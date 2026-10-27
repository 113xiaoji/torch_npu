"""Isolated process/signal fixtures; real process tests require Linux /proc."""
import contextlib
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def module():
    spec=importlib.util.spec_from_file_location('owned_group_cleanup_fixture',HERE/'owned_process_r23.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class Process:
    pid=4321
    def __init__(self):self.owns_unreaped_session=True;self.returncode=None;self.peek=None;self.actions=[];self.poll_error=None
    def poll(self):
        if self.poll_error is not None:
            error,self.poll_error=self.poll_error,None;raise error
        return self.peek
    def wait(self,timeout):
        self.actions.append('reap');self.owns_unreaped_session=False;self.returncode=self.peek
        return self.returncode


class OwnedGroupCleanupTests(unittest.TestCase):
    def test_extracted_nodes_match_frozen_source_ast(self):
        m=module()
        self.assertEqual(m.SOURCE_SHA256,'7fb65f05dc62cd322852f26625bd7b960c9664617af533706c9ca9a997342db2')
        expected={
            'TERM_SECONDS':'c03afe833147ac6b30f976ec70fee1c5ac1ac89139f7fe81731594dd730e3808',
            'REAP_SECONDS':'944ce8c5ec921307162ceed3ccded1021b92a5fc352b0b3cc83985ab01010469',
            'require':'2407e108a9bab0667a8b0fc9e12a4ef08a70d8fbae9d87b3cfe5ea649dd9ba19',
            'parse_proc_stat':'8e6ad3517a4b1539b106fb8933f41e9a7b9bccec456ed847e164d58aa1d657d2',
            'group_members':'9447af8fad45f0bce2bfe1e6b099af623862832c600f88207ab853570dd974bd',
            'OwnedProcess':'adbb617729b417b8abbafe70a32db88bc66683ac7e1c4a7a9c2cbb6a6db4c417',
            'stop_owned_group':'3631700942a0541db65e29492d98abad3d0d8cce583e9af08ee506b525461bb9'}
        actual={}
        for node in ast.parse((HERE/'owned_process_r23.py').read_text(encoding='utf-8')).body:
            names=([n.id for n in node.targets if isinstance(n,ast.Name)] if isinstance(node,ast.Assign) else [getattr(node,'name','')])
            for name in names:
                if name in expected:actual[name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        self.assertEqual(actual,expected);self.assertEqual(m.SOURCE_AST_SHA256,expected)

    def fixture(self):
        m=module();p=Process()
        identity=dict(pid=p.pid,state='S',ppid=10,pgid=p.pid,session=p.pid,starttime=123)
        text=f'{p.pid} (fixture) S 10 {p.pid} {p.pid} '+'0 '*15+'123\n'
        def kill(pid,signum):
            self.assertEqual(pid,p.pid);self.assertTrue(p.owns_unreaped_session)
            p.actions.append('TERM' if signum==m.signal.SIGTERM else 'KILL')
            p.peek=-signum
        def members(*_args,**_kwargs):
            self.assertTrue(p.owns_unreaped_session,'inventory must precede releasing the PID instance')
            p.actions.append('inventory');return []
        stack=contextlib.ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(m.signal,'SIGKILL',9,create=True))
        stack.enter_context(patch.object(m.os,'killpg',side_effect=kill,create=True))
        stack.enter_context(patch.object(m.Path,'read_text',return_value=text))
        stack.enter_context(patch.object(m,'group_members',side_effect=members))
        return m,p,identity

    def test_term_exit_still_attempts_group_kill_before_reap_and_checks_group(self):
        m,p,identity=self.fixture();result=m.stop_owned_group(p,identity)
        self.assertEqual(p.actions,['TERM','KILL','inventory','reap'])
        self.assertTrue(result['leader_reaped']);self.assertEqual(result['group_live_members'],[])

    def test_interrupted_term_wait_still_kills_and_reaps_preserving_first_error(self):
        m,p,identity=self.fixture();original=KeyboardInterrupt('fixture stop interrupt');p.poll_error=original
        with self.assertRaises(KeyboardInterrupt) as caught:m.stop_owned_group(p,identity)
        self.assertIs(caught.exception,original)
        self.assertIn('KILL',p.actions);self.assertEqual(p.actions[-1],'reap')

    def test_initial_proc_identity_error_still_terminates_owned_birth_and_reaps(self):
        m,p,identity=self.fixture()
        with patch.object(m.Path,'read_text',side_effect=PermissionError('fixture own proc unreadable')):
            with self.assertRaises(PermissionError):m.stop_owned_group(p,None)
        self.assertIn('TERM',p.actions);self.assertIn('KILL',p.actions);self.assertEqual(p.actions[-1],'reap')

    def test_inventory_failure_cannot_prevent_last_kill_or_reap(self):
        m,p,identity=self.fixture()
        with patch.object(m,'group_members',side_effect=PermissionError('fixture inventory unreadable')):
            with self.assertRaises(PermissionError):m.stop_owned_group(p,identity)
        self.assertEqual(p.actions[:2],['TERM','KILL']);self.assertEqual(p.actions[-1],'reap')

    def test_already_reaped_identity_is_never_signalled(self):
        m,p,identity=self.fixture();p.owns_unreaped_session=False
        with self.assertRaises(ValueError):m.stop_owned_group(p,identity)
        self.assertEqual(p.actions,[])

    def test_owned_process_poll_uses_wnowait_and_never_reaps(self):
        m=module();calls=[]
        underlying=SimpleNamespace(pid=4321,returncode=None)
        underlying.wait=lambda timeout:calls.append(('wait',timeout)) or 0
        with contextlib.ExitStack() as stack:
            for name,value in {'P_PID':1,'WEXITED':4,'WNOHANG':1,'WNOWAIT':0x1000000,'CLD_EXITED':1,'CLD_KILLED':2,'CLD_DUMPED':3}.items():
                stack.enter_context(patch.object(m.os,name,value,create=True))
            waitid=stack.enter_context(patch.object(m.os,'waitid',return_value=SimpleNamespace(si_code=1,si_status=0),create=True))
            process=m.OwnedProcess(underlying)
            self.assertEqual(process.poll(),0)
            self.assertTrue(process.owns_unreaped_session);self.assertEqual(calls,[])
            self.assertEqual(waitid.call_args.args,(1,4321,4|1|0x1000000))

    def test_wait_that_reaps_then_raises_keeps_first_error_and_clears_birth_ownership(self):
        m=module();underlying=SimpleNamespace(pid=4321,returncode=None)
        original=KeyboardInterrupt('wait reaped before raising')
        def wait(timeout):underlying.returncode=-15;raise original
        underlying.wait=wait;process=m.OwnedProcess(underlying)
        with self.assertRaises(KeyboardInterrupt) as caught:process.wait(1)
        self.assertIs(caught.exception,original)
        self.assertEqual(process.returncode,-15)
        self.assertFalse(process.owns_unreaped_session)

    def test_external_reap_is_checked_before_the_first_signal(self):
        m,p,identity=self.fixture();original=ChildProcessError('fixture external reap')
        def poll():p.owns_unreaped_session=False;raise original
        p.poll=poll
        with self.assertRaises(ChildProcessError) as caught:m.stop_owned_group(p,identity)
        self.assertIs(caught.exception,original)
        self.assertNotIn('TERM',p.actions);self.assertNotIn('KILL',p.actions)

    @unittest.skipUnless(os.name=='posix' and hasattr(os,'waitid') and hasattr(os,'WNOWAIT'),'Linux nonreaping process fixture unavailable')
    def test_real_short_lived_process_is_group_terminated_then_reaped(self):
        m=module()
        original=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(20)'],start_new_session=True)
        process=m.OwnedProcess(original)
        try:
            identity=m.parse_proc_stat(Path(f'/proc/{process.pid}/stat').read_text())
            result=m.stop_owned_group(process,identity)
            self.assertTrue(result['leader_reaped']);self.assertTrue(result['group_empty_observed_before_reap'])
            self.assertFalse(process.owns_unreaped_session)
            self.assertIn(result['exit_code'],(-m.signal.SIGTERM,-m.signal.SIGKILL))
        finally:
            if process.owns_unreaped_session:
                try:os.killpg(process.pid,m.signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(10)

    @unittest.skipUnless(sys.platform=='linux' and hasattr(os,'fork') and hasattr(os,'waitid') and hasattr(os,'WNOWAIT'),'Linux parent-exit/TERM-ignore fixture unavailable')
    def test_real_exited_leader_keeps_term_ignoring_child_owned_until_group_kill(self):
        m=module()
        script='''import os,signal,time
r,w=os.pipe()
pid=os.fork()
if pid==0:
 os.close(r);signal.signal(signal.SIGTERM,signal.SIG_IGN)
 os.write(w,b'R');os.close(w);time.sleep(20);os._exit(0)
os.close(w);assert os.read(r,1)==b'R';os.close(r)
print(pid,flush=True)
'''
        original=subprocess.Popen([sys.executable,'-I','-B','-c',script],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True)
        process=m.OwnedProcess(original)
        try:
            child=int(original.stdout.readline().strip())
            identity=m.parse_proc_stat(Path(f'/proc/{process.pid}/stat').read_text())
            deadline=time.monotonic()+5
            while process.poll() is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(process.poll(),0);self.assertTrue(process.owns_unreaped_session)
            self.assertIn(child,[row['pid'] for row in m.group_members(process.pid,process.pid)])
            result=m.stop_owned_group(process,identity)
            self.assertEqual(result['exit_code'],0)  # Observed leader CLI is not the descendant's cleanup status.
            self.assertEqual(result['signals'],['SIGTERM','SIGKILL'])
            self.assertTrue(result['group_empty_observed_before_reap']);self.assertTrue(result['leader_reaped'])
            self.assertEqual(m.group_members(process.pid,process.pid),[])
        finally:
            if process.owns_unreaped_session:
                try:os.killpg(process.pid,m.signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(10)
            original.stdout.close()


if __name__=='__main__':unittest.main()
