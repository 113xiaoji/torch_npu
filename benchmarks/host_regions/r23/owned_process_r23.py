"""Exact standalone extraction of the reviewed R22 owned-process cleanup.

The caller must use Popen(start_new_session=True) and retain this unreaped child.
Poll only through OwnedProcess; Popen.poll()/wait() would release PID ownership.
No SDK, fault injection, fetch, environment mutation, or application is launched.
"""
import os
from pathlib import Path
import signal
import time

SOURCE_SHA256 = '7fb65f05dc62cd322852f26625bd7b960c9664617af533706c9ca9a997342db2'
SOURCE_RELATIVE_PATH = 'submit-failure-region-prefix-20261027-20/run_submit_failure_gate_r22.py'
SOURCE_AST_SHA256 = {'TERM_SECONDS': 'c03afe833147ac6b30f976ec70fee1c5ac1ac89139f7fe81731594dd730e3808', 'REAP_SECONDS': '944ce8c5ec921307162ceed3ccded1021b92a5fc352b0b3cc83985ab01010469', 'require': '2407e108a9bab0667a8b0fc9e12a4ef08a70d8fbae9d87b3cfe5ea649dd9ba19', 'parse_proc_stat': '8e6ad3517a4b1539b106fb8933f41e9a7b9bccec456ed847e164d58aa1d657d2', 'group_members': '9447af8fad45f0bce2bfe1e6b099af623862832c600f88207ab853570dd974bd', 'OwnedProcess': 'adbb617729b417b8abbafe70a32db88bc66683ac7e1c4a7a9c2cbb6a6db4c417', 'stop_owned_group': '3631700942a0541db65e29492d98abad3d0d8cce583e9af08ee506b525461bb9'}

TERM_SECONDS=5

REAP_SECONDS=10

def require(value,message):
    if not value:raise ValueError(message)

def parse_proc_stat(text):
    first,tail=text.split('(',1);_comm,tail=tail.rsplit(')',1);fields=tail.split()
    require(len(fields)>=20,'incomplete proc stat')
    return dict(pid=int(first.strip()),state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),session=int(fields[3]),starttime=int(fields[19]))

def group_members(pgid,session,proc_root=Path('/proc')):
    rows=[]
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit():continue
        try:row=parse_proc_stat((folder/'stat').read_text())
        except (FileNotFoundError,ProcessLookupError):continue
        if row['pgid']==pgid and row['session']==session and row['state'] not in ('Z','X'):rows.append(row)
    return rows

class OwnedProcess:
    """Keep this setsid child unreaped until the final group signal is sent."""
    def __init__(self,process):
        self.process,self.pid=process,process.pid
        self.owns_unreaped_session=True;self.returncode=None
    def poll(self):
        if not self.owns_unreaped_session:return self.returncode
        try:info=os.waitid(os.P_PID,self.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
        except ChildProcessError:
            self.owns_unreaped_session=False
            raise
        if info is None:return None
        if info.si_code==os.CLD_EXITED:return info.si_status
        require(info.si_code in (os.CLD_KILLED,os.CLD_DUMPED),'unexpected child waitid status')
        return -info.si_status
    def wait(self,timeout):
        try:code=self.process.wait(timeout=timeout)
        finally:
            # Popen.wait may reap in its KeyboardInterrupt handler, then raise.
            # Never retain permission to signal that now-reusable PID/group.
            if self.process.returncode is not None:
                self.returncode=self.process.returncode;self.owns_unreaped_session=False
        self.returncode=code;self.owns_unreaped_session=False
        return code

def stop_owned_group(process,identity):
    """Last KILL and group inventory precede releasing the owned leader PID."""
    require(process.owns_unreaped_session is True,'refusing to signal an already reaped/reusable process group')
    signals=[];first=None;remaining=None
    current=None
    try:current=parse_proc_stat(Path(f'/proc/{process.pid}/stat').read_text())
    except OSError as error:first=error
    if current is not None:
        require(current['pgid']==current['session']==process.pid and
            (identity is None or current['starttime']==identity['starttime']),
            'refusing to signal a changed child process/group')
    try:
        process.poll()  # WNOWAIT also detects an unexpected external reap.
        require(process.owns_unreaped_session is True,'child ownership was lost before first signal')
        # Successful Popen(start_new_session=True) plus an unreaped child also
        # binds cleanup when the first own-/proc identity read was unavailable.
        try:os.killpg(process.pid,signal.SIGTERM);signals.append('SIGTERM')
        except ProcessLookupError:pass
        deadline=time.monotonic()+TERM_SECONDS
        while process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
    except BaseException as error:
        if first is None:first=error
    finally:
        reap_deadline=time.monotonic()+REAP_SECONDS
        try:
            require(process.owns_unreaped_session is True,'child was unexpectedly reaped before final group kill')
            # Never let leader exit or a failed inventory suppress this KILL.
            # WNOWAIT above still reserves the original PID/session identity.
            try:os.killpg(process.pid,signal.SIGKILL);signals.append('SIGKILL')
            except ProcessLookupError:pass
            while True:
                remaining=group_members(process.pid,process.pid)
                if not remaining:break
                require(time.monotonic()<reap_deadline,'owned group retained live members after KILL')
                time.sleep(.05)
        except BaseException as error:
            if first is None:first=error
        finally:
            try:process.wait(timeout=max(.01,reap_deadline-time.monotonic()))
            except BaseException as error:
                if first is None:first=error
    if first is not None:raise first
    require(process.returncode is not None and remaining==[],'owned process group did not terminate/reap')
    return dict(exit_code=process.returncode,signals=signals,leader_reaped=True,group_live_members=remaining,
        group_empty_observed_before_reap=True,
        boundary='external termination after observation; -SIGTERM does not prove the child handled TERM')
