import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import subprocess


class ProfilerContracts(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('run_msprof_r23.py')
        self.assertTrue(path.is_file(), 'missing profiler controller')
        spec = importlib.util.spec_from_file_location('msprof_contract_fixture', path)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)

    def test_four_cases_reuse_exact_off_argv_except_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.m.parse_args(['--root', str(root), '--tools', str(root/'tools'),
                '--control', str(root/'control'), '--cohort-result', str(root/'cohort/RESULT.json'),
                '--output', str(root/'new'), '--window-wrapper', str(root/'window.py')])
            rows = []
            for abi in ('std', 'ft'):
                for label in ('original-serial', 'candidate-serial', 'worker4'):
                    for mode in ('off', 'stages'):
                        rows.append(dict(abi=abi, label=label, profile_mode=mode, round=1, kind='perf',
                            runner_arm='worker4' if label=='worker4' else 'serial',
                            argv=['/python', '-I', '-B', '/runner.py', '--kind', 'perf', '--output', '/old/'+abi+label+mode],
                            output='/old/'+abi+label+mode, binding={'abi':abi}))
            cohort = dict(plan=dict(repeat=rows)); old=copy.deepcopy(cohort)
            plan = self.m.make_plan(args, cohort)
            self.assertEqual([(r['abi'],r['label']) for r in plan['cases']],
                [('std','candidate-serial'),('std','worker4'),('ft','candidate-serial'),('ft','worker4')])
            for row in plan['cases']:
                original=row['cohort_item']['argv']; changed=row['runner_argv']
                index=original.index('--output')+1
                self.assertEqual(changed[:index],original[:index]);self.assertEqual(changed[index+1:],original[index+1:])
                self.assertNotEqual(changed[index],original[index]);self.assertIn('--application=',row['argv'][1])
                self.assertEqual(row['timeout_seconds'],1800)
            self.assertEqual(cohort,old);self.assertFalse(args.output.exists())

    def test_npu_idle_requires_success_health_and_no_process_rows(self):
        raw=b'| NPU Name | Health |\n| 6 Ascend910 | OK | 10 |\n| NPU Chip | Process id |\n| No running processes found in NPU 6 |\n'
        self.m.validate_idle(0,raw)
        for code,data in [(1,raw),(0,raw.replace(b' OK ',b' Alarm ')),(0,raw.replace(b'No running processes found',b'unknown')),
                (0,raw+b'| 6 0 | 1234 | python | 20 |\n')]:
            with self.subTest(code=code,data=data),self.assertRaises(ValueError):self.m.validate_idle(code,data)

    def test_campaign_birth_identity_rejects_live_parent_but_records_reuse(self):
        launch=dict(pid=123,pgid=123,starttime_ticks=456)
        with self.assertRaisesRegex(ValueError,'still live'):
            self.m.require_campaign_stopped(launch,dict(pid=123,exists=True,starttime_ticks=456))
        reused=self.m.require_campaign_stopped(launch,dict(pid=123,exists=True,starttime_ticks=999))
        self.assertTrue(reused['pid_reused']);self.assertFalse(reused['original_instance_live'])
        gone=self.m.require_campaign_stopped(launch,dict(pid=123,exists=False))
        self.assertFalse(gone['pid_reused']);self.assertFalse(gone['original_instance_live'])

    def test_window_uses_actual_wrapper_schema_and_rejects_incomplete_or_wrong_argv(self):
        case=dict(runner_argv=['/python','-I','-B','/runner.py','--output','/new/RUNNER.json'])
        row=dict(schema=1,window_complete=True,temporary_module_bindings_restored=True,
            diagnostic_only=True,performance_conclusion_allowed=False,original_exit_code=0,
            original_report_passed=True,failure=None,samples=2000,common_wall_ns_matches=True,
            tool_sha256=self.m.WINDOW_SHA,runner=dict(sha256='a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'),
            original_argv=case['runner_argv'][4:],application_process=dict(proc_identity_observed=True,
                pid=123,pgid=122,starttime_ticks=456,host_boot_id=self.m.EXPECTED_BOOT))
        self.m.validate_window(row,case)
        for key,value in [('window_complete',False),('temporary_module_bindings_restored',False),
                ('original_exit_code',1),('samples',1999),('original_argv',[]),('application_process',{})]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.m.validate_window(bad,case)

    def test_owned_run_observes_original_cli_then_cleans_group_before_return(self):
        events=[]
        class Owned:
            def __init__(self,process):self.pid=process.pid
            def poll(self):events.append('observe-unreaped');return 0
        def stop(process,identity):
            events.append('cleanup')
            return dict(exit_code=0,leader_reaped=True,group_live_members=[],group_empty_observed_before_reap=True)
        ownership=SimpleNamespace(OwnedProcess=Owned,stop_owned_group=stop)
        def popen(*args,**kw):
            self.assertTrue(kw['start_new_session']);events.append('spawn');return SimpleNamespace(pid=123)
        with tempfile.TemporaryDirectory() as tmp:
            row={}
            code=self.m.run_owned(['/msprof'],{},None,Path(tmp),row,ownership,
                popen=popen,identity_reader=lambda pid:dict(pid=pid,starttime=77))
            self.assertEqual(code,0);self.assertEqual(row['exit_code'],0)
            self.assertEqual(events,['spawn','observe-unreaped','cleanup'])
            self.assertTrue(row['termination']['group_empty_observed_before_reap'])
            self.assertTrue((Path(tmp)/'TERMINATION.json').is_file())

    def test_owned_timeout_keeps_first_error_and_still_attempts_group_cleanup(self):
        calls=[]
        class Owned:
            def __init__(self,process):self.pid=process.pid
            def poll(self):return None
        def cleanup(process,identity):calls.append('cleanup');raise RuntimeError('secondary cleanup failure')
        ownership=SimpleNamespace(OwnedProcess=Owned,stop_owned_group=cleanup)
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(self.m.time,'monotonic',side_effect=[0,2000]):
            row={}
            with self.assertRaises(subprocess.TimeoutExpired):
                self.m.run_owned(['/msprof'],{},None,Path(tmp),row,ownership,
                    popen=lambda *a,**k:SimpleNamespace(pid=123),identity_reader=lambda pid:dict(pid=pid,starttime=77))
            self.assertEqual(calls,['cleanup']);self.assertEqual(row['termination_error']['error'],'secondary cleanup failure')
            self.assertTrue((Path(tmp)/'TERMINATION.json').is_file())


if __name__=='__main__':unittest.main()
