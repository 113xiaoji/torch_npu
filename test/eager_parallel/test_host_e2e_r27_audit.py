"""Independent report-failure and numeric-evidence rejection contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


class ReportAudit(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('r27_e2e_audit',ROOT/'benchmarks/eager_parallel_r27/host_e2e.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def invoke(self, root, runner):
        config=root/'config.json';config.write_text(json.dumps(dict(artifact_pins={},source_pins={})))
        argv=['host_e2e.py','--config',str(config),'--output',str(root/'result'),'--arm','ei','--kind','cold','--round','0']
        with patch.object(self.module,'run',runner),patch('sys.argv',argv):
            self.module.main()

    def test_failure_records_native_first_error_and_keeps_original_exception(self):
        original=RuntimeError('first numerical failure')
        def failed(config,report):
            report['_failure_snapshot_callback']=lambda:dict(native=dict(first_error='kernel failed',quarantined=True))
            raise original
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(RuntimeError) as raised:self.invoke(root,failed)
            self.assertIs(raised.exception,original)
            result=json.loads((root/'result/result.json').read_text())
        self.assertFalse(result['passed'])
        self.assertEqual(result['error']['message'],'first numerical failure')
        self.assertEqual(result['failure_snapshot']['native']['first_error'],'kernel failed')
        self.assertTrue(result['failure_snapshot']['native']['quarantined'])

    def test_failed_snapshot_does_not_replace_first_error(self):
        original=RuntimeError('first caller failure')
        def broken_snapshot():raise ValueError('diagnostics unavailable')
        def failed(config,report):
            report['_failure_snapshot_callback']=broken_snapshot
            raise original
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(RuntimeError) as raised:self.invoke(root,failed)
            self.assertIs(raised.exception,original)
            result=json.loads((root/'result/result.json').read_text())
        self.assertEqual(result['failure_snapshot_error'],'ValueError')
        self.assertEqual(result['error']['message'],'first caller failure')

    def test_failed_report_write_does_not_replace_first_error(self):
        original=RuntimeError('first kernel failure')
        def failed(config,report):raise original
        write=Path.write_text
        def reject_result(path,*args,**kwargs):
            if path.name=='result.json':raise OSError('disk full')
            return write(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch.object(Path,'write_text',reject_result):
            with self.assertRaises(RuntimeError) as raised:self.invoke(Path(directory),failed)
        self.assertIs(raised.exception,original)

    def test_runtime_rejects_inexact_counters_and_false_numeric_qualification(self):
        valid=dict(backend_kind='HOST_SIM_R27_CPU',submitted=1,completed=1,errors=0,
                   numeric_supported=True,real_sdk_qualified=False,injected_faults=0,fault_active=False)
        self.module.check_runtime(valid)
        for change in ({'submitted':True},{'completed':1.0},{'errors':False},{'numeric_supported':1},
                       {'real_sdk_qualified':0},{'injected_faults':1},{'fault_active':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.module.check_runtime(dict(valid,**change))

    def test_statistics_reject_duplicate_arms_nonfinite_and_bool_timing(self):
        rows=[dict(round=0,arm=arm,passed=True,kind='perf',samples=3,window_wall_ns=2_000_000_000,
                   median_ns=100,p95_ns=120) for arm in ('off','e','ei')]
        self.module.paired_summary(rows,1)
        for change in ({'arm':'e'},{'median_ns':True},{'p95_ns':float('inf')},{'round':True},{'samples':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.module.paired_summary([dict(rows[0],**change)]+rows[1:],1)

    def test_unpinned_or_changed_helper_is_rejected_before_loading_code(self):
        runtime=SimpleNamespace(platform='linux',version_info=(3,14,6),_is_gil_enabled=lambda:False,
                                gettrace=lambda:None,getprofile=lambda:None)
        environment=dict(TASK_QUEUE_ENABLE='0',QWEN_ACL_HOST_SIM='1',PER_STREAM_QUEUE='0',
                         ASCEND_LAUNCH_BLOCKING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config={}
            for key in ('measurement_helper','reference','mapping_helper','frozen_model_helper'):
                path=root/(key+'.py');path.write_text("raise RuntimeError('must not execute unverified code')\n")
                config[key]=str(path)
            worker=str(Path(self.module.__file__).resolve())
            config['source_pins']={path:self.module.sha(path) for path in [worker,*config.values()]}
            variants=[dict(config['source_pins']),dict(config['source_pins'])]
            variants[0].pop(config['measurement_helper'])
            variants[1][config['measurement_helper']]='0'*64
            for pins in variants:
                with patch.object(self.module,'sys',runtime),patch.dict('os.environ',environment,clear=True),\
                     patch.object(self.module,'load',side_effect=AssertionError('unverified helper executed')) as loader:
                    with self.assertRaisesRegex(ValueError,'source pin'):
                        self.module.run(dict(config,source_pins=pins),{})
                    loader.assert_not_called()


if __name__=='__main__':unittest.main()
