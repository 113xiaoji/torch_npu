"""Lifecycle tests use ordinary host Python; not actual benchmark evidence."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'benchmarks/hostsim_r25/run.py'
        self.assertTrue(path.exists(),'owned-process controller must exist')
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('_hostsim_r25_controller',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_output_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):self.m.fresh_output(Path(directory))

    @unittest.skipUnless(os.name=='posix','actual process-group behavior tested on Linux only')
    def test_timeout_owns_and_terminates_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.m.run_owned([sys.executable,'-c','import time; time.sleep(10)'],os.environ.copy(),Path(directory)/'worker.log',.05)
            self.assertTrue(result['timed_out'])
            self.assertIsNotNone(result['returncode'])
            self.assertTrue(result['group_closed'])

    def test_row_validation_rejects_foreign_cold_gate_identity(self):
        row=dict(runtime='ft',arm='queue_w4',kind='perf',round=0,workers=4,warmup=20,samples=200)
        result=dict(row,passed=True,performance_eligible=True,forward_calls=220,durations_ns=[1]*200,
            worker_sha256='a'*64,contracts_sha256='b'*64,mechanism=dict(passed=True,embedding_commands_per_forward=26),
            ready=dict(cpu_identity=dict(state_sha256='x',input_tensor_tree_sha256='y'),artifact_pins={'new':'pin'},mapping=[]))
        gate=dict(result,kind='cold',cold_gate=dict(passed=True))
        self.m.validate_row(result,row,'a'*64,'b'*64,gate)
        gate['ready']=dict(gate['ready'],artifact_pins={'other':'pin'})
        with self.assertRaises(ValueError):self.m.validate_row(result,row,'a'*64,'b'*64,gate)


if __name__=='__main__':unittest.main()
