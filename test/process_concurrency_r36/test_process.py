import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmarks/process_concurrency_r36'


class ProcessContracts(unittest.TestCase):
    def load_contracts(self):
        self.assertTrue((BASE / 'contracts.py').exists(), 'process contracts not implemented')
        spec = importlib.util.spec_from_file_location('r36_test_contracts', BASE / 'contracts.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def reports(self):
        return [dict(rank=rank, world_size=2, total_requests=4, passed=True, start_ns=100,
                     device='cpu', runtime=dict(runtime_kind='standard-gil', gil_before=True),
                     config_sha256='a'*64, worker_sha256='b'*64, gil_after=True,
                     comparison_identity=dict(model_sha256='c'*64, input_hashes=['d'*64]*8),
                     source_pins={'worker.py': 'b'*64}, original_full_model_executed=True,
                     requests=[dict(request_id=i, input_id=i % 8, arrival_ns=100 + i * 10,
                                    forward_return_ns=105 + i * 10, response_ns=110 + i * 10)
                               for i in range(rank, 4, 2)],
                     numeric_validation=dict(passed=True, validated_requests=2),
                     shared_weights_mutated=False) for rank in range(2)]

    def test_aggregate_common_window_not_sum_of_worker_rates(self):
        c = self.load_contracts()
        value = c.aggregate(self.reports(), 2, 4, 100)
        self.assertEqual(value['requests_per_second'], 4e9 / 40)
        self.assertEqual(value['completed_requests'], 4)
        self.assertEqual(value['response_ms']['p99'], 10 / 1e6)

    def test_reject_duplicate_missing_or_wrong_input(self):
        c = self.load_contracts()
        for mutation in ('duplicate', 'input', 'numeric', 'start', 'runtime', 'source', 'config', 'bank'):
            rows = self.reports()
            if mutation == 'duplicate': rows[1]['requests'][0]['request_id'] = 0
            if mutation == 'input': rows[1]['requests'][0]['input_id'] = 7
            if mutation == 'numeric': rows[1]['numeric_validation']['passed'] = False
            if mutation == 'start': rows[1]['start_ns'] = 101
            if mutation == 'runtime': rows[1]['runtime']['runtime_kind'] = 'sameft-gilon'
            if mutation == 'source': rows[1]['source_pins']['worker.py'] = 'x'*64
            if mutation == 'config': rows[1]['config_sha256'] = 'f'*64
            if mutation == 'bank': rows[1]['comparison_identity']['input_hashes'][0] = 'e'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.aggregate(rows, 2, 4, 100)


class SpawnedController(unittest.TestCase):
    def run_fake(self, failure=False):
        self.assertTrue((BASE / 'controller.py').exists(), 'process controller not implemented')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            worker = base / 'fake.py'
            worker.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('config','output'): p.add_argument('--'+k, required=True)
for k in ('rank','world-size','total-requests','warmup'): p.add_argument('--'+k,type=int,required=True)
a=p.parse_args(); root=Path(a.output); own=root/('rank-'+str(a.rank)); own.mkdir()
(own/'pid').write_text(str(os.getpid()))
if json.loads(Path(a.config).read_text())['failure'] and a.rank==0: raise SystemExit(5)
(own/'ready.json').write_text(json.dumps({'rank':a.rank,'pid':os.getpid()}))
while not (root/'go.json').exists(): time.sleep(.002)
start=json.loads((root/'go.json').read_text())['start_ns']
while time.monotonic_ns()<start: time.sleep(.001)
rows=[]
for i in range(a.rank,a.total_requests,a.world_size):
 t=time.monotonic_ns(); time.sleep(.002)
 rows.append(dict(request_id=i,input_id=i%8,arrival_ns=t,forward_return_ns=t+1,response_ns=time.monotonic_ns()))
(own/'done.json').write_text('{}')
while not (root/'validate.json').exists(): time.sleep(.002)
r=dict(rank=a.rank,world_size=a.world_size,total_requests=a.total_requests,passed=True,start_ns=start,
 requests=rows,numeric_validation=dict(passed=True,validated_requests=len(rows)),shared_weights_mutated=False)
r.update(device='cpu',runtime=dict(runtime_kind='standard-gil',gil_before=True),
 config_sha256='a'*64,worker_sha256='b'*64,gil_after=True,
 comparison_identity=dict(model_sha256='c'*64,input_hashes=['d'*64]*8),source_pins={'worker.py':'b'*64},
 original_full_model_executed=True)
(own/'result.json').write_text(json.dumps(r))
''')
            config = base / 'config.json'
            config.write_text(json.dumps(dict(failure=failure)))
            launch = base / 'launch.json'
            launch.write_text(json.dumps(dict(argv_prefix=[sys.executable])))
            output = base / 'out'
            proc = subprocess.run([sys.executable, str(BASE / 'controller.py'), '--launch', str(launch),
                '--config', str(config), '--worker', str(worker), '--world-size', '2', '--total-requests', '8',
                '--warmup', '0', '--output', str(output), '--timeout', '5'], capture_output=True, timeout=10)
            if failure:
                self.assertNotEqual(proc.returncode, 0)
                receipt = json.loads((output / 'controller.json').read_text())
                self.assertFalse(receipt['passed'])
                self.assertTrue(all(p['returncode'] is not None for p in receipt['children']))
            else:
                self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                receipt = json.loads((output / 'controller.json').read_text())
                self.assertTrue(receipt['passed'])
                self.assertEqual(receipt['metrics']['completed_requests'], 8)
                self.assertIn('memory_at_all_ready', receipt)
                self.assertEqual(len(receipt['memory_at_all_ready']['processes']), 2)
                if sys.platform == 'linux':
                    self.assertTrue(receipt['memory_at_all_ready']['available'])
                    self.assertGreater(receipt['memory_at_all_ready']['sum_pss_bytes'], 0)
                starts = [json.loads((output / f'rank-{i}/result.json').read_text())['start_ns'] for i in range(2)]
                self.assertEqual(starts[0], starts[1])
                self.assertTrue((output / 'validate.json').exists())

    def test_real_subprocesses_start_and_validate_together(self): self.run_fake()
    def test_failed_child_terminates_and_reaps_other_children(self): self.run_fake(True)


class RuntimeGuards(unittest.TestCase):
    def test_build_modes_cannot_be_mislabeled(self):
        self.assertTrue((BASE / 'worker.py').exists(), 'process worker not implemented')
        spec = importlib.util.spec_from_file_location('r36_test_worker', BASE / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_runtime_mode('standard-gil', None, True, {}, {})
        module.validate_runtime_mode('sameft-gilon', 1, True, {'gil': '1'}, {})
        for args in [('standard-gil', 1, True, {}, {}),
                     ('sameft-gilon', 1, False, {'gil': '1'}, {}),
                     ('sameft-gilon', 1, True, {}, {}),
                     ('standard-gil', None, True, {}, {'PYTHON_GIL': '1'})]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.validate_runtime_mode(*args)


if __name__ == '__main__': unittest.main()
