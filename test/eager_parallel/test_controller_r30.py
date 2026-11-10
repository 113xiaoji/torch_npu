"""Mixed inputs retain the existing timing, runtime and numerical gates."""
import copy
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LENGTHS = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MixedController(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'benchmarks/eager_parallel_r30/controller.py'
        self.assertTrue(path.is_file(), 'R30 mixed-input controller is missing')
        self.module = load(path, 'r30_controller_test')

    def provenance(self, profile='mixed'):
        identity = copy.deepcopy(self.module.IDENTITY['global'])
        identity['input_sha256'] = self.module.IDENTITY['per_l']['mixed']['input_sha256']
        return dict(identity=identity, workload=dict(bag_length=profile, bag_lengths=LENGTHS[:],
            batch=128, tables=26, embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
            original_input_sha256='239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6',
            base_input_sha256='9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782',
            extra_input_seed=20260920,
            input_policy='per-table prefixes of frozen R29 L=256 IDs; fixed MLPerf multi-hot length vector',
            tables_detail=[dict(table=i, indices_shape=[128*n], logical_indices_shape=[128,n],
                offsets_shape=[128], offset_first=0, offset_last=127*n,
                bag_length_min=n, bag_length_max=n) for i,n in enumerate(LENGTHS)]))

    def fixture(self):
        old = load(ROOT / 'test/eager_parallel/test_controller_r28.py', 'r30_report_fixture')
        fixture = old.Controller()
        fixture.setUp()
        fixture.module = self.module
        fixture.provenance = self.provenance
        return fixture

    def test_fixed_profile_has_three_cold_and_thirty_six_balanced_perf_workers(self):
        self.assertEqual(self.module.LENGTHS, ('mixed',))
        self.assertEqual(list(self.module.BAG_LENGTHS), LENGTHS)
        cold, perf = self.module.make_plan('cold'), self.module.make_plan('perf')
        self.assertEqual(len(cold), 3)
        self.assertEqual(len(perf), 36)
        self.assertEqual(len({r['label'] for r in cold+perf}), 36)
        self.assertEqual(sum(r['samples'] for r in perf), 108000)
        self.assertTrue(all(r['warmup']==0 and r['samples']==1 for r in cold))
        self.assertTrue(all(r['warmup']==50 and r['samples']==3000 for r in perf))
        orders = [tuple(r['arm'] for r in perf if r['round']==i) for i in range(12)]
        for order in itertools.permutations(('off','e','ei')):
            self.assertEqual(orders.count(order), 2)
        self.assertTrue(all(r['bag_length']=='mixed' for r in cold+perf))
        for kwargs in (dict(rounds=1),dict(samples=1),dict(rounds=True)):
            with self.assertRaises(ValueError): self.module.make_plan('perf',**kwargs)

    def test_identity_rejects_wrong_table_lengths_offsets_hashes_or_model(self):
        self.module.workload_identity(self.provenance(),'mixed')
        for mutation in ('vector','offset','shape','hash','model','prefix','tag'):
            p=self.provenance()
            if mutation=='vector': p['workload']['bag_lengths'][20]=64
            if mutation=='offset': p['workload']['tables_detail'][2]['offset_last']=254
            if mutation=='shape': p['workload']['tables_detail'][20]['logical_indices_shape']=[128,64]
            if mutation=='hash': p['identity']['input_sha256']='0'*64
            if mutation=='model': p['identity']['state_sha256']='0'*64
            if mutation=='prefix': p['workload']['base_input_sha256']='0'*64
            if mutation=='tag': p['workload']['bag_length']=8
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.module.workload_identity(p,'mixed')
        for tag in (8,True,None,'other'):
            with self.assertRaises(ValueError): self.module.workload_identity(self.provenance(),tag)

    def test_config_preserves_runtime_native_and_dense_configuration(self):
        fixture=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base=fixture.base_config(directory); original=copy.deepcopy(base)
            configs=self.module.prepare_configs(base,Path(directory))
            self.assertEqual(base,original)
            self.assertEqual(set(configs),{'mixed'})
            c=configs['mixed']
            self.assertEqual(c['bag_length'],'mixed')
            self.assertTrue(c['validate_stages'])
            self.assertEqual(c['reference'],str(Path(directory)/'benchmarks/eager_parallel_r30/workload.py'))
            for key in ('package','python','ld_library_path','runtime_identity','libraries','original_source'):
                self.assertEqual(c[key],base[key])

    def test_report_retains_raw_count_stream_stage_and_timing_rejection_gates(self):
        self.fixture().test_report_validation_recomputes_raw_counts_and_rejects_false_or_saturated_evidence()

    def test_serial_gate_rejects_pid_reuse_and_overlap(self):
        self.fixture().test_serial_intervals_reject_overlap_pid_reuse_and_noninteger_timestamps()


if __name__=='__main__':
    unittest.main()
