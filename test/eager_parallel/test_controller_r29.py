"""The high-L continuation retains an identical fresh L32 control."""
import copy
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class HighLengthController(unittest.TestCase):
    def setUp(self):
        self.module = load(ROOT / 'benchmarks/eager_parallel_r29/controller.py', 'r29_controller_test')

    def provenance(self, length):
        identity = copy.deepcopy(self.module.IDENTITY['global'])
        identity['input_sha256'] = self.module.IDENTITY['per_l'][str(length)]['input_sha256']
        return dict(identity=identity, workload=dict(bag_length=length, batch=128, tables=26,
            embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
            original_input_sha256='239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6',
            base_input_sha256=self.module.IDENTITY['per_l']['32']['input_sha256'], extra_input_seed=20260920,
            input_policy='original R28 L=32 IDs per bag plus fixed per-table Bx224 prefix',
            tables_detail=[dict(table=i, indices_shape=[128*length], logical_indices_shape=[128,length],
                offsets_shape=[128], offset_first=0, offset_last=127*length,
                bag_length_min=length, bag_length_max=length) for i in range(26)]))

    def test_balanced_plan_keeps_fresh_32_control_and_high_lengths(self):
        lengths = (32,64,128,256)
        self.assertEqual(self.module.LENGTHS, lengths)
        cold = self.module.make_plan('cold')
        perf = self.module.make_plan('perf')
        self.assertEqual(len(cold),12)
        self.assertEqual(len(perf),144)
        self.assertEqual(sum(r['samples'] for r in perf),432000)
        self.assertEqual(len({r['label'] for r in perf}),144)
        orders = list(itertools.permutations(('off','e','ei')))
        for length in lengths:
            permutations = [tuple(r['arm'] for r in perf if r['round']==n and r['bag_length']==length)
                            for n in range(12)]
            self.assertTrue(all(permutations.count(order)==2 for order in orders))
        for n in range(12):
            rows = [r for r in perf if r['round']==n]
            self.assertEqual([rows[i]['bag_length'] for i in range(0,12,3)], list(lengths[n%4:]+lengths[:n%4]))
            self.assertTrue(all(r['warmup']==50 and r['samples']==3000 for r in rows))

    def test_identities_pin_exact_new_inputs_and_unchanged_model(self):
        for length in self.module.LENGTHS:
            result = self.module.workload_identity(self.provenance(length),length)
            self.assertEqual(result['global'],self.module.IDENTITY['global'])
        for mutation in ('hash','prefix','seed','shape','model'):
            p = self.provenance(64)
            if mutation=='hash':p['identity']['input_sha256']=self.module.IDENTITY['per_l']['32']['input_sha256']
            if mutation=='prefix':p['workload']['base_input_sha256']='0'*64
            if mutation=='seed':p['workload']['extra_input_seed']=20260919
            if mutation=='shape':p['workload']['tables_detail'][0]['offset_last']=127*32
            if mutation=='model':p['identity']['state_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.module.workload_identity(p,64)

    def test_config_extends_only_benchmark_source_and_inputs(self):
        old_tests = load(ROOT / 'test/eager_parallel/test_controller_r28.py','r29_old_fixture')
        fixture = old_tests.Controller()
        fixture.module = self.module
        with tempfile.TemporaryDirectory() as directory:
            base = fixture.base_config(directory)
            original = copy.deepcopy(base)
            configs = self.module.prepare_configs(base,Path(directory))
            self.assertEqual(base,original)
            self.assertEqual(set(configs),set(self.module.LENGTHS))
            for length,config in configs.items():
                for key in ('package','python','ld_library_path','runtime_identity','libraries','original_source'):
                    self.assertEqual(config[key],base[key])
                self.assertEqual(config['bag_length'],length)
                self.assertEqual(config['reference'],str(Path(directory)/'benchmarks/eager_parallel_r29/workload.py'))
                self.assertTrue(config['validate_stages'])
                self.assertEqual(config['source_pins'][str(Path(directory)/'benchmarks/eager_parallel_r28/workload.py')],
                    '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960')


if __name__=='__main__':unittest.main()
