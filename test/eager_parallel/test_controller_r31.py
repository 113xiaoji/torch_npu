"""Five treatments share one frozen MetaSyn projection and native build."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Controller(unittest.TestCase):
    def setUp(self):
        path=ROOT/'benchmarks/eager_parallel_r31/controller.py'
        self.assertTrue(path.is_file(),'R31 selective controller is missing')
        self.module=load(path,'r31_controller_test')

    def provenance(self):
        p=json.loads((ROOT/'benchmarks/eager_parallel_r31/meta26_inputs.json').read_text())
        identity=dict(self.module.IDENTITY['global'],input_sha256='c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1')
        return dict(identity=identity,workload=dict(bag_length='meta26',batch=128,tables=26,
            embedding_dim=16,vocabulary=1024,dense_shape=[128,13],
            original_input_sha256='239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6',
            source_git=p['source_git'],source_sha256=p['source_sha256'],
            projection_sha256='5eb2663c2d00d64db83c9d5b858c78f80849841216b242c93cfcd72d2dfbade5',
            selected_source_tables=p['selected_source_tables'],selection=p['selection'],source_shape=[856,65536],
            input_policy='MetaSyn first128 bags; lossless int64 IDs modulo1024',
            routing_profile=dict(large_embeddings=2,small_embeddings=24,min_embedding_bag_length=64),
            tables_detail=[dict(table=row['model_table'],source_table=row['source_table'],
                indices_shape=[row['indices_total']],offsets_shape=[128],offset_first=0,
                offset_last=row['offsets'][-1],bag_lengths=row['lengths'],
                bag_length_min=min(row['lengths']),bag_length_max=max(row['lengths']),
                mean_length=row['mean_length']) for row in p['tables']]))

    def test_five_arm_williams_plan_balances_positions_and_directed_neighbors(self):
        cold=self.module.make_plan('cold');perf=self.module.make_plan('perf')
        self.assertEqual(len(cold),5);self.assertEqual(len(perf),50)
        self.assertEqual(sum(r['samples'] for r in perf),150000)
        arms=('off','e','ei','ae','aei')
        self.assertEqual(tuple(r['arm'] for r in cold),arms)
        self.assertTrue(all(r['warmup']==50 and r['samples']==3000 for r in perf))
        orders=[tuple(r['arm'] for r in perf if r['round']==i) for i in range(10)]
        self.assertEqual(len(set(orders)),10)
        for arm in arms:
            self.assertTrue(all(sum(order[position]==arm for order in orders)==2 for position in range(5)))
        for a,b in itertools.permutations(arms,2):
            self.assertEqual(sum(sum(pair==(a,b) for pair in zip(order,order[1:])) for order in orders),2)
        for kwargs in (dict(rounds=12),dict(rounds=True),dict(samples=1)):
            with self.assertRaises(ValueError):self.module.make_plan('perf',**kwargs)

    def test_workload_identity_rejects_per_bag_or_policy_drift(self):
        self.module.workload_identity(self.provenance(),'meta26')
        for change in ('bag','source_table','offset','policy','hash','state'):
            p=self.provenance()
            if change=='bag':p['workload']['tables_detail'][0]['bag_lengths'][0]+=1
            if change=='source_table':p['workload']['tables_detail'][0]['source_table']+=1
            if change=='offset':p['workload']['tables_detail'][1]['offset_last']=1
            if change=='policy':p['workload']['routing_profile']['min_embedding_bag_length']=32
            if change=='hash':p['identity']['input_sha256']='0'*64
            if change=='state':p['identity']['state_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.module.workload_identity(p,'meta26')

    def test_configs_keep_one_native_runtime_and_bind_projection_and_routing(self):
        old=load(ROOT/'test/eager_parallel/test_controller_r28.py','r31_config_fixture')
        fixture=old.Controller();fixture.module=self.module
        with tempfile.TemporaryDirectory() as directory:
            base=fixture.base_config(directory);original=copy.deepcopy(base)
            configs=self.module.prepare_configs(base,Path(directory))
            self.assertEqual(base,original);self.assertEqual(set(configs),{'meta26'})
            config=configs['meta26']
            for key in ('package','python','runtime_identity','libraries','original_source','ld_library_path'):
                self.assertEqual(config[key],base[key])
            self.assertEqual(config['routing_profile'],dict(large_embeddings=2,small_embeddings=24,min_embedding_bag_length=64))
            self.assertEqual(config['reference'],str(Path(directory)/'benchmarks/eager_parallel_r31/workload.py'))
            self.assertTrue(config['validate_stages'])


if __name__=='__main__':
    unittest.main()
