"""Host-only fixtures. These tests are not host-sim or device execution evidence."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmarks/hostsim_r25/contracts.py'


def load():
    if not SOURCE.exists():
        return None
    spec = importlib.util.spec_from_file_location('_hostsim_r25_contracts', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContractsTests(unittest.TestCase):
    def setUp(self):
        self.m = load()
        self.assertIsNotNone(self.m, 'dedicated host-sim contracts must exist')

    def config(self):
        pin = dict(path='/isolated/new/libopapi.so', sha256='a' * 64)
        return dict(schema=1, cpus=[144,146,148,150,152,154,156,158], numa_node=1,
            helpers=dict(r22=dict(path='/tools/real_dlrm_runner_r22.py', sha256='a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'),
                         topology=dict(path='/tools/measurement_contracts_r22.py', sha256='5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a')),
            sdk=dict(backend='host_sim', real_sdk_qualified=False, source_pins={'/isolated/source/sdk.cpp': 'b'*64},
                     libraries={'libopapi.so': pin, 'libascendcl.so':dict(path='/isolated/new/libascendcl.so',sha256='c'*64)}),
            runtimes=[dict(name='ft',python='/venv/bin/python3',expected_gil=False,
                pythonpath=['/isolated/package'],ld_library_path=['/isolated/new'],
                artifact_pins={'/isolated/package/torch_npu/lib/libtorch_npu.so':'d'*64},
                torch_npu_package='/isolated/package/torch_npu',
                model_manifest=dict(path='/assets/ft-dlrm-assets.json',sha256='e'*64))])

    def test_three_round_four_arm_plan_is_counterbalanced_and_small(self):
        plan = self.m.make_plan(self.config())
        self.assertEqual(len(plan['cold']), 4)
        self.assertEqual(len(plan['perf']), 12)
        for round_id in range(3):
            rows=[r for r in plan['perf'] if r['round']==round_id]
            self.assertEqual({r['arm'] for r in rows},{'off','queue_w1','queue_w2','queue_w4'})
            self.assertEqual({r['warmup'] for r in rows},{20})
            self.assertEqual({r['samples'] for r in rows},{200})
        self.assertNotEqual([r['arm'] for r in plan['perf'][:4]],[r['arm'] for r in plan['perf'][4:8]])

    def test_natural_gil_and_exact_sim_scope_cannot_be_overridden(self):
        for mutate in (lambda c:c.update(env={'PYTHON_GIL':'0'}),
                       lambda c:c['sdk'].update(real_sdk_qualified=True),
                       lambda c:c['sdk'].update(backend='real'),
                       lambda c:c['runtimes'][0].update(expected_gil=True)):
            c=self.config();mutate(c)
            with self.assertRaises(ValueError): self.m.validate_config(c)

    def test_exact_cpu_budget_and_new_sdk_pair_are_required(self):
        for mutate in (lambda c:c.update(cpus=[1]*8),lambda c:c.update(cpus=list(range(7))),
                       lambda c:c['sdk']['libraries'].pop('libascendcl.so'),
                       lambda c:c['sdk'].update(source_pins={})):
            c=self.config();mutate(c)
            with self.assertRaises(ValueError): self.m.validate_config(c)

    def test_runtime_python_lexical_path_is_preserved(self):
        self.assertEqual(self.m.validate_config(self.config())['runtimes'][0]['python'],'/venv/bin/python3')

    def test_legacy_dependency_paths_cannot_precede_native_interpreter_packages(self):
        config=self.config();config['runtimes'][0]['pythonpath'].append('/legacy/deps-ft')
        with self.assertRaisesRegex(ValueError,'candidate-only'):
            self.m.validate_config(config)
        with self.assertRaisesRegex(ValueError,'candidate-only'):
            self.m.worker_environment(config,config['runtimes'][0],self.m.make_plan(self.config())['cold'][0],{})

    def test_clean_environment_removes_gil_injection_and_keeps_new_mapping(self):
        c=self.config();row=self.m.make_plan(c)['perf'][0]
        env=self.m.worker_environment(c,c['runtimes'][0],row,{'PATH':'/bin','PYTHON_GIL':'1',
            'LD_PRELOAD':'evil.so','PYTHONPATH':'/old','FTLAB_HOST_REGIONS':'1','OMP_NUM_THREADS':'9'})
        self.assertNotIn('PYTHON_GIL',env)
        self.assertNotIn('LD_PRELOAD',env)
        self.assertEqual(env['PYTHONPATH'],'/isolated/package')
        self.assertEqual(env['LD_LIBRARY_PATH'],'/isolated/new')
        self.assertEqual(env['OMP_NUM_THREADS'],'1')
        self.assertEqual(env['QWEN_ACL_HOST_SIM'],'1')
        self.assertEqual(env['FTLAB_HOST_REGIONS'],'0')

    def snap(self, queued=0):
        return dict(schema=1,backend='host_sim',real_sdk_qualified=False,simulated_prepare=True,
            qualified_ops=[],qualified_sim_ops=['aclnnEmbeddingBag'],first_error=None,
            order_evidence=dict(passed=True,admitted_count=queued,executed_count=queued,admitted_sequence_hash=queued,executed_sequence_hash=queued),
            operators={'aclnnEmbeddingBag':dict(seen=queued,eligible=queued,queued=queued,
                prepared=queued,executed=queued,returned=queued,fallback=0)},
            queue=dict(admitted=queued,retired=queued,slots_in_use=0,retained_bytes=0,workers_started=4),
            closure=dict(admission_closed=True,fifo_drained=True,device_synchronized=True,
                all_retired=True,workers_stopped=True,close_attempted=True,passed=True))

    def test_queue_gate_requires_actual_embedding_commands_and_balanced_counts(self):
        before=self.snap();after=self.snap(52)
        gate=self.m.validate_mechanism(before,after,self.snap(52),forwards=2,workers=4)
        self.assertEqual(gate['embedding_commands_per_forward'],26)
        self.assertEqual(gate['coverage_scope'],'partial native operator coverage')
        self.assertFalse(gate['real_sdk_qualified'])
        for field in ('prepared','executed','returned'):
            bad=copy.deepcopy(after);bad['operators']['aclnnEmbeddingBag'][field]-=1
            with self.assertRaises(ValueError):self.m.validate_mechanism(before,bad,self.snap(52),forwards=2,workers=4)

    def test_fixed_dlrm_rejects_balanced_extra_or_missing_native_embedding_commands(self):
        for forwards,count in ((1,52),(1,25),(2,104),(2,50)):
            with self.subTest(forwards=forwards,count=count), self.assertRaises(ValueError):
                self.m.validate_mechanism(self.snap(),self.snap(count),self.snap(count),forwards=forwards,workers=4)

    def test_all_fallback_cannot_pass_as_pipeline(self):
        with self.assertRaises(ValueError):self.m.validate_mechanism(self.snap(),self.snap(),self.snap(),forwards=2,workers=4)

    def test_close_requires_no_live_queue_resource_and_preserves_first_error(self):
        for mutate in (lambda s:s['queue'].update(slots_in_use=1),
                       lambda s:s['closure'].update(workers_stopped=False),
                       lambda s:s.update(first_error='prepare failed')):
            closed=self.snap(26);mutate(closed)
            with self.assertRaises(ValueError):self.m.validate_mechanism(self.snap(),self.snap(26),closed,forwards=1,workers=4)

    def test_disabled_arm_has_no_started_core_to_close(self):
        snapshot=self.snap();snapshot['operators']={};snapshot['queue']['workers_started']=0
        snapshot['closure']={key:False for key in snapshot['closure']}
        snapshot['closure']['admission_closed']=snapshot['closure']['all_retired']=True
        gate=self.m.validate_mechanism(snapshot,snapshot,snapshot,forwards=1,workers=0)
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['optimization_active'])

    def test_cold_diagnostics_require_closed_sequence_hash_and_distinct_tids(self):
        native=self.snap(26)
        native['order_evidence']=dict(passed=True,admitted_count=26,executed_count=26,admitted_sequence_hash=123,executed_sequence_hash=123)
        native['diagnostic_trace']=dict(enabled=True,complete=True,dropped=0,incomplete=False,
            events=[dict(native_tid=20+i%4,caller_tid=10,consumer_tid=30) for i in range(26)])
        sdk=dict(tracing_enabled=True,trace_dropped=0,prepare_tids=[20,21,22,23],execute_tids=[30])
        self.assertTrue(self.m.validate_cold(native,sdk,4)['passed'])
        native['order_evidence']['executed_sequence_hash']=124
        with self.assertRaises(ValueError):self.m.validate_cold(native,sdk,4)

    def test_fixed_cold_forward_rejects_extra_or_missing_order_counts_and_traces(self):
        sdk=dict(tracing_enabled=True,trace_dropped=0,prepare_tids=[20,21,22,23],execute_tids=[30])
        for count,traces in ((52,52),(52,26),(25,26),(26,27),(26,25)):
            native=self.snap(count)
            native['diagnostic_trace']=dict(enabled=True,complete=True,dropped=0,incomplete=False,
                events=[dict(native_tid=20+i%4,caller_tid=10,consumer_tid=30) for i in range(traces)])
            with self.subTest(count=count,traces=traces), self.assertRaises(ValueError):
                self.m.validate_cold(native,sdk,4)

    def test_sdk_prepare_and_execute_delta_matches_native_embedding_admission(self):
        before=dict(operations={'aclnnEmbeddingBag':dict(prepared=5,executed=5,cancelled=0)})
        after=dict(operations={'aclnnEmbeddingBag':dict(prepared=57,executed=57,cancelled=0)})
        gate=self.m.validate_sdk_delta(before,after,forwards=2,queued_embeddings=52)
        self.assertEqual(gate['embedding_commands_per_forward'],26)
        after['operations']['aclnnEmbeddingBag']['executed']=56
        with self.assertRaises(ValueError):self.m.validate_sdk_delta(before,after,forwards=2,queued_embeddings=52)

    def test_fixed_dlrm_rejects_matching_sdk_and_native_overcounts_and_sdk_undercounts(self):
        before=dict(operations={'aclnnEmbeddingBag':dict(prepared=5,executed=5,cancelled=0)})
        for forwards,count in ((1,52),(1,25),(2,104),(2,50)):
            after=dict(operations={'aclnnEmbeddingBag':dict(prepared=5+count,executed=5+count,cancelled=0)})
            for queued in (0,count):
                with self.subTest(forwards=forwards,count=count,queued=queued), self.assertRaises(ValueError):
                    self.m.validate_sdk_delta(before,after,forwards=forwards,queued_embeddings=queued)

    def test_stub_mapping_rejects_missing_duplicate_and_wrong_sha(self):
        pins=self.config()['sdk']['libraries']
        rows=[dict(name=k,**v) for k,v in pins.items()]
        self.m.validate_mapping(rows,pins)
        for bad in (rows[:1], rows+rows[:1], [dict(rows[0],sha256='0'*64),rows[1]]):
            with self.assertRaises(ValueError):self.m.validate_mapping(bad,pins)

    def test_summary_pairs_by_round_without_pooling_correlated_samples(self):
        rows=[]
        for r in range(3):
            for arm,mult in [('off',1),('queue_w1',.9),('queue_w2',1.2),('queue_w4',1)]:
                rows.append(dict(runtime='ft',round=r,arm=arm,durations_ns=[int(1000*mult)]*200))
        summary=self.m.summarize(rows)
        self.assertAlmostEqual(summary['ft']['queue_w1']['median_paired_speedup'],1/.9)
        self.assertEqual(summary['ft']['queue_w1']['paired_rounds'],3)
        self.assertEqual(summary['ft']['queue_w2']['classification'],'regression observed in first screen')
        self.assertFalse(summary['ft']['queue_w1']['true_device_kernel_claim'])

    def test_first_screen_rejects_missing_round_arm(self):
        with self.assertRaises(ValueError):self.m.summarize([])


if __name__=='__main__':unittest.main()
