"""Reject mislabeled selective scheduling and incomplete R31 evidence."""
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'benchmarks/eager_parallel_r31/host_e2e.py'


def load_worker(path=WORKER):
    spec = importlib.util.spec_from_file_location('r31_e2e', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def profile(large):
    return dict(large_embeddings=large, small_embeddings=26-large, min_embedding_bag_length=64)


def native_snapshot(arm, large=13, forwards=0):
    selective = arm in ('ae', 'aei')
    side = large if selective else 26
    interaction = arm == 'ei' or (arm == 'aei' and side > 0)
    scheduled = (side + (4 if interaction else 0)) * forwards
    row = dict(schema=1, backend='native_immediate', mode='ei' if arm in ('ei', 'aei') else 'e',
        protocol='synchronous_host_submit_groups', enabled=True, actual_stream_scheduling=True,
        real_sdk_qualified=False, device_parallel_qualified=False, host_workers=0,
        task_queue_required=0, capacity=64, byte_limit=536870912,
        min_embedding_bag_length=64 if selective else 0,
        embedding_length_policy='metadata_mean_indices_per_bag',
        caller_embedding_small=(26-side)*forwards, caller_interaction=4*forwards if arm == 'aei' and side == 0 else 0,
        small_dependency_drains=0, policy_uncertain_fallback=0, caller_frontier_dirty=selective and side == 0 and forwards > 0,
        caller_frontier_records=forwards if selective and side else 0,
        caller_frontier_waits=forwards if selective and side else 0,
        active_calls=0, retained_bytes=0, peak_retained_bytes=1024, retained_owners=0,
        owned_event_handles=129 if selective else 65,
        scheduled_embedding=side*forwards, scheduled_cat=2*forwards if interaction else 0,
        scheduled_bmm=forwards if interaction else 0, scheduled_index=forwards if interaction else 0,
        fallback=0, aliases=0, nested_entries=0, drains=forwards, budget_drains=0,
        failed=0, boundaries=forwards, event_records=scheduled + (forwards if side else 0),
        event_waits=forwards if side else 0, caller_joins=forwards if side else 0,
        first_error=None, quarantined=False, stream_ids=[0, 1, 2, 3],
        closure=dict(admission_closed=False, no_owned_device_work=not (selective and side == 0 and forwards > 0),
                     device_synchronized=False, passed=False))
    return row


def sdk_snapshot(worker, arm, large=13, forwards=1):
    operations = {name: dict(prepared=value*forwards, executed=value*forwards, failed=0, cancelled=0,
                            stream_executions={'11': value*forwards} if forwards else {})
                  for name, value in worker.ORIGINAL_COUNTS.items()}
    if arm != 'off':
        side = large if arm in ('ae', 'aei') else 26
        streams = {}
        if side < 26 and forwards:
            streams['11'] = (26-side)*forwards
        for handle, number in (('12', (side+1)//2), ('13', side//2)):
            if number and forwards:
                streams[handle] = number*forwards
        operations['aclnnEmbeddingBag']['stream_executions'] = streams
        if arm == 'ei' or (arm == 'aei' and side):
            for name in ('aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex'):
                operations[name]['stream_executions'] = {'14': operations[name]['executed']} if forwards else {}
    return dict(operations=operations, live_objects=0, descriptors_created=42*forwards,
                descriptors_destroyed=42*forwards)


class WorkerContract(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()

    def test_new_worker_exposes_five_pinned_arm_policies(self):
        self.assertTrue(WORKER.is_file(), 'R31 worker is missing')
        spec = importlib.util.spec_from_file_location('r31_e2e', WORKER)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.assertEqual(worker.ARMS, ('off', 'e', 'ei', 'ae', 'aei'))
        self.assertEqual(worker.ARM_POLICIES, {
            'off': dict(mode='shadow', min_embedding_bag_length=0),
            'e': dict(mode='e', min_embedding_bag_length=0),
            'ei': dict(mode='ei', min_embedding_bag_length=0),
            'ae': dict(mode='e', min_embedding_bag_length=64),
            'aei': dict(mode='ei', min_embedding_bag_length=64),
        })

    def test_frozen_runtime_sdk_and_loader_helpers_are_identical(self):
        frozen = load_worker(ROOT / 'benchmarks/eager_parallel_r27/host_e2e.py')
        for name in ('require', 'sha', 'load', 'count', 'check_runtime', 'sdk_delta', 'load_workload'):
            with self.subTest(helper=name):
                self.assertEqual(inspect.getsource(getattr(self.worker, name)), inspect.getsource(getattr(frozen, name)))
        self.assertEqual(sum(self.worker.ORIGINAL_COUNTS.values()), 42)
        self.assertEqual(self.worker.sha(ROOT / 'benchmarks/eager_parallel_r27/host_e2e.py'),
                         '66a752d345233f8f20a64b18d94b0966d609fcd490c31652567788c902d103ca')

    def test_meta26_loader_and_optional_stage_callbacks(self):
        calls = []
        reference = SimpleNamespace(load_original_model=lambda *args, **kwargs: calls.append((args, kwargs)))
        self.worker.load_workload(reference, 'upstream', dict(bag_length='meta26'))
        self.assertEqual(calls, [(('upstream',), dict(bag_length='meta26'))])
        with self.assertRaisesRegex(ValueError, 'stage callbacks'):
            self.worker.load_workload(reference, 'upstream', dict(validate_stages=True))

    def test_timing_call_and_stage_check_order_are_preserved(self):
        source = inspect.getsource(self.worker.run)
        frozen = inspect.getsource(load_worker(ROOT / 'benchmarks/eager_parallel_r27/host_e2e.py').run)
        measurement = "common.measure_forward_window(model, inputs, torch.npu.synchronize,\n                warmup=report['warmup'], samples=report['samples'], snapshot=snapshot)"
        self.assertIn(measurement, source)
        self.assertIn(measurement, frozen)
        positions = [source.index(value) for value in ('reference.capture_stages(', "model.to('npu:0')",
            'common.measure_forward_window(', "require(report['numeric_final']['passed']",
            'reference.validate_stages(', "report['closed'] = snapshot()")]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('.item(', source)
        self.assertIn("min_embedding_bag_length=policy['min_embedding_bag_length']", source)
        self.assertIn("routing=config['routing_profile']", source)

    def test_cli_accepts_all_arms_and_records_pinned_policy_and_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = folder / 'config.json'
            config.write_text(json.dumps(dict(artifact_pins={}, source_pins={}, routing_profile=profile(7))))
            real_run = self.worker.run
            for arm in self.worker.ARMS:
                output = folder / arm
                def run(config, report):
                    # Exercise the real metadata entry point before its pinned-runtime guard.
                    with self.assertRaisesRegex(ValueError, 'pinned Linux FT Python required'):
                        real_run(config, report)
                    report.update(passed=True, timing={})
                argv = ['host_e2e.py', '--config', str(config), '--output', str(output),
                        '--arm', arm, '--kind', 'cold', '--round', '0']
                with patch.object(self.worker.sys, 'argv', argv), patch.object(self.worker.sys, 'platform', 'win32'), \
                        patch.object(self.worker, 'run', side_effect=run), patch('builtins.print'):
                    self.worker.main()
                report = json.loads((output / 'result.json').read_text())
                self.assertEqual((report['schema'], report['arm'], report['warmup'], report['samples']), (1, arm, 0, 1))
                self.assertEqual(report['policy'], self.worker.ARM_POLICIES[arm])
                self.assertEqual(report['routing_profile'], profile(7))


class NativeGates(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()

    def gate(self, arm, large=13, forwards=3, before=None, after=None, routing=None):
        self.assertIn('routing', inspect.signature(self.worker.native_delta).parameters,
                      'native gate needs the metadata routing profile')
        return self.worker.native_delta(before if before is not None else native_snapshot(arm, large, 2),
            after if after is not None else native_snapshot(arm, large, 2+forwards), arm, forwards,
            routing=profile(large) if routing is None else routing)

    def test_five_arm_native_counts_follow_policy_not_label(self):
        off = dict(backend='shadow', enabled=False, entered=0)
        self.assertTrue(self.worker.native_delta(off, off, 'off', 3)['passed'])
        for arm in ('e', 'ei', 'ae', 'aei'):
            for large in (0, 1, 13, 26):
                with self.subTest(arm=arm, large=large):
                    result = self.gate(arm, large)
                    side = large if arm.startswith('a') else 26
                    self.assertEqual(result['scheduled_embedding'], side*3)
                    self.assertEqual(result['caller_embedding_small'], (26-side)*3)
                    self.assertEqual(result['caller_interaction'], 12 if arm == 'aei' and large == 0 else 0)

    def test_stage_delta_allows_healthy_closed_endpoint(self):
        after = native_snapshot('aei', 13, 5)
        after.update(enabled=False, owned_event_handles=0,
                     closure=dict(admission_closed=True, no_owned_device_work=True, device_synchronized=True, passed=True))
        self.assertTrue(self.gate('aei', after=after)['passed'])

    def test_auto_modes_require_valid_shape_only_profile(self):
        malformed = ({}, profile(-1), profile(27), dict(profile(13), large_embeddings=True),
                     dict(profile(13), small_embeddings=12), dict(profile(13), min_embedding_bag_length=0),
                     dict(profile(13), small_embeddings=13.0))
        for arm in ('ae', 'aei'):
            with self.assertRaisesRegex(ValueError, 'routing'):
                self.worker.native_delta(native_snapshot(arm), native_snapshot(arm, forwards=1), arm, 1)
            for routing in malformed:
                with self.subTest(arm=arm, routing=routing), self.assertRaisesRegex(ValueError, 'routing'):
                    self.gate(arm, routing=routing)
        for arm in ('e', 'ei'):
            self.worker.native_delta(native_snapshot(arm), native_snapshot(arm, forwards=1), arm, 1)

    def test_policy_and_health_tampering_are_rejected(self):
        changes = ({'mode': 'aei'}, {'min_embedding_bag_length': 0}, {'min_embedding_bag_length': True},
                   {'embedding_length_policy': 'offset_values'}, {'actual_stream_scheduling': False},
                   {'backend': 'shadow'}, {'first_error': 'failure'}, {'quarantined': True},
                   {'active_calls': 1}, {'retained_owners': 1}, {'retained_bytes': 4}, {'failed': 1},
                   {'stream_ids': [0, 1, 1, 3]}, {'caller_frontier_dirty': 0}, {'host_workers': 1},
                   {'task_queue_required': 1}, {'real_sdk_qualified': True}, {'device_parallel_qualified': True})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.gate('aei', after=dict(native_snapshot('aei', forwards=5), **change))

    def test_wrong_routing_policy_counters_and_dependencies_are_rejected(self):
        for name in ('scheduled_embedding', 'scheduled_cat', 'scheduled_bmm', 'scheduled_index',
                     'caller_embedding_small', 'caller_interaction', 'small_dependency_drains',
                     'policy_uncertain_fallback'):
            for arm, large in (('ei', 13), ('ae', 13), ('aei', 13), ('aei', 0)):
                after = native_snapshot(arm, large, 5)
                after[name] += 1
                with self.subTest(counter=name, arm=arm, large=large), self.assertRaises(ValueError):
                    self.gate(arm, large, after=after)
        for name in ('event_records', 'event_waits', 'caller_frontier_records', 'caller_frontier_waits'):
            before = native_snapshot('aei', 13, 2)
            after = native_snapshot('aei', 13, 5)
            after[name] = before[name]
            with self.subTest(counter=name), self.assertRaises(ValueError):
                self.gate('aei', before=before, after=after)

    def test_zero_side_and_full_modes_do_not_fabricate_frontier_work(self):
        for arm, large in (('e', 13), ('ei', 13), ('ae', 0), ('aei', 0)):
            for name in ('caller_frontier_records', 'caller_frontier_waits'):
                after = native_snapshot(arm, large, 5)
                after[name] += 1
                with self.subTest(arm=arm, counter=name), self.assertRaises(ValueError):
                    self.gate(arm, large, after=after)
        for arm in ('ae', 'aei'):
            self.gate(arm, 0)

    def test_nonzero_side_frontiers_are_clean_between_forwards(self):
        for arm in ('e', 'ei', 'ae', 'aei'):
            for endpoint in ('before', 'after'):
                row = native_snapshot(arm, 13, 2 if endpoint == 'before' else 5)
                row['caller_frontier_dirty'] = True
                with self.subTest(arm=arm, endpoint=endpoint), self.assertRaisesRegex(ValueError, 'frontier'):
                    self.gate(arm, **{endpoint: row})

    def test_zero_frontier_contract_is_cumulative(self):
        for arm, large in (('e', 13), ('ei', 13), ('ae', 0), ('aei', 0)):
            for name in ('caller_frontier_records', 'caller_frontier_waits'):
                before, after = native_snapshot(arm, large, 2), native_snapshot(arm, large, 5)
                before[name] = after[name] = 1
                with self.subTest(arm=arm, counter=name), self.assertRaisesRegex(ValueError, 'frontier'):
                    self.gate(arm, large, before=before, after=after)

    def test_native_gate_rejects_unknown_arm_and_nonpositive_forward_count(self):
        for arm, forwards in (('other', 1), ('ei', 0), ('ei', -1), ('ei', True), ('ei', 1.0)):
            with self.subTest(arm=arm, forwards=forwards), self.assertRaises(ValueError):
                self.worker.native_delta(native_snapshot('ei'), native_snapshot('ei', forwards=1), arm, forwards)


class StreamGates(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()

    def gate(self, arm, large=13, forwards=3, before=None, after=None, routing=None):
        self.assertIn('routing', inspect.signature(self.worker.sdk_stream_gate).parameters,
                      'stream gate needs the metadata routing profile')
        return self.worker.sdk_stream_gate(after if after is not None else sdk_snapshot(self.worker, arm, large, 2+forwards),
            arm, before if before is not None else sdk_snapshot(self.worker, arm, large, 2), forwards,
            routing=profile(large) if routing is None else routing)

    def test_all_five_arms_allow_only_exact_metadata_routes(self):
        for arm in self.worker.ARMS:
            for large in (0, 1, 2, 13, 25, 26):
                with self.subTest(arm=arm, large=large):
                    self.assertTrue(self.gate(arm, large)['passed'])
                    self.worker.sdk_stream_gate(sdk_snapshot(self.worker, arm, large, 3), arm, routing=profile(large))

    def test_auto_stream_modes_require_valid_routing(self):
        for arm in ('ae', 'aei'):
            with self.assertRaisesRegex(ValueError, 'routing'):
                self.worker.sdk_stream_gate(sdk_snapshot(self.worker, arm), arm)
            for routing in ({}, dict(profile(13), small_embeddings=12), dict(profile(13), large_embeddings=True)):
                with self.assertRaisesRegex(ValueError, 'routing'):
                    self.gate(arm, routing=routing)

    def test_embedding_counts_cannot_hide_wrong_caller_or_producer_routes(self):
        for streams in ({'11': 40, '12': 20, '13': 18}, {'11': 36, '12': 21, '13': 21},
                        {'11': 39, '12': 39}, {'12': 39, '13': 39},
                        {'11': 39, '12': 18, '13': 18, '15': 3}):
            after = sdk_snapshot(self.worker, 'aei', 13, 3)
            after['operations']['aclnnEmbeddingBag']['stream_executions'] = streams
            # Counts are all for the exact timed window, avoiding warmup contamination.
            with self.subTest(streams=streams), self.assertRaises(ValueError):
                self.gate('aei', after=after, before=sdk_snapshot(self.worker, 'aei', 13, 0))

    def test_timed_window_cannot_borrow_second_producer_from_warmup(self):
        before = sdk_snapshot(self.worker, 'ei', 13, 2)
        after = sdk_snapshot(self.worker, 'ei', 13, 3)
        after['operations']['aclnnEmbeddingBag']['stream_executions'] = {'12': 52, '13': 26}
        with self.assertRaises(ValueError):
            self.gate('ei', forwards=1, before=before, after=after)

    def test_every_interaction_kernel_obeys_consumer_or_caller_policy(self):
        for arm, large, wrong in (('off', 13, '14'), ('e', 13, '14'), ('ei', 13, '11'),
                                  ('ae', 13, '14'), ('aei', 13, '11'), ('aei', 0, '14')):
            for name in ('aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex'):
                after = sdk_snapshot(self.worker, arm, large, 3)
                after['operations'][name]['stream_executions'] = {wrong: after['operations'][name]['executed']}
                with self.subTest(arm=arm, large=large, operation=name), self.assertRaises(ValueError):
                    self.gate(arm, large, before=sdk_snapshot(self.worker, arm, large, 0), after=after)

    def test_consumer_cannot_overlap_any_embedding_or_mlp_stream(self):
        for handle in ('11', '12', '13'):
            after = sdk_snapshot(self.worker, 'aei', 13, 3)
            for name in ('aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex'):
                after['operations'][name]['stream_executions'] = {handle: after['operations'][name]['executed']}
            with self.subTest(handle=handle), self.assertRaises(ValueError):
                self.gate('aei', before=sdk_snapshot(self.worker, 'aei', 13, 0), after=after)

    def test_correct_totals_do_not_allow_wrong_original_operation_count(self):
        for name in self.worker.ORIGINAL_COUNTS:
            after = sdk_snapshot(self.worker, 'aei', 13, 3)
            operation = after['operations'][name]
            operation['executed'] += 1
            handle = next(iter(operation['stream_executions']))
            operation['stream_executions'][handle] += 1
            with self.subTest(operation=name), self.assertRaises(ValueError):
                self.gate('aei', before=sdk_snapshot(self.worker, 'aei', 13, 0), after=after)

    def test_mlp_requires_one_shared_caller_and_exact_kernel_counts(self):
        for name in ('aclnnAddmm', 'aclnnRelu', 'aclnnSigmoid'):
            after = sdk_snapshot(self.worker, 'aei', 13, 3)
            operation = after['operations'][name]
            operation['stream_executions'] = {'15': operation['executed']}
            with self.subTest(operation=name), self.assertRaises(ValueError):
                self.gate('aei', before=sdk_snapshot(self.worker, 'aei', 13, 0), after=after)

    def test_counter_regression_boolean_and_zero_stream_counts_fail(self):
        for value in (-1, True, 0, 1.0):
            after = sdk_snapshot(self.worker, 'aei', 13, 3)
            after['operations']['aclnnEmbeddingBag']['stream_executions']['12'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate('aei', before=sdk_snapshot(self.worker, 'aei', 13, 0), after=after)

    def test_stream_gate_rejects_unknown_arm_and_nonpositive_forward_count(self):
        for arm, forwards in (('other', 1), ('ei', 0), ('ei', -1), ('ei', True), ('ei', 1.0)):
            with self.subTest(arm=arm, forwards=forwards), self.assertRaises(ValueError):
                self.worker.sdk_stream_gate(sdk_snapshot(self.worker, 'ei'), arm, forwards=forwards)


class PairedStatistics(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()
        self.rows = [dict(round=i, arm=arm, median_ns=value, p95_ns=value*2, passed=True,
                          kind='perf', samples=3000, window_wall_ns=2_000_000_000)
                     for i, values in enumerate(((100, 50, 200, 25, 400), (10, 20, 5, 40, 2.5)))
                     for arm, value in zip(self.worker.ARMS, values)]

    def test_all_four_candidates_use_per_round_off_ratios(self):
        result = self.worker.paired_summary(self.rows, 2)
        self.assertEqual(set(result), {'e', 'ei', 'ae', 'aei'})
        self.assertEqual(result['e']['paired_speedups'], [2.0, 0.5])
        self.assertEqual(result['ei']['paired_speedups'], [0.5, 2.0])
        self.assertEqual(result['ae']['paired_speedups'], [4.0, 0.25])
        self.assertEqual(result['aei']['paired_speedups'], [0.25, 4.0])
        self.assertEqual(result['e']['paired_speedup_median'], 1.25)
        self.assertEqual(result['e']['paired_latency_change_pct'], 25.0)
        self.assertEqual(result['aei']['faster_rounds'], 1)

    def test_missing_duplicate_unknown_or_wrong_round_arms_fail(self):
        variants = (self.rows[:-1], self.rows + [self.rows[0]],
                    self.rows[:-1] + [dict(self.rows[-1], arm='unknown')],
                    self.rows[:-1] + [dict(self.rows[-1], round=2)])
        for rows in variants:
            with self.assertRaises(ValueError):
                self.worker.paired_summary(rows, 2)

    def test_unsuccessful_nonfinite_or_mismatched_windows_fail(self):
        for change in ({'passed': False}, {'kind': 'cold'}, {'samples': 1}, {'samples': True}, {'round': True},
                       {'median_ns': 0}, {'median_ns': True}, {'p95_ns': float('nan')}, {'median_ns': float('inf')},
                       {'window_wall_ns': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.worker.paired_summary([dict(self.rows[0], **change)] + self.rows[1:], 2)


if __name__ == '__main__':
    unittest.main()
