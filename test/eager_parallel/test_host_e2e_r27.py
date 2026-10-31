"""Gate actual scheduling and numerical evidence, not a renamed shadow run."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Gates(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('r27_e2e', ROOT / 'benchmarks/eager_parallel_r27/host_e2e.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_expected_original_sdk_counts(self):
        operations = {name: dict(prepared=count * 3, executed=count * 3, failed=0, cancelled=0)
                      for name, count in self.module.ORIGINAL_COUNTS.items()}
        old = dict(operations={}, live_objects=0, descriptors_created=0, descriptors_destroyed=0)
        new = dict(operations=operations, live_objects=0, descriptors_created=100, descriptors_destroyed=100)
        self.module.sdk_delta(old, new, 3)
        new['operations']['aclnnIndex']['executed'] -= 1
        with self.assertRaisesRegex(ValueError, 'aclnnIndex'):
            self.module.sdk_delta(old, new, 3)

    def test_noop_or_incomplete_runtime_is_rejected(self):
        good = dict(backend_kind='HOST_SIM_R27_CPU', submitted=8, completed=8, errors=0,
                    numeric_supported=True, real_sdk_qualified=False, injected_faults=0, fault_active=False)
        self.module.check_runtime(good)
        for change in ({'backend_kind': 'HOST_SIM_R25'}, {'completed': 7}, {'errors': 1},
                       {'numeric_supported': False}, {'real_sdk_qualified': True}):
            with self.assertRaises(ValueError):
                self.module.check_runtime(dict(good, **change))

    def test_native_ei_requires_actual_scheduled_pipeline(self):
        before = dict(backend='native_immediate', mode='ei', actual_stream_scheduling=True,
            stream_ids=[0, 1, 2, 3], first_error=None, quarantined=False,
            active_calls=0, retained_owners=0, failed=0, event_records=0, event_waits=0,
            scheduled_embedding=0, scheduled_cat=0, scheduled_bmm=0, scheduled_index=0)
        after = dict(before, scheduled_embedding=26, scheduled_cat=2, scheduled_bmm=1,
                     scheduled_index=1, event_records=33, event_waits=8)
        self.module.native_delta(before, after, 'ei', 1)
        for change in ({'backend': 'shadow'}, {'scheduled_index': 0}, {'first_error': 'failure'},
                       {'stream_ids': [0, 0, 0, 0]}, {'actual_stream_scheduling': False}):
            with self.assertRaises(ValueError):
                self.module.native_delta(before, dict(after, **change), 'ei', 1)

    def test_sdk_stream_handles_prove_actual_three_arm_routing(self):
        import copy
        for arm in ('off', 'e', 'ei'):
            operations = {name: dict(executed=value, stream_executions={'11': value})
                          for name, value in self.module.ORIGINAL_COUNTS.items()}
            if arm != 'off':
                operations['aclnnEmbeddingBag']['stream_executions'] = {'12': 13, '13': 13}
            if arm == 'ei':
                for name in ('aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex'):
                    operations[name]['stream_executions'] = {'14': operations[name]['executed']}
            self.module.sdk_stream_gate(dict(operations=operations), arm)
            if arm == 'ei':
                bad = copy.deepcopy(operations)
                bad['aclnnIndex']['stream_executions'] = {'11': 1}
                with self.assertRaises(ValueError):
                    self.module.sdk_stream_gate(dict(operations=bad), arm)
                after = copy.deepcopy(operations)
                for value in after.values():
                    value['executed'] *= 2
                    value['stream_executions'] = {key: count * 2 for key, count in value['stream_executions'].items()}
                self.module.sdk_stream_gate(dict(operations=after), arm, dict(operations=operations), 1)
                # Warmup used both producers, but the timed window used only one.
                after['aclnnEmbeddingBag']['stream_executions'] = {'12': 39, '13': 13}
                with self.assertRaises(ValueError):
                    self.module.sdk_stream_gate(dict(operations=after), arm, dict(operations=operations), 1)

    def test_paired_statistics_do_not_pool_or_divide_aggregate_medians(self):
        rows = [dict(round=i, arm=arm, median_ns=value, p95_ns=value, passed=True,
                     kind='perf', samples=3000, window_wall_ns=2_000_000_000)
                for i, values in enumerate(((100, 50, 200), (10, 20, 5)))
                for arm, value in zip(('off', 'e', 'ei'), values)]
        result = self.module.paired_summary(rows, 2)
        self.assertEqual(result['e']['paired_speedup_median'], 1.25)
        self.assertEqual(result['ei']['paired_speedup_median'], 1.25)
        self.assertEqual(result['e']['paired_latency_change_pct'], 25.0)
        with self.assertRaises(ValueError):
            self.module.paired_summary(rows[:-1], 2)
        for change in ({'passed': False}, {'kind': 'cold'}, {'samples': 1}, {'median_ns': 0},
                       {'p95_ns': float('nan')}, {'window_wall_ns': 1}):
            with self.assertRaises(ValueError):
                self.module.paired_summary([dict(rows[0], **change)] + rows[1:], 2)


if __name__ == '__main__':
    unittest.main()
