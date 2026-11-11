import importlib.util
from pathlib import Path
import unittest


class PhaseAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2]/'benchmarks/eager_parallel_r33/analyze.py'
        spec = importlib.util.spec_from_file_location('r33_analysis_test', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_overlapping_and_adjacent_intervals(self):
        self.assertEqual(self.module.merge([(5, 15), (0, 10), (15, 20), (24, 30)]),
                         [(0, 20), (24, 30)])

    def test_clip(self):
        self.assertEqual(self.module.clip([(0, 10), (5, 15), (18, 21)], 2, 12), [(2, 12)])

    def test_intersection_endpoints_have_no_duration(self):
        self.assertEqual(self.module.intersection([(0, 10)], [(10, 20)]), [])
        self.assertEqual(self.module.intersection([(0, 10), (12, 20)], [(5, 15)]),
                         [(5, 10), (12, 15)])

    def test_epoch_precision_does_not_round_away_nanoseconds(self):
        self.assertEqual(self.module.ns('1800000000000000.007'), 1800000000000000007)
        self.assertEqual(self.module.ns('0.013'), 13)

    def test_device_sum_is_distinct_from_busy_union(self):
        row = self.module.device_window([(2, 6), (4, 9), (9, 10)], 0, 8, 12)
        self.assertEqual(row['kernel_sum_ns'], 10)
        self.assertEqual(row['kernel_union_ns'], 8)
        self.assertEqual(row['kernel_overlap_ns'], 2)
        self.assertEqual(row['kernel_final_sync_ns'], 2)
        self.assertEqual(row['kernel_forward_ns'], 6)

    def test_host_partition_adds_once(self):
        counters = dict(forward=[dict(category='event', exclusive_ns=10),
                                  dict(category='sync', exclusive_ns=3),
                                  dict(category='copy_sync', exclusive_ns=2),
                                  dict(category='launch', exclusive_ns=40)],
                        final_sync=[dict(category='sync', exclusive_ns=8)])
        row = self.module.host_window(dict(samples=1, rows=[dict(forward_ns=80,
            final_sync_ns=20, e2e_ns=100)], api_counters=counters))
        self.assertEqual(row['host_prepare_submit_ns'], 65)
        self.assertEqual(row['event_ns'], 10)
        self.assertEqual(row['internal_sync_ns'], 5)
        self.assertEqual(row['final_sync_ns'], 20)
        self.assertEqual(row['sdk_launch_ns'], 40)
        self.assertEqual(row['host_other_ns'], 25)
        self.assertEqual(row['partition_sum_ns'], 100)

    def test_summary_lane_is_not_counted_as_another_kernel(self):
        events = [dict(ph='M', name='process_name', pid=1, args={'name':'Ascend Hardware'}),
                  dict(ph='M', name='process_name', pid=2, args={'name':'Overlap Analysis'}),
                  dict(ph='X', name='aclnnRelu', pid=1, tid=7, ts='1.0', dur='0.5',
                       args={'Task Type':'AI_VECTOR_CORE'}),
                  dict(ph='X', name='Computing', pid=2, tid=0, ts='1.0', dur='0.5')]
        self.assertEqual(len(self.module.kernels(events)), 1)

    def test_unknown_device_task_is_explicit(self):
        events = [dict(ph='M', name='process_name', pid=1, args={'name':'Ascend Hardware'}),
                  dict(ph='X', name='aclnnUnknown', pid=1, tid=7, ts='1.0', dur='0.5',
                       args={'Task Type':'UNRECOGNIZED_COMPUTE'})]
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.module.kernels(events)

    def test_level0_kernel_type_names(self):
        events = [dict(ph='M', name='process_name', pid=1, args={'name':'Ascend Hardware'})]
        for kind in ('KERNEL_AICORE', 'KERNEL_AIVEC'):
            events.append(dict(ph='X', name='aclnnCompute', pid=1, tid=7, ts='1.0', dur='0.5',
                               args={'Task Type':kind}))
        self.assertEqual(len(self.module.kernels(events)), 2)

    def test_device_order_groups_are_independent_of_host_clock_offset(self):
        events = [dict(name='work', ts='1', dur='1'),
                  dict(name='aclnnSigmoid_last', ts='3', dur='1'),
                  dict(name='work', ts='8', dur='2'),
                  dict(name='aclnnSigmoid_last', ts='12', dur='1')]
        groups = self.module.device_groups(events, 2, expected_count=2)
        self.assertEqual([[r['name'] for r in g] for g in groups],
                         [['work', 'aclnnSigmoid_last'], ['work', 'aclnnSigmoid_last']])


if __name__ == '__main__':
    unittest.main()
