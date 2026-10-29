import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class HostE2ETests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'benchmarks/eager_parallel_r26/host_e2e.py'
        self.assertTrue(path.is_file(), 'independent original-model E2E worker missing')
        spec = importlib.util.spec_from_file_location('r26_host_e2e_test', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_window_excludes_warmup_and_snapshots_and_synchronizes_each_forward(self):
        ticks = dict(wall=0, cpu=0, voluntary=0, involuntary=0)
        events = []
        inputs = (object(), object(), object())
        token = object()

        def model(*actual):
            self.assertEqual(actual, inputs)
            events.append('model')
            ticks['wall'] += 10
            ticks['cpu'] += 2
            ticks['voluntary'] += 1
            return token

        def sync():
            events.append('sync')
            ticks['wall'] += 2

        def snapshot():
            events.append('snapshot')
            ticks['wall'] += 1000
            ticks['cpu'] += 200
            return dict(sequence=len(events))

        row, result = self.module.measure_forward_window(model, inputs, sync,
            warmup=2, samples=3, snapshot=snapshot,
            wall_clock=lambda: ticks['wall'], cpu_clock=lambda: ticks['cpu'],
            usage=lambda: dict(voluntary=ticks['voluntary'], involuntary=0, max_rss_kib=12))
        self.assertIs(result, token)
        self.assertEqual(events, ['model', 'sync'] * 2 + ['snapshot'] +
                         ['model', 'sync'] * 3 + ['snapshot'])
        self.assertEqual(row['durations_ns'], [12, 12, 12])
        self.assertEqual(row['window_wall_ns'], 36)
        self.assertEqual(row['window_process_cpu_ns'], 6)
        self.assertEqual(row['usage_delta']['voluntary'], 3)
        self.assertEqual(row['forward_calls'], 5)
        self.assertEqual(row['synchronize_calls'], 5)

    def test_model_failure_stops_samples_without_replacing_original_error(self):
        first = RuntimeError('original model failure')
        events = []

        def model():
            events.append('model')
            raise first

        with self.assertRaises(RuntimeError) as raised:
            self.module.measure_forward_window(model, (), lambda: events.append('sync'),
                warmup=0, samples=2, snapshot=lambda: {}, usage=lambda: {})
        self.assertIs(raised.exception, first)
        self.assertEqual(events, ['model'])

    def test_noop_output_reads_only_shape_dtype_and_device(self):
        class MetadataOnly:
            shape = (128, 1)
            dtype = 'torch.float32'
            device = 'npu:0'

            def __getattr__(self, name):
                raise AssertionError('NoOp value access: ' + name)

        self.assertEqual(self.module.output_metadata(MetadataOnly()),
                         dict(shape=[128, 1], dtype='torch.float32', device='npu:0',
                              numeric_validated=False))

    @staticmethod
    def sdk(count=0):
        return dict(backend_kind='HOST_SIM_R25', numeric_supported=False,
            real_sdk_qualified=False, artificial_prepare_delay_ns=0,
            tracing_enabled=False, errors=0, cache_stub_calls=0,
            live_descriptors=0, live_executors=0, huge_pending=0, huge_active_threads=0,
            descriptors_created=count, descriptors_destroyed=count,
            executors_created=count, executors_executed=count, executors_cancelled=0,
            huge_init=count, huge_uninit=count,
            operations={'aclnnEmbeddingBag': dict(prepared=count, executed=count, cancelled=0)})

    def test_sdk_gate_rejects_missing_extra_or_cancelled_original_embeddings(self):
        before, after = self.sdk(52), self.sdk(130)
        self.assertEqual(self.module.validate_sdk_delta(before, after, 3)['executed'], 78)
        for amount in (77, 79):
            wrong = self.sdk(52 + amount)
            with self.assertRaisesRegex(ValueError, '26'):
                self.module.validate_sdk_delta(before, wrong, 3)
        after['operations']['aclnnEmbeddingBag']['cancelled'] = 1
        with self.assertRaises(ValueError):
            self.module.validate_sdk_delta(before, after, 3)

    def test_sdk_gate_rejects_simulation_delay_trace_and_leaked_resources(self):
        self.module.validate_sdk(self.sdk())
        for field, value in [('artificial_prepare_delay_ns', 1), ('tracing_enabled', True),
                             ('live_descriptors', 1), ('huge_pending', 1), ('errors', 1)]:
            row = self.sdk()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.validate_sdk(row)

    @staticmethod
    def native(forwards):
        rows = {
            'aten::_embedding_bag_forward_only': dict(entered=26, completed=26, failed=0, inputs=78, outputs=104),
            'aten::cat': dict(entered=2, completed=2, failed=0, inputs=29, outputs=2),
            'aten::bmm': dict(entered=1, completed=1, failed=0, inputs=2, outputs=1),
            'aten::index.Tensor': dict(entered=1, completed=1, failed=0, inputs=3, outputs=1)}
        return dict(schema=1, backend='shadow', enabled=True, entered=30*forwards,
            completed=30*forwards, failed=0, nested_entries=0, active_calls=0,
            foreign_thread_calls=0, first_error=None, coverage_incomplete=False,
            real_sdk_qualified=False, device_parallel_qualified=False, host_workers=0,
            operator_counts={name: {key: count*forwards for key, count in row.items()}
                             for name, row in rows.items()}, boundaries={})

    def test_native_timed_delta_excludes_warmup_and_catches_counter_regression(self):
        before, after = self.native(2), self.native(5)
        row = self.module.validate_native_delta(before, after, 'shadow', 3)
        self.assertEqual(row['embedding_calls'], 78)
        after['operator_counts']['aten::cat']['completed'] -= 1
        with self.assertRaises(ValueError):
            self.module.validate_native_delta(before, after, 'shadow', 3)
        with self.assertRaises(ValueError):
            self.module.validate_native_delta(self.native(5), self.native(2), 'shadow', 3)

    def test_disabled_native_gate_rejects_any_entry(self):
        off = self.native(0)
        off['enabled'] = False
        self.assertTrue(self.module.validate_native_delta(off, off, 'off', 3)['passed'])
        after = copy.deepcopy(off)
        after['entered'] = 1
        with self.assertRaises(ValueError):
            self.module.validate_native_delta(off, after, 'off', 3)

    def test_disabled_native_gate_rejects_stale_operator_counters(self):
        off = self.native(0)
        off['enabled'] = False
        off['operator_counts']['aten::cat']['completed'] = 1
        with self.assertRaises(ValueError):
            self.module.validate_native_delta(off, off, 'off', 3)

    def test_summary_pairs_process_medians_without_pooling_or_accepting_cold(self):
        rows = []
        for round_index, values in enumerate(((100, 50), (1000, 1000))):
            for arm, value in zip(('off', 'shadow'), values):
                rows.append(dict(passed=True, kind='perf', arm=arm, round=round_index,
                    samples=2, warmup=50, host_latency_measured=True,
                    durations_ns=[value, value], window_wall_ns=2_000_000_000))
        result = self.module.summarize_pairs(rows, rounds=2, samples=2)
        self.assertEqual([row['median_ratio_off_over_shadow'] for row in result['pairs']], [2, 1])
        self.assertEqual(result['median_paired_ratio'], 1.5)
        self.assertFalse(result['speedup_claim_proven'])
        with self.assertRaises(ValueError):
            self.module.summarize_pairs(rows[:-1], rounds=2, samples=2)
        rows[0]['kind'] = 'cold'
        with self.assertRaises(ValueError):
            self.module.summarize_pairs(rows, rounds=2, samples=2)

    def test_late_failure_cannot_write_pass_and_existing_output_cannot_be_overwritten(self):
        def late_failure(config, output, report):
            report['passed'] = True
            raise RuntimeError('late artifact mismatch')

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = folder / 'config.json'
            config.write_text('{}')
            argv = ['--config', str(config), '--output', str(folder/'out'),
                    '--arm', 'off', '--kind', 'cold', '--round', '0', '--warmup', '0', '--samples', '1']
            with patch.object(self.module, 'run', side_effect=late_failure):
                with self.assertRaisesRegex(RuntimeError, 'late artifact mismatch'):
                    self.module.main(argv)
            result_path = folder / 'out/result.json'
            saved = result_path.read_text()
            self.assertFalse(json.loads(saved)['passed'])
            with self.assertRaises(FileExistsError):
                self.module.main(argv)
            self.assertEqual(result_path.read_text(), saved)

    def test_receipt_io_error_cannot_replace_original_failure(self):
        first = RuntimeError('first body failure')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = folder / 'config.json'
            config.write_text('{}')
            original_dump = self.module.dump

            def broken_receipt(path, value):
                if path.name == 'result.json':
                    raise OSError('receipt failed')
                original_dump(path, value)

            with patch.object(self.module, 'run', side_effect=first), \
                    patch.object(self.module, 'dump', side_effect=broken_receipt):
                with self.assertRaises(RuntimeError) as raised:
                    self.module.main(['--config', str(config), '--output', str(folder/'out'),
                        '--arm', 'shadow', '--kind', 'cold', '--round', '0'])
            self.assertIs(raised.exception, first)
            self.assertIn('receipt failed', ' '.join(first.__notes__))


if __name__ == '__main__':
    unittest.main()
