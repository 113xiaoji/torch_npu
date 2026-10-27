"""Host-only timing semantics. No Torch, SDK or device imports."""
import importlib.util
import os
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).with_name('instrument_frozen_runner_r23.py')


def module():
    if not PATH.is_file():
        raise AssertionError('standalone instrumentation harness missing')
    spec = importlib.util.spec_from_file_location('r23_instrument_test', PATH)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class TimingTests(unittest.TestCase):
    def collector(self):
        return module().Collector({'phase': 'descriptive-timing'})

    def test_inactive_phase_records_nothing_and_calls_once(self):
        c = module().Collector({'phase': 'warmup'}); calls = []; result = object()
        self.assertIs(c.invoke('forward', lambda: calls.append(1) or result, new_request=True), result)
        self.assertEqual(c.events(), []); self.assertEqual(calls, [1])

    def test_nested_spans_have_request_parent_tid_and_cpu_clock(self):
        c = self.collector()
        c.invoke('forward', lambda: c.invoke('guard', lambda: c.invoke('policy', lambda: 3)), new_request=True)
        rows = {row['name']: row for row in c.events()}
        self.assertEqual({row['request'] for row in rows.values()}, {1})
        self.assertEqual(rows['policy']['parent'], rows['guard']['id'])
        self.assertEqual(rows['guard']['parent'], rows['forward']['id'])
        for row in rows.values():
            self.assertGreater(row['native_tid'], 0)
            self.assertGreaterEqual(row['end_ns'], row['start_ns'])
            self.assertGreaterEqual(row['end_thread_time_ns'], row['start_thread_time_ns'])

    def test_exception_identity_and_return_value_are_preserved(self):
        c = self.collector(); error = KeyboardInterrupt('unique'); calls = []
        def fail(): calls.append(1); raise error
        with self.assertRaises(KeyboardInterrupt) as caught:
            c.invoke('forward', fail, new_request=True)
        self.assertIs(caught.exception, error); self.assertEqual(calls, [1])
        self.assertEqual(c.events()[0]['error_type'], 'KeyboardInterrupt')

    def test_request_indexes_and_tail_sync_share_identity(self):
        c = self.collector()
        for _ in range(2):
            c.invoke('forward', lambda: None, new_request=True)
            c.invoke('tail_sync', lambda: None)
        self.assertEqual([(r['name'], r['request']) for r in c.events()],
                         [('forward', 1), ('tail_sync', 1), ('forward', 2), ('tail_sync', 2)])

    def test_union_does_not_sum_overlapping_worker_spans(self):
        m = module()
        self.assertEqual(m.interval_union([(0, 10), (5, 20), (25, 30)]), 25)
        self.assertEqual(m.interval_union([]), 0)

    def test_cli_separator_is_removed_without_changing_runner_arguments(self):
        m = module(); original = ['--abi', 'std', '--output', '/new/result.json']
        self.assertEqual(m.runner_arguments(['--'] + original), original)
        self.assertEqual(m.runner_arguments(original), original)

    def test_instance_and_class_methods_restore_original_ownership(self):
        c = self.collector()
        class Model:
            def forward(self): return 7
        obj = Model(); original = Model.forward
        c.wrap(obj, 'forward', 'forward', new_request=True)
        self.assertEqual(obj.forward(), 7); self.assertIn('forward', vars(obj))
        c.wrap(Model, 'forward', 'class_forward')
        rows = c.restore()
        self.assertNotIn('forward', vars(obj)); self.assertIs(Model.forward, original)
        self.assertTrue(all(row['restored'] for row in rows))

    def test_region_body_uses_original_callable_once_per_element(self):
        c = self.collector(); c.invoke('forward', lambda: None, new_request=True)
        calls = []
        def work(runtime, session, region, body, items):
            return [body(i, items[i]) for i in range(region.start, region.stop)]
        wrapped = c.worker_wrapper(work)
        region = SimpleNamespace(index=2, start=1, stop=3)
        self.assertEqual(wrapped(None, None, region, lambda i, v: calls.append(i) or v, [4, 5, 6]), [5, 6])
        self.assertEqual(calls, [1, 2])
        self.assertEqual([row['item'] for row in c.events() if row['name'] == 'region.body'], [1, 2])

    def test_actual_cold_probe_restores_before_worker_timer_and_does_not_revive_it(self):
        m = module(); c = m.Collector({'phase': 'warmup'})
        default = Path(__file__).resolve().parents[2] / '.worktrees/torch-npu-r22-publish-20261027/benchmarks/host_regions/real_dlrm_runner_r22.py'
        frozen_path = Path(os.environ.get('R23_FROZEN_RUNNER', str(default))).resolve(strict=True)
        self.assertEqual(m.sha(frozen_path), m.RUNNER_SHA)
        spec = importlib.util.spec_from_file_location('r23_actual_cold_probe', frozen_path)
        frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
        original = lambda rt, session, region, body, items: body(region.start, items[region.start])
        runtime = SimpleNamespace(_native_work=original)
        probe = frozen.ColdWorkerProbe(runtime); probe.install()
        c.defer_worker(runtime, probe.saved)
        self.assertFalse(c.attach_worker_if_ready())
        probe.restore()
        self.assertIs(runtime._native_work, probe.saved)  # Original runner's identity gate.
        self.assertTrue(c.attach_worker_if_ready())
        c.report['phase'] = 'descriptive-timing'; c.invoke('forward', lambda: None, new_request=True)
        region = SimpleNamespace(index=0, start=0, stop=1)
        self.assertEqual(runtime._native_work(None, None, region, lambda i, value: value, [9]), 9)
        self.assertEqual(len([e for e in c.events() if e['name'] == 'worker.region']), 1)
        self.assertTrue(all(row['restored'] for row in c.restore()))
        probe.restore()  # Original main cleanup must not reinstall a timing wrapper.
        self.assertIs(runtime._native_work, original)

    def test_serial_cpu_is_untouched_and_exception_restores_before_outer_cleanup(self):
        m = module(); c = m.Collector(); error = RuntimeError('original unique failure')
        class Model:
            def __init__(self, device): self.device = device; self.bot_l = object(); self.top_l = object()
            def parameters(self): return iter([SimpleNamespace(device=SimpleNamespace(type=self.device))])
            def forward(self): return 1
            def apply_emb(self): return 2
            def apply_mlp(self, value, layers): return value
            def interact_features(self): return 3
        cpu, npu = Model('cpu'), Model('npu'); sync = lambda: None
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=sync))
        def run(args, report, cleanup):
            runner.eager_proof(cpu, torch); self.assertNotIn('forward', vars(cpu))
            runner.eager_proof(npu, torch); self.assertIn('forward', vars(npu))
            raise error
        runner = SimpleNamespace(run=run, eager_proof=lambda *a: {}, install_real_worker_model=lambda *a: None)
        m.install_hooks(runner, c)
        with self.assertRaises(RuntimeError) as caught:
            runner.run(SimpleNamespace(arm='serial'), {'phase': 'functional'}, [])
        self.assertIs(caught.exception, error)
        self.assertIs(runner.run, run); self.assertNotIn('forward', vars(npu))
        self.assertIs(torch.npu.synchronize, sync)
        self.assertTrue(all(row['restored'] for row in c.restoration))

    def test_actual_cold_barrier_keeps_original_sync_after_timers_are_restored(self):
        m = module(); c = m.Collector({'phase': 'warmup'})
        self.assertIn('R23_PROVIDER_SOURCE', os.environ, 'explicit pinned provider source path required')
        source = Path(os.environ['R23_PROVIDER_SOURCE']).resolve(strict=True)
        self.assertEqual(m.sha(source), m.PYTHON_PINS['torch_npu_provider'])
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_NativeWorkerRetirement')
        namespace = {}; exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        original_sync = lambda device=0: None
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=original_sync, default_stream=lambda device: None))
        model = SimpleNamespace(forward=lambda: 1, apply_emb=lambda: 2, apply_mlp=lambda *a: 3,
                                interact_features=lambda: 4, bot_l=object(), top_l=object())
        c.attach_model(model, torch)
        barrier = namespace['_NativeWorkerRetirement'](torch.npu, None, (0, 0, 0))
        self.assertIs(barrier._synchronize, original_sync)
        worker = lambda *a: None; runtime = SimpleNamespace(_native_work=worker)
        c.defer_worker(runtime, worker); c.attach_after_warmup(torch)
        self.assertIsNot(torch.npu.synchronize, original_sync)
        c.restore()
        self.assertIs(torch.npu.synchronize, original_sync)
        self.assertIs(barrier._synchronize, original_sync)


if __name__ == '__main__':
    unittest.main()
