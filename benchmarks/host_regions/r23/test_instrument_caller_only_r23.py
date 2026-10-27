"""Caller-only host protocol fixtures; no SDK/device execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('instrument_caller_only_r23.py')


def module():
    if not PATH.is_file(): raise AssertionError('caller-only wrapper missing')
    spec = importlib.util.spec_from_file_location('caller_only_test', PATH)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class Policy:
    accepted = True
    def _sequence(self, value): return tuple(value)
    def check(self, arguments):
        self._sequence([1]); self._sequence([2]); return True


class Provider:
    def __init__(self): self.policies = [Policy()]
    def guard(self, policy, arguments): return policy.check(arguments)


class Model:
    def __init__(self, device, provider):
        self.device = device; self.provider = provider; self.bot_l = object(); self.top_l = object()
    def parameters(self): return iter([SimpleNamespace(device=SimpleNamespace(type=self.device))])
    def apply_mlp(self, value, layers): return value
    def apply_emb(self): return self.provider.guard(self.provider.policies[0], {})
    def interact_features(self): return 1
    def forward(self):
        self.apply_mlp(1, self.bot_l); self.apply_emb(); self.interact_features(); self.apply_mlp(1, self.top_l)
        return 9


class CallerOnlyTests(unittest.TestCase):
    def test_exact_base_source_is_loaded_and_bad_source_is_rejected(self):
        m = module(); base = m.load_base(PATH.with_name('instrument_frozen_runner_r23.py'))
        self.assertEqual(base.RUNNER_SHA, m.RUNNER_SHA)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.py'; path.write_text('raise AssertionError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'base wrapper source'):
                m.load_base(path)

    def test_worker_case_records_only_caller_tags_and_late_sync_leaves_original_alias(self):
        m = module(); base = m.load_base(PATH.with_name('instrument_frozen_runner_r23.py')); c = base.Collector()
        provider = Provider(); npu = Model('npu', provider); cpu = Model('cpu', provider)
        original_sync = lambda device=0: None
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=original_sync))
        saved_work = lambda *a: None; cold_work = lambda *a: None
        runtime = SimpleNamespace(_native_work=cold_work)
        probe = SimpleNamespace(module=runtime, saved=saved_work)
        runner = SimpleNamespace(eager_proof=lambda *a: {}, install_real_worker_model=lambda *a, **k: (provider, probe, None))
        def run(args, report, cleanup):
            runner.eager_proof(cpu, torch); self.assertNotIn('forward', vars(cpu))
            runner.install_real_worker_model(npu, torch)
            self.assertIs(runtime._native_work, cold_work)
            # A real cold barrier stores this callback during functional execution.
            saved_barrier_sync = torch.npu.synchronize
            self.assertIs(saved_barrier_sync, original_sync)
            report['phase'] = 'warmup'; runtime._native_work = probe.saved
            runner.eager_proof(npu, torch)
            self.assertIs(runtime._native_work, saved_work)
            self.assertIsNot(torch.npu.synchronize, original_sync)
            report['phase'] = 'descriptive-timing'
            for _ in range(2000):
                self.assertEqual(npu.forward(), 9); torch.npu.synchronize()
            report['phase'] = 'complete'
            self.assertIs(saved_barrier_sync, original_sync)
        runner.run = run
        with patch.object(m, 'verify_python_sources', return_value={}):
            m.install_caller_hooks(base, runner, c)
            runner.run(SimpleNamespace(arm='worker4'), {'phase': 'functional'}, [])
        complete, counts, snapshots = m.coverage(c, 0, 'worker4', 'caller_only')
        self.assertTrue(complete); self.assertEqual(counts['policy.container_snapshot'], 4000)
        self.assertEqual(len(snapshots), 2000); self.assertEqual(snapshots[0]['count'], 2)
        self.assertEqual(len({r['native_tid'] for r in c.events()}), 1)
        self.assertFalse(any(r['name'].startswith(('worker.', 'region.', 'session.', 'executor.', 'runtime.')) for r in c.events()))
        self.assertIs(torch.npu.synchronize, original_sync); self.assertNotIn('forward', vars(npu))
        self.assertTrue(all(r['restored'] for r in c.restoration))
        c.report['phase'] = 'descriptive-timing'; c.invoke('worker.region', lambda: None)
        self.assertFalse(m.coverage(c, 0, 'worker4', 'caller_only')[0])

    def test_serial_cpu_is_untouched_and_original_exception_is_preserved(self):
        m = module(); base = m.load_base(PATH.with_name('instrument_frozen_runner_r23.py')); c = base.Collector()
        p = Provider(); cpu, npu = Model('cpu', p), Model('npu', p)
        sync = lambda: None; torch = SimpleNamespace(npu=SimpleNamespace(synchronize=sync)); error = KeyboardInterrupt('unique')
        runner = SimpleNamespace(eager_proof=lambda *a: {}, install_real_worker_model=lambda *a: None)
        def run(args, report, cleanup):
            runner.eager_proof(cpu, torch); self.assertNotIn('forward', vars(cpu))
            runner.eager_proof(npu, torch); self.assertIn('forward', vars(npu)); raise error
        runner.run = run; m.install_caller_hooks(base, runner, c)
        with self.assertRaises(KeyboardInterrupt) as caught:
            runner.run(SimpleNamespace(arm='serial'), {'phase': 'functional'}, [])
        self.assertIs(caught.exception, error); self.assertNotIn('forward', vars(npu))
        self.assertIs(runner.run, run); self.assertIs(torch.npu.synchronize, sync)

    def test_off_coverage_requires_zero_wrappers_and_events(self):
        m = module(); base = m.load_base(PATH.with_name('instrument_frozen_runner_r23.py')); c = base.Collector()
        self.assertTrue(m.coverage(c, 0, 'serial', 'off')[0])
        c.report = {'phase': 'descriptive-timing'}; c.invoke('forward', lambda: None, new_request=True)
        self.assertFalse(m.coverage(c, 0, 'serial', 'off')[0])

    def test_unobserved_workers_are_null_instead_of_zero_duration(self):
        m = module(); base = m.load_base(PATH.with_name('instrument_frozen_runner_r23.py'))
        c = base.Collector({'phase': 'descriptive-timing'}); c.invoke('forward', lambda: None, new_request=True)
        summary = m.caller_summary(base, c.events())
        self.assertIsNone(summary['requests'][0]['worker_cpu_ns'])
        self.assertIsNone(summary['requests'][0]['worker_span_ns'])
        self.assertFalse(summary['requests'][0]['worker_metrics_observed'])


if __name__ == '__main__': unittest.main()
