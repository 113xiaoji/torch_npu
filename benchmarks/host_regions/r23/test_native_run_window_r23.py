"""Host-only clock and temporary-module-boundary tests; no SDK mock."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name('native_run_window_r23.py')
RUNNER = Path(os.environ['R23_FROZEN_RUNNER']) if os.environ.get('R23_FROZEN_RUNNER') else (
    Path(__file__).parents[1] / 'real-4worker-20261026-r22' / 'continuation-tools-20261027-14' / 'real_dlrm_runner_r22.py')


def load_module():
    if not SOURCE.is_file():
        raise AssertionError('native run-window wrapper is not implemented')
    spec = importlib.util.spec_from_file_location('_test_native_run_window_r23', SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Clock:
    CLOCK_MONOTONIC_RAW = 4
    def __init__(self):
        self.calls = []; self.value = 100
    def perf_counter_ns(self):
        self.calls.append('perf'); self.value += 10; return self.value
    def monotonic_ns(self):
        self.calls.append('monotonic'); self.value += 1; return self.value
    def clock_gettime_ns(self, kind):
        if kind != self.CLOCK_MONOTONIC_RAW:
            raise AssertionError('wrong clock domain')
        self.calls.append('raw'); self.value += 1; return self.value
    def time_ns(self):
        self.calls.append('epoch'); self.value += 1; return self.value
    def process_time_ns(self):
        return 1234


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_proxy_preserves_return_values_and_records_only_two_original_boundaries(self):
        clock = Clock(); report = dict(phase='setup', timing=dict(durations_ns=[]))
        proxy = self.module.TimeProxy(clock, report)
        self.assertEqual(proxy.perf_counter_ns(), 110)
        self.assertIsNone(proxy.start)
        self.assertEqual(clock.calls, ['perf'])
        report['phase'] = 'descriptive-timing'
        self.assertEqual(proxy.perf_counter_ns(), 120)
        self.assertEqual(proxy.start['perf_counter_ns'], 120)
        self.assertEqual(clock.calls, ['perf', 'perf', 'monotonic', 'raw', 'epoch', 'monotonic'])
        self.assertEqual(proxy.process_time_ns(), 1234)
        report['timing']['durations_ns'] = [1] * 1999
        self.assertEqual(proxy.perf_counter_ns(), 134)
        self.assertIsNone(proxy.end)
        report['timing']['durations_ns'].append(1)
        self.assertEqual(proxy.perf_counter_ns(), 144)
        self.assertEqual(proxy.end['perf_counter_ns'], 144)
        self.assertEqual(proxy.end['samples_observed'], 2000)
        self.assertEqual(proxy.start['read_order'], ['perf_counter_ns', 'monotonic_ns_before',
            'clock_gettime_ns_MONOTONIC_RAW', 'time_ns', 'monotonic_ns_after'])
        calls = list(clock.calls)
        report['phase'] = 'complete'
        self.assertEqual(proxy.perf_counter_ns(), 158)
        self.assertEqual(clock.calls, calls + ['perf'])
        self.assertIs(time.perf_counter_ns, self.saved_global_perf)

    def test_missing_raw_clock_is_explicit_and_cannot_complete_a_supported_window(self):
        clock = Clock()
        clock.clock_gettime_ns = None
        report = dict(phase='descriptive-timing', timing=dict(durations_ns=[]))
        proxy = self.module.TimeProxy(clock, report)
        proxy.perf_counter_ns()
        report['timing']['durations_ns'] = [1] * 2000
        proxy.perf_counter_ns()
        proof = proxy.proof()
        self.assertFalse(proof['window_complete'])
        self.assertEqual(proof['raw_clock_status'], 'unsupported')
        self.assertIsNone(proxy.start['clock_monotonic_raw_ns'])

    def test_application_identity_uses_proc_birth_and_boot_with_explicit_unsupported_host(self):
        self.assertTrue(callable(getattr(self.module, 'process_identity', None)),
            'application process birth/boot observer is not implemented')
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / 'self').mkdir()
            boot = proc / 'sys/kernel/random/boot_id'
            boot.parent.mkdir(parents=True)
            boot.write_text('3bb55319-5ec2-4dff-8cbc-23790efd6517\n')
            fields = ['S', '1', '4321'] + ['0'] * 16 + ['987654']
            (proc / 'self/stat').write_text('4321 (fixture ) command) ' + ' '.join(fields))
            with mock.patch.object(self.module.sys, 'platform', 'linux'), \
                    mock.patch.object(self.module.os, 'getpid', return_value=4321):
                identity = self.module.process_identity(proc)
            self.assertEqual(identity, dict(pid=4321, pgid=4321,
                host_boot_id='3bb55319-5ec2-4dff-8cbc-23790efd6517',
                starttime_ticks=987654, proc_identity_observed=True))
            with mock.patch.object(self.module.sys, 'platform', 'win32'), \
                    mock.patch.object(self.module.os, 'getpid', return_value=4321):
                unsupported = self.module.process_identity(proc)
            self.assertEqual(unsupported, dict(pid=4321, pgid=None, host_boot_id=None,
                starttime_ticks=None, proc_identity_observed=False))

    def test_original_main_and_run_called_once_restored_and_incomplete_not_upgraded(self):
        clock = Clock(); calls = []
        report = dict(phase='setup', passed=True, timing=dict(durations_ns=[]))
        runner = SimpleNamespace(time=clock)
        def original_run(args, row, cleanup):
            calls.append(('run', args, cleanup))
            self.assertIsNot(runner.time, clock)
            row['phase'] = 'descriptive-timing'
            runner.time.perf_counter_ns()
            row['timing']['durations_ns'].append(9)
            row['phase'] = 'complete'
        def original_main(argv):
            calls.append(('main', argv))
            runner.run(SimpleNamespace(kind='perf', warmup=20, iterations=2000), report, [])
            return 0
        runner.run = original_run; runner.main = original_main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'window.json'
            self.assertEqual(self.module.run_with_window(runner, ['--fixture-argument'], output, RUNNER), 0)
            proof = json.loads(output.read_text())
            self.assertFalse(proof['window_complete'])
            self.assertFalse(proof['performance_conclusion_allowed'])
            self.assertTrue(proof['diagnostic_only'])
            self.assertTrue(proof['temporary_module_bindings_restored'])
            self.assertEqual(proof['original_exit_code'], 0)
            self.assertEqual(proof['report_phase'], 'complete')
            self.assertEqual(proof['samples'], 1)
            self.assertIn('application_process', proof, 'sidecar has no application PID/birth identity')
            self.assertEqual(proof['application_process']['pid'], os.getpid())
            self.assertIs(type(proof['application_process']['proc_identity_observed']), bool)
            self.assertIs(runner.run, original_run)
            self.assertIs(runner.time, clock)
            self.assertEqual([row[0] for row in calls], ['main', 'run'])
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                self.module.run_with_window(runner, [], output, RUNNER)
            self.assertEqual(len(calls), 2)

    def test_original_exception_restores_bindings_and_keeps_failure_metadata(self):
        clock = Clock(); runner = SimpleNamespace(time=clock)
        def original_run(args, row, cleanup):
            row['phase'] = 'descriptive-timing'
            runner.time.perf_counter_ns()
            raise RuntimeError('original failure')
        def original_main(argv):
            return runner.run(SimpleNamespace(kind='perf', warmup=20, iterations=2000),
                dict(phase='setup', timing=dict(durations_ns=[])), [])
        runner.run = original_run; runner.main = original_main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'window.json'
            with self.assertRaisesRegex(RuntimeError, 'original failure'):
                self.module.run_with_window(runner, [], output, RUNNER)
            proof = json.loads(output.read_text())
            self.assertTrue(proof['temporary_module_bindings_restored'])
            self.assertFalse(proof['window_complete'])
            self.assertIn('original failure', proof['failure']['error'])
            self.assertIsNone(proof['original_exit_code'])
            self.assertIs(runner.run, original_run)
            self.assertIs(runner.time, clock)

    def test_exact_two_thousand_sample_common_wall_qualifies_only_the_original_boundaries(self):
        clock = Clock(); runner = SimpleNamespace(time=clock)
        report = dict(phase='setup', passed=True, timing=dict(durations_ns=[]))
        def original_run(args, row, cleanup):
            row['phase'] = 'descriptive-timing'
            started = runner.time.perf_counter_ns()
            for _ in range(2000):
                sample_start = runner.time.perf_counter_ns()
                row['timing']['durations_ns'].append(runner.time.perf_counter_ns() - sample_start)
            row['timing']['common_wall_ns'] = runner.time.perf_counter_ns() - started
            row['phase'] = 'complete'
        def original_main(argv):
            runner.run(SimpleNamespace(kind='perf', warmup=20, iterations=2000), report, [])
            return 0
        runner.run = original_run; runner.main = original_main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'window.json'
            self.assertEqual(self.module.run_with_window(runner, [], output, RUNNER), 0)
            proof = json.loads(output.read_text())
            self.assertTrue(proof['window_complete'])
            self.assertTrue(proof['common_wall_ns_matches'])
            self.assertEqual(proof['start']['phase_perf_call'], 1)
            self.assertEqual(proof['end']['phase_perf_call'], 4002)
            self.assertEqual(proof['end']['perf_counter_ns'] - proof['start']['perf_counter_ns'],
                report['timing']['common_wall_ns'])
            self.assertEqual(len(report['timing']['durations_ns']), 2000)
            self.assertEqual(set(report['timing']['durations_ns']), {10})
            self.assertEqual(clock.calls.count('perf'), 4002)
            self.assertEqual(clock.calls.count('raw'), 2)
            self.assertEqual(clock.calls.count('epoch'), 2)
            self.assertEqual(clock.calls.count('monotonic'), 4)

    saved_global_perf = time.perf_counter_ns


if __name__ == '__main__':
    unittest.main()
