import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmarks/forward_concurrency_r35/real_service.py'


class ServicePolicyTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SOURCE.is_file(), 'real full-forward service adapter is not implemented')
        spec = importlib.util.spec_from_file_location('r35_service_test', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def helper(self, module, name):
        helper = getattr(module, name, None)
        self.assertTrue(callable(helper), name + ' safety helper is not implemented')
        return helper

    def test_gil_off_is_natural_and_gil_on_uses_same_ft_build(self):
        m = self.module()
        self.assertFalse(m.validate_gil('off', False, 1, {}, {}))
        self.assertTrue(m.validate_gil('on', True, 1, {'gil': '1'}, {}))

    def test_wrong_or_forced_gil_policy_rejects_before_framework_setup(self):
        m = self.module()
        for mode, enabled, build, options, env in (
            ('off', True, 1, {}, {}), ('on', False, 1, {'gil': '1'}, {}),
            ('off', False, 0, {}, {}), ('off', False, 1, {'gil': '0'}, {}),
            ('on', True, 1, {}, {}), ('off', False, 1, {}, {'PYTHON_GIL': '0'})):
            with self.subTest(mode=mode, enabled=enabled, options=options), self.assertRaises(ValueError):
                m.validate_gil(mode, enabled, build, options, env)

    def test_valid_native_close_is_acknowledged_without_retry(self):
        m = self.module()
        close = self.helper(m, 'close_native_domain')
        snapshot = {'native': 'closed'}
        calls = []
        validated = []

        def drain_close():
            calls.append('drain')
            return snapshot

        def validate_close(value, backend):
            validated.append((value, backend))
            return {'passed': True}

        report = {}
        prepare = SimpleNamespace(drain_close=drain_close)
        contracts = SimpleNamespace(validate_close=validate_close)
        self.assertIs(close(prepare, contracts, 'mutex4', report), snapshot)
        self.assertEqual(calls, ['drain'])
        self.assertEqual(validated, [(snapshot, 'mutex4')])
        self.assertIs(report['closed'], snapshot)
        self.assertTrue(report['close_gate']['passed'])
        self.assertTrue(report['native_close_attempted'])
        self.assertTrue(report['native_close_acknowledged'])
        self.assertFalse(report.get('requires_process_termination', False))
        with self.assertRaises(RuntimeError):
            close(prepare, contracts, 'mutex4', report)
        self.assertEqual(calls, ['drain'], 'native close must never be retried')

    def test_native_close_exception_requires_termination_in_both_paths(self):
        m = self.module()
        close = self.helper(m, 'close_native_domain')
        for failure_path in (False, True):
            with self.subTest(failure_path=failure_path):
                error = RuntimeError('quarantined native workers remain')
                calls = []

                def drain_close():
                    calls.append('drain')
                    raise error

                def validate_close(value, backend):
                    self.fail('a failed drain must not be validated')

                report = {}
                prepare = SimpleNamespace(drain_close=drain_close)
                contracts = SimpleNamespace(validate_close=validate_close)
                with self.assertRaises(RuntimeError) as caught:
                    close(prepare, contracts, 'mutex4', report, failure=failure_path)
                self.assertIs(caught.exception, error)
                self.assertTrue(report['requires_process_termination'])
                self.assertTrue(report['native_close_attempted'])
                self.assertFalse(report.get('native_close_acknowledged', False))
                with self.assertRaises(RuntimeError):
                    close(prepare, contracts, 'mutex4', report, failure=failure_path)
                self.assertEqual(calls, ['drain'])

    def test_invalid_native_close_ack_requires_termination_in_both_paths(self):
        m = self.module()
        close = self.helper(m, 'close_native_domain')
        for failure_path in (False, True):
            for invalid in ('raise', 'false', 'missing'):
                with self.subTest(failure_path=failure_path, invalid=invalid):
                    snapshot = {'native': 'not acknowledged'}
                    calls = []

                    def drain_close():
                        calls.append('drain')
                        return snapshot

                    def validate_close(value, backend):
                        self.assertIs(value, snapshot)
                        if invalid == 'raise':
                            raise ValueError('workers_stopped ACK missing')
                        return {'passed': False} if invalid == 'false' else None

                    report = {}
                    with self.assertRaises(ValueError):
                        close(SimpleNamespace(drain_close=drain_close),
                              SimpleNamespace(validate_close=validate_close),
                              'mpsc4', report, failure=failure_path)
                    self.assertTrue(report['requires_process_termination'])
                    self.assertFalse(report.get('native_close_acknowledged', False))
                    self.assertIs(report['failure_close' if failure_path else 'closed'], snapshot)
                    self.assertEqual(calls, ['drain'])

    def output_path(self, writer):
        class OutputPath:
            def __truediv__(inner, name):
                self.assertEqual(name, 'result.json')
                return inner

            def write_text(inner, text):
                return writer(text)

        return OutputPath()

    def test_report_persistence_precedes_forced_exit_only_when_required(self):
        m = self.module()
        persist = self.helper(m, 'persist_report')
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe):
                operations = []
                report = {'passed': False, 'requires_process_termination': unsafe}

                def write(text):
                    self.assertEqual(json.loads(text), report)
                    operations.append('write')

                persist(self.output_path(write), report,
                        terminate=lambda code: operations.append(('exit', code)))
                self.assertEqual(operations, ['write', ('exit', 2)] if unsafe else ['write'])

    def test_report_write_failure_still_forces_exit_with_active_callers(self):
        m = self.module()
        persist = self.helper(m, 'persist_report')
        error = OSError('disk full')
        exits = []

        def write(text):
            raise error

        with self.assertRaises(OSError) as caught:
            persist(self.output_path(write), {'requires_process_termination': True},
                    terminate=exits.append)
        self.assertIs(caught.exception, error)
        self.assertEqual(exits, [2])

    def test_report_serialization_failure_still_forces_exit_with_active_callers(self):
        m = self.module()
        persist = self.helper(m, 'persist_report')
        exits = []

        def write(text):
            self.fail('invalid JSON must not reach the writer')

        with self.assertRaises(ValueError):
            persist(self.output_path(write),
                    {'requires_process_termination': True, 'invalid': float('nan')},
                    terminate=exits.append)
        self.assertEqual(exits, [2])

    def test_ordinary_report_write_failure_does_not_force_exit(self):
        m = self.module()
        persist = self.helper(m, 'persist_report')
        error = OSError('disk full')
        exits = []

        def write(text):
            raise error

        with self.assertRaises(OSError) as caught:
            persist(self.output_path(write), {'passed': False}, terminate=exits.append)
        self.assertIs(caught.exception, error)
        self.assertEqual(exits, [])

    def test_source_identity_uses_canonical_posix_relative_paths(self):
        m = self.module()
        paths = [m.HERE / name for name in ('real_service.py', 'engine.py', 'contracts.py', 'requests.py')]
        pins = {str(path.resolve()): m.sha(path) for path in paths}
        values = sorted((Path(path).resolve().relative_to(ROOT).as_posix(), digest)
                        for path, digest in pins.items())
        expected = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
        config = dict(source_pins=pins, source_root=str(ROOT), source_sha256=expected)
        try:
            actual = m.source_identity(config)
        except ValueError as error:
            self.fail('canonical POSIX source identity rejected: ' + str(error))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
