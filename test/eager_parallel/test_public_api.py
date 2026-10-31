"""Python bridge unit tests. A fake extension checks forwarding only, not NPU qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]


class PublicApiTests(unittest.TestCase):
    def load_bridge(self, extension):
        path = ROOT / 'torch_npu/_eager_parallel.py'
        self.assertTrue(path.is_file(), 'R26 public native bridge is missing')
        spec = importlib.util.spec_from_file_location('r26_public_api', path)
        module = importlib.util.module_from_spec(spec)
        package = types.SimpleNamespace(_C=extension)
        with patch.dict(sys.modules, {'torch_npu': package}):
            spec.loader.exec_module(module)
        return module

    def test_configuration_forwards_to_native_and_defaults_off(self):
        calls = []
        extension = types.SimpleNamespace(_npu_eager_parallel_configure=lambda *args:
            calls.append(args) or json.dumps({'schema': 1, 'enabled': args[0], 'qualified': False}))
        result = self.load_bridge(extension).configure()
        self.assertFalse(result['enabled'])
        self.assertFalse(result['qualified'])
        self.assertEqual(calls, [(False, 'shadow', 2, 64, 536870912)])

    def test_missing_actual_binding_is_reported_not_emulated(self):
        bridge = self.load_bridge(types.SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, 'native eager-parallel binding'):
            bridge.configure(enabled=True)

    def test_e_and_ei_reach_native_without_emulating_execution(self):
        calls = []
        extension = types.SimpleNamespace(_npu_eager_parallel_configure=lambda *args:
            calls.append(args) or json.dumps({'schema': 1, 'mode': args[1]}))
        bridge = self.load_bridge(extension)
        with torch.inference_mode():
            for mode in ('e', 'ei'):
                self.assertEqual(bridge.configure(enabled=True, mode=mode)['mode'], mode)
        self.assertEqual([row[:3] for row in calls], [(True, 'e', 2), (True, 'ei', 2)])

    def test_execution_mode_requires_inference_before_native_configuration(self):
        calls = []
        extension = types.SimpleNamespace(_npu_eager_parallel_configure=lambda *args:
            calls.append(args) or '{"schema":1}')
        bridge = self.load_bridge(extension)
        for mode in ('e', 'ei'):
            with self.assertRaisesRegex(RuntimeError, 'inference_mode'):
                bridge.configure(enabled=True, mode=mode)
        self.assertEqual(calls, [])

    def test_unsupported_mode_and_invalid_limits_never_enter_native(self):
        bridge = self.load_bridge(types.SimpleNamespace())
        for kwargs in ({'mode': 'parallel'}, {'streams': 3}, {'capacity': 0}, {'byte_limit': 0}):
            with self.assertRaises(ValueError):
                bridge.configure(**kwargs)
        with self.assertRaises(TypeError):
            bridge.configure(enabled=1)

    def test_snapshot_and_close_use_native_json_schema(self):
        extension = types.SimpleNamespace(
            _npu_eager_parallel_snapshot=lambda: '{"schema":1,"state":"open"}',
            _npu_eager_parallel_drain_close=lambda: '{"schema":1,"state":"closed"}')
        bridge = self.load_bridge(extension)
        self.assertEqual(bridge.snapshot()['state'], 'open')
        self.assertEqual(bridge.drain_close()['state'], 'closed')
        extension._npu_eager_parallel_snapshot = lambda: '{"schema":true}'
        with self.assertRaisesRegex(RuntimeError, 'schema'):
            bridge.snapshot()

    def test_scope_requires_caller_inference_mode_and_closes_on_exit(self):
        calls = []
        extension = types.SimpleNamespace(
            _npu_eager_parallel_configure=lambda *args: calls.append('open') or '{"schema":1}',
            _npu_eager_parallel_drain_close=lambda: calls.append('close') or '{"schema":1}')
        bridge = self.load_bridge(extension)
        self.assertTrue(hasattr(bridge, 'scope'), 'scoped native lifecycle bridge is missing')
        with self.assertRaisesRegex(RuntimeError, 'inference_mode'):
            with bridge.scope():
                self.fail('scope admitted a training caller')
        self.assertEqual(calls, [])
        with torch.inference_mode(), bridge.scope():
            self.assertEqual(calls, ['open'])
        self.assertEqual(calls, ['open', 'close'])

    def test_scope_preserves_body_error_if_close_also_fails(self):
        error = ValueError('first body failure')
        def close():
            raise RuntimeError('later close failure')
        bridge = self.load_bridge(types.SimpleNamespace(
            _npu_eager_parallel_configure=lambda *args: '{"schema":1}',
            _npu_eager_parallel_drain_close=close))
        self.assertTrue(hasattr(bridge, 'scope'), 'scoped native lifecycle bridge is missing')
        with self.assertRaises(ValueError) as raised:
            with torch.inference_mode(), bridge.scope():
                raise error
        self.assertIs(raised.exception, error)

    def assert_scope_preserves_error(self, error, close_error):
        def close():
            raise close_error
        bridge = self.load_bridge(types.SimpleNamespace(
            _npu_eager_parallel_configure=lambda *args: '{"schema":1}',
            _npu_eager_parallel_drain_close=close))
        try:
            with torch.inference_mode(), bridge.scope():
                raise error
        except BaseException as actual:
            self.assertIs(actual, error, 'diagnostic note replaced the original body exception')
        else:
            self.fail('original body exception was swallowed')

    def test_scope_preserves_body_error_when_notes_is_not_a_list(self):
        error = ValueError('first body failure')
        error.__notes__ = {'invalid': 'notes'}
        self.assert_scope_preserves_error(error, RuntimeError('later close failure'))

    def test_scope_preserves_body_error_when_custom_add_note_raises(self):
        class NoteFailure(ValueError):
            def add_note(self, note):
                raise KeyboardInterrupt('diagnostic add_note failure')
        self.assert_scope_preserves_error(NoteFailure('first body failure'), RuntimeError('later close failure'))

    def test_scope_preserves_body_error_when_close_error_repr_raises(self):
        class ReprFailure(RuntimeError):
            def __repr__(self):
                raise SystemExit('diagnostic repr failure')
        self.assert_scope_preserves_error(ValueError('first body failure'), ReprFailure('later close failure'))


if __name__ == '__main__':
    unittest.main()
