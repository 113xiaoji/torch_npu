"""Qualification must reject a placebo or mismatched native treatment."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / 'benchmarks/linear_scaling_r41/native_policy.py'
spec = importlib.util.spec_from_file_location('r41_policy_test', PATH)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(native_sha256='a' * 64,
                           r41_treatment=dict(event_wait='condition', workspace_cache=1))
        self.env = dict(TORCH_NPU_EVENT_HOST_WAIT='condition', TORCH_NPU_WORKSPACE_CACHE='1')

    def test_explicit_treatment_and_identity(self):
        self.assertEqual(P.expected_treatment(self.config, self.env), self.config['r41_treatment'])
        for key in self.env:
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.expected_treatment(self.config, {k: v for k, v in self.env.items() if k != key})
        with self.assertRaises(ValueError):
            P.expected_treatment(dict(self.config, native_sha256='a'*63), self.env)

    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            P.expected_treatment(self.config, dict(self.env, TORCH_NPU_EVENT_HOST_WAIT='spin'))
        with self.assertRaises(ValueError):
            P.expected_treatment(self.config, dict(self.env, TORCH_NPU_WORKSPACE_CACHE='true'))

    def test_every_caller_must_reach_cache(self):
        row = dict(enabled=1, hits=50, allocations=1, growths=0, evictions=0,
                   bypasses=0, retained_bytes=4096, entries=1)
        self.assertTrue(P.audit_callers([row, row], 2, 1)['passed'])
        with self.assertRaises(ValueError):
            P.audit_callers([row, dict(row, hits=0)], 2, 1)
        with self.assertRaises(ValueError):
            P.audit_callers([row, None], 2, 1)

    def test_off_cannot_contain_treatment_activity(self):
        row = dict(enabled=0, hits=0, allocations=0, growths=0, evictions=0,
                   bypasses=0, retained_bytes=0, entries=0)
        self.assertTrue(P.audit_callers([row], 1, 0)['passed'])
        with self.assertRaises(ValueError):
            P.audit_callers([dict(row, hits=1)], 1, 0)

    def test_effective_mode_and_bounds(self):
        # Exercise the real receipt parser independently of a native library.
        native = object.__new__(P.NativeTreatment)
        native.expected = self.config['r41_treatment']
        native.mode = lambda: b'spin'
        native.stats = lambda values, count: None
        with self.assertRaises(ValueError):
            native.read()
        native.mode = lambda: b'condition'
        def stats(values, count):
            values[0] = 1
            values[6] = 64*1024*1024 + 1
        native.stats = stats
        with self.assertRaises(ValueError):
            native.read()


if __name__ == '__main__':
    unittest.main()
