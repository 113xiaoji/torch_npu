import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[2]


def load(name):
    path = ROOT / 'benchmarks/eager_mpsc_r34' / (name + '.py')
    if not path.exists():
        raise AssertionError('R34 ' + name + ' is not implemented')
    spec = importlib.util.spec_from_file_location('r34_' + name + '_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CampaignTests(unittest.TestCase):
    def test_launcher_selects_ingress_without_shell_reparsing(self):
        campaign = load('campaign')
        row = dict(kind='perf', arm='mpsc4', round=2, samples=3000, warmup=50)
        argv = campaign.worker_argv(dict(argv_prefix=['docker', 'exec', 'container', 'env',
            'TORCH_NPU_EAGER_PREPARE_INGRESS={ingress}', '/path with spaces/python']),
            '/worker.py', '/config.json', '/new output', row)
        self.assertIn('TORCH_NPU_EAGER_PREPARE_INGRESS=mpsc', argv)
        self.assertIn('/path with spaces/python', argv)
        self.assertEqual(argv[argv.index('--output') + 1], '/new output')

    def test_offline_audit_recomputes_native_delta_and_summary(self):
        audit = load('analyze')
        spec = importlib.util.spec_from_file_location('r34_fixture', Path(__file__).with_name('test_contracts.py'))
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        before, after = fixtures.snapshot(), fixtures.snapshot(count=75000)
        closed = copy.deepcopy(after)
        closed.update(enabled=False, mode='off')
        closed['queue']['closed'] = True
        row = dict(schema=1, passed=True, arm='mpsc2', kind='perf', round=0, samples=3000,
            warmup=50, before=before, after=after, closed=closed,
            durations_ns=[1_000_000] * 3000, window_wall_ns=3_100_000_000,
            timing=dict(count=3000, mean_ns=1_000_000, median_ns=1_000_000,
                        p95_ns=1_000_000, p99_ns=1_000_000),
            numeric_final={'passed': True}, stage_validation={'passed': True},
            native_timed_gate={'passed': True}, close_gate={'passed': True})
        result = audit.audit_result(row)
        self.assertEqual(result['native_timed_gate']['queued'], 75000)
        bad = copy.deepcopy(row)
        bad['after']['operators']['aclnnEmbeddingBag']['sdk_prepare_calls'] = 0
        with self.assertRaises(ValueError):
            audit.audit_result(bad)
        bad = copy.deepcopy(row)
        bad['timing']['median_ns'] = 999
        with self.assertRaises(ValueError):
            audit.audit_result(bad)


if __name__ == '__main__':
    unittest.main()
