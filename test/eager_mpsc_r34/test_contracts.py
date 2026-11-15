"""Offline adversarial gates; these fixtures never certify a device run."""
import copy
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).parents[2] / 'benchmarks/eager_mpsc_r34/contracts.py'


def module():
    if not SOURCE.exists():
        raise AssertionError('R34 measurement contracts are not implemented')
    spec = importlib.util.spec_from_file_location('r34_contracts_test', SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def snapshot(*, count=0):
    counters = {key: count for key in ('seen', 'eligible', 'queued', 'prepared', 'executed',
                                      'returned', 'sdk_prepare_calls')}
    counters.update(fallback=0, sdk_query_calls=0, sdk_cache_hits=count)
    return dict(schema=1, enabled=True, mode='eager_prepare', ingress='mpsc', workers=2,
                backend='real_sdk', real_sdk_qualified=True, simulated_prepare=False,
                qualified_ops=['aclnnEmbeddingBag'], first_error=None,
                diagnostic_trace=dict(enabled=False),
                operators={'aclnnEmbeddingBag': counters},
                queue=dict(admitted=count, retired=count, slots_in_use=0, retained_bytes=0,
                           closed=False, quarantined=False, workers_started=2),
                peaks=dict(slots_in_use=2, retained_bytes=4096),
                closure={key: True for key in ('admission_closed', 'fifo_drained',
                    'device_synchronized', 'all_retired', 'workers_stopped', 'passed')},
                order_evidence=dict(enabled=True, passed=True, admitted_count=count,
                    executed_count=count, admitted_sequence_hash=123, executed_sequence_hash=123))


class ContractsTests(unittest.TestCase):
    def test_warm_cache_hits_are_real_prepare_without_forcing_workspace_queries(self):
        c = module()
        gate = c.native_delta(snapshot(), snapshot(count=75), 'mpsc2', 3)
        self.assertEqual(gate['queued'], 75)
        self.assertEqual(gate['sdk_query_calls'], 0)
        self.assertEqual(gate['sdk_cache_hits'], 75)

    def test_zero_admission_or_missing_real_prepare_is_rejected(self):
        c = module()
        with self.assertRaisesRegex(ValueError, 'admission'):
            c.native_delta(snapshot(), snapshot(), 'mpsc2', 3)
        for field in ('sdk_prepare_calls', 'prepared', 'returned'):
            bad = snapshot(count=75)
            bad['operators']['aclnnEmbeddingBag'].pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.native_delta(snapshot(), bad, 'mpsc2', 3)

    def test_off_cannot_borrow_on_qualification_and_timed_snapshots_must_be_drained(self):
        c = module()
        off = snapshot()
        off.update(mode='off', enabled=False, qualified_ops=[], operators={})
        off['queue']['workers_started'] = 0
        self.assertTrue(c.native_delta(off, copy.deepcopy(off), 'off', 3)['passed'])
        bad = copy.deepcopy(off)
        bad['qualified_ops'] = ['aclnnEmbeddingBag']
        with self.assertRaises(ValueError):
            c.native_delta(off, bad, 'off', 3)
        bad = snapshot(count=75)
        bad['queue']['slots_in_use'] = 1
        with self.assertRaises(ValueError):
            c.native_delta(snapshot(), bad, 'mpsc2', 3)

    def test_simulation_wrong_ingress_immediate_and_trace_are_rejected(self):
        c = module()
        for key, value in [('backend', 'native_immediate'), ('simulated_prepare', True),
                           ('real_sdk_qualified', False), ('ingress', 'mutex'), ('workers', 4)]:
            bad = snapshot(count=75)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.native_delta(snapshot(), bad, 'mpsc2', 3)
        bad = snapshot(count=75)
        bad['diagnostic_trace']['enabled'] = True
        with self.assertRaises(ValueError):
            c.native_delta(snapshot(), bad, 'mpsc2', 3)

    def test_incomplete_fifo_or_retirement_never_closes_successfully(self):
        c = module()
        good = snapshot(count=75)
        good['enabled'] = False
        good['mode'] = 'off'
        good['queue']['closed'] = True
        self.assertTrue(c.validate_close(good, 'mpsc2')['passed'])
        for field in ('fifo_drained', 'device_synchronized', 'all_retired', 'workers_stopped'):
            bad = copy.deepcopy(good)
            bad['closure'][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.validate_close(bad, 'mpsc2')
        for field, value in [('enabled', False), ('passed', False), ('executed_sequence_hash', 456)]:
            bad = copy.deepcopy(good)
            bad['order_evidence'][field] = value
            with self.subTest(order=field), self.assertRaises(ValueError):
                c.validate_close(bad, 'mpsc2')

    def test_seven_arm_plan_requires_correctness_first_and_fixed_exposure(self):
        c = module()
        plan = c.campaign_plan(3)
        self.assertEqual(len(plan), 28)
        self.assertEqual([p['arm'] for p in plan[:7]], list(c.ARMS))
        self.assertTrue(all(p['kind'] == 'correctness' for p in plan[:7]))
        for round_no in range(3):
            rows = [p for p in plan if p['kind'] == 'perf' and p['round'] == round_no]
            self.assertEqual({p['arm'] for p in rows}, set(c.ARMS))
            self.assertTrue(all((p['warmup'], p['samples']) == (50, 3000) for p in rows))
        formal = c.campaign_plan(14)[7:]
        from collections import Counter
        positions = Counter((p['arm'], i % 7) for i, p in enumerate(formal))
        self.assertEqual(set(positions.values()), {2})
        pairs = Counter((a['arm'], b['arm']) for a, b in zip(formal, formal[1:])
                        if a['round'] == b['round'])
        self.assertEqual(len(pairs), 42)
        self.assertEqual(set(pairs.values()), {2})

    def test_summary_rejects_changed_artifact_or_missing_arm(self):
        c = module()
        rows = []
        for arm in c.ARMS:
            rows.append(dict(passed=True, kind='perf', arm=arm, round=0, samples=3000,
                warmup=50, durations_ns=[1_000_000] * 3000, window_wall_ns=3_100_000_000,
                window_process_cpu_ns=2_000_000_000, comparison_identity={'native': 'fixed'},
                native_timed_gate={'passed': True}, close_gate={'passed': True},
                numeric_final={'passed': True}, stage_validation={'passed': True}))
        self.assertEqual(c.summarize(rows, 1)['mpsc2']['paired_vs_off_pct'], [0.0])
        with self.assertRaises(ValueError):
            c.summarize(rows[:-1], 1)
        bad = copy.deepcopy(rows)
        bad[-1]['comparison_identity']['native'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity'):
            c.summarize(bad, 1)
        bad = copy.deepcopy(rows)
        bad[-1]['numeric_final']['passed'] = False
        with self.assertRaises(ValueError):
            c.summarize(bad, 1)


if __name__ == '__main__':
    unittest.main()
