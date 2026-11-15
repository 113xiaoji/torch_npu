"""Campaign isolation, bounded launch and offline audit checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('r35_fixtures', Path(__file__).with_name('test_contracts.py'))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class CampaignTests(unittest.TestCase):
    def test_grid_has_twelve_off_npu_cases_and_two_explicit_prepare_cases(self):
        c = F.load()
        npu = c.case_grid('npu')
        self.assertEqual(len(npu), 12)
        self.assertEqual({r['stream_mode'] for r in npu}, {'shared', 'private'})
        self.assertEqual({r['backend'] for r in npu}, {'off'})
        self.assertEqual(len(c.case_grid('npu', include_prepare=True)), 14)
        self.assertEqual(len(c.case_grid('cpu')), 6)
        with self.assertRaises(ValueError):
            c.case_grid('cpu', include_prepare=True)
        plan = c.campaign_plan(device='npu', rounds=2, include_prepare=True)
        self.assertEqual(len(plan), 14 * 3)
        self.assertTrue(all(r['kind'] == 'correctness' and r['total_requests'] == 32 and r['warmup_per_caller'] == 0 for r in plan[:14]))
        self.assertTrue(all(r['kind'] == 'perf' and r['total_requests'] == 3000 and r['warmup_per_caller'] == 24 for r in plan[14:]))

    def test_gil_on_uses_same_binary_explicit_xgil_and_structured_tokens(self):
        campaign = F.load('campaign')
        c = F.load()
        off, on = c.case_grid('cpu')[0], c.case_grid('cpu')[3]
        launch = dict(argv_prefix=['docker', 'exec', 'box', 'timeout', '600', '/path with spaces/python'],
                      worker_timeout_seconds=600)
        for row, expected in [(off, False), (on, True)]:
            row = dict(row, kind='perf', round=0, total_requests=3000, warmup_per_caller=24)
            argv = campaign.worker_argv(launch, '/worker path.py', '/config path.json', '/output path', row)
            self.assertEqual(('-X' in argv), expected)
            if expected:
                self.assertEqual(argv[6:9], ['-X', 'gil=1', '/worker path.py'])
            self.assertIn('/path with spaces/python', argv)
            self.assertEqual(argv[argv.index('--output') + 1], '/output path')
        for bad in (
            dict(argv_prefix='python', worker_timeout_seconds=600),
            dict(argv_prefix=['python'], worker_timeout_seconds=600),
            dict(argv_prefix=['timeout', '601', 'python'], worker_timeout_seconds=601),
            dict(argv_prefix=['timeout', '600', 'python', '-X', 'gil=0'], worker_timeout_seconds=600),
        ):
            with self.assertRaises(ValueError):
                campaign.worker_argv(bad, '/w.py', '/c.json', '/out', dict(off, kind='perf', round=0, total_requests=3000, warmup_per_caller=24))

    def test_offline_campaign_requires_serial_fresh_process_lifecycle(self):
        c, audit = F.load(), F.load('analyze')
        plan = c.campaign_plan(device='cpu', rounds=1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'plan.json').write_text(json.dumps(dict(schema=1, device='cpu', rounds=1, include_prepare=False, plan=plan)))
            life = []
            for i, task in enumerate(plan):
                label = audit.receipt_label(i, task)
                row = F.receipt(kind=task['kind'], callers=task['callers'], gil_mode=task['gil_mode'])
                row.update(round=task['round'])
                (folder / label).mkdir()
                (folder / label / 'result.json').write_text(json.dumps(row))
                life.append(dict(label=label, plan_index=i, launcher_pid=i+100, returncode=0,
                    monotonic_start_ns=i * 1000 + 1, monotonic_end_ns=i * 1000 + 900))
            lifecycle = folder / 'lifecycle.json'
            lifecycle.write_text(json.dumps(life))
            self.assertEqual(audit.audit_campaign(folder)['worker_count'], 12)
            bad = copy.deepcopy(life)
            bad[1]['monotonic_start_ns'] = 2
            lifecycle.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                audit.audit_campaign(folder)
            bad = copy.deepcopy(life)
            bad[1]['launcher_pid'] = bad[0]['launcher_pid']
            lifecycle.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                audit.audit_campaign(folder)


if __name__ == '__main__':
    unittest.main()
