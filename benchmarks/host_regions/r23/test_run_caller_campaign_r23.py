"""Host-only planning/validation fixtures; never launches a device process."""
import copy
import contextlib
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('run_caller_campaign_r23.py')


def module():
    if not PATH.is_file(): raise AssertionError('caller campaign driver missing')
    spec = importlib.util.spec_from_file_location('caller_campaign_test', PATH)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def cohort(m):
    rows = []
    for round_index in range(1, 4):
        for abi in ('std', 'ft'):
            for label in ('original-serial', 'candidate-serial', 'worker4'):
                for mode in ('off', 'stages'):
                    output = f'/old/{round_index}-{abi}-{label}-{mode}.json'
                    forwarded = ['--abi', abi, '--arm', 'worker4' if label == 'worker4' else 'serial',
                        '--kind', 'perf', '--warmup', '20', '--iterations', '2000', '--output', output,
                        '--topology-proof', '/old/TOPOLOGY.json', '--gate-proof', '/old/gate.json']
                    rows.append(dict(abi=abi, label=label, kind='perf', round=round_index, profile_mode=mode,
                        runner_arm='worker4' if label == 'worker4' else 'serial',
                        argv=['/python', '-I', '-B', '/tools/real_dlrm_runner_r22.py']+forwarded,
                        runner_argv=forwarded, output=output, binding={}))
    return dict(passed=True, diagnostic_only=True, formal_performance_conclusion_allowed=False,
                completed_cases=42, results=[{}]*42,
                plan=dict(rounds=3, gates=[{}]*6, repeat=rows),
                source_pins={str(i): value for i, value in enumerate(m.REQUIRED_COHORT_SOURCES)})


class CallerCampaignTests(unittest.TestCase):
    def test_exact_eight_adjacent_alternating_pairs_only_replace_output(self):
        m = module(); source = cohort(m); previous = copy.deepcopy(source)
        args = SimpleNamespace(output=Path('/new'), caller_wrapper=Path('/caller.py'), base_wrapper=Path('/base.py'), execute=False)
        plan = m.make_plan(args, source)
        self.assertEqual(len(plan['cases']), 8)
        self.assertEqual([r['mode'] for r in plan['cases']],
                         ['off', 'caller_only', 'caller_only', 'off', 'off', 'caller_only', 'caller_only', 'off'])
        for row in plan['cases']:
            old, new = row['cohort_item']['argv'], row['runner_argv']; index = old.index('--output')+1
            self.assertEqual(new[:index], old[:index]); self.assertEqual(new[index+1:], old[index+1:])
            self.assertNotEqual(new[index], old[index])
            if row['mode'] == 'off': self.assertEqual(row['argv'], new)
            else:
                self.assertIn('--base-wrapper', row['argv']); self.assertIn('--', row['argv'])
                self.assertEqual(row['argv'][row['argv'].index('--')+1:], new[4:])
        self.assertEqual(source, previous); self.assertTrue(plan['plan_only'])

    def test_main_fixed42_and_source_header_are_mandatory(self):
        m = module(); source = cohort(m); m.basic_cohort(source)
        for key, value in (('passed', False), ('completed_cases', 41), ('source_pins', {})):
            bad = copy.deepcopy(source); bad[key] = value
            with self.assertRaises(ValueError): m.basic_cohort(bad)

    def test_observer_pair_is_reported_separately_without_adjusting_stages(self):
        m = module(); rows = []
        for abi in ('std', 'ft'):
            for label in ('candidate-serial', 'worker4'):
                rows += [dict(abi=abi, label=label, mode='off', raw_summary=dict(mean_ns=100)),
                         dict(abi=abi, label=label, mode='caller_only', raw_summary=dict(mean_ns=120),
                              caller_stages={'emb': {'mean_ns': 30}})]
        prior = copy.deepcopy(rows); pairs = m.pair_summary(rows)
        self.assertEqual(len(pairs), 4); self.assertEqual(pairs[0]['caller_only_over_off'], 1.2)
        self.assertEqual(rows, prior); self.assertFalse(pairs[0]['stage_values_corrected'])

    def test_sidecar_failure_flags_are_rejected_before_any_event_processing(self):
        m = module()
        case = dict(runner_argv=['/python', '-I', '-B', '/runner', '--output', '/new/raw.json'])
        for row in ({}, dict(profile_mode='caller_only', instrumentation_complete=False),
                    dict(profile_mode='caller_only', worker_hooks_installed=1)):
            with self.assertRaises(ValueError): m.validate_sidecar(row, case, None, None, 123)

    def test_full2000_serial_sidecar_and_seven_stage_summary_are_validated(self):
        m = module()
        caller = m.load('caller_fixture', PATH.with_name('instrument_caller_only_r23.py'), m.CALLER_SHA)
        analyzer = m.load('partition_fixture', PATH.with_name('analyze_profile_r23.py'), m.ANALYZER_SHA)
        events = []
        spans = [('forward', 10, 100), ('model.bottom_mlp', 20, 30), ('model.apply_emb', 40, 60),
                 ('model.interact_features', 70, 80), ('model.top_mlp', 85, 90), ('tail_sync', 110, 130)]
        for request in range(1, 2001):
            for index, (name, start, end) in enumerate(spans):
                events.append(dict(id=f'123:{request}:{index}', parent=None, request=request, name=name, native_tid=123,
                    start_ns=request*1000+start, end_ns=request*1000+end, start_thread_time_ns=start, end_thread_time_ns=end))
        counts = dict(Counter(row['name'] for row in events))
        case = dict(runner_argv=['/python', '-I', '-B', '/runner', '--output', '/new/raw.json'], cohort_item=dict(runner_arm='serial'))
        profile = dict(schema=1, profile_mode='caller_only', tool_sha256=m.CALLER_SHA,
            base_wrapper=dict(sha256=m.BASE_SHA), runner=dict(sha256=m.RUNNER_SHA), original_exit_code=0,
            failure=None, diagnostic_only=True, performance_conclusion_allowed=False, instrumentation_complete=True,
            temporary_methods_restored=True, requests=2000, original_argv=case['runner_argv'][4:],
            worker_hooks_installed=0, worker_events=0, restoration=[dict(restored=True)], events=events,
            actual_event_counts=counts, expected_event_counts=counts, snapshot_per_request=[], worker_metrics_observed=False,
            native_measurements={key: None for key in ('query_ns', 'execute_ns', 'queue_wait_ns', 'device_ns')})
        m.validate_sidecar(profile, case, caller, analyzer, 123)
        summary = m.caller_stages(dict(timing=dict(durations_ns=[150]*2000)), profile, analyzer)
        self.assertEqual(sum(row['mean_ns'] for row in summary.values()), 150)
        profile['worker_events'] = 1
        with self.assertRaises(ValueError): m.validate_sidecar(profile, case, caller, analyzer, 123)
        profile['worker_events'] = 0; profile['events'][0]['native_tid'] = 124
        with self.assertRaises(ValueError): m.validate_sidecar(profile, case, caller, analyzer, 123)
        profile['events'][0]['native_tid'] = 123
        for row in profile['events']:
            row['start_ns'] -= row['request']*1000; row['end_ns'] -= row['request']*1000
        with self.assertRaisesRegex(ValueError, 'request timelines'):
            m.caller_stages(dict(timing=dict(durations_ns=[150]*2000)), profile, analyzer)

    def test_default_main_only_prints_plan_and_never_invokes_device_tools(self):
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root/'cohort.json'; path.write_text(json.dumps(cohort(m)))
            output = root/'new'; console = io.StringIO()
            with patch.object(m, 'execute', side_effect=AssertionError('must not execute')), \
                 patch.object(m.subprocess, 'run', side_effect=AssertionError('must not invoke SDK/device tool')), \
                 contextlib.redirect_stdout(console):
                code = m.main(['--root', str(root), '--tools', str(root/'tools'), '--control', str(root/'control'),
                               '--cohort-result', str(path), '--output', str(output)])
            self.assertEqual(code, 0); self.assertTrue(json.loads(console.getvalue())['plan_only'])
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
