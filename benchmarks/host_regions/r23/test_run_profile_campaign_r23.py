"""Host-only contract tests for the R23 six-cohort campaign controller."""
import builtins
from collections import Counter
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


CONTROLLER = Path(__file__).with_name('run_profile_campaign_r23.py')
LABELS = ('original-serial', 'candidate-serial', 'worker4')


def load_controller():
    if not CONTROLLER.is_file():
        raise AssertionError('R23 campaign controller is not implemented')
    spec = importlib.util.spec_from_file_location('_test_r23_campaign', CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag(argv, name):
    return argv[argv.index(name) + 1]


def row_mode(row):
    return flag(row['argv'], '--profile-mode') if '--profile-mode' in row['argv'] else 'off'


def bindings_fixture():
    result = {}
    for index, (abi, arm) in enumerate((a, b) for a in ('std', 'ft') for b in ('serial', 'worker4')):
        stem = '/fixture/' + abi + '/' + arm
        result[abi, arm] = dict(abi=abi, arm=arm, python=stem + '/bin/python',
            wheel=stem + '/candidate.whl', wheel_sha256=str(index + 1) * 64,
            native_sha256=str(index + 5) * 64, build_receipt=stem + '/BUILD_RESULT.json',
            build_receipt_sha256='a' * 64,
            dependency_adapter=({'kind': 'none'} if arm == 'serial' else
                dict(kind='pth', path=stem + '/adapter.pth', target='/fixture/dependencies', sha256='b' * 64)))
    return result


class CampaignContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_controller()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.wrapper = self.root / 'wrapper.py'
        self.wrapper.write_text('# host fixture only\n', encoding='utf-8')
        self.bindings = bindings_fixture()
        self.before = copy.deepcopy(self.bindings)
        self.helper = SimpleNamespace(load_arm_manifest=mock.Mock(return_value=self.bindings))

    def args(self, *extra):
        return self.module.parse_args(['--root', str(self.root), '--tools', str(self.tools),
            '--output', str(self.root / 'campaign'), '--wrapper', str(self.wrapper), *extra])

    def test_import_has_no_main_process_or_accelerator_side_effects(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torch_npu', 'numpy', 'sklearn'):
                raise AssertionError('campaign import attempted accelerator/dependency import: ' + name)
            return original_import(name, *args, **kwargs)
        with mock.patch.object(builtins, '__import__', side_effect=guarded_import), \
                mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('import launched a child')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('import ran a command')), \
                mock.patch.object(sys, 'argv', ['controller-import', '--invalid-main-argument']):
            loaded = load_controller()
        self.assertTrue(callable(loaded.make_plan))

    def test_default_plan_has_six_gates_and_thirty_six_fixed_exposures(self):
        args = self.args()
        self.assertEqual(args.rounds, 3)
        self.assertFalse(args.execute)
        self.assertIsNone(args.wrapper_sha256)
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('planning launched a child')), \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('planning ran a command')):
            plan = self.module.make_plan(args, self.helper)
        self.helper.load_arm_manifest.assert_called_once_with(args.arms_manifest)
        self.assertEqual(len(plan['gates']), 6)
        self.assertEqual(len(plan['repeat']), 36)
        self.assertFalse(plan.get('stability'))
        expected = {(abi, label) for abi in ('std', 'ft') for label in LABELS}
        self.assertEqual({(r['abi'], r['label']) for r in plan['gates']}, expected)
        for row in plan['gates']:
            self.assertEqual(flag(row['argv'], '--kind'), 'gate')
            self.assertEqual(flag(row['argv'], '--warmup'), '0')
            self.assertEqual(flag(row['argv'], '--iterations'), '1')
        counts = Counter((r['abi'], r['label'], row_mode(r)) for r in plan['repeat'])
        self.assertEqual(counts, Counter({(abi, label, mode): 3
            for abi, label in expected for mode in ('off', 'stages')}))
        for round_number in (1, 2, 3):
            rows = [r for r in plan['repeat'] if r['round'] == round_number]
            self.assertEqual(len(rows), 12)
            self.assertEqual({(r['abi'], r['label'], row_mode(r)) for r in rows}, set(counts))
        for row in plan['repeat']:
            self.assertEqual(flag(row['argv'], '--kind'), 'perf')
            self.assertEqual(flag(row['argv'], '--warmup'), '20')
            self.assertEqual(flag(row['argv'], '--iterations'), '2000')
        outputs = [r['output'] for r in plan['gates'] + plan['repeat']]
        self.assertEqual(len(set(outputs)), 42)
        self.assertFalse((self.root / 'campaign').exists(), 'pure planning created output state')
        self.assertEqual(self.bindings, self.before, 'planning mutated frozen physical arm bindings')

    def test_candidate_serial_reuses_candidate_artifact_and_transmits_serial_runner_arm(self):
        digest = hashlib.sha256(self.wrapper.read_bytes()).hexdigest()
        args = self.args('--wrapper-sha256', digest, '--rounds', '1')
        plan = self.module.make_plan(args, self.helper)
        for abi in ('std', 'ft'):
            candidate = self.bindings[abi, 'worker4']
            rows = [r for r in plan['gates'] + plan['repeat']
                if r['abi'] == abi and r['label'] == 'candidate-serial']
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row['binding'], candidate)
                self.assertEqual(row['binding']['arm'], 'worker4')
                self.assertEqual(row['runner_arm'], 'serial')
                argv = row['argv']
                self.assertEqual(argv[:3], [candidate['python'], '-I', '-B'])
                self.assertEqual(flag(argv, '--arm'), 'serial')
                self.assertEqual(flag(argv, '--wheel'), candidate['wheel'])
                self.assertEqual(flag(argv, '--expected-wheel-sha256'), candidate['wheel_sha256'])
                self.assertEqual(flag(argv, '--expected-native-sha256'), candidate['native_sha256'])
            off = next(r for r in rows if flag(r['argv'], '--kind') == 'perf' and row_mode(r) == 'off')
            stage = next(r for r in rows if row_mode(r) == 'stages')
            self.assertEqual(Path(off['argv'][3]), self.tools / 'real_dlrm_runner_r22.py')
            self.assertEqual(Path(stage['argv'][3]), self.wrapper)
            self.assertEqual(flag(stage['argv'], '--runner'), off['argv'][3])
            self.assertTrue(flag(stage['argv'], '--profile-output'))
            separator = stage['argv'].index('--')
            forwarded = stage['argv'][separator + 1:]
            self.assertEqual(forwarded[forwarded.index('--arm') + 1], 'serial')
            self.assertEqual(forwarded[forwarded.index('--output') + 1], stage['output'])
            self.assertNotEqual(flag(stage['argv'], '--profile-output'), stage['output'])
        self.assertEqual(len(plan['repeat']), 12)
        self.assertEqual(self.bindings, self.before)

    def test_execute_rejects_existing_output_before_any_child_or_sdk_preflight(self):
        args = self.args('--execute')
        plan = self.module.make_plan(args, self.helper)
        args.output.mkdir()
        sentinel = args.output / 'RESULT.json'
        original = b'{"preserve": "existing evidence"}\n'
        sentinel.write_bytes(original)
        self.helper.run_child = mock.Mock(side_effect=AssertionError('existing output launched a child'))
        self.helper.sdk_environment = mock.Mock(side_effect=AssertionError('existing output entered SDK setup'))
        with mock.patch.object(self.module.sys, 'platform', 'linux'), \
                mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('existing output launched a process')):
            with self.assertRaisesRegex(ValueError, 'exclusive output'):
                self.module.execute(args, plan, self.helper)
        self.helper.run_child.assert_not_called()
        self.helper.sdk_environment.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), original)
        self.assertEqual(list(args.output.iterdir()), [sentinel])

    def test_profile_acceptance_rejects_incomplete_coverage_or_wrong_identity(self):
        plan = self.module.make_plan(self.args('--rounds', '1'), self.helper)
        item = next(r for r in plan['repeat'] if r['label'] == 'candidate-serial' and row_mode(r) == 'stages')
        counts = {name: 2000 for name in ('forward', 'tail_sync', 'model.apply_emb',
            'model.bottom_mlp', 'model.interact_features', 'model.top_mlp')}
        wrapper_sha = hashlib.sha256(self.wrapper.read_bytes()).hexdigest()
        profile = dict(diagnostic_only=True, performance_conclusion_allowed=False,
            profile_mode='stages', original_exit_code=0, failure=None, tool_sha256=wrapper_sha,
            runner=dict(sha256=self.module.PINS['real_dlrm_runner_r22.py']),
            original_argv=copy.deepcopy(item['runner_argv']), temporary_methods_restored=True,
            instrumentation_complete=True, requests=2000, expected_event_counts=counts,
            actual_event_counts=copy.deepcopy(counts))
        self.module.validate_profile(profile, item, wrapper_sha)
        for change in ('incomplete', 'count', 'argv', 'wrapper', 'restoration'):
            bad = copy.deepcopy(profile)
            if change == 'incomplete':
                bad['instrumentation_complete'] = False
            elif change == 'count':
                bad['actual_event_counts']['model.apply_emb'] = 1999
            elif change == 'argv':
                bad['original_argv'].extend(['--unexpected-extra-observer'])
            elif change == 'wrapper':
                bad['tool_sha256'] = '0' * 64
            else:
                bad['temporary_methods_restored'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.validate_profile(bad, item, wrapper_sha)


if __name__ == '__main__':
    unittest.main()
