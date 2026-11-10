"""R28 scheduling and evidence contracts, without launching a benchmark."""
import copy
import importlib.util
import itertools
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Controller(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'benchmarks/eager_parallel_r28/controller.py'
        self.assertTrue(path.is_file(), 'reusable R28 controller is missing')
        self.module = load(path, 'r28_controller_test')
        self.worker = load(ROOT / 'benchmarks/eager_parallel_r27/host_e2e.py', 'r28_controller_worker_test')

    def test_plan_balances_lengths_arm_permutations_and_fresh_process_labels(self):
        lengths, arms = (2, 8, 16, 32), ('off', 'e', 'ei')
        orders = list(itertools.permutations(arms))
        cold = self.module.make_plan('cold')
        perf = self.module.make_plan('perf')
        self.assertEqual(len(cold), 12)
        self.assertEqual(len(perf), 144)
        self.assertEqual(sum(row['samples'] for row in perf), 432000)
        self.assertEqual(len({row['label'] for row in perf}), 144)
        for index, length in enumerate(lengths):
            rows = [row for row in cold if row['bag_length'] == length]
            self.assertEqual([row['arm'] for row in rows], list(arms[index % 3:] + arms[:index % 3]))
            self.assertTrue(all(row['round'] == 0 and row['warmup'] == 0 and row['samples'] == 1 for row in rows))
        for number in range(12):
            rows = [row for row in perf if row['round'] == number]
            expected_lengths = lengths[number % 4:] + lengths[:number % 4]
            self.assertEqual([rows[i]['bag_length'] for i in range(0, 12, 3)], list(expected_lengths))
            for length in lengths:
                selected = [row for row in rows if row['bag_length'] == length]
                self.assertEqual(tuple(row['arm'] for row in selected), orders[(number + lengths.index(length)) % 6])
                self.assertTrue(all(row['warmup'] == 50 and row['samples'] == 3000 for row in selected))
        self.assertEqual(perf[0]['label'], '000-L2-r0-off')
        for args in (dict(rounds=1), dict(samples=1), dict(rounds=True), dict(samples=True)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'fixed profile'):
                self.module.make_plan('perf', **args)

    def provenance(self, length):
        identity = copy.deepcopy(self.module.IDENTITY['global'])
        identity['input_sha256'] = self.module.IDENTITY['per_l'][str(length)]['input_sha256']
        return dict(identity=identity, workload=dict(bag_length=length, batch=128, tables=26,
            embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
            original_input_sha256=self.module.IDENTITY['per_l']['2']['input_sha256'],
            extra_input_seed=20260919,
            input_policy='original two IDs per bag plus fixed per-table Bx30 prefix',
            tables_detail=[dict(table=table, indices_shape=[128 * length], logical_indices_shape=[128, length],
                offsets_shape=[128], offset_first=0, offset_last=127 * length,
                bag_length_min=length, bag_length_max=length) for table in range(26)]))

    def test_identity_keeps_global_weights_and_dense_but_pins_each_l_input(self):
        for length in (2, 8, 16, 32):
            identity = self.module.workload_identity(self.provenance(length), length)
            self.assertEqual(identity['global'], self.module.IDENTITY['global'])
            self.assertEqual(identity['per_l'], self.module.IDENTITY['per_l'][str(length)])
        variants = [self.provenance(8) for _ in range(6)]
        variants[0]['identity']['input_sha256'] = self.module.IDENTITY['per_l']['2']['input_sha256']
        variants[1]['identity']['dense_sha256'] = '0' * 64
        variants[2]['workload']['tables_detail'][0]['offset_last'] += 1
        variants[3]['workload']['tables_detail'][0]['indices_shape'] = [256]
        variants[4]['workload']['extra_input_seed'] += 1
        variants[5]['workload']['bag_length'] = True
        for provenance in variants:
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                self.module.workload_identity(provenance, 8)

    def snapshot(self, forwards, arm):
        operations = {}
        for name, count in self.worker.ORIGINAL_COUNTS.items():
            executed = count * forwards
            streams = {'11': executed} if executed else {}
            if arm != 'off' and name == 'aclnnEmbeddingBag':
                streams = {'12': 13 * forwards, '13': 13 * forwards} if forwards else {}
            if arm == 'ei' and name in ('aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex'):
                streams = {'14': executed} if executed else {}
            if forwards:
                operations[name] = dict(prepared=executed, executed=executed, failed=0,
                                        cancelled=0, stream_executions=streams)
        native = dict(backend='shadow', enabled=False, entered=0, closure=dict(passed=True))
        if arm != 'off':
            native.update(backend='native_immediate', mode=arm, actual_stream_scheduling=True,
                stream_ids=[11, 12, 13, 14], first_error=None, quarantined=False,
                active_calls=0, retained_owners=0, failed=0, event_records=33 * forwards,
                event_waits=8 * forwards, scheduled_embedding=26 * forwards,
                scheduled_cat=2 * forwards if arm == 'ei' else 0,
                scheduled_bmm=forwards if arm == 'ei' else 0, scheduled_index=forwards if arm == 'ei' else 0)
        return dict(native=native, runtime=dict(backend_kind='HOST_SIM_R27_CPU', submitted=forwards,
            completed=forwards, errors=0, numeric_supported=True, real_sdk_qualified=False,
            injected_faults=0, fault_active=False), opapi=dict(operations=operations, active_kernels=0,
            live_objects=0, descriptors_created=42 * forwards, descriptors_destroyed=42 * forwards))

    @staticmethod
    def error(shape, atol=1e-5):
        return dict(passed=True, shape=shape, finite=True, rtol=1e-4, atol=atol, max_abs_error=0.0)

    def report(self, row):
        samples, warmup, arm = row['samples'], row['warmup'], row['arm']
        checks = 0 if row['kind'] == 'cold' else 8
        before_warmup = self.snapshot(checks, arm)
        before = self.snapshot(checks + warmup, arm)
        after = self.snapshot(checks + warmup + samples, arm)
        report = dict(passed=True, pid=101, kind=row['kind'], round=row['round'], arm=arm,
            samples=samples, warmup=warmup, worker_sha256='worker', source_pins={'worker': 'worker'},
            artifact_pins={'artifact': 'artifact'}, original_model_executed=True,
            host_sim_numeric_validated=True, gil_enabled=False, real_npu_performance_claim=False,
            numerical_checks_in_timing=False, startup_included=False,
            compute_backend='host_sim_numerical_cpu', mapping_before={'library': 'pinned'},
            mapping_after={'library': 'pinned'}, durations_ns=[1_000_000] * samples,
            window_wall_ns=1_000_000 * samples + 1, window_process_cpu_ns=1_000_000 * samples,
            forward_calls=warmup + samples, synchronize_calls=warmup + samples,
            timing=dict(samples=samples, median_ns=1_000_000, p95_ns=1_000_000),
            before_warmup=before_warmup, before=before, after=after,
            closed=self.snapshot(checks + warmup + samples + 1, arm),
            provenance=self.provenance(row['bag_length']),
            numeric_final=self.error([128, 1]), correctness=[self.error([128, 1])] * (checks or 1),
            stage_validation=dict(passed=True, checks_in_timing=False,
                embedding_errors=[self.error([128, 16]) for _ in range(26)],
                interaction_error=self.error([128, 367], 1e-4), output_error=self.error([128, 1])))
        report['sdk_timed_gate'] = self.worker.sdk_delta(before['opapi'], after['opapi'], samples)
        report['native_timed_gate'] = self.worker.native_delta(before['native'], after['native'], arm, samples)
        report['sdk_stream_gate'] = self.worker.sdk_stream_gate(after['opapi'], arm, before['opapi'], samples)
        config = dict(source_pins=report['source_pins'], artifact_pins=report['artifact_pins'],
                      bag_length=row['bag_length'])
        life = dict(pid=101, pgid=101, returncode=0, timed_out=False, group_closed=True,
                    monotonic_start_ns=100, monotonic_end_ns=200)
        return report, config, life

    def test_report_validation_recomputes_raw_counts_and_rejects_false_or_saturated_evidence(self):
        for kind in ('cold', 'perf'):
            for arm in ('off', 'e', 'ei'):
                row = next(row for row in self.module.make_plan(kind) if row['arm'] == arm)
                report, config, life = self.report(row)
                self.module.validate_report(report, row, config, life, self.worker, worker_sha256='worker')
        row = self.module.make_plan('perf')[2]
        report, config, life = self.report(row)
        variants = []
        for mutate in (
            lambda r: r['after']['opapi']['operations']['aclnnEmbeddingBag'].__setitem__('executed', 1),
            lambda r: r['before']['opapi']['operations']['aclnnAddmm'].__setitem__('executed', 1),
            lambda r: r['durations_ns'].__setitem__(0, True),
            lambda r: r['timing'].__setitem__('median_ns', 2),
            lambda r: r['stage_validation']['embedding_errors'][0].__setitem__('passed', False),
            lambda r: r['stage_validation']['interaction_error'].__setitem__('shape', [128, 1]),
            lambda r: r['stage_validation'].__setitem__('checks_in_timing', True),
            lambda r: r['closed']['native']['closure'].__setitem__('passed', False),
            lambda r: r.__setitem__('gil_enabled', True),
        ):
            changed = copy.deepcopy(report)
            mutate(changed)
            variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.module.validate_report(changed, row, config, life, self.worker, worker_sha256='worker')
        row = self.module.make_plan('cold')[0]
        report, config, life = self.report(row)
        report['before']['opapi']['operations'] = self.snapshot(1, 'off')['opapi']['operations']
        with self.assertRaisesRegex(ValueError, 'first NPU forward'):
            self.module.validate_report(report, row, config, life, self.worker, worker_sha256='worker')

    def test_serial_intervals_reject_overlap_pid_reuse_and_noninteger_timestamps(self):
        rows = [dict(pid=11, monotonic_start_ns=1, monotonic_end_ns=2),
                dict(pid=12, monotonic_start_ns=2, monotonic_end_ns=3)]
        self.module.check_serial(rows)
        for change in ({'pid': 11}, {'monotonic_start_ns': 1}, {'monotonic_end_ns': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.check_serial([rows[0], dict(rows[1], **change)])

    def base_config(self, directory):
        root = Path(directory)
        for relative in self.module.SOURCE_HASHES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        package = root / 'candidate/torch_npu'
        package.mkdir(parents=True)
        artifacts = {}
        for relative in ('_C.cpython-314t-aarch64-linux-gnu.so', '__init__.py', '_eager_parallel.py', 'lib/libtorch_npu.so'):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('existing candidate ' + relative)
            artifacts[str(path)] = self.module.sha(path)
        libraries = {}
        for name in ('libopapi.so', 'libascendcl.so', 'libacl_op_compiler.so', 'libr27_hostsim_runtime.so'):
            path = root / 'sdk' / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('existing SDK ' + name)
            libraries[name] = dict(path=str(path), sha256=self.module.sha(path))
            artifacts[str(path)] = self.module.sha(path)
        path = package / 'lib/libtorch_npu.so'
        libraries['libtorch_npu.so'] = dict(path=str(path), sha256=self.module.sha(path))
        sources = {}
        helpers = {}
        for key, relative in (('measurement_helper', 'benchmarks/eager_parallel_r26/host_e2e.py'),
                              ('mapping_helper', 'benchmarks/eager_parallel_r26/shadow_hostsim.py'),
                              ('frozen_model_helper', 'benchmarks/host_regions/real_dlrm_runner_r22.py'),
                              ('reference', 'benchmarks/eager_parallel_r26/reference.py')):
            path = root / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
            helpers[key] = str(path)
            sources[str(path)] = self.module.sha(path)
        executable = root / 'runtime/python'
        executable.parent.mkdir()
        executable.write_text('externally qualified exact interpreter')
        runtime_identity = dict(python_version='3.14.6', python_full_version='3.14.6 experimental free-threading build',
            gil_enabled=False, torch_git_version='cf30153c4c131c8164ee7798e5022d810682e2cb',
            artifact_pins={str(executable.resolve()): self.module.sha(executable)})
        return dict(schema=1, package=str(package), cpus=list(self.module.CPUS), python=str(executable),
            runtime_identity=runtime_identity, ld_library_path=str(root / 'sdk'),
            legacy_deps=str(root / 'legacy'), original_source=str(root / 'original'),
            libraries=libraries, source_pins=sources, artifact_pins=artifacts, **helpers)

    def test_configs_preserve_candidate_and_helpers_and_pin_runtime_and_new_sources(self):
        self.assertTrue(hasattr(self.module, 'prepare_configs'), 'pinned configuration preparation is missing')
        with tempfile.TemporaryDirectory() as directory:
            base = self.base_config(directory)
            unchanged = copy.deepcopy(base)
            configs = self.module.prepare_configs(base, Path(directory))
            self.assertEqual(base, unchanged)
            self.assertEqual(set(configs), {2, 8, 16, 32})
            for length, config in configs.items():
                self.assertEqual(config['libraries'], base['libraries'])
                for key in ('package', 'legacy_deps', 'original_source', 'measurement_helper', 'mapping_helper', 'python'):
                    self.assertEqual(config[key], base[key])
                self.assertEqual(config['bag_length'], length)
                self.assertIs(config['validate_stages'], True)
                self.assertEqual(config['reference'], str(Path(directory) / 'benchmarks/eager_parallel_r28/workload.py'))
                self.assertTrue(base['artifact_pins'].items() <= config['artifact_pins'].items())
                self.assertTrue(base['runtime_identity']['artifact_pins'].items() <= config['artifact_pins'].items())
                for relative, digest in self.module.SOURCE_HASHES.items():
                    self.assertEqual(config['source_pins'][str(Path(directory) / relative)], digest)
            for key, value in (('python_version', '3.14.7'), ('gil_enabled', True), ('python_full_version', '3.14.5')):
                changed = copy.deepcopy(base)
                changed['runtime_identity'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.module.prepare_configs(changed, Path(directory))
            changed = copy.deepcopy(base)
            changed['cpus'] = [0] * 8
            with self.assertRaises(ValueError):
                self.module.prepare_configs(changed, Path(directory))

    def test_environment_uses_exact_declared_runtime_path_and_removes_gil_override(self):
        self.assertTrue(hasattr(self.module, 'build_environment'), 'fixed worker environment preparation is missing')
        base = dict(package='/candidate/torch_npu', ld_library_path='/unchanged/sdk:/qualified/torch/lib')
        environment = self.module.build_environment(base, dict(PATH='/bin', PYTHON_GIL='0',
            PYTHONPATH='/wrong', LD_PRELOAD='/wrong.so', QWEN_ACL_HOST_SIM='0', CUSTOM='keep'))
        self.assertNotIn('PYTHON_GIL', environment)
        self.assertNotIn('LD_PRELOAD', environment)
        self.assertEqual(environment['LD_LIBRARY_PATH'], base['ld_library_path'])
        self.assertEqual(environment['PYTHONPATH'], str(Path(base['package']).parent))
        self.assertEqual(environment['TASK_QUEUE_ENABLE'], '0')
        self.assertEqual(environment['OMP_NUM_THREADS'], '1')
        self.assertEqual(environment['CUSTOM'], 'keep')

    def test_unverified_helper_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'malicious.py'
            path.write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaisesRegex(ValueError, 'before import'):
                self.module.load_pinned(path, '0' * 64, 'unverified_r28_helper')


if __name__ == '__main__':
    unittest.main()
