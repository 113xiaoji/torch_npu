"""Fixed mixed-L inputs over the same existing build.

No build or installation entry points are provided. The base config must name
an externally qualified exact free-threaded interpreter and its runtime pins.
"""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

CPUS = [144, 146, 148, 150, 152, 154, 156, 158]
LENGTHS = ('mixed',)
BAG_LENGTHS = (3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1)
ARMS = ('off', 'e', 'ei')
OWNED_SHA256 = 'cfce70d0a84253448f6c97a3b08f15e8d5afa4486e6c49c0b286cba0c23792e3'
TOPOLOGY_SHA256 = '5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a'
SOURCE_HASHES = {
    'benchmarks/eager_parallel_r27/host_e2e.py': '66a752d345233f8f20a64b18d94b0966d609fcd490c31652567788c902d103ca',
    'benchmarks/eager_parallel_r30/workload.py': '8c058e4a9cfadaf5cd42083a1a0e135372a93568b7537b66e59989f3290db23a',
    'benchmarks/eager_parallel_r29/workload.py': '361851afe6fe1e31f5ab78b88c7f191592403ffef96c90372b8028e78ac4a3f9',
    'benchmarks/eager_parallel_r28/workload.py': '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960',
    'benchmarks/eager_parallel_r26/reference.py': '162cd92ddc2cb58819ee6e9a921b86c29933ab0651ccf5ed66c1fb4a29c17c4f',
    'benchmarks/host_regions/real_dlrm_runner_r22.py': 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f',
}
IDENTITY = {
    'global': dict(model_sha256='9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5',
        state_sha256='6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19',
        dense_sha256='352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9'),
    'per_l': {'mixed': dict(bag_length='mixed', input_sha256='40d6a16803f328885b66357bbcfdb680eb5578b15a641a61421069e4755be511')},
}

PROFILE = dict(profiles=list(LENGTHS), bag_lengths=list(BAG_LENGTHS), cpus=CPUS, numa_node=1, physical_cores=8, llc_count=1,
    state_seed=20260917, input_seed=20260918, extra_input_seed=20260920,
    cold=dict(rounds=1, warmup=0, samples=1, processes=3),
    perf=dict(rounds=12, warmup=50, samples=3000, processes=36),
    process_policy='serial fresh owned process; 240-second timeout; start_new_session',
    ordering='one fixed mixed profile; arm permutation round modulo 6',
    real_npu_performance_claim=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def load_pinned(path, digest, name):
    require(Path(path).is_absolute() and sha(path) == digest, 'source pin mismatch before import: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pins(pins, label):
    require(type(pins) is dict and pins, label + ' pins required')
    for path, digest in pins.items():
        require(Path(path).is_absolute() and type(digest) is str and sha(path) == digest,
                label + ' pin mismatch: ' + str(path))


def make_plan(kind, *, rounds=12, samples=3000):
    require(kind in ('cold', 'perf'), 'kind must be cold or perf')
    require(type(rounds) is int and rounds == 12 and type(samples) is int and samples == 3000,
            'fixed profile requires 12 rounds and 3000 perf samples')
    plan, orders = [], list(itertools.permutations(ARMS))
    for number in range(1 if kind == 'cold' else rounds):
        arms = ARMS if kind == 'cold' else orders[number % 6]
        for arm in arms:
            plan.append(dict(kind=kind, bag_length='mixed', round=number, arm=arm,
                warmup=0 if kind == 'cold' else 50, samples=1 if kind == 'cold' else samples,
                label=f'{len(plan):03d}-Lmixed-r{number}-{arm}'))
    return plan


def workload_identity(provenance, length):
    require(type(length) is str and length == 'mixed', 'unqualified mixed profile')
    actual = provenance['identity']
    global_identity = {key: actual.get(key) for key in IDENTITY['global']}
    require(global_identity == IDENTITY['global'], 'original model/state/dense identity differs')
    per_l = dict(bag_length=length, input_sha256=actual.get('input_sha256'))
    require(per_l == IDENTITY['per_l']['mixed'], 'mixed input identity differs')
    workload = provenance['workload']
    expected = dict(bag_length='mixed', bag_lengths=list(BAG_LENGTHS), batch=128, tables=26,
        embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
        original_input_sha256='239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6',
        base_input_sha256='9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782',
        extra_input_seed=20260920,
        input_policy='per-table prefixes of frozen R29 L=256 IDs; fixed MLPerf multi-hot length vector')
    require(type(workload.get('bag_length')) is str and
            all(workload.get(key) == value for key, value in expected.items()), 'mixed workload profile differs')
    details = workload.get('tables_detail')
    require(type(details) is list and len(details) == 26, '26 table input contracts required')
    for table, (row, size) in enumerate(zip(details, BAG_LENGTHS)):
        require(row == dict(table=table, indices_shape=[128 * size], logical_indices_shape=[128, size],
            offsets_shape=[128], offset_first=0, offset_last=127 * size,
            bag_length_min=size, bag_length_max=size), 'table shape/offset contract differs')
    return {'global': global_identity, 'per_l': per_l}


def check_serial(rows):
    seen, previous_end = set(), 0
    for row in rows:
        start, end, pid = (row.get(key) for key in ('monotonic_start_ns', 'monotonic_end_ns', 'pid'))
        require(type(pid) is int and pid > 0 and pid not in seen, 'fresh unique worker PID required')
        require(type(start) is int and type(end) is int and previous_end <= start < end,
                'serial monotonic process intervals overlap or are invalid')
        seen.add(pid)
        previous_end = end


def check_error(record, shape, *, atol=1e-5):
    require(record.get('passed') is True and record.get('finite') is True and record.get('shape') == shape
            and record.get('rtol') == 1e-4 and record.get('atol') == atol,
            'numeric stage shape/tolerance/pass differs')
    error = record.get('max_abs_error')
    require(type(error) in (int, float) and math.isfinite(error) and error >= 0,
            'numeric error must be finite and nonnegative')


def validate_report(result, row, config, life, worker, *, worker_sha256):
    require(type(life.get('pid')) is int and life['pid'] > 0 and life.get('pgid') == life['pid']
            and type(life.get('returncode')) is int and life['returncode'] == 0
            and life.get('timed_out') is False and life.get('group_closed') is True, 'owned worker failed')
    check_serial([life])
    require(result.get('pid') == life['pid'] and result.get('passed') is True and 'error' not in result,
            'worker result/PID failed')
    for key in ('kind', 'round', 'arm', 'samples', 'warmup'):
        require(type(result.get(key)) is type(row[key]) and result[key] == row[key], 'worker metadata differs: ' + key)
    require(result.get('worker_sha256') == worker_sha256 and result.get('source_pins') == config['source_pins']
            and result.get('artifact_pins') == config['artifact_pins'], 'worker source/artifact pins differ')
    require(result.get('original_model_executed') is True and result.get('host_sim_numeric_validated') is True
            and result.get('gil_enabled') is False and result.get('real_npu_performance_claim') is False
            and result.get('numerical_checks_in_timing') is False and result.get('startup_included') is False
            and result.get('compute_backend') == 'host_sim_numerical_cpu', 'runtime/measurement identity differs')
    require(bool(result.get('mapping_before')) and result['mapping_before'] == result.get('mapping_after'),
            'loaded library mapping drift')
    samples, warmup, arm = row['samples'], row['warmup'], row['arm']
    durations = result.get('durations_ns')
    require(type(durations) is list and len(durations) == samples and
            all(type(value) is int and value > 0 for value in durations), 'raw duration count/value differs')
    ordered = sorted(durations)
    require(result.get('timing') == dict(samples=samples, median_ns=statistics.median(ordered),
            p95_ns=ordered[math.ceil(.95 * samples) - 1]), 'raw timing statistic differs')
    require(type(result.get('window_wall_ns')) is int and result['window_wall_ns'] >= sum(durations)
            and (row['kind'] != 'perf' or result['window_wall_ns'] >= 2_000_000_000), 'measurement wall window differs')
    require(result.get('forward_calls') == result.get('synchronize_calls') == warmup + samples,
            'measured forward/synchronize count differs')
    if row['kind'] == 'cold':
        require(result['before']['opapi']['operations'] == {} and
                result['before_warmup']['opapi']['operations'] == {}, 'cold must time the first NPU forward')
    for phase in ('before_warmup', 'before', 'after', 'closed'):
        snapshot = result[phase]
        worker.check_runtime(snapshot['runtime'])
        require(type(snapshot['opapi'].get('active_kernels')) is int and snapshot['opapi']['active_kernels'] == 0,
                'SDK kernel still active')
    require(result['closed']['native']['closure']['passed'] is True, 'native closure failed')
    before, after = result['before'], result['after']
    require(result.get('sdk_timed_gate') == worker.sdk_delta(before['opapi'], after['opapi'], samples),
            'declared timed SDK gate differs from raw counts')
    require(result.get('native_timed_gate') == worker.native_delta(before['native'], after['native'], arm, samples),
            'declared timed native gate differs from raw counts')
    require(result.get('sdk_stream_gate') == worker.sdk_stream_gate(after['opapi'], arm, before['opapi'], samples),
            'declared stream gate differs from raw counts')
    if warmup:
        earlier = result['before_warmup']
        worker.sdk_delta(earlier['opapi'], before['opapi'], warmup)
        worker.native_delta(earlier['native'], before['native'], arm, warmup)
        worker.sdk_stream_gate(before['opapi'], arm, earlier['opapi'], warmup)
    # The single staged check follows the timed after snapshot and precedes closure.
    worker.sdk_delta(after['opapi'], result['closed']['opapi'], 1)
    worker.native_delta(after['native'], result['closed']['native'], arm, 1)
    worker.sdk_stream_gate(result['closed']['opapi'], arm, after['opapi'], 1)
    correctness = result.get('correctness')
    require(type(correctness) is list and len(correctness) == (1 if row['kind'] == 'cold' else 8),
            'original full-forward correctness count differs')
    for error in [*correctness, result['numeric_final']]:
        check_error(error, [128, 1])
    stages = result['stage_validation']
    require(stages.get('passed') is True and stages.get('checks_in_timing') is False and
            type(stages.get('embedding_errors')) is list and len(stages['embedding_errors']) == 26,
            'all 26 intermediate bag checks must pass outside timing')
    for error in stages['embedding_errors']:
        check_error(error, [128, 16])
    check_error(stages['interaction_error'], [128, 367], atol=1e-4)
    check_error(stages['output_error'], [128, 1])
    require(config.get('bag_length') == row['bag_length'], 'config bag length differs')
    return workload_identity(result['provenance'], row['bag_length'])


def prepare_configs(base, runner):
    require(base.get('schema') == 1 and base.get('cpus') == CPUS, 'fixed CPU/schema profile differs')
    runtime = base.get('runtime_identity', {})
    full = runtime.get('python_full_version', runtime.get('python_full', ''))
    require(runtime.get('python_version') == '3.14.6' and runtime.get('gil_enabled') is False
            and runtime.get('torch_git_version') == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
            and full.startswith('3.14.6 ') and 'free-thread' in full, 'qualified runtime identity differs')
    verify_pins(base['artifact_pins'], 'native artifact')
    verify_pins(runtime['artifact_pins'], 'runtime artifact')
    verify_pins(base['source_pins'], 'existing helper')
    require(Path(base['python']).is_absolute() and str(Path(base['python']).resolve()) in
            {str(Path(p).resolve()) for p in runtime['artifact_pins']}, 'interpreter must be artifact-pinned')
    require(type(base.get('ld_library_path')) is str and bool(base['ld_library_path']), 'qualified library path required')
    sources = dict(base['source_pins'])
    for relative, digest in SOURCE_HASHES.items():
        path = str(Path(runner) / relative)
        require(sha(path) == digest, 'new source pin mismatch: ' + relative)
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    result = {}
    for length in LENGTHS:
        config = copy.deepcopy(base)
        config.update(bag_length=length, validate_stages=True,
            reference=str(Path(runner) / 'benchmarks/eager_parallel_r30/workload.py'), source_pins=dict(sources))
        for path, digest in runtime['artifact_pins'].items():
            require(path not in config['artifact_pins'] or config['artifact_pins'][path] == digest,
                    'conflicting runtime artifact pin')
            config['artifact_pins'][path] = digest
        result[length] = config
    return result


def build_environment(base, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = {key: value for key, value in inherited.items() if not key.startswith(
        ('PYTHON', 'LD_', 'TORCH', 'ASCEND', 'FTLAB', 'HDB', 'QWEN', 'TASK_QUEUE', 'PER_STREAM'))}
    env.update(PYTHONPATH=str(Path(base['package']).parent), PYTHONDONTWRITEBYTECODE='1',
        TORCH_DEVICE_BACKEND_AUTOLOAD='0', TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1',
        LD_LIBRARY_PATH=base['ld_library_path'], QWEN_ACL_HOST_SIM='1', TASK_QUEUE_ENABLE='0',
        PER_STREAM_QUEUE='0', ASCEND_LAUNCH_BLOCKING='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return env


def check_qualification(path, context, configs, worker):
    qualification = read(path)
    require(qualification.get('passed') is True and qualification.get('kind') == 'cold'
            and len(qualification.get('rows', [])) == 3, 'complete cold qualification required')
    for key in ('source_pins', 'artifact_pins', 'runtime_identity', 'identity', 'environment', 'profile', 'boot'):
        require(qualification['context'][key] == context[key], 'cold qualification drift: ' + key)
    expected = make_plan('cold')
    rows = qualification['rows']
    require([row['label'] for row in rows] == [row['label'] for row in expected], 'cold arm set differs')
    check_serial(rows)
    for row, planned in zip(rows, expected):
        result = read(row['result'])
        require(sha(row['result']) == row['result_sha256'], 'cold result changed')
        life = read(row['lifecycle'])
        require(sha(row['lifecycle']) == row['lifecycle_sha256'], 'cold lifecycle changed')
        validate_report(result, planned, configs[planned['bag_length']], life, worker,
                        worker_sha256=context['source_pins'][str(Path(worker.__file__).resolve())])
    return qualification


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--base-config', required=True, type=Path)
    parser.add_argument('--runner-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kind', required=True, choices=('cold', 'perf'))
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--samples', type=int, default=3000)
    args = parser.parse_args(argv)
    require(not sys.flags.optimize, 'optimized controller forbidden')
    plan = make_plan(args.kind, rounds=args.rounds, samples=args.samples)
    require(args.kind != 'perf' or args.qualification is not None, 'perf requires cold qualification')
    base = read(args.base_config)
    runner = args.runner_root.resolve(strict=True)
    configs = prepare_configs(base, runner)
    owned_path = Path(base['controller_helper'])
    topology_path = Path(base['topology_helper'])
    require(base['source_pins'].get(str(owned_path)) == OWNED_SHA256 and
            base['source_pins'].get(str(topology_path)) == TOPOLOGY_SHA256, 'control helpers must be pinned')
    support = load_pinned(owned_path, OWNED_SHA256, 'r30_owned_process')
    topology = load_pinned(topology_path, TOPOLOGY_SHA256, 'r30_topology').capture_topology(CPUS)
    require(topology['selected_cpus'] == CPUS and topology['choice']['numa_node'] == 1
            and topology['choice']['physical_cores'] == 8 and topology['choice']['llc_count'] == 1,
            'eight physical CPUs on NUMA1/one LLC required')
    worker_path = runner / 'benchmarks/eager_parallel_r27/host_e2e.py'
    worker = load_pinned(worker_path, SOURCE_HASHES['benchmarks/eager_parallel_r27/host_e2e.py'], 'r30_measured_worker')
    env = build_environment(base)
    env_contract = {key: env[key] for key in env if key.startswith(
        ('PYTHON', 'LD_', 'TORCH', 'ASCEND', 'QWEN', 'TASK_QUEUE', 'PER_STREAM', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_'))}
    context = dict(controller_sha256=sha(__file__), environment=env_contract,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), topology=topology,
        profile=PROFILE, runtime_identity=base['runtime_identity'], identity=IDENTITY,
        source_pins=configs['mixed']['source_pins'], artifact_pins=configs['mixed']['artifact_pins'])
    if args.kind == 'perf':
        check_qualification(args.qualification, context, configs, worker)
    args.output.mkdir(parents=True, exist_ok=False)
    for length, config in configs.items():
        dump(args.output / f'config-L{length}.json', config)
    first = support.cpu_ticks()
    time.sleep(.25)
    second = support.cpu_ticks()
    dump(args.output / 'plan.json', dict(kind=args.kind, rows=plan, context=context,
        exclusive_cpu_ownership=False, loadavg=Path('/proc/loadavg').read_text().strip(),
        cpu_sample=[dict(cpu=i, total=second[i][0]-first[i][0], idle=second[i][1]-first[i][1]) for i in CPUS]))
    rows = []
    try:
        for planned in plan:
            label, length = planned['label'], planned['bag_length']
            result_path = args.output / label / 'result.json'
            life_path = args.output / (label + '-lifecycle.json')
            command = ['numactl', '--physcpubind=' + ','.join(map(str, CPUS)), '--membind=1',
                base['python'], '-B', str(worker_path), '--config', str(args.output / f'config-L{length}.json'),
                '--output', str(result_path.parent), '--arm', planned['arm'], '--kind', args.kind,
                '--round', str(planned['round']), '--warmup', str(planned['warmup']), '--samples', str(planned['samples'])]
            dump(args.output / (label + '-command.json'), dict(argv=command))
            start = time.monotonic_ns()
            life = support.owned(command, env, args.output / (label + '-worker.log'), 240)
            life.update(monotonic_start_ns=start, monotonic_end_ns=time.monotonic_ns())
            dump(life_path, life)
            require(life['returncode'] == 0 and not life['timed_out'] and life['group_closed'], label + ' failed')
            result = read(result_path)
            identity = validate_report(result, planned, configs[length], life, worker,
                                       worker_sha256=SOURCE_HASHES['benchmarks/eager_parallel_r27/host_e2e.py'])
            row = dict(planned, **life, passed=True, result=str(result_path), result_sha256=sha(result_path),
                lifecycle=str(life_path), lifecycle_sha256=sha(life_path), identity=identity,
                median_ns=result['timing']['median_ns'], p95_ns=result['timing']['p95_ns'],
                window_wall_ns=result['window_wall_ns'])
            rows.append(row)
            check_serial(rows)
            dump(args.output / 'completed.json', dict(kind=args.kind, rows=rows, context=context))
            print(json.dumps(dict(label=label, passed=True, median_ns=row['median_ns'], p95_ns=row['p95_ns'])), flush=True)
        verify_pins(context['source_pins'], 'final source')
        verify_pins(context['artifact_pins'], 'final artifact')
        paired = {str(length): worker.paired_summary([row for row in rows if row['bag_length'] == length], args.rounds)
                  for length in LENGTHS} if args.kind == 'perf' else None
        dump(args.output / 'complete.json', dict(passed=True, kind=args.kind, rows=rows, context=context,
            timed_forwards=sum(row['samples'] for row in rows), paired=paired))
    except BaseException as error:
        dump(args.output / 'error.json', dict(passed=False, type=type(error).__name__, message=str(error), completed=len(rows)))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
