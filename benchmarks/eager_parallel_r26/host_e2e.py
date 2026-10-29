"""Original DLRM host-sim latency, off versus original-order native shadow.

NoOp values are never read. This worker measures host latency only; the R26
shadow adapter does not implement parallel E/EI execution or device speedup.
"""
import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import sysconfig
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def integer(row, key):
    value = row.get(key)
    require(type(value) is int and value >= 0, 'missing/inexact counter: ' + key)
    return value


def process_usage():
    import resource
    row = resource.getrusage(resource.RUSAGE_SELF)
    return dict(voluntary=row.ru_nvcsw, involuntary=row.ru_nivcsw, max_rss_kib=row.ru_maxrss)


def measure_forward_window(model, inputs, synchronize, *, warmup, samples, snapshot,
                           wall_clock=time.perf_counter_ns, cpu_clock=time.process_time_ns,
                           usage=process_usage):
    """Measure original forward+sync; callers own inference mode and scope.

    Snapshot, warmup, rusage, and setup/close work cannot enter sample durations.
    The wider wall/CPU window includes timing and list-append loop overhead.
    """
    require(type(warmup) is int and warmup >= 0, 'nonnegative exact warmup required')
    require(type(samples) is int and samples > 0, 'positive exact samples required')
    for _ in range(warmup):
        result = model(*inputs)
        synchronize()
    before = snapshot()
    usage_before = usage()
    durations = []
    cpu_start = cpu_clock()
    start = wall_clock()
    for _ in range(samples):
        tick = wall_clock()
        result = model(*inputs)
        synchronize()
        durations.append(wall_clock() - tick)
    wall = wall_clock() - start
    cpu = cpu_clock() - cpu_start
    usage_after = usage()
    after = snapshot()
    delta = {key: usage_after[key] - usage_before[key] for key in ('voluntary', 'involuntary')}
    require(all(value >= 0 for value in delta.values()), 'process usage counters regressed')
    return dict(durations_ns=durations, window_wall_ns=wall, window_process_cpu_ns=cpu,
        usage_before=usage_before, usage_after=usage_after, usage_delta=delta,
        max_rss_kib=usage_after['max_rss_kib'], before=before, after=after,
        forward_calls=warmup + samples, synchronize_calls=warmup + samples), result


def output_metadata(output):
    metadata = dict(shape=list(output.shape), dtype=str(output.dtype), device=str(output.device),
                    numeric_validated=False)
    require(metadata == dict(shape=[128, 1], dtype='torch.float32', device='npu:0',
                             numeric_validated=False), 'original output metadata differs')
    return metadata


def validate_sdk(snapshot):
    require(snapshot.get('backend_kind') == 'HOST_SIM_R25'
        and snapshot.get('numeric_supported') is False
        and snapshot.get('real_sdk_qualified') is False
        and snapshot.get('artificial_prepare_delay_ns') == 0,
        'NoOp provider identity or artificial delay differs')
    require(snapshot.get('tracing_enabled') is False, 'SDK tracing must remain disabled')
    for key in ('errors', 'cache_stub_calls', 'live_descriptors', 'live_executors',
                'huge_pending', 'huge_active_threads'):
        require(integer(snapshot, key) == 0, 'SDK resource/error not zero: ' + key)
    require(integer(snapshot, 'descriptors_created') == integer(snapshot, 'descriptors_destroyed'),
            'SDK descriptors not balanced')
    require(integer(snapshot, 'executors_created') == integer(snapshot, 'executors_executed')
            + integer(snapshot, 'executors_cancelled'), 'SDK executors not balanced')
    require(integer(snapshot, 'huge_init') == integer(snapshot, 'huge_uninit'),
            'SDK huge thread scopes not balanced')
    return True


def validate_sdk_delta(before, after, forwards):
    require(type(forwards) is int and forwards > 0, 'positive exact forward count required')
    validate_sdk(before)
    validate_sdk(after)
    old, new = before['operations']['aclnnEmbeddingBag'], after['operations']['aclnnEmbeddingBag']
    delta = {key: integer(new, key) - integer(old, key) for key in ('prepared', 'executed', 'cancelled')}
    require(delta['prepared'] == delta['executed'] == 26 * forwards and delta['cancelled'] == 0,
            'SDK must prepare/execute exactly 26 original Embedding calls per forward')
    return dict(passed=True, forwards=forwards, **delta)


def validate_native_delta(before, after, arm, forwards):
    require(arm in ('off', 'shadow'), 'unsupported native arm')
    require(type(forwards) is int and forwards > 0, 'positive exact forward count required')
    for snapshot in (before, after):
        require(snapshot.get('schema') == 1 and snapshot.get('backend') == 'shadow'
            and snapshot.get('enabled') is (arm == 'shadow')
            and snapshot.get('coverage_incomplete') is False
            and snapshot.get('first_error') is None
            and snapshot.get('device_parallel_qualified') is False
            and snapshot.get('real_sdk_qualified') is False, 'native snapshot identity differs')
        for key in ('active_calls', 'foreign_thread_calls', 'failed', 'host_workers'):
            require(integer(snapshot, key) == 0, 'native work/error not zero: ' + key)
    counters = {}
    for name in set(before['operator_counts']) | set(after['operator_counts']):
        old = before['operator_counts'].get(name, dict.fromkeys(('entered', 'completed', 'failed', 'inputs', 'outputs'), 0))
        new = after['operator_counts'].get(name, {})
        counters[name] = {key: integer(new, key) - integer(old, key)
                          for key in ('entered', 'completed', 'failed', 'inputs', 'outputs')}
        row = counters[name]
        require(all(value >= 0 for value in row.values()) and row['entered'] == row['completed']
                and row['failed'] == 0, 'native counters regressed or did not complete: ' + name)
    if arm == 'off':
        require(all(integer(snapshot, key) == 0 for snapshot in (before, after)
                    for key in ('entered', 'completed', 'nested_entries'))
                and all(integer(row, key) == 0 for snapshot in (before, after)
                        for row in snapshot['operator_counts'].values()
                        for key in ('entered', 'completed', 'failed', 'inputs', 'outputs')),
                'disabled native gate activated')
        return dict(passed=True, disabled=True, forwards=forwards, embedding_calls=0, operators=counters)
    embedding = [row for name, row in counters.items()
                 if name in ('aten::_embedding_bag', 'aten::_embedding_bag_forward_only')]
    require(sum(row['completed'] for row in embedding) == 26 * forwards
            and sum(row['outputs'] for row in embedding) == 104 * forwards,
            'timed native Embedding count or four-output contract differs')
    for name, count in (('aten::cat', 2), ('aten::bmm', 1), ('aten::index.Tensor', 1)):
        require(counters.get(name, {}).get('completed') == count * forwards,
                'timed native pipeline count differs: ' + name)
    require(counters['aten::cat']['inputs'] == 29 * forwards, 'timed Cat TensorList count differs')
    require(integer(after, 'entered') - integer(before, 'entered') ==
            integer(after, 'completed') - integer(before, 'completed') ==
            sum(row['completed'] for row in counters.values()), 'native aggregate count differs')
    return dict(passed=True, forwards=forwards, embedding_calls=26 * forwards, operators=counters)


def distribution(values):
    require(type(values) is list and values and all(type(value) is int and value > 0 for value in values),
            'positive exact host duration samples required')
    ordered = sorted(values)
    return dict(samples=len(values), median_ns=statistics.median(values),
                p95_ns=ordered[math.ceil(.95 * len(ordered)) - 1])


def summarize_pairs(rows, *, rounds=10, samples=3000):
    """Describe matched process medians; never pool samples or prove a speedup."""
    require(type(rounds) is int and rounds > 0 and type(samples) is int and samples > 0,
            'positive exact summary dimensions required')
    index = {(row['round'], row['arm']): row for row in rows}
    require(len(index) == len(rows) == 2 * rounds and
            set(index) == {(number, arm) for number in range(rounds) for arm in ('off', 'shadow')},
            'incomplete/duplicate matched off/shadow rounds')
    require(all(row.get('passed') is True and row.get('kind') == 'perf'
                and row.get('host_latency_measured') is True and row.get('samples') == samples
                and len(row['durations_ns']) == samples and row.get('window_wall_ns', 0) >= 2_000_000_000
                for row in rows), 'failed, cold, short, or incomplete performance row')
    require(len({row['warmup'] for row in rows}) == 1, 'warmup differs across paired cohort')
    pairs = []
    for number in range(rounds):
        off = distribution(index[number, 'off']['durations_ns'])
        shadow = distribution(index[number, 'shadow']['durations_ns'])
        pairs.append(dict(round=number, off=off, shadow=shadow,
            median_ratio_off_over_shadow=off['median_ns'] / shadow['median_ns'],
            p95_ratio_off_over_shadow=off['p95_ns'] / shadow['p95_ns']))
    return dict(pairs=pairs, rounds=rounds, samples_per_process=samples,
        median_paired_ratio=statistics.median(row['median_ratio_off_over_shadow'] for row in pairs),
        median_paired_p95_ratio=statistics.median(row['p95_ratio_off_over_shadow'] for row in pairs),
        aggregation='paired process median/p95 ratios; samples never pooled',
        speedup_claim_proven=False, performance_claim=False, device_overlap_claim=False,
        real_device_numerical_claim=False, host_latency_measured=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pins(pins, label):
    require(type(pins) is dict and pins, label + ' pins required')
    for path, digest in pins.items():
        require(Path(path).is_absolute() and sha(path) == digest, label + ' pin mismatch: ' + path)


def cpu_identity(cpus):
    require(len(cpus) == len(set(cpus)) == 8 and all(type(cpu) is int and cpu >= 0 for cpu in cpus),
            'eight distinct CPUs required')
    require(set(os.sched_getaffinity(0)) == set(cpus), 'externally bound CPU budget differs')
    rows = []
    for cpu in cpus:
        root = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu))
        require((root / 'node1').exists(), 'all CPUs must belong to NUMA node 1')
        rows.append(dict(cpu=cpu,
            package=int((root / 'topology/physical_package_id').read_text()),
            core=int((root / 'topology/core_id').read_text())))
    require(len({(row['package'], row['core']) for row in rows}) == 8, 'eight physical cores required')
    return rows


def thread_affinities(cpus):
    rows = []
    for task in sorted(Path('/proc/self/task').iterdir()):
        try:
            mask = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        require(set(mask) <= set(cpus), 'thread escaped eight-core affinity')
        rows.append(dict(tid=int(task.name), cpus=mask))
    return rows


def run(config, output, report):
    startup = time.perf_counter_ns()
    require(sys.platform == 'linux' and sys.version_info[:3] == (3, 14, 6),
            'target Linux CPython 3.14.6 required')
    require(os.environ.get('PYTHON_GIL') is None and sys._is_gil_enabled() is False,
            'natural free-threaded runtime required')
    require(config.get('schema') == 1 and config['compute_backend'] == 'host_sim_noop',
            'schema 1 NoOp host simulation required')
    expected_env = dict(QWEN_ACL_HOST_SIM='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TASK_QUEUE_ENABLE='1', PER_STREAM_QUEUE='0', ASCEND_LAUNCH_BLOCKING='0')
    for key, value in expected_env.items():
        require(os.environ.get(key) == value, 'runtime environment differs: ' + key)
    require(sys.gettrace() is None and sys.getprofile() is None, 'Python profiling/trace active')
    report['cpu_topology'] = cpu_identity(config['cpus'])
    verify_pins(config['artifact_pins'], 'artifact')
    verify_pins(config['source_pins'], 'source')
    base = Path(__file__).resolve().parent
    required_sources = [Path(__file__).resolve(), base / 'reference.py', base / 'shadow_hostsim.py',
                        base.parent / 'host_regions/real_dlrm_runner_r22.py']
    require(all(str(path) in config['source_pins'] for path in required_sources),
            'worker/reference/shadow validator/frozen helper must all be source-pinned')
    shadow = load_module(base / 'shadow_hostsim.py', 'r26_e2e_pinned_shadow')
    reference = load_module(base / 'reference.py', 'r26_e2e_pinned_reference')
    package = Path(config['torch_npu_package']).resolve(strict=True)
    require(os.environ.get('PYTHONPATH') == str(package.parent), 'candidate-only PYTHONPATH required')
    sys.path.insert(0, str(package.parent))
    sys.path.append(config['legacy_deps'])
    import torch
    import numpy
    import torch_npu
    from torch_npu import _eager_parallel
    require(Path(torch_npu.__file__).resolve().parent == package
        and Path(torch_npu._C.__file__).resolve().parent == package, 'import escaped pinned candidate')
    extension = str(Path(torch_npu._C.__file__).resolve())
    require(extension in config['artifact_pins'], 'actual native extension must be artifact-pinned')
    require(torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
        and numpy.__version__ == '2.4.4' and sys._is_gil_enabled() is False,
        'target Torch/NumPy/free-threaded identity changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Torch thread budget differs')
    model, inputs, provenance = reference.load_original_model(config['original_source'])
    report['model_provenance'] = provenance
    torch.npu.set_device(0)
    model = model.to('npu:0')
    inputs = inputs[0].to('npu:0'), [value.to('npu:0') for value in inputs[1]], [value.to('npu:0') for value in inputs[2]]
    torch.npu.synchronize()
    library = ctypes.CDLL(config['libraries']['libopapi.so']['path'])
    library.R25HostSimBackendKind.argtypes = []
    library.R25HostSimBackendKind.restype = ctypes.c_char_p
    library.R25HostSimDiagnosticsJson.argtypes = []
    library.R25HostSimDiagnosticsJson.restype = ctypes.c_char_p
    library.R25HostSimSetTracing.argtypes = [ctypes.c_bool]
    library.R25HostSimSetTracing.restype = ctypes.c_int
    require(library.R25HostSimBackendKind() == b'HOST_SIM_R25', 'unexpected NoOp simulation provider')
    require(library.R25HostSimSetTracing(False) == 0, 'SDK trace disable failed')
    def sdk():
        return json.loads(library.R25HostSimDiagnosticsJson().decode())
    def snapshot():
        return dict(native=_eager_parallel.snapshot(), sdk=sdk())
    report['mapping_before'] = shadow.mapped(config['libraries'])
    report['ready'] = dict(python=sys.version, executable=sys.executable,
        abi=sysconfig.get_config_var('SOABI'), gil_enabled=sys._is_gil_enabled(),
        device='npu:0', torch_git=torch.version.git_version, numpy_version=numpy.__version__,
        torch_npu_package=str(package), native_extension=extension, native_extension_sha256=sha(extension),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        thread_affinities=thread_affinities(config['cpus']))
    with torch.inference_mode():
        if report['arm'] == 'shadow':
            context = _eager_parallel.scope(mode='shadow')
        else:
            _eager_parallel.configure(enabled=False, mode='shadow')
            context = nullcontext()
        with context:
            report['native_before_warmup'] = _eager_parallel.snapshot()
            report['sdk_before_warmup'] = sdk()
            validate_sdk(report['sdk_before_warmup'])
            report['setup_startup_ns'] = time.perf_counter_ns() - startup
            measurement, result = measure_forward_window(model, inputs, torch.npu.synchronize,
                warmup=report['warmup'], samples=report['samples'], snapshot=snapshot)
            before, after = measurement.pop('before'), measurement.pop('after')
            report.update(measurement, native_before=before['native'], native_after=after['native'],
                          sdk_before=before['sdk'], sdk_after=after['sdk'], original_model_executed=True)
        report['native_closed'] = _eager_parallel.snapshot()
        report['sdk_closed'] = sdk()
    report['metadata'] = output_metadata(result)
    report['timing'] = distribution(report['durations_ns'])
    report['native_timed_gate'] = validate_native_delta(report['native_before'], report['native_after'],
                                                       report['arm'], report['samples'])
    report['sdk_timed_gate'] = validate_sdk_delta(report['sdk_before'], report['sdk_after'], report['samples'])
    total = report['warmup'] + report['samples']
    report['sdk_total_gate'] = validate_sdk_delta(report['sdk_before_warmup'], report['sdk_closed'], total)
    if report['arm'] == 'shadow':
        report['native_total_gate'] = shadow.validate_shadow(report['native_closed'], total)
    else:
        report['native_total_gate'] = validate_native_delta(report['native_before_warmup'],
                                                            report['native_closed'], 'off', total)
    validate_sdk(report['sdk_closed'])
    require(report['native_closed']['closure'].get('passed') is True
        and report['native_closed']['closure'].get('admission_closed') is True,
        'native scope did not close')
    report['mapping'] = shadow.mapped(config['libraries'])
    report['thread_affinities_closed'] = thread_affinities(config['cpus'])
    report['gil_enabled'] = sys._is_gil_enabled()
    require(report['gil_enabled'] is False, 'GIL changed during model execution')
    verify_pins(config['artifact_pins'], 'final artifact')
    verify_pins(config['source_pins'], 'final source')
    if report['kind'] == 'perf':
        require(report['window_wall_ns'] >= 2_000_000_000,
                'performance window below two seconds; rerun all pairs with a larger fixed sample count')
    report.update(host_latency_measured=report['kind'] == 'perf', passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('off', 'shadow'), required=True)
    parser.add_argument('--kind', choices=('cold', 'perf'), required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--samples', type=int)
    args = parser.parse_args(argv)
    warmup = args.warmup if args.warmup is not None else (0 if args.kind == 'cold' else 50)
    samples = args.samples if args.samples is not None else (1 if args.kind == 'cold' else 3000)
    require(args.round >= 0 and warmup >= 0 and samples > 0, 'nonnegative round/warmup and positive samples required')
    require(args.kind != 'cold' or (warmup == 0 and samples == 1), 'cold requires zero warmup and one sample')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, passed=False, arm=args.arm, kind=args.kind, round=args.round,
        warmup=warmup, samples=samples, pid=os.getpid(), worker_sha256=sha(__file__),
        compute_backend='host_sim_noop', schedule_backend='shadow',
        host_latency_measured=False, diagnostic_only=args.kind == 'cold',
        original_model_executed=False, synthetic_prepare_cpu_or_delay=False,
        numerical_claim=False, real_device_numerical_claim=False, real_sdk_qualified=False,
        device_parallel_qualified=False, device_overlap_claim=False, performance_claim=False,
        speedup_claim_proven=False, full_entry_coverage_qualified=False,
        startup_included_in_samples=False, warmup_included_in_samples=False,
        snapshot_included_in_samples=False, scope_close_included_in_samples=False,
        boundary='original model(*inputs) through torch.npu.synchronize return; host latency only',
        order_scope='original kernels and stream order; no E/EI parallel adapter')
    try:
        config = json.loads(args.config.read_text())
        report.update(artifact_pins=config.get('artifact_pins'), source_pins=config.get('source_pins'),
                      native_header_sha256=config.get('native_header_sha256'))
        dump(args.output / 'config.json', config)
        run(config, args.output, report)
    except BaseException as error:
        report['passed'] = False
        report['host_latency_measured'] = False
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        try:
            dump(args.output / 'result.json', report)
        except BaseException as receipt_error:
            try:
                error.add_note('failure receipt also failed: ' + repr(receipt_error))
            except BaseException:
                pass
        raise
    dump(args.output / 'result.json', report)
    print(json.dumps(dict(passed=True, arm=args.arm, kind=args.kind, round=args.round,
                         samples=samples, host_latency_measured=report['host_latency_measured'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
