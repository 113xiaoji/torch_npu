"""Unchanged original DLRM on the stream-aware numerical CPU host simulator.

Compare off/e/ei/ae/aei in fresh processes. This measures simulated E2E including CPU
arithmetic and simulated stream queues, never CANN/NPU device performance.
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
import traceback

ORIGINAL_COUNTS = dict(aclnnEmbeddingBag=26, aclnnCat=2, aclnnBatchMatMul=1,
                       aclnnIndex=1, aclnnAddmm=6, aclnnRelu=5, aclnnSigmoid=1)
ARMS = ('off', 'e', 'ei', 'ae', 'aei')
ARM_POLICIES = {
    'off': dict(mode='shadow', min_embedding_bag_length=0),
    'e': dict(mode='e', min_embedding_bag_length=0),
    'ei': dict(mode='ei', min_embedding_bag_length=0),
    'ae': dict(mode='e', min_embedding_bag_length=64),
    'aei': dict(mode='ei', min_embedding_bag_length=64),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count(row, name):
    value = row.get(name)
    require(type(value) is int and value >= 0, 'invalid counter ' + name)
    return value


def check_runtime(row):
    require(row.get('backend_kind') == 'HOST_SIM_R27_CPU' and row.get('numeric_supported') is True
            and row.get('real_sdk_qualified') is False, 'numerical stream runtime required')
    require(count(row, 'submitted') == count(row, 'completed') and count(row, 'errors') == 0,
            'simulated stream work did not complete cleanly')
    require(count(row, 'injected_faults') == 0 and row.get('fault_active') is False,
            'fault injection cannot enter a performance/correctness cohort')


def sdk_delta(before, after, forwards):
    for row in (before, after):
        require(count(row, 'live_objects') == 0, 'live SDK descriptor/executor')
        require(count(row, 'descriptors_created') == count(row, 'descriptors_destroyed'),
                'descriptor lifetime imbalance')
    result = {}
    for name in set(before['operations']) | set(after['operations']):
        old = before['operations'].get(name, dict(prepared=0, executed=0, failed=0, cancelled=0))
        new = after['operations'][name]
        delta = {key: count(new, key) - count(old, key) for key in ('prepared', 'executed', 'failed', 'cancelled')}
        require(all(value >= 0 for value in delta.values()), 'counter regressed: ' + name)
        require(delta['prepared'] == delta['executed'] == ORIGINAL_COUNTS.get(name, 0) * forwards
                and delta['failed'] == delta['cancelled'] == 0, 'original execution count differs: ' + name)
        result[name] = delta
    require(all(name in result for name in ORIGINAL_COUNTS), 'original operator missing')
    return result


def validate_routing(routing):
    require(type(routing) is dict and set(routing) == {
        'large_embeddings', 'small_embeddings', 'min_embedding_bag_length'}, 'invalid routing profile')
    require(type(routing['large_embeddings']) is int and 0 <= routing['large_embeddings'] <= 26
            and type(routing['small_embeddings']) is int
            and routing['small_embeddings'] == 26 - routing['large_embeddings']
            and type(routing['min_embedding_bag_length']) is int
            and routing['min_embedding_bag_length'] == 64, 'invalid metadata routing profile')
    return dict(routing)


def arm_routing(arm, routing):
    require(arm in ARMS, 'unknown arm')
    if arm in ('ae', 'aei'):
        routing = validate_routing(routing)
        return routing['large_embeddings'], routing['small_embeddings']
    return (0, 26) if arm == 'off' else (26, 0)


def native_delta(before, after, arm, forwards, *, routing=None):
    large, small = arm_routing(arm, routing)
    require(type(forwards) is int and forwards > 0, 'positive integer forwards required')
    if arm == 'off':
        require(all(row.get('backend') == 'shadow' and row.get('enabled') is False
                    and count(row, 'entered') == 0 for row in (before, after)),
                'off unexpectedly entered native scheduling')
        return dict(passed=True, disabled=True)
    policy = ARM_POLICIES[arm]
    for row in (before, after):
        require(row.get('backend') == 'native_immediate' and row.get('mode') == policy['mode']
                and row.get('actual_stream_scheduling') is True,
                'actual native scheduling required')
        require(count(row, 'min_embedding_bag_length') == policy['min_embedding_bag_length']
                and row.get('embedding_length_policy') == 'metadata_mean_indices_per_bag',
                'native metadata length policy differs')
        require(row.get('real_sdk_qualified') is False and row.get('device_parallel_qualified') is False
                and count(row, 'host_workers') == count(row, 'task_queue_required') == 0,
                'native execution flags differ')
        require(row.get('first_error') is None and row.get('quarantined') is False,
                'native domain failed or quarantined')
        require(all(count(row, key) == 0 for key in ('active_calls', 'retained_owners', 'retained_bytes', 'failed',
                                                    'small_dependency_drains', 'policy_uncertain_fallback')),
                'native work/owners did not retire')
        require(type(row.get('caller_frontier_dirty')) is bool, 'invalid caller frontier dirty flag')
        if large:
            require(row['caller_frontier_dirty'] is False, 'caller frontier remained dirty between forwards')
        if not policy['min_embedding_bag_length'] or not large:
            require(count(row, 'caller_frontier_records') == count(row, 'caller_frontier_waits') == 0,
                    'unexpected cumulative caller frontier work')
        require(len(row.get('stream_ids', [])) == len(set(row.get('stream_ids', []))) == 4,
                'distinct caller/producer/consumer streams required')
    interaction = arm == 'ei' or (arm == 'aei' and large > 0)
    expected = dict(scheduled_embedding=large * forwards, scheduled_cat=2 * forwards if interaction else 0,
                    scheduled_bmm=forwards if interaction else 0, scheduled_index=forwards if interaction else 0,
                    caller_embedding_small=small * forwards,
                    caller_interaction=4 * forwards if arm == 'aei' and large == 0 else 0,
                    small_dependency_drains=0, policy_uncertain_fallback=0)
    for key, value in expected.items():
        require(count(after, key) - count(before, key) == value, 'native scheduled count differs: ' + key)
    records = count(after, 'event_records') - count(before, 'event_records')
    waits = count(after, 'event_waits') - count(before, 'event_waits')
    scheduled = sum(value for key, value in expected.items() if key.startswith('scheduled_'))
    require(records >= scheduled and (waits > 0 if large else records == waits == 0),
            'actual event dependency work missing')
    for key in ('caller_frontier_records', 'caller_frontier_waits'):
        delta = count(after, key) - count(before, key)
        require(delta > 0 if policy['min_embedding_bag_length'] and large else delta == 0,
                'caller frontier work differs: ' + key)
        require(delta <= (records if key.endswith('records') else waits), 'caller frontier exceeds event work')
    return dict(passed=True, **expected)


def sdk_stream_gate(row, arm, before=None, forwards=None, *, routing=None):
    large, small = arm_routing(arm, routing)
    require(forwards is None or (type(forwards) is int and forwards > 0), 'positive integer forwards required')
    operations = row['operations']
    if before is not None:
        delta = {}
        for name in ORIGINAL_COUNTS:
            new = operations[name]
            old = before['operations'].get(name, dict(executed=0, stream_executions={}))
            streams = {}
            for handle in set(new['stream_executions']) | set(old['stream_executions']):
                new_value = new['stream_executions'].get(handle, 0)
                old_value = old['stream_executions'].get(handle, 0)
                require(type(new_value) is int and type(old_value) is int
                        and new_value >= old_value >= 0, 'stream counter regressed')
                value = new_value - old_value
                if value:
                    streams[handle] = value
            delta[name] = dict(executed=count(new, 'executed') - count(old, 'executed'), stream_executions=streams)
        operations = delta
    if forwards is None:
        executed = count(operations['aclnnEmbeddingBag'], 'executed')
        require(executed > 0 and executed % 26 == 0, 'original Embedding execution count differs')
        forwards = executed // 26
    observed = {}
    for name, expected in ORIGINAL_COUNTS.items():
        counts = operations[name]['stream_executions']
        require(all(type(value) is int and value > 0 for value in counts.values()), 'invalid stream count')
        require(sum(counts.values()) == count(operations[name], 'executed') == expected * forwards,
                'original stream execution count differs: ' + name)
        observed[name] = set(counts)
    caller = observed['aclnnAddmm']
    require(len(caller) == 1 and observed['aclnnRelu'] == observed['aclnnSigmoid'] == caller,
            'MLP must stay on the caller stream')
    embedding = observed['aclnnEmbeddingBag']
    interaction = observed['aclnnCat']
    require(len(interaction) == 1 and observed['aclnnBatchMatMul'] == observed['aclnnIndex'] == interaction,
            'Interaction kernels did not execute on one consumer stream')
    if arm == 'off':
        require(embedding == interaction == caller, 'off moved work to a side stream')
    else:
        streams = operations['aclnnEmbeddingBag']['stream_executions']
        caller_handle = next(iter(caller))
        require(streams.get(caller_handle, 0) == small * forwards,
                'small Embedding did not stay on the caller stream')
        side = embedding - caller
        expected_counts = sorted(value * forwards for value in ((large+1)//2, large//2) if value)
        require(len(side) == min(large, 2)
                and sorted(streams[handle] for handle in side) == expected_counts,
                'Embedding producer stream distribution differs from metadata routing')
        if arm == 'ei' or (arm == 'aei' and large > 0):
            require(interaction.isdisjoint(embedding | caller), 'EI did not use its separate consumer stream')
        else:
            require(interaction == caller, 'Interaction moved off the caller stream')
    return dict(passed=True, streams={name: sorted(values) for name, values in observed.items()})


def paired_summary(rows, rounds):
    require(type(rounds) is int and rounds > 0 and bool(rows), 'positive rounds required')
    samples = rows[0].get('samples')
    require(type(samples) is int and samples > 0, 'positive samples required')
    for row in rows:
        require(row.get('passed') is True and row.get('kind') == 'perf'
                and row.get('samples') == samples and type(row.get('samples')) is int
                and type(row.get('round')) is int
                and type(row.get('window_wall_ns')) is int and row['window_wall_ns'] >= 2_000_000_000,
                'only successful equal-sample performance windows are comparable')
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] > 0
                    for key in ('median_ns', 'p95_ns')), 'invalid timing statistic')
    index = {(row['round'], row['arm']): row for row in rows}
    require(len(index) == len(rows) == rounds * len(ARMS) and
            all((i, arm) in index for i in range(rounds) for arm in ARMS),
            'one off/e/ei/ae/aei result per round required')
    result = {}
    for arm in ARMS[1:]:
        speedups = [index[i, 'off']['median_ns'] / index[i, arm]['median_ns'] for i in range(rounds)]
        p95 = [index[i, 'off']['p95_ns'] / index[i, arm]['p95_ns'] for i in range(rounds)]
        result[arm] = dict(paired_speedup_median=statistics.median(speedups), paired_speedups=speedups,
            paired_p95_speedup_median=statistics.median(p95), faster_rounds=sum(x > 1 for x in speedups),
            paired_latency_change_pct=statistics.median((1 / value - 1) * 100 for value in speedups),
            paired_p95_latency_change_pct=statistics.median((1 / value - 1) * 100 for value in p95))
    return result


def load_workload(reference, upstream, config):
    require(type(config.get('validate_stages', False)) is bool, 'validate_stages must be boolean')
    if config.get('validate_stages', False):
        require(all(callable(getattr(reference, name, None)) for name in ('capture_stages', 'validate_stages')),
                'optional stage callbacks are required')
    kwargs = dict(bag_length=config['bag_length']) if 'bag_length' in config else {}
    return reference.load_original_model(upstream, **kwargs)


def run(config, report):
    require(report.get('arm') in ARMS, 'unknown arm')
    policy = dict(ARM_POLICIES[report['arm']])
    report.update(policy=policy, routing_profile=validate_routing(config.get('routing_profile')))
    require(sys.platform == 'linux' and sys.version_info[:3] == (3, 14, 6), 'pinned Linux FT Python required')
    require(os.environ.get('PYTHON_GIL') is None and not sys._is_gil_enabled(), 'natural free threading required')
    for key, value in dict(TASK_QUEUE_ENABLE='0', QWEN_ACL_HOST_SIM='1', PER_STREAM_QUEUE='0',
                          ASCEND_LAUNCH_BLOCKING='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0').items():
        require(os.environ.get(key) == value, 'environment differs: ' + key)
    require(sys.gettrace() is None and sys.getprofile() is None, 'profiling must be off')
    mandatory_sources = [str(Path(__file__).resolve()), config['measurement_helper'], config['reference'],
                         config['mapping_helper'], config['frozen_model_helper']]
    require(all(path in config['source_pins'] for path in mandatory_sources), 'required helper source pin missing')
    for path, digest in config['source_pins'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'source pin mismatch: ' + path)
    common = load(config['measurement_helper'], 'r27_measurement')
    common.verify_pins(config['artifact_pins'], 'artifact')
    common.verify_pins(config['source_pins'], 'source')
    require(str(Path(__file__).resolve()) in config['source_pins'], 'worker must be source pinned')
    report['topology'] = common.cpu_identity(config['cpus'])
    package = Path(config['package']).resolve(strict=True)
    require(os.environ.get('PYTHONPATH') == str(package.parent), 'candidate PYTHONPATH differs')
    sys.path.insert(0, str(package.parent))
    sys.path.append(config['legacy_deps'])
    import torch
    import numpy
    import torch_npu
    from torch_npu import _eager_parallel
    require(Path(torch_npu.__file__).resolve().parent == package
            and Path(torch_npu._C.__file__).resolve().parent == package, 'wrong native candidate')
    mandatory_artifacts = [str(Path(torch_npu._C.__file__).resolve()), str(package / '__init__.py'),
                           str(package / '_eager_parallel.py')]
    mandatory_artifacts.extend(row['path'] for row in config['libraries'].values())
    require(all(path in config['artifact_pins'] for path in mandatory_artifacts), 'required artifact pin missing')
    require(all(config['artifact_pins'][row['path']] == row['sha256'] for row in config['libraries'].values()),
            'library/artifact pin identities disagree')
    require(torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
            and numpy.__version__ == '2.4.4' and not sys._is_gil_enabled(), 'runtime identity differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = load(config['reference'], 'r27_reference')
    model, inputs, provenance = load_workload(reference, config['original_source'], config)
    report['provenance'] = provenance
    with torch.inference_mode():
        expected = model(*inputs).clone()
        expected_stages = reference.capture_stages(model, inputs) if config.get('validate_stages', False) else None
        model = model.to('npu:0')
        inputs = (inputs[0].to('npu:0'), [v.to('npu:0') for v in inputs[1]],
                  [v.to('npu:0') for v in inputs[2]])
        torch.npu.synchronize()
    runtime = ctypes.CDLL(config['libraries']['libr27_hostsim_runtime.so']['path'])
    opapi = ctypes.CDLL(config['libraries']['libopapi.so']['path'])
    runtime.R27HostSimDiagnosticsJson.restype = ctypes.c_char_p
    opapi.R27HostSimOpDiagnosticsJson.restype = ctypes.c_char_p
    def snapshot():
        return dict(native=_eager_parallel.snapshot(), runtime=json.loads(runtime.R27HostSimDiagnosticsJson()),
                    opapi=json.loads(opapi.R27HostSimOpDiagnosticsJson()))
    report['_failure_snapshot_callback'] = snapshot
    mappings = load(config['mapping_helper'], 'r27_mapping')
    report['mapping_before'] = mappings.mapped(config['libraries'])
    with torch.inference_mode():
        if report['arm'] == 'off':
            _eager_parallel.configure(enabled=False, mode='shadow', min_embedding_bag_length=0)
            context = nullcontext()
        else:
            context = _eager_parallel.scope(mode=policy['mode'],
                min_embedding_bag_length=policy['min_embedding_bag_length'])
        with context:
            # Cold measures the first NPU forward; perf validates before warmup.
            report['correctness'] = []
            for _ in range(0 if report['kind'] == 'cold' else 8):
                actual = model(*inputs)
                torch.npu.synchronize()
                error = reference.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
                report['correctness'].append(error)
                require(error['passed'], 'scheduled original forward differs from CPU reference')
            report['before_warmup'] = snapshot()
            measurement, actual = common.measure_forward_window(model, inputs, torch.npu.synchronize,
                warmup=report['warmup'], samples=report['samples'], snapshot=snapshot)
            report.update(measurement)
            report['timing'] = common.distribution(report['durations_ns'])
            report['numeric_final'] = reference.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
            require(report['numeric_final']['passed'], 'final timed output differs from CPU reference')
            if report['kind'] == 'cold':
                report['correctness'].append(report['numeric_final'])
            if expected_stages is not None:
                report['stage_validation'] = reference.validate_stages(model, inputs, expected_stages,
                                                                        torch.npu.synchronize)
                require(report['stage_validation']['passed'], 'original intermediate stages differ from CPU reference')
        report['closed'] = snapshot()
    for name in ('before', 'after', 'closed'):
        check_runtime(report[name]['runtime'])
        require(count(report[name]['opapi'], 'active_kernels') == 0, 'numeric kernel still active')
    report['sdk_timed_gate'] = sdk_delta(report['before']['opapi'], report['after']['opapi'], report['samples'])
    report['native_timed_gate'] = native_delta(report['before']['native'], report['after']['native'],
                                              report['arm'], report['samples'], routing=config['routing_profile'])
    report['sdk_stream_gate'] = sdk_stream_gate(report['after']['opapi'], report['arm'],
                                               report['before']['opapi'], report['samples'], routing=config['routing_profile'])
    require(report['closed']['native']['closure']['passed'], 'native domain did not close')
    if report['arm'] != 'off':
        require(report['closed']['native']['caller_frontier_dirty'] is False,
                'caller frontier did not close')
    report['mapping_after'] = mappings.mapped(config['libraries'])
    require(report['mapping_after'] == report['mapping_before'], 'loaded artifacts changed')
    report['thread_affinities'] = common.thread_affinities(config['cpus'])
    require(not sys._is_gil_enabled(), 'GIL enabled during execution')
    common.verify_pins(config['artifact_pins'], 'final artifact')
    common.verify_pins(config['source_pins'], 'final source')
    if report['kind'] == 'perf':
        require(report['window_wall_ns'] >= 2_000_000_000, 'window below two seconds')
    report.update(passed=True, gil_enabled=False, original_model_executed=True,
                  host_sim_numeric_validated=True, real_npu_performance_claim=False)
    report.pop('_failure_snapshot_callback')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--kind', required=True, choices=('cold', 'perf'))
    parser.add_argument('--round', required=True, type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--samples', type=int)
    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = 0 if args.kind == 'cold' else 50
    if args.samples is None:
        args.samples = 1 if args.kind == 'cold' else 3000
    require(args.round >= 0 and args.warmup >= 0 and args.samples > 0, 'invalid measurement sizes')
    require(args.kind != 'cold' or (args.warmup == 0 and args.samples == 1),
            'cold requires no warmup and exactly one first forward')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, passed=False, arm=args.arm, kind=args.kind, round=args.round,
                  warmup=args.warmup, samples=args.samples, pid=os.getpid(), worker_sha256=sha(__file__),
                  compute_backend='host_sim_numerical_cpu', startup_included=False,
                  numerical_checks_in_timing=False, original_model_executed=False,
                  host_sim_numeric_validated=False, real_npu_performance_claim=False)
    try:
        config = json.loads(args.config.read_text())
        report.update(artifact_pins=config['artifact_pins'], source_pins=config['source_pins'])
        run(config, report)
    except BaseException as error:
        report['passed'] = False
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        snapshot = report.pop('_failure_snapshot_callback', None)
        if snapshot:
            try:
                report['failure_snapshot'] = snapshot()
            except BaseException as diagnostic_error:
                report['failure_snapshot_error'] = type(diagnostic_error).__name__
        try:
            (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        except BaseException:
            pass  # A failed diagnostic write must never replace the original failure.
        raise
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, arm=args.arm, **report['timing'])), flush=True)


if __name__ == '__main__':
    main()
