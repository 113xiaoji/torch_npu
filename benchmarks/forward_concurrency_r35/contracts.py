"""Pure stdlib R35 receipt contracts, with no framework or device imports.

Request times are perf_counter_ns snapshots: arrival_ns is immediately before
the shared model call; forward_return_ns is its host return; response_ns is
after this request's completion token wait. Thread CPU snapshots use
thread_time_ns at the same boundaries. They measure the caller, not native
Prepare workers. Quantiles, including median_ns, use nearest rank.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics

IDENTITY = dict(model_sha256='9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5',
                state_sha256='6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19',
                input_sha256='c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1',
                dense_sha256='352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9')
CASE_KEYS = ('case_id', 'device', 'gil_mode', 'callers', 'stream_mode', 'backend')
TASK_KEYS = CASE_KEYS + ('kind', 'round', 'total_requests', 'warmup_per_caller')
NATIVE_SHA256 = 'd8943e56a5b234b2dc7cb0091861087a611437661b0ddd6479d5ca610a7ccaa4'
_NATIVE_CONTRACTS = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer: ' + name)
    return value


def sha256(value):
    return (type(value) is str and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def case_grid(device='npu', include_prepare=False):
    require(device in ('cpu', 'npu'), 'device must be cpu or npu')
    require(type(include_prepare) is bool and not (device == 'cpu' and include_prepare),
            'CPU does not support Prepare backends')
    streams = ('cpu',) if device == 'cpu' else ('shared', 'private')
    rows = [dict(device=device, gil_mode=gil, callers=callers, stream_mode=stream, backend='off')
            for gil in ('off', 'on') for callers in (1, 2, 4) for stream in streams]
    if include_prepare:
        rows += [dict(device='npu', gil_mode='off', callers=4, stream_mode='shared', backend=backend)
                 for backend in ('mutex4', 'mpsc4')]
    for row in rows:
        row['case_id'] = f"gil{row['gil_mode']}-c{row['callers']}-{row['stream_mode']}-{row['backend']}"
    return rows


def validate_case(row):
    matching = [case for case in case_grid(row.get('device'), row.get('backend') in ('mutex4', 'mpsc4'))
                if all(case[key] == row.get(key) for key in CASE_KEYS)]
    require(len(matching) == 1, 'unsupported case or case_id differs')


def campaign_plan(device='npu', rounds=1, include_prepare=False):
    integer(rounds, 'rounds', 1)
    require(rounds <= 100, 'rounds must be 1..100')
    cases = case_grid(device, include_prepare)
    result = [dict(row, kind='correctness', round=0, total_requests=32, warmup_per_caller=0)
              for row in cases]
    # Alternate direction and rotate to avoid always ending on the same arm.
    # This is deterministic order control, not a claim of formal balance.
    for index in range(rounds):
        order = cases[index % len(cases):] + cases[:index % len(cases)]
        if index % 2:
            order = list(reversed(order))
        result.extend(dict(row, kind='perf', round=index, total_requests=3000, warmup_per_caller=24)
                      for row in order)
    return result


def distribution(values):
    require(type(values) is list and bool(values)
            and all(type(value) is int and value > 0 for value in values),
            'positive integer raw durations required')
    ordered = sorted(values)
    def rank(probability):
        return ordered[math.ceil(probability * len(ordered)) - 1]
    return dict(count=len(values), mean_ns=statistics.mean(values), median_ns=rank(.5),
                p95_ns=rank(.95), p99_ns=rank(.99), min_ns=ordered[0], max_ns=ordered[-1])


def _overlap(intervals):
    # End-before-start at equal timestamps: touching intervals do not overlap.
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    active = maximum = overlap = 0
    previous = events[0][0]
    for instant, delta in events:
        if active > 1:
            overlap += instant - previous
        active += delta
        maximum = max(maximum, active)
        previous = instant
    return dict(overlap_ns=overlap, max_concurrency=maximum,
                interpretation='Host forward call overlap does not prove concurrent Python bytecode or NPU kernels.')


def request_metrics(row):
    """Validate exact request accounting and recompute all request/window metrics."""
    total = integer(row.get('total_requests'), 'total_requests', 1)
    callers = integer(row.get('callers'), 'callers', 1)
    require(callers in (1, 2, 4), 'callers must be 1, 2 or 4')
    start = integer(row.get('start_ns'), 'start_ns')
    end = integer(row.get('end_ns'), 'end_ns')
    require(end > start, 'measurement window must be positive')
    records = row.get('requests')
    workers = row.get('workers')
    require(type(records) is list and len(records) == total, 'request count differs')
    require(type(workers) is list and len(workers) == callers, 'worker count differs')
    require(all(type(record) is dict for record in records), 'invalid request record')
    ids = [integer(record.get('request_id'), 'request_id') for record in records]
    require(sorted(ids) == list(range(total)), 'request IDs must be exactly unique 0..total-1')
    worker_ids = [integer(worker.get('worker_index'), 'worker_index') for worker in workers]
    require(sorted(worker_ids) == list(range(callers)), 'worker IDs must be exactly unique 0..callers-1')
    indexed = {worker['worker_index']: worker for worker in workers}
    for key in ('thread_ident', 'native_thread_id'):
        ids = [integer(worker.get(key), key, 1) for worker in workers]
        require(len(set(ids)) == callers, 'callers must have distinct ' + key)
    forward, response, forward_cpu, response_cpu = [], [], [], []
    per_worker = {index: [] for index in range(callers)}
    intervals = []
    for record in records:
        worker_index = integer(record.get('worker_index'), 'worker_index')
        require(worker_index in indexed, 'request refers to unknown worker')
        worker = indexed[worker_index]
        require(all(record.get(key) == worker[key] for key in ('thread_ident', 'native_thread_id')),
                'request thread identity differs from its worker')
        require(type(record.get('input_id')) is int and record['input_id'] == record['request_id'] % 8,
                'request input bank assignment differs')
        arrival, returned, completed = [integer(record.get(key), key) for key in
                                       ('arrival_ns', 'forward_return_ns', 'response_ns')]
        cpu_start, cpu_return, cpu_end = [integer(record.get(key), key) for key in
            ('thread_cpu_start_ns', 'forward_thread_cpu_end_ns', 'thread_cpu_end_ns')]
        require(start <= arrival < returned <= completed <= end, 'invalid host/completion interval')
        require(cpu_start < cpu_return <= cpu_end, 'invalid caller thread CPU interval')
        forward.append(returned - arrival)
        response.append(completed - arrival)
        forward_cpu.append(cpu_return - cpu_start)
        response_cpu.append(cpu_end - cpu_start)
        intervals.append((arrival, returned))
        per_worker[worker_index].append(record)
    for index, worker_records in per_worker.items():
        worker_records.sort(key=lambda item: item['arrival_ns'])
        worker = indexed[index]
        require(bool(worker_records), 'every caller must serve requests')
        require(integer(worker.get('completed_requests'), 'completed_requests') == len(worker_records),
                'worker completed count differs')
        worker_cpu = integer(worker.get('thread_cpu_ns'), 'worker thread_cpu_ns', 1)
        require(worker_cpu >= sum(record['thread_cpu_end_ns'] - record['thread_cpu_start_ns']
                                  for record in worker_records), 'worker CPU undercounts request CPU')
        for previous, current in zip(worker_records, worker_records[1:]):
            require(previous['response_ns'] <= current['arrival_ns'], 'caller is not closed loop')
            require(previous['thread_cpu_end_ns'] <= current['thread_cpu_start_ns'],
                    'caller thread CPU clock moved backwards')
    return dict(completed_requests=total, window_wall_ns=end - start,
                throughput_requests_per_second=total * 1_000_000_000 / (end - start),
                host_forward=distribution(forward), end_to_end=distribution(response),
                forward_thread_cpu=distribution(forward_cpu), response_thread_cpu=distribution(response_cpu),
                worker_thread_cpu_ns=sum(worker['thread_cpu_ns'] for worker in workers),
                host_forward_overlap=_overlap(intervals))


def validate_identity(row):
    validate_case(row)
    expected = row['gil_mode'] == 'on'
    gil = row.get('gil', {})
    require(all(gil.get(key) is expected for key in ('expected', 'before', 'after'))
            and type(gil.get('build_disabled')) is int and gil['build_disabled'] == 1,
            'effective GIL before/after differs or interpreter is not a free-threaded build')
    require(row.get('execution_device') == row['device'] == row.get('actual_device_type'),
            'CPU and real NPU execution identities differ')
    identity = row.get('comparison_identity')
    require(type(identity) is dict, 'comparison identity missing')
    require(all(identity.get(key) == value for key, value in IDENTITY.items()),
            'frozen model/state/input template identity differs')
    hashes = identity.get('input_hashes')
    require(type(hashes) is list and len(hashes) == len(set(hashes)) == 8
            and all(sha256(value) for value in hashes), 'eight distinct input hashes required')
    bank = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
    require(identity.get('input_bank_sha256') == bank, 'input bank hash differs')
    require(type(identity.get('python_executable')) is str and bool(identity['python_executable'])
            and sha256(identity.get('python_sha256')) and sha256(identity.get('source_sha256')),
            'actual interpreter/source pins required')
    policy = row.get('execution_policy', {})
    for key in ('closed_loop', 'shared_model', 'eval_mode', 'independent_input_slots',
                'outputs_validated_outside_window'):
        require(policy.get(key) is True, 'execution policy missing: ' + key)
    require(policy.get('device_synchronize_in_request_loop') is False,
            'global device synchronization is forbidden in request loop')
    require(policy.get('event_completion') is (row['device'] == 'npu'), 'completion policy differs')
    require(type(policy.get('prepare_multi_caller')) is int and policy['prepare_multi_caller'] == 1,
            'Prepare multiCaller must remain 1')
    return copy.deepcopy(identity)


def _native_contracts():
    """Load only the frozen stdlib audit rules, never the device backend."""
    global _NATIVE_CONTRACTS
    if _NATIVE_CONTRACTS is None:
        path = Path(__file__).resolve().parents[1] / 'eager_mpsc_r34/contracts.py'
        spec = importlib.util.spec_from_file_location('_r35_frozen_native_contracts_audit', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NATIVE_CONTRACTS = module
    return _NATIVE_CONTRACTS


def validate_native_evidence(row):
    """Recompute recorded native gates without accessing provider files/devices.

    Provider bytes and the source closure are verified by the producing worker.
    This offline check binds its recorded provider pins to both mapped snapshots.
    """
    evidence = ('before', 'after', 'closed', 'native_gate', 'close_gate',
                'mapping_before', 'mapping_after', 'provider_pins', 'prepare_configuration')
    if row['device'] == 'cpu':
        require(all(row.get(key) is None for key in evidence), 'CPU receipt contains NPU evidence')
        return
    for key in ('before', 'after', 'closed', 'native_gate', 'close_gate', 'provider_pins'):
        require(type(row.get(key)) is dict and bool(row[key]), 'missing NPU evidence: ' + key)
    for key in ('before', 'after', 'closed'):
        require(row[key].get('backend') == 'real_sdk' and row[key].get('simulated_prepare') is False,
                'native snapshot must use the real non-simulated SDK: ' + key)
    native = _native_contracts()
    expected_gate = native.native_delta(row['before'], row['after'], row['backend'], row['total_requests'])
    expected_close = native.validate_close(row['closed'], row['backend'])
    require(row['native_gate'] == expected_gate, 'reported native gate differs from raw snapshots')
    require(row['close_gate'] == expected_close, 'reported close gate differs from raw snapshot')
    require(all(row['closed']['queue'][key] == row['after']['queue'][key]
                for key in ('admitted', 'retired')), 'native admissions changed between quiescence and close')
    before, after = row.get('mapping_before'), row.get('mapping_after')
    require(type(before) is list and bool(before) and before == after, 'NPU provider mappings changed or missing')
    pins = row['provider_pins']
    for path, digest in pins.items():
        require(type(path) is str and PurePosixPath(path).is_absolute()
                and '..' not in PurePosixPath(path).parts and sha256(digest), 'invalid recorded provider pin')
    mapped = {}
    for entry in before:
        require(type(entry) is dict, 'invalid provider mapping')
        path, digest = entry.get('path'), entry.get('sha256')
        require(type(path) is str and path in pins and sha256(digest) and pins[path] == digest,
                'mapped NPU provider differs from recorded pins')
        require(path not in mapped, 'duplicate provider mapping')
        require(not any(marker in path.lower() for marker in ('hostsim', 'phase_api_tap', 'sdk-007')),
                'simulation/interposer provider in NPU receipt')
        mapped[path] = digest
    names = {PurePosixPath(path).name for path in mapped}
    require(any(name.startswith('libascendcl.so') for name in names), 'real AscendCL mapping missing')
    require(any(name.startswith('libopapi.so') for name in names)
            or {'libopapi_nn.so', 'libopapi_math.so'} <= names,
            'real monolithic/split opAPI mapping missing')
    natives = [digest for path, digest in mapped.items() if PurePosixPath(path).name == 'libtorch_npu.so']
    require(bool(natives) and all(digest == NATIVE_SHA256 for digest in natives),
            'frozen D894 native mapping missing or changed')


def audit_result(row):
    require(row.get('schema') == 1 and row.get('passed') is True, 'unsuccessful worker receipt')
    require(row.get('kind') in ('correctness', 'perf'), 'unknown receipt purpose')
    exposure = (3000, 24) if row['kind'] == 'perf' else (32, 0)
    require((row.get('total_requests'), row.get('warmup_per_caller')) == exposure,
            'fixed request/warmup exposure differs')
    integer(row.get('round'), 'round')
    validate_identity(row)
    validate_native_evidence(row)
    numeric = row.get('numeric_validation', {})
    require(numeric.get('passed') is True, 'numeric validation failed')
    for key in ('validated_requests', 'finite_count', 'shape_count', 'reference_compared_count'):
        require(type(numeric.get(key)) is int and numeric[key] == exposure[0],
                'all outputs must be validated: ' + key)
    shape = numeric.get('expected_shape')
    require(type(shape) is list and bool(shape)
            and all(type(size) is int and size > 0 for size in shape), 'invalid output shape')
    for key in ('max_abs_error', 'atol', 'rtol'):
        value = numeric.get(key)
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                'invalid numeric result: ' + key)
    metrics = request_metrics(row)
    if 'metrics' in row:
        require(row['metrics'] == metrics, 'reported metrics differ from raw records')
    result = copy.deepcopy(row)
    result['metrics'] = metrics
    return result


def summarize(rows, device='npu', rounds=1, include_prepare=False):
    expected = [task for task in campaign_plan(device, rounds, include_prepare) if task['kind'] == 'perf']
    require(type(rows) is list and len(rows) == len(expected), 'performance receipt count differs')
    indexed, identity = {}, None
    for row in rows:
        require(row.get('kind') == 'perf' and row.get('device') == device,
                'correctness or another device cannot supply performance evidence')
        audited = audit_result(row)
        current = audited['comparison_identity']
        if identity is None:
            identity = current
        require(current == identity, 'comparison identity changed across GIL/caller/stream arms')
        key = row['round'], row['case_id']
        require(key not in indexed, 'duplicate performance receipt')
        indexed[key] = audited
    require(set(indexed) == {(task['round'], task['case_id']) for task in expected},
            'missing or unexpected performance case/round')
    cases = {}
    for case in case_grid(device, include_prepare):
        windows = [indexed[index, case['case_id']]['metrics'] for index in range(rounds)]
        cases[case['case_id']] = dict(case, rounds=rounds, windows=windows,
            median_throughput_requests_per_second=statistics.median(
                metric['throughput_requests_per_second'] for metric in windows))
    return dict(schema=1, device=device, rounds=rounds, cases=cases,
                performance_request_count=sum(row['total_requests'] for row in rows),
                comparison_identity=identity,
                interpretation='Caller and host-call concurrency; host overlap alone does not prove device kernel overlap.')
