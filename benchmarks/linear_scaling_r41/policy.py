"""R38 setup/receipt policy gates; never called inside a request loop."""
import ctypes
import os
from pathlib import Path
import time

TASK_GETTER = '_Z23OpApiGetTaskQueueEnablev'
PROOF_KIND = 'fresh_process_environment_and_private_repository_thread_topology'
ENVIRONMENT_KEYS = ('TASK_QUEUE_ENABLE', 'PER_STREAM_QUEUE', 'ASCEND_LAUNCH_BLOCKING')
CPUS = set(range(8, 16))


def require(value, message):
    if not value:
        raise ValueError(message)


def environment_receipt(environment):
    return {key: environment.get(key) for key in ENVIRONMENT_KEYS}


def validate_environment(config, report, environment):
    expected = config.get('per_stream_queue')
    require(type(expected) is int and expected in (0, 1), 'explicit per_stream_queue must be integer0/1')
    require(all(report.get(k) == v for k, v in dict(device='npu', gil_mode='off', backend='off',
                                                  stream_mode='private').items()),
            'R38 supports real NPU, naturalGILoff, private streams, Prepareoff only')
    require(environment.get('TASK_QUEUE_ENABLE') == '1'
            and environment.get('PER_STREAM_QUEUE') == str(expected)
            and environment.get('ASCEND_LAUNCH_BLOCKING') == '0', 'queue environment differs from configured arm')
    return expected


def read_native_options(library_path, *, loader=ctypes.CDLL, noload_flag=None, local_flag=None):
    """Read the exported task getter on the already-loaded exact library.

    D894 does not export a per-stream OptionsManager getter. A missing direct
    getter is explicit evidence metadata, never fabricated as effectivePSQ.
    """
    if noload_flag is None: noload_flag = getattr(os, 'RTLD_NOLOAD', None)
    if local_flag is None: local_flag = getattr(os, 'RTLD_LOCAL', None)
    require(type(noload_flag) is int and noload_flag != 0 and type(local_flag) is int,
            'safe RTLD_NOLOAD library lookup unavailable')
    try:
        library = loader(str(library_path), mode=noload_flag | local_flag)
        getter = getattr(library, TASK_GETTER)
    except (OSError, AttributeError) as error:
        raise ValueError('already-loaded native task getter unavailable') from error
    getter.argtypes = []
    getter.restype = ctypes.c_uint32
    result = dict(task_queue_enable=int(getter()), task_queue_getter_symbol=TASK_GETTER,
                  per_stream_queue_native_getter_available=False,
                  per_stream_queue_native_value=None, proof_kind=PROOF_KIND)
    validate_native_options(result)
    return result


def validate_native_options(observed):
    require(type(observed) is dict and type(observed.get('task_queue_enable')) is int
            and observed['task_queue_enable'] == 1, 'effective native task queue is not1')
    require(observed.get('per_stream_queue_native_getter_available') is False
            and observed.get('per_stream_queue_native_value') is None
            and observed.get('proof_kind') == PROOF_KIND,
            'direct native per-stream policy evidence is unavailable, not inferred')


def process_snapshot():
    """Raw host state outside ROI; start/end make sampling duration explicit."""
    result = dict(available=False, sample_started_ns=time.monotonic_ns(), process_cpu_ns=time.process_time_ns(),
                  tasks=[], memory={}, cpu_budget=sorted(CPUS))
    try:
        result['clock_ticks_per_second'] = int(os.sysconf('SC_CLK_TCK'))
        for path in sorted(Path('/proc/self/task').iterdir(), key=lambda p: int(p.name)):
            try:
                tid = int(path.name)
                stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                result['tasks'].append(dict(tid=tid, comm=(path / 'comm').read_text().strip(),
                    affinity=sorted(os.sched_getaffinity(tid)), user_ticks=int(stat[11]), system_ticks=int(stat[12])))
            except (OSError, ValueError) as error:
                result.setdefault('task_read_errors', []).append(dict(tid=path.name, error=repr(error)))
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
            if line.startswith(('Rss:', 'Pss:')):
                name, value, unit = line.split()
                require(unit == 'kB', 'unexpected memory unit')
                result['memory'][name[:-1].lower()+'_bytes'] = int(value)*1024
        require(result['tasks'] and {'rss_bytes', 'pss_bytes'} <= result['memory'].keys(),
                'process snapshot incomplete')
        result['available'] = True
    except (OSError, ValueError, AttributeError) as error:
        result['error'] = repr(error)
    result['sample_ended_ns'] = time.monotonic_ns()
    return result


def _named_threads(snapshot):
    require(type(snapshot) is dict and snapshot.get('available') is True, 'host topology snapshot unavailable')
    tasks = snapshot.get('tasks')
    require(type(tasks) is list, 'task snapshot missing')
    errors = snapshot.get('task_read_errors', [])
    require(type(errors) is list and not errors, 'host topology observation incomplete: task read failed')
    seen = set()
    names = {'acl_thread': set(), 'release_thread': set()}
    for row in tasks:
        tid = row.get('tid')
        require(type(tid) is int and tid > 0 and tid not in seen, 'duplicate/invalid native threadID')
        seen.add(tid)
        affinity = row.get('affinity')
        require(type(affinity) is list and bool(affinity)
                and all(type(cpu) is int for cpu in affinity) and set(affinity) <= CPUS,
                'process thread escaped fixed CPU8..15 budget')
        if row.get('comm') in names:
            names[row['comm']].add(tid)
    return names


def validate_topology(baseline, current, expected, callers, *, initialized):
    require(type(expected) is int and expected in (0, 1) and callers in (1, 2, 4)
            and type(initialized) is bool, 'invalid topology expectation')
    old, new = _named_threads(baseline), _named_threads(current)
    needed = callers if expected == 1 and initialized else 0
    delta = {}
    for name in old:
        require(old[name] <= new[name], 'baseline PTA thread disappeared: '+name)
        delta[name] = sorted(new[name]-old[name])
        require(len(delta[name]) == needed, 'private repository thread creation differs: '+name)
    return dict(passed=True, initialized=initialized, expected_new_per_type=needed, new_thread_ids=delta)


def audit_policy_receipt(report, *, expected_native_sha256):
    expected = report.get('per_stream_queue_requested')
    config = dict(per_stream_queue=expected)
    for phase in ('before', 'after'):
        validate_environment(config, report, report.get('queue_policy_environment_'+phase, {}))
        validate_native_options(report.get('queue_policy_'+phase))
    require(report['queue_policy_before'] == report['queue_policy_after'], 'native policy changed during run')
    callers = report.get('callers')
    require(type(callers) is int and callers in (1, 2, 4), 'invalid caller count')
    warmup = report.get('warmup_per_caller')
    require(type(warmup) is int and warmup in (0, 24), 'unexpected exposure for topology audit')
    ids = report.get('private_stream_ids')
    require(type(ids) is list and len(ids) == callers and len(set(ids)) == callers
            and all(type(x) is int and x != 0 for x in ids), 'distinct nondefault streamIDs required')
    library = report.get('queue_policy_library', {})
    require(type(library) is dict and type(library.get('path')) is str
            and library['path'].startswith('/') and library['path'].endswith('/libtorch_npu.so')
            and library.get('sha256') == expected_native_sha256, 'native getter library identity differs')
    for phase in ('before', 'after'):
        require(library in report.get('mapping_'+phase, []), 'native getter library differs from mapped provider')
    baseline = report.get('queue_topology_baseline')
    first = validate_topology(baseline, report.get('queue_topology_before_timing'), expected, callers,
                              initialized=warmup > 0)
    final = validate_topology(baseline, report.get('queue_topology_after_timing'), expected, callers,
                              initialized=True)
    return dict(passed=True, proof_kind=PROOF_KIND, before_timing=first, after_timing=final)
