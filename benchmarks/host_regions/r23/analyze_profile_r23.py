"""Offline, fail-closed decomposition of the fixed R23 stages sidecar."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

RUNNER_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
HARNESS_SHA = 'd7461dbe7f1d6f0481e865381d9a07aa9bb64bf9ceb02a335a6776e2b0b39865'
MODEL = {'bottom': 'model.bottom_mlp', 'emb': 'model.apply_emb',
         'interaction': 'model.interact_features', 'top': 'model.top_mlp'}
NESTED = ('provider.guard', 'policy.check', 'policy.container_snapshot', 'runtime.map',
          'runtime.native_parallel', 'runtime.prepare_executor', 'session.open', 'session.drain',
          'session.close', 'executor.submit')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value):
    return type(value) is int


def union(intervals):
    total = 0; left = right = None
    for start, end in sorted(intervals):
        if left is None: left, right = start, end
        elif start > right: total += right-left; left, right = start, end
        else: right = max(right, end)
    return total + (right-left if left is not None else 0)


def distribution(values):
    require(values and all(integer(v) and v >= 0 for v in values), 'nonnegative integer ns required')
    ordered = sorted(values); count = len(values)
    return dict(count=count, mean_ns=sum(values)/count,
                p50_ns=ordered[math.ceil(.50*count)-1], p95_ns=ordered[math.ceil(.95*count)-1],
                p99_ns=ordered[math.ceil(.99*count)-1], min_ns=ordered[0], max_ns=ordered[-1],
                quantiles='nearest rank')


def argument(argv, flag):
    require(type(argv) is list and all(type(v) is str for v in argv) and argv.count(flag) == 1,
            'single original runner argument required: ' + flag)
    index = argv.index(flag); require(index+1 < len(argv), 'missing runner argument value')
    return argv[index+1]


def validate_headers(raw, profile):
    require(raw.get('passed') is True and raw.get('phase') == 'complete' and raw.get('kind') == 'perf'
            and raw.get('abi') in ('std', 'ft') and raw.get('arm') in ('serial', 'worker4'), 'raw runner did not pass fixed perf')
    require(raw.get('script_sha256') == RUNNER_SHA and profile.get('runner', {}).get('sha256') == RUNNER_SHA
            and profile.get('tool_sha256') == HARNESS_SHA, 'runner/harness source SHA differs')
    require(profile.get('schema') == 1 and profile.get('profile_mode') == 'stages'
            and profile.get('diagnostic_only') is True and profile.get('performance_conclusion_allowed') is False
            and profile.get('instrumentation_complete') is True and profile.get('temporary_methods_restored') is True
            and integer(profile.get('original_exit_code')) and profile['original_exit_code'] == 0
            and profile.get('failure') is None and integer(profile.get('requests')) and profile['requests'] == 2000,
            'profile is incomplete, unrestored, failed, or not the 2000-request stage mode')
    restores = profile.get('restoration')
    require(type(restores) is list and restores and all(r.get('restored') is True for r in restores),
            'temporary method restoration evidence missing')
    timing = raw.get('timing', {}); durations = timing.get('durations_ns')
    require(type(durations) is list and len(durations) == 2000 and all(integer(v) and v > 0 for v in durations)
            and timing.get('samples') == timing.get('requested_samples') == 2000 and timing.get('warmup') == 20
            and raw.get('forward_calls') == 2021, 'raw exposure differs from perf20+2000')
    argv = profile.get('original_argv')
    for flag, wanted in (('--abi', raw['abi']), ('--arm', raw['arm']), ('--kind', 'perf'),
                         ('--warmup', '20'), ('--iterations', '2000')):
        require(argument(argv, flag) == wanted, 'raw/profile runner cohort differs: ' + flag)
    wheel = argument(argv, '--expected-wheel-sha256'); native = argument(argv, '--expected-native-sha256')
    require(re.fullmatch('[0-9a-f]{64}', wheel) and re.fullmatch('[0-9a-f]{64}', native)
            and raw.get('requested_wheel', {}).get('sha256') == raw.get('wheel', {}).get('sha256') == wheel
            and raw.get('native_sha256') == native, 'raw/profile wheel or native identity differs')
    return durations


def validate_events(events):
    require(type(events) is list and events, 'profile events missing')
    by_id = {}
    for row in events:
        require(type(row) is dict and type(row.get('id')) is str and row['id'] not in by_id
                and type(row.get('name')) is str, 'invalid or duplicate event identity')
        for key in ('request', 'native_tid', 'start_ns', 'end_ns', 'start_thread_time_ns', 'end_thread_time_ns'):
            require(integer(row.get(key)) and row[key] >= 0, 'invalid event integer: ' + key)
        require(1 <= row['request'] <= 2000 and row['native_tid'] > 0
                and row['end_ns'] >= row['start_ns'] and row['end_thread_time_ns'] >= row['start_thread_time_ns']
                and 'error_type' not in row, 'invalid/failed event interval')
        by_id[row['id']] = row
    for row in events:
        if row.get('parent') is None: continue
        require(row['parent'] in by_id, 'event parent missing')
        parent = by_id[row['parent']]
        require(parent['native_tid'] == row['native_tid'], 'cross-TID parent cannot be deducted')
        require(parent['request'] == row['request'] and parent['start_ns'] < row['start_ns'] <= row['end_ns'] <= parent['end_ns'],
                'event parent interval/request differs')
    return by_id


def one(rows, name):
    selected = [r for r in rows if r['name'] == name]
    require(len(selected) == 1, 'one stage per request required: ' + name)
    return selected[0]


def partition_request(rows, raw_duration):
    forward, sync = one(rows, 'forward'), one(rows, 'tail_sync'); caller = forward['native_tid']
    require(integer(raw_duration) and raw_duration > 0 and sync['native_tid'] == caller, 'invalid raw duration/caller TID')
    require(forward['start_ns'] <= forward['end_ns'] <= sync['start_ns'] <= sync['end_ns'], 'forward/sync overlap or order differs')
    stages = {}; end = forward['start_ns']
    for label, name in MODEL.items():
        event = one(rows, name)
        require(event['native_tid'] == caller, 'model stage differs from caller TID')
        require(forward['start_ns'] <= event['start_ns'] <= event['end_ns'] <= forward['end_ns'], 'model stage outside forward')
        require(event['start_ns'] >= end, 'model stages overlap or order differs')
        stages[label] = event['end_ns']-event['start_ns']; end = event['end_ns']
    forward_ns = forward['end_ns']-forward['start_ns']; sync_ns = sync['end_ns']-sync['start_ns']
    require(raw_duration >= sync['end_ns']-forward['start_ns'], 'raw duration smaller than observed request envelope')
    stages.update({'forward.misc': forward_ns-sum(stages.values()), 'tail_sync': sync_ns,
                   'outer.misc': raw_duration-forward_ns-sync_ns})
    require(all(v >= 0 for v in stages.values()) and sum(stages.values()) == raw_duration, 'nonconserving/negative host partition')
    return dict(caller_tid=caller, stages_ns=stages, raw_ns=raw_duration,
                start_ns=forward['start_ns'], end_ns=sync['end_ns'])


def worker_metrics(rows):
    workers = [r for r in rows if r['name'] == 'worker.region']
    return dict(regions=len(workers), tids=sorted({r['native_tid'] for r in workers}),
                span_ns=max((r['end_ns'] for r in workers), default=0)-min((r['start_ns'] for r in workers), default=0),
                union_ns=union([(r['start_ns'], r['end_ns']) for r in workers]),
                thread_cpu_ns=sum(r['end_thread_time_ns']-r['start_thread_time_ns'] for r in workers))


def nested_metrics(rows):
    children = defaultdict(list)
    for row in rows: children[row.get('parent')].append(row)
    result = {}
    for name in NESTED:
        selected = [r for r in rows if r['name'] == name]
        if not selected: continue
        wall = cpu = exclusive = 0
        for row in selected:
            value = row['end_ns']-row['start_ns']; wall += value
            cpu += row['end_thread_time_ns']-row['start_thread_time_ns']
            nested = union([(c['start_ns'], c['end_ns']) for c in children[row['id']]
                            if c['native_tid'] == row['native_tid']])
            require(nested <= value, 'invalid nested union')
            exclusive += value-nested
        result[name] = dict(count=len(selected), inclusive_wall_ns=wall, inclusive_thread_cpu_ns=cpu, exclusive_wall_ns=exclusive)
    return result


def analyze(raw, profile):
    durations = validate_headers(raw, profile); events = profile.get('events'); validate_events(events)
    counts = dict(Counter(row['name'] for row in events))
    require(profile.get('actual_event_counts') == counts, 'profile event counts differ from actual events')
    expected = {name: 2000 for name in ('forward', 'tail_sync', *MODEL.values())}
    if raw['arm'] == 'worker4':
        expected.update({name: 2000 for name in ('runtime.map', 'runtime.native_parallel', 'runtime.prepare_executor',
            'provider.guard', 'policy.check', 'session.open', 'session.drain', 'session.close')})
        expected.update({'worker.region': 8000, 'executor.submit': 8000, 'region.body': 52000})
    require(profile.get('expected_event_counts') == expected and all(counts.get(k) == v for k, v in expected.items()),
            'fixed hook coverage differs')
    groups = defaultdict(list)
    for event in events: groups[event['request']].append(event)
    require(set(groups) == set(range(1, 2001)), 'request indexes are incomplete')
    requests = []; callers = set(); previous = -1
    for index, duration in enumerate(durations, 1):
        rows = groups[index]; row = partition_request(rows, duration); callers.add(row['caller_tid'])
        require(row['start_ns'] >= previous, 'request timelines overlap or are reordered'); previous = row['end_ns']
        require(all(r['native_tid'] == row['caller_tid'] for r in rows if r['name'] in NESTED),
                'caller nested metric appeared on a worker TID')
        workers = worker_metrics(rows)
        if raw['arm'] == 'worker4':
            local_counts = Counter(r['name'] for r in rows)
            for name in ('runtime.map', 'runtime.native_parallel', 'runtime.prepare_executor', 'provider.guard',
                         'policy.check', 'session.open', 'session.drain', 'session.close'):
                require(local_counts[name] == 1, 'per-request caller event count differs: ' + name)
            require(local_counts['executor.submit'] == 4, 'per-request caller submit count differs')
            require(workers['regions'] == 4 and row['caller_tid'] not in workers['tids'], 'worker region count/TIDs differ')
            embedding = one(rows, 'model.apply_emb')
            for region, (start, stop) in enumerate(((0, 6), (6, 13), (13, 19), (19, 26))):
                spans = [r for r in rows if r['name'] == 'worker.region' and r.get('region') == region]
                body = [r for r in rows if r['name'] == 'region.body' and r.get('region') == region]
                require(len(spans) == 1 and spans[0].get('first') == start and spans[0].get('stop') == stop
                        and sorted(r.get('item') for r in body) == list(range(start, stop)), 'worker region/body coverage differs')
                require(all(r.get('parent') == spans[0]['id'] and r['native_tid'] == spans[0]['native_tid'] for r in body),
                        'worker body identity differs from its region')
                require(embedding['start_ns'] <= spans[0]['start_ns'] <= spans[0]['end_ns'] <= embedding['end_ns'],
                        'worker region outside its request embedding envelope')
        else:
            require(workers['regions'] == 0 and not any(r['name'] == 'region.body' for r in rows), 'serial contains worker events')
        row.update(request=index, worker=workers, nested=nested_metrics(rows)); requests.append(row)
    require(len(callers) == 1, 'original caller TID changed')
    stages = {name: distribution([row['stages_ns'][name] for row in requests]) for name in requests[0]['stages_ns']}
    nested_summary = {name: {key: distribution([row['nested'][name][key] for row in requests])
                            for key in ('inclusive_wall_ns', 'inclusive_thread_cpu_ns', 'exclusive_wall_ns')}
                      for name in NESTED if all(name in row['nested'] for row in requests)}
    worker_summary = {key: distribution([row['worker'][key] for row in requests])
                      for key in ('span_ns', 'union_ns', 'thread_cpu_ns')}
    worker_summary['tids'] = sorted({tid for row in requests for tid in row['worker']['tids']})
    return dict(passed=True, abi=raw['abi'], arm=raw['arm'], requests=requests, stages=stages,
                nested_summary=nested_summary, worker_summary=worker_summary, raw_distribution=distribution(durations),
                native_measurements=dict(query_ns=None, execute_ns=None, queue_wait_ns=None, device_ns=None),
                observer_overhead=dict(measured_here=False, off_on_ratio=None,
                    boundary='Requires separate campaign off/on cohorts; never subtract observer overhead from stage values.'),
                boundaries=['Seven host stages are disjoint, nonnegative integers summing exactly to each original duration.',
                    'forward.misc is unmeasured work within forward; outer.misc includes wrapper/bookkeeping/gaps and clock-call overhead.',
                    'tail_sync includes host queue and eligible owner/event retirement plus device wait; it is not pure kernel time.',
                    'Nested wall/CPU metrics are inclusive; same-TID direct-child union alone is deducted for exclusive wall.',
                    'Worker CPU/span/union remain separate and must not be added to caller wall stages.',
                    'Pairing uses original argv cohort/wheel/native identities and supplied artifact SHA; campaign provenance remains authoritative.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('raw', 'profile', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv); output = args.output.resolve(); markdown = output.with_suffix('.md')
    require(not output.exists() and not markdown.exists() and output != markdown, 'new exclusive JSON/Markdown outputs required')
    payloads = {name: getattr(args, name).read_bytes() for name in ('raw', 'profile')}
    inputs = {name: dict(path=str(getattr(args, name).resolve()), sha256=hashlib.sha256(payload).hexdigest())
              for name, payload in payloads.items()}
    row = dict(schema=1, passed=False, diagnostic_only=True, performance_conclusion_allowed=False,
               tool_sha256=sha(__file__), source_pins=dict(runner=RUNNER_SHA, harness=HARNESS_SHA), inputs=inputs)
    try:
        raw = json.loads(payloads['raw']); profile = json.loads(payloads['profile'])
        analysis = analyze(raw, profile)
        require(all(sha(getattr(args, name)) == pin['sha256'] for name, pin in inputs.items()), 'input changed during analysis')
        row.update(analysis)
    except Exception as error:
        row.update(passed=False, error=str(error), error_type=type(error).__name__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(row, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    lines = ['# R23 host stage analysis', '', 'Diagnostic only; no new formal performance conclusion.', '']
    if row['passed']:
        lines += ['| Stage | Mean ms | P50 ms | P95 ms | P99 ms |', '|---|---:|---:|---:|---:|']
        for name, values in row['stages'].items():
            lines.append('| '+name+' | '+' | '.join(f"{values[key]/1e6:.6f}" for key in ('mean_ns', 'p50_ns', 'p95_ns', 'p99_ns'))+' |')
        lines += ['', 'Each request conserves the original duration exactly. Native query/execute/queue-wait/device durations are unmeasured (null).',
                  'Worker and nested metrics are separate in JSON. Observer overhead requires campaign off/on comparison.']
    else: lines += ['Rejected: '+row['error']]
    with markdown.open('x', encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=row['passed'], output=str(output), error=row.get('error'))), flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
