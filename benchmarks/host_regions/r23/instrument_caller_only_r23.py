"""Bounded caller-only supplement around the unchanged R22 perf runner."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

BASE_SHA = 'd7461dbe7f1d6f0481e865381d9a07aa9bb64bf9ceb02a335a6776e2b0b39865'
RUNNER_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
CORE = ('forward', 'model.bottom_mlp', 'model.apply_emb', 'model.interact_features', 'model.top_mlp', 'tail_sync')
POLICY = ('provider.guard', 'policy.check')
SNAPSHOT = 'policy.container_snapshot'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value: raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def load_base(path):
    require(sha(path) == BASE_SHA, 'frozen base wrapper source SHA mismatch')
    return load_module('_r23_caller_frozen_collector', path)


def verify_python_sources(base, collector):
    for short, expected in base.PYTHON_PINS.items():
        name = 'torch_npu._host_regions.'+short; module = sys.modules[name]
        require(sha(module.__file__) == expected, 'installed caller policy source differs: '+name)
        collector.module_sources[name] = dict(path=str(Path(module.__file__).resolve()), sha256=expected)


def install_caller_hooks(base, runner, collector):
    original_run, original_eager, original_install = runner.run, runner.eager_proof, runner.install_real_worker_model
    state = dict(arm=None, probe=None, late_sync_pending=False)
    def run(args, report, cleanup):
        state['arm'] = args.arm; collector.report = report
        try: return original_run(args, report, cleanup)
        finally: collector.restore()  # Before original main's token.close/runtime.close.
    def eager(model, torch):
        value = original_eager(model, torch)
        if state['arm'] == 'serial' and next(model.parameters()).device.type == 'npu':
            collector.attach_model(model, torch, tail_sync=True)
        elif state['arm'] == 'worker4' and state['late_sync_pending']:
            probe = state['probe']
            if probe.module._native_work is probe.saved:
                # The original restore+identity gate has already passed. Never modify _native_work.
                collector.wrap(torch.npu, 'synchronize', 'tail_sync')
                state['late_sync_pending'] = False
        return value
    def install(model, torch, *args, **kwargs):
        value = original_install(model, torch, *args, **kwargs); provider, probe = value[:2]
        verify_python_sources(base, collector)
        policies = [policy for policy in provider.policies if policy.accepted]
        require(len(policies) == 1, 'one accepted EmbeddingBag loop policy required')
        collector.wrap(provider, 'guard', 'provider.guard')
        collector.wrap(policies[0], 'check', 'policy.check')
        collector.wrap(policies[0], '_sequence', SNAPSHOT)
        collector.attach_model(model, torch)  # No sync wrapper until functional has built its cold barrier.
        state.update(probe=probe, late_sync_pending=True)
        return value
    collector.replace(runner, 'run', run, 'caller-only run adapter')
    collector.replace(runner, 'eager_proof', eager, 'caller-only eager adapter')
    collector.replace(runner, 'install_real_worker_model', install, 'caller-only install adapter')


def coverage(collector, code, arm, mode):
    events = collector.events(); counts = dict(Counter(row['name'] for row in events))
    if mode == 'off': return code == 0 and not events and not collector.restoration, counts, []
    required = CORE + (POLICY if arm == 'worker4' else ())
    allowed = set(required) | ({SNAPSHOT} if arm == 'worker4' else set())
    okay = code == 0 and collector.request == 2000 and set(counts) <= allowed
    okay = okay and all(counts.get(name) == 2000 for name in required)
    groups = defaultdict(list)
    for row in events: groups[row['request']].append(row)
    okay = okay and set(groups) == set(range(1, 2001)) and len({row['native_tid'] for row in events}) == 1
    snapshots = []; first_count = None
    for index in range(1, 2001):
        rows = groups[index]; local = Counter(row['name'] for row in rows)
        okay = okay and all(local[name] == 1 for name in required)
        selected = [row for row in rows if row['name'] == SNAPSHOT]
        if arm == 'worker4':
            first_count = len(selected) if first_count is None else first_count
            okay = okay and first_count > 0 and len(selected) == first_count
            snapshots.append(dict(request=index, count=len(selected),
                inclusive_wall_ns=sum(row['end_ns']-row['start_ns'] for row in selected),
                inclusive_thread_cpu_ns=sum(row['end_thread_time_ns']-row['start_thread_time_ns'] for row in selected)))
    okay = okay and not collector.worker_hook_installed and collector.worker_target is None
    return bool(okay), counts, snapshots


def caller_summary(base, events):
    summary = base.summarize(events)
    for row in summary['requests']:
        for key in ('worker_span_ns', 'worker_union_ns', 'worker_cpu_ns', 'worker_tids'): row[key] = None
        row['worker_metrics_observed'] = False
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--base-wrapper', type=Path, default=Path(__file__).with_name('instrument_frozen_runner_r23.py'))
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--profile-mode', choices=('off', 'caller_only'), default='caller_only')
    outer, forwarded = parser.parse_known_args(argv)
    base_path = outer.base_wrapper.resolve(strict=True); base = load_base(base_path)
    forwarded = base.runner_arguments(forwarded)
    runner_path = outer.runner.resolve(strict=True); output = outer.profile_output.resolve()
    require(sha(runner_path) == RUNNER_SHA, 'original frozen runner SHA mismatch')
    require(not output.exists(), 'new exclusive caller profile output required')
    runner = load_module('_r23_caller_original_runner', runner_path); args = runner.parse_args(forwarded)
    require(args.kind == 'perf', 'caller-only supplement retains exact perf20+2000 exposure')
    require(output not in (runner_path, base_path, args.output.resolve(), args.wheel.resolve())
            and not output.is_relative_to(args.upstream.resolve()), 'caller profile overlaps original inputs/result')
    collector = base.Collector(); code = None; failure = None
    try:
        if outer.profile_mode == 'caller_only': install_caller_hooks(base, runner, collector)
        code = runner.main(forwarded)
        return code
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        collector.restore(); events = collector.events()
        complete, counts, snapshots = coverage(collector, code, args.arm, outer.profile_mode)
        expected = {} if outer.profile_mode == 'off' else {name: 2000 for name in CORE+(POLICY if args.arm == 'worker4' else ())}
        if snapshots: expected[SNAPSHOT] = snapshots[0]['count']*2000
        worker_prefixes = ('worker.', 'region.', 'session.', 'executor.', 'runtime.')
        worker_events = sum(count for name, count in counts.items() if name.startswith(worker_prefixes))
        worker_hooks = int(collector.worker_hook_installed or collector.worker_target is not None
                           or any(r['name'] == '_native_work' or r['label'].startswith(worker_prefixes) for r in collector.restoration))
        row = dict(schema=1, profile_mode=outer.profile_mode, diagnostic_only=True, performance_conclusion_allowed=False,
            tool_sha256=sha(__file__), runner=dict(path=str(runner_path), sha256=RUNNER_SHA),
            base_wrapper=dict(path=str(base_path), sha256=BASE_SHA), original_argv=forwarded,
            original_exit_code=code, failure=failure, requests=collector.request,
            temporary_methods_restored=all(r['restored'] for r in collector.restoration), restoration=collector.restoration,
            installed_python_sources=collector.module_sources, instrumentation_complete=complete,
            expected_event_counts=expected, actual_event_counts=counts, events=events, summary=caller_summary(base, events),
            worker_hooks_installed=worker_hooks, worker_events=worker_events, snapshot_per_request=snapshots,
            worker_metrics_observed=False, native_measurements=dict(query_ns=None, execute_ns=None, queue_wait_ns=None, device_ns=None),
            limits=['Caller-only Python observation still changes caller/GIL scheduling; measure separate off/on overhead.',
                    'No runtime/session/submit/worker/body timing wrappers are installed.',
                    'Snapshot atomic calls can exceed 2000; exactly 2000 per-request aggregates preserve their true counts.',
                    'Query/execute/queue-wait/device durations remain unmeasured; tail_sync is not pure kernel time.'])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(row, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        require(row['temporary_methods_restored'], 'caller-only temporary method restoration failed')
        if code == 0: require(complete, 'caller-only request/event coverage differs')


if __name__ == '__main__': raise SystemExit(main())
