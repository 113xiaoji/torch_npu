"""Diagnostic method spans around the unchanged, pinned R22 runner."""
import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import traceback

RUNNER_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
PYTHON_PINS = {
    'runtime': 'ef7bbc6b683ba22aaa48152e38ebf6cd25c04f564ebd77905ba9f112bda26cf8',
    'torch_npu_provider': 'd4b9f2aa7b1181f39a0d50fa2778dd1b349cd6e39e06eebe3e8338911cade568',
    'torch_npu_policy': '9153cbfdc2ad85e7dc81ea29f321e522973e1adc9f73831b495589daf49a41ec',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def runner_arguments(values):
    return values[1:] if values[:1] == ['--'] else values


def interval_union(intervals):
    total = 0; left = right = None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left; left, right = start, end
        else:
            right = max(right, end)
    return total + (right - left if left is not None else 0)


class Collector:
    def __init__(self, report=None):
        self.report = {} if report is None else report
        self.local = threading.local(); self.buffers = []; self.buffer_lock = threading.Lock()
        self.request = 0; self.patches = []; self.patched = set(); self.restoration = []
        self.models = set(); self.module_sources = {}
        self.worker_target = None; self.worker_hook_installed = False

    def state(self):
        if not hasattr(self.local, 'rows'):
            self.local.rows = []; self.local.stack = []; self.local.sequence = 0
            with self.buffer_lock:
                self.buffers.append(self.local.rows)
        return self.local

    def invoke(self, label, function, *args, new_request=False, details=None, **kwargs):
        state = self.state()  # Buffers are allocated on first wrapped call, even outside timing.
        if self.report.get('phase') != 'descriptive-timing':
            return function(*args, **kwargs)
        if new_request:
            self.request += 1
        request = self.request
        if request <= 0:
            return function(*args, **kwargs)
        tid = threading.get_native_id(); state.sequence += 1
        event = dict(id=f'{tid}:{state.sequence}', parent=state.stack[-1] if state.stack else None,
                     request=request, name=label, native_tid=tid)
        if details:
            event.update(details)
        state.stack.append(event['id'])
        event['start_ns'] = time.monotonic_ns(); event['start_thread_time_ns'] = time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            event['error_type'] = type(error).__name__
            raise
        finally:
            event['end_thread_time_ns'] = time.thread_time_ns(); event['end_ns'] = time.monotonic_ns()
            state.stack.pop(); state.rows.append(event)

    def replace(self, owner, name, wrapper, label):
        key = id(owner), name
        if key in self.patched:
            return
        namespace = vars(owner); own = name in namespace; raw = namespace.get(name)
        self.patches.append((owner, name, own, raw, wrapper, label))
        self.patched.add(key); setattr(owner, name, wrapper)

    def wrap(self, owner, name, label, *, new_request=False):
        if (id(owner), name) in self.patched:
            return
        saved = getattr(owner, name)
        @functools.wraps(saved)
        def observed(*args, **kwargs):
            actual_label = label(args, kwargs) if callable(label) else label
            return self.invoke(actual_label, saved, *args, new_request=new_request, **kwargs)
        self.replace(owner, name, observed, str(label) if isinstance(label, str) else name)

    def restore(self):
        while self.patches:
            owner, name, own, raw, wrapper, label = self.patches.pop()
            row = dict(name=name, label=label, owner_type=type(owner).__name__, restored=False)
            try:
                require(vars(owner).get(name) is wrapper, 'temporary method changed before restoration: ' + name)
                if own:
                    setattr(owner, name, raw)
                else:
                    delattr(owner, name)
                row['restored'] = True
            except BaseException as error:
                row['error'] = repr(error)
            self.restoration.append(row)
        return list(self.restoration)

    def events(self):
        return sorted((row for rows in self.buffers for row in rows), key=lambda row: row['start_ns'])

    def worker_wrapper(self, saved):
        @functools.wraps(saved)
        def observed(runtime, session, region, body, items):
            def timed_body(index, value):
                return self.invoke('region.body', body, index, value,
                                   details=dict(region=region.index, item=index))
            return self.invoke('worker.region', saved, runtime, session, region, timed_body, items,
                               details=dict(region=region.index, first=region.start, stop=region.stop))
        return observed

    def defer_worker(self, runtime, saved):
        self.worker_target = runtime, saved

    def attach_worker_if_ready(self):
        if self.worker_target is None:
            return self.worker_hook_installed
        runtime, saved = self.worker_target
        if runtime._native_work is not saved:
            return False  # Keep the original cold probe and its restoration gate intact.
        self.replace(runtime, '_native_work', self.worker_wrapper(saved), 'worker.region')
        self.worker_hook_installed = True; self.worker_target = None
        return True

    def attach_after_warmup(self, torch):
        if self.attach_worker_if_ready():
            self.wrap(torch.npu, 'synchronize', 'tail_sync')

    def attach_model(self, model, torch, *, tail_sync=False):
        if id(model) in self.models:
            return
        self.models.add(id(model))
        self.wrap(model, 'forward', 'forward', new_request=True)
        self.wrap(model, 'apply_emb', 'model.apply_emb')
        def mlp_label(args, _kwargs):
            layer = args[1] if len(args) > 1 else None
            return 'model.bottom_mlp' if layer is getattr(model, 'bot_l', None) else (
                'model.top_mlp' if layer is getattr(model, 'top_l', None) else 'model.apply_mlp')
        self.wrap(model, 'apply_mlp', mlp_label)
        self.wrap(model, 'interact_features', 'model.interact_features')
        if tail_sync:
            self.wrap(torch.npu, 'synchronize', 'tail_sync')

    def attach_native(self, provider):
        modules = {}
        for short, expected in PYTHON_PINS.items():
            name = 'torch_npu._host_regions.' + short
            module = sys.modules[name]; digest = sha(module.__file__)
            require(digest == expected, 'installed diagnostic Python source differs: ' + name)
            self.module_sources[name] = dict(path=str(Path(module.__file__).resolve()), sha256=digest)
            modules[short] = module
        runtime = modules['runtime']; native = modules['torch_npu_provider']; policy = modules['torch_npu_policy']
        self.wrap(runtime.RegionRuntime, 'map', 'runtime.map')
        self.wrap(runtime.RegionRuntime, '_native_parallel', 'runtime.native_parallel')
        saved = runtime.RegionRuntime._prepare_native_executor
        @functools.wraps(saved)
        def prepare(instance):
            value = self.invoke('runtime.prepare_executor', saved, instance)
            if instance._executor is not None:
                self.wrap(instance._executor, 'submit', 'executor.submit')
            return value
        self.replace(runtime.RegionRuntime, '_prepare_native_executor', prepare, 'runtime.prepare_executor')
        self.wrap(native.NativeCaptureProvider, 'guard', 'provider.guard')
        self.wrap(policy.NpuLoopPolicy, 'check', 'policy.check')
        self.wrap(policy.NpuLoopPolicy, '_sequence', 'policy.container_snapshot')
        for name in ('open', 'begin', 'end', 'done', 'drain', 'close'):
            self.wrap(native.NativeRunSession, name, 'session.' + name)


def install_hooks(runner, collector):
    original_run = runner.run; original_eager = runner.eager_proof; original_install = runner.install_real_worker_model
    arm = [None]
    def run(args, report, cleanup):
        arm[0] = args.arm; collector.report = report
        try:
            return original_run(args, report, cleanup)
        finally:
            # Original main executes token.close/runtime.close only after run returns.
            collector.restore()
    def eager(model, torch):
        value = original_eager(model, torch)
        if arm[0] == 'serial' and next(model.parameters()).device.type == 'npu':
            collector.attach_model(model, torch, tail_sync=True)
        elif arm[0] == 'worker4':
            # eager_after_warmup follows frozen probe.restore() and its identity check.
            collector.attach_after_warmup(torch)
        return value
    def install(model, torch, *args, **kwargs):
        value = original_install(model, torch, *args, **kwargs)
        collector.attach_native(value[0]); collector.attach_model(model, torch)
        collector.defer_worker(value[1].module, value[1].saved)
        return value
    collector.replace(runner, 'run', run, 'runner.run adapter')
    collector.replace(runner, 'eager_proof', eager, 'runner.eager_proof adapter')
    collector.replace(runner, 'install_real_worker_model', install, 'runner.install adapter')


def summarize(events):
    children = {}; groups = {}
    for row in events:
        children.setdefault(row['parent'], []).append(row)
        groups.setdefault(row['request'], []).append(row)
    totals = {}
    for row in events:
        wall = row['end_ns'] - row['start_ns']; cpu = row['end_thread_time_ns'] - row['start_thread_time_ns']
        nested = interval_union([(child['start_ns'], child['end_ns']) for child in children.get(row['id'], [])
                                 if child['native_tid'] == row['native_tid']])
        item = totals.setdefault(row['name'], dict(count=0, inclusive_wall_ns=0, thread_cpu_ns=0, exclusive_wall_ns=0))
        item['count'] += 1; item['inclusive_wall_ns'] += wall; item['thread_cpu_ns'] += cpu
        item['exclusive_wall_ns'] += max(0, wall - nested)
    requests = []
    for index, rows in sorted(groups.items()):
        workers = [row for row in rows if row['name'] == 'worker.region']
        requests.append(dict(request=index, worker_span_ns=(max(r['end_ns'] for r in workers) - min(r['start_ns'] for r in workers)) if workers else 0,
                             worker_union_ns=interval_union([(r['start_ns'], r['end_ns']) for r in workers]),
                             worker_cpu_ns=sum(r['end_thread_time_ns'] - r['start_thread_time_ns'] for r in workers),
                             worker_tids=sorted({r['native_tid'] for r in workers})))
    return dict(methods=totals, requests=requests,
                boundary='Wall and thread CPU method totals are inclusive. Exclusive wall subtracts direct same-thread child unions only. Cross-thread workers overlap caller drain; never add those wall spans.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--profile-mode', choices=('off', 'stages'), required=True)
    outer, forwarded = parser.parse_known_args(argv)
    forwarded = runner_arguments(forwarded)
    path = outer.runner.resolve(strict=True); output = outer.profile_output.resolve()
    require(sha(path) == RUNNER_SHA, 'original frozen runner SHA mismatch')
    require(not output.exists(), 'new exclusive profile output required')
    spec = importlib.util.spec_from_file_location('_r23_frozen_runner', path)
    runner = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runner; spec.loader.exec_module(runner)
    original_args = runner.parse_args(forwarded)
    require(outer.profile_mode == 'off' or original_args.kind == 'perf', 'stage diagnosis uses the unchanged perf20+2000 exposure')
    require(output not in (path, original_args.output.resolve(), original_args.wheel.resolve())
            and not output.is_relative_to(original_args.upstream.resolve()), 'profile output overlaps original inputs/result')
    collector = Collector(); code = None; failure = None
    try:
        if outer.profile_mode == 'stages':
            install_hooks(runner, collector)
        code = runner.main(forwarded)
        return code
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        collector.restore(); events = collector.events()
        counts = {}
        for event in events:
            counts[event['name']] = counts.get(event['name'], 0) + 1
        expected = {} if outer.profile_mode == 'off' else {'forward': 2000, 'tail_sync': 2000,
            'model.apply_emb': 2000, 'model.bottom_mlp': 2000, 'model.top_mlp': 2000, 'model.interact_features': 2000}
        if outer.profile_mode == 'stages' and original_args.arm == 'worker4':
            expected.update({'worker.region': 8000, 'region.body': 52000, 'runtime.map': 2000,
                'runtime.native_parallel': 2000, 'runtime.prepare_executor': 2000, 'executor.submit': 8000,
                'provider.guard': 2000, 'policy.check': 2000, 'session.open': 2000, 'session.drain': 2000, 'session.close': 2000})
        coverage = code == 0 and all(counts.get(name, 0) == count for name, count in expected.items())
        if outer.profile_mode == 'off':
            coverage = coverage and not events and not collector.restoration
        row = dict(schema=1, profile_mode=outer.profile_mode, diagnostic_only=True,
                   performance_conclusion_allowed=False, device_result_belongs_to_original_runner=True,
                   tool_sha256=sha(__file__), runner=dict(path=str(path), sha256=RUNNER_SHA),
                   original_argv=forwarded, original_exit_code=code, failure=failure,
                   temporary_methods_restored=all(r['restored'] for r in collector.restoration),
                   restoration=collector.restoration, installed_python_sources=collector.module_sources,
                   instrumentation_complete=coverage, expected_event_counts=expected, actual_event_counts=counts,
                   requests=collector.request, events=events, summary=summarize(events),
                   limits=['No native query/execute/device duration instrumented by these Python spans.',
                           'Session drain includes preparation-head wait and queue publication, not device completion.',
                           'Tail synchronize includes original queue, eligible owner/event retirement and device wait.',
                           'Off installs zero timing wrappers. On retains the original runner exposure and all validation guards.'])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(row, stream, indent=2, sort_keys=True); stream.write('\n')
        require(row['temporary_methods_restored'], 'temporary diagnostic method restoration failed')
        if code == 0:
            require(coverage, 'diagnostic hook coverage differs; original runner result is retained separately')


if __name__ == '__main__':
    raise SystemExit(main())
