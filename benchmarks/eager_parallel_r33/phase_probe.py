"""Phase timing and short real-device traces using the frozen R32 package.

Tracing is a diagnostic cohort; its durations must not replace unprofiled E2E
results. No model methods, operators, parameters, or input tensors are replaced.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(rows):
    return {key: dict(mean_ns=statistics.mean(r[key] for r in rows),
                     median_ns=statistics.median(r[key] for r in rows))
            for key in ('forward_ns', 'final_sync_ns', 'e2e_ns')}


def run(args, report):
    config = json.loads(args.config.read_text())
    old = load(args.r32_worker, 'r33_frozen_r32_worker')
    old.pins(config)
    os.sched_setaffinity(0, config['cpus'])
    import torch
    import numpy
    import torch_npu
    from torch_npu import _eager_parallel as ep
    assert not sys._is_gil_enabled() and os.environ.get('PYTHON_GIL') is None
    assert torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
    assert numpy.__version__ == '2.4.4'
    assert Path(torch_npu.__file__).resolve().parent == Path(config['package']).resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    w = load(config['workload'], 'r33_workload')
    gates = load(config['routing_gates'], 'r33_native_gates')
    report.update(pins=config['pins'], policy=old.POLICIES[args.arm],
                  runtime=dict(torch=torch.__version__, numpy=numpy.__version__,
                               python=sys.version, gil=sys._is_gil_enabled(),
                               pta=torch_npu.__file__, device=torch.npu.get_device_name(0)))
    tap = None
    if args.tap != 'none':
        from api_tap import ApiTap
        tap = ApiTap(args.tap == 'active')
        mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if 'libphase_api_tap.so' in line}
        assert len(mapped) == 1
        binary = Path(next(iter(mapped)))
        report['tap_identity'] = dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                      frequency_hz=tap.frequency_hz,
                                      reader_sha256=hashlib.sha256(Path(__file__).with_name('api_tap.py').read_bytes()).hexdigest())
    report['environment'] = {name: os.environ.get(name) for name in
                             ('TASK_QUEUE_ENABLE', 'PER_STREAM_QUEUE', 'ASCEND_LAUNCH_BLOCKING',
                              'OMP_NUM_THREADS', 'ASCEND_RT_VISIBLE_DEVICES')}
    with torch.inference_mode():
        model, inputs, report['provenance'] = w.load_original_model(config['upstream'])
        expected = model(*inputs).clone()
        stages = w.capture_stages(model, inputs)
        model = model.npu()
        inputs = (inputs[0].npu(), [v.npu() for v in inputs[1]], [v.npu() for v in inputs[2]])
        torch.npu.synchronize()
        if args.arm == 'off':
            ep.configure(enabled=False, mode='shadow')
            scope = nullcontext()
        else:
            scope = ep.scope(**old.POLICIES[args.arm])
        with scope:
            if tap:
                tap.reset()
                tap.phase(0)
                cold_output = model(*inputs)
                tap.phase(1)
                torch.npu.synchronize()
                tap.phase(-1)
                report['cold_api_check'] = tap.snapshot()
                error = w.error_record(cold_output.cpu(), expected, rtol=1e-4, atol=1e-5)
                assert error['passed']
            for _ in range(args.warmup):
                output = model(*inputs)
                torch.npu.synchronize()
            report['stage_validation'] = w.validate_stages(model, inputs, stages, torch.npu.synchronize)
            assert report['stage_validation']['passed']
            report['mapping_before'] = old.mappings()

            def window(count, annotate=False, step=None):
                rows = []
                before = ep.snapshot()
                if tap:
                    tap.reset()
                for i in range(count):
                    outer = torch.autograd.profiler.record_function('r33.iter') if annotate else nullcontext()
                    with outer:
                        forward = torch.autograd.profiler.record_function('r33.forward') if annotate else nullcontext()
                        sync = torch.autograd.profiler.record_function('r33.final_sync') if annotate else nullcontext()
                        with forward:
                            if tap:
                                tap.phase(0)
                            realtime_start = time.time_ns() if args.kind == 'trace' else None
                            start = time.perf_counter_ns()
                            result = model(*inputs)
                            middle = time.perf_counter_ns()
                            realtime_middle = time.time_ns() if args.kind == 'trace' else None
                        with sync:
                            if tap:
                                tap.phase(1)
                            torch.npu.synchronize()
                            end = time.perf_counter_ns()
                            realtime_end = time.time_ns() if args.kind == 'trace' else None
                            if tap:
                                tap.phase(-1)
                    rows.append(dict(iteration=i, forward_ns=middle-start, final_sync_ns=end-middle,
                                     e2e_ns=end-start, realtime_start_ns=realtime_start,
                                     realtime_middle_ns=realtime_middle, realtime_end_ns=realtime_end))
                    if step:
                        step()
                after = ep.snapshot()
                gate = gates.native_delta(before, after, args.arm, count, routing=w.ROUTING_PROFILE)
                error = w.error_record(result.cpu(), expected, rtol=1e-4, atol=1e-5)
                assert error['passed']
                counters = tap.snapshot() if tap else None
                return dict(samples=count, rows=rows, timing=distribution(rows), before=before,
                            after=after, gate=gate, numeric=error, api_counters=counters)

            if args.kind == 'split':
                report['measurement'] = window(args.samples)
            elif args.kind == 'calibrate':
                assert tap is not None
                report['windows'] = []
                for number in range(args.rounds):
                    for active in ([False, True] if number % 2 == 0 else [True, False]):
                        tap.active = active
                        measured = window(args.samples)
                        measured.update(active=active, round=number)
                        report['windows'].append(measured)
                report['measurement'] = report['windows'][-1]
            else:
                report['control_before'] = window(args.samples)
                ec = torch_npu.profiler._ExperimentalConfig(
                    profiler_level=getattr(torch_npu.profiler.ProfilerLevel, args.level),
                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                    data_simplification=False)
                with torch_npu.profiler.profile(
                    activities=([torch_npu.profiler.ProfilerActivity.CPU] if args.cpu_trace else [])+
                               [torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=args.samples, repeat=1),
                    record_shapes=False, profile_memory=False, with_stack=False,
                    experimental_config=ec,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output/'profile'), analyse_flag=False)) as profiler:
                    report['measurement'] = window(args.samples, annotate=args.cpu_trace, step=profiler.step)
                report['raw_profile_path'] = profiler.prof_if.prof_path
                report['control_after'] = window(args.samples)
                # Analysis is deliberately separate from all measurement windows.
                saved = {name: os.environ.get(name) for name in ('PYTHONHOME', 'PYTHONPATH')}
                try:
                    # The external CANN exporter uses the image's Python, while
                    # this already-initialized FT process retains its sys.path.
                    os.environ.pop('PYTHONHOME', None)
                    os.environ['PYTHONPATH'] = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages'
                    profiler.export_chrome_trace(str(args.output/'trace.json'))
                finally:
                    for name, value in saved.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                report['trace_exists'] = (args.output/'trace.json').exists()
                assert report['trace_exists']
                trace = json.loads((args.output/'trace.json').read_text())
                events = trace.get('traceEvents', []) if isinstance(trace, dict) else trace
                assert any(v.get('name') == 'process_name' and v.get('args', {}).get('name') == 'Ascend Hardware'
                           for v in events), 'trace lacks actual device hardware events'
            report['mapping_after'] = old.mappings()
        report['closed'] = ep.snapshot()
        assert report['closed']['closure']['passed']
    old.pins(config)
    assert not sys._is_gil_enabled()
    report['passed'] = True


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--r32-worker', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', required=True, choices=['off', 'e', 'ei', 'ae', 'aei'])
    p.add_argument('--kind', required=True, choices=['split', 'trace', 'calibrate'])
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--tap', choices=['none', 'disabled', 'active'], default='none')
    p.add_argument('--level', choices=['Level0', 'Level1', 'Level2'], default='Level1')
    p.add_argument('--cpu-trace', action='store_true')
    p.add_argument('--rounds', type=int, default=3)
    args = p.parse_args()
    assert args.samples > 0 and args.warmup >= 0
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, kind=args.kind, arm=args.arm, pid=os.getpid(),
                  warmup=args.warmup, samples=args.samples, tap=args.tap, level=args.level,
                  cpu_trace=args.cpu_trace, worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        run(args, report)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'], arm=args.arm,
                          timing=report['measurement']['timing'])), flush=True)


if __name__ == '__main__':
    main()
