"""One fresh process, one full-model caller, one private completion stream."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import time
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value: raise ValueError(message)


def validate_runtime_mode(kind, build_disabled, enabled, options, environment):
    require('PYTHON_GIL' not in environment, 'ambiguous PYTHON_GIL')
    require(enabled is True, 'GIL must remain enabled')
    if kind == 'sameft-gilon':
        require(build_disabled == 1 and options.get('gil') == '1', 'same FT requires explicit -Xgil=1')
    elif kind == 'standard-gil':
        require(build_disabled in (None, 0) and 'gil' not in options, 'standard build required')
    else: raise ValueError('unknown runtime kind')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wait_json(path, deadline):
    while not path.exists():
        if time.monotonic() > deadline: raise TimeoutError('waiting for ' + str(path))
        time.sleep(.002)
    return json.loads(path.read_text())


def memory(torch, device):
    result = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        if line.startswith(('Rss:', 'Pss:')):
            key, value, unit = line.split()
            require(unit == 'kB', 'unknown memory unit')
            result[key[:-1].lower() + '_bytes'] = int(value) * 1024
    if device == 'npu':
        result['npu_allocated_bytes'] = torch.npu.memory_allocated(0)
        result['npu_reserved_bytes'] = torch.npu.memory_reserved(0)
    return result


def standard_mappings(S):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5]:
            path = str(Path(parts[5]).resolve(strict=True))
            if '/Ascend/' in path or Path(path).name.startswith(('libtorch_npu', 'libopapi', 'libcust_opapi', 'libnnopbase', 'libaclnn')):
                paths.add(path)
    require(any(Path(p).name == 'libtorch_npu.so' for p in paths), 'PTA mapping absent')
    require(any(Path(p).name.startswith('libascendcl.so') for p in paths), 'AscendCL mapping absent')
    require(not any(x in p.lower() for p in paths for x in ('hostsim', 'phase_api_tap', 'sdk-007')), 'simulation in NPU process')
    return [dict(path=p, sha256=S.sha(p)) for p in sorted(paths)]


def run(args, config, report):
    root = Path(config.get('frozen_r35_source_root', Path(__file__).resolve().parents[1]))
    S = load(root / 'forward_concurrency_r35/real_service.py', '_r36_frozen_r35_service')
    require(platform.machine() == 'aarch64' and sys.platform == 'linux', 'AArch64 Linux required')
    require(args.world_size in (1, 2, 4) and args.total_requests >= args.world_size and 0 <= args.rank < args.world_size,
            'invalid request partition')
    require(sys.getprofile() is None and sys.gettrace() is None, 'profiler forbidden')
    for key in ('QWEN_ACL_HOST_SIM', 'LD_PRELOAD', 'LD_AUDIT'):
        require(key not in os.environ, 'interposer forbidden: ' + key)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'CPU budget differs: ' + key)
    require(config['cpus'] == list(range(8, 16)), 'fixed CPU8..15 required')
    os.sched_setaffinity(0, config['cpus'])
    S.verify_pins(config['pins'])
    S.verify_pins(config['source_pins'])
    executable = str(Path(sys.executable).resolve())
    require(S.sha(executable) == config['python_sha256'], 'interpreter binary changed')
    mode = config['runtime_kind']
    build_disabled = sysconfig.get_config_var('Py_GIL_DISABLED')
    enabled = sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else build_disabled in (0, None)
    validate_runtime_mode(mode, build_disabled, enabled, sys._xoptions, os.environ)
    import numpy
    import torch
    actual = dict(python=platform.python_version(), torch=torch.__version__, torch_git=torch.version.git_version,
                  numpy=numpy.__version__, build_disabled=build_disabled)
    require(actual == config['expected_runtime'], 'runtime identity differs: ' + repr(actual))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    prepare = native_contracts = None
    device = config['device']
    require(device in ('cpu', 'npu'), 'invalid device')
    report['runtime'] = dict(actual, executable=executable, python_sha256=S.sha(executable),
                             runtime_kind=mode, affinity=sorted(os.sched_getaffinity(0)), gil_before=enabled)
    if device == 'npu':
        require(os.environ.get('TASK_QUEUE_ENABLE') == '1' and os.environ.get('PER_STREAM_QUEUE') == '0'
                and os.environ.get('ASCEND_LAUNCH_BLOCKING') == '0', 'NPU policy differs')
        import torch_npu
        report['runtime']['torch_npu'] = torch_npu.__version__
        if mode == 'standard-gil' and config.get('standard_empty_fix'):
            fix = str(Path(config['standard_empty_fix']).resolve())
            require(config['pins'].get(fix) == S.sha(fix) ==
                    '762fa14fb495ccff8cfd30a338f75c616aa99ab76f51a14b25b17b27c4646c30',
                    'unqualified standard empty-input fix')
            torch.ops.load_library(fix)
            report['extra_native_fixes'] = [dict(path=fix, sha256=S.sha(fix))]
        torch.npu.set_device(0)
        if mode == 'sameft-gilon':
            from torch_npu import _eager_prepare as prepare
            package = Path(config['package']).resolve()
            require(Path(torch_npu.__file__).resolve().parent == package, 'wrong private PTA package')
            require(S.sha(package / 'lib/libtorch_npu.so') == S.C.NATIVE_SHA256, 'frozen D894 required')
            native_contracts = S.load(config['native_contracts'], '_r36_native_contracts')
            prepare.configure(enabled=False, workers=4, capacity=8, byte_limit=536870912,
                              allowed_ops=('aclnnEmbeddingBag',), diagnostic_trace=False)
            report['native_configured'] = True
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    elif 'torch_npu' in sys.modules:
        raise ValueError('CPU process imported PTA')
    enabled = sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else True
    validate_runtime_mode(mode, build_disabled, enabled, sys._xoptions, os.environ)
    workload = S.load(config['workload'], '_r36_original_workload')
    with torch.inference_mode():
        model, template, provenance = workload.load_original_model(config['upstream'])
        require(all(provenance['identity'].get(k) == v for k, v in S.C.IDENTITY.items()), 'frozen model changed')
        specs = S.R.build_request_bank(template[0].tolist(), [x.tolist() for x in template[1]],
                                      [x.tolist() for x in template[2]])
        bank = [(torch.tensor(row['dense'], dtype=torch.float32),
                 [torch.tensor(x, dtype=torch.int64) for x in row['offsets']],
                 [torch.tensor(x, dtype=torch.int64) for x in row['indices']]) for row in specs]
        input_hashes = [workload.helper.tensor_tree_sha256(row, torch) for row in bank]
        require(input_hashes[0] == S.C.IDENTITY['input_sha256'] and len(set(input_hashes)) == 8, 'input bank changed')
        references = [model(*row).clone() for row in bank]
        initial_state = workload.helper.tensor_tree_sha256(dict(model.state_dict()), torch)
        require(initial_state == S.C.IDENTITY['state_sha256'], 'weights changed')
        target = 'cpu' if device == 'cpu' else 'npu:0'
        model = model.to(target).eval().requires_grad_(False)
        bank = [(row[0].to(target, copy=True), [t.to(target, copy=True) for t in row[1]],
                 [t.to(target, copy=True) for t in row[2]]) for row in bank]
        stream = torch.npu.Stream(device=0) if device == 'npu' else None
        event = torch.npu.Event(enable_timing=False) if device == 'npu' else None
        if device == 'npu': torch.npu.synchronize()
        context = torch.npu.stream(stream) if stream else nullcontext()
        with context:
            def invoke(request_id):
                value = model(*bank[request_id % 8])
                if event is not None:
                    event.record(torch.npu.current_stream(0))
                    event.synchronize()
                return value
            for i in range(args.warmup): invoke(i)
            if device == 'npu': torch.npu.synchronize()
            if prepare: report['before'] = S.quiesce(torch, prepare)
            mappings = (S.native_mappings(config) if prepare else standard_mappings(S)) if device == 'npu' else []
            report['mapping_before'] = mappings
            report['setup_memory'] = memory(torch, device)
            ready = dict(rank=args.rank, pid=os.getpid(), runtime=report['runtime'], setup_memory=report['setup_memory'])
            write_json(args.own / 'ready.json', ready)
            deadline = time.monotonic() + config.get('request_timeout_seconds', 540)
            start_ns = wait_json(args.output / 'go.json', deadline)['start_ns']
            while time.monotonic_ns() < start_ns:
                time.sleep(min(.0005, max(0, (start_ns-time.monotonic_ns())/1e9)))
            report['start_ns'] = start_ns
            requests, outputs = [], []
            for request_id in range(args.rank, args.total_requests, args.world_size):
                arrival = time.monotonic_ns()
                cpu_start = time.thread_time_ns()
                value = model(*bank[request_id % 8])
                returned = time.monotonic_ns()
                cpu_returned = time.thread_time_ns()
                if event is not None:
                    event.record(torch.npu.current_stream(0))
                    event.synchronize()
                response = time.monotonic_ns()
                cpu_end = time.thread_time_ns()
                requests.append(dict(request_id=request_id, input_id=request_id % 8,
                    arrival_ns=arrival, forward_return_ns=returned, response_ns=response,
                    thread_cpu_start_ns=cpu_start, forward_thread_cpu_end_ns=cpu_returned, thread_cpu_end_ns=cpu_end))
                outputs.append(value)
            # No device copies, report serialization, or weight hashing until all callers finish.
            write_json(args.own / 'done.json', dict(end_ns=requests[-1]['response_ns']))
            wait_json(args.output / 'validate.json', deadline)
        report['requests'] = requests
        report['after_memory'] = memory(torch, device)
        maximum = 0.0
        for row, output in zip(requests, outputs):
            require(output.device.type == device, 'wrong execution device')
            actual_output = output.cpu()
            reference = references[row['input_id']]
            require(bool(torch.isfinite(actual_output).all()), 'nonfinite output')
            torch.testing.assert_close(actual_output, reference, rtol=1e-4, atol=1e-5)
            maximum = max(maximum, float((actual_output-reference).abs().max()))
        report['numeric_validation'] = dict(passed=True, validated_requests=len(outputs), max_abs_error=maximum)
        require(workload.helper.tensor_tree_sha256(dict(model.state_dict()), torch) == initial_state, 'weights mutated')
        if prepare:
            report['after'] = S.quiesce(torch, prepare)
            report['native_gate'] = native_contracts.native_delta(report['before'], report['after'], 'off', len(outputs))
            S.close_native_domain(prepare, native_contracts, 'off', report)
        if device == 'npu':
            after = S.native_mappings(config) if prepare else standard_mappings(S)
            require(after == mappings, 'provider mappings changed')
            report['mapping_after'] = after
            torch.npu.synchronize()
        enabled = sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else True
        validate_runtime_mode(mode, build_disabled, enabled, sys._xoptions, os.environ)
        S.verify_pins(config['pins'])
        S.verify_pins(config['source_pins'])
        report.update(passed=True, shared_weights_mutated=False, original_full_model_executed=True,
            network_and_preprocessing_included=False, kernel_overlap_proven=False,
            comparison_identity=dict(S.C.IDENTITY, input_hashes=input_hashes), gil_after=enabled,
            config_sha256=S.sha(args.config), worker_sha256=S.sha(Path(__file__)),
            source_pins=config['source_pins'], device=device)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key in ('config', 'output'): parser.add_argument('--'+key, type=Path, required=True)
    for key in ('rank', 'world-size', 'total-requests', 'warmup'): parser.add_argument('--'+key, type=int, required=True)
    args = parser.parse_args()
    args.own = args.output / f'rank-{args.rank}'
    args.own.mkdir()
    report = dict(schema=1, passed=False, pid=os.getpid(), rank=args.rank,
                  world_size=args.world_size, total_requests=args.total_requests, warmup_per_process=args.warmup)
    code = 0
    try: run(args, json.loads(args.config.read_text()), report)
    except BaseException as error:
        code = 1
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if report.get('native_configured') and not report.get('native_close_attempted'):
            try:
                S = sys.modules['_r36_frozen_r35_service']
                S.close_native_domain(sys.modules['torch_npu._eager_prepare'],
                    sys.modules['_r36_native_contracts'], 'off', report, failure=True)
            except BaseException as close_error:
                report['requires_process_termination'] = True
                report['failure_close_error'] = repr(close_error)
    finally:
        try: write_json(args.own / 'result.json', report)
        finally:
            if report.get('requires_process_termination'): os._exit(2)
    if code: print(json.dumps(report['error']), file=sys.stderr, flush=True)
    return code


if __name__ == '__main__': sys.exit(main())
