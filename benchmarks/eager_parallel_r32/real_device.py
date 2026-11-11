"""Five-arm real A3 validation of the frozen R31 model and input.

Each invocation runs one arm in a fresh process. No Python model, dispatcher,
operator, or arithmetic replacement is installed by this benchmark.
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

POLICIES = {
    'off': dict(mode='shadow', min_embedding_bag_length=0),
    'e': dict(mode='e', min_embedding_bag_length=0),
    'ei': dict(mode='ei', min_embedding_bag_length=0),
    'ae': dict(mode='e', min_embedding_bag_length=64),
    'aei': dict(mode='ei', min_embedding_bag_length=64),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pins(config):
    for path, digest in config['pins'].items():
        require(sha(path) == digest, 'artifact/source pin changed: ' + path)


def mappings():
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '/' in line and '.so' in line})
    require(not any('hostsim' in p.lower() or 'sdk-007' in p for p in paths),
            'host-sim library entered real-device process')
    for name in ('libascendcl.so', 'libtorch_npu.so'):
        require(any(name in p for p in paths), 'required real-device library absent: ' + name)
    real_opapi = {Path(p).name for p in paths if '/usr/local/Ascend/' in p}
    # CANN 9.1 can resolve these operators from its split libraries before
    # opening the monolithic libopapi.so (see GetOpApiFuncAddr in op-plugin).
    require('libopapi.so' in real_opapi or
            {'libopapi_nn.so', 'libopapi_math.so'} <= real_opapi,
            'monolithic or split opapi must come from the installed real CANN toolkit')
    return paths


def run(config, report):
    require(sys.version_info[:3] == (3, 14, 6) and not sys._is_gil_enabled(),
            'matched natural free-threaded runtime required')
    for name, value in {'TASK_QUEUE_ENABLE': '0', 'PER_STREAM_QUEUE': '0',
                        'ASCEND_LAUNCH_BLOCKING': '0', 'OMP_NUM_THREADS': '1',
                        'OPENBLAS_NUM_THREADS': '1', 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}.items():
        require(os.environ.get(name) == value, 'environment differs: ' + name)
    require(os.environ.get('QWEN_ACL_HOST_SIM') is None and os.environ.get('PYTHON_GIL') is None,
            'simulation and forced GIL settings are forbidden')
    os.sched_setaffinity(0, config['cpus'])
    pins(config)
    import numpy
    import torch
    import torch_npu
    from torch_npu import _eager_parallel as ep
    require(torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
            and numpy.__version__ == '2.4.4' and not sys._is_gil_enabled(), 'runtime identity differs')
    require(Path(torch_npu.__file__).resolve().parent == Path(config['package']).resolve(),
            'unexpected torch_npu package imported')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workload = load(config['workload'], 'r32_frozen_workload')
    gates = load(config['routing_gates'], 'r32_frozen_routing_gates')
    common = load(config['measurement'], 'r32_frozen_measurement')
    report['runtime'] = dict(python=sys.version, torch=torch.__version__,
        torch_git=torch.version.git_version, numpy=numpy.__version__, pta=torch_npu.__version__,
        device=torch.npu.get_device_name(0), gil_enabled=sys._is_gil_enabled(),
        affinity=sorted(os.sched_getaffinity(0)))
    policy = POLICIES[report['arm']]
    report['policy'] = policy
    with torch.inference_mode():
        model, inputs, report['provenance'] = workload.load_original_model(config['upstream'])
        expected = model(*inputs).clone()
        expected_stages = workload.capture_stages(model, inputs)
        model = model.to('npu:0')
        inputs = (inputs[0].npu(), [v.npu() for v in inputs[1]], [v.npu() for v in inputs[2]])
        torch.npu.synchronize()
        if report['arm'] == 'off':
            ep.configure(enabled=False, mode='shadow')
            scope = nullcontext()
        else:
            scope = ep.scope(**policy)
        with scope:
            report['correctness'] = []
            for _ in range(8):
                actual = model(*inputs)
                torch.npu.synchronize()
                error = workload.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
                report['correctness'].append(error)
                require(error['passed'], 'full original forward differs from CPU reference')
            report['stage_validation'] = workload.validate_stages(
                model, inputs, expected_stages, torch.npu.synchronize)
            require(report['stage_validation']['passed'], 'intermediate stage differs from CPU reference')
            report['mapping_before'] = mappings()
            measured, actual = common.measure_forward_window(model, inputs, torch.npu.synchronize,
                warmup=report['warmup'], samples=report['samples'], snapshot=ep.snapshot)
            report.update(measured)
            report['numeric_final'] = workload.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
            require(report['numeric_final']['passed'], 'last timed output differs from CPU reference')
            report['native_timed_gate'] = gates.native_delta(report['before'], report['after'],
                report['arm'], report['samples'], routing=workload.ROUTING_PROFILE)
        report['closed'] = ep.snapshot()
        require(report['closed']['closure']['passed'], 'native scope did not close cleanly')
    report['mapping_after'] = mappings()
    require(not sys._is_gil_enabled(), 'GIL enabled during execution')
    pins(config)
    report['timing'] = common.distribution(report['durations_ns'])
    if report['kind'] == 'perf':
        require(report['window_wall_ns'] >= 2_000_000_000, 'performance window below two seconds')
    report.update(passed=True, original_model_executed=True, compute_backend='real_a3_cann',
                  numerical_checks_in_timing=False, startup_included=False, gil_enabled=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', choices=tuple(POLICIES), required=True)
    parser.add_argument('--kind', choices=('correctness', 'perf'), required=True)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--samples', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=50)
    args = parser.parse_args()
    require(args.samples > 0 and args.warmup >= 0 and args.round >= 0, 'invalid sample counts')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, passed=False, arm=args.arm, kind=args.kind, round=args.round,
                  samples=args.samples, warmup=args.warmup, pid=os.getpid(), worker_sha256=sha(__file__))
    try:
        config = json.loads(args.config.read_text())
        report['pins'] = config['pins']
        run(config, report)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(arm=args.arm, passed=True, **report['timing'])), flush=True)


if __name__ == '__main__':
    main()
