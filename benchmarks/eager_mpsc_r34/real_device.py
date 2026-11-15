"""One real A3 arm per process; original frozen DLRM forward is called directly."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


C = load(Path(__file__).with_name('contracts.py'), 'r34_worker_contracts')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pins(pins):
    C.require(type(pins) is dict and pins, 'artifact/source pins required')
    for path, digest in pins.items():
        C.require(Path(path).is_absolute() and sha(path) == digest, 'artifact/source pin changed: ' + path)


def validate_environment(env, arm):
    policy = C.arm_policy(arm)
    required = dict(TASK_QUEUE_ENABLE='1', PER_STREAM_QUEUE='0', ASCEND_LAUNCH_BLOCKING='0',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    TORCH_DEVICE_BACKEND_AUTOLOAD='0', TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1',
                    TORCH_NPU_EAGER_PREPARE_INGRESS=policy['ingress'])
    for key, value in required.items():
        C.require(env.get(key) == value, 'environment differs: ' + key)
    for key in ('PYTHON_GIL', 'QWEN_ACL_HOST_SIM', 'LD_PRELOAD', 'LD_AUDIT'):
        C.require(key not in env, 'forced runtime, simulation or instrumentation is forbidden: ' + key)
    for key in ('FTLAB_HOST_REGIONS', 'FTLAB_HOST_NATIVE_RUN', 'HDB_REGION_PACKET', 'HDB_REGION_REAL_OPAPI'):
        C.require(env.get(key, '0') == '0', 'legacy capture path is enabled: ' + key)


def validate_mappings(paths, pins):
    C.require(not any(any(token in p.lower() for token in ('hostsim', 'sdk-007', 'phase_api_tap')) for p in paths),
              'simulation or diagnostic interposer entered real performance process')
    names = {Path(p).name for p in paths}
    C.require(any(n.startswith('libascendcl.so') for n in names)
              and any(n.startswith('libtorch_npu.so') for n in names), 'real AscendCL/PTA libraries are absent')
    C.require(any(n.startswith('libopapi.so') for n in names)
              or {'libopapi_nn.so', 'libopapi_math.so'} <= names, 'real monolithic/split opapi is absent')
    prefixes = ('libascendcl', 'libopapi', 'libcust_opapi', 'libnnopbase', 'libaclnn', 'libtorch_npu')
    providers = [p for p in paths if '/Ascend/' in p or Path(p).name.startswith(prefixes)]
    for path in providers:
        C.require(path in pins and sha(path) == pins[path], 'unapproved/changed mapped real provider: ' + path)
    return [dict(path=p, sha256=pins[p]) for p in providers]


def mappings(pins):
    paths = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5]:
            paths.append(str(Path(parts[5]).resolve(strict=True)))
    return validate_mappings(sorted(set(paths)), pins)


def precision(torch):
    result = dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                  conv_allow_hf32=torch.npu.conv.allow_hf32)
    C.require(all(value is False for value in result.values()), 'FP32 precision changed')
    return result


def run(config, report):
    policy = C.arm_policy(report['arm'])
    # Selection is cold and precedes imports; an inherited contradictory selector
    # is rejected rather than silently overwritten.
    os.environ.setdefault('TORCH_NPU_EAGER_PREPARE_INGRESS', policy['ingress'])
    validate_environment(os.environ, report['arm'])
    C.require(sys.platform == 'linux' and platform.machine() == 'aarch64'
              and sys.version_info[:3] == (3, 14, 6) and not sys._is_gil_enabled()
              and 'gil' not in sys._xoptions, 'pinned natural free-threaded A3 runtime required')
    C.require(sys.getprofile() is None and sys.gettrace() is None, 'Python profiling must be disabled')
    C.require(config.get('cpus') == list(range(8, 16)), 'fixed CPU8..15 budget required')
    C.require(config.get('expected_queued_per_forward', 25) == 25,
              'frozen workload requires 25 nonempty EmbeddingBag admissions per forward')
    verify_pins(config['pins'])
    mandatory = [str(Path(__file__).resolve()), str(Path(__file__).with_name('contracts.py').resolve()),
                 config['workload'], config['measurement']]
    package = Path(config['package']).resolve(strict=True)
    mandatory += [str(package / '__init__.py'), str(package / '_eager_prepare.py'),
                  str(package / 'lib/libtorch_npu.so')]
    C.require(all(path in config['pins'] for path in mandatory), 'measurement/package source closure pin missing')
    os.sched_setaffinity(0, config['cpus'])
    import numpy
    import torch
    import torch_npu
    from torch_npu import _eager_prepare as prepare
    C.require(torch.__version__ == '2.13.0+cpu'
              and torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb'
              and numpy.__version__ == '2.4.4' and not sys._is_gil_enabled(), 'framework identity differs')
    C.require(Path(torch_npu.__file__).resolve().parent == package
              and Path(torch_npu._C.__file__).resolve().parent == package, 'wrong candidate package imported')
    c_path = str(Path(torch_npu._C.__file__).resolve())
    C.require(c_path in config['pins'], 'actual _C binding is not pinned')
    verify_pins(config['pins'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    report['precision_before_setup'] = precision(torch)
    workload = load(config['workload'], 'r34_frozen_r31_workload')
    common = load(config['measurement'], 'r34_frozen_r26_measurement')
    report.update(policy=policy, environment={k: os.environ.get(k) for k in
        ('TASK_QUEUE_ENABLE', 'PER_STREAM_QUEUE', 'ASCEND_LAUNCH_BLOCKING', 'TORCH_NPU_EAGER_PREPARE_INGRESS')},
        runtime=dict(python=sys.version, torch=torch.__version__, torch_git=torch.version.git_version,
                     numpy=numpy.__version__, pta=torch_npu.__version__, device=torch.npu.get_device_name(0),
                     gil_enabled=sys._is_gil_enabled(), affinity=sorted(os.sched_getaffinity(0))),
        pins=config['pins'])
    configured = closed = False
    try:
        with torch.inference_mode():
            model, inputs, report['provenance'] = workload.load_original_model(config['upstream'])
            for key, digest in C.IDENTITY.items():
                C.require(report['provenance']['identity'].get(key) == digest, 'frozen model/input identity differs: ' + key)
            C.require(len(model.emb_l) == 26 and all(embedding.sparse is True for embedding in model.emb_l),
                      'original sparse=True embedding modules changed')
            expected = model(*inputs).clone()
            stages = workload.capture_stages(model, inputs)
            model = model.to('npu:0')
            inputs = (inputs[0].npu(), [v.npu() for v in inputs[1]], [v.npu() for v in inputs[2]])
            torch.npu.synchronize()
            report['configuration'] = prepare.configure(enabled=policy['enabled'], workers=policy['workers'],
                capacity=8, byte_limit=536870912, allowed_ops=('aclnnEmbeddingBag',), diagnostic_trace=False)
            configured = policy['enabled']
            report['correctness'] = []
            for _ in range(8):
                actual = model(*inputs)
                torch.npu.synchronize()
                error = workload.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
                report['correctness'].append(error)
                C.require(error['passed'], 'original full forward differs from CPU reference')
            report['stage_validation'] = workload.validate_stages(model, inputs, stages, torch.npu.synchronize)
            C.require(report['stage_validation']['passed'], 'original intermediate stage differs')
            report['precision_before_timing'] = precision(torch)
            report['mapping_before'] = mappings(config['pins'])
            measured, actual = common.measure_forward_window(model, inputs, torch.npu.synchronize,
                warmup=report['warmup'], samples=report['samples'], snapshot=prepare.snapshot)
            report.update(measured)
            report['numeric_final'] = workload.error_record(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
            C.require(report['numeric_final']['passed'], 'last timed output differs')
            report['native_timed_gate'] = C.native_delta(report['before'], report['after'], report['arm'], report['samples'])
            # Exactly one close attempt. Failed SDK retirement is not retried.
            closed = True
            report['closed'] = prepare.drain_close()
            report['close_gate'] = C.validate_close(report['closed'], report['arm'])
            report['precision_after_run'] = precision(torch)
        report['mapping_after'] = mappings(config['pins'])
        C.require(report['mapping_after'] == report['mapping_before'], 'mapped native/SDK providers changed')
        report['thread_affinities'] = common.thread_affinities(config['cpus'])
        C.require(not sys._is_gil_enabled(), 'GIL enabled during execution')
        verify_pins(config['pins'])
        report['comparison_identity'] = dict(pins=config['pins'], workload=C.IDENTITY,
            precision=report['precision_after_run'], runtime=report['runtime'], package=str(package),
            mappings=report['mapping_after'], capacity=8, byte_limit=536870912, sparse=True)
        report['timing'] = C.distribution(report['durations_ns'])
        if report['kind'] == 'perf':
            C.require(report['window_wall_ns'] >= 2_000_000_000, 'performance window below two seconds')
        report.update(passed=True, original_model_executed=True, compute_backend='real_a3_cann',
                      numerical_checks_in_timing=False, startup_included=False, gil_enabled=False)
    except BaseException:
        if configured and not closed:
            closed = True
            try:
                report['failure_close'] = prepare.drain_close()
            except BaseException as error:
                report['failure_close_error'] = dict(type=type(error).__name__, message=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', choices=C.ARMS, required=True)
    parser.add_argument('--kind', choices=('correctness', 'perf'), required=True)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--samples', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=50)
    args = parser.parse_args()
    C.require((args.warmup, args.samples) == ((50, 3000) if args.kind == 'perf' else (0, 3))
              and args.round >= 0, 'fixed R34 exposure required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, passed=False, arm=args.arm, kind=args.kind, round=args.round,
                  samples=args.samples, warmup=args.warmup, pid=os.getpid(), worker_sha256=sha(__file__))
    try:
        run(json.loads(args.config.read_text()), report)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(arm=args.arm, passed=True, **report['timing'])), flush=True)


if __name__ == '__main__':
    main()
