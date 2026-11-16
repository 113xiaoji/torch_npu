"""Complete-model closed-loop clients; CPU and real-NPU evidence stay separate."""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
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


HERE = Path(__file__).resolve().parent
C = load(HERE / 'contracts.py', '_r35_contracts_runtime')
E = load(HERE / 'engine.py', '_r35_engine_runtime')
R = load(HERE / 'requests.py', '_r35_requests_runtime')
P = load(HERE / 'policy.py', '_r38_queue_policy_runtime')
N = load(HERE / 'native_policy.py', '_r41_native_policy_runtime')


def validate_gil(mode, enabled, build_disabled, options, environment):
    C.require(mode in ('off', 'on') and type(enabled) is bool, 'invalid GIL mode')
    C.require(type(build_disabled) is int and build_disabled == 1, 'same free-threaded build required')
    C.require('PYTHON_GIL' not in environment, 'ambiguous PYTHON_GIL environment is forbidden')
    expected = mode == 'on'
    C.require(enabled is expected, 'actual GIL state differs')
    C.require(options.get('gil') == '1' if expected else 'gil' not in options,
              'on requires -Xgil=1; off must be natural, not forced')
    return expected


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pins(pins):
    C.require(type(pins) is dict and bool(pins), 'explicit file pins required')
    for path, digest in pins.items():
        C.require(Path(path).is_absolute() and C.sha256(digest) and sha(path) == digest,
                  'pinned file changed: ' + path)


def source_identity(config):
    pins = config.get('source_pins', {})
    verify_pins(pins)
    root = Path(config['source_root']).resolve()
    values = sorted((Path(path).resolve().relative_to(root).as_posix(), digest) for path, digest in pins.items())
    required = {str((HERE / name).resolve()) for name in ('real_service.py', 'engine.py', 'contracts.py', 'requests.py', 'policy.py', 'native_policy.py')}
    C.require(required <= set(pins), 'runner source closure missing')
    digest = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    C.require(digest == config.get('source_sha256'), 'source closure identity changed')
    return digest


def native_mappings(config):
    pins = config.get('provider_pins', {})
    C.require(bool(pins), 'NPU provider pins required')
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5]:
            path = str(Path(parts[5]).resolve(strict=True))
            if '/Ascend/' in path or Path(path).name.startswith(
                    ('libtorch_npu', 'libopapi', 'libcust_opapi', 'libnnopbase', 'libaclnn')):
                paths.add(path)
    C.require(any(Path(p).name == 'libtorch_npu.so' for p in paths), 'real PTA mapping absent')
    C.require(any(Path(p).name.startswith('libascendcl.so') for p in paths), 'real AscendCL absent')
    C.require(not any(x in p.lower() for p in paths for x in ('hostsim', 'phase_api_tap', 'sdk-007')),
              'simulation/interposer in real NPU process')
    for path in paths:
        C.require(path in pins and sha(path) == pins[path], 'unexpected/changed NPU provider: ' + path)
    return [dict(path=path, sha256=pins[path]) for path in sorted(paths)]


def quiesce(torch, prepare, timeout=10):
    # All client threads are paused or joined. Never called inside request loop.
    torch.npu.synchronize()
    deadline = time.monotonic() + timeout
    while True:
        snap = prepare.snapshot()
        if snap.get('first_error') is not None or snap['queue']['quarantined']:
            raise RuntimeError('native preparation domain failed: ' + json.dumps(snap['first_error']))
        if snap['queue']['slots_in_use'] == snap['queue']['retained_bytes'] == 0:
            return snap
        if time.monotonic() >= deadline:
            raise TimeoutError('device completed but native retirement did not quiesce')
        time.sleep(.001)


def close_native_domain(prepare, native_contracts, backend, report, *, failure=False):
    """Attempt native close once; only a validated snapshot acknowledges it."""
    if report.get('native_close_attempted'):
        raise RuntimeError('native close was already attempted; SDK close cannot be retried')
    report['native_close_attempted'] = True
    snapshot_key = 'failure_close' if failure else 'closed'
    gate_key = 'failure_close_gate' if failure else 'close_gate'
    try:
        report[snapshot_key] = prepare.drain_close()
        report[gate_key] = native_contracts.validate_close(report[snapshot_key], backend)
        C.require(type(report[gate_key]) is dict and report[gate_key].get('passed') is True,
                  'native close acknowledgement missing')
    except BaseException:
        report['requires_process_termination'] = True
        raise
    report['native_close_acknowledged'] = True
    return report[snapshot_key]


def persist_report(output, report, *, terminate=os._exit):
    """Persist best-effort evidence and bypass unsafe interpreter finalization."""
    try:
        (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    finally:
        if report.get('requires_process_termination'):
            terminate(2)


def run(config, report):
    treatment = N.expected_treatment(config, os.environ)
    report['native_sha256'] = config['native_sha256']
    report['r41_treatment_requested'] = treatment
    C.validate_case(report)
    report['per_stream_queue_requested'] = P.validate_environment(config, report, os.environ)
    report['queue_policy_environment_before'] = P.environment_receipt(os.environ)
    validate_gil(report['gil_mode'], sys._is_gil_enabled(),
                 sysconfig.get_config_var('Py_GIL_DISABLED'), sys._xoptions, os.environ)
    C.require(sys.version_info[:3] == (3, 14, 6) and sys.platform == 'linux'
              and platform.machine() == 'aarch64', 'pinned AArch643.14.6t runtime required')
    C.require(sys.getprofile() is None and sys.gettrace() is None, 'Python profiler not allowed')
    for key in ('QWEN_ACL_HOST_SIM', 'LD_PRELOAD', 'LD_AUDIT'):
        C.require(key not in os.environ, 'simulation or interposer forbidden: ' + key)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        C.require(os.environ.get(key) == '1', 'CPU thread budget differs: ' + key)
    C.require(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'backend autoload must be explicit')
    C.require(config['cpus'] == list(range(8, 16)), 'fixed CPU8..15 budget required')
    os.sched_setaffinity(0, config['cpus'])
    verify_pins(config['pins'])
    source_sha = source_identity(config)
    executable = str(Path(sys.executable).resolve())
    python_sha = sha(executable)
    C.require(python_sha == config['python_sha256'], 'interpreter binary differs')
    os.environ['TORCH_NPU_EAGER_PREPARE_INGRESS'] = 'mpsc' if report['backend'] == 'mpsc4' else 'mutex'
    os.environ['TORCH_NPU_EAGER_PREPARE_MULTI_CALLER'] = '1'
    import numpy
    import torch
    C.require(torch.__version__ == '2.13.0+cpu' and numpy.__version__ == '2.4.4'
              and torch.version.git_version == 'cf30153c4c131c8164ee7798e5022d810682e2cb',
              'framework identity changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before_gil = sys._is_gil_enabled()
    expected_gil = validate_gil(report['gil_mode'], before_gil,
        sysconfig.get_config_var('Py_GIL_DISABLED'), sys._xoptions, os.environ)
    prepare = native_contracts = None
    configured = False
    if report['device'] == 'npu':
        P.validate_environment(config, report, os.environ)
        import torch_npu
        from torch_npu import _eager_prepare as prepare
        package = Path(config['package']).resolve()
        C.require(Path(torch_npu.__file__).resolve().parent == package, 'wrong private NPU package')
        report['provider_pins'] = dict(config['provider_pins'])
        C.require(sha(package / 'lib/libtorch_npu.so') ==
                  config['native_sha256'], 'pinned R41 native candidate required')
        native_contracts = load(config['native_contracts'], '_r35_frozen_native_contracts')
        native_library = (package / 'lib/libtorch_npu.so').resolve()
        report['queue_policy_library'] = dict(path=str(native_library), sha256=sha(native_library))
        report['queue_policy_before'] = P.read_native_options(native_library)
        native_treatment = N.NativeTreatment(native_library, config, os.environ)
        report['r41_treatment_before'] = native_treatment.read()
        report['r41_workspace_callers'] = [None] * report['callers']
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    elif 'torch_npu' in sys.modules:
        raise RuntimeError('CPU evidence must not import the NPU backend')
    workload = load(config['workload'], '_r35_frozen_model_workload')
    try:
        with torch.inference_mode():
            model, template, provenance = workload.load_original_model(config['upstream'])
            C.require(all(provenance['identity'].get(k) == v for k, v in C.IDENTITY.items()),
                      'frozen model/state/template changed')
            C.require(len(model.emb_l) == 26 and all(m.sparse for m in model.emb_l), 'model changed')
            specs = R.build_request_bank(template[0].tolist(), [x.tolist() for x in template[1]],
                                         [x.tolist() for x in template[2]])
            bank = [(torch.tensor(row['dense'], dtype=torch.float32),
                     [torch.tensor(x, dtype=torch.int64) for x in row['offsets']],
                     [torch.tensor(x, dtype=torch.int64) for x in row['indices']]) for row in specs]
            input_hashes = [workload.helper.tensor_tree_sha256(row, torch) for row in bank]
            C.require(input_hashes[0] == C.IDENTITY['input_sha256'] and len(set(input_hashes)) == 8,
                      'request bank must retain template0 and contain eight distinct requests')
            references = [model(*args).clone() for args in bank]
            initial_state = workload.helper.tensor_tree_sha256(dict(model.state_dict()), torch)
            C.require(initial_state == C.IDENTITY['state_sha256'], 'model weight state changed')
            device = 'cpu' if report['device'] == 'cpu' else 'npu:0'
            model = model.to(device).eval()
            model.requires_grad_(False)
            slots = [[(row[0].to(device, copy=True), [t.to(device, copy=True) for t in row[1]],
                       [t.to(device, copy=True) for t in row[2]]) for row in bank]
                     for _ in range(report['callers'])]
            # Independent client inputs, including physically distinct sparse/dense storage.
            for input_id in range(8):
                for tensor_id in range(53):
                    values = []
                    for caller in slots:
                        args = caller[input_id]
                        tensors = [args[0]] + args[1] + args[2]
                        if tensors[tensor_id].numel(): values.append(tensors[tensor_id].data_ptr())
                    C.require(len(values) == len(set(values)), 'clients share nonempty input storage')
            streams = None
            if prepare:
                torch.npu.synchronize()
                streams = ([torch.npu.Stream(device=0) for _ in range(report['callers'])]
                           if report['stream_mode'] == 'private' else
                           [torch.npu.default_stream(0) for _ in range(report['callers'])])
                enabled = report['backend'] != 'off'
                report['prepare_configuration'] = prepare.configure(enabled=enabled, workers=4, capacity=8,
                    byte_limit=536870912, allowed_ops=('aclnnEmbeddingBag',), diagnostic_trace=False)
                configured = enabled
                if enabled: C.require(report['prepare_configuration']['real_sdk_qualified'], 'SDK unqualified')
                report['private_stream_ids'] = [int(stream.stream_id) for stream in streams]
                report['queue_topology_baseline'] = P.process_snapshot()

            class Session:
                def __init__(self, index):
                    self.inputs = slots[index]
                    self.event = torch.npu.Event(enable_timing=False) if prepare else None
                def forward(self, request_id):
                    return model(*self.inputs[request_id % 8])
                def complete(self, output):
                    if self.event is not None:
                        self.event.record(torch.npu.current_stream(0))
                        self.event.synchronize()

            @contextmanager
            def factory(index):
                with torch.inference_mode():
                    if prepare: torch.npu.set_device(0)
                    context = torch.npu.stream(streams[index]) if prepare else nullcontext()
                    with context:
                        try:
                            yield Session(index)
                        finally:
                            report['r41_workspace_callers'][index] = native_treatment.counters()

            def before_timing():
                report['before'] = quiesce(torch, prepare) if prepare else None
                if prepare: report['mapping_before'] = native_mappings(config)
                report['queue_topology_before_timing'] = P.process_snapshot()
                report['queue_topology_pre_gate'] = P.validate_topology(
                    report['queue_topology_baseline'], report['queue_topology_before_timing'],
                    report['per_stream_queue_requested'], report['callers'],
                    initialized=report['warmup_per_caller'] > 0)

            result = E.run_closed_loop(concurrency=report['callers'], total_requests=report['total_requests'],
                warmup_per_worker=report['warmup_per_caller'], session_factory=factory,
                before_timing=before_timing, timeout_seconds=config.get('request_timeout_seconds', 180))
            report['queue_topology_after_timing'] = P.process_snapshot()
            report['queue_topology_post_gate'] = P.validate_topology(
                report['queue_topology_baseline'], report['queue_topology_after_timing'],
                report['per_stream_queue_requested'], report['callers'], initialized=True)
            report['queue_policy_after'] = P.read_native_options(native_library)
            report['r41_treatment_after'] = native_treatment.read()
            C.require(report['r41_treatment_after'] == report['r41_treatment_before'],
                      'R41 native treatment changed during execution')
            report['r41_workspace_gate'] = N.audit_callers(report['r41_workspace_callers'],
                report['callers'], treatment['workspace_cache'])
            report['after'] = quiesce(torch, prepare) if prepare else None
            report.update(start_ns=result.start_ns, end_ns=result.end_ns,
                requests=[dict(asdict(row), input_id=row.request_id % 8) for row in result.requests],
                workers=[asdict(row) for row in result.workers])
            if prepare:
                report['native_gate'] = native_contracts.native_delta(report['before'], report['after'],
                    {'off': 'off', 'mutex4': 'mutex4', 'mpsc4': 'mpsc4'}[report['backend']], report['total_requests'])
            total = len(result.outputs)
            finite = shapes = compared = 0
            maximum = 0.0
            expected_shape = list(references[0].shape)
            for request_id, output in enumerate(result.outputs):
                C.require(output.device.type == report['device'], 'wrong execution device')
                actual = output.cpu()
                if list(actual.shape) == expected_shape: shapes += 1
                if bool(torch.isfinite(actual).all()): finite += 1
                torch.testing.assert_close(actual, references[request_id % 8], rtol=1e-4, atol=1e-5)
                compared += 1
                maximum = max(maximum, float((actual - references[request_id % 8]).abs().max()))
            report['numeric_validation'] = dict(passed=finite == shapes == compared == total,
                validated_requests=total, finite_count=finite, shape_count=shapes,
                reference_compared_count=compared, expected_shape=expected_shape,
                max_abs_error=maximum, rtol=1e-4, atol=1e-5)
            C.require(workload.helper.tensor_tree_sha256(dict(model.state_dict()), torch) == initial_state,
                      'shared inference mutated model weights')
            if prepare:
                close_native_domain(prepare, native_contracts, report['backend'], report)
                report['mapping_after'] = native_mappings(config)
                C.require(report['mapping_after'] == report['mapping_before'], 'provider mappings changed')
        after_gil = sys._is_gil_enabled()
        validate_gil(report['gil_mode'], after_gil, sysconfig.get_config_var('Py_GIL_DISABLED'), sys._xoptions, os.environ)
        report.update(gil=dict(expected=expected_gil, before=before_gil, after=after_gil, build_disabled=1),
            execution_device=report['device'], actual_device_type=result.outputs[0].device.type,
            execution_policy=dict(closed_loop=True, shared_model=True, eval_mode=True,
                independent_input_slots=True, event_completion=prepare is not None,
                device_synchronize_in_request_loop=False, outputs_validated_outside_window=True,
                prepare_multi_caller=1),
            comparison_identity=dict(C.IDENTITY, input_hashes=input_hashes,
                input_bank_sha256=hashlib.sha256(json.dumps(input_hashes, separators=(',', ':')).encode()).hexdigest(),
                python_executable=executable, python_sha256=python_sha, source_sha256=source_sha),
            runtime=dict(python=sys.version, torch=torch.__version__, torch_git=torch.version.git_version,
                numpy=numpy.__version__, affinity=sorted(os.sched_getaffinity(0)),
                framework_intra_op=torch.get_num_threads(), framework_inter_op=torch.get_num_interop_threads()),
            original_full_model_executed=True, network_and_preprocessing_included=False,
            kernel_overlap_proven=False, shared_weights_mutated=False, passed=True)
        verify_pins(config['pins'])
        source_identity(config)
        report['queue_policy_environment_after'] = P.environment_receipt(os.environ)
        report['queue_policy_gate'] = P.audit_policy_receipt(report, expected_native_sha256=config['native_sha256'])
        report['metrics'] = C.request_metrics(report)
        C.audit_result(report)
    except BaseException as error:
        unfinished = getattr(error, 'unfinished_worker_indices', ())
        if unfinished:
            report['unsafe_active_callers'] = list(unfinished)
            report['requires_process_termination'] = True
        elif configured and not report.get('native_close_attempted'):
            try: close_native_domain(prepare, native_contracts, report['backend'], report, failure=True)
            except BaseException as close_error: report['failure_close_error'] = repr(close_error)
        raise


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key, choices in (('device', ('cpu', 'npu')), ('gil-mode', ('off', 'on')),
                         ('stream-mode', ('cpu', 'shared', 'private')), ('backend', ('off', 'mutex4', 'mpsc4')),
                         ('kind', ('correctness', 'perf'))):
        parser.add_argument('--' + key, choices=choices, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--callers', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--total-requests', type=int, required=True)
    parser.add_argument('--warmup-per-caller', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {key: getattr(args, key) for key in C.TASK_KEYS}
    report.update(schema=1, passed=False, pid=os.getpid())
    code = 0
    try:
        run(json.loads(args.config.read_text()), report)
    except BaseException as error:
        report['passed'] = False
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        code = 1
    finally:
        persist_report(args.output, report)
    if code:
        print(json.dumps(report['error']), file=sys.stderr, flush=True)
        raise SystemExit(code)
    print(json.dumps(dict(case_id=report['case_id'], passed=True, metrics=report['metrics'])), flush=True)


if __name__ == '__main__':
    main()
