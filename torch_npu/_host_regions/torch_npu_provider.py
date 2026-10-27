"""C ABI bridge for independently certified TorchNPU capture modes.

The declarations apply only to NpuLoopPolicy-admitted intrinsic loops. They do
not certify arbitrary Torch operations. Real OpAPI requires a separate native
capability and creation entry; legacy HOSTSIM/builder ABI is not that evidence.
The native DSO captures owned commands, and the original queue consumes them.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import threading

from .backends import CaptureCapabilities, CapturedCommand, make_torch_npu_backend
from .contracts import RegionConfig, UnsupportedRegion, _NativeRetirementSession
from .integration import install_from_env
from .torch_npu_policy import NpuLoopPolicy, logical_stream_identity


_EMIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p)
_EMIT_BATCH = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p)
_MAX_TEXT = 4 * 1024 * 1024


def _native_run_enabled(settings):
    flag = settings.get('FTLAB_HOST_NATIVE_RUN', '0')
    if flag not in ('0', '1'):
        raise ValueError('FTLAB_HOST_NATIVE_RUN must be 0 or 1')
    return flag == '1'


def _native_builder_enabled(settings):
    flag = settings.get('HDB_REGION_NATIVE_BUILDER', '0')
    if flag not in ('0', '1'):
        raise ValueError('HDB_REGION_NATIVE_BUILDER must be 0 or 1')
    if flag == '1' and (not _native_run_enabled(settings) or settings.get('HDB_REGION_PACKET') != '1'):
        raise ValueError('native binding builder requires FTLAB_HOST_NATIVE_RUN=1 and HDB_REGION_PACKET=1')
    return flag == '1'


def _real_opapi_enabled(settings):
    flag = settings.get('HDB_REGION_REAL_OPAPI', '0')
    if flag not in ('0', '1'):
        raise ValueError('HDB_REGION_REAL_OPAPI must be 0 or 1')
    if flag == '1':
        if not _native_run_enabled(settings) or settings.get('HDB_REGION_PACKET') != '1':
            raise ValueError('real OpAPI requires FTLAB_HOST_NATIVE_RUN=1 and HDB_REGION_PACKET=1')
        if settings.get('QWEN_ACL_HOST_SIM') == '1' or settings.get('HDB_REGION_NATIVE_BUILDER', '0') != '0':
            raise ValueError('real OpAPI excludes HOSTSIM and the legacy native binding builder')
    return flag == '1'


class NativeBridge:
    """One loaded DSO. ctypes releases the calling thread state for C calls."""
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library)) if isinstance(library, (str, Path)) else library
        signatures = {
            'begin': (ctypes.c_int, [_EMIT, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]),
            'end': (ctypes.c_int, []),
            'publish': (ctypes.c_int, [ctypes.c_uint64]),
            'discard': (ctypes.c_int, [ctypes.c_uint64]),
            'stats': (ctypes.c_size_t, [ctypes.c_void_p, ctypes.c_size_t]),
            'error': (ctypes.c_size_t, [ctypes.c_void_p, ctypes.c_size_t]),
        }
        self._functions = {}
        for name, (result, arguments) in signatures.items():
            try:
                fn = getattr(self.library, 'hdb_region_' + name)
            except AttributeError as error:
                raise UnsupportedRegion('TorchNPU native capture ABI is unavailable: ' + name) from error
            fn.restype, fn.argtypes = result, arguments
            self._functions[name] = fn
        evidence = self.stats()
        if evidence.get('abi') != 1 or evidence.get('backend') != 'torch_npu_hostsim_capture':
            raise UnsupportedRegion('TorchNPU native capture ABI/backend mismatch')
        self.batch_available = type(evidence.get('batch_abi')) is int and evidence['batch_abi'] == 1
        if self.batch_available:
            for name, arguments in (
                    ('activate', []),
                    ('begin_batch', [_EMIT_BATCH, ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_uint64, ctypes.c_uint32])):
                try:
                    fn = getattr(self.library, 'hdb_region_' + name)
                except AttributeError as error:
                    raise UnsupportedRegion('TorchNPU declared incomplete batch ABI: ' + name) from error
                fn.restype, fn.argtypes = ctypes.c_int, arguments
                self._functions[name] = fn

    def _text(self, name):
        fn = self._functions[name]
        required = int(fn(None, 0))
        # Stats can grow concurrently between the size query and the copy.
        for _ in range(4):
            if not 1 <= required <= _MAX_TEXT:
                raise RuntimeError('native ' + name + ' text size is invalid or too large')
            buffer = ctypes.create_string_buffer(required)
            actual = int(fn(buffer, required))
            if not 1 <= actual <= _MAX_TEXT:
                raise RuntimeError('native ' + name + ' text size is invalid or too large')
            if actual <= required:
                return buffer.value.decode('utf-8', errors='strict')
            required = actual
        raise RuntimeError('native ' + name + ' text size did not stabilize')

    def error(self, action):
        return RuntimeError('native region ' + action + ' failed: ' + self._text('error'))

    def stats(self):
        result = json.loads(self._text('stats'))
        if type(result) is not dict:
            raise RuntimeError('native region stats must be a JSON object')
        return result

    def begin(self, callback, device, region):
        if self._functions['begin'](callback, None, device, region) != 0:
            raise self.error('begin')

    def activate(self):
        if self.batch_available and self._functions['activate']() != 0:
            raise self.error('activate')

    def begin_batch(self, callback, device, region, limit):
        if self._functions['begin_batch'](callback, None, device, region, limit) != 0:
            raise self.error('begin_batch')

    def end_error(self):
        return self.error('end') if self._functions['end']() != 0 else None

    def publish(self, token):
        # Native publication claims the token even on failed/partial acceptance.
        if self._functions['publish'](token) != 0:
            raise self.error('publish')

    def discard(self, token):
        if self._functions['discard'](token) != 0:
            raise self.error('discard')

    def enable_native_run(self):
        evidence = self.stats()
        if type(evidence.get('native_run_abi')) is not int or evidence['native_run_abi'] != 1:
            raise UnsupportedRegion('TorchNPU native-run requires declared ABI1')
        signatures = (
            ('run_create', ctypes.c_uint64, [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]),
            ('run_create_out', ctypes.c_int, [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]),
            ('run_begin', ctypes.c_int, [ctypes.c_uint64, ctypes.c_uint32]),
            ('run_done', ctypes.c_int, [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int]),
            ('run_drain', ctypes.c_int, [ctypes.c_uint64]),
            ('run_abort', ctypes.c_int, [ctypes.c_uint64]),
            ('run_destroy', ctypes.c_int, [ctypes.c_uint64]),
            ('run_destroy_out', ctypes.c_int, [ctypes.POINTER(ctypes.c_uint64)]),
        )
        for name, result, arguments in signatures:
            try:
                fn = getattr(self.library, 'hdb_region_' + name)
            except AttributeError as error:
                raise UnsupportedRegion('TorchNPU declared incomplete native-run ABI: ' + name) from error
            fn.restype, fn.argtypes = result, arguments
            self._functions[name] = fn


    def enable_native_binding_builder(self):
        evidence = self.stats()
        if type(evidence.get('native_builder_abi')) is not int or evidence['native_builder_abi'] != 1:
            raise UnsupportedRegion('TorchNPU native binding builder requires declared ABI1')
        try:
            fn = self.library.hdb_region_run_create_native_builder_out
        except AttributeError as error:
            raise UnsupportedRegion('TorchNPU native binding builder out-slot ABI is unavailable') from error
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        # This provider declares its audited intrinsic policy's plain-native
        # contract. Other native-run clients retain their original creation ABI.
        self._functions['run_create_out'] = fn


    def enable_real_opapi(self):
        evidence = self.stats()
        for name in ('real_capture_abi', 'opapi_owned_abi'):
            if type(evidence.get(name)) is not int or evidence[name] != 1:
                raise UnsupportedRegion('real OpAPI requires declared ABI1: ' + name)
        if evidence.get('real_capture_supported') is not True:
            raise UnsupportedRegion('native real OpAPI capability is unavailable')
        if evidence.get('real_prepare_strategy') not in ('consumer', 'worker'):
            raise UnsupportedRegion('unreviewed real OpAPI preparation strategy')
        try:
            fn = self.library.hdb_region_run_create_real_opapi_out
        except AttributeError as error:
            raise UnsupportedRegion('real OpAPI out-slot ABI is unavailable') from error
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        self._functions['run_create_out'] = fn
        self._requires_worker_tls_retirement = True
        return {name: evidence[name] for name in ('real_capture_abi', 'opapi_owned_abi',
                'real_capture_supported', 'real_prepare_strategy')}


class NativeRunSession(_NativeRetirementSession):
    """Opaque C-owned channels; no Python callback or command crosses into C++."""
    def __init__(self, native, regions, capacity, batch_limit, *, retire_worker_tls=None):
        self.native, self.closed = native, False
        self.retire_worker_tls = retire_worker_tls
        # Native writes ownership into a pre-held slot before returning. A
        # signal on FFI re-entry cannot hide a successfully created run id.
        self._slot = ctypes.c_uint64(0)
        self._dimensions, self._opened = (regions, capacity, batch_limit), False
        self._validate_retirement()

    @property
    def native_retirement_required(self):
        return isinstance(self.native, NativeBridge) and getattr(self.native, '_requires_worker_tls_retirement', False) is True

    def _validate_retirement(self):
        if self.native_retirement_required and (type(self.retire_worker_tls) is not _NativeWorkerRetirement
                or self.retire_worker_tls._native is not self.native):
            raise UnsupportedRegion('real native session requires its bridge-bound retirement capability before open')

    def open(self):
        if self.closed or self._opened:
            raise RuntimeError('native run session was already opened or closed')
        self._validate_retirement()
        self._opened = True
        status = self.native._functions['run_create_out'](*self._dimensions, ctypes.byref(self._slot))
        if status != 0 or not self.run:
            raise self.native.error('run_create')

    @property
    def run(self): return self._slot.value

    def _call(self, name, *args):
        if self.closed or not self.run:
            raise RuntimeError('native run session is not open')
        if self.native._functions[name](self.run, *args) != 0:
            raise self.native.error(name)

    def begin(self, region): self._call('run_begin', region)

    def end(self):
        error = self.native.end_error()
        if error is not None:
            raise error

    def done(self, region, failed): self._call('run_done', region, int(bool(failed)))

    def drain(self):
        if self.closed:
            raise RuntimeError('native run session is closed')
        status = int(self.native._functions['run_drain'](self.run))
        if status < 0:
            raise self.native.error('run_drain')
        return status

    def abort(self):
        if self.run:
            self._call('run_abort')

    def close(self):
        interrupted = None
        while not self.closed:
            if not self.run:
                self.closed = True
                break
            try:
                status = self.native._functions['run_destroy_out'](ctypes.byref(self._slot))
            except (KeyboardInterrupt, InterruptedError) as error:
                if interrupted is None:
                    interrupted = error
                else:
                    interrupted.add_note(f'native close interrupted again by {type(error).__name__}')
                continue
            if status != 0 or self.run:
                error = self.native.error('run_destroy')
                if interrupted is not None:
                    interrupted.add_note(f'native close failed: {type(error).__name__}: {error}')
                    raise interrupted from error
                raise error
            self.closed = True
        if interrupted is not None:
            raise interrupted


@dataclass(frozen=True)
class _CallerContext:
    device: int
    stream: object
    include: object
    exclude: object
    inference: bool


class _NativeWorkerRetirement:
    """Cold device0/default-stream barrier; never called by forward/map.

    The original synchronize(device) selects that explicit device, drains the
    original host queues and synchronizes the device before returning. Native
    closure checks then reject a cleanup fault or any still-live owned plan.
    """
    def __init__(self, npu, native, stream_identity):
        self._default_stream = npu.default_stream
        self._synchronize = npu.synchronize
        self._native, self.domain = native, stream_identity

    def __call__(self):
        if logical_stream_identity(self._default_stream(0)) != self.domain:
            raise RuntimeError('native retirement default-stream domain changed')
        if self._synchronize(0) is not None:
            raise RuntimeError('native retirement original synchronize did not return its expected ACK')
        evidence = self._native.stats()
        # The original C++ synchronize logs some host queue failures without
        # propagating them. A normal Python return is therefore insufficient.
        for name in ('owned_opapi_device_pending', 'owned_opapi_payloads_live', 'owned_opapi_retirement_failures',
                     'owned_opapi_retained_bytes', 'queue_pending_total', 'native_retirement_pending',
                     'packet_blocks_live', 'native_runs_live', 'failures'):
            if type(evidence.get(name)) is not int or evidence[name] != 0:
                raise RuntimeError('native retirement closure failed: '+name)
        return True


class NativeCaptureProvider:
    """A policy-gated provider, not a general-purpose Tensor capture mechanism."""
    capabilities = CaptureCapabilities(
        abi_version=1, backend='torch_npu',
        audit_id='hostsim-default-stream-embedding-loop-v1',
        owns_commands=True, captures_all_effects=True, restores_thread_state=True,
        rejects_unsupported_before_effect=True, preserves_owner_retirement=True,
        operation_policy=True, audited=True)

    def __init__(self, torch, torch_npu, *, library=None, batch_size=32, native_run=None, native_builder=None,
                 real_opapi=None):
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError('native batch size must be an integer in 1..256')
        if native_run is None:
            native_run = _native_run_enabled(os.environ)
        if type(native_run) is not bool:
            raise TypeError('native_run must be bool or None')
        if native_builder is None:
            native_builder = _native_builder_enabled(os.environ)
        if type(native_builder) is not bool:
            raise TypeError('native_builder must be bool or None')
        if native_builder and (not native_run or os.environ.get('HDB_REGION_PACKET') != '1'):
            raise ValueError('native binding builder requires native_run and HDB_REGION_PACKET=1')
        if real_opapi is None:
            real_opapi = _real_opapi_enabled(os.environ)
        if type(real_opapi) is not bool:
            raise TypeError('real_opapi must be bool or None')
        if real_opapi and (not native_run or native_builder or os.environ.get('HDB_REGION_PACKET') != '1'
                           or os.environ.get('HDB_REGION_REAL_OPAPI') != '1'
                           or os.environ.get('QWEN_ACL_HOST_SIM') == '1'):
            raise ValueError('real OpAPI requires its explicit native-run/packet mode and excludes legacy HOSTSIM builder')
        self.torch, self.torch_npu = torch, torch_npu
        if library is None:
            library = Path(torch_npu.__file__).resolve().parent / 'lib' / 'libtorch_npu.so'
        self.native = NativeBridge(library)
        self.batch_size = batch_size
        if native_run:
            self.native.enable_native_run()
        if native_builder:
            self.native.enable_native_binding_builder()
        self.real_capture_certificate = self.native.enable_real_opapi() if real_opapi else None
        self.real_opapi_enabled = real_opapi
        self.native_retirement_required = real_opapi
        self._retirement_barrier = None
        if real_opapi:
            self.capabilities = replace(self.capabilities, audit_id='real-opapi-default-stream-embedding-loop-r22')
        self.native_builder_enabled = native_builder
        self.native_run = self._create_native_run if native_run else None
        # Account for ordinary setup copies before the first capture scope.
        self.native.activate()
        self.policies = []
        # The runtime copies contextvars separately for each scheduled region.
        # No mutable shared "last caller TLS" crosses concurrent invocations.
        self._caller = ContextVar('torch_npu_host_region_caller_' + str(id(self)), default=None)

    def _create_native_run(self, regions, capacity):
        caller = self._caller.get()
        if caller is None:
            raise UnsupportedRegion('native run requires a successful invocation guard')
        barrier = None
        if self.real_opapi_enabled:
            if not isinstance(caller, _CallerContext) or type(caller.device) is not int or caller.device != 0:
                raise UnsupportedRegion('real native retirement requires the guarded device0 caller')
            domain = logical_stream_identity(caller.stream)
            if domain[1] != 0 or domain != logical_stream_identity(self.torch_npu.npu.default_stream(0)):
                raise UnsupportedRegion('real native retirement requires the guarded default-stream domain')
            barrier = getattr(self, '_retirement_barrier', None)
            if barrier is None:
                barrier = _NativeWorkerRetirement(self.torch_npu.npu, self.native, domain)
                self._retirement_barrier = barrier
            elif barrier.domain != domain:
                raise UnsupportedRegion('real native retirement domain changed across invocations')
        # Only native caller TLS enters the session. Clear Python stream/keyset
        # owners before runtime copies contextvars for the worker bodies.
        self._caller.set(None)
        return NativeRunSession(self.native, regions, capacity, min(capacity, self.batch_size), retire_worker_tls=barrier)

    def guard(self, policy, arguments):
        self._caller.set(None)
        if not policy.check(arguments):
            return self._decline('policy: ' + policy.last_reason)
        try:
            native = self.torch._C
            npu = self.torch_npu.npu
            context = _CallerContext(
                device=npu.current_device(), stream=npu.current_stream(),
                include=native._dispatch_tls_local_include_set(),
                exclude=native._dispatch_tls_local_exclude_set(),
                inference=self.torch.is_inference_mode_enabled())
            # InferenceMode reconstructs AutogradState; admit only its defaults.
            expected_grad_modes = not context.inference
            if (native._is_fwd_grad_enabled() is not expected_grad_modes
                    or native._is_multithreading_enabled() is not expected_grad_modes
                    or native._is_view_replay_enabled() is not False
                    or native._is_grad_layout_enforcement_enabled() is not True
                    or native._get_graph_exec_group() is not None):
                return self._decline('unsupported autograd/Torch TLS')
            current_identity = logical_stream_identity(context.stream)
            default_identity = logical_stream_identity(npu.default_stream(0))
            if (type(context.device) is not int or context.device != 0
                    or current_identity != default_identity or current_identity[1] != 0):
                return self._decline('current device/stream differs from admitted default stream')
            self._caller.set(context)
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            if getattr(self, 'real_opapi_enabled', False):
                raise UnsupportedRegion('real OpAPI invocation TLS guard failed before body') from error
            return False

    def _decline(self, reason):
        if getattr(self, 'real_opapi_enabled', False):
            raise UnsupportedRegion('real OpAPI invocation rejected before body: ' + reason)
        return False

    def stats(self):
        return self.native.stats()

    def bind_policy(self, plan):
        policy = NpuLoopPolicy(self.torch, self.torch_npu, plan, real_opapi=self.real_opapi_enabled)
        self.policies.append(policy)
        if not policy.accepted:
            return None
        return make_torch_npu_backend(self, guard=lambda arguments: self.guard(policy, arguments))

    @contextmanager
    def capture(self, region, sink):
        caller = self._caller.get()
        if caller is None:
            raise UnsupportedRegion('native region capture requires a successful invocation guard')
        if type(region) is not int or not 0 <= region < 2**64 or not callable(sink):
            raise ValueError('region must be a uint64 and sink callable')
        failures = []
        worker = threading.get_ident()

        def emit(token, units, _context):
            # No exception may cross the libffi callback boundary. The C side
            # discards any token still live after a rejected callback.
            try:
                if threading.get_ident() != worker:
                    raise RuntimeError('native command callback arrived on a different worker')
                if failures:
                    return -1
                if type(units) is not int or not 1 <= units <= self.batch_size:
                    raise ValueError('native batch command count is invalid')
                command = CapturedCommand(
                    lambda token=token: self.native.publish(token),
                    lambda token=token: self.native.discard(token),
                    owners=(self.native,), label='torch_npu:owned:' + str(token), units=units)
                sink(command)
                return 0
            except BaseException as error:
                failures.append(error.with_traceback(None))
                return -1

        batched = self.native.batch_available
        callback = _EMIT_BATCH(emit) if batched else _EMIT(lambda token, context: emit(token, 1, context))
        with (
            self.torch.inference_mode(caller.inference),
            self.torch.no_grad(),
            self.torch._C._ForceDispatchKeyGuard(caller.include, caller.exclude),
            self.torch_npu.npu.device(caller.device),
            self.torch_npu.npu.stream(caller.stream),
        ):
            if batched:
                self.native.begin_batch(callback, caller.device, region, self.batch_size)
            else:
                self.native.begin(callback, caller.device, region)
            body_error = None
            try:
                yield
            except BaseException as error:
                body_error = error
            # Native TLS closes before Torch stream/device/dispatch restoration.
            try:
                end_error = self.native.end_error()
            except BaseException as error:
                end_error = error
            primary = body_error if body_error is not None else failures[0] if failures else end_error
            if primary is not None:
                for extra in [*failures, end_error]:
                    if extra is not None and extra is not primary:
                        primary.add_note('native capture cleanup: ' + type(extra).__name__ + ': ' + str(extra))
                raise primary
        # ctypes callback finalization runs only after all native callbacks stop.
        del callback


class Registration:
    def __init__(self, torch, torch_npu, provider, installation, policies):
        self.torch, self.torch_npu = torch, torch_npu
        self.provider, self.installation, self.policies = provider, installation, policies
        self.closed = False

    @property
    def installed_count(self): return self.installation.installed_count

    @property
    def reports(self): return self.installation.reports

    def stats(self):
        return self.provider.stats() if self.provider is not None else {'available': False}

    def close(self):
        self.installation.close()
        self.closed = True


_registration = None
_registration_lock = threading.RLock()

# Descriptive public alias used by framework/experiment integrations.
TorchNpuHostSimProvider = NativeCaptureProvider


def get_registration():
    with _registration_lock:
        return _registration


def bootstrap(torch, torch_npu, *, env=None, library=None):
    """Framework initializer. Missing required native ABI is an explicit error.

    The disabled path does not load the native bridge or install a Module hook.
    Enabled policies are structural/intrinsic; no model name is consulted.
    """
    global _registration
    with _registration_lock:
        if _registration is not None and not _registration.closed:
            if _registration.torch is torch and _registration.torch_npu is torch_npu:
                return _registration
            raise RuntimeError('host-region bootstrap already belongs to another framework instance')
        config = RegionConfig.from_env(env)
        provider = None
        if config.enabled:
            settings = os.environ if env is None else env
            batch_size = int(settings.get('FTLAB_HOST_REGION_BATCH', '32'))
            if not 1 <= batch_size <= 256:
                raise ValueError('FTLAB_HOST_REGION_BATCH must be in 1..256')
            provider = NativeCaptureProvider(torch, torch_npu, library=library,
                batch_size=min(batch_size, config.commands_per_region),
                native_run=_native_run_enabled(settings), native_builder=_native_builder_enabled(settings),
                real_opapi=_real_opapi_enabled(settings))
        policies = provider.policies if provider is not None else []

        def factory(plan):
            return provider.bind_policy(plan) if provider is not None else None

        installation = install_from_env(torch, factory, env=env)
        _registration = Registration(torch, torch_npu, provider, installation, policies)
        return _registration
