"""Local synthetic provider contracts; no Torch, device or performance execution."""
from contextvars import ContextVar
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO_ROOT = Path(os.environ.get('HOST_REGIONS_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
ROOT = REPO_ROOT / 'torch_npu' / '_host_regions'
NAME = 'real_provider_r22_tests'
spec = importlib.util.spec_from_file_location(NAME, ROOT / '__init__.py',
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[NAME] = package
spec.loader.exec_module(package)
provider = importlib.import_module(NAME + '.torch_npu_provider')
policy_module = importlib.import_module(NAME + '.torch_npu_policy')
REAL_ENV = {'HDB_REGION_REAL_OPAPI': '1', 'FTLAB_HOST_NATIVE_RUN': '1', 'HDB_REGION_PACKET': '1'}


class FakeFunction:
    def __init__(self, function): self.function = function
    def __call__(self, *args): return self.function(*args)


class FakeLibrary:
    def __init__(self, changes=None, *, new_symbol=True):
        self.calls = []
        self.evidence = dict(abi=1, batch_abi=1, backend='torch_npu_hostsim_capture', native_run_abi=1,
            native_builder_abi=1, real_capture_abi=1, opapi_owned_abi=1,
            real_capture_supported=True, real_prepare_strategy='consumer')
        self.evidence.update(changes or {})
        names = ('begin','end','publish','discard','stats','error','activate','begin_batch',
                 'run_create','run_create_out','run_begin','run_done','run_drain',
                 'run_abort','run_destroy','run_destroy_out')
        if new_symbol: names += ('run_create_real_opapi_out',)
        for name in names:
            setattr(self, 'hdb_region_' + name, FakeFunction(lambda *args, name=name: self.call(name, *args)))

    def call(self, name, *args):
        self.calls.append(name)
        if name in ('stats', 'error'):
            text = (json.dumps(self.evidence) if name == 'stats' else 'synthetic error').encode() + b'\0'
            buffer, capacity = args
            if buffer is not None and capacity >= len(text): ctypes.memmove(buffer, text, len(text))
            return len(text)
        if name in ('run_create_out', 'run_create_real_opapi_out'):
            ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_uint64))[0] = 17
        if name == 'run_destroy_out': ctypes.cast(args[0], ctypes.POINTER(ctypes.c_uint64))[0] = 0
        return 0


class RealProviderTests(unittest.TestCase):
    def test_real_embedding_bag_rejects_mixed_index_types_before_body(self):
        class Tensor:
            def __init__(self, dtype, rank=1):
                self.dtype, self.rank = dtype, rank
                self.device = SimpleNamespace(type='npu', index=0)
                self.layout, self.requires_grad = 'strided', False
            def is_contiguous(self): return True
            def storage_offset(self): return 0
            def numel(self): return 8
            def is_conj(self): return False
            def is_neg(self): return False
            def dim(self): return self.rank
        class Parameter(Tensor): pass
        class EmbeddingBag:
            def __init__(self):
                self._parameters = {'weight': Parameter('f32', 2)}
                self._buffers, self._modules = {}, {}
                self.mode, self.max_norm, self.sparse = 'sum', None, True
                self.scale_grad_by_freq, self.include_last_offset = False, False
                self.padding_idx, self._compiled_call_impl = None, None
        torch = SimpleNamespace(Tensor=Tensor, nn=SimpleNamespace(Parameter=Parameter,
            EmbeddingBag=EmbeddingBag), strided='strided', float32='f32', int32='i32', int64='i64')
        policy = policy_module.NpuLoopPolicy.__new__(policy_module.NpuLoopPolicy)
        policy.torch, policy.real_opapi = torch, True
        for left, right in (('i32', 'i64'), ('i64', 'i32')):
            with self.subTest(left=left, right=right), self.assertRaisesRegex(
                    policy_module._Decline, 'matching indices/offsets dtypes'):
                policy._module(EmbeddingBag(), Tensor(left), Tensor(right), None)
        for dtype in ('i32', 'i64'):
            self.assertIs(policy._module(EmbeddingBag(), Tensor(dtype), Tensor(dtype), None),
                          policy_module._RESULT)
        policy.real_opapi = False
        self.assertIs(policy._module(EmbeddingBag(), Tensor('i32'), Tensor('i64'), None),
                      policy_module._RESULT)

    def real_provider(self, library=None):
        library = library or FakeLibrary()
        with patch.dict(provider.os.environ, REAL_ENV, clear=True):
            obj = provider.NativeCaptureProvider(SimpleNamespace(), SimpleNamespace(), library=library,
                                                 native_run=True, real_opapi=True)
        return obj, library

    def test_real_flag_requires_native_packet_and_rejects_simulator_or_legacy_builder(self):
        self.assertTrue(hasattr(provider, '_real_opapi_enabled'), 'new real OpAPI selection is missing')
        self.assertFalse(provider._real_opapi_enabled({}))
        self.assertTrue(provider._real_opapi_enabled(REAL_ENV))
        for changes in ({'HDB_REGION_REAL_OPAPI': 'yes'}, {'FTLAB_HOST_NATIVE_RUN': '0'},
                        {'HDB_REGION_PACKET': '0'}, {'QWEN_ACL_HOST_SIM': '1'},
                        {'HDB_REGION_NATIVE_BUILDER': '1'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                provider._real_opapi_enabled({**REAL_ENV, **changes})

    def test_real_mode_selects_only_new_outslot_after_native_capability(self):
        self.assertTrue(hasattr(provider.NativeBridge, 'enable_real_opapi'), 'real native capability gate missing')
        obj, lib = self.real_provider()
        self.assertTrue(obj.real_opapi_enabled)
        self.assertIs(obj.native._functions['run_create_out'], lib.hdb_region_run_create_real_opapi_out)
        # Real sessions now require the actual guarded device/default-stream
        # shape even though this test only fakes the native out-slot boundary.
        stream = SimpleNamespace(stream_id=0, device_index=0, device_type=20)
        obj.torch_npu.npu = SimpleNamespace(default_stream=lambda _device: stream,
                                             synchronize=lambda _device: None)
        obj._caller.set(provider._CallerContext(0, stream, None, None, True))
        session = obj.native_run(4, 32); session.open(); session.close()
        self.assertEqual(lib.calls.count('run_create_real_opapi_out'), 1)
        self.assertNotIn('run_create_out', lib.calls)
        self.assertEqual(session.run, 0)

    def test_missing_or_inexact_native_certificate_never_activates(self):
        self.assertTrue(hasattr(provider.NativeBridge, 'enable_real_opapi'), 'real native capability gate missing')
        for changes in ({'real_capture_abi': True}, {'real_capture_abi': 0}, {'opapi_owned_abi': 1.0},
                        {'real_capture_supported': False}, {'real_capture_supported': 1},
                        {'real_prepare_strategy': 'unknown'}, {'real_prepare_strategy': None}):
            lib = FakeLibrary(changes)
            with self.subTest(changes=changes), self.assertRaises(provider.UnsupportedRegion):
                self.real_provider(lib)
            self.assertNotIn('activate', lib.calls)
            self.assertNotIn('run_create_out', lib.calls)
        lib = FakeLibrary(); del lib.evidence['real_capture_abi']
        with self.assertRaises(provider.UnsupportedRegion): self.real_provider(lib)
        lib = FakeLibrary(new_symbol=False)
        with self.assertRaises(provider.UnsupportedRegion): self.real_provider(lib)

    def test_declared_worker_strategy_is_explicitly_distinct_from_consumer(self):
        try: obj,_=self.real_provider(FakeLibrary({'real_prepare_strategy':'worker'}))
        except provider.UnsupportedRegion: self.fail('native-certified worker strategy was incorrectly rejected')
        self.assertEqual(obj.real_capture_certificate['real_prepare_strategy'],'worker')

    def test_default_legacy_mode_does_not_promote_from_native_real_support(self):
        lib = FakeLibrary()
        with patch.dict(provider.os.environ, {}, clear=True):
            obj = provider.NativeCaptureProvider(SimpleNamespace(), SimpleNamespace(), library=lib, native_run=True)
        self.assertIs(obj.native._functions['run_create_out'], lib.hdb_region_run_create_out)
        self.assertFalse(getattr(obj, 'real_opapi_enabled', False))

    def test_explicit_real_guard_rejection_raises_before_any_body_instead_of_fallback(self):
        self.assertTrue(hasattr(provider.NativeBridge, 'enable_real_opapi'), 'real native capability gate missing')
        obj, lib = self.real_provider()
        policy = SimpleNamespace(check=lambda arguments: False, last_reason='synthetic unknown effect')
        with self.assertRaisesRegex(provider.UnsupportedRegion, 'synthetic unknown effect'):
            obj.guard(policy, {})
        self.assertIsNone(obj._caller.get())
        self.assertNotIn('run_create_real_opapi_out', lib.calls)

    def test_caller_snapshot_retains_exact_inference_state(self):
        self.assertIn('inference', provider._CallerContext.__dataclass_fields__,
                      'guard passes inference but the caller snapshot drops it')
        context = provider._CallerContext(device=0, stream=object(), include=object(),
                                          exclude=object(), inference=True)
        self.assertIs(context.inference, True)

    def test_valid_inference_guard_does_not_silently_decline_on_context_constructor(self):
        stream = SimpleNamespace(stream_id=0, device_index=0, device_type=20)
        native = SimpleNamespace(
            _dispatch_tls_local_include_set=lambda: object(),
            _dispatch_tls_local_exclude_set=lambda: object(),
            _is_fwd_grad_enabled=lambda: False, _is_multithreading_enabled=lambda: False,
            _is_view_replay_enabled=lambda: False,
            _is_grad_layout_enforcement_enabled=lambda: True,
            _get_graph_exec_group=lambda: None)
        obj = provider.NativeCaptureProvider.__new__(provider.NativeCaptureProvider)
        obj.torch = SimpleNamespace(_C=native, is_inference_mode_enabled=lambda: True)
        obj.torch_npu = SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 0,
            current_stream=lambda: stream, default_stream=lambda device: stream))
        obj._caller = ContextVar('synthetic_r22_caller', default=None)
        self.assertTrue(obj.guard(SimpleNamespace(check=lambda arguments: True), {}),
                        'valid inference guard silently fell back')
        self.assertIs(obj._caller.get().inference, True)


if __name__ == '__main__':
    unittest.main()
