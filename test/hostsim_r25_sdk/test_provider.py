"""Portable typed ABI/lifetime contract tests; no torch or CANN dependency."""
import ctypes as C
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "tools/hostsim_r25/sdk"


class ProviderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This assertion is also the first TDD red gate: the isolated provider
        # does not exist in the R24 base checkout.
        if not (SDK / "r25_hostsim_opapi.cpp").is_file():
            return
        spec = importlib.util.spec_from_file_location("build_provider", SDK / "build_provider.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.lib = C.CDLL(str(module.build_provider()))
        cls.lib.R25HostSimAbi.restype = C.c_int
        cls.lib.R25HostSimBackendKind.restype = C.c_char_p
        cls.lib.R25HostSimDiagnosticsJson.restype = C.c_char_p
        cls.lib.R25HostSimSetTracing.argtypes = [C.c_bool]
        cls.lib.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int,
            C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        cls.lib.aclCreateTensor.restype = C.c_void_p
        cls.lib.aclDestroyTensor.argtypes = [C.c_void_p]
        cls.lib.aclnnReluGetWorkspaceSize.argtypes = [C.c_void_p, C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        cls.lib.aclnnRelu.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        cls.lib.R25HostSimExecutorId.argtypes = [C.c_void_p]
        cls.lib.R25HostSimExecutorId.restype = C.c_uint64

    def setUp(self):
        if hasattr(self, "lib"):
            self.assertEqual(self.lib.R25HostSimResetDiagnostics(), 0)
            self.assertEqual(self.lib.R25HostSimSetTracing(False), 0)

    def snapshot(self):
        return json.loads(self.lib.R25HostSimDiagnosticsJson())

    def make_tensor(self):
        dims = (C.c_int64 * 2)(2, 4)
        strides = (C.c_int64 * 2)(4, 1)
        storage = (C.c_int64 * 1)(8)
        result = self.lib.aclCreateTensor(dims, 2, 0, strides, 0, 2, storage, 1, 4096)
        self.assertTrue(result)
        return result

    def prepare(self, source, output):
        workspace, executor = C.c_uint64(123), C.c_void_p()
        self.assertEqual(self.lib.aclnnReluGetWorkspaceSize(source, output,
            C.byref(workspace), C.byref(executor)), 0)
        self.assertEqual(workspace.value, 0)
        return executor

    def test_isolated_provider_exists(self):
        self.assertTrue((SDK / "r25_hostsim_opapi.cpp").is_file(), "R25 typed host-sim provider missing")

    def test_marker_explicitly_excludes_numeric_and_real_sdk_claims(self):
        self.assertTrue(hasattr(self, "lib"), "R25 typed host-sim provider missing")
        self.assertEqual(self.lib.R25HostSimAbi(), 1)
        self.assertEqual(self.lib.R25HostSimBackendKind(), b"HOST_SIM_R25")
        info = json.loads(self.lib.R25HostSimDiagnosticsJson())
        self.assertFalse(info["numeric_supported"])
        self.assertFalse(info["real_sdk_qualified"])
        self.assertFalse(info["pta_cache_supported"])
        self.assertEqual(info["artificial_prepare_delay_ns"], 0)

    def test_worker_prepare_and_caller_execute_order_are_observable(self):
        self.assertEqual(self.lib.R25HostSimSetTracing(True), 0)
        source, output = self.make_tensor(), self.make_tensor()
        executors, worker_tids = [], []
        def worker():
            worker_tids.append(threading.get_native_id())
            executors.append(self.prepare(source, output))
        try:
            for _ in range(2):
                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
            ids = [self.lib.R25HostSimExecutorId(item) for item in executors]
            self.assertEqual(len(set(ids)), 2)
            for executor in reversed(executors):
                self.assertEqual(self.lib.aclnnRelu(None, 0, executor, None), 0)
            info = self.snapshot()
            prepares = [item for item in info["events"] if item["phase"] == "prepare"]
            executes = [item for item in info["events"] if item["phase"] == "execute"]
            self.assertEqual([item["executor_id"] for item in prepares], ids)
            self.assertEqual([item["executor_id"] for item in executes], ids[::-1])
            self.assertEqual([item["tid"] for item in prepares], worker_tids)
            self.assertEqual(info["execute_tids"], [threading.get_native_id()])
            self.assertEqual(info["executors_created"], info["executors_executed"])
            self.assertEqual(info["live_executors"], 0)
            self.assertEqual(info["errors"], 0)
        finally:
            self.assertEqual(self.lib.aclDestroyTensor(source), 0)
            self.assertEqual(self.lib.aclDestroyTensor(output), 0)
        self.assertEqual(self.lib.R25HostSimResetDiagnostics(), 0)

    def test_legacy_inactive_cache_cleanup_is_harmless(self):
        self.lib.UnInitPTACacheThreadLocal()
        info = self.snapshot()
        self.assertFalse(info["pta_cache_supported"])
        self.assertEqual(info["cache_stub_calls"], 0)
        self.assertEqual(info["errors"], 0)
        self.assertEqual(info["cache_cleanup_calls"], 1)

    def test_ordinary_nullable_descriptor_cleanup_is_a_noop(self):
        # Ordinary OpAPI ReleaseConvertTypes releases absent optional arguments
        # as typed nullptr handles. These are not allocated/stale descriptors.
        before = self.snapshot()
        for name in ("aclDestroyTensor", "aclDestroyScalar", "aclDestroyIntArray",
                     "aclDestroyFloatArray", "aclDestroyBoolArray",
                     "aclDestroyTensorList", "aclDestroyScalarList"):
            with self.subTest(name=name):
                destroy = getattr(self.lib, name)
                destroy.argtypes = [C.c_void_p]
                self.assertEqual(destroy(None), 0)
                self.assertEqual(destroy(None), 0)
        self.assertEqual(self.snapshot(), before)

    def test_actual_ordinary_release_tuple_accepts_absent_optionals(self):
        # Compile the real provider with the exact ordinary tuple cleanup, so
        # this covers the OFF path rather than OwnedConvertPolicy's null guard.
        source = (ROOT / "patches/eager_prepare_r25/op-plugin/op_plugin/utils/op_api_common_base.h").read_text(encoding="utf-8")
        start = source.index("inline void Release(aclTensor *p)")
        end = source.index("template <typename... Ts> constexpr auto ConvertTypes", start)
        spec = importlib.util.spec_from_file_location("_r25_native_compile", ROOT / "test/host_regions/test_eager_prepare_integration.py")
        integration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integration)
        fixture = r'''
#define NOMINMAX
#include "/*PROVIDER*/"
#include <cassert>
#include <tuple>
#include <utility>
void* resolve(const char* name) {
#define RESOLVE(symbol) if (std::strcmp(name,#symbol)==0) return reinterpret_cast<void*>(&::symbol)
    RESOLVE(aclDestroyTensor); RESOLVE(aclDestroyScalar); RESOLVE(aclDestroyIntArray);
    RESOLVE(aclDestroyBoolArray); RESOLVE(aclDestroyTensorList); RESOLVE(aclDestroyScalarList);
#undef RESOLVE
    return nullptr;
}
#define GET_OP_API_FUNC(name) reinterpret_cast<decltype(&::name)>(resolve(#name))
/*RELEASE*/
int main() {
    assert(R25HostSimResetDiagnostics()==0);
    assert(InitHugeMemThreadLocal(nullptr,false)==0);
    int64_t one=1;
    auto* tensor=aclCreateTensor(&one,1,ACL_FLOAT,&one,0,ACL_FORMAT_ND,&one,1,reinterpret_cast<void*>(0x1000));
    assert(tensor);
    uint64_t ws=123; aclOpExecutor* executor=nullptr;
    assert(aclnnReluGetWorkspaceSize(tensor,tensor,&ws,&executor)==0);
    assert(ws==0 && aclnnRelu(nullptr,ws,executor,nullptr)==0);
    auto parameters=std::make_tuple(tensor,static_cast<aclTensor*>(nullptr),static_cast<aclScalar*>(nullptr),
        static_cast<aclIntArray*>(nullptr),static_cast<aclBoolArray*>(nullptr),
        static_cast<aclTensorList*>(nullptr),static_cast<aclScalarList*>(nullptr));
    ReleaseConvertTypes(parameters);
    ReleaseHugeMem(nullptr,false); UnInitHugeMemThreadLocal(nullptr,false);
    const std::string report=R25HostSimDiagnosticsJson();
    assert(report.find("\"errors\":0")!=std::string::npos);
    assert(report.find("\"live_descriptors\":0")!=std::string::npos);
    assert(report.find("\"live_executors\":0")!=std::string::npos);
    assert(report.find("\"huge_pending\":0")!=std::string::npos);
    assert(report.find("\"huge_active_threads\":0")!=std::string::npos);
    assert(R25HostSimResetDiagnostics()==0);
    assert(aclDestroyTensor(tensor)!=0); // Nonnull stale handles remain errors.
    assert(aclDestroyAclOpExecutor(nullptr)!=0); // Executors are never nullable cleanup.
}
'''
        fixture = fixture.replace("/*PROVIDER*/", (SDK / "r25_hostsim_opapi.cpp").as_posix()).replace("/*RELEASE*/", source[start:end])
        with tempfile.TemporaryDirectory(prefix="r25-ordinary-null-cleanup-") as folder:
            path = Path(folder) / "ordinary_cleanup.cpp"
            path.write_text(fixture, encoding="utf-8")
            integration.NativeOwnedJobTests().compile_run(path)

    def test_default_trace_is_off_and_cold_trace_is_bounded(self):
        source, output = self.make_tensor(), self.make_tensor()
        try:
            executor = self.prepare(source, output)
            self.assertEqual(self.lib.aclnnRelu(None, 0, executor, None), 0)
            self.assertEqual(self.snapshot()["events"], [])
            self.assertEqual(self.lib.R25HostSimSetTracing(True), 0)
            for _ in range(2050):
                executor = self.prepare(source, output)
                self.assertEqual(self.lib.aclnnRelu(None, 0, executor, None), 0)
            info = self.snapshot()
            self.assertEqual(len(info["events"]), 4096)
            self.assertEqual(info["trace_dropped"], 4)
            self.assertEqual(info["executors_created"], 2051)
            self.assertEqual(info["executors_executed"], 2051)
            self.assertEqual(info["live_executors"], 0)
        finally:
            self.assertEqual(self.lib.aclDestroyTensor(source), 0)
            self.assertEqual(self.lib.aclDestroyTensor(output), 0)
        self.assertEqual(self.lib.R25HostSimResetDiagnostics(), 0)


if __name__ == "__main__":
    unittest.main()
