"""CPU compile/run the actual R41 admission function with dependency stubs.

This supplements the cache helper tests: the function is extracted verbatim
from op_api_common.cpp, so missing production gate wiring fails at runtime.
"""
from pathlib import Path
import os
import subprocess
import tempfile

from event_wait_native import compiler_environment

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "op_api_workspace_cache_r41.h"
#if __has_include("torch_npu/csrc/core/npu/RuntimeResourceEpoch.h")
#include "torch_npu/csrc/core/npu/RuntimeResourceEpoch.h"
#else
namespace c10_npu {
inline uint64_t GetRuntimeResourceEpoch() noexcept { return 0; }
inline void InvalidateRuntimeResources() noexcept {}
}
#endif
using aclrtStream = void*;
struct State {
    bool native_allocator = true, pool = false, force_uncached = false;
    bool unsafe = false, initialized = true, graph = false;
    unsigned allocations = 0;
} state;
namespace at {
struct Tensor { std::shared_ptr<unsigned> owner; };
}
namespace at_npu { namespace native {
struct OpPreparation {
    static at::Tensor unsafe_empty_workspace(uint64_t) {
        return {std::make_shared<unsigned>(++state.allocations)};
    }
};
}}
namespace c10_npu {
struct MockStream {
    int device_index() const { return 0; }
    bool isSyncLaunchStream() const { return false; }
    aclrtStream stream(bool) const { return reinterpret_cast<aclrtStream>(0x1000); }
};
inline MockStream getCurrentNPUStream() { return {}; }
namespace option { struct OptionsManager {
    static unsigned GetTaskQueueEnable() { return 1; }
    static bool CheckForceUncached() { return state.force_uncached; }
}; }
namespace detail { inline bool isCurrentStreamExternal() { return false; } }
namespace host_regions {
inline bool IsCapturing() { return false; }
inline bool OwnedOpApiEnabled() { return false; }
}
namespace eager_prepare { inline bool Enabled() { return false; } }
namespace NPUCachingAllocator {
inline bool isNativeAllocator() noexcept { return state.native_allocator; }
inline bool hasCapturesUnderway(int) { return state.pool; }
}
inline bool get_npu_data_unsafe_flag() { return state.unsafe; }
class NpuSysCtrl {
public:
    static NpuSysCtrl& GetInstance() { static NpuSysCtrl value; return value; }
    bool GetInitFlag() { return state.initialized; }
};
enum class CaptureStatus { None, Active };
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
    return state.graph ? CaptureStatus::Active : CaptureStatus::None;
}
}
"""

SUFFIX = r"""
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    unsigned failures = 0;
    auto run = [&](const char* name, auto operation) {
        try { operation(); std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& error) {
            ++failures; std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    bool retain = false;
    auto original = GetOpApiWorkspaceR41(64, stream, retain);
    require(retain, "healthy native/default-pool request must qualify");
    require(GetOpApiWorkspaceR41(32, stream, retain).owner == original.owner,
        "healthy request must reuse cache");
    auto bypass = [&] {
        auto first = GetOpApiWorkspaceR41(32, stream, retain);
        require(!retain, "unsupported production mode must not qualify");
        auto second = GetOpApiWorkspaceR41(32, stream, retain);
        require(first.owner != original.owner && second.owner != original.owner && first.owner != second.owner,
            "unsupported production mode must allocate fresh fallback");
    };
    run("actual custom allocator gate", [&] {
        state.native_allocator = false; bypass(); state.native_allocator = true;
    });
    state.native_allocator = true;
    run("actual private allocator pool gate", [&] {
        state.pool = true; bypass(); state.pool = false;
    });
    state.pool = false;
    run("actual force-uncached gate", [&] {
        state.force_uncached = true; bypass(); state.force_uncached = false;
    });
    state.force_uncached = false;
    run("actual unsafe-storage gate", [&] {
        state.unsafe = true; bypass(); state.unsafe = false;
    });
    state.unsafe = false;
    run("actual runtime-initialization gate", [&] {
        state.initialized = false; bypass(); state.initialized = true;
    });
    state.initialized = true;
    run("unsupported scopes leave native cache intact", [&] {
        require(GetOpApiWorkspaceR41(32, stream, retain).owner == original.owner,
            "temporary unsupported scope must not admit its buffer");
    });
    run("resource transition permanently bypasses actual production cache", [&] {
        require(c10_npu::GetRuntimeResourceEpoch() == 0, "fresh process must have epoch zero");
        c10_npu::InvalidateRuntimeResources();
        require(c10_npu::GetRuntimeResourceEpoch() != 0, "resource invalidation must publish nonzero epoch");
        bypass();
        state.unsafe = true;
        state.unsafe = false;
        bypass();
        c10_npu::InvalidateRuntimeResources();
        bypass();
    });
    return failures ? 1 : 0;
}
"""


def function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[opening + 1:cursor - 1]


def verify_lifecycle_wiring():
    hooks = {
        "torch_npu/csrc/core/npu/NPUFunctions.cpp": [
            ("aclError ResetUsedDevices()", "aclrtResetDevice("),
            ("aclError DestroyUsedStreams()", "AclrtDestroyStreamForce(")],
        "torch_npu/csrc/core/npu/NPUStream.cpp": [
            ("void recovery_all_npu_streams(", ".stream.store(")],
        "torch_npu/csrc/npu/Module.cpp": [
            ("PyObject* THNPModule_stopDevice_wrap(", "setDefaultStreamsStatus("),
            ("PyObject* THNPModule_restart_device_wrap(", "IsExistAclrtRepairError(")],
        "torch_npu/csrc/core/npu/NPUCachingAllocator.cpp": [
            ("void markAllBlockUnsafe(int device) override", "->markAllBlockUnsafe(")],
    }
    for relative, signatures in hooks.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        for signature, mutation in signatures:
            body = function_body(source, signature)
            if "InvalidateRuntimeResources();" not in body:
                raise AssertionError(f"resource transition does not invalidate workspace admission: {relative}: {signature}")
            if body.index("InvalidateRuntimeResources();") >= body.index(mutation):
                raise AssertionError(f"resource invalidation follows mutation: {relative}: {signature}")
    source = (ROOT / "torch_npu/csrc/core/npu/NPUCachingAllocator.cpp").read_text(encoding="utf-8")
    identity = function_body(source, "bool isNativeAllocator() noexcept")
    if "&caching_allocator" not in identity or "allocator.load(" not in identity:
        raise AssertionError("native allocator query must compare exact allocator identity")
    print("PASS lifecycle hooks and exact native allocator identity are wired", flush=True)


def main():
    compiler, env = compiler_environment()
    source = (ROOT / "patches/eager_mpsc_r34/op-plugin/op_plugin/utils/op_api_common.cpp").read_text(encoding="utf-8")
    begin = source.index("namespace {\nbool WorkspaceCacheEnabledR41()")
    end = source.index("// Caller-thread qualification only:", begin)
    with tempfile.TemporaryDirectory(prefix="workspace_admission_r41_") as temporary:
        build = Path(temporary)
        unit = build / "workspace_admission.cpp"
        unit.write_text(PREFIX + source[begin:end] + SUFFIX, encoding="utf-8")
        executable = build / ("workspace_admission.exe" if os.name == "nt" else "workspace_admission")
        epoch = ROOT / "torch_npu/csrc/core/npu/RuntimeResourceEpoch.cpp"
        sources = [str(unit)] + ([str(epoch)] if epoch.exists() else [])
        helper = ROOT / "patches/linear_scaling_r41/op-plugin/op_plugin/utils"
        if os.name == "nt":
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/O2",
                       "/D_CRT_SECURE_NO_WARNINGS", f"/I{ROOT}", f"/I{helper}", *sources,
                       f"/Fe:{executable}", f"/Fo:{build}{os.sep}"]
        else:
            command = [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror", "-O2",
                       f"-I{ROOT}", f"-I{helper}", *sources, "-o", str(executable)]
        subprocess.run(command, env=env, cwd=build, check=True, timeout=120)
        env["TORCH_NPU_WORKSPACE_CACHE"] = "1"
        subprocess.run([str(executable)], env=env, check=True, timeout=30)
    verify_lifecycle_wiring()


if __name__ == "__main__":
    main()
