"""Compile the real event manager with CPU-only dependency stubs, then test it.

No Torch, Python extension, ACL runtime, or NPU is needed. The stubs replace
dependencies outside the counter/wait path; the production manager source and
header are compiled unchanged. Run: python test/linear_scaling_r41/event_wait_native.py
"""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
STUBS = {
    "c10/core/thread_pool.h": r"""#pragma once
#include <functional>
#include <memory>
namespace c10 {
class TaskThreadPool {
public:
    explicit TaskThreadPool(int) {}
    template <typename F> void run(F&& f) { f(); }
    void waitWorkComplete() {}
};
}
""",
    "c10/util/flat_hash_map.h": r"""#pragma once
#include <unordered_map>
namespace ska { template <typename K, typename V> using flat_hash_map = std::unordered_map<K, V>; }
""",
    "third_party/acl/inc/acl/acl.h": r"""#pragma once
using aclrtEvent = void*;
using aclError = int;
constexpr int ACL_ERROR_NONE = 0;
inline int aclrtDestroyEvent(aclrtEvent) { return 0; }
""",
    "torch_npu/csrc/core/npu/NPUException.h": r"""#pragma once
#include <sstream>
#include <stdexcept>
#include "third_party/acl/inc/acl/acl.h"
template <typename... Args> void EventWaitTestCheck(bool value, Args&&... args) {
    if (!value) {
        std::ostringstream text;
        (text << ... << args);
        throw std::runtime_error(text.str());
    }
}
#define TORCH_CHECK(...) EventWaitTestCheck(__VA_ARGS__)
#define PTA_ERROR(...) ""
#define CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(value) ((void)(value))
#define C10_NPU_SHOW_ERR_MSG() ((void)0)
#define NPU_CHECK_ERROR(value) ((void)(value))
#define NPU_CHECK_WARN(value) ((void)(value))
namespace c10_npu { namespace acl {
enum aclrtEventRecordedStatus { ACL_EVENT_RECORDED_STATUS_NOT_READY, ACL_EVENT_RECORDED_STATUS_COMPLETE };
enum aclrtEventWaitStatus { ACL_EVENT_WAIT_STATUS_RESERVED, ACL_EVENT_WAIT_STATUS_COMPLETE };
inline int AclQueryEventRecordedStatus(aclrtEvent, aclrtEventRecordedStatus* value) {
    *value = ACL_EVENT_RECORDED_STATUS_COMPLETE; return 0;
}
inline int AclQueryEventWaitStatus(aclrtEvent, aclrtEventWaitStatus* value) {
    *value = ACL_EVENT_WAIT_STATUS_COMPLETE; return 0;
}
inline bool IsExistCreateEventExWithFlag() { return true; }
}}
""",
    "torch_npu/csrc/core/npu/npu_log.h": r"""#pragma once
#define ASCEND_LOGI(...) ((void)0)
#define ASCEND_LOGD(...) ((void)0)
""",
    "torch_npu/csrc/core/npu/register/OptionsManager.h": r"""#pragma once
namespace c10_npu { namespace option {
struct OptionsManager { static bool GetPerStreamQueue() { return false; } };
}}
""",
}


def compiler_environment():
    env = os.environ.copy()
    if os.name != "nt":
        compiler = shutil.which(env.get("CXX", "c++"))
        if not compiler:
            raise RuntimeError("a C++17 compiler is required")
        return compiler, env
    compiler = shutil.which("cl.exe")
    if compiler:
        return compiler, env
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        text=True,
    ).strip()
    if not installation:
        raise RuntimeError("MSVC C++ build tools are required")
    vcvars = Path(installation) / "VC/Auxiliary/Build/vcvarsall.bat"
    output = subprocess.check_output(
        f'call "{vcvars}" x64 >nul && set', shell=True, text=True,
    )
    for line in output.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            env[name.upper()] = value
    compiler = shutil.which("cl.exe", path=env.get("PATH"))
    if not compiler:
        raise RuntimeError("vcvarsall did not expose cl.exe")
    return compiler, env


def main():
    compiler, env = compiler_environment()
    with tempfile.TemporaryDirectory(prefix="torch_npu_event_wait_") as temporary:
        build = Path(temporary)
        stubs = build / "stubs"
        for name, contents in STUBS.items():
            target = stubs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        native = ROOT / "test/linear_scaling_r41/event_wait_native.cpp"
        manager = ROOT / "torch_npu/csrc/core/npu/NPUEventManager.cpp"
        executable = build / ("event_wait_native.exe" if os.name == "nt" else "event_wait_native")
        if os.name == "nt":
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/O2",
                       "/D_CRT_SECURE_NO_WARNINGS",
                       "/DBUILD_LIBTORCH", "/DC10_NPU_BUILD_MAIN_LIB", "/DC10_NPU_BUILD_SHARED_LIBS",
                       f"/I{stubs}", f"/I{ROOT}", str(native), str(manager),
                       f"/Fe:{executable}", f"/Fo:{build}{os.sep}"]
        else:
            command = [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror", "-O2",
                       "-DBUILD_LIBTORCH", "-rdynamic", f"-I{stubs}", f"-I{ROOT}",
                       str(native), str(manager), "-ldl", "-o", str(executable)]
        print("Compiling production NPUEventManager.cpp with CPU-only dependency stubs", flush=True)
        subprocess.run(command, env=env, cwd=build, check=True, timeout=120)
        for value, expected in [(None, "spin"), ("spin", "spin"), ("condition", "condition"),
                                ("invalid-mode", "invalid"), ("", "invalid")]:
            child_env = env.copy()
            for name in list(child_env):
                if name.upper() == "TORCH_NPU_EVENT_HOST_WAIT":
                    del child_env[name]
            if value is not None:
                child_env["TORCH_NPU_EVENT_HOST_WAIT"] = value
            print(f"Running TORCH_NPU_EVENT_HOST_WAIT={value!r} expected={expected}", flush=True)
            subprocess.run([str(executable), expected], env=child_env, check=True, timeout=30)
    print("All CPU-native event host-wait cases passed", flush=True)


if __name__ == "__main__":
    main()
