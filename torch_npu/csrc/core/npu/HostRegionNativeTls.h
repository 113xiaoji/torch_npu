#pragma once
#include <cstdint>

// The standalone capture-TU fixtures deliberately do not emulate Torch or
// CPython. Real std/FT probes must compile the branch below without this macro.
#ifdef HDB_NATIVE_CORE_TEST
namespace hdb_native {
struct CallerTls {
    explicit CallerTls(bool = false) noexcept {}
    int device() const noexcept { return 0; }
    void* context() const noexcept { return reinterpret_cast<void*>(uintptr_t(1)); }
};
struct TaskTls {
    explicit TaskTls(const CallerTls&, bool = false) noexcept {}
    TaskTls(const TaskTls&) = delete;
    TaskTls(TaskTls&&) = delete;
    TaskTls& operator=(const TaskTls&) = delete;
    TaskTls& operator=(TaskTls&&) = delete;
};
struct WaitDetach {
    WaitDetach() noexcept = default;
    WaitDetach(const WaitDetach&) = delete;
    WaitDetach(WaitDetach&&) = delete;
    WaitDetach& operator=(const WaitDetach&) = delete;
    WaitDetach& operator=(WaitDetach&&) = delete;
};
} // namespace hdb_native
#else
#include <Python.h>
#include <c10/core/AutogradState.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace hdb_native {
class TaskTls;

class CallerTls {
public:
    explicit CallerTls(bool require_primary_context = false) : state_(capture(require_primary_context)) {}
    c10::DeviceIndex device() const noexcept { return state_.device; }
    void* context() const noexcept { return state_.context; }

private:
    friend class TaskTls;
#ifdef HDB_NATIVE_TLS_PROBE
    friend struct NativeTlsProbeAccess;
#endif
    struct Snapshot {
        c10::DeviceIndex device;
        c10::Stream stream;
        c10::DispatchKeySet include, exclude;
        bool inference;
        aclrtContext context;
    };

    static Snapshot capture(bool require_primary_context)
    {
        // Do not copy AutogradState: a nondefault graph group can contain a
        // SafePyObject. Only native values from the admitted caller cross threads.
        const auto& autograd = c10::AutogradState::get_tls_state();
        const bool inference = autograd.get_inference_mode();
        TORCH_CHECK(!autograd.get_grad_mode(), "native regions require no-grad caller");
        TORCH_CHECK(autograd.get_fw_grad_mode() == !inference &&
                    autograd.get_multithreading_enabled() == !inference &&
                    !autograd.get_view_replay_enabled() &&
                    autograd.get_grad_layout_enforcement_enabled() &&
                    !autograd.get_graph_exec_group().has_value(),
                    "native regions require default caller AutogradState");
        TORCH_CHECK(c10::impl::TorchDispatchModeTLS::stack_len() == 0,
                    "native regions do not support Python dispatch modes");
        const auto keys = c10::impl::tls_local_dispatch_key_set();
        TORCH_CHECK(!keys.included_.has_any(c10::python_ks),
                    "native regions do not support included Python dispatch keys");
        const auto device = c10_npu::current_device();
        TORCH_CHECK(device == 0, "native regions require audited NPU device0");
        const auto stream = c10_npu::getCurrentNPUStream(device);
        TORCH_CHECK(stream.id() == 0 && stream == c10_npu::getDefaultNPUStream(device),
                    "native regions require the audited default NPU stream");
        // Legacy HOSTSIM stubs may not expose a Context. Only the new real run
        // requires one; it never promotes a missing simulator value to a token.
        aclrtContext context = nullptr;
        if (require_primary_context && aclrtGetCurrentContext(&context) != ACL_ERROR_NONE) context = nullptr;
        return Snapshot{device, stream.unwrap(), keys.included_, keys.excluded_, inference, context};
    }

    Snapshot state_;
};

class TaskTls {
public:
    explicit TaskTls(const CallerTls& caller, bool require_primary_context = false)
        : inference_(caller.state_.inference),
          no_grad_(),
          dispatch_(caller.state_.include, caller.state_.exclude),
          device_(caller.state_.device),
          stream_(caller.state_.stream) {
        if (require_primary_context) {
            // GetDevice reports logical device0 on a fresh thread even when
            // ACL has no current Context, so NPUGuard alone may skip binding.
            // Bind a private worker's first use exactly as Torch lazy device
            // initialization does; later task guards restore native TLS.
            NPU_CHECK_ERROR(c10_npu::SetDevice(caller.state_.device));
            aclrtContext actual = nullptr;
            NPU_CHECK_ERROR(aclrtGetCurrentContext(&actual));
            TORCH_CHECK(actual && actual == caller.state_.context,
                "native region worker is outside the captured primary Context");
        }
    }
    TaskTls(const TaskTls&) = delete;
    TaskTls(TaskTls&&) = delete;
    TaskTls& operator=(const TaskTls&) = delete;
    TaskTls& operator=(TaskTls&&) = delete;
    ~TaskTls() noexcept = default;

private:
    // Member order is the original five-context entry order. InferenceMode(false)
    // enables gradients, so no-grad must follow it. Reverse destruction restores
    // streams/device/dispatch before the original complete worker AutogradState.
    c10::InferenceMode inference_;
    c10::NoGradGuard no_grad_;
    c10::impl::ForceDispatchKeyGuard dispatch_;
    c10_npu::NPUGuard device_;
    c10_npu::NPUStreamGuard stream_;
};

class WaitDetach {
public:
    WaitDetach() noexcept
        : saved_(PyThreadState_GetUnchecked() != nullptr ? PyEval_SaveThread() : nullptr) {}
    WaitDetach(const WaitDetach&) = delete;
    WaitDetach(WaitDetach&&) = delete;
    WaitDetach& operator=(const WaitDetach&) = delete;
    WaitDetach& operator=(WaitDetach&&) = delete;
    ~WaitDetach() noexcept
    {
        if (saved_ != nullptr) PyEval_RestoreThread(saved_);
    }

private:
    // Same-thread interpreter bookkeeping only. No new thread state, Python
    // callback or object crosses threads. Construct before channel locks and
    // destroy after unlock: reacquiring Python state while holding them can
    // deadlock GIL-on workers and free-threaded stop-the-world GC.
    PyThreadState* saved_;
};
} // namespace hdb_native
#endif
