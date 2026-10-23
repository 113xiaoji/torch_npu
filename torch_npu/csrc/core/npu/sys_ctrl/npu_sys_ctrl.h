#pragma once

#include <third_party/acl/inc/acl/acl.h>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include "c10/macros/Export.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#define NpuSysStatus c10_npu::NpuSysCtrl::SysStatus

namespace c10_npu {
using ReleaseFn = std::function<void()>;

enum class ReleasePriority : uint8_t {
    PriorityFirst = 0,
    PriorityMiddle = 5,
    PriorityLast = 10
};

class NpuSysCtrl {
public:
    ~NpuSysCtrl() = default;

    enum SysStatus {
        INIT_SUCC = 0,
        INIT_ALREADY,
        INIT_FAILED,
        CREATE_SESS_SUCC,
        CREATE_SESS_FAILED,
        FINALIZE_SUCC,
        FINALIZE_FAILED,
    };

    // Get NpuSysCtrl singleton instance
    C10_NPU_API static NpuSysCtrl &GetInstance();

    // Environment Initialize, return SysStatus
    SysStatus Initialize(int device_id = -1);

    SysStatus LazyInitialize(int device_id = -1);

    // Change current device from pre_device to device
    SysStatus ExchangeDevice(int device);

    // Init backwards thread
    SysStatus BackwardsInit();

    // Set overflow switch
    SysStatus OverflowSwitchEnable();

    // Environment Finalize, return SysStatus
    C10_NPU_API SysStatus Finalize();

    // Get Init_flag
    C10_NPU_API bool GetInitFlag();

    // Get lazyInit_flag
    bool GetLazyInitFlag();

    C10_NPU_API void HostFinalize();

    bool GetHostFinalize();

    int InitializedDeviceID();

    void RegisterLazyFn(const option::OptionCallBack &call_, const std::string &in);

    // Register fn to be called during stage of exit and
    // the callability of fn is guaranteed by the caller.
    void RegisterReleaseFn(ReleaseFn release_fn, ReleasePriority priority = ReleasePriority::PriorityMiddle);

private:
    enum class RuntimeLifecycle : uint8_t {
        Uninitialized,
        Initializing,
        Running,
        Finalizing,
        Failed,
    };

    NpuSysCtrl();

private:
    bool repeat_init_acl_flag_;
    std::atomic<bool> init_flag_{false};
    std::atomic<bool> lazy_init_flag_{false};
    std::atomic<bool> host_finalize_flag_{false}; // during shutdown, tell hostallocator empty_cache to avoid stream lifetime issues
    std::atomic<int> device_id_{0};
    std::map<ReleasePriority, std::vector<ReleaseFn>> release_fn_;
    std::vector<std::pair<option::OptionCallBack, std::string>> lazy_fn_;
    RuntimeLifecycle lifecycle_{RuntimeLifecycle::Uninitialized};
    std::exception_ptr initialization_error_;
    std::mutex init_mutex_;
    std::condition_variable lifecycle_cv_;
};

aclError SetCurrentDevice();

} // namespace c10_npu
