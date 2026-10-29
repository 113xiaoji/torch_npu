#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_numbers.h>
#endif

#include <nlohmann/json.hpp>
#include <fstream>

#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/EagerPrepare.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"
#include "third_party/acl/inc/acl/acl_op_compiler.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/LazyInitAclops.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#ifdef SUCCESS
#undef SUCCESS
#endif
#ifdef FAILED
#undef FAILED
#endif


namespace {
const uint32_t kMaxOpExecuteTimeOut = 547U;
const size_t kMaxPathLen = 4096U;

void SetDefaultAllowInternalFromatDisable()
{
    auto allow_internal_format = c10_npu::option::GetOption("ALLOW_INTERNAL_FORMAT");
    if (allow_internal_format.has_value() && allow_internal_format.value() != "") {
        return;
    }

    c10_npu::option::SetOption("ALLOW_INTERNAL_FORMAT", "disable");
    ASCEND_LOGI("Set ALLOW_INTERNAL_FORMAT default value disable.");
}

void SetDeterministicFromLevel()
{
    const static bool isAclStrongConsistencyExist = []() {
        const std::string kMinRuntimeVersion = "8.5.0";
        if (IsGteCANNVersion(kMinRuntimeVersion, "RUNTIME")) {
            return true;
        }
        return false;
    }();
    if (!isAclStrongConsistencyExist) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
        return;
    }

    uint32_t level = c10_npu::GetDeterministicLevel();
    if (level == 1) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 1));
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 0));
        return;
    } else if (level == 2) {
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 1));
        NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 1));
        return;
    } else if (level != 0) {
        TORCH_CHECK(false, "'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !", PTA_ERROR(ErrCode::VALUE));
    }
    NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, 0));
    NPU_CHECK_ERROR(at_npu::native::AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_STRONG_CONSISTENCY, 0));
}

#ifndef BUILD_LIBTORCH
std::string GetTorchNpuFile()
{
    pybind11::gil_scoped_acquire get_gil;
    auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
    if (!torch) {
        throw python_error();
    }
    auto file_attr = THPObjectPtr(PyObject_GetAttrString(torch.get(), "__file__"));
    if (!file_attr) {
        PyErr_Clear();
        ASCEND_LOGW("Failed to get __file__ attribute.");
        return "";
    }
    const char *file_path = PyUnicode_AsUTF8(file_attr.get());
    if (file_path == nullptr) {
        throw python_error();
    }
    std::string file_path_str(file_path);
    const std::string key_word = "torch";
    const size_t pos = file_path_str.rfind(key_word);
    if (pos != std::string::npos) {
        return file_path_str.substr(0, pos);
    }
    ASCEND_LOGW("Failed to get __file__ attribute.");
    return "";
}
#endif

std::string GetAclConfigJsonPath()
{
#ifndef BUILD_LIBTORCH
    const char* acl_init_path = c10_npu::option::OptionsManager::GetAclInitPath();
    if (acl_init_path != nullptr) {
        std::string json_path = std::string(acl_init_path);
        std::string json_path_str = torch_npu::toolkit::profiler::Utils::RealPath(json_path);
        if (json_path_str.empty()) {
            TORCH_CHECK(false, "TORCH_ACL_INIT_CONFIG_PATH ", acl_init_path, " is invalid. ",
                        "Please set the environment variable 'TORCH_ACL_INIT_CONFIG_PATH' to a valid JSON config file path.",
                        PTA_ERROR(ErrCode::UNAVAIL));
        }

        std::ifstream config_file(json_path_str);
        if (!config_file.is_open()) {
            TORCH_CHECK(false, "Failed to open user acl json ", json_path_str,
                        ". Please ensure the file exists and has read permission.",
                        PTA_ERROR(ErrCode::UNAVAIL));
        }

        using json = nlohmann::json;
        json config;
        try {
            config_file >> config;
        } catch (const std::exception& e) {
            TORCH_CHECK(false, "Failed to parse user acl json ", json_path_str,
                        ". Please check that the file contains valid JSON syntax. Error: ", e.what(),
                        PTA_ERROR(ErrCode::UNAVAIL));
        }

        if (c10_npu::is_lazy_set_device()) {
            if (!config.contains("defaultDevice") || !config["defaultDevice"].is_object()) {
                TORCH_CHECK(false, "User acl json ", json_path_str,
                            " lacks 'defaultDevice' object, required for lazy set_device mode. ",
                            "Please add: \"defaultDevice\": {\"default_device\": \"0\"}",
                            PTA_ERROR(ErrCode::VALUE));
            }
            const auto& default_dev = config["defaultDevice"];
            if (!default_dev.contains("default_device") ||
                !default_dev["default_device"].is_string() ||
                default_dev["default_device"].get<std::string>() != "0") {
                TORCH_CHECK(false, "User acl json ", json_path_str,
                            " requires 'defaultDevice.default_device' to be \"0\" in lazy set_device mode. ",
                            "Please set: \"defaultDevice\": {\"default_device\": \"0\"}",
                            PTA_ERROR(ErrCode::VALUE));
            }
        } else {
            if (config.contains("defaultDevice")) {
                TORCH_CHECK(false, "User acl json ", json_path_str,
                            " contains 'defaultDevice' which is not expected in non-lazy set_device mode. ",
                            "Please remove the 'defaultDevice' field from your JSON config.",
                            PTA_ERROR(ErrCode::VALUE));
            }
        }

        return json_path_str;
    }
    std::string npu_path = GetTorchNpuFile();
    if (npu_path == "") {
        ASCEND_LOGW("Failed to get npu path!");
        return "";
    }
    std::string json_path = "";
    if (c10_npu::is_lazy_set_device()) {
        json_path = npu_path.append("torch_npu/acl_default.json");
    } else {
        json_path = npu_path.append("torch_npu/acl.json");
    }
    std::string json_path_str = torch_npu::toolkit::profiler::Utils::RealPath(json_path);
    if (json_path_str == "") {
        ASCEND_LOGW("this path:%s is not exist!", json_path.c_str());
    }
    return json_path_str;
#else
    return "";
#endif
}
} // namespace

namespace c10_npu {

NpuSysCtrl::NpuSysCtrl() : repeat_init_acl_flag_(true) {}

// Get NpuSysCtrl singleton instance
NpuSysCtrl &NpuSysCtrl::GetInstance()
{
    static NpuSysCtrl instance;
    return instance;
}

// Environment Initialize, return Status: SUCCESS, FAILED
NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id)
{
    {
        std::unique_lock<std::mutex> lifecycle_lock(init_mutex_);
        lifecycle_cv_.wait(lifecycle_lock, [this]() {
            return lifecycle_ != RuntimeLifecycle::Initializing &&
                lifecycle_ != RuntimeLifecycle::Finalizing;
        });
        if (lifecycle_ == RuntimeLifecycle::Running) {
            return INIT_SUCC;
        }
        if (lifecycle_ == RuntimeLifecycle::Failed) {
            const std::exception_ptr error = initialization_error_;
            lifecycle_lock.unlock();
            if (error) {
                std::rethrow_exception(error);
            }
            TORCH_CHECK(false, "NPU runtime initialization previously failed.", PTA_ERROR(ErrCode::INTERNAL));
        }
        lifecycle_ = RuntimeLifecycle::Initializing;
        initialization_error_ = nullptr;
        repeat_init_acl_flag_ = true;
        host_finalize_flag_.store(false, std::memory_order_release);
    }

    bool owns_acl_runtime = false;
    try {
    std::string json_path = GetAclConfigJsonPath();
    const char *json_path_ptr = json_path == "" ? nullptr : json_path.c_str();
    ASCEND_LOGD("get acl json path:%s.", json_path_ptr);
    auto init_ret = aclInit(json_path_ptr);
    // The aclinit function is an important low-level aclInit interface that can only be called once in PTA.
    // However, because of the different business codes, aclinit may be successfully invoked by other frameworks or components.
    // ACL_ERROR_REPEAT_INITIALIZE means that aclInit is not called by PTA, so we save the flag variable to control aclFinalize.
    if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
        repeat_init_acl_flag_ = false;
        ASCEND_LOGI("acl has already init by other component.");
    } else if (init_ret != ACL_ERROR_NONE) {
        NPU_CHECK_ERROR(init_ret, "aclInit");
    } else {
        owns_acl_runtime = true;
    }

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        NPU_CHECK_ERROR(aclmdlInitDump());
        ASCEND_LOGD("dump init success");
    }

    c10_npu::NPUCachingAllocator::init();
    ASCEND_LOGD("Npu caching allocator initialize successfully");
    c10_npu::NPUWorkspaceAllocator::init();
    ASCEND_LOGD("Npu workspace allocator initialize successfully");
    c10_npu::option::OptionsManager::IsOomSnapshotEnable();

    if (!c10_npu::is_lazy_set_device()) {
        // There's no need to call c10_npu::GetDevice at the start of the process, because device 0 may not be needed
        int initialized_device = device_id_.load(std::memory_order_acquire);
        auto ret = aclrtGetDevice(&initialized_device);
        if (ret != ACL_ERROR_NONE) {
            initialized_device = (device_id == -1) ? 0 : device_id;
            NPU_CHECK_ERROR(c10_npu::SetDevice(initialized_device));
        } else {
            ASCEND_LOGW("Npu device %d has been set before global init.", initialized_device);
        }
        device_id_.store(initialized_device, std::memory_order_release);
    } else {
        if (device_id >= 0) {
            NPU_CHECK_ERROR(c10_npu::SetDevice(device_id));
            device_id_.store(device_id, std::memory_order_release);
        }
    }

    if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
        const char *aclConfigPath = "acl.json";
        NPU_CHECK_ERROR(aclmdlSetDump(aclConfigPath));
        ASCEND_LOGD("set dump config success");
    }

    auto soc_name = c10_npu::acl::AclGetSocName();
    // set global soc name
    c10_npu::SetSocVersion(soc_name);

    if (!c10_npu::is_lazy_set_device()) {
        if (c10_npu::IsSupportInfNan()) {
            c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
        } else {
            c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
        }
    }

    auto acl_op_init_mode = c10_npu::option::OptionsManager::GetAclOpInitMode();
    if (acl_op_init_mode == 0) {
        at_npu::aclops::InitAclops();
    } else {
        at_npu::aclops::InitializeJitCompilationMode();
    }

    // set default allow_internal_format value
    if (((c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) &&
        (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1)) ||
        (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_9391)) {
        SetDefaultAllowInternalFromatDisable();
    }

    if (!c10_npu::is_lazy_set_device()) {
        SetDeterministicFromLevel();
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSetOpExecuteTimeOut(kMaxOpExecuteTimeOut));
    }

    if (SetThreadAffinityInInitialize()) {
        SetThreadAffinity(device_id_.load(std::memory_order_acquire));
    }

    for (;;) {
        std::vector<std::pair<option::OptionCallBack, std::string>> pending_callbacks;
        {
            std::lock_guard<std::mutex> lifecycle_lock(init_mutex_);
            pending_callbacks.swap(lazy_fn_);
            if (pending_callbacks.empty()) {
                init_flag_.store(true, std::memory_order_release);
                lifecycle_ = RuntimeLifecycle::Running;
            }
        }
        if (pending_callbacks.empty()) {
            break;
        }
        for (const auto& callback : pending_callbacks) {
            ASCEND_LOGD("start setoption for the lazy call.");
            callback.first(callback.second);
        }
    }
    lifecycle_cv_.notify_all();
    ASCEND_LOGD("Npu sys ctrl initialize successfully.");

    return INIT_SUCC;
    } catch (...) {
        const std::exception_ptr initialization_error = std::current_exception();
        if (owns_acl_runtime) {
            try {
                NPU_CHECK_WARN(aclFinalize());
            } catch (...) {
                ASCEND_LOGE("ACL rollback after NPU initialization failure also failed.");
            }
        }
        {
            std::lock_guard<std::mutex> lifecycle_lock(init_mutex_);
            init_flag_.store(false, std::memory_order_release);
            lifecycle_ = RuntimeLifecycle::Failed;
            initialization_error_ = initialization_error;
        }
        lifecycle_cv_.notify_all();
        std::rethrow_exception(initialization_error);
    }
}

NpuSysCtrl::SysStatus NpuSysCtrl::LazyInitialize(int device_id)
{
    if (!c10_npu::is_lazy_set_device()) {
        return INIT_SUCC;
    }

    std::unique_lock<std::mutex> lifecycle_lock(init_mutex_);
    lifecycle_cv_.wait(lifecycle_lock, [this]() {
        return lifecycle_ != RuntimeLifecycle::Initializing &&
            lifecycle_ != RuntimeLifecycle::Finalizing;
    });
    TORCH_CHECK(
        lifecycle_ == RuntimeLifecycle::Running,
        "NPU runtime must be initialized before lazy device initialization.",
        PTA_ERROR(ErrCode::INTERNAL));
    if (lazy_init_flag_.load(std::memory_order_acquire)) {
        return INIT_SUCC;
    }

    // There's no need to call c10_npu::GetDevice at the start of the process, because device 0 may not be needed
    int initialized_device = device_id_.load(std::memory_order_acquire);
    auto ret = aclrtGetDevice(&initialized_device);
    if (ret == ACL_ERROR_NONE) {
        device_id_.store(initialized_device, std::memory_order_release);
    }

    if (c10_npu::IsSupportInfNan()) {
        c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN);
    } else {
        c10_npu::acl::AclrtSetDeviceSatMode(aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION);
    }

    SetDeterministicFromLevel();
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSetOpExecuteTimeOut(kMaxOpExecuteTimeOut));

    lazy_init_flag_.store(true, std::memory_order_release);
    ASCEND_LOGD("Npu sys ctrl Lazyinitialize successfully.");

    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int device)
{
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device));
    device_id_.store(device, std::memory_order_release);
    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit()
{
    NPU_CHECK_ERROR(c10_npu::SetDevice(device_id_.load(std::memory_order_acquire)));
    return INIT_SUCC;
}

NpuSysCtrl::SysStatus NpuSysCtrl::OverflowSwitchEnable()
{
    if (!c10_npu::IsSupportInfNan()) {
        c10_npu::acl::AclrtSetStreamOverflowSwitch(c10_npu::getCurrentNPUStream(), 1);
        ASCEND_LOGI("Npu overflow check switch set successfully.");
    }
    return INIT_SUCC;
}

// Environment Finalize, return SysStatus
NpuSysCtrl::SysStatus NpuSysCtrl::Finalize()
{
    // No SDK/init/finalization lock may precede this cold gate. A failed ACK
    // keeps Prepare workers/TLS/owners and never enters forced SDK cleanup.
    eager_prepare::BeforeRuntimeFinalize();
    std::map<ReleasePriority, std::vector<ReleaseFn>> release_callbacks;
    {
        std::unique_lock<std::mutex> lifecycle_lock(init_mutex_);
        lifecycle_cv_.wait(lifecycle_lock, [this]() {
            return lifecycle_ != RuntimeLifecycle::Initializing &&
                lifecycle_ != RuntimeLifecycle::Finalizing;
        });
        if (lifecycle_ == RuntimeLifecycle::Uninitialized) {
            return FINALIZE_SUCC;
        }
        if (lifecycle_ == RuntimeLifecycle::Failed) {
            return FINALIZE_FAILED;
        }

        const bool finalize_acl_runtime = repeat_init_acl_flag_;
        release_fn_[ReleasePriority::PriorityLast].emplace_back(
            []() -> void {
                c10_npu::NPUEventManager::GetInstance().ClearEvent();
            });
        release_fn_[ReleasePriority::PriorityLast].emplace_back(
            []() -> void {
                NPU_CHECK_WARN(c10_npu::DestroyUsedStreams());
            });
        release_fn_[ReleasePriority::PriorityLast].emplace_back(
            []() -> void {
                NPU_CHECK_WARN(c10_npu::ResetUsedDevices());
            });
#ifndef BUILD_LIBTORCH
        release_fn_[ReleasePriority::PriorityLast].emplace_back(
            []() -> void {
                if (c10d::symmetric_memory::Aclshmem_finalize_exist()) {
                    auto ret = c10d::symmetric_memory::Aclshmem_finalize();
                    ASCEND_LOGI("shmem_finalize emd, ret is %d", ret);
                }
            });
#endif
        // Only the component that acquired the ACL runtime releases it.
        if (finalize_acl_runtime) {
            release_fn_[ReleasePriority::PriorityLast].emplace_back(
                []() -> void {
                    NPU_CHECK_WARN(aclFinalize());
                });
        }
        lifecycle_ = RuntimeLifecycle::Finalizing;
        init_flag_.store(false, std::memory_order_release);
        release_callbacks.swap(release_fn_);
    }

    bool finalize_failed = false;
    try {
        if (c10_npu::option::OptionsManager::CheckAclDumpDateEnable()) {
            NPU_CHECK_WARN(aclmdlFinalizeDump());
        }
    } catch (const std::exception& error) {
        finalize_failed = true;
        ASCEND_LOGE("NPU dump finalization failed: %s", error.what());
    } catch (...) {
        finalize_failed = true;
        ASCEND_LOGE("NPU dump finalization failed with an unknown exception.");
    }

    for (const auto& iter : release_callbacks) {
        const auto& fn_vec = iter.second;
        for (const auto& fn : fn_vec) {
            try {
                fn();
            } catch (const std::exception& error) {
                finalize_failed = true;
                ASCEND_LOGE("NPU release callback failed: %s", error.what());
            } catch (...) {
                finalize_failed = true;
                ASCEND_LOGE("NPU release callback failed with an unknown exception.");
            }
        }
    }

    {
        std::lock_guard<std::mutex> lifecycle_lock(init_mutex_);
        lazy_init_flag_.store(false, std::memory_order_release);
        lifecycle_ = RuntimeLifecycle::Uninitialized;
    }
    lifecycle_cv_.notify_all();

    ASCEND_LOGD("Npu sys ctrl finalize successfully.");
    return finalize_failed ? FINALIZE_FAILED : FINALIZE_SUCC;
}

bool NpuSysCtrl::GetInitFlag()
{
    return init_flag_.load(std::memory_order_acquire);
}

bool NpuSysCtrl::GetLazyInitFlag()
{
    return lazy_init_flag_.load(std::memory_order_acquire);
}

void NpuSysCtrl::HostFinalize()
{
    host_finalize_flag_.store(true, std::memory_order_release);
}

bool NpuSysCtrl::GetHostFinalize()
{
    return host_finalize_flag_.load(std::memory_order_acquire);
}

int NpuSysCtrl::InitializedDeviceID()
{
    if (GetInitFlag()) {
        return device_id_.load(std::memory_order_acquire);
    }
    TORCH_CHECK(false, "no npu device has been initialized!", PTA_ERROR(ErrCode::INTERNAL));
    return -1;
}

void NpuSysCtrl::RegisterLazyFn(const option::OptionCallBack& call_, const std::string& in)
{
    bool call_immediately = false;
    {
        std::lock_guard<std::mutex> lifecycle_lock(init_mutex_);
        TORCH_CHECK(
            lifecycle_ != RuntimeLifecycle::Failed,
            "Cannot register an NPU option after runtime initialization failed.",
            PTA_ERROR(ErrCode::INTERNAL));
        if (lifecycle_ == RuntimeLifecycle::Running) {
            call_immediately = true;
        } else {
            lazy_fn_.emplace_back(std::make_pair(call_, in));
        }
    }
    if (call_immediately) {
        call_(in);
    }
}

void NpuSysCtrl::RegisterReleaseFn(ReleaseFn release_fn, ReleasePriority priority)
{
    std::lock_guard<std::mutex> lifecycle_lock(init_mutex_);
    TORCH_CHECK(
        lifecycle_ != RuntimeLifecycle::Finalizing &&
            lifecycle_ != RuntimeLifecycle::Failed,
        "Cannot register an NPU release callback while the runtime is stopping.",
        PTA_ERROR(ErrCode::INTERNAL));
    const auto& iter = this->release_fn_.find(priority);
    if (iter != release_fn_.end()) {
        release_fn_[priority].emplace_back(release_fn);
    } else {
        release_fn_[priority] = (std::vector<ReleaseFn>({release_fn}));
    }
}

aclError SetCurrentDevice()
{
    if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        c10_npu::SetDevice(c10_npu::NpuSysCtrl::GetInstance().InitializedDeviceID());
        return ACL_SUCCESS;
    }
    TORCH_CHECK(false, "npu device has not been inited.", PTA_ERROR(ErrCode::INTERNAL));
}

} // namespace c10_npu
