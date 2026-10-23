#include <algorithm>
#include <atomic>
#include <cstring>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace c10_npu {
namespace option {
namespace {
std::atomic<c10_npu::option::OptionInterface*> fast_allow_internal_format{nullptr};
std::atomic<c10_npu::option::OptionInterface*> fast_compatible_impl{nullptr};
}
OptionInterface::OptionInterface(OptionCallBack callback)
{
    this->callback = callback;
}

void OptionInterface::Set(const std::string &in)
{
    this->val = in;
    if (this->callback != nullptr) {
        if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
            ASCEND_LOGD("setoption call immediately.");
            this->callback(in);
        } else {
            ASCEND_LOGD("setoption will lazy call.");
            c10_npu::NpuSysCtrl::GetInstance().RegisterLazyFn(this->callback, in);
        }
    }
}

std::string OptionInterface::Get()
{
    return val;
}


namespace register_options {
OptionRegister *OptionRegister::GetInstance()
{
    static OptionRegister instance;
    return &instance;
}

void OptionRegister::Register(const std::string &name, ::std::unique_ptr<OptionInterface> &ptr)
{
    std::lock_guard<std::mutex> lock(mu_);
    auto result = registry.emplace(name, std::move(ptr));
    auto* value = result.first->second.get();
    if (name == "ALLOW_INTERNAL_FORMAT") {
        fast_allow_internal_format.store(value, std::memory_order_release);
    } else if (name == "TORCH_NPU_USE_COMPATIBLE_IMPL") {
        fast_compatible_impl.store(value, std::memory_order_release);
    }
}

void OptionRegister::Set(const std::string &name, const std::string &val)
{
    auto itr = registry.find(name);
    if (itr != registry.end()) {
        itr->second->Set(val);
    } else {
        AT_ERROR("invalid npu option name:", name);
    }
}

c10::optional<std::string> OptionRegister::Get(const std::string &name)
{
    auto itr = registry.find(name);
    if (itr != registry.end()) {
        return itr->second->Get();
    }
    return c10::nullopt; // default value
}

OptionInterfaceBuilder::OptionInterfaceBuilder(const std::string &name, ::std::unique_ptr<OptionInterface> &ptr,
    const std::string &type)
{
    OptionRegister::GetInstance()->Register(name, ptr);

    // init the value if env variable.
    if (type == "env") {
        std::string env_name = name;
        std::transform(env_name.begin(), env_name.end(), env_name.begin(), ::toupper);
        char *env_val = std::getenv(env_name.c_str());
        if (env_val != nullptr) {
            std::string val(env_val);
            OptionRegister::GetInstance()->Set(name, val);
        }
    }
}
} // namespace register_options


namespace {
struct FastCacheLifetime {
    ~FastCacheLifetime() {
        fast_allow_internal_format.store(nullptr, std::memory_order_release);
        fast_compatible_impl.store(nullptr, std::memory_order_release);
    }
};
}
c10::optional<std::string> GetOptionFast(const char* key)
{
    auto* registry = register_options::OptionRegister::GetInstance();
    static FastCacheLifetime lifetime;
    OptionInterface* value = nullptr;
    if (key != nullptr && std::strcmp(key, "ALLOW_INTERNAL_FORMAT") == 0) {
        value = fast_allow_internal_format.load(std::memory_order_acquire);
    } else if (key != nullptr && std::strcmp(key, "TORCH_NPU_USE_COMPATIBLE_IMPL") == 0) {
        value = fast_compatible_impl.load(std::memory_order_acquire);
    }
    if (value != nullptr) return value->Get();
    return registry->Get(key == nullptr ? std::string{} : std::string(key));
}

void SetOption(const std::string &key, const std::string &val)
{
    if (c10_npu::IsAclnnOnly()) {
        if (key == "jitCompile" && val == "enable") {
            TORCH_NPU_WARN_ONCE("Current device only support jit_compile=False, ",
                "the requested value True is invalid and has been reverted to False.");
            return register_options::OptionRegister::GetInstance()->Set(key, "disable");
        }
        if (key == "ALLOW_INTERNAL_FORMAT" && val == "enable") {
            TORCH_NPU_WARN_ONCE("Current device only support allow_internal_format=False, ",
                "the requested value True is invalid and has been reverted to False.");
            return register_options::OptionRegister::GetInstance()->Set(key, "disable");
        }
    }
    register_options::OptionRegister::GetInstance()->Set(key, val);
}

void SetOption(const std::map<std::string, std::string> &options)
{
    for (auto item : options) {
        SetOption(item.first, item.second);
    }
}

c10::optional<std::string> GetOption(const std::string &key)
{
    return register_options::OptionRegister::GetInstance()->Get(key);
}
} // namespace option
} // namespace c10_npu
