#include "torch_npu/csrc/core/npu/HostRegionCapture.h"
#include "torch_npu/csrc/core/npu/HostRegionRetirement.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/PrefixCompletion.h"
#include "torch_npu/csrc/core/npu/PrefixH2DGuard.h"
#include "torch_npu/csrc/core/npu/PrefixH2DPrewait.h"
#include "torch_npu/csrc/core/npu/EagerPrepare.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "third_party/op-plugin/op_plugin/ops/dvm/lazy_fusion_kernel.h"

#ifndef BUILD_LIBTORCH
#include <Python.h>
#include "torch_npu/csrc/utils/PythonThreadState.h"
#endif

#include <ATen/record_function.h>
#include <atomic>
#include <optional>
#include <sstream>
#include <third_party/acl/inc/acl/acl_rt.h>

namespace c10_npu {

namespace {
std::atomic<uint64_t> native_batch_fallback_calls{0};

class CallBackManager {
public:
    CallBackManager() {}
    ~CallBackManager() {}
    void SetExec(const ACL_EXEC_FUNC &func)
    {
        this->execFunc = func;
    }

    void SetCopy(const ACL_COPY_FUNC &func)
    {
        this->copyFunc = func;
    }

    void SetRelease(const ACL_RELEASE_FUNC &func)
    {
        this->releaseFunc = func;
    }

    void SetCopyReleaseParam(const ACL_COPY_RELEASE_PARM_FUNC &func)
    {
        this->copyReleaseParamFunc = func;
    }

    void SetReleaseParam(const ACL_RELEASE_PARAM_FUNC &func)
    {
        this->releaseParamFunc = func;
    }

    void SetNew(const ACL_NEW_FUNC &func)
    {
        this->newFunc = func;
    }

    void SetDelete(const ACL_DELETE_FUNC &func)
    {
        this->deleteFunc = func;
    }

    void *getCurrentParams(void *head, int offset)
    {
        return (uint8_t *)head + ParameterSize() * offset;
    }

    int Call(void *head, int offset)
    {
        TORCH_CHECK(this->execFunc, "Failed to find execution function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)head + ParameterSize() * offset;
        return this->execFunc(dstPtr);
    }

    void Copy(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyFunc, "Failed to find copy function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + ParameterSize() * offset;
        return this->copyFunc(dstPtr, src);
    }

    void Release(void *head, int offset, ReleaseQueue &releaseQueue)
    {
        TORCH_CHECK(this->releaseFunc, "Failed to find release function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + ParameterSize() * offset;
        this->releaseFunc(ptr, releaseQueue);
        host_regions::QueueReleased(ptr);
    }

    void CopyRealseParam(void *dstHead, int offset, void *src)
    {
        TORCH_CHECK(this->copyReleaseParamFunc, "Failed to find copy release params function.",
            PTA_ERROR(ErrCode::NOT_FOUND));
        auto dstPtr = (uint8_t *)dstHead + ParameterSize() * offset;
        return this->copyReleaseParamFunc(dstPtr, src);
    }

    void ReleaseParam(void *head, int offset)
    {
        TORCH_CHECK(this->releaseParamFunc, "Failed to find release params function.", PTA_ERROR(ErrCode::NOT_FOUND));
        auto ptr = (uint8_t *)head + ParameterSize() * offset;
        return this->releaseParamFunc(ptr);
    }

    void *Init(int capacity)
    {
        TORCH_CHECK(this->newFunc, "Failed to find new function.", PTA_ERROR(ErrCode::NOT_FOUND));
        int discovered_size = 0;
        void *ptr = this->newFunc(capacity, discovered_size); // not check as CUDA
        if (discovered_size <= 0) {
            if (ptr != nullptr && this->deleteFunc != nullptr) {
                this->deleteFunc(ptr);
            }
            TORCH_CHECK(
                false,
                "Queue callback returned an invalid parameter size.",
                PTA_ERROR(ErrCode::INTERNAL));
        }
        int expected_size = 0;
        if (!sizePerParams.compare_exchange_strong(
                expected_size,
                discovered_size,
                std::memory_order_acq_rel,
                std::memory_order_acquire) &&
            expected_size != discovered_size) {
            if (ptr != nullptr && this->deleteFunc != nullptr) {
                this->deleteFunc(ptr);
            }
            TORCH_CHECK(
                false,
                "Queue callback parameter size changed across repository initialization.",
                PTA_ERROR(ErrCode::INTERNAL));
        }
        return ptr;
    }

    void DeInit(void *ptr)
    {
        if (ptr != nullptr) {
            TORCH_CHECK(this->deleteFunc, "Failed to find delete function.", PTA_ERROR(ErrCode::NOT_FOUND));
            this->deleteFunc(ptr);
            ptr = nullptr;
        }
    }

private:
    int ParameterSize() const
    {
        const int size = sizePerParams.load(std::memory_order_acquire);
        TORCH_CHECK(size > 0, "Queue callback parameter size is not initialized.", PTA_ERROR(ErrCode::INTERNAL));
        return size;
    }

    std::atomic<int> sizePerParams{0};
    ACL_EXEC_FUNC execFunc = nullptr;
    ACL_COPY_FUNC copyFunc = nullptr;
    ACL_RELEASE_FUNC releaseFunc = nullptr;
    ACL_NEW_FUNC newFunc = nullptr;
    ACL_DELETE_FUNC deleteFunc = nullptr;
    ACL_COPY_RELEASE_PARM_FUNC copyReleaseParamFunc = nullptr;
    ACL_RELEASE_PARAM_FUNC releaseParamFunc = nullptr;
}; // class CallBackManager

CallBackManager &manager()
{
    static CallBackManager instance;
    return instance;
}

CallBackManager &releaseManager()
{
    static CallBackManager releaseinstance;
    return releaseinstance;
}

class QueueWaitHooks {
public:
    explicit QueueWaitHooks(bool detach_python) : detach_python_(detach_python) {}

    void BeforeWait()
    {
#ifndef BUILD_LIBTORCH
        if (detach_python_) {
            python_state_.emplace();
        }
#endif
    }

    void AfterWait()
    {
#ifndef BUILD_LIBTORCH
        python_state_.reset();
#endif
    }

private:
    bool detach_python_ = false;
#ifndef BUILD_LIBTORCH
    std::optional<torch_npu::utils::PythonThreadStateDetachGuard> python_state_;
#endif
};

prefix::Registry& PrefixRegistry()
{
    // Explicit repository retirement owns cleanup; avoid static teardown order.
    static auto* registry = new prefix::Registry;
    return *registry;
}

std::exception_ptr PrefixStoppedError()
{
    static const auto error = std::make_exception_ptr(
        std::runtime_error("FIFO prefix cancelled before successful event record"));
    return error;
}

const void* PrefixEventKey(void* parameters)
{
    const auto* task = static_cast<queue::QueueParas*>(parameters);
    if (task->paramType != queue::RECORD_EVENT) {
        return nullptr;
    }
    return static_cast<queue::EventParas*>(task->paramVal)->event;
}

void PrefixStartGeneration(const Repository* repo)
{
    // Existing lifecycle has finished all old producer/consumer leases.
    PrefixRegistry().FailQueue(repo, PrefixStoppedError());
    PrefixRegistry().Quiescent(repo);
    PrefixRegistry().StartQueue(repo);
}

void PrefixQuiescentIfStopped(const Repository* repo, const queue::QueueSnapshot& state)
{
    if (state.mode == queue::QueueMode::Stopped && state.active_producers == 0 && !state.consumer_active) {
        PrefixRegistry().FailQueue(repo, PrefixStoppedError());
        PrefixRegistry().Quiescent(repo);
    }
}

class PrefixAclEvent final : public prefix::DeviceEvent {
public:
    explicit PrefixAclEvent(const NPUStream& stream) : device_(stream.device_index()), stream_(stream)
    {
        NPUGuard guard(device_);
        LazySetDevice(device_);
        flags_ = acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
        const auto ret = acl::AclrtCreateEventWithFlag(&event_, flags_);
        NPU_CHECK_ERROR(ret);
        TORCH_CHECK(ret == ACL_ERROR_NONE && event_ != nullptr,
            "FIFO prefix event creation did not succeed with a nonnull handle, ret=", ret,
            PTA_ERROR(ErrCode::ACL));
    }

    ~PrefixAclEvent() override
    {
        try {
            if (!NpuSysCtrl::GetInstance().GetInitFlag()) {
                return;
            }
            NPUGuard guard(device_);
            // This private event has at most one outstanding record. Rejected
            // publication and metadata-only discard may leave its manager
            // count behind, after all actual raw users have quiesced.
            auto& manager = NPUEventManager::GetInstance();
            try {
                if (!manager.IsEventRecorded(event_)) {
                    manager.DecreaseUnrecordedCount(event_);
                }
            } catch (...) {
                // ClearUnrecordedCount may have concurrently removed it.
            }
            // Existing flag-aware policy, without enqueueing a destructor task.
            NPU_CHECK_ERROR(manager.LazyDestroy(event_));
            if (!acl::IsExistCreateEventExWithFlag()) {
                NPU_CHECK_ERROR(manager.QueryAndDestroyEvent());
            }
        } catch (...) {
            ASCEND_LOGE("FIFO prefix event destruction failed.");
        }
    }

    void Wait() override
    {
        NPUGuard guard(device_);
        const auto ret = aclrtSynchronizeEvent(event_);
        NPU_CHECK_ERROR(ret);
        // NPU_CHECK_ERROR intentionally warns for some unsupported features.
        // A completion boundary requires the original raw return to be zero.
        TORCH_CHECK(ret == ACL_ERROR_NONE,
            "FIFO prefix event wait did not complete successfully, ret=", ret,
            PTA_ERROR(ErrCode::ACL));
    }

    aclrtEvent Event() const { return event_; }
    unsigned int Flags() const { return flags_; }
    c10::DeviceIndex Device() const { return device_; }
    const NPUStream& Stream() const { return stream_; }
    std::shared_ptr<const prefix::QueueRunToken> run_token;
    std::mutex reuse_mu;
    std::exception_ptr poisoned;

private:
    c10::DeviceIndex device_;
    const NPUStream stream_;
    aclrtEvent event_ = nullptr;
    unsigned int flags_ = ACL_EVENT_DEFAULT;
};

thread_local std::weak_ptr<PrefixAclEvent> h2d_prefix_owner;
thread_local const void* h2d_marker_event = nullptr;
thread_local const prefix::QueueRunToken* h2d_marker_token = nullptr;
thread_local bool h2d_prefix_active = false;

class PrefixH2DMarkerScope final {
public:
    PrefixH2DMarkerScope(const std::shared_ptr<PrefixAclEvent>& owner, bool active)
        : old_event_(h2d_marker_event), old_token_(h2d_marker_token), old_active_(h2d_prefix_active)
    {
        if (active) {
            h2d_marker_event = owner->Event();
            h2d_marker_token = owner->run_token.get();
            h2d_prefix_active = true;
        }
    }
    ~PrefixH2DMarkerScope()
    {
        h2d_marker_event = old_event_;
        h2d_marker_token = old_token_;
        h2d_prefix_active = old_active_;
    }
private:
    const void* old_event_;
    const prefix::QueueRunToken* old_token_;
    bool old_active_;
};

[[noreturn]] void PrefixPoisonAndRethrow(
    const std::shared_ptr<PrefixAclEvent>& owner, const prefix::Registry::Ticket& ticket,
    std::exception_ptr error)
{
    owner->poisoned = std::move(error);
    if (ticket) {
        PrefixRegistry().Abandon(owner->Event(), owner->poisoned);
    }
    if (ticket && ticket->Published()) {
        try { emptyAllNPUStream(); }
        catch (...) { owner->poisoned = std::current_exception(); throw; }
    }
    std::rethrow_exception(owner->poisoned);
}

// Caller owns reuse_mu and is detached from Python. The caller's outer scope
// must release reuse_mu before restoring Python state on return or exception.
bool PrefixWaitLocked(const std::shared_ptr<PrefixAclEvent>& owner, bool h2d)
{
    if (owner->poisoned) {
        std::rethrow_exception(owner->poisoned);
    }
    prefix::Registry::Ticket ticket;
    try {
        TORCH_CHECK(getCurrentNPUStream() == owner->Stream() && current_device() == owner->Device(),
            "FIFO prefix event owner moved to another stream/device.", PTA_ERROR(ErrCode::NOT_SUPPORT));
        if (owner->run_token && !owner->run_token->running.load(std::memory_order_acquire)) {
            if (h2d) {
                return false;
            }
            TORCH_CHECK(false, "FIFO prefix owner generation is inactive; recreate the factory.",
                PTA_ERROR(ErrCode::INTERNAL));
        }
        ticket = PrefixRegistry().Register(owner->Event(), owner);
        PrefixH2DMarkerScope marker_scope(owner, h2d);
        const auto ret = queue::LaunchRecordEventTask(owner->Event(), owner->Stream(), owner->Flags());
        NPU_CHECK_ERROR(ret);
        TORCH_CHECK(ret == ACL_ERROR_NONE && ticket->Published(),
            "FIFO prefix marker was not accepted successfully by its repository.", PTA_ERROR(ErrCode::INTERNAL));
        auto token = ticket->QueueToken();
        TORCH_CHECK(token && (!owner->run_token || owner->run_token == token),
            "FIFO prefix owner crossed a repository generation; recreate the factory.",
            PTA_ERROR(ErrCode::INTERNAL));
        if (h2d && !token->running.load(std::memory_order_acquire)) {
            throw prefix::PrefixNotRunning();
        }
        QueueWaitHooks wait_hooks(false);
        ticket->WaitDevice(wait_hooks);
        owner->run_token = std::move(token);
        return true;
    } catch (...) {
        PrefixPoisonAndRethrow(owner, ticket, std::current_exception());
    }
}

bool PrefixTryH2DDetached(const std::shared_ptr<PrefixAclEvent>& owner, const NPUStream& stream)
{
    if (h2d_prefix_active) {
        return false;
    }
    std::unique_lock<std::mutex> reuse(owner->reuse_mu, std::try_to_lock);
    TORCH_CHECK(reuse.owns_lock(), "FIFO prefix H2D event owner used concurrently.", PTA_ERROR(ErrCode::INTERNAL));
    if (owner->poisoned) {
        std::rethrow_exception(owner->poisoned);
    }
    prefix::H2DPrewaitGuard guard;
    guard.bound_owner = owner->run_token != nullptr;
    guard.default_stream = stream == getDefaultNPUStream(stream.device_index());
    guard.same_owner_stream_device = stream == owner->Stream() && current_device() == owner->Device();
    guard.task_queue = option::OptionsManager::GetTaskQueueEnable() != 0;
    guard.shared_queue = option::OptionsManager::GetPerStreamQueue() == 0;
    guard.running_generation = owner->run_token &&
        owner->run_token->running.load(std::memory_order_acquire);
    guard.resume_mode = option::OptionsManager::IsResumeModeEnable();
    guard.stream_timeout = option::OptionsManager::GetACLExecTimeout();
    guard.sync_debug = warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED;
#ifndef BUILD_LIBTORCH
    guard.trace_active = impl::NPUTrace::getTrace() != nullptr;
#endif
    guard.sync_launch = stream.isSyncLaunchStream();
    guard.reentrant = h2d_prefix_active;
    if (!prefix::AllowsH2DPrewait(guard)) {
        return false;
    }
    return PrefixWaitLocked(owner, true);
}

} // namespace

NativeQueueBatchSnapshot GetNativeQueueBatchSnapshot() noexcept
{
    NativeQueueBatchSnapshot result;
#ifndef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
    auto& counters = queue::GetOrderedBatchCounters();
    result.fast_calls = counters.calls.load(std::memory_order_relaxed);
    result.committed_bursts = counters.bursts.load(std::memory_order_relaxed);
    result.committed_items = counters.items.load(std::memory_order_relaxed);
#endif
    result.fallback_calls = native_batch_fallback_calls.load(std::memory_order_relaxed);
    return result;
}

namespace register_queue_cb {
NPUCallBackRegisterBuilder::NPUCallBackRegisterBuilder(const ACL_EXEC_FUNC &execFunc, const ACL_COPY_FUNC &copyFunc,
    const ACL_RELEASE_FUNC &releaseFunc, const ACL_NEW_FUNC &newFunc, const ACL_DELETE_FUNC &deleteFunc,
    const ACL_COPY_RELEASE_PARM_FUNC &copyReleaseParamF, const ACL_RELEASE_PARAM_FUNC &releaseParamF)
{
    manager().SetExec(execFunc);
    manager().SetCopy(copyFunc);
    manager().SetRelease(releaseFunc);
    manager().SetNew(newFunc);
    manager().SetDelete(deleteFunc);
    releaseManager().SetCopyReleaseParam(copyReleaseParamF);
    releaseManager().SetReleaseParam(releaseParamF);
    releaseManager().SetNew(newFunc);
    releaseManager().SetDelete(deleteFunc);
}
} // namespace register_queue_cb


// If the capacity is too large, when the queue is full,
// a large amount of device memory is occupied at the same time;
// if the capacity is too small, and the main thread is fast enough,
// it does not make full use of concurrent design capabilities.
std::unordered_map<RepoStatus, std::string> deviceErrorMap = {
    {RepoStatus::UCE_EXIT, "UCE ERROR"},
    {RepoStatus::HBM_ECC_EXIT, "HBM MULTI BIT ECC ERROR"},
    {RepoStatus::STOP_EXIT, "FORCE STOP"},
    {RepoStatus::SUSPECT_MEM_EXIT, "SUSPECT MEM ERROR"},
    {RepoStatus::HCCS_LINK_EXIT, "HCCS LINK ERROR"},
    {RepoStatus::HCCL_OP_RETRY_EXIT, "HCCL OP RETRY FAILED"},
    {RepoStatus::SUSPECT_REMOTE_EXIT, "SUSPECT REMOTE ERROR"}
};

std::string get_func_error_msg(void *error_paras)
{
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(error_paras);
    if (queueParam->paramVal == nullptr) {
        return "captured queue parameter already retired";
    }
    auto type = queueParam->paramType;
    std::stringstream result;
    if (type == c10_npu::queue::EXECUTE_OPAPI) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApi *>(queueParam->paramVal);
        auto op_name = cur_paras->opType;
        result << "the current working operator name is " << op_name;
    } else if (type == c10_npu::queue::EXECUTE_OPAPI_V2) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApiV2 *>(queueParam->paramVal);
        auto op_name = cur_paras->opName;
        result << "the current working operator name is " << *op_name;
    } else if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
        auto cur_paras = static_cast<at_npu::native::ExecuteParas *>(queueParam->paramVal);
        auto op_name = cur_paras->opType;
        // Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
        result << "the current working operator name is " << op_name;
    } else if (type == c10_npu::queue::ASYNC_MEMCPY) {
        auto cur_paras = static_cast<c10_npu::queue::CopyParas *>(queueParam->paramVal);
        result << "the current copy params are srclen=" << cur_paras->srcLen << ", dstlen=" << cur_paras->dstLen <<
            ", kind=" << cur_paras->kind;
    } else {
        auto cur_paras = static_cast<c10_npu::queue::EventParas *>(queueParam->paramVal);
        result << "the current working event is " << cur_paras->event;
    }
    return result.str();
}

ErrorRecord Repository::SnapshotError() const
{
    std::lock_guard<std::mutex> lock(error_mu_);
    return error_;
}

void Repository::RecordRepositoryError(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.repository = std::move(message);
}

void Repository::RecordAclError(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.acl = std::move(message);
}

void Repository::RecordCurrentParameter(std::string message)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.current_parameter = std::move(message);
}

RepoStatus Repository::GetStatus() const
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call GetStatus(). !!");
    }

    return repo_status.load();
}

void Repository::SetStatus(RepoStatus desired)
{
    if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
        if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
            ASCEND_LOGE("Task queue is not initialized, shouldn't call SetStatus(). !!");
            return;
        }
        repo_status.store(desired, std::memory_order_release);
        PrefixRegistry().SetQueueRunning(this, desired == RepoStatus::RUN);
        ++status_epoch_;
        queue_core_.RequestStop();
        if (desired != RepoStatus::NEED_EXIT) {
            PrefixRegistry().FailQueue(this, PrefixStoppedError());
        }
        return;
    }

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    RepoStatus previous_status = RepoStatus::INIT;
    uint64_t previous_epoch = 0;
    {
        std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mu_);
        while (lifecycle_.load(std::memory_order_acquire) == Lifecycle::Initializing) {
            wait_hooks.BeforeWait();
            try {
                lifecycle_cv_.wait(lifecycle_lock);
            } catch (...) {
                if (lifecycle_lock.owns_lock()) {
                    lifecycle_lock.unlock();
                }
                wait_hooks.AfterWait();
                throw;
            }
            lifecycle_lock.unlock();
            wait_hooks.AfterWait();
            lifecycle_lock.lock();
        }
        const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
        TORCH_CHECK(
            lifecycle == Lifecycle::Running,
            "Task queue cannot activate from its current lifecycle state.",
            PTA_ERROR(ErrCode::INTERNAL));
        previous_status = repo_status.load(std::memory_order_acquire);
        previous_epoch = status_epoch_;
        lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    }

    try {
        const queue::QueueMode mode = queue_core_.Snapshot().mode;
        if (mode == queue::QueueMode::Failed || mode == queue::QueueMode::Stopping) {
            PrefixRegistry().FailQueue(this, PrefixStoppedError());
        }
        if (mode == queue::QueueMode::Failed) {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } else if (mode == queue::QueueMode::Stopping) {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                wait_hooks);
        }
        PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
            // A STOP/error published while cleanup was in progress wins over
            // this restart. The caller can retry after observing that status.
            if (status_epoch_ == previous_epoch &&
                repo_status.load(std::memory_order_acquire) == previous_status) {
                if (queue_core_.Snapshot().mode == queue::QueueMode::Stopped) {
                    PrefixStartGeneration(this);
                    queue_core_.Start();
                    PrefixRegistry().SetQueueRunning(this, GetStatus() == RepoStatus::RUN);
                }
                TORCH_CHECK(
                    queue_core_.Snapshot().mode == queue::QueueMode::Running,
                    "Task queue recovery did not restart the ordered queue core.",
                    PTA_ERROR(ErrCode::INTERNAL));
                repo_status.store(desired, std::memory_order_release);
                PrefixRegistry().SetQueueRunning(this, desired == RepoStatus::RUN);
                ++status_epoch_;
            }
            lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
    } catch (const std::exception& error) {
        RecordRepositoryError(std::string("Task queue recovery failed: ") + error.what());
        try {
            // A cleanup callback may have released only part of its payload.
            // Drop metadata without retrying that callback.
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } catch (...) {
            ASCEND_LOGE("Task queue recovery metadata cleanup failed.");
        }
        PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
            lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
        throw;
    } catch (...) {
        RecordRepositoryError("Task queue recovery failed with an unknown exception.");
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } catch (...) {
            ASCEND_LOGE("Task queue recovery metadata cleanup failed.");
        }
        PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
        {
            std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
            lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
        throw;
    }
}

void Repository::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }
    if (repo_status.load(std::memory_order_acquire) != expected) {
        return;
    }
    if (desired == RepoStatus::RUN || desired == RepoStatus::INIT) {
        TORCH_CHECK(
            lifecycle == Lifecycle::Running &&
                queue_core_.Snapshot().mode == queue::QueueMode::Running,
            "Task queue cannot publish an active status while its core is stopped.",
            PTA_ERROR(ErrCode::INTERNAL));
        repo_status.store(desired, std::memory_order_release);
        PrefixRegistry().SetQueueRunning(this, desired == RepoStatus::RUN);
        ++status_epoch_;
    } else {
        repo_status.store(desired, std::memory_order_release);
        PrefixRegistry().SetQueueRunning(this, desired == RepoStatus::RUN);
        ++status_epoch_;
        queue_core_.RequestStop();
        if (desired != RepoStatus::NEED_EXIT) {
            PrefixRegistry().FailQueue(this, PrefixStoppedError());
        }
    }
}

NPUStatus Repository::MakeSureQueueEmpty(bool check_error)
{
    host_regions::RejectEffect("queue drain");
    TORCH_NPU_QUEUE_LOGD("MakeSureQueueEmpty: start, device = %d, queued = %zu, status = %d",
        device_idx, queue_core_.Snapshot().size, GetStatus());
    std::string error_msg;
    std::string runtime_error;
    if (!CheckInit()) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call MakeSureQueueEmpty(). !!");
        return NPU_STATUS_FAILED;
    }
    ASCEND_LOGI("Begin to makesure taskqueue empty.");
    if (lazy_fusion::IsEnabled()) {
        lazy_fusion::LazyFusionFlush();
    }
    // While waiting for ACL thread to launch tasks,
    // the current thread should not hold GIL.
    // When the operator compilation is triggered in the ACL thread,
    // the TE module attempts to obtain the GIL.
    // If the current thread does not release the GIL, a deadlock will
    // occur.
    if (consumer.joinable()) {
#ifndef BUILD_LIBTORCH
        QueueWaitHooks wait_hooks(g_used_aclop);
#else
        QueueWaitHooks wait_hooks(false);
#endif
        queue_core_.WaitDrained(wait_hooks);
    }

    // ACL submission completion is not retirement of the eligible native
    // Tensor/Storage owners moved to the bounded release lane. Keep that work
    // inside the unchanged caller synchronization/timing boundary.
    releaseQueue.WaitForEligibleOwners();

    const ErrorRecord error = SnapshotError();

    const RepoStatus current_status = GetStatus();
    auto iter = deviceErrorMap.find(current_status);
    if (iter != deviceErrorMap.end()) {
        std::string throwError = iter->second;
        std::string device_detail;
        if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
            device_detail = c10_npu::c10_npu_get_error_message();
        }
        runtime_error = throwError + ", " + device_detail + PTA_ERROR(ErrCode::ACL);
        error_msg = throwError + " happened.";
    }

    if (current_status == RepoStatus::CAN_EXIT) {
        error_msg = "Inner error happened with CAN_EXIT status, detail: " + error.repository;
    }

    auto errmsg = GetQueueErrMsg();
    bool cannOOM = isCannOOM(errmsg);
    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        if (c10_npu::option::OptionsManager::IsOomSnapshotEnable() && cannOOM) {
            c10_npu::option::oom_observer();
        }

        runtime_error = "The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            error.repository + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + error.acl;
        error_msg = "Inner error happened, detail: " + error.repository;
    }

    if (!error_msg.empty()) {
        ASCEND_LOGE("%s", error_msg.c_str());
    }
    if (check_error && !runtime_error.empty()) {
        ASCEND_LOGE("runtime_error: %s", runtime_error.c_str());
        if (cannOOM) {
            TORCH_CHECK_WITH(OutOfMemoryError, false, runtime_error.c_str());
        } else {
            throw std::runtime_error(runtime_error);
        }
    }
    TORCH_NPU_QUEUE_LOGD("MakeSureQueueEmpty: clearing successful, device = %d, queued = %zu, status = %d",
        device_idx, queue_core_.Snapshot().size, GetStatus());

    return NPU_STATUS_SUCCESS;
}

void Repository::CheckDeviceError(int ret, std::string& err_msg)
{
    std::string acl_error;
    if (ret != ACL_ERROR_RT_DEVICE_TASK_ABORT && ret != ACL_ERROR_RT_DEVICE_MEM_ERROR) {
        acl_error = c10_npu::c10_npu_get_error_message();
        RecordAclError(acl_error);
    }
    if (ret == ACL_ERROR_RT_DEVICE_MEM_ERROR ||
        acl_error.find(DEVICE_MEM_ERROR) != std::string::npos ||
        acl_error.find(DEVICE_MEM_ERROR_V2) != std::string::npos) {
        if (checkUceErrAndRepair(false, err_msg)) {
            ASCEND_LOGE("UCE ERROR happened, set task queue status to UCE_EXIT");
            SetStatus(UCE_EXIT);
        }
    } else if (ret == ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR ||
               acl_error.find(DEVICE_HBM_ECC_ERROR) != std::string::npos ||
               acl_error.find(DEVICE_HBM_ECC_ERROR_V2) != std::string::npos) {
        record_mem_hbm_ecc_error();
        SetStatus(HBM_ECC_EXIT);
    } else if (ret == ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR ||
               acl_error.find(SUSPECT_DEVICE_MEM_ERROR) != std::string::npos ||
               acl_error.find(SUSPECT_DEVICE_MEM_ERROR_V2) != std::string::npos) {
        ASCEND_LOGE("SUSPECT MEM ERROR happened, set task queue status to SUSPECT_MEM_EXIT");
        SetStatus(SUSPECT_MEM_EXIT);
    } else if (ret == ACL_ERROR_RT_LINK_ERROR ||
               acl_error.find(HCCS_LINK_ERROR) != std::string::npos ||
               acl_error.find(HCCS_LINK_ERROR_V2) != std::string::npos ||
               acl_error.find(HCCS_LINK_ERROR_V3) != std::string::npos) {
        ASCEND_LOGE("HCCS LINK ERROR happened, set task queue status to HCCS_LINK_EXIT");
        SetStatus(HCCS_LINK_EXIT);
    } else if (ret == ACL_ERROR_RT_COMM_OP_RETRY_FAIL ||
               acl_error.find(HCCL_OP_RETRY_FAILED) != std::string::npos ||
               acl_error.find(HCCL_OP_RETRY_FAILED_V2) != std::string::npos) {
        ASCEND_LOGE("HCCL OP RETRY FAILED happened, set task queue status to HCCL_OP_RETRY_EXIT");
        SetStatus(HCCL_OP_RETRY_EXIT);
    } else if (ret == ACL_ERROR_RT_SUSPECT_REMOTE_ERROR ||
               acl_error.find(SUSPECT_REMOTE_ERROR) != std::string::npos ||
               acl_error.find(SUSPECT_REMOTE_ERROR_V2) != std::string::npos) {
        ASCEND_LOGE("SUSPECT REMOTE ERROR happened, set task queue status to SUSPECT_REMOTE_EXIT");
        SetStatus(SUSPECT_REMOTE_EXIT);
    } else if (GetStatus() != STOP_EXIT) {
        SetStatus(ERROR_EXIT);
    }
}

bool Repository::ReadQueue()
{
    bool execution_failed = false;
    const bool consumed = queue_core_.ConsumeOne([this, &execution_failed](uint32_t slot) {
        const void* prefix_event = PrefixEventKey(manager().getCurrentParams(data, slot));
        auto prefix_lease = prefix_event ? PrefixRegistry().Acquire(prefix_event) : prefix::Registry::Ticket{};
        const std::string current_para = get_func_error_msg(manager().getCurrentParams(data, slot));
        RecordCurrentParameter(current_para);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(2, manager().getCurrentParams(data, slot), 0);
        const auto ret = manager().Call(data, slot);
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(3, manager().getCurrentParams(data, slot), 0);
#else
        const auto ret = manager().Call(data, slot);
#endif
        host_regions::QueueConsumed(manager().getCurrentParams(data, slot), ret);
        if (ret != 0) {
            if (prefix_lease) {
                // This receipt describes this record attempt, never an old event state.
                PrefixRegistry().Recorded(prefix_event, prefix_lease,
                    std::make_exception_ptr(std::runtime_error(
                        "FIFO prefix event record failed, ACL error=" + std::to_string(ret))));
            }
            execution_failed = true;
            std::string repository_error = current_para;
            RecordRepositoryError(repository_error);
            ASCEND_LOGE("---Thread---%llu: device = %d, slot = %u, status = %d, ret = %d",
                std::this_thread::get_id(), device_idx, slot, GetStatus(), ret);
            TORCH_NPU_QUEUE_LOGI("ReadQueue: read failed, %s, device = %d, slot = %u, status = %d, ret = %d",
                repository_error.c_str(), device_idx, slot, GetStatus(), ret);
            manager().Release(data, slot, releaseQueue);
            std::string err_msg;
            CheckDeviceError(ret, err_msg);
            if (!err_msg.empty()) {
                repository_error += ". Other error information exists:" + err_msg;
                RecordRepositoryError(repository_error);
            }
            if (GetStatus() == RepoStatus::STOP_EXIT) {
                const auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
                ASCEND_LOGI("ReadQueue: SynchronizeDevice with FORCE STOP, device = %d, slot = %u, "
                            "status = %d, ret = %d, acl_ret = %d",
                    device_idx, slot, GetStatus(), ret, acl_ret);
            }
            return;
        }

        manager().Release(data, slot, releaseQueue);
        RecordCurrentParameter({});
        TORCH_NPU_QUEUE_LOGD("ReadQueue: read success, %s, device = %d, slot = %u, status = %d",
            current_para.c_str(), device_idx, slot, GetStatus());
        if (GetStatus() == RepoStatus::STOP_EXIT) {
            const auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
            ASCEND_LOGI("ReadQueue: SynchronizeDevice with FORCE STOP, device = %d, slot = %u, "
                        "status = %d, ret = %d, acl_ret = %d",
                device_idx, slot, GetStatus(), ret, acl_ret);
        }
        // Acknowledge only after real record and parameter release succeeded.
        if (prefix_lease) {
            PrefixRegistry().Recorded(prefix_event, prefix_lease);
        }
    });
    if (execution_failed) {
        ClearQueue();
        c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
        return false;
    }
    return consumed;
}

void Repository::ThrowDeviceError(RepoStatus current_status, void* cur_paras)
{
    auto iter = deviceErrorMap.find(current_status);
    if (iter == deviceErrorMap.end()) {
        return;
    }
    std::string throwError = iter->second;
    auto queueParam = static_cast<c10_npu::queue::QueueParas *>(cur_paras);
    auto type = queueParam->paramType;
    // The RECORD_EVENT in the destructor process should not throw an exception.
    if (type == c10_npu::queue::LAZY_DESTROY_EVENT || type == c10_npu::queue::RECORD_EVENT) {
        return;
    }
    ASCEND_LOGE("getUceErrorFlag in Enqueue, throw %s.", throwError.c_str());
    std::string device_error_msg;
    if (current_status != RepoStatus::STOP_EXIT && current_status != RepoStatus::UCE_EXIT) {
        device_error_msg = c10_npu::c10_npu_get_error_message();
    }
    throw std::runtime_error(throwError + ", " + device_error_msg + PTA_ERROR(ErrCode::ACL));
}

void Repository::Enqueue(void *cur_paras)
{
    TORCH_CHECK(
        lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
        "Task queue is not ready to accept a producer.",
        PTA_ERROR(ErrCode::INTERNAL));

    TORCH_NPU_QUEUE_LOGD("Enqueue: start, %s, device = %d, queued = %zu, status = %d",
        get_func_error_msg(cur_paras).c_str(), device_idx, queue_core_.Snapshot().size, GetStatus());

    const RepoStatus current_status = GetStatus();
    ThrowDeviceError(current_status, cur_paras);
    const ErrorRecord error =
        (current_status == RepoStatus::CAN_EXIT || current_status == RepoStatus::ERROR_EXIT)
        ? SnapshotError() : ErrorRecord{};

    if (current_status == RepoStatus::CAN_EXIT) {
        ASCEND_LOGE("Inner error happened with CAN_EXIT status, detail: %s", error.repository.c_str());
    }

    if (current_status == RepoStatus::ERROR_EXIT) {
        // Avoid repeatedly throwing exceptions
        SetStatus(CAN_EXIT);

        auto retmsg = std::string("The Inner error is reported as above. "
            "The process exits for this inner error, and " +
            error.repository + ".\n" +
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace, "
            "please set the environment variable ASCEND_LAUNCH_BLOCKING=1.\n" +
            "Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, "
            "resulting in performance degradation. "
            "Please unset ASCEND_LAUNCH_BLOCKING in time after debugging." +
            PTA_ERROR(ErrCode::ACL) + ".\n" + error.acl);

        auto errmsg = GetQueueErrMsg();
        if (isCannOOM(errmsg)) {
            if (c10_npu::option::OptionsManager::IsOomSnapshotEnable()) {
                c10_npu::option::oom_observer();
            }
            ASCEND_LOGE("%s", retmsg.c_str());
            TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
        }
        throw std::runtime_error(retmsg);
    }

    if (current_status != RUN && current_status != INIT) {
        auto queueParam = static_cast<c10_npu::queue::QueueParas *>(cur_paras);
        auto type = queueParam->paramType;
        if (type == c10_npu::queue::EXECUTE_OPAPI) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApi *>(queueParam->paramVal);
            auto op_name = cur_paras->opType;
            ASCEND_LOGE("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name);
        } else if (type == c10_npu::queue::EXECUTE_OPAPI_V2) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParasOpApiV2 *>(queueParam->paramVal);
            auto op_name = cur_paras->opName;
            ASCEND_LOGE("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name->c_str());
        } else if (type == c10_npu::queue::COMPILE_AND_EXECUTE) {
            auto cur_paras = static_cast<at_npu::native::ExecuteParas *>(queueParam->paramVal);
            auto op_name = cur_paras->opType;
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for executing and op name is=%s.", op_name);
        } else if (type == c10_npu::queue::ASYNC_MEMCPY) {
            auto cur_paras = static_cast<c10_npu::queue::CopyParas *>(queueParam->paramVal);
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for copy, srclen=%zu, dstlen is %zu, kind=%d",
                cur_paras->srcLen, cur_paras->dstLen, cur_paras->kind);
        } else {
            auto cur_paras = static_cast<c10_npu::queue::EventParas *>(queueParam->paramVal);
            ASCEND_LOGW("Task queue thread is exit, can't call Enqueue() for event, event is=%p", cur_paras->event);
        }
        return;
    }

    if (lazy_fusion::IsEnabled()) {
        auto type = static_cast<c10_npu::queue::QueueParas *>(cur_paras)->paramType;
        // LAZY_DESTROY_EVENT may execute in a separate thread, parallel with forward and backward threads.
        if (type != c10_npu::queue::LAZY_DESTROY_EVENT) {
            lazy_fusion::LazyFusionFlush();
        }
    }

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    const bool accepted = queue_core_.Enqueue(
        [this, cur_paras](uint32_t slot) {
            ThrowDeviceError(GetStatus(), cur_paras);
            // The deep copy is deliberately inside the core's short producer
            // commit section. Publishing tail/size before this returns would
            // expose a partially copied parameter record to the ACL thread.
            const void* event = PrefixEventKey(cur_paras);
            const auto* expected = event && event == h2d_marker_event ? h2d_marker_token : nullptr;
            if (expected && GetStatus() != RepoStatus::RUN) {
                throw prefix::PrefixNotRunning();
            }
            manager().Copy(data, slot, cur_paras);
            host_regions::QueueCopyAccepted(cur_paras);
            eager_prepare::QueueCopyAccepted(cur_paras);
            if (event) {
                PrefixRegistry().Published(event, this, expected);
            }
        },
        wait_hooks);
    if (!accepted) {
        const RepoStatus rejected_status = GetStatus();
        ThrowDeviceError(rejected_status, cur_paras);
        TORCH_CHECK(
            rejected_status != RepoStatus::RUN && rejected_status != RepoStatus::INIT,
            "active task queue rejected a producer",
            PTA_ERROR(ErrCode::INTERNAL));
        TORCH_NPU_QUEUE_LOGI("Enqueue: producer admission closed, device = %d, status = %d",
            device_idx, rejected_status);
    }
}

void Repository::EnqueueBatch(void* const* items, size_t count,
    void (*prepare)(size_t, void*), void* context)
{
    if (count == 0) {
        return;
    }
    TORCH_CHECK(items != nullptr && prepare != nullptr,
        "Batch queue requires items and a prepare callback.", PTA_ERROR(ErrCode::PARAM));
    for (size_t i = 0; i != count; ++i) {
        TORCH_CHECK(items[i] != nullptr, "Batch queue item is null.", PTA_ERROR(ErrCode::PARAM));
    }
    const bool packet = host_regions::IsPublishingPacketCommand(items[0]);
    if (packet && GetStatus() == RepoStatus::INIT) {
        // Packet parameters are still private here. Finish the original INIT
        // drain before taking one core producer lease for the complete batch.
        MakeSureQueueEmpty();
        ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
    }
    // INIT requires the original per-item drain/status transition, and lazy
    // fusion may publish work of its own. Do not fold either protocol into a
    // running queue's batch commit section.
    bool fallback = GetStatus() != RepoStatus::RUN || lazy_fusion::IsEnabled();
#ifdef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
    fallback = true;
#endif
    TORCH_CHECK(!packet || !fallback,
        "Captured packets require a running ordered batch queue; no item was accepted.",
        PTA_ERROR(ErrCode::INTERNAL));
    if (fallback) {
        native_batch_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        for (size_t i = 0; i != count; ++i) {
            prepare(i, context);
            Enqueue(items[i]);
            host_regions::CheckCapturedPublishAccepted(items[i]);
            if (GetStatus() == RepoStatus::INIT) {
                MakeSureQueueEmpty();
                ChangeStatus(RepoStatus::INIT, RepoStatus::RUN);
            }
        }
        return;
    }
    TORCH_CHECK(lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
        "Task queue is not ready to accept a batch producer.", PTA_ERROR(ErrCode::INTERNAL));
#ifndef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    struct BatchStopped { RepoStatus status; };
    size_t accepted = 0;
    try {
        accepted = queue_core_.EnqueueBatch(count,
        [this, items, prepare, context](uint32_t slot, size_t index) {
            void* parameter = items[index];
            const RepoStatus status = GetStatus();
            ThrowDeviceError(status, parameter);
            // Preserve per-item device/error checks. Normal status changes can
            // close admission while a batch is waiting for queue capacity.
            if (status != RepoStatus::RUN) {
                throw BatchStopped{status};
            }
            prepare(index, context);
            const void* event = PrefixEventKey(parameter);
            const auto* expected = event && event == h2d_marker_event ? h2d_marker_token : nullptr;
            if (expected && GetStatus() != RepoStatus::RUN) {
                throw prefix::PrefixNotRunning();
            }
            manager().Copy(data, slot, parameter);
            host_regions::QueueCopyAccepted(parameter);
            if (event) {
                PrefixRegistry().Published(event, this, expected);
            }
        }, wait_hooks);
    } catch (const BatchStopped& stopped) {
        // Status publication and OOM hooks can acquire queue/lifecycle locks
        // or reenter. Run them only after the batch commit mutex is released.
        if (stopped.status == RepoStatus::ERROR_EXIT) {
            SetStatus(CAN_EXIT);
            const auto error = SnapshotError();
            const std::string message = "The Inner error is reported as above. " +
                error.repository + ".\n" + error.acl + PTA_ERROR(ErrCode::ACL);
            if (isCannOOM(GetQueueErrMsg())) {
                if (c10_npu::option::OptionsManager::IsOomSnapshotEnable()) {
                    c10_npu::option::oom_observer();
                }
                TORCH_CHECK_WITH(OutOfMemoryError, false, message.c_str());
            }
            throw std::runtime_error(message);
        }
        TORCH_CHECK(false,
            "Task queue stopped before accepting a batch item.", PTA_ERROR(ErrCode::INTERNAL));
    }
    TORCH_CHECK(accepted == count,
        "Task queue stopped after accepting ", accepted, " of ", count, " batch items.",
        PTA_ERROR(ErrCode::INTERNAL));
#endif
}

void Repository::Dequeue()
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Task queue is not initialized, shouldn't call Dequeue(). !!");
        return;
    }

    while (GetStatus() != RepoStatus::CAN_EXIT) {
        if (deviceErrorMap.find(GetStatus()) != deviceErrorMap.end()) {
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            continue;
        }
        bool consumed = false;
        try {
            consumed = ReadQueue();
        } catch (const std::exception& error) {
            RecordRepositoryError(error.what());
            SetStatus(RepoStatus::ERROR_EXIT);
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            break;
        } catch (...) {
            RecordRepositoryError("unknown exception in task queue consumer");
            SetStatus(RepoStatus::ERROR_EXIT);
            ClearQueue();
            c10_npu::NPUEventManager::GetInstance().ClearUnrecordedCount();
            break;
        }
        if (!consumed) {
            if (GetStatus() == RepoStatus::NEED_EXIT) {
                ChangeStatus(NEED_EXIT, CAN_EXIT);
                break;
            }

            if (GetStatus() == RepoStatus::ERROR_EXIT) {
                break;
            }

            if (GetStatus() == RepoStatus::STOP_EXIT) {
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                continue;
            }
        }
    }
}

void Repository::ReleaseResource()
{
    manager().DeInit(data);
    data = nullptr;
}

void Repository::ClearQueue()
{
    RepoStatus status = RepoStatus::INIT;
    uint64_t epoch = 0;
    queue::QueueSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        if (lifecycle_.load(std::memory_order_acquire) != Lifecycle::Running) {
            // Initialization, restart, another cleanup, or destruction owns
            // the transition. It will leave the core in a coherent state.
            return;
        }
        status = repo_status.load(std::memory_order_acquire);
        epoch = status_epoch_;
        snapshot = queue_core_.Snapshot();

        // A consumer may have decided to clear under STOP just before another
        // thread completed restart. Once the new generation is Running, stale
        // cleanup must not touch it.
        if ((status == RepoStatus::RUN || status == RepoStatus::INIT) &&
            snapshot.mode == queue::QueueMode::Running) {
            return;
        }
        lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    }

#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    PrefixRegistry().FailQueue(this, PrefixStoppedError());
    bool cleanup_failed = false;
    try {
        if (snapshot.failed) {
            // The callback that owns the head may have partially released it.
            // Metadata-only discard prevents a non-idempotent retry.
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } else {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                wait_hooks);
        }
    } catch (const std::exception& error) {
        cleanup_failed = true;
        RecordRepositoryError(std::string("Task queue cleanup failed: ") + error.what());
        ASCEND_LOGE("Task queue cleanup failed, discarding metadata: %s", error.what());
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } catch (...) {
            ASCEND_LOGE("Task queue metadata cleanup failed.");
        }
    } catch (...) {
        cleanup_failed = true;
        RecordRepositoryError("Task queue cleanup failed with an unknown exception.");
        ASCEND_LOGE("Task queue cleanup failed, discarding metadata.");
        try {
            queue_core_.StopAndDiscardMetadata(wait_hooks);
            host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
        } catch (...) {
            ASCEND_LOGE("Task queue metadata cleanup failed.");
        }
    }

    PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
        if (cleanup_failed) {
            repo_status.store(RepoStatus::ERROR_EXIT, std::memory_order_release);
            ++status_epoch_;
        } else if (
            status_epoch_ == epoch &&
            repo_status.load(std::memory_order_acquire) == status &&
            (status == RepoStatus::RUN || status == RepoStatus::INIT) &&
            queue_core_.Snapshot().mode == queue::QueueMode::Stopped) {
            PrefixStartGeneration(this);
            queue_core_.Start();
            PrefixRegistry().SetQueueRunning(this, GetStatus() == RepoStatus::RUN);
        }
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
    }
    lifecycle_cv_.notify_all();
}

void Repository::SetQueueErrMsg(const char *errmsg)
{
    std::lock_guard<std::mutex> lock(error_mu_);
    error_.queue = errmsg == nullptr ? std::string{} : std::string(errmsg);
}

std::string Repository::GetQueueErrMsg()
{
    std::lock_guard<std::mutex> lock(error_mu_);
    return error_.queue;
}

Repository::~Repository()
{
    {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        const Lifecycle lifecycle = lifecycle_.load(std::memory_order_relaxed);
        if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
            return;
        }
        lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
    }

    repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
    PrefixRegistry().SetQueueRunning(this, false);
    queue_core_.RequestStop();
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    try {
        // Drain with runtime-aware wait hooks before joining. Once this
        // returns, the consumer has no leased task and join only observes its
        // loop exit; an attached Python thread never blocks the task callback.
        queue_core_.StopAndDrain(wait_hooks);
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue drain failed, discarding pending tasks: %s", error.what());
        PrefixRegistry().FailQueue(this, std::current_exception());
        try {
            // StopAndDrain waits for active producers before surfacing a
            // shared callback error, so this fallback has no producer wait.
            if (queue_core_.Snapshot().failed) {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
                host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
            } else {
                queue_core_.StopAndDiscard(
                    [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                    wait_hooks);
            }
        } catch (const std::exception& discard_error) {
            ASCEND_LOGE("Task queue cleanup failed: %s", discard_error.what());
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
                host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        } catch (...) {
            ASCEND_LOGE("Task queue cleanup failed with an unknown exception.");
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
                host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        }
    } catch (...) {
        ASCEND_LOGE("Task queue drain failed with an unknown exception.");
        PrefixRegistry().FailQueue(this, std::current_exception());
        try {
            if (queue_core_.Snapshot().failed) {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
                host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
            } else {
                queue_core_.StopAndDiscard(
                    [this](uint32_t slot) { manager().Release(data, slot, releaseQueue); },
                    wait_hooks);
            }
        } catch (...) {
            ASCEND_LOGE("Task queue cleanup failed with an unknown exception.");
            try {
                queue_core_.StopAndDiscardMetadata(wait_hooks);
                host_regions::DiscardCapturedBufferOwners(data);
            host_regions::QueueAbandoned(device_idx);
            } catch (...) {
                ASCEND_LOGE("Task queue metadata cleanup failed.");
            }
        }
    }
    if (consumer.joinable()) {
        consumer.join();
    }
    try {
        releaseQueue.Shutdown();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue shutdown failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue shutdown failed with an unknown exception.");
    }
    try {
        ReleaseResource();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue storage cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Task queue storage cleanup failed with an unknown exception.");
    }
    PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
    lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
    lifecycle_cv_.notify_all();
}

bool Repository::CheckInit() const
{
    return lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running;
}

void StartConsume(Repository *repo, c10::DeviceIndex device_id)
{
    host_regions::CapturedRetirementWorkerScope retirement_worker;
    try {
        SetThreadType(ThreadType::ACL_THREAD);
        SetThreadAffinity(device_id);

        aclError ret = c10_npu::SetDevice(device_id);
        if (ret != 0) {
            C10_NPU_SHOW_ERR_MSG();
            ASCEND_LOGE("***Thread*%d: set device (%d): ret = %d", std::this_thread::get_id(), device_id, ret);
        }

        while (repo->GetStatus() != RepoStatus::CAN_EXIT and repo->GetStatus() != RepoStatus::ERROR_EXIT) {
            repo->Dequeue();
        }
    } catch (const std::exception& error) {
        ASCEND_LOGE("Task queue consumer escaped its dequeue loop: %s", error.what());
        try {
            repo->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Task queue consumer could not publish ERROR_EXIT.");
        }
    } catch (...) {
        ASCEND_LOGE("Task queue consumer escaped its dequeue loop with an unknown exception.");
        try {
            repo->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Task queue consumer could not publish ERROR_EXIT.");
        }
    }
}

void Repository::InitRepo(c10::DeviceIndex device_id)
{
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    const Lifecycle current = lifecycle_.load(std::memory_order_acquire);
    if (current == Lifecycle::Running) {
        return;
    }
    TORCH_CHECK(current == Lifecycle::Uninitialized || current == Lifecycle::Stopped,
        "Task queue cannot initialize from its current lifecycle state.", PTA_ERROR(ErrCode::INTERNAL));
    lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);

    try {
        device_idx = device_id;
        repo_status.store(RepoStatus::INIT, std::memory_order_release);
        ++status_epoch_;
        {
            std::lock_guard<std::mutex> error_lock(error_mu_);
            error_ = ErrorRecord{};
        }
        data = manager().Init(static_cast<int>(queue_core_.Capacity()));
        ASCEND_LOGI("TaskQueue is enable");
        PrefixStartGeneration(this);
        queue_core_.Start();
        PrefixRegistry().SetQueueRunning(this, GetStatus() == RepoStatus::RUN);

        // Release storage must be live before the main consumer can transfer
        // ownership to it.
        releaseQueue.InitReleaseQueue(device_id);
        std::thread cur_consumer(StartConsume, this, device_id);
        consumer = std::move(cur_consumer);
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
        lifecycle_cv_.notify_all();
    } catch (...) {
        const std::exception_ptr initialization_error = std::current_exception();
        queue_core_.RequestStop();
        if (consumer.joinable()) {
            repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
            try {
                consumer.join();
            } catch (...) {
                ASCEND_LOGE("Task queue initialization consumer cleanup failed.");
            }
        }
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization metadata cleanup failed.");
        }
        try {
            releaseQueue.Shutdown();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization release cleanup failed.");
        }
        try {
            ReleaseResource();
        } catch (...) {
            ASCEND_LOGE("Task queue initialization storage cleanup failed.");
        }
        data = nullptr;
        PrefixQuiescentIfStopped(this, queue_core_.Snapshot());
        lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
        lifecycle_cv_.notify_all();
        std::rethrow_exception(initialization_error);
    }
}

std::string Repository::GetPara()
{
    const ErrorRecord error = SnapshotError();
    if (!error.current_parameter.empty()) {
        return error.current_parameter;
    }
    std::string repo_para = "EmptyQueue";
    queue_core_.InspectHead([this, &repo_para](uint32_t slot) {
        repo_para = get_func_error_msg(manager().getCurrentParams(data, slot));
    });
    return repo_para;
}

namespace prefix {
namespace {
std::function<void()> MakePrefixWaiter(bool enable_h2d)
{
    TORCH_CHECK(option::OptionsManager::GetTaskQueueEnable(),
        "FIFO prefix completion requires task queues.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(!option::OptionsManager::GetPerStreamQueue(),
        "FIFO prefix completion requires the shared default-stream queue.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    const auto stream = getCurrentNPUStream();
    TORCH_CHECK(stream == getDefaultNPUStream(stream.device_index()) && !stream.isSyncLaunchStream(),
        "FIFO prefix completion requires the current default stream.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    auto owner = std::make_shared<PrefixAclEvent>(stream);
    if (enable_h2d) {
        TORCH_CHECK(h2d_prefix_owner.expired(), "A FIFO prefix H2D factory is already active on this producer.",
            PTA_ERROR(ErrCode::INTERNAL));
        h2d_prefix_owner = owner;
    }
    return [owner] {
        std::unique_lock<std::mutex> reuse(owner->reuse_mu, std::try_to_lock);
        TORCH_CHECK(reuse.owns_lock(), "FIFO prefix event owner used concurrently.", PTA_ERROR(ErrCode::INTERNAL));
        (void)PrefixWaitLocked(owner, false);
    };
}
} // namespace

C10_NPU_API std::function<void()> MakeDefaultStreamPrefixWaiter()
{
    return MakePrefixWaiter(false);
}

C10_NPU_API std::function<void()> MakeDefaultStreamPrefixH2DWaiter()
{
    return MakePrefixWaiter(true);
}

C10_NPU_API bool TryH2DPrewait(const NPUStream& stream)
{
    auto owner = h2d_prefix_owner.lock();
    if (!owner) {
        return false;
    }
#ifndef BUILD_LIBTORCH
    // Declare before invoking the inner function. Its reuse/ticket/registry
    // locks all unwind first, including exceptions, before Python restoration.
    torch_npu::utils::PythonThreadStateDetachGuard detach(true);
#endif
    return PrefixTryH2DDetached(owner, stream);
}

} // namespace prefix

void ReleaseQueue::PushToReleaseQueue(void *cur_paras)
{
    TORCH_CHECK(
        lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running,
        "Release queue is not ready to accept a parameter.",
        PTA_ERROR(ErrCode::INTERNAL));

    const bool accepted = queue_core_.Enqueue(
        [this, cur_paras](uint32_t slot) { releaseManager().CopyRealseParam(data, slot, cur_paras); },
        queue::NoopWaitHooks{});
    TORCH_CHECK(accepted, "Release queue stopped before accepting a parameter.", PTA_ERROR(ErrCode::INTERNAL));
}

void ReleaseQueue::PopFromReleaseQueue()
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call PopFromReleaseQueue(). !!");
        return;
    }

    const bool consumed = queue_core_.ConsumeOne(
        [this](uint32_t slot) { releaseManager().ReleaseParam(data, slot); });
    if (!consumed && GetStatus() == RepoStatus::NEED_EXIT) {
        ChangeStatus(NEED_EXIT, CAN_EXIT);
    }
}

void StartRelease(ReleaseQueue *releaseQue)
{
    host_regions::CapturedRetirementWorkerScope retirement_worker;
    try {
        SetThreadType(ThreadType::RELEASE_THREAD);
        SetThreadAffinity(releaseQue->GetDeviceID());

        while (releaseQue->GetStatus() != RepoStatus::CAN_EXIT) {
            releaseQue->PopFromReleaseQueue();
        }
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue consumer failed: %s", error.what());
        try {
            releaseQue->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Release queue consumer could not publish ERROR_EXIT.");
        }
    } catch (...) {
        ASCEND_LOGE("Release queue consumer failed with an unknown exception.");
        try {
            releaseQue->SetStatus(RepoStatus::ERROR_EXIT);
        } catch (...) {
            ASCEND_LOGE("Release queue consumer could not publish ERROR_EXIT.");
        }
    }
}

void ReleaseQueue::InitReleaseQueue(c10::DeviceIndex device_id)
{
    std::lock_guard<std::mutex> lock(lifecycle_mu_);
    const Lifecycle current = lifecycle_.load(std::memory_order_acquire);
    if (current == Lifecycle::Running) {
        return;
    }
    TORCH_CHECK(current == Lifecycle::Uninitialized || current == Lifecycle::Stopped,
        "Release queue cannot initialize from its current lifecycle state.", PTA_ERROR(ErrCode::INTERNAL));
    lifecycle_.store(Lifecycle::Initializing, std::memory_order_release);
    try {
        device_idx = device_id;
        repo_status.store(RepoStatus::INIT, std::memory_order_release);
        data = releaseManager().Init(static_cast<int>(queue_core_.Capacity()));
        queue_core_.Start();
        std::thread cur_releaser(StartRelease, this);
        releaser = std::move(cur_releaser);
        lifecycle_.store(Lifecycle::Running, std::memory_order_release);
    } catch (...) {
        const std::exception_ptr initialization_error = std::current_exception();
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Release queue initialization metadata cleanup failed.");
        }
        try {
            releaseManager().DeInit(data);
        } catch (...) {
            ASCEND_LOGE("Release queue initialization storage cleanup failed.");
        }
        data = nullptr;
        lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
        std::rethrow_exception(initialization_error);
    }
}

ReleaseQueue::~ReleaseQueue()
{
    try {
        Shutdown();
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue cleanup failed with an unknown exception.");
    }
}

void ReleaseQueue::WaitForEligibleOwners()
{
    // Never wait for work whose completion includes this very ACL/release
    // thread. Ordinary queues without moved capture owners retain the old
    // fast path, rather than draining all ACL descriptor releases per request.
    if (host_regions::IsCapturedRetirementWorker() || data == nullptr ||
        !host_regions::HasCapturedOwnerRetirement(data)) {
        return;
    }
#ifndef BUILD_LIBTORCH
    QueueWaitHooks wait_hooks(g_used_aclop);
#else
    QueueWaitHooks wait_hooks(false);
#endif
    try {
        queue_core_.WaitDrained(wait_hooks);
    } catch (...) {
        const auto error = std::current_exception();
        // A partially executed ReleaseParam must never run again. Shutdown
        // joins the failed worker and DeleteFunc only retires the additional
        // capture owners in its constructed slots, without retrying ACL.
        try {
            Shutdown();
        } catch (...) {
            ASCEND_LOGE("Captured owner retirement cleanup also failed.");
        }
        std::rethrow_exception(error);
    }
    TORCH_CHECK(!host_regions::HasCapturedOwnerRetirement(data),
        "Release queue drained before captured native owner retirement completed.",
        PTA_ERROR(ErrCode::INTERNAL));
}

void ReleaseQueue::Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        const Lifecycle current = lifecycle_.load(std::memory_order_relaxed);
        if (current == Lifecycle::Uninitialized || current == Lifecycle::Stopped ||
            current == Lifecycle::Stopping) {
            return;
        }
        lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
    }

    if (repo_status.load(std::memory_order_acquire) != RepoStatus::ERROR_EXIT) {
        repo_status.store(RepoStatus::NEED_EXIT, std::memory_order_release);
    }
    queue_core_.RequestStop();
    if (releaser.joinable()) {
        releaser.join();
    }

    const bool release_failed = queue_core_.Snapshot().failed;
    if (release_failed) {
        // ConsumeOne retains the failing head. ReleaseParam is not idempotent,
        // so a callback that partially destroyed it must never be retried.
        try {
            queue_core_.StopAndDiscard();
        } catch (...) {
            ASCEND_LOGE("Release queue metadata cleanup failed.");
        }
    } else {
        try {
            queue_core_.StopAndDiscard(
                [this](uint32_t slot) { releaseManager().ReleaseParam(data, slot); });
        } catch (const std::exception& error) {
            ASCEND_LOGE("Release queue cleanup failed, discarding metadata: %s", error.what());
            try {
                queue_core_.StopAndDiscard();
            } catch (...) {
                ASCEND_LOGE("Release queue metadata cleanup failed.");
            }
        } catch (...) {
            ASCEND_LOGE("Release queue cleanup failed, discarding metadata.");
            try {
                queue_core_.StopAndDiscard();
            } catch (...) {
                ASCEND_LOGE("Release queue metadata cleanup failed.");
            }
        }
    }
    try {
        releaseManager().DeInit(data);
    } catch (const std::exception& error) {
        ASCEND_LOGE("Release queue storage cleanup failed: %s", error.what());
    } catch (...) {
        ASCEND_LOGE("Release queue storage cleanup failed with an unknown exception.");
    }
    data = nullptr;
    lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
}

RepoStatus ReleaseQueue::GetStatus() const
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call GetStatus(). !!");
    }

    return repo_status.load();
}

c10::DeviceIndex ReleaseQueue::GetDeviceID() const
{
    return device_idx;
}


void ReleaseQueue::SetStatus(RepoStatus desired)
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call SetStatus(). !!");
        return;
    }

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    repo_status = desired;
    if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
        queue_core_.RequestStop();
    } else if (queue_core_.Snapshot().mode == queue::QueueMode::Stopped) {
        queue_core_.Start();
    }
}

void ReleaseQueue::ChangeStatus(RepoStatus expected, RepoStatus desired)
{
    const Lifecycle lifecycle = lifecycle_.load(std::memory_order_acquire);
    if (lifecycle == Lifecycle::Uninitialized || lifecycle == Lifecycle::Stopped) {
        ASCEND_LOGE("Release queue is not initialized, shouldn't call ChangeStatus(). !!");
        return;
    }

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    if (repo_status.compare_exchange_strong(expected, desired)) {
        if (desired != RepoStatus::RUN && desired != RepoStatus::INIT) {
            queue_core_.RequestStop();
        } else if (queue_core_.Snapshot().mode == queue::QueueMode::Stopped) {
            queue_core_.Start();
        }
    }
}
} // namespace c10_npu
