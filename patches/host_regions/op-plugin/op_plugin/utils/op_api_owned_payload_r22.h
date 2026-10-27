#pragma once
#include "op_api_owned_r22.h"
#include "torch_npu/csrc/core/npu/HostRegionOpApiCommand.h"
#include <optional>

namespace hdb_opapi {
inline constexpr bool HasOwnedOpApiSchema(const char* name)
{
    const char* expected = "aclnnEmbeddingBag";
    if (!name) return false;
    while (*name && *expected && *name == *expected) { ++name; ++expected; }
    return *name == '\0' && *expected == '\0';
}
struct OwnedSdkFunctions {
    void* query;
    OpApiFunc execute;
    int (*destroy)(aclOpExecutor*);
    int (*init)(void*, bool);
    void (*uninit)(void*, bool);
    void (*release)(void*, bool);
};
TORCH_NPU_API const OwnedSdkFunctions& GetOwnedSdkFunctions();
TORCH_NPU_API bool BeginOwnedCache(bool v1, const CacheParams& config, aclrtStream stream);
TORCH_NPU_API aclOpExecutor* FindOwnedCache(bool v1, uint64_t* workspace);
TORCH_NPU_API void EndOwnedCache() noexcept;
TORCH_NPU_API void PoisonOwnedSdkCleanup() noexcept;
TORCH_NPU_API at::Tensor AllocateOwnedWorkspace(uint64_t bytes, aclrtStream stream, bool scratch);

inline void HashOwnedArgument(const OwnedTensor& value) { ::add_param_to_buf_v2(value.metadata); }
template<class T> void HashOwnedArgument(const T& value) { ::add_param_to_buf_v2(value); }
template<class T> void HashOwnedArgument(const c10::optional<T>& value) {
    ::add_param_to_buf_v2(value.has_value());
    if (value.has_value()) HashOwnedArgument(value.value());
}
template<class T> void HashOwnedArgument(const std::vector<T>& value) {
    ::add_param_to_buf_v2(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if constexpr (std::is_same<T, bool>::value) HashOwnedArgument(bool(value[i]));
        else HashOwnedArgument(value[i]);
    }
}

// One stable allocation owns source snapshots and all preparation state. Its
// only SDK preparation entry is called by Capture on the Python worker after
// command/event reservations; Execute runs directly on the original consumer.
template<class Arguments>
class OwnedOpApiCommand final : public c10_npu::host_regions::OwnedOpApiPayload {
    using Converted = ConvertedPack<Arguments, OwnedConvertPolicy>;
public:
    OwnedOpApiCommand(NativeOwners&& owners, Arguments&& arguments, bool v1)
        : owners_(std::move(owners)), arguments_(std::move(arguments)), functions_(GetOwnedSdkFunctions()), v1_(v1) {}
    ~OwnedOpApiCommand() noexcept override { Cancel(); }

    void Prepare(void* actual_stream) override
    {
        TORCH_CHECK(!prepare_attempted_, "owned OpAPI preparation cannot be retried");
        prepare_attempted_ = true;
        stream_ = GetAclStream();
        TORCH_CHECK(actual_stream == stream_, "owned OpAPI preparation stream changed");
        try {
            const auto config = GetCacheParams();
            cache_active_ = BeginOwnedCache(v1_, config, stream_);
            if (cache_active_) {
                std::apply([](const auto&... value) { (HashOwnedArgument(value), ...); }, arguments_);
                if (auto* cached = FindOwnedCache(v1_, &workspace_size_)) {
                    executor_.Adopt(true, cached);
                    cache_hit_ = true;
                    EndOwnedCache(); cache_active_ = false;
                    return;
                }
            }
            const auto init_status = functions_.init(nullptr, false);
            TORCH_CHECK(init_status == 0, "owned OpAPI HugeMem initialization failed");
            pool_pending_ = true;
            converted_.emplace(arguments_, OwnedConvertPolicy{v1_, &conversion_cleanup_failed_, &cleanup_failure_});
            aclOpExecutor* candidate = nullptr;
            auto parameters = std::tuple_cat(converted_->Get(), std::make_tuple(&workspace_size_, &candidate));
            const auto query = ConvertToOpApiFunc(parameters, functions_.query);
            const auto status = call(query, parameters);
            // Exact EmbeddingBag CREATE_EXECUTOR/ReleaseTo transfers only on
            // success. A failed phase1's UniqueExecutor owns its own rollback.
            TORCH_CHECK(status == 0, "owned OpAPI GetWorkspaceSize failed, status=", status);
            executor_.Adopt(true, candidate);
            functions_.uninit(nullptr, false);
            if (cache_active_) { EndOwnedCache(); cache_active_ = false; }
        } catch (...) {
            CleanupHost();
            functions_.uninit(nullptr, false);
            throw;
        }
    }

    int Execute(void* actual_stream) override
    {
        TORCH_CHECK(executor_.Prepared() && actual_stream == stream_, "owned OpAPI execution has no prepared plan or changed stream");
        CheckOwnedSdkProfile("aclnnEmbeddingBag");
        if (c10_npu::check_dequeue_need_use(stream_)) c10_npu::UseStreamResInCurrentThread(stream_);
        if (workspace_size_) workspace_ = AllocateOwnedWorkspace(workspace_size_, stream_, !v1_);
        void* workspace = workspace_.defined() ? const_cast<void*>(workspace_.storage().data()) : nullptr;
        auto* executor = executor_.TakeForExecute(); // SDK owns it before entry, including failure.
        struct HostCleanup {
            OwnedOpApiCommand& command;
            ~HostCleanup() noexcept { command.CleanupHost(); }
        } cleanup{*this};
        return functions_.execute(workspace, workspace_size_, executor, stream_);
    }

    void Cancel() noexcept override
    {
        if (!executor_.MayHaveSubmitted()) CleanupHost();
    }
    bool MayHaveSubmitted() const noexcept override { return executor_.MayHaveSubmitted(); }
    std::exception_ptr CleanupFailure() const noexcept override { return cleanup_failure_; }
    size_t RetainedBytes() const noexcept override
    {
        if (workspace_size_ > std::numeric_limits<size_t>::max() - owners_.Bytes())
            return std::numeric_limits<size_t>::max();
        return owners_.Bytes() + static_cast<size_t>(workspace_size_);
    }

private:
    void CleanupHost() noexcept
    {
        if (host_cleaned_) return;
        host_cleaned_ = true;
        // A failed ConvertedPack constructor has already attempted its prefix
        // rollback. The policy preserves a cleanup fault even though optional
        // emplace never made that partially constructed pack visible here.
        bool safe_pool_release = !conversion_cleanup_failed_;
        try {
            executor_.Cancel([this](aclOpExecutor* executor) {
                const auto status = functions_.destroy(executor);
                TORCH_CHECK(status == 0, "owned OpAPI executor cancellation failed");
            });
        } catch (...) {
            if (!cleanup_failure_) cleanup_failure_ = std::current_exception();
            safe_pool_release = false; PoisonOwnedSdkCleanup();
        }
        // Converted ACL objects may themselves reside in this HugeMem pool.
        // They must be destroyed after executor consumption and before Release.
        if (converted_ && !converted_->Reset()) { safe_pool_release = false; PoisonOwnedSdkCleanup(); }
        converted_.reset();
        if (pool_pending_ && safe_pool_release) functions_.release(nullptr, false);
        pool_pending_ = false;
        if (cache_active_) { EndOwnedCache(); cache_active_ = false; }
        // Queue2 scratch wrappers are not independent leases; release their
        // accounting immediately after the same-stream Execute, as originally.
        // Queue1 caching workspace remains a real owner until the device fence.
        if (!v1_) workspace_ = at::Tensor{};
    }
    NativeOwners owners_;
    Arguments arguments_;
    const OwnedSdkFunctions& functions_;
    bool v1_;
    aclrtStream stream_ = nullptr;
    uint64_t workspace_size_ = 0;
    ExecutorOwnership<aclOpExecutor*> executor_;
    std::optional<Converted> converted_;
    at::Tensor workspace_;
    bool prepare_attempted_ = false, host_cleaned_ = false, cache_active_ = false, cache_hit_ = false, pool_pending_ = false;
    bool conversion_cleanup_failed_ = false;
    std::exception_ptr cleanup_failure_;
};

template<class... Args>
void CaptureOwnedOpApiArguments(const char* name, bool v1, Args&&... values)
{
    TORCH_CHECK(c10_npu::host_regions::UseCapturedOwnedOpApi(), "real owned OpAPI scope has failed or stopped");
    CheckOwnedSdkProfile(name);
    NativeOwners owners;
    owners.Reserve(sizeof...(Args));
    auto arguments = SnapshotArgs(OwnedSnapshotter{owners}, std::forward<Args>(values)...);
    auto payload = std::make_unique<OwnedOpApiCommand<decltype(arguments)>>(std::move(owners), std::move(arguments), v1);
    at_npu::native::OpCommand::RunOwnedOpApi(name, std::move(payload));
}
} // namespace hdb_opapi
