#pragma once
#include "op_api_owned_payload_r22.h"
#include "torch_npu/csrc/core/npu/EagerPrepare.h"
#include <limits>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include "torch_npu/csrc/core/npu/EagerPrepareEligibility.h"

namespace hdb_opapi {
inline constexpr bool EagerNameEquals(const char* a,const char* b) {
    while (*a && *b && *a==*b) { ++a; ++b; } return *a=='\0' && *b=='\0';
}
inline constexpr bool HasEagerTypedSchema(const char* name) {
    return EagerNameEquals(name,"aclnnEmbeddingBag") || EagerNameEquals(name,"aclnnAdd") ||
        EagerNameEquals(name,"aclnnRelu") || EagerNameEquals(name,"aclnnMm");
}
TORCH_NPU_API void CheckEagerSdkFunctions(const char* name);
TORCH_NPU_API OwnedSdkFunctions GetEagerSdkFunctions(const char* name);
inline bool EagerSdkQualified(const char* name) noexcept {
    return c10_npu::eager_prepare::SdkContractAvailable(name);
}
inline void CheckEagerSdkProfile(const char* name) {
    CheckEagerSdkFunctions(name); // Recheck provider and cleanup poison at every SDK entry.
}
inline OwnedSdkFunctions EagerFunctions(const char* name) {
    return GetEagerSdkFunctions(name);
}
TORCH_NPU_API OwnedTensor SnapshotEagerOwnedTensor(NativeOwners&,const at::Tensor&,
    c10::optional<aclDataType> forced_type=c10::nullopt);
struct EagerSnapshotter : OwnedSnapshotter {
    using OwnedSnapshotter::operator();
    explicit EagerSnapshotter(NativeOwners& owners) : OwnedSnapshotter{owners} {}
    OwnedTensor operator()(const at::Tensor& value) const { return SnapshotEagerOwnedTensor(owners,value); }
    OwnedTensor operator()(const TensorWrapper& value) const { return SnapshotEagerOwnedTensor(owners,value.tensor_,value.dtype); }
    template<class T> auto operator()(const c10::optional<T>& value) const {
        using Item=std::decay_t<decltype((*this)(std::declval<const T&>()))>;
        if (!value.has_value()) return c10::optional<Item>{};
        return c10::optional<Item>{(*this)(value.value())};
    }
};

template<class Arguments>
class EagerOwnedCommand final : public c10_npu::host_regions::OwnedOpApiPayload {
    using Converted=ConvertedPack<Arguments,OwnedConvertPolicy>;
public:
    EagerOwnedCommand(const char* operation,NativeOwners&& owners,Arguments&& arguments,bool v1,CacheParams config)
        : operation_(operation),owners_(std::move(owners)),arguments_(std::move(arguments)),
          functions_(EagerFunctions(operation)),v1_(v1),config_(config) {}
    ~EagerOwnedCommand() noexcept override { Cancel(); }
    void Prepare(void* actual_stream) override {
        TORCH_CHECK(!prepare_attempted_,"eager SDK preparation cannot be retried"); prepare_attempted_=true;
        CheckEagerSdkProfile(operation_); stream_=GetAclStream();
        TORCH_CHECK(actual_stream==stream_,"eager SDK preparation stream changed");
        const auto worker_config=GetCacheParams();
        TORCH_CHECK(worker_config.GetDeterministicStatus()==config_.GetDeterministicStatus() &&
            worker_config.GetAicNum()==config_.GetAicNum() && worker_config.GetAivNum()==config_.GetAivNum(),
            "eager caller/worker cache configuration changed before SDK preparation");
        try {
            // Cache semantics are currently qualified only for EmbeddingBag.
            cache_active_=!c10_npu::eager_prepare::BuildUsesHostSimulation() &&
                EagerNameEquals(operation_,"aclnnEmbeddingBag") && BeginOwnedCache(v1_,config_,stream_);
            if (cache_active_) {
                std::apply([](const auto&... value) { (HashOwnedArgument(value),...); },arguments_);
                if (auto* cached=FindOwnedCache(v1_,&workspace_size_)) {
                    executor_.Adopt(true,cached); EndOwnedCache(); cache_active_=false; return;
                }
            }
            const auto init_status=functions_.init(nullptr,false);
            TORCH_CHECK(init_status==0,"eager HugeMem initialization failed"); pool_pending_=true;
            converted_.emplace(arguments_,OwnedConvertPolicy{v1_,&conversion_cleanup_failed_,&cleanup_failure_});
            aclOpExecutor* candidate=nullptr;
            auto parameters=std::tuple_cat(converted_->Get(),std::make_tuple(&workspace_size_,&candidate));
            const auto query=ConvertToOpApiFunc(parameters,functions_.query);
            const auto status=call(query,parameters);
            TORCH_CHECK(status==0,"eager GetWorkspaceSize failed, status=",status);
            executor_.Adopt(true,candidate); functions_.uninit(nullptr,false);
            if (cache_active_) { EndOwnedCache(); cache_active_=false; }
        } catch (...) { CleanupHost(); functions_.uninit(nullptr,false); throw; }
    }
    int Execute(void* actual_stream) override {
        CheckEagerSdkProfile(operation_);
        TORCH_CHECK(executor_.Prepared() && actual_stream==stream_,"eager Execute has no prepared plan or changed stream");
        if (c10_npu::check_dequeue_need_use(stream_)) c10_npu::UseStreamResInCurrentThread(stream_);
        if (workspace_size_) workspace_=AllocateOwnedWorkspace(workspace_size_,stream_,!v1_);
        void* workspace=workspace_.defined() ? const_cast<void*>(workspace_.storage().data()) : nullptr;
        auto* executor=executor_.TakeForExecute();
        struct Cleanup { EagerOwnedCommand& command; ~Cleanup() noexcept { command.CleanupHost(); } } cleanup{*this};
        return functions_.execute(workspace,workspace_size_,executor,stream_);
    }
    void Cancel() noexcept override { if (!executor_.MayHaveSubmitted()) CleanupHost(); }
    bool MayHaveSubmitted() const noexcept override { return executor_.MayHaveSubmitted(); }
    std::exception_ptr CleanupFailure() const noexcept override { return cleanup_failure_; }
    size_t RetainedBytes() const noexcept override {
        if (workspace_size_>std::numeric_limits<size_t>::max()-owners_.Bytes()) return std::numeric_limits<size_t>::max();
        return owners_.Bytes()+static_cast<size_t>(workspace_size_);
    }
private:
    void CleanupHost() noexcept {
        if (host_cleaned_) return; host_cleaned_=true;
        bool safe=!conversion_cleanup_failed_;
        try { executor_.Cancel([this](aclOpExecutor* p) { TORCH_CHECK(functions_.destroy(p)==0,"eager executor cancellation failed"); }); }
        catch (...) { if (!cleanup_failure_) cleanup_failure_=std::current_exception(); safe=false; PoisonOwnedSdkCleanup(); }
        if (converted_ && !converted_->Reset()) { safe=false; PoisonOwnedSdkCleanup(); }
        converted_.reset();
        if (pool_pending_ && safe) functions_.release(nullptr,false);
        pool_pending_=false;
        if (cache_active_) { EndOwnedCache(); cache_active_=false; }
        if (!v1_) workspace_=at::Tensor{};
    }
    const char* operation_;
    NativeOwners owners_;
    Arguments arguments_;
    const OwnedSdkFunctions functions_;
    bool v1_;
    CacheParams config_; // Caller snapshot: never read worker-default core/cache configuration.
    aclrtStream stream_=nullptr;
    uint64_t workspace_size_=0;
    ExecutorOwnership<aclOpExecutor*> executor_;
    std::optional<Converted> converted_;
    at::Tensor workspace_;
    bool prepare_attempted_=false,host_cleaned_=false,cache_active_=false,pool_pending_=false,conversion_cleanup_failed_=false;
    std::exception_ptr cleanup_failure_;
};

template<class Tuple,std::size_t... Index>
auto EagerDecayedTuple(std::index_sequence<Index...>) ->
    std::tuple<std::decay_t<std::tuple_element_t<Index,Tuple>>...>;
// Arity must be checked before any tuple_element is instantiated. Unrelated
// SDK macros are type-checked even in a discarded non-template macro branch.
// Only this exact EmbeddingBag signature may instantiate tensor/domain reads.
template<class Tuple,bool = (std::tuple_size<Tuple>::value==13)>
struct EagerEmbeddingBagSignature : std::false_type {};
template<class Tuple>
struct EagerEmbeddingBagSignature<Tuple,true> : std::is_same<
    decltype(EagerDecayedTuple<Tuple>(std::make_index_sequence<13>{})),
    std::tuple<at::Tensor,at::Tensor,at::Tensor,bool,int64_t,bool,
        c10::optional<at::Tensor>,bool,int64_t,at::Tensor,at::Tensor,at::Tensor,at::Tensor>> {};

template<class Tuple>
bool EagerArgumentDomain(const char* name,const Tuple& args) {
    if (EagerNameEquals(name,"aclnnEmbeddingBag")) {
        if constexpr (EagerEmbeddingBagSignature<Tuple>::value) {
            const auto& weight=std::get<0>(args); const auto& indices=std::get<1>(args); const auto& offsets=std::get<2>(args);
            if (!weight.defined() || !indices.defined() || !offsets.defined()) return false;
            bool native=c10_npu::eager_prepare::CanQueueTensor(weight) && c10_npu::eager_prepare::CanQueueTensor(indices) &&
                c10_npu::eager_prepare::CanQueueTensor(offsets);
            bool disjoint=true;
            auto output=[&](const at::Tensor& value) {
                native=native && c10_npu::eager_prepare::CanQueueTensor(value);
                if (!native) return;
                const auto* storage=value.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                disjoint=disjoint && storage!=weight.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl() &&
                    storage!=indices.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl() &&
                    storage!=offsets.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            };
            output(std::get<9>(args)); output(std::get<10>(args)); output(std::get<11>(args)); output(std::get<12>(args));
            // ATen's inner forward-only route wraps None as an undefined Tensor.
            const auto& per_sample_weights=std::get<6>(args);
            const c10_npu::eager_prepare::EmbeddingBagDomain domain{native,weight.scalar_type()==at::kFloat,
                indices.scalar_type()==at::kLong && offsets.scalar_type()==at::kLong,
                weight.dim()==2 && indices.dim()==1 && offsets.dim()==1,
                weight.numel()>0 && indices.numel()>0 && offsets.numel()>0,disjoint,
                std::get<3>(args),std::get<4>(args),std::get<5>(args),
                per_sample_weights.has_value() && per_sample_weights.value().defined(),
                std::get<7>(args),std::get<8>(args)<0};
            return domain.Admitted();
        } else return false;
    }
    // Other typed registrations are not SDK-qualified yet. Their argument
    // domains must be supplied along with their independent SDK qualification.
    return false;
}

template<class... Args>
void EnqueueEagerArguments(const char* name,bool v1,Args&&... values) {
    CheckEagerSdkProfile(name);
    c10_npu::host_regions::PrepareOwnedOpApiCallerState();
    (void)GetAclStream(); // Bind the admitted stream resource before freezing caller CacheParams.
    const auto config=GetCacheParams();
    NativeOwners owners; owners.Reserve(sizeof...(Args));
    auto arguments=SnapshotArgs(EagerSnapshotter{owners},std::forward<Args>(values)...);
    std::string signature;
    if (c10_npu::eager_prepare::DiagnosticTracing()) {
        std::ostringstream out; size_t position=0;
        auto record=[&](const auto& value) {
            using Value=std::decay_t<decltype(value)>;
            if (position++) out << ';';
            if constexpr (std::is_same<Value,OwnedTensor>::value) {
                if (!value.metadata) { out << "tensor:null"; return; }
                const auto& metadata=*value.metadata;
                out << "acl_dtype=" << static_cast<int>(metadata.acl_type) << ",shape=[";
                for (size_t i=0;i<metadata.sizes.size();++i) { if (i) out << ','; out << metadata.sizes[i]; }
                out << "],stride=[";
                for (size_t i=0;i<metadata.strides.size();++i) { if (i) out << ','; out << metadata.strides[i]; }
                out << "],offset=" << metadata.storage_offset;
            } else out << "scalar_or_optional";
        };
        std::apply([&](const auto&... value) { (record(value),...); },arguments);
        signature=out.str();
        TORCH_CHECK(signature.size()<=4096,"eager cold shape signature exceeds fixed diagnostic bound");
    }
    auto payload=std::make_unique<EagerOwnedCommand<decltype(arguments)>>(name,std::move(owners),std::move(arguments),v1,config);
    c10_npu::eager_prepare::Submit(name,std::move(payload),std::move(signature));
}
} // namespace hdb_opapi

#include "torch_npu/csrc/core/npu/EagerPrepareDispatch.h"
