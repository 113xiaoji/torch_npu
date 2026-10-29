// HOST_SIM_R25: metadata/lifetime simulation only. No numerical kernels, sleeps,
// artificial preparation work, Torch linkage, SDK qualification, or real cache.
#define R25_HOSTSIM_BUILD
#include "r25_hostsim_api.h"
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/syscall.h>
# include <unistd.h>
#endif

namespace {
enum class Kind { Tensor, Scalar, IntArray, FloatArray, BoolArray, TensorList, ScalarList, Executor };
enum class Op : size_t { EmbeddingBag, Add, Relu, Mm };
constexpr const char* names[]={"aclnnEmbeddingBag","aclnnAdd","aclnnRelu","aclnnMm"};
constexpr size_t kMaxRank=64, kMaxArray=1048576, kMaxThreads=256, kMaxEvents=4096;
struct Object {
    Kind kind=Kind::Tensor;
    uint64_t id=0, prepare_tid=0;
    aclDataType dtype=ACL_DT_UNDEFINED;
    aclFormat format=ACL_FORMAT_ND;
    int64_t offset=0;
    void* data=nullptr;
    std::vector<int64_t> shape, strides, storage_shape;
    std::vector<unsigned char> bytes;
    std::vector<uintptr_t> children; // List owns children; executor only borrows.
    uintptr_t parent=0;
    Op operation=Op::EmbeddingBag;
    std::array<int64_t,5> attributes{};
};
struct OpCounts { uint64_t prepared=0, executed=0, cancelled=0, execute_failed=0; };
struct Event { uint64_t sequence=0,id=0,tid=0,prepare_tid=0; Op op=Op::EmbeddingBag; const char* phase=""; int status=0; };
struct ThreadCount { uint64_t id=0,count=0; };
struct State {
    std::mutex mutex;
    // Opaque non-dereferenceable monotonically increasing tokens avoid ABA
    // without retaining heap tombstones. Objects are freed on destroy/consume.
    uintptr_t next_token=0x10000;
    std::unordered_map<uintptr_t,std::unique_ptr<Object>> live;
    uint64_t descriptors_created=0,descriptors_destroyed=0,executors_created=0,
        executors_executed=0,executors_cancelled=0,executors_failed=0,
        errors=0,cache_stub_calls=0,cache_checks=0,cache_cleanup_calls=0,huge_init=0,huge_uninit=0,
        huge_release=0,huge_pending=0,huge_active_threads=0,sequence=0,trace_dropped=0;
    std::array<OpCounts,4> operations{};
    std::array<ThreadCount,kMaxThreads> prepare_tids{},execute_tids{};
    bool tracing=false;
    std::array<Event,kMaxEvents> events{};
    size_t event_count=0;
};
State state;
thread_local const char* error_message="";
thread_local std::string diagnostic;
thread_local bool huge_active=false;
uint64_t tid() noexcept {
#if defined(_WIN32)
    return static_cast<uint64_t>(GetCurrentThreadId());
#else
    return static_cast<uint64_t>(syscall(SYS_gettid));
#endif
}
uintptr_t key(const void* value) noexcept { return reinterpret_cast<uintptr_t>(value); }
int fail(int code,const char* message) noexcept { ++state.errors; error_message=message; return code; }
int exception_status() noexcept { error_message="C++ allocation/internal failure"; return R25_OUT_OF_MEMORY; }
Object* lookup(const void* value,Kind kind) {
    const auto found=state.live.find(key(value));
    if (found==state.live.end()) { fail(R25_STALE_HANDLE,"null, unknown, or stale handle"); return nullptr; }
    if (found->second->kind!=kind) { fail(R25_WRONG_KIND,"wrong descriptor kind"); return nullptr; }
    return found->second.get();
}
size_t scalar_size(aclDataType type) noexcept {
    switch(type) {
        case ACL_FLOAT: case ACL_INT32: case ACL_UINT32: return 4;
        case ACL_FLOAT16: case ACL_INT16: case ACL_UINT16: case ACL_BF16: return 2;
        case ACL_INT8: case ACL_UINT8: case ACL_BOOL: return 1;
        case ACL_INT64: case ACL_UINT64: case ACL_DOUBLE: case ACL_COMPLEX64: return 8;
        case ACL_COMPLEX128: return 16;
        default: return 0;
    }
}
uintptr_t insert(std::unique_ptr<Object> object) {
    if (state.next_token>std::numeric_limits<uintptr_t>::max()-16) throw std::bad_alloc();
    const auto handle=state.next_token; state.next_token+=16;
    object->id=static_cast<uint64_t>(handle/16);
    const bool executor=object->kind==Kind::Executor;
    state.live.emplace(handle,std::move(object));
    if (executor) ++state.executors_created; else ++state.descriptors_created;
    return handle;
}
void trace(const Object& object,const char* phase,int status=0) noexcept {
    if (!state.tracing) return;
    if (state.event_count==kMaxEvents) { ++state.trace_dropped; return; }
    state.events[state.event_count++]={++state.sequence,object.id,tid(),object.prepare_tid,object.operation,phase,status};
}
void count_thread(std::array<ThreadCount,kMaxThreads>& counts,uint64_t id) noexcept {
    for (auto& entry:counts) {
        if (entry.id==id) { ++entry.count; return; }
        if (!entry.id) { entry={id,1}; return; }
    }
}
int destroy_locked(const void* value,Kind kind,bool from_parent=false) {
    auto* object=lookup(value,kind); if (!object) return R25_STALE_HANDLE;
    if (object->parent && !from_parent) return fail(R25_LIFECYCLE_ERROR,"list owns this child descriptor");
    // List destruction is atomic with respect to registry clients.
    if (kind==Kind::TensorList || kind==Kind::ScalarList) {
        for (const auto child:object->children) {
            auto found=state.live.find(child);
            if (found!=state.live.end()) { state.live.erase(found); ++state.descriptors_destroyed; }
        }
    }
    state.live.erase(key(value)); ++state.descriptors_destroyed; return R25_SUCCESS;
}
template<class Pointer> int destroy(Pointer value,Kind kind) noexcept {
    // Ordinary OpAPI tuple cleanup releases absent optional descriptors too.
    // No allocation or lifetime exists for nullptr; stale nonnull tokens fail.
    if (!value) return R25_SUCCESS;
    try { std::lock_guard<std::mutex> lock(state.mutex); return destroy_locked(value,kind); }
    catch (...) { return exception_status(); }
}
template<class Result,class Value> Result* create_array(const Value* values,uint64_t size,Kind kind) noexcept {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (size>kMaxArray || (size && !values)) { fail(R25_INVALID_ARGUMENT,"invalid array pointer/length"); return nullptr; }
        auto object=std::make_unique<Object>(); object->kind=kind;
        object->bytes.resize(static_cast<size_t>(size)*sizeof(Value));
        if (size) std::memcpy(object->bytes.data(),values,object->bytes.size());
        return reinterpret_cast<Result*>(insert(std::move(object)));
    } catch (...) { exception_status(); return nullptr; }
}
template<class Result,class Child> Result* create_list(const Child* const* values,uint64_t size,Kind kind,Kind childkind) noexcept {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (size>kMaxArray || (size && !values)) { fail(R25_INVALID_ARGUMENT,"invalid list pointer/length"); return nullptr; }
        auto object=std::make_unique<Object>(); object->kind=kind; object->children.reserve(static_cast<size_t>(size));
        for (uint64_t i=0;i<size;++i) {
            auto* child=lookup(values[i],childkind);
            if (!child || child->parent) { fail(R25_INVALID_ARGUMENT,"invalid or already owned list child"); return nullptr; }
            for (const auto previous:object->children) if (previous==key(values[i])) {
                fail(R25_INVALID_ARGUMENT,"duplicate list child ownership"); return nullptr;
            }
            object->children.push_back(key(values[i]));
        }
        const auto handle=insert(std::move(object));
        for (auto child:state.live.at(handle)->children) state.live.at(child)->parent=handle;
        return reinterpret_cast<Result*>(handle);
    } catch (...) { exception_status(); return nullptr; }
}
bool tensor_ok(const aclTensor* tensor,aclDataType type=ACL_DT_UNDEFINED) {
    auto* object=lookup(tensor,Kind::Tensor); if (!object) return false;
    if (type!=ACL_DT_UNDEFINED && object->dtype!=type) { fail(R25_INVALID_ARGUMENT,"unsupported tensor dtype for schema"); return false; }
    return true;
}
int prepare_locked(Op op,std::vector<uintptr_t> descriptors,std::array<int64_t,5> attrs,
    const Object* scalar,uint64_t* ws,aclOpExecutor** executor) {
    auto object=std::make_unique<Object>(); object->kind=Kind::Executor; object->operation=op;
    object->prepare_tid=tid(); object->children=std::move(descriptors); object->attributes=attrs;
    if (scalar) object->bytes=scalar->bytes; // Immutable scalar import, no pointer back into source values.
    const auto native_tid=object->prepare_tid;
    count_thread(state.prepare_tids,native_tid);
    const auto handle=insert(std::move(object));
    ++state.operations[static_cast<size_t>(op)].prepared;
    trace(*state.live.at(handle),"prepare");
    *ws=0; *executor=reinterpret_cast<aclOpExecutor*>(handle); return R25_SUCCESS;
}
bool out_args(uint64_t* ws,aclOpExecutor** executor) {
    if (ws) *ws=0;
    if (executor) *executor=nullptr;
    if (!ws || !executor) { fail(R25_INVALID_ARGUMENT,"workspace/executor output pointer is null"); return false; }
    return true;
}
int execute(Op op,void* workspace,uint64_t size,aclOpExecutor* executor,aclrtStream stream) noexcept {
    (void)workspace; (void)stream;
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        auto* object=lookup(executor,Kind::Executor); if (!object) return R25_STALE_HANDLE;
        // Transfer out of the live registry BEFORE validation: phase2 consumes
        // every recognized unique executor, including its failure paths.
        auto owned=std::move(state.live.at(key(executor))); state.live.erase(key(executor));
        int status=R25_SUCCESS;
        if (owned->operation!=op) status=fail(R25_WRONG_KIND,"executor operation differs from phase2 entry");
        else if (size!=0) status=fail(R25_INVALID_ARGUMENT,"simulation workspace size must be zero");
        else for (auto handle:owned->children) {
            if (!state.live.count(handle)) { status=fail(R25_STALE_HANDLE,"borrowed descriptor destroyed before Execute"); break; }
        }
        auto& counts=state.operations[static_cast<size_t>(owned->operation)];
        count_thread(state.execute_tids,tid());
        if (status==0) { ++state.executors_executed; ++counts.executed; }
        else { ++state.executors_failed; ++counts.execute_failed; }
        trace(*owned,"execute",status); return status;
    } catch (...) { return exception_status(); }
}
template<class Values> void json_array(std::ostream& out,const Values& values) {
    out << '['; bool comma=false; for (auto value:values) { if (comma) out << ','; comma=true; out << value; } out << ']';
}
void json_threads(std::ostream& out,const std::array<ThreadCount,kMaxThreads>& values) {
    out << '['; bool comma=false;
    for (const auto& item:values) {
        if (!item.id) break;
        if (comma) out << ',';
        comma=true; out << item.id;
    }
    out << ']';
}
void json_bytes(std::ostream& out,const std::vector<unsigned char>& values) {
    constexpr char digits[]="0123456789abcdef"; out << '"';
    for (auto byte:values) out << digits[byte/16] << digits[byte%16];
    out << '"';
}
void cache_stub() noexcept {
    try { std::lock_guard<std::mutex> lock(state.mutex); ++state.cache_stub_calls;
        fail(R25_UNSUPPORTED,"PTA cache is explicitly unsupported by HOST_SIM_R25"); } catch (...) { exception_status(); }
}
} // namespace

extern "C" {
int R25HostSimAbi() { return 1; }
const char* R25HostSimBackendKind() { return "HOST_SIM_R25"; }
const char* R25HostSimLastError() { return error_message; }
aclTensor* aclCreateTensor(const int64_t* dims,uint64_t rank,aclDataType dtype,const int64_t* strides,
    int64_t offset,aclFormat format,const int64_t* storage,uint64_t storage_rank,void* data) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (rank>kMaxRank || storage_rank>kMaxRank || (rank && (!dims || !strides)) ||
            (storage_rank && !storage) || offset<0 || scalar_size(dtype)==0) {
            fail(R25_INVALID_ARGUMENT,"invalid tensor metadata or dtype"); return nullptr;
        }
        auto object=std::make_unique<Object>(); object->kind=Kind::Tensor; object->dtype=dtype;
        object->format=format; object->offset=offset; object->data=data;
        if (rank) { object->shape.assign(dims,dims+rank); object->strides.assign(strides,strides+rank); }
        if (storage_rank) object->storage_shape.assign(storage,storage+storage_rank);
        bool empty=false;
        for (uint64_t i=0;i<rank;++i) {
            if (dims[i]<0 || strides[i]<0) { fail(R25_INVALID_ARGUMENT,"negative tensor dimension/stride"); return nullptr; }
            empty=empty || dims[i]==0;
        }
        for (uint64_t i=0;i<storage_rank;++i) if (storage[i]<0) { fail(R25_INVALID_ARGUMENT,"negative storage dimension"); return nullptr; }
        if (!data && !empty) { fail(R25_INVALID_ARGUMENT,"nonempty tensor has null data"); return nullptr; }
        return reinterpret_cast<aclTensor*>(insert(std::move(object)));
    } catch (...) { exception_status(); return nullptr; }
}
int aclDestroyTensor(const aclTensor* value) { return destroy(value,Kind::Tensor); }
aclScalar* aclCreateScalar(void* value,aclDataType type) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); const auto bytes=scalar_size(type);
        if (!value || !bytes) { fail(R25_INVALID_ARGUMENT,"invalid scalar pointer/dtype"); return nullptr; }
        auto object=std::make_unique<Object>(); object->kind=Kind::Scalar; object->dtype=type;
        object->bytes.resize(bytes); std::memcpy(object->bytes.data(),value,bytes);
        return reinterpret_cast<aclScalar*>(insert(std::move(object)));
    } catch (...) { exception_status(); return nullptr; }
}
int aclDestroyScalar(const aclScalar* value) { return destroy(value,Kind::Scalar); }
aclIntArray* aclCreateIntArray(const int64_t* values,uint64_t size) { return create_array<aclIntArray>(values,size,Kind::IntArray); }
int aclDestroyIntArray(const aclIntArray* value) { return destroy(value,Kind::IntArray); }
aclFloatArray* aclCreateFloatArray(const float* values,uint64_t size) { return create_array<aclFloatArray>(values,size,Kind::FloatArray); }
int aclDestroyFloatArray(const aclFloatArray* value) { return destroy(value,Kind::FloatArray); }
aclBoolArray* aclCreateBoolArray(const bool* values,uint64_t size) { return create_array<aclBoolArray>(values,size,Kind::BoolArray); }
int aclDestroyBoolArray(const aclBoolArray* value) { return destroy(value,Kind::BoolArray); }
aclTensorList* aclCreateTensorList(const aclTensor* const* values,uint64_t size) { return create_list<aclTensorList>(values,size,Kind::TensorList,Kind::Tensor); }
int aclDestroyTensorList(const aclTensorList* value) { return destroy(value,Kind::TensorList); }
aclScalarList* aclCreateScalarList(const aclScalar* const* values,uint64_t size) { return create_list<aclScalarList>(values,size,Kind::ScalarList,Kind::Scalar); }
int aclDestroyScalarList(const aclScalarList* value) { return destroy(value,Kind::ScalarList); }

int aclnnEmbeddingBagGetWorkspaceSize(const aclTensor* weight,const aclTensor* indices,const aclTensor* offsets,
    bool scale,int64_t mode,bool sparse,const aclTensor* samples,bool last,int64_t padding,
    aclTensor* output,aclTensor* offset2bag,aclTensor* bagsize,aclTensor* maxindices,uint64_t* ws,aclOpExecutor** executor) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); if (!out_args(ws,executor)) return R25_INVALID_ARGUMENT;
        if (!tensor_ok(weight,ACL_FLOAT) || !tensor_ok(indices,ACL_INT64) || !tensor_ok(offsets,ACL_INT64) ||
            !tensor_ok(output,ACL_FLOAT) || !tensor_ok(offset2bag,ACL_INT64) || !tensor_ok(bagsize,ACL_INT64) ||
            !tensor_ok(maxindices,ACL_INT64) || (samples && !tensor_ok(samples,ACL_FLOAT))) return R25_INVALID_ARGUMENT;
        const auto& w=*state.live.at(key(weight)); const auto& idx=*state.live.at(key(indices)); const auto& off=*state.live.at(key(offsets));
        if (w.shape.size()!=2 || idx.shape.size()!=1 || off.shape.size()!=1 || mode<0 || mode>2) return fail(R25_INVALID_ARGUMENT,"unsupported EmbeddingBag metadata/domain");
        std::vector<uintptr_t> borrowed={key(weight),key(indices),key(offsets),key(output),key(offset2bag),key(bagsize),key(maxindices)};
        if (samples) borrowed.push_back(key(samples));
        return prepare_locked(Op::EmbeddingBag,std::move(borrowed),{{scale,mode,sparse,last,padding}},nullptr,ws,executor);
    } catch (...) { return exception_status(); }
}
int aclnnEmbeddingBag(void* ws,uint64_t size,aclOpExecutor* executor,aclrtStream stream) { return execute(Op::EmbeddingBag,ws,size,executor,stream); }
int aclnnAddGetWorkspaceSize(const aclTensor* self,const aclTensor* other,const aclScalar* alpha,
    aclTensor* out,uint64_t* ws,aclOpExecutor** executor) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); if (!out_args(ws,executor)) return R25_INVALID_ARGUMENT;
        if (!tensor_ok(self) || !tensor_ok(other) || !tensor_ok(out)) return R25_INVALID_ARGUMENT;
        const auto* scalar=lookup(alpha,Kind::Scalar); if (!scalar) return R25_INVALID_ARGUMENT;
        return prepare_locked(Op::Add,{key(self),key(other),key(alpha),key(out)},{{0,0,0,0,0}},scalar,ws,executor);
    } catch (...) { return exception_status(); }
}
int aclnnAdd(void* ws,uint64_t size,aclOpExecutor* executor,aclrtStream stream) { return execute(Op::Add,ws,size,executor,stream); }
int aclnnReluGetWorkspaceSize(const aclTensor* self,const aclTensor* out,uint64_t* ws,aclOpExecutor** executor) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); if (!out_args(ws,executor)) return R25_INVALID_ARGUMENT;
        if (!tensor_ok(self) || !tensor_ok(out)) return R25_INVALID_ARGUMENT;
        return prepare_locked(Op::Relu,{key(self),key(out)},{{0,0,0,0,0}},nullptr,ws,executor);
    } catch (...) { return exception_status(); }
}
int aclnnRelu(void* ws,uint64_t size,aclOpExecutor* executor,aclrtStream stream) { return execute(Op::Relu,ws,size,executor,stream); }
int aclnnMmGetWorkspaceSize(const aclTensor* self,const aclTensor* mat2,aclTensor* out,int8_t cube,
    uint64_t* ws,aclOpExecutor** executor) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); if (!out_args(ws,executor)) return R25_INVALID_ARGUMENT;
        if (!tensor_ok(self) || !tensor_ok(mat2) || !tensor_ok(out)) return R25_INVALID_ARGUMENT;
        return prepare_locked(Op::Mm,{key(self),key(mat2),key(out)},{{cube,0,0,0,0}},nullptr,ws,executor);
    } catch (...) { return exception_status(); }
}
int aclnnMm(void* ws,uint64_t size,aclOpExecutor* executor,aclrtStream stream) { return execute(Op::Mm,ws,size,executor,stream); }
int aclDestroyAclOpExecutor(aclOpExecutor* value) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); auto* object=lookup(value,Kind::Executor);
        if (!object) return R25_STALE_HANDLE;
        ++state.executors_cancelled; ++state.operations[static_cast<size_t>(object->operation)].cancelled;
        trace(*object,"cancel"); state.live.erase(key(value)); return R25_SUCCESS;
    } catch (...) { return exception_status(); }
}
int InitHugeMemThreadLocal(void*,bool) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (huge_active) return fail(R25_LIFECYCLE_ERROR,"HugeMem TLS already active");
        huge_active=true; ++state.huge_init; ++state.huge_pending; ++state.huge_active_threads; return R25_SUCCESS;
    } catch (...) { return exception_status(); }
}
void UnInitHugeMemThreadLocal(void*,bool) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!huge_active) { fail(R25_LIFECYCLE_ERROR,"HugeMem TLS not active"); return; }
        huge_active=false; ++state.huge_uninit; --state.huge_active_threads;
    } catch (...) { exception_status(); }
}
void ReleaseHugeMem(void*,bool) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.huge_pending) { fail(R25_LIFECYCLE_ERROR,"HugeMem release has no initialized lease"); return; }
        --state.huge_pending; ++state.huge_release;
    } catch (...) { exception_status(); }
}
bool CanUsePTACache(const char*) {
    try { std::lock_guard<std::mutex> lock(state.mutex); ++state.cache_checks; } catch (...) { exception_status(); }
    return false;
}
void InitPTACacheThreadLocal() { cache_stub(); }
void UnInitPTACacheThreadLocal() {
    // The ordinary OpAPI macros invoke this cleanup even after CanUsePTACache
    // returns false. There is no cache state to clear in this provider.
    try { std::lock_guard<std::mutex> lock(state.mutex); ++state.cache_cleanup_calls; }
    catch (...) { exception_status(); }
}
void SetPTAHashKey(uint64_t) { cache_stub(); }
void SetPTACacheHashKey(uint8_t*,size_t) { cache_stub(); }
aclOpExecutor* PTAGetExecCache(uint64_t,uint64_t* ws) { if (ws) *ws=0; cache_stub(); return nullptr; }
aclOpExecutor* PTAFindExecCache(uint8_t*,size_t,uint64_t* ws) { if (ws) *ws=0; cache_stub(); return nullptr; }
void AddTensorAddrToCachedList(void*) { cache_stub(); }

uint64_t R25HostSimExecutorId(const aclOpExecutor* value) {
    try { std::lock_guard<std::mutex> lock(state.mutex); auto* object=lookup(value,Kind::Executor); return object ? object->id : 0; }
    catch (...) { exception_status(); return 0; }
}
int R25HostSimSetTracing(bool enabled) {
    try { std::lock_guard<std::mutex> lock(state.mutex); state.tracing=enabled; return 0; }
    catch (...) { return exception_status(); }
}
int R25HostSimResetDiagnostics() {
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.live.empty() || state.huge_pending || state.huge_active_threads) return fail(R25_LIFECYCLE_ERROR,"cannot reset while descriptors/executors/pools live");
        state.descriptors_created=state.descriptors_destroyed=state.executors_created=state.executors_executed=
            state.executors_cancelled=state.executors_failed=state.errors=state.cache_stub_calls=state.cache_checks=state.cache_cleanup_calls=
            state.huge_init=state.huge_uninit=state.huge_release=state.sequence=state.trace_dropped=0;
        state.operations={}; state.prepare_tids={}; state.execute_tids={}; state.event_count=0;
        error_message=""; return 0;
    } catch (...) { return exception_status(); }
}
const char* R25HostSimTensorMetadataJson(const aclTensor* value) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); auto* object=lookup(value,Kind::Tensor);
        if (!object) return "{\"error\":\"invalid tensor\"}";
        std::ostringstream out; out << "{\"id\":" << object->id << ",\"dtype\":" << static_cast<int>(object->dtype)
            << ",\"offset\":" << object->offset << ",\"shape\":"; json_array(out,object->shape);
        out << ",\"strides\":"; json_array(out,object->strides); out << ",\"storage_shape\":";
        json_array(out,object->storage_shape); out << '}'; diagnostic=out.str(); return diagnostic.c_str();
    } catch (...) { exception_status(); return "{\"error\":\"diagnostic allocation failed\"}"; }
}
const char* R25HostSimExecutorMetadataJson(const aclOpExecutor* value) {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); auto* object=lookup(value,Kind::Executor);
        if (!object) return "{\"error\":\"invalid executor\"}";
        std::ostringstream out; out << "{\"id\":" << object->id << ",\"operation\":\"" << names[static_cast<size_t>(object->operation)]
            << "\",\"prepare_tid\":" << object->prepare_tid << ",\"attributes\":"; json_array(out,object->attributes);
        out << ",\"scalar_bytes_hex\":"; json_bytes(out,object->bytes); out << '}'; diagnostic=out.str(); return diagnostic.c_str();
    } catch (...) { exception_status(); return "{\"error\":\"diagnostic allocation failed\"}"; }
}
const char* R25HostSimDiagnosticsJson() {
    try {
        std::lock_guard<std::mutex> lock(state.mutex); std::ostringstream out;
        out << "{\"abi\":1,\"backend_kind\":\"HOST_SIM_R25\",\"numeric_supported\":false,\"real_sdk_qualified\":false,"
            "\"pta_cache_supported\":false,\"artificial_prepare_delay_ns\":0";
#define R25_COUNTER(name) out << ",\"" #name "\":" << state.name
        R25_COUNTER(descriptors_created); R25_COUNTER(descriptors_destroyed); R25_COUNTER(executors_created);
        R25_COUNTER(executors_executed); R25_COUNTER(executors_cancelled); R25_COUNTER(executors_failed);
        R25_COUNTER(errors); R25_COUNTER(cache_stub_calls); R25_COUNTER(cache_checks); R25_COUNTER(cache_cleanup_calls);
        R25_COUNTER(huge_init); R25_COUNTER(huge_uninit); R25_COUNTER(huge_release); R25_COUNTER(huge_pending);
        R25_COUNTER(huge_active_threads); R25_COUNTER(trace_dropped);
#undef R25_COUNTER
        out << ",\"live_descriptors\":" << state.descriptors_created-state.descriptors_destroyed
            << ",\"live_executors\":" << state.executors_created-state.executors_executed-state.executors_cancelled-state.executors_failed
            << ",\"prepare_tids\":"; json_threads(out,state.prepare_tids);
        out << ",\"execute_tids\":"; json_threads(out,state.execute_tids);
        out << ",\"tracing_enabled\":" << (state.tracing?"true":"false") << ",\"operations\":{";
        for (size_t i=0;i<4;++i) {
            if (i) out << ',';
            const auto& counts=state.operations[i];
            out << '"' << names[i] << "\":{\"prepared\":" << counts.prepared << ",\"executed\":" << counts.executed
                << ",\"cancelled\":" << counts.cancelled << ",\"execute_failed\":" << counts.execute_failed << '}';
        }
        out << "},\"events\":[";
        for (size_t i=0;i<state.event_count;++i) {
            if (i) out << ',';
            const auto& event=state.events[i];
            out << "{\"sequence\":" << event.sequence << ",\"phase\":\"" << event.phase << "\",\"executor_id\":" << event.id
                << ",\"operation\":\"" << names[static_cast<size_t>(event.op)] << "\",\"tid\":" << event.tid
                << ",\"prepare_tid\":" << event.prepare_tid << ",\"status\":" << event.status << '}';
        }
        out << "]}"; diagnostic=out.str(); return diagnostic.c_str();
    } catch (...) { exception_status(); return "{\"error\":\"diagnostic allocation failed\"}"; }
}
} // extern C
