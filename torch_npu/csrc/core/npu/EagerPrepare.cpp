#include "torch_npu/csrc/core/npu/EagerPrepare.h"
#include "torch_npu/csrc/core/npu/EagerPrepareOrder.h"
#include "torch_npu/csrc/core/npu/EagerPrepareOwnedJob.h"
#include "torch_npu/csrc/core/npu/EagerPrepareLifecycle.h"
#include "torch_npu/csrc/core/npu/EagerPrepareStorage.h"
#include "torch_npu/csrc/core/npu/HostRegionNativeTls.h"
#include "torch_npu/csrc/core/npu/HostRegionTensorOwner.h"
#include "torch_npu/csrc/core/npu/HostRegionRetirement.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace c10_npu::eager_prepare {
using Core = queue::EagerPrepareQueueCore;
using Fence = host_regions::device_retirement_detail::CompletionFence;
namespace {
std::atomic<bool> enabled_flag{false}; // Off fast path never allocates runtime state.
std::atomic<bool> trace_flag{false};
struct Counts {
    uint64_t seen=0, eligible=0, queued=0, prepared=0, executed=0, returned=0, fallback=0;
    std::map<std::string,uint64_t> reasons;
    bool qualified=false;
    uint64_t fallback_fences=0;
};
struct Identity {
    std::atomic<uint64_t> sequence{std::numeric_limits<uint64_t>::max()};
    uint64_t caller_tid=0;
    std::atomic<uint64_t> consumer_tid{0};
};
struct Trace {
    std::shared_ptr<Identity> identity;
    std::string operation;
    std::string shape_signature;
    uint64_t native_tid=0,start_ns=0,end_ns=0,start_thread_cpu_ns=0,end_thread_cpu_ns=0;
    bool succeeded=false;
};
struct State {
    std::mutex mu, lifecycle_mu;
    std::condition_variable submitters_changed;
    std::mutex fail_stop_mutex;
    std::condition_variable fail_stop_changed;
    ColdCloseOnce cold_close;
    std::atomic<bool> enabled{false};
    std::atomic<bool> sdk_stop{false};
    Configuration config;
    std::shared_ptr<Core> core;
    std::thread::id caller_thread;
    uint64_t caller_tid=0;
    size_t active_submitters=0;
    std::map<std::string,Counts> operators;
    SequenceOrderEvidence admission_order, execution_order;
    std::exception_ptr first_error;
    std::string error_phase, error_operation;
    uint64_t error_sequence=0;
    bool fifo_drained=false, device_synchronized=false, workers_stopped=true, closure_passed=true;
    size_t peak_slots=0, peak_bytes=0;
    std::vector<Trace> trace;
    uint64_t trace_dropped=0;
    std::atomic<uint64_t> trace_errors{0};
    // Process-lifetime only on a failed ACK. Destroying an unacknowledged
    // completion event or the producer cache TLS would be an unsafe cleanup.
    std::vector<std::unique_ptr<Fence>> quarantined_fences;
};
State& state() { static auto* value = new State; return *value; }
thread_local const void* publication_parameter = nullptr;
thread_local std::atomic<bool>* publication_ack = nullptr;
thread_local Core* publication_core = nullptr;
thread_local const Core::TicketPtr* publication_ticket = nullptr;
thread_local std::atomic<bool>* publication_commit_failed = nullptr;
struct PublicationMarker {
    PublicationMarker(const void* parameter,std::atomic<bool>& ack,Core& core,
                      const Core::TicketPtr& ticket,std::atomic<bool>& commit_failed) {
        TORCH_CHECK(!publication_parameter,"nested eager Prepare FIFO publication");
        publication_parameter=parameter; publication_ack=&ack;
        publication_core=&core; publication_ticket=&ticket; publication_commit_failed=&commit_failed;
    }
    ~PublicationMarker() noexcept {
        publication_parameter=nullptr; publication_ack=nullptr; publication_core=nullptr;
        publication_ticket=nullptr; publication_commit_failed=nullptr;
    }
};
thread_local bool native_job_active=false;
struct NativeJobScope {
    NativeJobScope() { TORCH_CHECK(!native_job_active,"recursive eager Prepare native job"); native_job_active=true; }
    ~NativeJobScope() noexcept { native_job_active=false; }
};
uint64_t native_tid() noexcept { return static_cast<uint64_t>(::syscall(SYS_gettid)); }
uint64_t clock_ns(clockid_t clock) {
    timespec value{};
    TORCH_CHECK(clock_gettime(clock,&value)==0,"eager diagnostic clock failed");
    return static_cast<uint64_t>(value.tv_sec)*1000000000ULL+static_cast<uint64_t>(value.tv_nsec);
}
void record_trace(Trace trace) noexcept {
    try {
        auto& s=state(); std::lock_guard<std::mutex> lock(s.mu);
        if (s.trace.size()<s.config.trace_limit) s.trace.emplace_back(std::move(trace));
        else ++s.trace_dropped;
    } catch (...) { state().trace_errors.fetch_add(1,std::memory_order_relaxed); }
}

std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        switch (c) { case '"': out << "\\\""; break; case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break; case '\r': out << "\\r"; break; case '\t': out << "\\t"; break;
        default: if (c < 32) { const char* hex="0123456789abcdef"; out << "\\u00" << hex[c>>4] << hex[c&15]; }
                 else out << static_cast<char>(c); }
    }
    out << '"'; return out.str();
}
const char* boolean(bool value) { return value ? "true" : "false"; }
std::string describe(std::exception_ptr error) {
    try { if (error) std::rethrow_exception(error); }
    catch (const std::exception& value) { return value.what(); }
    catch (...) { return "non-standard native exception"; }
    return "";
}
void note_error(const char* phase, const std::string& operation, uint64_t sequence, std::exception_ptr error) noexcept {
    try {
        auto& s=state(); std::lock_guard<std::mutex> lock(s.mu);
        if (!s.first_error) { s.first_error=error; s.error_phase=phase; s.error_operation=operation; s.error_sequence=sequence; }
        s.closure_passed=false;
    } catch (...) { /* The core separately retains the actual first exception. */ }
}
void count_event(const std::string& operation, const char* event, const Identity& identity) {
    auto& s=state(); std::lock_guard<std::mutex> lock(s.mu); auto& n=s.operators[operation];
    if (std::strcmp(event,"prepared")==0) ++n.prepared;
    else if (std::strcmp(event,"executed")==0) {
        ++n.executed;
        if (BuildUsesHostSimulation()) s.execution_order.Record(identity.sequence.load(std::memory_order_acquire),operation.c_str());
    }
    else if (std::strcmp(event,"returned")==0) ++n.returned;
}
void check_queue_configuration() {
    TORCH_CHECK(c10_npu::option::OptionsManager::GetTaskQueueEnable()==1 &&
        !c10_npu::option::OptionsManager::GetPerStreamQueue() &&
        !c10_npu::option::OptionsManager::CheckBlockingEnable(),
        "R24 eager Prepare requires asynchronous task queue1 and its original default stream repository");
}
void check_context(void* expected) {
    aclrtContext actual=nullptr; NPU_CHECK_ERROR(c10_npu::SetDevice(0));
    NPU_CHECK_ERROR(aclrtGetCurrentContext(&actual));
    TORCH_CHECK(actual && actual==expected,"eager Prepare primary Context changed");
}
void* create_event(void* context) {
    c10_npu::NPUGuard guard(0); check_context(context); aclrtEvent event=nullptr;
    const auto flags=c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event,flags)); return event;
}
void record_event(void* event,void* stream,void* context) {
    c10_npu::NPUGuard guard(0); check_context(context); NPU_CHECK_ERROR(aclrtRecordEvent(event,stream));
}
void wait_event(void* event,void* context) {
    c10_npu::NPUGuard guard(0); check_context(context); NPU_CHECK_ERROR(aclrtSynchronizeEvent(event));
}
void destroy_event(void* event,void* context) noexcept {
    try { c10_npu::NPUGuard guard(0); check_context(context); NPU_CHECK_ERROR(aclrtDestroyEvent(event)); }
    catch (...) { state().sdk_stop=true; note_error("event_destroy","",0,std::current_exception()); }
}
host_regions::device_retirement_detail::CompletionHooks completion_hooks{create_event,record_event,wait_event,destroy_event};

class NativeContext final : public OwnedExecutionContext {
public:
    NativeContext(std::shared_ptr<hdb_native::CallerTls> caller,std::shared_ptr<Identity> identity,
                  std::string operation,bool tracing,std::string signature)
        : caller_(std::move(caller)),identity_(std::move(identity)),operation_(std::move(operation)),
          signature_(std::move(signature)),tracing_(tracing) {}
    void Prepare(host_regions::OwnedOpApiPayload& payload,void* stream) override {
        TORCH_CHECK(!state().sdk_stop.load(std::memory_order_acquire),"eager Prepare SDK domain is quarantined");
        check_queue_configuration();
        NativeJobScope native_scope;
        if (!tracing_) { hdb_native::TaskTls guard(*caller_,true); payload.Prepare(stream); return; }
        Trace trace;
        try { trace.identity=identity_; trace.operation=operation_; trace.shape_signature=signature_; trace.native_tid=native_tid(); }
        catch (...) { state().trace_errors.fetch_add(1,std::memory_order_relaxed); throw; }
        trace.start_ns=clock_ns(CLOCK_MONOTONIC); trace.start_thread_cpu_ns=clock_ns(CLOCK_THREAD_CPUTIME_ID);
        try {
            hdb_native::TaskTls guard(*caller_,true); payload.Prepare(stream); trace.succeeded=true;
        } catch (...) {
            const auto first=std::current_exception();
            try { trace.end_thread_cpu_ns=clock_ns(CLOCK_THREAD_CPUTIME_ID); trace.end_ns=clock_ns(CLOCK_MONOTONIC); record_trace(std::move(trace)); }
            catch (...) {}
            std::rethrow_exception(first);
        }
        trace.end_thread_cpu_ns=clock_ns(CLOCK_THREAD_CPUTIME_ID); trace.end_ns=clock_ns(CLOCK_MONOTONIC); record_trace(std::move(trace));
    }
    int Execute(host_regions::OwnedOpApiPayload& payload,void* stream) override {
        check_queue_configuration();
        NativeJobScope native_scope;
        if (tracing_) identity_->consumer_tid.store(native_tid(),std::memory_order_release);
        c10_npu::NPUGuard guard(0); check_context(caller_->context()); return payload.Execute(stream);
    }
    void Cancel(host_regions::OwnedOpApiPayload& payload) noexcept override {
        try { NativeJobScope native_scope; c10_npu::NPUGuard guard(0); check_context(caller_->context()); payload.Cancel(); }
        catch (...) { cleanup_failed_=true; state().sdk_stop=true; note_error("cancel",operation_,identity_->sequence.load(),std::current_exception()); }
    }
    bool CleanupFailed() const noexcept override { return cleanup_failed_.load(std::memory_order_acquire); }
    void Failed(const char* phase,std::exception_ptr error) noexcept override {
        note_error(phase,operation_,identity_->sequence.load(std::memory_order_acquire),error);
    }
private:
    std::shared_ptr<hdb_native::CallerTls> caller_;
    std::shared_ptr<Identity> identity_;
    std::string operation_;
    std::string signature_;
    bool tracing_;
    std::atomic<bool> cleanup_failed_{false};
};

struct Retirement {
    std::shared_ptr<Core> core;
    Core::TicketPtr ticket;
    std::shared_ptr<OwnedQueueJob> job;
    std::unique_ptr<Fence> fence;
    std::string operation;
    ~Retirement() noexcept {
        if (!ticket) return; // Publication never began; caller owns cancellation.
        try {
            const auto publication=core->Publication(ticket);
            if (publication==Core::PublishState::NotAccepted) return; // Core owns rollback/ACK.
            TORCH_CHECK(publication==Core::PublishState::Committed,"eager Prepare publication identity is uncertain");
            core->CancelAndWait(ticket);
            if (!fence->Complete(job->MayHaveSubmitted())) std::rethrow_exception(fence->Failure());
            TORCH_CHECK(!job->CleanupFailed(),"eager Prepare SDK host cleanup failed");
            fence.reset();
            { std::lock_guard<std::mutex> lock(state().mu);
              TORCH_CHECK(!state().first_error,"eager Prepare completion domain has failed"); }
            core->Retire(ticket,true);
        } catch (...) {
            auto error=std::current_exception(); state().sdk_stop=true; note_error("retirement",operation,ticket->sequence,error);
            try { core->Quarantine(ticket,error); } catch (...) {}
            // Preserve an unacknowledged event together with the core's owners
            // and live producer workers; do not retry SDK calls in a destructor.
            if (fence) { try { std::lock_guard<std::mutex> lock(state().mu); state().quarantined_fences.push_back(std::move(fence)); }
                         catch (...) { (void)fence.release(); } }
        }
    }
};
bool registered(const std::string& name) {
    return name=="aclnnEmbeddingBag" || name=="aclnnAdd" || name=="aclnnRelu" || name=="aclnnMm";
}
} // namespace

bool Enabled() noexcept { return enabled_flag.load(std::memory_order_acquire); }
bool DiagnosticTracing() noexcept { return trace_flag.load(std::memory_order_acquire); }
bool NativeJobActive() noexcept { return native_job_active; }
bool Admit(const char* name,bool qualified,const char* decline_reason,bool argument_domain) {
    if (!Enabled()) return false;
    TORCH_CHECK(!native_job_active,"eager Prepare workers/consumer cannot reenter the original eager route");
    TORCH_CHECK(name,"invalid eager Prepare operation name");
    check_queue_configuration();
    auto& s=state(); std::shared_ptr<Core> core;
    { std::lock_guard<std::mutex> lock(s.mu); core=s.core;
      TORCH_CHECK(s.caller_thread==std::this_thread::get_id(),"R24 eager Prepare currently admits one native caller thread per configured domain");
      if (s.first_error) std::rethrow_exception(s.first_error); }
    core->RethrowIfFailed();
    host_regions::PrepareOwnedOpApiCallerState();
    bool allowed;
    { std::lock_guard<std::mutex> lock(s.mu); auto& n=s.operators[name]; ++n.seen;
      allowed=std::find(s.config.allowed_ops.begin(),s.config.allowed_ops.end(),name)!=s.config.allowed_ops.end();
      TORCH_CHECK(!n.qualified || qualified,"eager Prepare previously qualified SDK contract became unavailable");
      if (qualified) n.qualified=true;
      if (allowed && qualified && argument_domain) { ++n.eligible; return true; }
    }
    // An unsupported phase1 is not proved metadata-only. Flush prior eager
    // jobs before returning to its original SDK path. This is intentionally
    // included in mixed-model timing; it says nothing about host-meta reads
    // which took place earlier than this macro interception point.
    try {
        hdb_native::CallerTls caller(true);
        hdb_native::WaitDetach detached;
        c10_npu::NPUGuard guard(0); check_context(caller.context());
        TORCH_CHECK(c10_npu::eagerPrepareDefaultQueueHealthy(),"eager fallback original FIFO has failed");
        const auto status=c10_npu::emptyAllNPUStream(true);
        TORCH_CHECK(status==NPU_STATUS_SUCCESS && c10_npu::eagerPrepareDefaultQueueHealthy(),"eager fallback FIFO drain failed");
        c10_npu::getDefaultNPUStream(0).synchronize();
        TORCH_CHECK(c10_npu::eagerPrepareDefaultQueueHealthy(),"eager fallback stream synchronization failed");
        core->RethrowIfFailed();
    } catch (...) { s.sdk_stop=true; core->Close(); note_error("fallback_fence",name,0,std::current_exception()); throw; }
    { std::lock_guard<std::mutex> lock(s.mu); auto& n=s.operators[name];
      ++n.fallback; ++n.fallback_fences;
      const char* reason=std::strcmp(decline_reason,"operator_unregistered")==0 ? "operator_unregistered" :
          !allowed ? "operator_not_allowed" : !qualified ? "sdk_contract_unvalidated" : decline_reason;
      ++n.reasons[reason]; }
    return false;
}
bool CanQueueTensor(const at::Tensor& value) {
    if (!value.defined()) return false;
    auto* storage=host_regions::CapturedStorageSourceKey(value,true);
    if (!storage || value.device().index()!=0 || typeid(*storage)!=typeid(torch_npu::NPUStorageImpl) ||
        storage->pyobj_slot()->load_pyobj() || static_cast<torch_npu::NPUStorageImpl*>(storage)->lazy_fusion_data_) return false;
    return c10_npu::NPUCachingAllocator::IsNativeEagerRetirementData(
            storage->data_ptr(),storage->allocator(),0,c10_npu::getDefaultNPUStream(0).stream(false));
}
void ValidateTensor(const at::Tensor& value) {
    TORCH_CHECK(Enabled() && CanQueueTensor(value),
        "eager Prepare requires admitted native device0 Storage/default-stream allocation without Python metadata, hooks or instrumentation");
}
void QueueCopyAccepted(const void* parameter) noexcept {
    if (parameter!=publication_parameter || !publication_ack) return;
    // The original FIFO now owns this exact deferred callback. INIT enqueue
    // may synchronously wait for its consumption, so wake Core's consumer here,
    // before enCurrentNPUStream or the enclosing PublishFn returns. Ownership
    // stays committed even if that later return reports an error.
    publication_ack->store(true,std::memory_order_release);
    if (!publication_core || !publication_ticket || !publication_core->MarkCommitted(*publication_ticket)) {
        if (publication_commit_failed) publication_commit_failed->store(true,std::memory_order_release);
    }
}

void SubmitImpl(const char* name,std::unique_ptr<host_regions::OwnedOpApiPayload> payload,std::string signature) {
    auto& s=state(); std::shared_ptr<Core> core;
    struct Submitter {
        State& state;
        ~Submitter() noexcept { std::lock_guard<std::mutex> lock(state.mu); --state.active_submitters; state.submitters_changed.notify_all(); }
    };
    { std::lock_guard<std::mutex> lock(s.mu);
      TORCH_CHECK(s.enabled && s.core && s.caller_thread==std::this_thread::get_id(),"eager Prepare admission is closed or caller changed");
      core=s.core; ++s.active_submitters; }
    Submitter submitter{s};
    TORCH_CHECK(!host_regions::OwnedOpApiEnabled(),"eager Prepare cannot nest the R22 capture domain");
    host_regions::PrepareOwnedOpApiCallerState();
    auto caller=std::make_shared<hdb_native::CallerTls>(true);
    TORCH_CHECK(caller->context(),"eager Prepare requires the current primary Context");
    auto stream=c10_npu::getDefaultNPUStream(0).stream(false);
    const std::string operation(name);
    bool tracing;
    { std::lock_guard<std::mutex> lock(s.mu); tracing=s.config.diagnostic_trace; }
    auto identity=std::make_shared<Identity>(); if (tracing) identity->caller_tid=native_tid();
    auto job=std::make_shared<OwnedQueueJob>(std::move(payload),std::make_shared<NativeContext>(caller,identity,operation,tracing,std::move(signature)),stream,
        [operation,identity](const char* event) { count_event(operation,event,*identity); });
    auto retirement=std::make_shared<Retirement>(); retirement->core=core; retirement->job=job; retirement->operation=operation;
    retirement->fence=std::make_unique<Fence>(completion_hooks,caller->context(),stream);
    try {
        // Detached before Submit's locks; native workers and the original FIFO
        // consumer never enter Python or wait for the calling interpreter.
        hdb_native::WaitDetach detached;
        auto ticket=core->Submit(job,[&](const Core::TicketPtr& value) {
            identity->sequence.store(value->sequence,std::memory_order_release);
            if (BuildUsesHostSimulation()) { std::lock_guard<std::mutex> lock(s.mu); s.admission_order.Record(value->sequence,name); }
            retirement->ticket=value;
            std::atomic<bool> accepted{false},commit_failed{false};
            std::exception_ptr error;
            try {
                at_npu::native::ExecuteParasOpApi parameters;
                std::strncpy(parameters.opType,name,sizeof(parameters.opType)-1);
                parameters.customHandler=[core,value,retirement]() { core->Consume(value); return 0; };
                queue::QueueParas queue_parameter(queue::EXECUTE_OPAPI,sizeof(parameters),&parameters);
                PublicationMarker marker(&queue_parameter,accepted,*core,value,commit_failed);
                c10_npu::enCurrentNPUStream(&queue_parameter,0);
            }
            catch (...) { error=std::current_exception(); }
            if (commit_failed.load(std::memory_order_acquire) && !error) {
                error=std::make_exception_ptr(std::runtime_error("eager Prepare original FIFO copy ACK could not commit its ticket"));
            }
            return Core::PublishReceipt{accepted.load(std::memory_order_acquire) ? Core::PublishState::Committed :
                Core::PublishState::NotAccepted,error};
        });
        if (core->Publication(ticket)!=Core::PublishState::Committed) core->RethrowIfFailed();
        const auto snap=core->Snapshot();
        { std::lock_guard<std::mutex> lock(s.mu); ++s.operators[operation].queued;
          s.peak_slots=std::max(s.peak_slots,snap.slots_in_use); s.peak_bytes=std::max(s.peak_bytes,snap.retained_bytes); }
        core->RethrowIfFailed();
    } catch (...) { note_error("submit",operation,retirement->ticket ? retirement->ticket->sequence : 0,std::current_exception()); throw; }
}
void Submit(const char* name,std::unique_ptr<host_regions::OwnedOpApiPayload> payload,std::string signature) {
    try { SubmitImpl(name,std::move(payload),std::move(signature)); }
    catch (...) {
        auto& s=state(); note_error("submit",name,std::numeric_limits<uint64_t>::max(),std::current_exception());
        std::shared_ptr<Core> core;
        { std::lock_guard<std::mutex> lock(s.mu); core=s.core; }
        if (core) core->Close();
        throw;
    }
}

std::string Configure(Configuration config) {
    TORCH_CHECK(config.workers>0 && config.workers<=8 && config.capacity>=config.workers && config.capacity<=1024 &&
        config.byte_limit>0 && config.byte_limit<=536870912,"eager Prepare configuration exceeds admitted bounds");
    TORCH_CHECK(config.trace_limit>0 && config.trace_limit<=256,"eager diagnostic trace exceeds fixed 256-record bound");
    TORCH_CHECK(!config.allowed_ops.empty(),"eager Prepare requires explicit operator schemas");
    std::vector<std::string> unique;
    for (const auto& name:config.allowed_ops) {
        TORCH_CHECK(registered(name) && std::find(unique.begin(),unique.end(),name)==unique.end(),"unregistered or duplicate eager Prepare schema");
        unique.push_back(name);
    }
    auto& s=state(); LifecycleGuard lifecycle(s.lifecycle_mu);
    { std::lock_guard<std::mutex> lock(s.mu); TORCH_CHECK(!s.enabled && !s.first_error,"close the existing eager Prepare domain before configuring"); }
    if (config.enabled) {
        check_queue_configuration();
        hdb_native::CallerTls caller(true);
        TORCH_CHECK(caller.context(),"eager Prepare configure requires the initialized device0 primary Context");
        host_regions::PrepareOwnedOpApiCallerState();
        CheckSdkConfiguration();
        std::vector<Trace> trace; trace.reserve(config.diagnostic_trace ? config.trace_limit : 0);
        auto core=std::make_shared<Core>(Core::Config{config.capacity,config.byte_limit,config.workers});
        std::lock_guard<std::mutex> lock(s.mu); s.core=std::move(core); s.config=config; s.operators.clear();
        s.fifo_drained=false; s.device_synchronized=false; s.workers_stopped=false; s.closure_passed=false;
        s.peak_slots=0; s.peak_bytes=0;
        s.admission_order=SequenceOrderEvidence{}; s.execution_order=SequenceOrderEvidence{};
        s.trace=std::move(trace); s.trace_dropped=0; s.trace_errors=0;
        s.cold_close.Reset();
        s.caller_thread=std::this_thread::get_id(); s.caller_tid=native_tid();
        s.sdk_stop=false;
        for (const auto& name:config.allowed_ops) s.operators[name].qualified=SdkContractAvailable(name.c_str());
        s.enabled.store(true,std::memory_order_release);
        enabled_flag.store(true,std::memory_order_release);
        trace_flag.store(config.diagnostic_trace,std::memory_order_release);
    } else { std::lock_guard<std::mutex> lock(s.mu); s.config=config; }
    return Snapshot();
}

std::string Snapshot() {
    InterpreterDetach detached;
    auto& s=state(); std::shared_ptr<Core> core;
    { std::lock_guard<std::mutex> lock(s.mu); core=s.core; }
    const auto q=core ? core->Snapshot() : Core::SnapshotData{};
    if (core) { try { core->RethrowIfFailed(); } catch (...) {
        note_error("queue","",q.first_failed_sequence,std::current_exception()); } }
    std::lock_guard<std::mutex> lock(s.mu); std::ostringstream out;
    out << "{\"schema\":1,\"mode\":" << quote(s.enabled ? "eager_prepare" : "off") << ",\"enabled\":" << boolean(s.enabled)
        << ",\"workers\":" << s.config.workers << ",\"capacity\":" << s.config.capacity << ",\"byte_limit\":" << s.config.byte_limit;
    out << ",\"caller_tid\":" << s.caller_tid;
    out << ",\"backend\":" << quote(BackendName())
        << ",\"real_sdk_qualified\":" << boolean(!BuildUsesHostSimulation() && std::any_of(s.operators.begin(),s.operators.end(),
            [](const auto& entry) { return entry.second.qualified; }))
        << ",\"simulated_prepare\":" << boolean(BuildUsesHostSimulation())
        << ",\"pta_cache_enabled\":" << (BuildUsesHostSimulation() ? "false" : "null")
        << ",\"numerical_kernels_verified\":false";
    out << ",\"allowed_ops\":["; bool first=true;
    for (const auto& name:s.config.allowed_ops) { if (!first) out << ','; first=false; out << quote(name); }
    out << "],\"qualified_ops\":["; first=true;
    for (const auto& entry:s.operators) if (entry.second.qualified && !BuildUsesHostSimulation()) { if (!first) out << ','; first=false; out << quote(entry.first); }
    out << "],\"qualified_sim_ops\":["; first=true;
    for (const auto& entry:s.operators) if (entry.second.qualified && BuildUsesHostSimulation()) { if (!first) out << ','; first=false; out << quote(entry.first); }
    out << "],\"operators\":{"; first=true;
    for (const auto& entry:s.operators) {
        if (!first) out << ','; first=false; const auto& n=entry.second;
        out << quote(entry.first) << ":{\"seen\":" << n.seen << ",\"eligible\":" << n.eligible << ",\"queued\":" << n.queued
            << ",\"prepared\":" << n.prepared << ",\"executed\":" << n.executed << ",\"returned\":" << n.returned
            << ",\"fallback\":" << n.fallback << ",\"fallback_reasons\":{"; bool rfirst=true;
        for (const auto& reason:n.reasons) { if (!rfirst) out << ','; rfirst=false; out << quote(reason.first) << ':' << reason.second; }
        out << "},\"fallback_fences\":" << n.fallback_fences << '}';
    }
    out << "},\"queue\":{\"slots_in_use\":" << q.slots_in_use << ",\"retained_bytes\":" << q.retained_bytes
        << ",\"queued\":" << q.queued << ",\"preparing\":" << q.preparing << ",\"ready\":" << q.ready
        << ",\"admitted\":" << q.admitted << ",\"retired\":" << q.retired << ",\"closed\":" << boolean(q.closed)
        << ",\"workers_started\":" << q.workers_started
        << ",\"quarantined\":" << boolean(q.quarantined) << ",\"first_failed_sequence\":";
    if (q.first_failed_sequence==std::numeric_limits<uint64_t>::max()) out << "null"; else out << q.first_failed_sequence;
    out << "},\"peaks\":{\"slots_in_use\":" << q.peak_slots_in_use << ",\"retained_bytes\":" << q.peak_retained_bytes << "},\"closure\":{\"admission_closed\":"
        << boolean(!s.enabled || q.closed) << ",\"fifo_drained\":" << boolean(s.fifo_drained) << ",\"device_synchronized\":" << boolean(s.device_synchronized)
        << ",\"all_retired\":" << boolean(q.slots_in_use==0) << ",\"workers_stopped\":" << boolean(s.workers_stopped)
        << ",\"close_attempted\":" << boolean(s.cold_close.Attempted()) << ",\"passed\":" << boolean(s.closure_passed)
        << "},\"first_error\":";
    if (!s.first_error) out << "null";
    else {
        out << "{\"phase\":" << quote(s.error_phase) << ",\"operator\":" << quote(s.error_operation) << ",\"sequence\":";
        if (s.error_sequence==std::numeric_limits<uint64_t>::max()) out << "null"; else out << s.error_sequence;
        out << ",\"message\":" << quote(describe(s.first_error)) << '}';
    }
    out << ",\"order_evidence\":{\"enabled\":" << boolean(BuildUsesHostSimulation())
        << ",\"admitted_count\":" << s.admission_order.count << ",\"executed_count\":" << s.execution_order.count
        << ",\"admitted_sequence_hash\":" << s.admission_order.hash << ",\"executed_sequence_hash\":" << s.execution_order.hash
        << ",\"passed\":" << boolean(BuildUsesHostSimulation() && s.closure_passed && s.admission_order.ordered &&
            s.execution_order.ordered && s.admission_order.count==s.execution_order.count && s.admission_order.hash==s.execution_order.hash) << '}';
    out << ",\"diagnostic_trace\":{\"enabled\":" << boolean(s.config.diagnostic_trace) << ",\"limit\":" << s.config.trace_limit
        << ",\"dropped\":" << s.trace_dropped+s.trace_errors.load(std::memory_order_relaxed)
        << ",\"incomplete\":" << boolean(s.trace_dropped!=0 || s.trace_errors.load(std::memory_order_relaxed)!=0)
        << ",\"events\":["; first=true;
    for (const auto& trace:s.trace) {
        if (!first) out << ','; first=false;
        out << "{\"sequence\":";
        const auto sequence=trace.identity->sequence.load(std::memory_order_acquire);
        if (sequence==std::numeric_limits<uint64_t>::max()) out << "null"; else out << sequence;
        out << ",\"operator\":" << quote(trace.operation) << ",\"native_tid\":" << trace.native_tid
            << ",\"shape_unknown\":" << boolean(trace.shape_signature.empty()) << ",\"shape_signature\":";
        if (trace.shape_signature.empty()) out << "null"; else out << quote(trace.shape_signature);
        out
            << ",\"caller_tid\":" << trace.identity->caller_tid << ",\"consumer_tid\":";
        const auto consumer=trace.identity->consumer_tid.load(std::memory_order_acquire);
        if (!consumer) out << "null"; else out << consumer;
        out << ",\"start_ns\":" << trace.start_ns << ",\"end_ns\":" << trace.end_ns
            << ",\"start_thread_cpu_ns\":" << trace.start_thread_cpu_ns << ",\"end_thread_cpu_ns\":" << trace.end_thread_cpu_ns
            << ",\"succeeded\":" << boolean(trace.succeeded) << '}';
    }
    uint64_t prepared=0;
    for (const auto& entry:s.operators) prepared+=entry.second.prepared;
    const bool trace_complete=s.config.diagnostic_trace && s.closure_passed && !s.first_error &&
        s.trace_dropped==0 && s.trace_errors.load(std::memory_order_relaxed)==0 && s.trace.size()==prepared;
    out << "],\"complete\":" << boolean(trace_complete)
        << "},\"counter_units\":\"seen/eligible/queued are per SDK call; executed is phase2 entry; returned is phase2 return including nonzero; snapshot is non-linearizable while active\"}";
    return out.str();
}

std::string DrainClose() {
    auto& s=state(); LifecycleGuard lifecycle(s.lifecycle_mu); std::shared_ptr<Core> core;
    bool active;
    { std::lock_guard<std::mutex> lock(s.mu); core=s.core; active=s.enabled.load(std::memory_order_acquire); }
    if (!active) return Snapshot();
    try {
      s.cold_close.Run([&] {
        core->Close();
        hdb_native::WaitDetach detached;
        { std::unique_lock<std::mutex> lock(s.mu); s.submitters_changed.wait(lock,[&] { return s.active_submitters==0; }); }
        c10_npu::NPUGuard guard(0); NPU_CHECK_ERROR(c10_npu::SetDevice(0));
        TORCH_CHECK(c10_npu::eagerPrepareDefaultQueueHealthy(),"eager Prepare original FIFO has failed");
        const auto status=c10_npu::emptyAllNPUStream(true);
        TORCH_CHECK(status==NPU_STATUS_SUCCESS && c10_npu::eagerPrepareDefaultQueueHealthy(),"eager Prepare original FIFO drain failed");
        { std::lock_guard<std::mutex> lock(s.mu); s.fifo_drained=true; }
        TORCH_CHECK(c10_npu::npuSynchronizeDevice(true) && c10_npu::eagerPrepareDefaultQueueHealthy(),"eager Prepare original device synchronization failed");
        { std::lock_guard<std::mutex> lock(s.mu); s.device_synchronized=true; }
        core->Shutdown(); core->RethrowIfFailed();
        { std::lock_guard<std::mutex> lock(s.mu); TORCH_CHECK(!s.first_error,"eager Prepare cleanup/retirement failed");
          s.workers_stopped=true; s.closure_passed=true; s.enabled.store(false,std::memory_order_release);
          enabled_flag.store(false,std::memory_order_release); }
        trace_flag.store(false,std::memory_order_release);
      });
    } catch (...) { s.sdk_stop=true; note_error("drain_close","",0,std::current_exception()); throw; }
    return Snapshot();
}
void BeforeRuntimeFinalize() {
    if (!Enabled()) return;
    try { (void)DrainClose(); }
    catch (...) {
        auto& s=state(); s.sdk_stop=true;
        note_error("runtime_finalize","",0,std::current_exception());
        FailStopFinalize(s.fail_stop_mutex,s.fail_stop_changed);
    }
}
} // namespace c10_npu::eager_prepare
