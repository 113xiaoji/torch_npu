#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.h"
#include "torch_npu/csrc/core/npu/eager_parallel/AllocatorEventScope.h"
#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateProtocol.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/HostRegionCapture.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/CPUAllocator.h>
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace c10_npu::eager_parallel {
namespace {
constexpr std::size_t kMaxInputs = 128;
constexpr std::size_t kMaxOutputs = 4;
thread_local unsigned internal_depth = 0;
thread_local unsigned entry_depth = 0;
thread_local unsigned qualified_body_depth = 0;
struct InternalScope { InternalScope() { ++internal_depth; } ~InternalScope() { --internal_depth; } };
enum class Kind { Unknown, Alias, Embedding, Cat, Bmm, Index };
Kind classify(const char* schema, bool interaction)
{
    if (!schema) return Kind::Unknown;
    if (!std::strcmp(schema,"aten::_embedding_bag") || !std::strcmp(schema,"aten::_embedding_bag_forward_only")) return Kind::Embedding;
    if (!std::strcmp(schema,"aten::view") || !std::strcmp(schema,"aten::_unsafe_view") ||
        !std::strcmp(schema,"aten::transpose.int") || !std::strcmp(schema,"aten::t") ||
        !std::strcmp(schema,"aten::as_strided") || !std::strcmp(schema,"aten::alias") || !std::strcmp(schema,"aten::detach")) return Kind::Alias;
    if (!interaction) return Kind::Unknown;
    if (!std::strcmp(schema,"aten::cat")) return Kind::Cat;
    if (!std::strcmp(schema,"aten::bmm")) return Kind::Bmm;
    if (!std::strcmp(schema,"aten::index.Tensor")) return Kind::Index;
    return Kind::Unknown;
}
bool scheduled(Kind kind) { return kind == Kind::Embedding || kind == Kind::Cat || kind == Kind::Bmm || kind == Kind::Index; }
void check_acl(aclError code, const char* operation)
{
    // NPU_CHECK_ERROR may warn for an unsupported status. A receipt needs exact success.
    if (code != ACL_ERROR_NONE) throw std::runtime_error(std::string("immediate ") + operation + " failed: " + std::to_string(code));
}
class AclBackend final : public ImmediateEventBackend {
public:
    explicit AclBackend(std::size_t capacity)
        : streams{{getDefaultNPUStream(0),getNPUStreamFromPool(0),getNPUStreamFromPool(0),getNPUStreamFromPool(0)}}, events(capacity + 1, nullptr)
    {
        InternalScope scope;
        for (std::size_t i = 0; i < streams.size(); ++i) {
            raw[i] = streams[i].stream(false);
            if (!raw[i]) throw std::runtime_error("immediate stream has no native handle");
            for (std::size_t j = 0; j < i; ++j)
                if (raw[j] == raw[i]) throw std::runtime_error("immediate stream pool returned duplicate streams");
        }
        const auto flags = acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
        try {
            for (auto& event : events) {
                check_acl(acl::AclrtCreateEventWithFlag(&event, flags), "event reserve");
                if (!event) throw std::runtime_error("immediate event reserve returned null");
            }
        } catch (...) { destroy(); throw; }
    }
    ~AclBackend() override { if (safe_to_destroy) destroy(); }
    std::size_t live_events() const noexcept {
        std::size_t count=0; for (const auto event:events) if (event) ++count; return count;
    }
    void retire() {
        InternalScope scope;
        if (!safe_to_destroy) throw std::runtime_error("immediate event retirement lacks completion proof");
        for (auto& event:events) if (event) {
            check_acl(aclrtDestroyEvent(event),"event destroy"); event=nullptr;
        }
    }
    void record(std::size_t event, unsigned stream) override
    {
        InternalScope scope; safe_to_destroy = false;
        check_acl(aclrtRecordEvent(events.at(event),raw.at(stream)), "event record");
    }
    void wait(unsigned stream, std::size_t event) override
    {
        InternalScope scope; safe_to_destroy = false;
        check_acl(aclrtStreamWaitEvent(raw.at(stream), events.at(event)), "event wait");
    }
    void synchronize(unsigned stream) override
    {
        InternalScope scope;
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            acl::aclrtStreamStatus status = acl::ACL_STREAM_STATUS_RESERVED;
            check_acl(acl::AclrtStreamQuery(raw.at(stream), &status), "stream completion query");
            if (status == acl::ACL_STREAM_STATUS_COMPLETE) return;
            if (std::chrono::steady_clock::now() >= end) throw std::runtime_error("immediate stream completion timed out");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void destroy() noexcept
    {
        InternalScope scope;
        for (auto& event : events) if (event) {
            // Handles are never recycled before all recorded waits have completed.
            // A failed destroy retains the handle rather than claiming retirement.
            if (aclrtDestroyEvent(event) == ACL_ERROR_NONE) event = nullptr;
        }
    }
    std::array<NPUStream,4> streams;
    std::array<aclrtStream,4> raw{};
    std::vector<aclrtEvent> events;
    bool safe_to_destroy = true;
};
struct Owner {
    at::Tensor tensor;
    const c10::StorageImpl* storage = nullptr;
    std::uintptr_t key = 0;
    std::size_t bytes = 0;
};
struct State {
    State(bool interaction, std::size_t capacity, std::size_t limit)
        : owner(std::this_thread::get_id()), interaction(interaction), capacity(capacity), byte_limit(limit),
          backend(capacity), protocol(backend,capacity,capacity*kMaxOutputs), owners(capacity*(kMaxInputs+kMaxOutputs)) {}
    void check_owner() const {
        if (owner != std::this_thread::get_id()) throw std::runtime_error("immediate E/EI requires one caller; foreign caller rejected before effects");
    }
    void check_healthy() const { if (protocol.poisoned()) throw std::runtime_error(first_error.data()); }
    void remember(const char* message) noexcept {
        if (first_error[0]) return;
        if (!message) message = "unknown immediate failure";
        std::strncpy(first_error.data(),message,first_error.size()-1);
    }
    void clear_owners() noexcept {
        InternalScope scope;
        for (std::size_t i=0;i<owner_count;++i) owners[i] = Owner{};
        for (auto& output:pending_outputs) output=at::Tensor{};
        owner_count=0; retained_bytes=0;
    }
    void drain() {
        const bool had_work=protocol.active();
        protocol.drain(); backend.safe_to_destroy=true; clear_owners(); check_healthy();
        if (had_work) ++drains;
    }
    bool recover() noexcept {
        const bool complete=protocol.recover();
        if (complete) { backend.safe_to_destroy=true; clear_owners(); }
        else { quarantined=true; }
        return complete;
    }
    void retain(const at::Tensor& tensor) {
        if (!tensor.defined() || !tensor.has_storage()) return;
        auto storage=tensor.storage();
        const auto* impl=storage.unsafeGetStorageImpl();
        for (std::size_t i=0;i<owner_count;++i) if (owners[i].storage==impl) return;
        if (owner_count==owners.size()) throw std::runtime_error("immediate owner capacity exhausted");
        // Tensor assignment is intrusive_ptr ownership, with no dynamic allocation.
        auto& slot=owners[owner_count++]; slot.tensor=tensor; slot.storage=impl;
        slot.key=tensor.device().type()==c10::DeviceType::PrivateUse1 ? reinterpret_cast<std::uintptr_t>(storage.data_ptr().get_context()) : 0;
        slot.bytes=storage.nbytes();
        if (slot.bytes>std::numeric_limits<std::size_t>::max()-retained_bytes) throw std::runtime_error("immediate retained size overflow");
        retained_bytes+=slot.bytes;
        if (retained_bytes>peak_bytes) peak_bytes=retained_bytes;
    }
    void record_use(const at::Tensor& tensor, unsigned stream) {
        if (!tensor.defined() || tensor.device().type()!=c10::DeviceType::PrivateUse1 || !tensor.has_storage()) return;
        InternalScope scope;
        NPUCachingAllocator::recordStream(tensor.storage().data_ptr(),backend.streams.at(stream));
        check_healthy();
    }
    const std::thread::id owner;
    const bool interaction;
    const std::size_t capacity,byte_limit;
    AclBackend backend;
    ImmediateProtocol protocol;
    std::vector<Owner> owners;
    // A whole result is captured here without metadata access before output
    // auditing/registration. Failure at output 0 must also retain outputs 1..3.
    std::array<at::Tensor,kMaxOutputs> pending_outputs;
    std::size_t owner_count=0,retained_bytes=0,peak_bytes=0;
    std::array<char,512> first_error{};
    std::atomic<bool> enabled{true};
    bool closed=false,quarantined=false;
    std::shared_ptr<State> quarantine_anchor;
    std::uint64_t embedding=0,cat=0,bmm=0,index=0,fallback=0,aliases=0,nested=0,drains=0,budget_drains=0,failed=0,boundaries=0;
};
std::shared_ptr<State> current;
std::atomic<bool> last_interaction{true};
std::atomic<bool> runtime_enabled{false};
std::shared_ptr<State> state() { return std::atomic_load_explicit(&current,std::memory_order_acquire); }
void check_environment()
{
    if (option::OptionsManager::GetTaskQueueEnable()!=0) throw std::runtime_error("immediate E/EI requires TASK_QUEUE_ENABLE=0 in the fresh process");
    if (option::OptionsManager::CheckForceUncached()) throw std::runtime_error("immediate E/EI requires the native caching allocator");
    if (at_npu::native::env::CheckOpHookEnable()) throw std::runtime_error("immediate E/EI requires operator hooks disabled");
#ifndef BUILD_LIBTORCH
    if (c10_npu::impl::NPUTrace::getTrace()) throw std::runtime_error("immediate E/EI requires NPU trace callbacks disabled");
#endif
    if (host_regions::IsCapturing()) throw std::runtime_error("immediate E/EI does not admit host capture");
    InternalScope scope;
    if (current_device()!=0 || getCurrentNPUStream(0)!=getDefaultNPUStream(0)) throw std::runtime_error("immediate E/EI requires device 0 and its default caller stream");
    if (NPUCachingAllocator::hasCapturesUnderway(0) || currentStreamCaptureStatus()!=CaptureStatus::None)
        throw std::runtime_error("immediate E/EI does not admit graph capture");
}
bool native_tensor(const at::Tensor& tensor)
{
    if (!tensor.defined() || tensor.device().type()!=c10::DeviceType::PrivateUse1 || tensor.device().index()!=0 ||
        tensor.layout()!=c10::kStrided || !tensor.has_storage() || !at_npu::native::FormatHelper::IsBaseFormatType(tensor)) return false;
    const auto storage=tensor.storage(); const auto& ptr=storage.data_ptr();
    if (storage.allocator()!=NPUCachingAllocator::get() || ptr.get_deleter()!=NPUCachingAllocator::get()->raw_deleter()) return false;
    // Aligned/interior/custom DataPtr ownership is not covered by recordStream's
    // native lookup. Views retain the same base DataPtr and are admitted.
    return !ptr.get() || ptr.get()==ptr.get_context();
}
std::uintptr_t key(const at::Tensor& tensor)
{
    return tensor.defined() && tensor.device().type()==c10::DeviceType::PrivateUse1 && tensor.has_storage()
        ? reinterpret_cast<std::uintptr_t>(tensor.storage().data_ptr().get_context()) : 0;
}
std::string quote(const char* value)
{
    std::string result="\"";
    for (const unsigned char ch:std::string(value)) {
        if (ch=='"' || ch=='\\') { result+='\\'; result+=static_cast<char>(ch); }
        else if (ch>=32) result+=static_cast<char>(ch);
    }
    return result+'"';
}
} // namespace

AllocatorEventScope::AllocatorEventScope() noexcept { ++internal_depth; }
AllocatorEventScope::~AllocatorEventScope() noexcept { --internal_depth; }

struct ImmediateGuardImpl {
    explicit ImmediateGuardImpl(std::shared_ptr<State> s,const char* schema)
        : state(std::move(s)),kind(classify(schema,state->interaction)) {}
    std::shared_ptr<State> state;
    Kind kind;
    bool nested=false,begun=false,launched=false,finished=false,switched=false,body_scope=false,disqualified=false;
    unsigned selected=0;
    std::size_t input_count=0,output_count=0;
    std::array<at::Tensor,kMaxInputs> inputs;
    std::array<at::Tensor,kMaxOutputs> outputs;
    std::array<std::uintptr_t,kMaxInputs> input_keys{};
    std::array<std::uintptr_t,kMaxOutputs> output_keys{};
    void restore() noexcept {
        if (body_scope) { --qualified_body_depth; body_scope=false; }
        if (switched) {
            InternalScope scope;
            try { setCurrentNPUStream(state->backend.streams[0]); }
            catch (...) { state->remember("immediate failed to restore caller stream"); state->protocol.poison(); }
            switched=false;
        }
    }
};

bool RuntimeEnabled() noexcept { return runtime_enabled.load(std::memory_order_acquire); }
ImmediateGuard::ImmediateGuard(const char* schema,const char* route)
{
    (void)route;
    auto s=state(); if (!s || !s->enabled.load(std::memory_order_acquire)) return;
    s->check_owner(); s->check_healthy();
    if (internal_depth) {
        s->remember("immediate backend callback attempted native reentry"); s->protocol.poison();
        throw std::runtime_error(s->first_error.data());
    }
    impl_=std::make_unique<ImmediateGuardImpl>(s,schema);
    impl_->nested=entry_depth++!=0;
    if (impl_->nested) ++s->nested;
}
ImmediateGuard::~ImmediateGuard() noexcept
{
    if (!impl_) return;
    if (!impl_->finished) fail();
    impl_->restore(); --entry_depth;
    // Member destruction occurs after this body. Release the final Tensor
    // references explicitly while allocator bookkeeping remains internal.
    InternalScope scope;
    impl_.reset();
}
void ImmediateGuard::disqualify() noexcept { if (impl_) impl_->disqualified=true; }
void ImmediateGuard::add_input(const at::Tensor& value)
{
    if (!impl_ || impl_->nested || (!scheduled(impl_->kind) && impl_->kind!=Kind::Alias) || !value.defined()) return;
    if (impl_->input_count==kMaxInputs) { impl_->disqualified=true; return; }
    impl_->inputs[impl_->input_count++]=value;
}
void ImmediateGuard::begin()
{
    if (!impl_) return;
    auto& g=*impl_; auto& s=*g.state;
    if (g.begun) throw std::runtime_error("immediate guard began twice");
    g.begun=true;
    if (g.nested) return;
    check_environment();
    if (g.kind==Kind::Alias && !g.disqualified) {
        bool valid=g.input_count>0;
        for (std::size_t i=0;valid && i<g.input_count;++i) valid=native_tensor(g.inputs[i]) && !g.inputs[i].is_quantized();
        if (valid) { ++s.aliases; return; }
    }
    bool eligible=scheduled(g.kind) && !g.disqualified && g.input_count>0 && !at::GradMode::is_enabled();
    if (g.kind==Kind::Embedding && g.input_count!=3) eligible=false;
    if (g.kind==Kind::Bmm && g.input_count!=2) eligible=false;
    for (std::size_t i=0;eligible && i<g.input_count;++i) {
        const auto& t=g.inputs[i];
        const bool integral=t.scalar_type()==at::kLong || t.scalar_type()==at::kInt;
        if (g.kind==Kind::Index && i>0 && t.device().is_cpu()) {
            eligible=integral && t.layout()==c10::kStrided && t.is_contiguous() && t.has_storage() &&
                t.storage().allocator()==c10::GetCPUAllocator() &&
                t.storage().data_ptr().get_deleter()==c10::GetCPUAllocator()->raw_deleter();
        } else {
            eligible=native_tensor(t);
            if (g.kind==Kind::Embedding && i>0) eligible=eligible && integral && t.is_contiguous();
            else if (g.kind==Kind::Index && i>0) eligible=eligible && integral;
            else eligible=eligible && t.scalar_type()==at::kFloat;
        }
        g.input_keys[i]=key(t);
    }
    if (!eligible) { s.drain(); ++s.fallback; return; }
    std::size_t incoming=0;
    for (std::size_t i=0;i<g.input_count;++i) {
        const auto n=g.inputs[i].storage().nbytes();
        if (n>s.byte_limit || incoming>s.byte_limit-n) { s.drain(); ++s.fallback; return; }
        incoming+=n;
    }
    if (!s.protocol.can_begin() || s.owner_count+g.input_count+kMaxOutputs>s.owners.size() || incoming>s.byte_limit-s.retained_bytes) {
        s.drain(); ++s.budget_drains;
    }
    // Owner slots and cross-stream allocator use are established before body.
    for (std::size_t i=0;i<g.input_count;++i) s.retain(g.inputs[i]);
    g.selected=s.protocol.begin(g.kind==Kind::Embedding,g.input_keys.data(),g.input_count);
    for (std::size_t i=0;i<g.input_count;++i) s.record_use(g.inputs[i],g.selected);
    {
        InternalScope scope; setCurrentNPUStream(s.backend.streams[g.selected]);
    }
    g.switched=true; g.launched=true; ++qualified_body_depth; g.body_scope=true;
}
void ImmediateGuard::add_output(const at::Tensor& value)
{
    if (!impl_ || !impl_->launched) return;
    auto& g=*impl_;
    if (g.output_count==kMaxOutputs) throw std::runtime_error("immediate output handle contract exceeded");
    // Exact admitted schemas return at most four tensors. This path only copies
    // intrusive handles; metadata, allocator and SDK calls happen after capture.
    g.outputs[g.output_count++]=value;
}
void ImmediateGuard::complete()
{
    if (!impl_) return;
    auto& g=*impl_; auto& s=*g.state;
    if (g.finished) return;
    if (g.launched) {
        for (std::size_t i=0;i<g.output_count;++i) s.pending_outputs[i]=g.outputs[i];
        const auto expected=g.kind==Kind::Embedding ? 4u : 1u;
        if (g.output_count!=expected) throw std::runtime_error("immediate output handle count differs from qualified schema");
        for (std::size_t i=0;i<g.output_count;++i) s.retain(g.outputs[i]);
        for (std::size_t i=0;i<g.output_count;++i) {
            const auto& value=g.outputs[i];
            if (!native_tensor(value)) throw std::runtime_error("immediate output has unsupported allocation or format");
            const auto output_key=key(value);
            for (std::size_t j=0;j<g.input_count;++j)
                if (output_key && output_key==g.input_keys[j]) throw std::runtime_error("immediate out-of-place output unexpectedly aliases an input");
            g.output_keys[i]=output_key; s.record_use(value,g.selected);
            // The returned allocation may next be consumed asynchronously by
            // baseline work on the caller stream after this group is drained.
            s.record_use(value,0);
        }
        s.protocol.commit(g.output_keys.data(),g.output_count);
        for (auto& output:s.pending_outputs) output=at::Tensor{};
        if (g.kind==Kind::Embedding) ++s.embedding;
        else if (g.kind==Kind::Cat) ++s.cat;
        else if (g.kind==Kind::Bmm) ++s.bmm;
        else if (g.kind==Kind::Index) ++s.index;
        g.restore();
        s.check_healthy(); // failed restoration cannot be reported as successful completion
        // Newly discovered outputs can exceed a byte budget; close before return.
        // No second call can enter with this over-budget live set.
        if (s.retained_bytes>s.byte_limit) { s.drain(); ++s.budget_drains; }
    }
    g.finished=true;
}
void ImmediateGuard::fail(std::exception_ptr error) noexcept
{
    if (!impl_ || impl_->finished) return;
    auto& g=*impl_; auto& s=*g.state;
    g.restore(); g.finished=true;
    if (g.nested) return; // the outer owner captures the original exception
    for (std::size_t i=0;i<g.output_count;++i) s.pending_outputs[i]=g.outputs[i];
    ++s.failed;
    try { if (error) std::rethrow_exception(error); s.remember("immediate entry ended without completion"); }
    catch (const std::exception& e) { s.remember(e.what()); }
    catch (...) { s.remember("non-standard immediate entry failure"); }
    s.protocol.poison();
    if (!s.recover()) s.quarantine_anchor=g.state;
}
void ImmediateBoundary(const char* reason)
{
    (void)reason;
    if (internal_depth || qualified_body_depth) return;
    auto s=state(); if (!s || !s->enabled.load(std::memory_order_acquire)) return;
    s->check_owner(); s->check_healthy(); ++s->boundaries;
    try { s->drain(); }
    catch (const std::exception& error) {
        s->remember(error.what()); s->protocol.poison(); if (!s->recover()) s->quarantine_anchor=s; throw;
    }
}
std::string ImmediateConfigure(bool enabled,const std::string& mode,uint64_t streams,uint64_t capacity,uint64_t bytes)
{
    if (mode!="e" && mode!="ei") throw std::invalid_argument("immediate mode must be e or ei");
    if (streams!=2 || capacity==0 || capacity>1024 || bytes==0 || bytes>512u*1024u*1024u)
        throw std::invalid_argument("invalid immediate runtime bounds");
    auto old=state();
    if (old) {
        old->check_owner();
        if (entry_depth) throw std::runtime_error("cannot reconfigure immediate runtime inside an operator");
        if (!old->closed) ImmediateDrainClose();
    }
    last_interaction.store(mode=="ei",std::memory_order_release);
    if (!enabled) {
        if (old) {
            old->enabled.store(false,std::memory_order_release); old->closed=true;
            runtime_enabled.store(false,std::memory_order_release);
        }
        return ImmediateSnapshot();
    }
    check_environment();
    InternalScope scope;
    auto next=std::make_shared<State>(mode=="ei",static_cast<std::size_t>(capacity),static_cast<std::size_t>(bytes));
    // An unclosed active runtime must retain owners even during library/process
    // destruction. Explicit successful close breaks this allocation-free anchor.
    next->quarantine_anchor=next;
    // Initial configuration can race from two host threads. Publication is a
    // CAS; no lock is held across event creation, drain or completion polling.
    if (!std::atomic_compare_exchange_strong_explicit(&current,&old,next,std::memory_order_acq_rel,std::memory_order_acquire)) {
        next->quarantine_anchor.reset();
        throw std::runtime_error("concurrent immediate configuration rejected");
    }
    runtime_enabled.store(true,std::memory_order_release);
    return ImmediateSnapshot();
}
std::string ImmediateDrainClose()
{
    auto s=state(); if (!s) return ImmediateSnapshot();
    s->check_owner();
    if (entry_depth) throw std::runtime_error("cannot close immediate runtime inside an operator");
    if (!s->closed) {
        if (s->protocol.poisoned()) {
            if (!s->recover()) { s->quarantine_anchor=s; throw std::runtime_error(s->first_error.data()); }
        } else {
            try { s->drain(); }
            catch (const std::exception& error) {
                s->remember(error.what()); s->protocol.poison(); if (!s->recover()) s->quarantine_anchor=s; throw;
            }
        }
        try { s->backend.retire(); }
        catch (const std::exception& error) {
            s->remember(error.what()); s->protocol.poison(); s->quarantined=true; s->quarantine_anchor=s; throw;
        }
        s->closed=true; s->enabled.store(false,std::memory_order_release);
        runtime_enabled.store(false,std::memory_order_release); s->quarantine_anchor.reset();
    }
    return ImmediateSnapshot();
}
std::string ImmediateSnapshot()
{
    auto s=state();
    if (!s) return std::string("{\"schema\":1,\"backend\":\"native_immediate\",\"mode\":\"")+
        (last_interaction.load(std::memory_order_acquire)?"ei":"e")+
        "\",\"enabled\":false,\"closure\":{\"admission_closed\":true,\"no_owned_device_work\":true,\"device_synchronized\":false,\"passed\":true}}";
    s->check_owner();
    std::ostringstream out;
    out << "{\"schema\":1,\"backend\":\"native_immediate\",\"protocol\":\"synchronous_host_submit_groups\",\"mode\":" << quote(s->interaction?"ei":"e")
        << ",\"enabled\":" << (s->enabled.load()?"true":"false")
        << ",\"actual_stream_scheduling\":true,\"real_sdk_qualified\":false,\"device_parallel_qualified\":false,\"host_workers\":0"
        << ",\"task_queue_required\":0,\"capacity\":" << s->capacity << ",\"byte_limit\":" << s->byte_limit
        << ",\"active_calls\":" << s->protocol.calls() << ",\"retained_bytes\":" << s->retained_bytes << ",\"peak_retained_bytes\":" << s->peak_bytes
        << ",\"retained_owners\":" << s->owner_count << ",\"scheduled_embedding\":" << s->embedding << ",\"scheduled_cat\":" << s->cat
        << ",\"owned_event_handles\":" << s->backend.live_events()
        << ",\"scheduled_bmm\":" << s->bmm << ",\"scheduled_index\":" << s->index << ",\"fallback\":" << s->fallback
        << ",\"aliases\":" << s->aliases << ",\"nested_entries\":" << s->nested << ",\"drains\":" << s->drains
        << ",\"budget_drains\":" << s->budget_drains << ",\"failed\":" << s->failed << ",\"boundaries\":" << s->boundaries
        << ",\"event_records\":" << s->protocol.records() << ",\"event_waits\":" << s->protocol.waits() << ",\"caller_joins\":" << s->protocol.joins()
        << ",\"first_error\":" << (s->first_error[0]?quote(s->first_error.data()):"null")
        << ",\"quarantined\":" << (s->quarantined?"true":"false")
        << ",\"stream_ids\":[" << s->backend.streams[0].id() << ',' << s->backend.streams[1].id() << ',' << s->backend.streams[2].id() << ',' << s->backend.streams[3].id()
        << "],\"closure\":{\"admission_closed\":" << (s->closed?"true":"false")
        << ",\"no_owned_device_work\":" << (!s->protocol.active() && !s->owner_count?"true":"false")
        << ",\"device_synchronized\":" << (s->closed && !s->protocol.active()?"true":"false")
        << ",\"passed\":" << (s->closed && !s->protocol.active() && !s->owner_count && !s->backend.live_events() && !s->protocol.poisoned()?"true":"false") << "}}";
    return out.str();
}
} // namespace c10_npu::eager_parallel
