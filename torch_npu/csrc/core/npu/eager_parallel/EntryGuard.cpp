#include "torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h"
#include "torch_npu/csrc/core/npu/eager_parallel/EntryDiagnostics.h"
#include "torch_npu/csrc/core/npu/eager_parallel/Runtime.h"
#include <sstream>

namespace c10_npu::eager_parallel {
namespace {
std::atomic<EntryDiagnostics*> active_diagnostics{nullptr};
std::atomic<bool> immediate_selected{false};
std::atomic_flag control_reserved = ATOMIC_FLAG_INIT;
struct ControlReservation {
    ControlReservation() {
        if (control_reserved.test_and_set(std::memory_order_acquire))
            throw std::logic_error("concurrent eager-parallel lifecycle call rejected");
    }
    ~ControlReservation() { control_reserved.clear(std::memory_order_release); }
};
EntryDiagnostics& diagnostics() {
    // Configuration allocates this process-lifetime diagnostic domain. Disabled
    // operator entries never allocate, including first use during module import.
    static auto* value=[] {
        auto* created=new EntryDiagnostics;
        active_diagnostics.store(created,std::memory_order_release);
        return created;
    }();
    return *value;
}
std::string quoted(const std::string& value) {
    std::ostringstream out;out << '"';
    for (unsigned char ch:value) {
        if (ch=='"' || ch=='\\') out << '\\' << char(ch);
        else if (ch>=32) out << char(ch); // Preserve UTF-8 diagnostics.
        else { const char* hex="0123456789abcdef";out << "\\u00" << hex[ch>>4] << hex[ch&15]; }
    }
    out << '"';return out.str();
}
}
std::string SnapshotImpl();
bool Enabled() noexcept {
    auto* value=active_diagnostics.load(std::memory_order_acquire);
    return value && value->enabled();
}
void Boundary(const char* reason) {
    if (RuntimeEnabled()) ImmediateBoundary(reason);
    auto* value=active_diagnostics.load(std::memory_order_acquire);
    if(value) value->boundary(reason);
}
EntryGuard::EntryGuard(const char* schema,const char* route,std::size_t inputs) noexcept {
    if(!Enabled()) return;
    if (inputs==detail::IncompleteTensorCount) { diagnostics().mark_incomplete();inputs=0; }
    token_=diagnostics().enter(schema,route,inputs);
}
EntryGuard::~EntryGuard() noexcept {
    if (token_ && !finished_) diagnostics().fail(token_,"entry ended without completion acknowledgement");
}
void EntryGuard::complete(std::size_t outputs) noexcept {
    if(!token_) { finished_=true;return; }
    if (outputs==detail::IncompleteTensorCount) { diagnostics().mark_incomplete();outputs=0; }
    if (!finished_) { diagnostics().complete(token_,outputs);finished_=true; }
}
void EntryGuard::fail(std::exception_ptr error) noexcept {
    if (finished_) return;
    if(!token_) { finished_=true;return; }
    try {
        if (error) std::rethrow_exception(error);
        diagnostics().fail(token_,"unknown native entry failure");
    } catch(const std::exception& original) { diagnostics().fail(token_,original.what()); }
      catch(...) { diagnostics().fail(token_,"non-standard native entry failure"); }
    finished_=true;
}
std::string Configure(bool enabled,const std::string& mode,uint64_t streams,uint64_t capacity,uint64_t bytes) {
    ControlReservation reservation;
    if (mode == "e" || mode == "ei") {
        if (Enabled()) throw std::logic_error("close the native shadow domain first");
        auto result = ImmediateConfigure(enabled,mode,streams,capacity,bytes);
        immediate_selected.store(true,std::memory_order_release);
        return result;
    }
    if (RuntimeEnabled()) throw std::logic_error("close the native execution domain first");
    if (mode!="shadow" || streams!=2) throw std::invalid_argument("only original-stream shadow diagnostics is available; device execution is not qualified");
    diagnostics().configure(enabled,capacity,bytes);
    immediate_selected.store(false,std::memory_order_release);
    return SnapshotImpl();
}
std::string Snapshot() {
    ControlReservation reservation;
    return SnapshotImpl();
}
std::string SnapshotImpl() {
    if (immediate_selected.load(std::memory_order_acquire)) return ImmediateSnapshot();
    const auto s=diagnostics().snapshot();std::ostringstream out;
    out << "{\"schema\":1,\"backend\":\"shadow\",\"mode\":\"shadow\",\"enabled\":" << (s.enabled?"true":"false")
        << ",\"real_sdk_qualified\":false,\"device_parallel_qualified\":false,\"host_workers\":0"
        << ",\"effects_registered\":false,\"full_entry_coverage_qualified\":false"
        << ",\"source_coverage\":\"generated_dispatcher_and_op_plugin\",\"capacity\":" << s.capacity
        << ",\"byte_limit\":" << s.byte_limit << ",\"entered\":" << s.entered
        << ",\"completed\":" << s.completed << ",\"failed\":" << s.failed << ",\"nested_entries\":" << s.nested
        << ",\"active_calls\":" << s.active << ",\"foreign_thread_calls\":" << s.foreign_calls
        << ",\"coverage_incomplete\":" << (s.coverage_incomplete?"true":"false")
        << ",\"first_error\":" << (s.first_error.empty()?"null":quoted(s.first_error))
        << ",\"operator_counts\":{";
    bool first=true;
    for (const auto& row:s.operators) {
        if(!first) { out << ','; }
        first=false;
        const auto& n=row.second;
        out << quoted(row.first) << ":{\"entered\":" << n.entered << ",\"completed\":" << n.completed
            << ",\"failed\":" << n.failed << ",\"inputs\":" << n.inputs << ",\"outputs\":" << n.outputs << '}';
    }
    out << "},\"boundaries\":{";
    first=true;
    for(const auto& row:s.boundaries) {
        if(!first) { out << ','; }
        first=false;
        out << quoted(row.first) << ':' << row.second;
    }
    out << "},\"closure\":{\"admission_closed\":" << (s.closed?"true":"false")
        << ",\"no_owned_device_work\":true,\"device_synchronized\":false,\"passed\":"
        << (s.closed && s.active==0?"true":"false") << "}}";
    return out.str();
}
std::string DrainClose() {
    ControlReservation reservation;
    if (immediate_selected.load(std::memory_order_acquire)) return ImmediateDrainClose();
    diagnostics().close();return SnapshotImpl();
}
} // namespace c10_npu::eager_parallel
