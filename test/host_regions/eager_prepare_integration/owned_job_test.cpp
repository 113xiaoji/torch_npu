#include "torch_npu/csrc/core/npu/EagerPrepareOwnedJob.h"
#include <cassert>
#include <stdexcept>

using namespace c10_npu::eager_prepare;
struct Stats { int prepares=0,executes=0,cancels=0,alive=0,status=0; bool submitted=false,failed=false; };
struct Payload final : c10_npu::host_regions::OwnedOpApiPayload {
    explicit Payload(Stats& stats) : s(stats) { ++s.alive; }
    ~Payload() noexcept override { --s.alive; }
    void Prepare(void*) override { ++s.prepares; }
    int Execute(void*) override { ++s.executes; s.submitted=true; return s.status; }
    void Cancel() noexcept override { if (!s.submitted) ++s.cancels; }
    bool MayHaveSubmitted() const noexcept override { return s.submitted; }
    size_t RetainedBytes() const noexcept override { return 7; }
    std::exception_ptr CleanupFailure() const noexcept override {
        return s.failed ? std::make_exception_ptr(std::runtime_error("cleanup")) : nullptr;
    }
    Stats& s;
};
struct Context final : OwnedExecutionContext {
    int entered=0;
    std::exception_ptr error;
    void Prepare(c10_npu::host_regions::OwnedOpApiPayload& p, void* stream) override { ++entered; p.Prepare(stream); }
    int Execute(c10_npu::host_regions::OwnedOpApiPayload& p, void* stream) override { ++entered; return p.Execute(stream); }
    void Cancel(c10_npu::host_regions::OwnedOpApiPayload& p) noexcept override { ++entered; p.Cancel(); }
    bool CleanupFailed() const noexcept override { return false; }
    void Failed(const char*,std::exception_ptr value) noexcept override { error=value; }
};
int main() {
    auto context=std::make_shared<Context>();
    Stats stats;
    auto payload=std::make_unique<Payload>(stats);
    int prepared=0, executed=0, returned=0;
    OwnedQueueJob job(std::move(payload),context,reinterpret_cast<void*>(1),
        [&](const char* event) { const std::string e(event); prepared+=e=="prepared"; executed+=e=="executed"; returned+=e=="returned"; });
    assert(job.RetainedBytes()==7);
    job.Prepare(); job.Execute(); job.Cancel();
    assert(stats.prepares==1 && stats.executes==1 && stats.cancels==0);
    assert(prepared==1 && executed==1 && returned==1 && context->entered==3);
    assert(job.MayHaveSubmitted() && !job.CleanupFailed());
    // Keep the Job alive just like the real Retirement closure and Submit
    // local do: accounting must follow actual owned resources, not wrapper refs.
    Stats retained;
    auto held=std::make_shared<OwnedQueueJob>(std::make_unique<Payload>(retained),std::make_shared<Context>(),
        reinterpret_cast<void*>(1),OwnedQueueJob::Observe{});
    using Core=c10_npu::queue::EagerPrepareQueueCore;
    Core core(Core::Config{1,64,1});
    auto ticket=core.Submit(held,[](const Core::TicketPtr&) { return Core::PublishReceipt{Core::PublishState::Committed}; });
    core.Consume(ticket); assert(retained.alive==1);
    core.Retire(ticket,true);
    assert(held && retained.alive==0); // Strong wrapper ref must no longer own bytes.
    assert(core.Snapshot().slots_in_use==0 && core.Snapshot().retained_bytes==0);
    core.Shutdown();
    Stats bad; bad.status=17;
    auto bad_context=std::make_shared<Context>();
    int bad_returns=0;
    OwnedQueueJob failure(std::make_unique<Payload>(bad),bad_context,reinterpret_cast<void*>(1),
        [&](const char* event) { bad_returns+=std::string(event)=="returned"; });
    failure.Prepare();
    std::exception_ptr original;
    try { failure.Execute(); } catch (...) { original=std::current_exception(); }
    assert(original && bad_context->error && bad_returns==1);
    // MSVC current_exception is permitted to copy exception objects. Verify
    // the latched original cause, without assuming exception_ptr address identity.
    std::string thrown,recorded;
    try { std::rethrow_exception(original); } catch (const std::exception& e) { thrown=e.what(); }
    try { std::rethrow_exception(bad_context->error); } catch (const std::exception& e) { recorded=e.what(); }
    assert(thrown==recorded && recorded=="Eager Prepare SDK Execute returned status=17");
    return 0;
}
