#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <exception>

namespace at { class Tensor; }
namespace c10_npu::host_regions {
// The typed op-plugin implementation owns metadata, native Storage and SDK
// resources once. FIFO slots borrow its enclosing Command through packet leases.
class OwnedOpApiPayload {
public:
    virtual ~OwnedOpApiPayload() noexcept = default;
    // Called on the worker only after command/event capacity has been reserved.
    // The exact SDK adapter may override this to prepare an executor. A throw
    // leaves the unique_ptr with its caller for legal never-submitted cleanup.
    virtual void Prepare(void* actual_stream) { (void)actual_stream; }
    virtual int Execute(void* actual_stream) = 0; // Direct SDK calls; never enqueue.
    virtual void Cancel() noexcept = 0; // Exactly-once cleanup of a never-submitted plan.
    virtual bool MayHaveSubmitted() const noexcept = 0;
    // Host SDK cleanup can fail after Execute returned or during cancellation.
    // Retirement must expose that same request's first failure, not just poison
    // the next run. Older/private payload fixtures have no such failure.
    virtual std::exception_ptr CleanupFailure() const noexcept { return {}; }
    virtual size_t RetainedBytes() const noexcept = 0;
};
// Supplied by the typed SDK adapter. Symbol presence alone is not this contract.
bool OwnedOpApiSdkContractAvailable() noexcept;
const char* OwnedOpApiPrepareStrategy() noexcept;
void PrepareOwnedOpApiCallerState();
bool UseCapturedOwnedOpApi() noexcept;
// Mode-only, including a failed/aborted real scope. Macros must not fall back to
// old SDK preparation after a dynamic failure; first require Use... above.
bool OwnedOpApiEnabled() noexcept;
void ValidateOwnedOpApiTensor(const at::Tensor&);
// false leaves payload untouched. Once moved, any subsequent error is terminal:
// an accepted prefix must never be rerun by a serial fallback.
bool CaptureOwnedOpApi(const char* name, std::unique_ptr<OwnedOpApiPayload>& payload, int device);

namespace device_retirement_detail {
struct CompletionHooks {
    void* (*create)(void* context);
    void (*record)(void* event, void* stream, void* context);
    void (*wait)(void* event, void* context);
    void (*destroy)(void* event, void* context) noexcept;
};
// Constructed before parameter adoption. Only the final publisher/host-slot
// lease calls Complete, after all actual SDK invocations have returned.
class CompletionFence {
public:
    CompletionFence(CompletionHooks hooks, void* context, void* stream)
        : hooks_(hooks), context_(context), stream_(stream), event_(hooks.create(context)) {
        if (!event_) throw std::bad_alloc();
    }
    CompletionFence(const CompletionFence&) = delete;
    CompletionFence& operator=(const CompletionFence&) = delete;
    ~CompletionFence() noexcept { if (event_) hooks_.destroy(event_, context_); }
    bool Complete(bool may_have_submitted) noexcept {
        if (attempted_) return complete_;
        attempted_ = true;
        if (!may_have_submitted) return complete_ = true;
        try { hooks_.record(event_, stream_, context_); recorded_ = true; hooks_.wait(event_, context_); complete_ = true; }
        catch (...) { failure_ = std::current_exception(); }
        return complete_;
    }
    bool Recorded() const noexcept { return recorded_; }
    std::exception_ptr Failure() const noexcept { return failure_; }
private:
    CompletionHooks hooks_;
    void *context_, *stream_, *event_;
    bool attempted_ = false, recorded_ = false, complete_ = false;
    std::exception_ptr failure_;
};
} // namespace device_retirement_detail
#ifdef HDB_NATIVE_CORE_TEST
void SetOwnedOpApiCompletionHooksForTest(device_retirement_detail::CompletionHooks) noexcept;
#endif
} // namespace c10_npu::host_regions
