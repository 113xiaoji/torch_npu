#pragma once
#include "torch_npu/csrc/core/npu/EagerPrepareQueueCore.h"
#include "torch_npu/csrc/core/npu/HostRegionOpApiCommand.h"
#include <functional>
#include <string>

namespace c10_npu::eager_prepare {
// Implemented exclusively by the native runtime. No Python callback/object is
// accepted: this context owns the captured native caller state and SDK domain.
struct OwnedExecutionContext {
    virtual ~OwnedExecutionContext() = default;
    virtual void Prepare(host_regions::OwnedOpApiPayload&, void* stream) = 0;
    virtual int Execute(host_regions::OwnedOpApiPayload&, void* stream) = 0;
    virtual void Cancel(host_regions::OwnedOpApiPayload&) noexcept = 0;
    virtual bool CleanupFailed() const noexcept = 0;
    virtual void Failed(const char*, std::exception_ptr) noexcept {}
};

class OwnedQueueJob final : public queue::EagerPrepareQueueCore::Job {
public:
    using Observe = std::function<void(const char*)>;
    OwnedQueueJob(std::unique_ptr<host_regions::OwnedOpApiPayload> payload,
                  std::shared_ptr<OwnedExecutionContext> context, void* stream, Observe observe)
        : payload_(std::move(payload)), context_(std::move(context)), stream_(stream), observe_(std::move(observe))
    {
        if (!payload_ || !context_ || !stream_) throw std::invalid_argument("Eager Prepare job requires owned native arguments/context/stream");
    }
    void Prepare() override {
        try {
            context_->Prepare(*payload_,stream_);
            if (CleanupFailed()) ThrowCleanupFailure();
            ObserveEvent("prepared");
        } catch (...) { context_->Failed("prepare",std::current_exception()); throw; }
    }
    void Execute() override {
        try {
            ObserveEvent("executed");
            const int status = context_->Execute(*payload_,stream_);
            ObserveEvent("returned");
            if (status != 0) throw std::runtime_error("Eager Prepare SDK Execute returned status="+std::to_string(status));
            if (CleanupFailed()) ThrowCleanupFailure();
        } catch (...) { context_->Failed("execute",std::current_exception()); throw; }
    }
    void Cancel() noexcept override {
        if (!payload_) return;
        may_have_submitted_=payload_->MayHaveSubmitted();
        context_->Cancel(*payload_);
        cleanup_failed_=context_->CleanupFailed() || bool(payload_->CleanupFailure());
        if (!cleanup_failed_) {
            // Called by core only after Prepare ACK or real device retirement.
            // Drop byte-owning payload BEFORE core returns its credit, even if
            // a FIFO retirement closure still strongly owns this Job wrapper.
            payload_.reset(); context_.reset();
        }
    }
    size_t RetainedBytes() const noexcept override { return payload_ ? payload_->RetainedBytes() : 0; }
    bool CleanupFailed() const noexcept override {
        return cleanup_failed_ || (context_ && context_->CleanupFailed()) || (payload_ && bool(payload_->CleanupFailure()));
    }
    bool MayHaveSubmitted() const noexcept { return payload_ ? payload_->MayHaveSubmitted() : may_have_submitted_; }
private:
    void ThrowCleanupFailure() const {
        if (const auto failure=payload_->CleanupFailure()) std::rethrow_exception(failure);
        throw std::runtime_error("Eager Prepare native context cleanup failed");
    }
    void ObserveEvent(const char* event) { if (observe_) observe_(event); }
    std::unique_ptr<host_regions::OwnedOpApiPayload> payload_;
    std::shared_ptr<OwnedExecutionContext> context_;
    void* stream_;
    Observe observe_;
    bool may_have_submitted_=false,cleanup_failed_=false;
};
} // namespace c10_npu::eager_prepare
