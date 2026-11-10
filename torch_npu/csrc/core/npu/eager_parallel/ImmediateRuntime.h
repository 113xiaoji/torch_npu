#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>
#include <array>
#include <exception>
#include <memory>
#include <string>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu::eager_parallel {
// Experimental actual stream scheduling. This is not the FrozenCall protocol:
// output handles are discovered after an original synchronous-host-submit body.
// The body owns new allocations on its selected stream; fixed owner/event slots
// are reserved before entry, and allocator recordStream protects cross-stream use.
C10_NPU_API bool RuntimeEnabled() noexcept;
C10_NPU_API std::string ImmediateConfigure(bool enabled, const std::string& mode,
    uint64_t streams, uint64_t capacity, uint64_t bytes, uint64_t min_embedding_bag_length=0);
C10_NPU_API std::string ImmediateSnapshot();
C10_NPU_API std::string ImmediateDrainClose();
C10_NPU_API void ImmediateBoundary(const char* reason);
// Lean opt-in path: three input handles, no per-call owner arrays or happy-path
// output traversal. State is type-erased here to keep backend details private.
class C10_NPU_API ImmediateCallerGuard {
public:
    ImmediateCallerGuard() = default;
    ~ImmediateCallerGuard() noexcept;
    ImmediateCallerGuard(const ImmediateCallerGuard&) = delete;
    ImmediateCallerGuard& operator=(const ImmediateCallerGuard&) = delete;
    bool begin(const char* schema, const at::Tensor& weight, const at::Tensor& indices,
        const at::Tensor& offsets, bool include_last_offset);
    bool disqualified() const noexcept { return disqualified_; }
    void complete();
    void add_failed_output(const at::Tensor& value) noexcept;
    void fail(std::exception_ptr error = {}) noexcept;
private:
    void leave() noexcept;
    std::shared_ptr<void> state_;
    std::array<at::Tensor,3> inputs_;
    std::size_t output_count_=0;
    bool entered_=false,body_scope_=false,finished_=false,disqualified_=false;
};
struct ImmediateGuardImpl;
class C10_NPU_API ImmediateGuard {
public:
    ImmediateGuard(const char* schema, const char* route);
    ~ImmediateGuard() noexcept;
    ImmediateGuard(const ImmediateGuard&) = delete;
    ImmediateGuard& operator=(const ImmediateGuard&) = delete;
    void disqualify() noexcept;
    void add_input(const at::Tensor& value);
    void begin();
    void add_output(const at::Tensor& value);
    void complete();
    void fail(std::exception_ptr error = {}) noexcept;
private:
    std::unique_ptr<ImmediateGuardImpl> impl_;
};
} // namespace c10_npu::eager_parallel
