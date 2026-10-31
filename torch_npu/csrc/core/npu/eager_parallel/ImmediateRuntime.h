#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>
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
    uint64_t streams, uint64_t capacity, uint64_t bytes);
C10_NPU_API std::string ImmediateSnapshot();
C10_NPU_API std::string ImmediateDrainClose();
C10_NPU_API void ImmediateBoundary(const char* reason);
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
