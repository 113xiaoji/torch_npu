#pragma once
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu::eager_parallel {
// Allocator event bookkeeping is not an eager observation/admission boundary.
// Scope only the known allocator event call; SDK errors keep existing behavior.
class C10_NPU_API AllocatorEventScope {
public:
    AllocatorEventScope() noexcept;
    ~AllocatorEventScope() noexcept;
    AllocatorEventScope(const AllocatorEventScope&) = delete;
    AllocatorEventScope& operator=(const AllocatorEventScope&) = delete;
};
} // namespace c10_npu::eager_parallel
