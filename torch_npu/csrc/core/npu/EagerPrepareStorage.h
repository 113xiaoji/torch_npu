#pragma once
#include "torch_npu/csrc/core/npu/NPUMacros.h"
namespace c10 { class DataPtr; struct Allocator; }
namespace c10_npu::NPUCachingAllocator {
// Explicit R24 activation; the legacy env-cached R22 predicate is unchanged.
C10_NPU_API bool IsNativeEagerRetirementData(const c10::DataPtr&,const c10::Allocator*,int device,void* stream) noexcept;
}
