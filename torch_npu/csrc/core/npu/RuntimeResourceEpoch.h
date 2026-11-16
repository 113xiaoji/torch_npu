#pragma once

#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {

// Cold resource-lifecycle transitions invalidate caches that have no generation
// token for their raw streams/storage. This is an observation, not a barrier:
// existing callers must still quiesce all work before reset/recovery/finalize.
TORCH_NPU_API uint64_t GetRuntimeResourceEpoch() noexcept;
TORCH_NPU_API void InvalidateRuntimeResources() noexcept;

} // namespace c10_npu
