#pragma once
#include "torch_npu/csrc/core/npu/NPUMacros.h"
namespace c10_npu {
class NPUStream;
namespace prefix {
// true: real prior prefix completed. false: old pre-sync is still required.
// Errors throw before this request submits its unchanged synchronous memcpy.
C10_NPU_API bool TryH2DPrewait(const NPUStream& stream);
}
}
