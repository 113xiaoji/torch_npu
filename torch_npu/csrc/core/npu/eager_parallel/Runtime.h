#pragma once
#include <cstdint>
#include <string>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
namespace c10_npu::eager_parallel {
C10_NPU_API std::string Configure(bool enabled,const std::string& mode,uint64_t streams,uint64_t capacity,uint64_t bytes);
C10_NPU_API std::string Snapshot();
C10_NPU_API std::string DrainClose();
}
