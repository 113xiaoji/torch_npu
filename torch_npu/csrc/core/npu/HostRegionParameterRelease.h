#pragma once

#include "torch_npu/csrc/core/npu/HostRegionParameterArrays.h"

namespace at_npu::native {
struct ExecuteParas;
// Internal free function: no field, virtual method, or published payload layout
// changes. Callers guard non-idempotent ACL attempts before entering this API.
void ReleaseLegacyParameters(ExecuteParas&, c10_npu::host_regions::ParameterArrayOwnership);
} // namespace at_npu::native
