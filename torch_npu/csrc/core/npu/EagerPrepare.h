#pragma once
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/HostRegionOpApiCommand.h"
#include <cstdint>
#include <string>
#include <vector>
namespace at { class Tensor; }
namespace c10_npu::eager_prepare {
struct Configuration {
    bool enabled = false;
    size_t workers = 4, capacity = 8, byte_limit = 536870912;
    bool diagnostic_trace = false;
    size_t trace_limit = 256;
    std::vector<std::string> allowed_ops;
};
C10_NPU_API bool Enabled() noexcept;
C10_NPU_API bool DiagnosticTracing() noexcept;
// Every enabled macro invocation is counted once. false means the original
// route is still responsible; true transfers no owner until Submit succeeds.
C10_NPU_API bool Admit(const char* operation, bool qualified, const char* decline_reason, bool argument_domain = true);
C10_NPU_API bool CanQueueTensor(const at::Tensor&);
C10_NPU_API void ValidateTensor(const at::Tensor&);
C10_NPU_API void Submit(const char* operation, std::unique_ptr<host_regions::OwnedOpApiPayload> payload,
    std::string diagnostic_signature = {});
C10_NPU_API std::string Configure(Configuration);
C10_NPU_API std::string Snapshot();
C10_NPU_API std::string DrainClose();
C10_NPU_API void BeforeRuntimeFinalize();
C10_NPU_API bool NativeJobActive() noexcept;
// Called in Repository's nonthrowing tail of its successful deep copy. The
// thread-local marker only recognizes this publication's exact QueueParas.
C10_NPU_API void QueueCopyAccepted(const void*) noexcept;
} // namespace c10_npu::eager_prepare
