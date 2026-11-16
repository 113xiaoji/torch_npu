#include "torch_npu/csrc/core/npu/RuntimeResourceEpoch.h"

#include <atomic>

namespace c10_npu {
namespace {
std::atomic<uint64_t> runtime_resource_epoch{0};
} // namespace

uint64_t GetRuntimeResourceEpoch() noexcept
{
    return runtime_resource_epoch.load(std::memory_order_acquire);
}

void InvalidateRuntimeResources() noexcept
{
    runtime_resource_epoch.fetch_add(1, std::memory_order_acq_rel);
}

} // namespace c10_npu
