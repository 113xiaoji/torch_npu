#pragma once

#include <atomic>

namespace c10_npu {
namespace detail {

// Published while the allocator's existing pool-mutation mutex is held. This
// is an eligibility observation, not synchronization with a concurrently
// changed stream/graph scope. Legal scope entry/exit ordering is unchanged.
struct alignas(64) CapturePoolActivity final {
    void Publish(bool active) noexcept
    {
        active_.store(active, std::memory_order_release);
    }

    bool Active() const noexcept
    {
        return active_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> active_{false};
};

static_assert(std::atomic<bool>::is_always_lock_free, "capture eligibility must be lock-free");
static_assert(sizeof(CapturePoolActivity) == 64, "keep hot allocator fields off the eligibility cache line");

} // namespace detail
} // namespace c10_npu
