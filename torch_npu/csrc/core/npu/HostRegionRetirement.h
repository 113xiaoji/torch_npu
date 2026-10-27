#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include "torch_npu/csrc/core/npu/HostRegionPacket.h"

namespace c10 { class DataPtr; struct Allocator; }
namespace at { class Tensor; }
namespace c10_npu::NPUCachingAllocator {
// Defined alongside the concrete allocator, never trust a pluggable allocator's
// name or raw_deleter() virtual method as evidence of a native-only deleter.
bool IsNativeCapturedRetirementData(const c10::DataPtr&, const c10::Allocator*) noexcept;
}

namespace c10_npu::host_regions {

// Worker-only admission for a private storage owner. The original Tensor's
// Python slot/autograd metadata are not copied into that never-exposed holder.
// Its Storage and allocator still require the complete native/immutable audit.
bool CanCreateCapturedStorageOwner(const at::Tensor& original) noexcept;

struct CapturedRetirementSnapshot {
    uint64_t pending, retired, peak, transferred, fallback, handoffs;
};

#ifdef HDB_NATIVE_CORE_TEST
inline CapturedRetirementSnapshot GetCapturedRetirementSnapshot() noexcept { return {}; }
inline bool HasCapturedOwnerRetirement(void*) noexcept { return false; }
inline bool IsCapturedRetirementWorker() noexcept { return false; }
class CapturedRetirementWorkerScope {
public:
    CapturedRetirementWorkerScope() noexcept = default;
};
inline void DiscardCapturedOwners(void*) noexcept {}
inline void DiscardCapturedBufferOwners(void*) noexcept {}
#else

namespace retirement_detail {
// One cold allocation per existing queue buffer. Parameter layouts are unchanged;
// the trailer is private to the callback's returned stride, never an ACL payload.
struct alignas(std::max_align_t) BufferState {
    size_t capacity, stride, metadata_offset;
    std::atomic<uint64_t> pending{0};
    BufferState(size_t count, size_t step, size_t offset) noexcept
        : capacity(count), stride(step), metadata_offset(offset) {}
};
struct alignas(std::max_align_t) SlotState {
    BufferState* buffer;
    void* payload = nullptr;
    void (*destroy)(void*) noexcept = nullptr;
    uint64_t owners = 0;
    unsigned kind = 0;
    bool captured = false;
    bool safe_cleanup = false;
    PacketCommandRef packet;
    bool packet_pending = false;
    explicit SlotState(BufferState* state) noexcept : buffer(state) {}
};
struct Counters {
    std::atomic<uint64_t> pending{0}, retired{0}, peak{0}, transferred{0}, fallback{0}, handoffs{0};
};
inline Counters counters;
inline thread_local unsigned worker_depth = 0;

inline size_t Align(size_t bytes) noexcept
{
    constexpr size_t alignment = alignof(std::max_align_t);
    return (bytes + alignment - 1) & ~(alignment - 1);
}
inline BufferState* Buffer(void* data) noexcept
{
    return data ? static_cast<BufferState*>(data) - 1 : nullptr;
}
inline SlotState& Slot(void* slot, size_t payload_bytes) noexcept
{
    return *reinterpret_cast<SlotState*>(static_cast<unsigned char*>(slot) + Align(payload_bytes));
}
inline void* Allocate(size_t payload_bytes, size_t capacity, size_t& stride)
{
    if (!capacity || payload_bytes > std::numeric_limits<size_t>::max() - sizeof(SlotState) - 32)
        throw std::bad_alloc();
    const size_t metadata = Align(payload_bytes);
    stride = Align(metadata + sizeof(SlotState));
    if (capacity > (std::numeric_limits<size_t>::max() - sizeof(BufferState)) / stride)
        throw std::bad_alloc();
    void* allocation = std::malloc(sizeof(BufferState) + capacity * stride);
    if (!allocation) throw std::bad_alloc();
    auto* state = new (allocation) BufferState(capacity, stride, metadata);
    void* data = state + 1;
    std::memset(data, 0, capacity * stride);
    for (size_t index = 0; index < capacity; ++index)
        new (static_cast<unsigned char*>(data) + index * stride + metadata) SlotState(state);
    return data;
}

// Called only by the slot's current owner, and after release worker join during
// final cleanup. No refcount destruction occurs under a queue-core mutex.
template<class Payload>
inline Payload& Construct(SlotState& slot, void* payload, unsigned kind = 0)
{
    assert(slot.owners == 0 && !slot.packet_pending);
    if (slot.destroy) slot.destroy(slot.payload);
    slot.payload = nullptr;
    slot.destroy = nullptr;
    slot.captured = false;
    slot.safe_cleanup = false;
    slot.kind = kind;
    auto* result = new (payload) Payload();
    slot.payload = result;
    slot.destroy = [](void* value) noexcept { static_cast<Payload*>(value)->~Payload(); };
    return *result;
}
// An acquired packet command is the payload. No ExecuteParas is constructed in
// either queue slot and no additional shared_ptr/refcount is copied per hop.
inline void BorrowPacket(SlotState& slot, const PacketCommandRef& packet, unsigned kind) noexcept
{
    assert(packet.packet && !slot.owners && !slot.packet_pending);
    if (slot.destroy) slot.destroy(slot.payload);
    slot.payload = nullptr;
    slot.destroy = nullptr;
    slot.kind = kind;
    slot.captured = true;
    slot.safe_cleanup = true;
    slot.packet = packet;
    slot.packet_pending = true;
    slot.buffer->pending.fetch_add(1, std::memory_order_release);
}
inline void TransferPacket(SlotState& destination, SlotState& source) noexcept
{
    assert(source.packet_pending && source.packet.packet);
    BorrowPacket(destination, source.packet, source.kind);
    source.packet = {};
    source.packet_pending = false;
    source.buffer->pending.fetch_sub(1, std::memory_order_release);
}
template<class Retire>
inline void RetirePacket(SlotState& slot, Retire&& retire) noexcept
{
    if (!slot.packet_pending) return;
    retire(slot.packet);
    slot.packet = {};
    slot.packet_pending = false;
    slot.buffer->pending.fetch_sub(1, std::memory_order_release);
}
template<class Header, class Retire>
inline void RetirePacketHeader(SlotState& slot, Header& header, Retire&& retire) noexcept
{
    if (!slot.packet_pending) return;
    // The callback may drop the packet's final lease and delete the borrowed
    // payload. Keep a failing retained queue head safe to inspect afterward.
    header.paramVal = nullptr;
    RetirePacket(slot, std::forward<Retire>(retire));
}
inline void Adopt(SlotState& slot, uint64_t count) noexcept
{
    assert(!slot.owners);
    slot.owners = count;
    slot.safe_cleanup = true;
    if (!count) return;
    slot.buffer->pending.fetch_add(count, std::memory_order_release);
    counters.transferred.fetch_add(count, std::memory_order_relaxed);
    const uint64_t current = counters.pending.fetch_add(count, std::memory_order_release) + count;
    uint64_t peak = counters.peak.load(std::memory_order_relaxed);
    while (peak < current && !counters.peak.compare_exchange_weak(
        peak, current, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}
template<class Clear>
inline void Retire(SlotState& slot, Clear&& clear) noexcept
{
    const uint64_t count = slot.owners;
    if (!count) return;
    clear(); // All accepted owners are proven native before the source move.
    slot.owners = 0;
    counters.retired.fetch_add(count, std::memory_order_relaxed);
    counters.pending.fetch_sub(count, std::memory_order_release);
    slot.buffer->pending.fetch_sub(count, std::memory_order_release);
}
template<class OwnerVector>
class OwnerGuard {
public:
    OwnerGuard(SlotState& slot, OwnerVector& owners) noexcept : slot_(slot), owners_(owners) {}
    ~OwnerGuard() noexcept { Retire(slot_, [this]() noexcept { owners_.clear(); }); }
    OwnerGuard(const OwnerGuard&) = delete;
    OwnerGuard& operator=(const OwnerGuard&) = delete;
private:
    SlotState& slot_;
    OwnerVector& owners_;
};

// Shared by the actual release callback and the standalone lifetime probes.
// reserve() is the sole allocating operation and precedes any source move.
// Tensor's move constructor is noexcept; after reserve each push is allocation-free.
template<class OwnerVector>
inline void Transfer(SlotState& slot, OwnerVector& destination, OwnerVector& source, uint64_t count,
    SlotState* previous = nullptr)
{
    static_assert(std::is_nothrow_move_constructible<typename OwnerVector::value_type>::value,
        "Retirement cannot recover from a throwing owner move");
    assert(destination.empty() && !slot.owners);
    destination.reserve(source.size());
    for (auto& value : source) destination.emplace_back(std::move(value));
    source.clear();
    if (previous && previous->owners) {
        assert(previous->owners == count);
        // The same references move from main queue to bounded release lane.
        // Overall live/transferred counters count them once, not once per hop.
        slot.owners = count;
        slot.safe_cleanup = true;
        slot.buffer->pending.fetch_add(count, std::memory_order_release);
        previous->owners = 0;
        previous->buffer->pending.fetch_sub(count, std::memory_order_release);
    } else {
        Adopt(slot, count);
    }
    counters.handoffs.fetch_add(count, std::memory_order_relaxed);
}

template<class Cleanup>
inline void ForEachSlot(void* data, Cleanup&& cleanup) noexcept
{
    auto* state = Buffer(data);
    if (!state) return;
    for (size_t index = 0; index < state->capacity; ++index) {
        void* raw = static_cast<unsigned char*>(data) + index * state->stride;
        auto& slot = *reinterpret_cast<SlotState*>(static_cast<unsigned char*>(raw) + state->metadata_offset);
        cleanup(raw, slot);
    }
}
template<class Cleanup>
inline void DestroyBuffer(void* data, Cleanup&& cleanup) noexcept
{
    auto* state = Buffer(data);
    if (!state) return;
    ForEachSlot(data, [&cleanup](void* raw, SlotState& slot) noexcept {
        cleanup(raw, slot);
        assert(!slot.owners && !slot.packet_pending);
        slot.~SlotState();
    });
    assert(!state->pending.load(std::memory_order_acquire));
    state->~BufferState();
    std::free(state);
}
} // namespace retirement_detail

inline CapturedRetirementSnapshot GetCapturedRetirementSnapshot() noexcept
{
    auto& values = retirement_detail::counters;
    return {values.pending.load(std::memory_order_acquire), values.retired.load(std::memory_order_relaxed),
        values.peak.load(std::memory_order_relaxed), values.transferred.load(std::memory_order_relaxed),
        values.fallback.load(std::memory_order_relaxed), values.handoffs.load(std::memory_order_relaxed)};
}
inline bool HasCapturedOwnerRetirement(void* release_buffer) noexcept
{
    auto* state = retirement_detail::Buffer(release_buffer);
    return state && state->pending.load(std::memory_order_acquire) != 0;
}
inline bool IsCapturedRetirementWorker() noexcept { return retirement_detail::worker_depth != 0; }
class CapturedRetirementWorkerScope {
public:
    CapturedRetirementWorkerScope() noexcept { ++retirement_detail::worker_depth; }
    ~CapturedRetirementWorkerScope() noexcept { --retirement_detail::worker_depth; }
    CapturedRetirementWorkerScope(const CapturedRetirementWorkerScope&) = delete;
    CapturedRetirementWorkerScope& operator=(const CapturedRetirementWorkerScope&) = delete;
};
// Defined with the actual ExecuteParas callback implementation. This only clears
// newly admitted native owners, including a partial-ACL-release failing head.
void DiscardCapturedOwners(void* queue_parameter) noexcept;
void DiscardCapturedBufferOwners(void* buffer) noexcept;
#endif

} // namespace c10_npu::host_regions
