#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <limits>

#define C10_NPU_MPSC_CLOSE_GATE 1

namespace c10_npu::queue {

// Close is an admission barrier, not a drain. A successful pre-Close permit
// belongs to the accepted prefix and may publish afterwards. Callers must wait
// for ActiveProducers()==0 AND drain pending requests before destroying storage.
// Packing the bit and count into one atomic prevents new permits after Close.
class LockFreeIngressGate final {
public:
    bool TryEnter() noexcept {
        size_t current = state_.load();
        do { if ((current & closed_bit) || current == closed_bit - 1) return false; }
        while (!state_.compare_exchange_weak(current, current + 1));
        return true;
    }
    void Leave() noexcept { state_.fetch_sub(1); }
    void Close() noexcept { state_.fetch_or(closed_bit); }
    bool Closed() const noexcept { return (state_.load() & closed_bit) != 0; }
    size_t ActiveProducers() const noexcept { return state_.load() & (closed_bit - 1); }
    bool IsLockFree() const noexcept { return state_.is_lock_free(); }
private:
    static constexpr size_t closed_bit = size_t(1) << (std::numeric_limits<size_t>::digits - 1);
    static_assert(std::atomic<size_t>::is_always_lock_free, "ingress requires a lock-free close gate");
    std::atomic<size_t> state_{0};
};

// Fixed storage, multiple producers, exactly one consumer. Claim is a bounded
// scan; Publish is a lock-free CAS loop. Allocation happens only at construction.
// A claimed but unpublished descriptor creates no hole in the visible queue.
template <typename Value>
class LockFreeMpscIngress final {
public:
    struct Node {
        Value value{};
        Node* next = nullptr;
        std::atomic<bool> claimed{false};
    };
    explicit LockFreeMpscIngress(size_t capacity) : capacity_(capacity), nodes_(new Node[capacity]) {
        static_assert(std::atomic<Node*>::is_always_lock_free, "ingress requires lock-free pointers");
        static_assert(std::atomic<bool>::is_always_lock_free, "ingress requires lock-free descriptor claims");
    }
    Node* TryClaim() noexcept {
        for (size_t i = 0; i < capacity_; ++i) {
            bool available = false;
            if (nodes_[i].claimed.compare_exchange_strong(available, true, std::memory_order_acquire)) return &nodes_[i];
        }
        return nullptr;
    }
    bool HasFree() const noexcept {
        for (size_t i = 0; i < capacity_; ++i) if (!nodes_[i].claimed.load(std::memory_order_acquire)) return true;
        return false;
    }
    // All Value initialization precedes this linearization point. Producers
    // copy head addresses but NEVER dereference head nodes. Consequently a
    // recycled head address is harmless: successful CAS links the current node
    // at that address. Static storage also rules out pointer lifetime UAF.
    void Publish(Node* node) noexcept {
        Node* head = head_.load(std::memory_order_relaxed);
        do { node->next = head; }
        while (!head_.compare_exchange_weak(head, node, std::memory_order_acq_rel, std::memory_order_relaxed));
    }
    Node* TakeBatch() noexcept {
        Node* batch = head_.exchange(nullptr, std::memory_order_acquire);
        Node* fifo = nullptr;
        while (batch) {
            Node* next = batch->next;
            batch->next = fifo; fifo = batch; batch = next;
        }
        return fifo;
    }
    bool Empty() const noexcept { return head_.load(std::memory_order_acquire) == nullptr; }
    bool IsLockFree() const noexcept {
        return head_.is_lock_free() && (capacity_ == 0 || nodes_[0].claimed.is_lock_free());
    }
    // Caller must finish Value cleanup, and the consumer must have saved next,
    // before Release. A descriptor is never released while any thread reads it.
    void Release(Node* node) noexcept { node->claimed.store(false, std::memory_order_release); }
private:
    const size_t capacity_;
    const std::unique_ptr<Node[]> nodes_;
    std::atomic<Node*> head_{nullptr};
};

} // namespace c10_npu::queue
