#pragma once

// A separate, deliberately narrower protocol than DependencyScheduler:
// single caller, synchronous host submission, pure out-of-place device calls,
// whole-allocation keys pinned until a group-wide device completion fence.
// Events and arrays are allocated before the first body. No state allocation
// occurs in begin/commit/drain. Mutation and unknown operations close the group.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <limits>

#include <vector>

namespace c10_npu::eager_parallel {
enum class EmbeddingLengthClass { Uncertain, Small, Large };
// Metadata only. Integer quotient avoids the threshold*bags overflow while
// preserving the exact comparison of ragged mean length to an integer limit.
inline EmbeddingLengthClass ClassifyEmbeddingLength(int64_t weight_rank,
    int64_t indices_rank, int64_t offsets_rank, int64_t indices, int64_t offsets,
    bool include_last_offset, uint64_t threshold) noexcept
{
    if (weight_rank!=2 || indices_rank!=1 || offsets_rank!=1 || indices<0 ||
        offsets<=static_cast<int64_t>(include_last_offset)) return EmbeddingLengthClass::Uncertain;
    const auto bags=static_cast<uint64_t>(offsets-static_cast<int64_t>(include_last_offset));
    return static_cast<uint64_t>(indices)/bags<threshold ? EmbeddingLengthClass::Small : EmbeddingLengthClass::Large;
}
struct StorageInterval { std::uintptr_t begin=0; std::size_t bytes=0; bool valid=false; };
inline StorageInterval MakeStorageInterval(std::uintptr_t begin, std::size_t bytes) noexcept
{
    return {begin,bytes,(!bytes || begin!=0) && bytes<=std::numeric_limits<std::uintptr_t>::max()-begin};
}
inline bool StorageIntervalsOverlap(StorageInterval a, StorageInterval b) noexcept
{
    if (!a.valid || !b.valid) return true; // uncertain ownership fences conservatively
    if (!a.bytes || !b.bytes) return false;
    return a.begin<b.begin+b.bytes && b.begin<a.begin+a.bytes;
}
class ImmediateEventBackend {
public:
    virtual ~ImmediateEventBackend() = default;
    virtual void record(std::size_t event, unsigned stream) = 0;
    virtual void wait(unsigned stream, std::size_t event) = 0;
    // Must be bounded and establish device completion, not host queue emptiness.
    virtual void synchronize(unsigned stream) = 0;
};

class ImmediateProtocol {
public:
    ImmediateProtocol(ImmediateEventBackend& backend, std::size_t capacity, std::size_t keys, bool selective=false)
        : backend_(backend), capacity_(capacity), selective_(selective), writes_(keys)
    {
        if (!capacity || keys < 4) throw std::invalid_argument("invalid immediate protocol capacity");
    }
    bool can_begin() const noexcept { return !poisoned_ && !inflight_ && calls_ < capacity_ && (!selective_ || event_cursor_+1<=2*capacity_) && writes_.size() - keys_ >= 4; }
    bool active() const noexcept { return active_; }
    bool poisoned() const noexcept { return poisoned_; }
    std::size_t calls() const noexcept { return calls_; }
    std::uint64_t waits() const noexcept { return waits_; }
    std::uint64_t records() const noexcept { return records_; }
    std::uint64_t joins() const noexcept { return joins_; }

    bool caller_dirty() const noexcept { return caller_dirty_; }
    std::uint64_t frontier_records() const noexcept { return frontier_records_; }
    std::uint64_t frontier_waits() const noexcept { return frontier_waits_; }
    void mark_caller_dirty() {
        if (!selective_ || poisoned_ || inflight_) throw std::runtime_error("invalid immediate caller frontier");
        caller_dirty_=true;
    }
    unsigned begin(bool producer, const std::uintptr_t* inputs, std::size_t count)
    {
        if (!can_begin()) throw std::runtime_error("immediate group cannot admit a call");
        selected_ = producer ? 1u + static_cast<unsigned>(producer_turn_ % 2) : 3u;
        try {
            if (!active_) {
                active_ = true; // a failing record may already have submitted work
                backend_.record(0, 0); check_callback(); ++records_;
                if (selective_) { ++frontier_records_; caller_dirty_=false; }
            } else if (selective_ && caller_dirty_) {
                frontier_=event_cursor_++;
                backend_.record(frontier_,0); check_callback(); ++records_; ++frontier_records_;
                caller_dirty_=false;
            }
            inflight_ = true;
            if (!entered_[selected_] || (selective_ && waited_frontier_[selected_]!=frontier_)) {
                backend_.wait(selected_,frontier_); check_callback(); ++waits_; entered_[selected_] = true;
                waited_frontier_[selected_]=frontier_;
                if (selective_) ++frontier_waits_;
            }
            std::array<std::size_t, 4> dependency{};
            for (std::size_t i = 0; i < count; ++i) {
                if (!inputs[i]) continue;
                for (std::size_t j = 0; j < keys_; ++j) {
                    const auto& write = writes_[j];
                    if (write.key == inputs[i] && write.stream != selected_ && write.event > dependency[write.stream])
                        dependency[write.stream] = write.event;
                }
            }
            for (unsigned s = 1; s <= 3; ++s) if (dependency[s]) {
                backend_.wait(selected_, dependency[s]); check_callback(); ++waits_;
            }
            if (producer) ++producer_turn_;
            return selected_;
        } catch (...) { poisoned_ = true; throw; }
    }

    void commit(const std::uintptr_t* outputs, std::size_t count)
    {
        if (!inflight_ || poisoned_ || count > 4 || writes_.size() - keys_ < count) {
            poisoned_ = true;
            throw std::runtime_error("invalid immediate completion");
        }
        try {
            // No handle is rerecorded while an earlier wait can still use it.
            const auto event = selective_ ? event_cursor_++ : calls_ + 1;
            backend_.record(event, selected_); check_callback(); ++records_;
            for (std::size_t i = 0; i < count; ++i) {
                if (!outputs[i]) continue;
                writes_[keys_++] = {outputs[i], event, selected_};
            }
            tail_[selected_] = event;
            ++calls_; inflight_ = false;
        } catch (...) { poisoned_ = true; throw; }
    }

    void poison() noexcept { poisoned_ = true; }
    void drain(bool synchronize_caller=false)
    {
        if (poisoned_ || inflight_) throw std::runtime_error("cannot normally drain a failed immediate group");
        if (!active_ && !(selective_ && caller_dirty_ && synchronize_caller)) return;
        try {
            for (unsigned s = 1; s <= 3; ++s) if (tail_[s]) {
                backend_.wait(0, tail_[s]); check_callback(); ++waits_; ++joins_;
            }
            backend_.synchronize(0); check_callback();
            reset();
        } catch (...) { poisoned_ = true; throw; }
    }
    bool recover() noexcept
    {
        // Every stream is checked independently: an exception after a device
        // wait or body submission may have left no usable completion event.
        bool complete = true;
        for (unsigned s = 0; s <= 3; ++s) {
            try { backend_.synchronize(s); } catch (...) { complete = false; }
        }
        if (complete) reset();
        return complete;
    }
private:
    struct Write { std::uintptr_t key = 0; std::size_t event = 0; unsigned stream = 0; };
    void check_callback() const {
        if (poisoned_) throw std::runtime_error("immediate backend callback violated admission");
    }
    void reset() noexcept
    {
        active_ = false; inflight_ = false; calls_ = keys_ = producer_turn_ = 0;
        entered_.fill(false); tail_.fill(0); waited_frontier_.fill(0);
        event_cursor_=1; frontier_=0; caller_dirty_=false;
    }
    ImmediateEventBackend& backend_;
    const std::size_t capacity_;
    const bool selective_;
    std::vector<Write> writes_;
    std::array<bool, 4> entered_{};
    std::array<std::size_t, 4> tail_{}, waited_frontier_{};
    std::size_t event_cursor_=1,frontier_=0;
    bool caller_dirty_=false;
    std::size_t calls_ = 0, keys_ = 0, producer_turn_ = 0;
    unsigned selected_ = 0;
    bool active_ = false, inflight_ = false, poisoned_ = false;
    std::uint64_t waits_ = 0, records_ = 0, joins_ = 0, frontier_records_=0, frontier_waits_=0;
};
} // namespace c10_npu::eager_parallel
