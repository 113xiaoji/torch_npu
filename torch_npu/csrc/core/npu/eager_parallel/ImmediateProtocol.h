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
#include <vector>

namespace c10_npu::eager_parallel {
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
    ImmediateProtocol(ImmediateEventBackend& backend, std::size_t capacity, std::size_t keys)
        : backend_(backend), capacity_(capacity), writes_(keys)
    {
        if (!capacity || keys < 4) throw std::invalid_argument("invalid immediate protocol capacity");
    }
    bool can_begin() const noexcept { return !poisoned_ && !inflight_ && calls_ < capacity_ && writes_.size() - keys_ >= 4; }
    bool active() const noexcept { return active_; }
    bool poisoned() const noexcept { return poisoned_; }
    std::size_t calls() const noexcept { return calls_; }
    std::uint64_t waits() const noexcept { return waits_; }
    std::uint64_t records() const noexcept { return records_; }
    std::uint64_t joins() const noexcept { return joins_; }

    unsigned begin(bool producer, const std::uintptr_t* inputs, std::size_t count)
    {
        if (!can_begin()) throw std::runtime_error("immediate group cannot admit a call");
        selected_ = producer ? 1u + static_cast<unsigned>(producer_turn_ % 2) : 3u;
        try {
            if (!active_) {
                active_ = true; // a failing record may already have submitted work
                backend_.record(0, 0); check_callback(); ++records_;
            }
            inflight_ = true;
            if (!entered_[selected_]) {
                backend_.wait(selected_, 0); check_callback(); ++waits_; entered_[selected_] = true;
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
            const auto event = calls_ + 1;
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
    void drain()
    {
        if (poisoned_ || inflight_) throw std::runtime_error("cannot normally drain a failed immediate group");
        if (!active_) return;
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
        entered_.fill(false); tail_.fill(0);
    }
    ImmediateEventBackend& backend_;
    const std::size_t capacity_;
    std::vector<Write> writes_;
    std::array<bool, 4> entered_{};
    std::array<std::size_t, 4> tail_{};
    std::size_t calls_ = 0, keys_ = 0, producer_turn_ = 0;
    unsigned selected_ = 0;
    bool active_ = false, inflight_ = false, poisoned_ = false;
    std::uint64_t waits_ = 0, records_ = 0, joins_ = 0;
};
} // namespace c10_npu::eager_parallel
