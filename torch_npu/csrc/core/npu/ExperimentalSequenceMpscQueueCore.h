#pragma once

// Experimental bounded MPSC sequence ring for task-owned TorchNPU validation.
// Public NPUQueue/QueueParas interfaces are unchanged.
#include "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace c10_npu::queue {

class ExperimentalSequenceMpscQueueCore final {
 public:
  explicit ExperimentalSequenceMpscQueueCore(size_t capacity) : capacity_(capacity), slots_(new Slot[capacity]) {
    if (capacity == 0 || capacity > static_cast<size_t>(UINT32_MAX)) {
      throw std::invalid_argument("sequence MPSC queue capacity out of range");
    }
    for (size_t i = 0; i < capacity_; ++i) slots_[i].sequence.store(i, std::memory_order_relaxed);
  }
  ~ExperimentalSequenceMpscQueueCore() { StopAndDiscardMetadata(); }
  ExperimentalSequenceMpscQueueCore(const ExperimentalSequenceMpscQueueCore&) = delete;
  ExperimentalSequenceMpscQueueCore& operator=(const ExperimentalSequenceMpscQueueCore&) = delete;

  size_t Capacity() const noexcept { return capacity_; }

  void Start() {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (mode_.load() != QueueMode::Stopped || active_producers_.load() != 0 || consumer_active_.load()) {
      throw std::logic_error("sequence queue can only start from stopped state");
    }
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].sequence.store(i, std::memory_order_relaxed);
      slots_[i].aborted.store(false, std::memory_order_relaxed);
    }
    error_ = nullptr;
    mode_.store(QueueMode::Running, std::memory_order_release);
  }

  template <typename Copy, typename WaitHooks>
  bool Enqueue(Copy&& copy, WaitHooks&& hooks) {
    active_producers_.fetch_add(1, std::memory_order_acq_rel);
    auto leave = [this]() { active_producers_.fetch_sub(1, std::memory_order_acq_rel); };
    if (mode_.load(std::memory_order_acquire) != QueueMode::Running) {
      leave();
      return false;
    }
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    bool waiting = false;
    try {
      for (;;) {
        if (mode_.load(std::memory_order_acquire) != QueueMode::Running) {
          if (waiting) hooks.AfterWait();
          leave();
          return false;
        }
        Slot& slot = slots_[pos % capacity_];
        size_t sequence = slot.sequence.load(std::memory_order_acquire);
        intptr_t difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos);
        if (difference == 0) {
          if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
        } else if (difference < 0) {
          if (!waiting) { hooks.BeforeWait(); waiting = true; }
          std::this_thread::yield();
          pos = enqueue_pos_.load(std::memory_order_relaxed);
        } else {
          pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
      }
      if (waiting) { hooks.AfterWait(); waiting = false; }
      Slot& slot = slots_[pos % capacity_];
      slot.aborted.store(false, std::memory_order_relaxed);
      try {
        std::forward<Copy>(copy)(static_cast<uint32_t>(pos % capacity_));
      } catch (...) {
        set_failed(std::current_exception());
        slot.aborted.store(true, std::memory_order_relaxed);
        slot.sequence.store(pos + 1, std::memory_order_release);
        leave();
        throw;
      }
      slot.sequence.store(pos + 1, std::memory_order_release);
      leave();
      return true;
    } catch (...) {
      if (waiting) hooks.AfterWait();
      throw;
    }
  }

  template <typename Consume>
  bool ConsumeOne(Consume&& consume) { NoopWaitHooks hooks; return ConsumeOne(std::forward<Consume>(consume), hooks); }

  template <typename Consume, typename WaitHooks>
  bool ConsumeOne(Consume&& consume, WaitHooks&& hooks) {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    bool waiting = false;
    for (;;) {
      Slot& slot = slots_[pos % capacity_];
      size_t sequence = slot.sequence.load(std::memory_order_acquire);
      intptr_t difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos + 1);
      if (difference == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (difference < 0) {
        QueueMode mode = mode_.load(std::memory_order_acquire);
        if (mode != QueueMode::Running && active_producers_.load(std::memory_order_acquire) == 0) {
          if (waiting) hooks.AfterWait();
          return false;
        }
        if (!waiting) { hooks.BeforeWait(); waiting = true; }
        std::this_thread::yield();
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      } else {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
    }
    if (waiting) hooks.AfterWait();
    consumer_active_.store(true, std::memory_order_release);
    Slot& slot = slots_[pos % capacity_];
    try {
      if (slot.aborted.load(std::memory_order_acquire)) {
        slot.sequence.store(pos + capacity_, std::memory_order_release);
        consumer_active_.store(false, std::memory_order_release);
        RethrowIfFailed();
        throw std::runtime_error("sequence queue aborted slot");
      }
      std::forward<Consume>(consume)(static_cast<uint32_t>(pos % capacity_));
    } catch (...) {
      set_failed(std::current_exception());
      slot.sequence.store(pos + capacity_, std::memory_order_release);
      consumer_active_.store(false, std::memory_order_release);
      throw;
    }
    slot.sequence.store(pos + capacity_, std::memory_order_release);
    consumer_active_.store(false, std::memory_order_release);
    return true;
  }

  template <typename WaitHooks>
  void WaitDrained(WaitHooks&& hooks) {
    bool waiting = false;
    while (dequeue_pos_.load(std::memory_order_acquire) != enqueue_pos_.load(std::memory_order_acquire) ||
           active_producers_.load(std::memory_order_acquire) != 0 || consumer_active_.load(std::memory_order_acquire)) {
      if (!waiting) { hooks.BeforeWait(); waiting = true; }
      std::this_thread::yield();
    }
    if (waiting) hooks.AfterWait();
    RethrowIfFailed();
  }

  void RequestStop() {
    QueueMode expected = QueueMode::Running;
    mode_.compare_exchange_strong(expected, QueueMode::Stopping, std::memory_order_acq_rel);
    mode_.store(QueueMode::Stopping, std::memory_order_release);
  }

  void StopAndDrain() { NoopWaitHooks hooks; StopAndDrain(hooks); }
  template <typename WaitHooks> void StopAndDrain(WaitHooks&& hooks) {
    RequestStop(); WaitDrained(hooks); mode_.store(QueueMode::Stopped, std::memory_order_release);
  }
  void StopAndDiscard() { StopAndDiscardMetadata(); }
  template <typename WaitHooks> void StopAndDiscardMetadata(WaitHooks&& hooks) {
    mode_.store(QueueMode::Stopping, std::memory_order_release);
    bool waiting = false;
    while (active_producers_.load(std::memory_order_acquire) != 0 || consumer_active_.load(std::memory_order_acquire)) {
      if (!waiting) { hooks.BeforeWait(); waiting = true; }
      std::this_thread::yield();
    }
    if (waiting) hooks.AfterWait();
    dequeue_pos_.store(enqueue_pos_.load(std::memory_order_acquire), std::memory_order_release);
    mode_.store(QueueMode::Stopped, std::memory_order_release);
  }
  void StopAndDiscardMetadata() {
    mode_.store(QueueMode::Stopping, std::memory_order_release);
    while (active_producers_.load(std::memory_order_acquire) != 0 || consumer_active_.load(std::memory_order_acquire)) std::this_thread::yield();
    dequeue_pos_.store(enqueue_pos_.load(std::memory_order_acquire), std::memory_order_release);
    mode_.store(QueueMode::Stopped, std::memory_order_release);
  }
  template <typename Discard, typename WaitHooks> void StopAndDiscard(Discard&& discard, WaitHooks&& hooks) {
    RequestStop();
    while (active_producers_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    while (dequeue_pos_.load(std::memory_order_acquire) != enqueue_pos_.load(std::memory_order_acquire)) {
      ConsumeOne(std::forward<Discard>(discard), hooks);
    }
    mode_.store(QueueMode::Stopped, std::memory_order_release);
  }
  template <typename Discard> void StopAndDiscard(Discard&& discard) {
    NoopWaitHooks hooks;
    StopAndDiscard(std::forward<Discard>(discard), hooks);
  }
  void Fail(std::exception_ptr error) { set_failed(error); }
  void RethrowIfFailed() const {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (error_) std::rethrow_exception(error_);
  }
  QueueSnapshot Snapshot() const {
    size_t enq = enqueue_pos_.load(std::memory_order_acquire), deq = dequeue_pos_.load(std::memory_order_acquire);
    size_t size = enq >= deq ? std::min(enq - deq, capacity_) : 0;
    return QueueSnapshot{capacity_, size, active_producers_.load(), consumer_active_.load(), mode_.load(), has_error()};
  }
  template <typename Inspect> bool InspectHead(Inspect&& inspect) const {
    if (consumer_active_.load(std::memory_order_acquire)) return false;
    size_t pos = dequeue_pos_.load(std::memory_order_acquire), enq = enqueue_pos_.load(std::memory_order_acquire);
    if (pos == enq) return false;
    if (slots_[pos % capacity_].sequence.load(std::memory_order_acquire) != pos + 1) return false;
    std::forward<Inspect>(inspect)(static_cast<uint32_t>(pos % capacity_));
    return true;
  }

 private:
  struct alignas(64) Slot { std::atomic<size_t> sequence{0}; std::atomic<bool> aborted{false}; };
  void set_failed(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!error_) error_ = error ? error : std::make_exception_ptr(std::runtime_error("sequence queue failed"));
    mode_.store(QueueMode::Failed, std::memory_order_release);
  }
  bool has_error() const { std::lock_guard<std::mutex> lock(state_mu_); return error_ != nullptr; }
  size_t capacity_;
  std::unique_ptr<Slot[]> slots_;
  std::atomic<size_t> enqueue_pos_{0}, dequeue_pos_{0}, active_producers_{0};
  std::atomic<bool> consumer_active_{false};
  std::atomic<QueueMode> mode_{QueueMode::Stopped};
  mutable std::mutex state_mu_;
  std::exception_ptr error_;
};

} // namespace c10_npu::queue
