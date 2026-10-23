#include "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using c10_npu::queue::NoopWaitHooks;
using c10_npu::queue::OrderedMpscQueueCore;
using c10_npu::queue::QueueMode;
using namespace std::chrono_literals;

template <class Predicate> bool Eventually(Predicate predicate, std::chrono::milliseconds budget)
{
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

[[noreturn]] void EmergencyExit(const char* message)
{
    std::cerr << "FAILED bounded cleanup: " << message << std::endl;
    // This test executable owns all its threads; exit prevents a failed test
    // from leaving detached native threads or blocking indefinitely in join.
    std::_Exit(72);
}

struct CountingHooks {
    std::atomic<std::uint64_t>* before;
    std::atomic<std::uint64_t>* after;
    void BeforeWait() { before->fetch_add(1, std::memory_order_release); }
    void AfterWait() { after->fetch_add(1, std::memory_order_release); }
};

int WakeContract()
{
    constexpr std::size_t kWaiters = 8;
    constexpr std::size_t kItems = 64;
    constexpr std::uint64_t kSpuriousAllowance = kWaiters * 4;
    OrderedMpscQueueCore queue(128);
    std::vector<std::size_t> slots(queue.Capacity());
    queue.Start();
    for (std::size_t item = 0; item < kItems; ++item) {
        if (!queue.Enqueue([&](std::uint32_t slot) { slots[slot] = item; }, NoopWaitHooks{})) {
            return 70;
        }
    }
    std::array<std::atomic<std::uint64_t>, kWaiters> before{};
    std::array<std::atomic<std::uint64_t>, kWaiters> after{};
    std::atomic<std::size_t> completed{0};
    std::atomic<std::size_t> errors{0};
    std::vector<std::thread> waiters;
    for (std::size_t index = 0; index < kWaiters; ++index) {
        waiters.emplace_back([&, index] {
            try {
                CountingHooks hooks{&before[index], &after[index]};
                queue.WaitDrained(hooks);
                ++completed;
            } catch (...) {
                ++errors;
            }
        });
    }
    if (!Eventually([&] {
            for (const auto& count : before) {
                if (count.load(std::memory_order_acquire) == 0) {
                    return false;
                }
            }
            return true;
        }, 2s)) {
        EmergencyExit("not all drain waiters entered their first wait");
    }
    bool fifo = true;
    for (std::size_t item = 0; item < kItems - 1; ++item) {
        fifo &= queue.ConsumeOne([&](std::uint32_t slot) { fifo &= slots[slot] == item; });
        std::this_thread::sleep_for(2ms);
    }
    const auto nonempty = queue.Snapshot();
    std::uint64_t nonempty_wait_entries = 0;
    for (const auto& count : before) {
        nonempty_wait_entries += count.load(std::memory_order_acquire);
    }
    const auto early_returns = completed.load();
    fifo &= queue.ConsumeOne([&](std::uint32_t slot) { fifo &= slots[slot] == kItems - 1; });
    if (!Eventually([&] { return completed.load() + errors.load() == kWaiters; }, 2s)) {
        queue.RequestStop();
        if (!Eventually([&] { return completed.load() + errors.load() == kWaiters; }, 2s)) {
            EmergencyExit("drain waiter did not finish after final consume/rescue");
        }
    }
    for (auto& waiter : waiters) {
        waiter.join();
    }
    queue.StopAndDrain();
    const auto repeated = nonempty_wait_entries >= kWaiters ? nonempty_wait_entries - kWaiters : 0;
    const bool infrastructure_ok = fifo && nonempty.size == 1 && !nonempty.consumer_active &&
                                   early_returns == 0 && completed == kWaiters && errors == 0;
    const bool contract_passed = infrastructure_ok && repeated <= kSpuriousAllowance;
    std::cout << "{\"test\":\"running_nonempty_drain_wake_contract\",\"waiters\":" << kWaiters
              << ",\"items\":" << kItems << ",\"nonempty_consume_steps\":" << kItems - 1
              << ",\"step_interval_ms\":2,\"nonempty_wait_entries\":" << nonempty_wait_entries
              << ",\"repeated_wait_entries_while_nonempty\":" << repeated
              << ",\"allowed_spurious_rewaits\":" << kSpuriousAllowance
              << ",\"waiters_completed_after_final_consume\":" << completed
              << ",\"early_drain_returns\":" << early_returns
              << ",\"infrastructure_ok\":" << (infrastructure_ok ? "true" : "false")
              << ",\"contract_passed\":" << (contract_passed ? "true" : "false") << "}\n";
    return infrastructure_ok ? (contract_passed ? 0 : 1) : 71;
}

int DiscardLifecycle(bool metadata_only)
{
    OrderedMpscQueueCore queue(8);
    std::vector<int> slots(queue.Capacity(), -1);
    queue.Start();
    for (int item = 0; item < 3; ++item) {
        if (!queue.Enqueue([&](std::uint32_t slot) { slots[slot] = item; }, NoopWaitHooks{})) {
            return 70;
        }
    }
    std::atomic<bool> consumer_entered{false};
    std::atomic<bool> release_consumer{false};
    std::atomic<bool> consumer_ok{false};
    std::atomic<std::uint64_t> before{0}, after{0};
    std::vector<int> discarded;
    std::thread consumer([&] {
        try {
            const bool consumed = queue.ConsumeOne([&](std::uint32_t slot) {
                consumer_ok = slots[slot] == 0;
                consumer_entered.store(true, std::memory_order_release);
                while (!release_consumer.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(100us);
                }
            });
            consumer_ok = consumer_ok.load() && consumed;
        } catch (...) {
            consumer_ok = false;
        }
    });
    if (!Eventually([&] { return consumer_entered.load(std::memory_order_acquire); }, 2s)) {
        EmergencyExit("lifecycle consumer never claimed the head");
    }
    std::promise<int> stopped;
    auto stopped_future = stopped.get_future();
    std::thread shutdown([&] {
        try {
            CountingHooks hooks{&before, &after};
            if (metadata_only) {
                queue.StopAndDiscardMetadata(hooks);
            } else {
                queue.StopAndDiscard([&](std::uint32_t slot) { discarded.push_back(slots[slot]); }, hooks);
            }
            stopped.set_value(0);
        } catch (...) {
            stopped.set_value(1);
        }
    });
    if (!Eventually([&] { return before.load(std::memory_order_acquire) > 0; }, 2s)) {
        EmergencyExit("discard did not wait for the active consumer");
    }
    const auto pending = queue.Snapshot();
    release_consumer.store(true, std::memory_order_release);
    const bool needed_rescue = stopped_future.wait_for(500ms) != std::future_status::ready;
    if (needed_rescue) {
        queue.RequestStop();  // Broadcast rescue, then fail after clean joins.
        if (stopped_future.wait_for(2s) != std::future_status::ready) {
            EmergencyExit("discard still blocked after RequestStop broadcast");
        }
    }
    const int stop_error = stopped_future.get();
    consumer.join();
    shutdown.join();
    const auto final = queue.Snapshot();
    const bool expected_discards = metadata_only ? discarded.empty() : discarded == std::vector<int>{1, 2};
    const bool passed = !needed_rescue && stop_error == 0 && consumer_ok.load() && expected_discards &&
                        pending.size == 3 && pending.consumer_active && pending.mode == QueueMode::Stopping &&
                        final.size == 0 && !final.consumer_active && final.mode == QueueMode::Stopped &&
                        before.load() == after.load();
    std::cout << "{\"test\":\"" << (metadata_only ? "StopAndDiscardMetadata" : "StopAndDiscard")
              << "_consumer_idle_with_pending_items\",\"queued_before_callback_release\":" << pending.size
              << ",\"consumer_steps\":1,\"remaining_items_after_single_consume\":2"
              << ",\"discard_callbacks\":" << discarded.size()
              << ",\"wait_before\":" << before << ",\"wait_after\":" << after
              << ",\"rescue_request_stop\":" << (needed_rescue ? "true" : "false")
              << ",\"bounded_cleanup_complete\":true,\"passed\":" << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 64;
    }
    const std::string mode = argv[1];
    if (mode == "wake-contract") {
        return WakeContract();
    }
    if (mode == "metadata-lifecycle") {
        return DiscardLifecycle(true);
    }
    if (mode == "discard-lifecycle") {
        return DiscardLifecycle(false);
    }
    return 64;
}
