#include "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using c10_npu::queue::NoopWaitHooks;
using c10_npu::queue::OrderedMpscQueueCore;
using c10_npu::queue::QueueMode;
using namespace std::chrono_literals;

struct Task {
    int producer = -1;
    int sequence = -1;
};

[[noreturn]] void Fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message)
{
    if (!condition) {
        Fail(message);
    }
}

template <typename Predicate>
void RequireEventually(Predicate&& predicate, const std::string& message)
{
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            Fail(message);
        }
        std::this_thread::sleep_for(1ms);
    }
}

class Gate {
public:
    void Open()
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            open_ = true;
        }
        cv_.notify_all();
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return open_; });
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool open_ = false;
};

struct CountingWaitHooks {
    std::atomic<int>* before;
    std::atomic<int>* after;

    void BeforeWait() { before->fetch_add(1, std::memory_order_release); }
    void AfterWait() { after->fetch_add(1, std::memory_order_release); }
};

struct BlockingRestoreHooks {
    std::atomic<bool>* before_wait;
    std::atomic<bool>* restoring;
    Gate* allow_restore;

    void BeforeWait() { before_wait->store(true, std::memory_order_release); }
    void AfterWait()
    {
        restoring->store(true, std::memory_order_release);
        allow_restore->Wait();
    }
};

void TestExactlyOnceAndPerProducerOrder(int producer_count)
{
    constexpr int kTasksPerProducer = 300;
    OrderedMpscQueueCore core(17);
    std::vector<Task> slots(core.Capacity());
    std::vector<Task> observed;
    observed.reserve(static_cast<size_t>(producer_count * kTasksPerProducer));
    core.Start();

    std::thread consumer([&] {
        while (core.ConsumeOne([&](uint32_t slot) { observed.push_back(slots.at(slot)); })) {
        }
    });

    std::vector<std::thread> producers;
    for (int producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&, producer] {
            for (int sequence = 0; sequence < kTasksPerProducer; ++sequence) {
                const bool accepted = core.Enqueue(
                    [&](uint32_t slot) { slots.at(slot) = Task{producer, sequence}; }, NoopWaitHooks{});
                Require(accepted, "running queue rejected a producer task");
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }
    core.WaitDrained(NoopWaitHooks{});
    core.StopAndDrain();
    consumer.join();

    Require(observed.size() == static_cast<size_t>(producer_count * kTasksPerProducer),
        "task count is not exactly once");
    std::vector<int> next(static_cast<size_t>(producer_count), 0);
    for (const Task& task : observed) {
        Require(task.producer >= 0 && task.producer < producer_count, "invalid producer id");
        Require(task.sequence == next.at(static_cast<size_t>(task.producer)),
            "per-producer order changed or a task was duplicated");
        ++next.at(static_cast<size_t>(task.producer));
    }
    for (int count : next) {
        Require(count == kTasksPerProducer, "producer task was lost");
    }
}

void TestCapacityTwoWakesEightWaiters()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    for (int value = 0; value < 2; ++value) {
        Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = value; }, NoopWaitHooks{}),
            "failed to prefill queue");
    }

    std::atomic<int> entered{0};
    std::atomic<int> accepted{0};
    std::atomic<int> before_wait{0};
    std::atomic<int> after_wait{0};
    CountingWaitHooks hooks{&before_wait, &after_wait};
    std::vector<std::thread> waiters;
    for (int value = 2; value < 10; ++value) {
        waiters.emplace_back([&, value] {
            entered.fetch_add(1, std::memory_order_relaxed);
            if (core.Enqueue([&](uint32_t slot) { slots.at(slot) = value; }, hooks)) {
                accepted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    RequireEventually([&] { return entered.load(std::memory_order_relaxed) == 8; },
        "not all full-queue waiters started");
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire) >= 8; },
        "not all eight producers blocked on the full queue");

    std::vector<int> observed;
    while (observed.size() != 10) {
        Require(core.ConsumeOne([&](uint32_t slot) { observed.push_back(slots.at(slot)); }),
            "consumer stopped while producers were running");
    }
    for (auto& waiter : waiters) {
        waiter.join();
    }
    core.StopAndDrain(NoopWaitHooks{});
    Require(accepted.load(std::memory_order_relaxed) == 8, "a full-queue waiter was not awakened");
    Require(after_wait.load(std::memory_order_acquire) ==
            before_wait.load(std::memory_order_acquire),
        "a full-queue producer wait was not restored");
    std::sort(observed.begin(), observed.end());
    for (int value = 0; value < 10; ++value) {
        Require(observed.at(static_cast<size_t>(value)) == value, "waiter task lost or duplicated");
    }
}

void TestCopyExceptionDoesNotPublishSlot()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    bool threw = false;
    try {
        (void)core.Enqueue([](uint32_t) { throw std::runtime_error("copy failed"); }, NoopWaitHooks{});
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "copy failed";
    }
    Require(threw, "copy exception was not propagated");
    Require(core.Snapshot().size == 0, "copy exception occupied a queue slot");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 7; }, NoopWaitHooks{}),
        "queue unusable after copy exception");
    int observed = 0;
    Require(core.ConsumeOne([&](uint32_t slot) { observed = slots.at(slot); }), "replacement task missing");
    Require(observed == 7, "replacement task corrupted");
    core.StopAndDrain(NoopWaitHooks{});
}

void TestWaitHooksRunOnlyForActualBlockingWaits()
{
    OrderedMpscQueueCore core(1);
    std::vector<int> slots(core.Capacity(), -1);
    std::atomic<int> before{0};
    std::atomic<int> after{0};
    CountingWaitHooks hooks{&before, &after};
    core.Start();

    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 1; }, hooks), "nonblocking enqueue failed");
    Require(before.load(std::memory_order_acquire) == 0, "enqueue hook ran without blocking");
    Require(core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }, hooks),
        "nonblocking consume failed");
    Require(before.load(std::memory_order_acquire) == 0, "consumer hook ran without blocking");

    std::thread blocked_consumer([&] {
        Require(core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }, hooks),
            "blocked consumer did not consume");
    });
    RequireEventually([&] { return before.load(std::memory_order_acquire) == 1; },
        "consumer did not enter an actual wait");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 2; }, hooks), "wake enqueue failed");
    blocked_consumer.join();
    Require(after.load(std::memory_order_acquire) == 1, "consumer wait hook was not restored");

    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 3; }, hooks), "full prefill failed");
    std::thread blocked_producer([&] {
        Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 4; }, hooks),
            "blocked producer was not admitted");
    });
    RequireEventually([&] { return before.load(std::memory_order_acquire) == 2; },
        "producer did not enter an actual wait");
    Require(core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }, hooks), "full dequeue failed");
    blocked_producer.join();
    Require(after.load(std::memory_order_acquire) == 2, "producer wait hook was not restored");
    Require(core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }, hooks), "final dequeue failed");
    core.StopAndDrain(hooks);
}

void TestWaitHookRestoresOutsideQueueMutex()
{
    OrderedMpscQueueCore core(1);
    std::vector<int> slots(core.Capacity(), -1);
    std::atomic<bool> before_wait{false};
    std::atomic<bool> restoring{false};
    std::atomic<bool> inspected{false};
    Gate allow_restore;
    BlockingRestoreHooks hooks{&before_wait, &restoring, &allow_restore};
    core.Start();

    std::thread consumer([&] {
        (void)core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }, hooks);
    });
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire); },
        "consumer did not begin waiting");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 1; }, NoopWaitHooks{}),
        "restore test enqueue failed");
    RequireEventually([&] { return restoring.load(std::memory_order_acquire); },
        "wait hook did not begin restoration");

    std::thread inspector([&] {
        (void)core.Snapshot();
        inspected.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + 250ms;
    while (!inspected.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    const bool inspected_during_restore = inspected.load(std::memory_order_acquire);
    allow_restore.Open();
    inspector.join();
    consumer.join();
    core.StopAndDrain(NoopWaitHooks{});
    Require(inspected_during_restore, "wait hook restored while holding the queue mutex");
}

void TestConsumerExceptionBroadcastsSameError()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 1; }, NoopWaitHooks{}), "prefill failed");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 2; }, NoopWaitHooks{}), "prefill failed");

    std::atomic<int> before_wait{0};
    std::atomic<int> after_wait{0};
    CountingWaitHooks hooks{&before_wait, &after_wait};
    std::mutex error_mu;
    std::vector<std::string> errors;
    std::vector<std::thread> producers;
    for (int i = 0; i < 4; ++i) {
        producers.emplace_back([&, i] {
            try {
                (void)core.Enqueue([&, i](uint32_t slot) { slots.at(slot) = i + 10; }, hooks);
                std::lock_guard<std::mutex> lock(error_mu);
                errors.emplace_back("no error");
            } catch (const std::runtime_error& error) {
                std::lock_guard<std::mutex> lock(error_mu);
                errors.emplace_back(error.what());
            }
        });
    }
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire) >= 4; },
        "all four producers did not block before the consumer failed");
    bool consumer_threw = false;
    try {
        (void)core.ConsumeOne([](uint32_t) { throw std::runtime_error("consumer failed once"); });
    } catch (const std::runtime_error& error) {
        consumer_threw = std::string(error.what()) == "consumer failed once";
    }
    Require(consumer_threw, "consumer exception was not propagated");
    for (auto& producer : producers) {
        producer.join();
    }
    Require(errors.size() == 4, "not all blocked producers woke on consumer error");
    for (const std::string& error : errors) {
        Require(error == "consumer failed once", "producer did not receive the shared consumer error");
    }
    Require(after_wait.load(std::memory_order_acquire) ==
            before_wait.load(std::memory_order_acquire),
        "a producer was not restored after the consumer failed");
    core.StopAndDiscard();
}

void TestFourDrainWaitersWake()
{
    OrderedMpscQueueCore core(8);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    for (int value = 0; value < 8; ++value) {
        Require(core.Enqueue([&, value](uint32_t slot) { slots.at(slot) = value; }, NoopWaitHooks{}),
            "drain prefill failed");
    }
    std::atomic<int> waiting{0};
    std::atomic<int> drained{0};
    std::vector<std::thread> waiters;
    for (int i = 0; i < 4; ++i) {
        waiters.emplace_back([&] {
            waiting.fetch_add(1, std::memory_order_relaxed);
            core.WaitDrained(NoopWaitHooks{});
            drained.fetch_add(1, std::memory_order_relaxed);
        });
    }
    RequireEventually([&] { return waiting.load(std::memory_order_relaxed) == 4; },
        "drain waiters did not start");
    std::thread consumer([&] {
        for (int i = 0; i < 8; ++i) {
            Require(core.ConsumeOne([&](uint32_t slot) { (void)slots.at(slot); }), "drain consume failed");
        }
    });
    consumer.join();
    for (auto& waiter : waiters) {
        waiter.join();
    }
    Require(drained.load(std::memory_order_relaxed) == 4, "not all drain waiters woke");
    core.StopAndDrain(NoopWaitHooks{});
}

void TestShutdownWakesFullWaiterAndWaitsForActiveConsumer()
{
    OrderedMpscQueueCore core(1);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 1; }, NoopWaitHooks{}), "prefill failed");

    Gate release_consumer;
    std::atomic<bool> consumer_active{false};
    std::thread consumer([&] {
        (void)core.ConsumeOne([&](uint32_t slot) {
            (void)slots.at(slot);
            consumer_active.store(true, std::memory_order_release);
            release_consumer.Wait();
        });
    });
    RequireEventually([&] { return consumer_active.load(std::memory_order_acquire); },
        "consumer did not acquire its lease");

    std::atomic<bool> producer_returned{false};
    std::atomic<bool> producer_accepted{true};
    std::atomic<int> before_wait{0};
    std::atomic<int> after_wait{0};
    CountingWaitHooks producer_hooks{&before_wait, &after_wait};
    std::thread producer([&] {
        producer_accepted.store(
            core.Enqueue([&](uint32_t slot) { slots.at(slot) = 2; }, producer_hooks),
            std::memory_order_release);
        producer_returned.store(true, std::memory_order_release);
    });
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire) >= 1; },
        "producer did not block on the full queue before shutdown");

    std::atomic<bool> shutdown_returned{false};
    std::thread shutdown([&] {
        core.StopAndDrain(NoopWaitHooks{});
        shutdown_returned.store(true, std::memory_order_release);
    });
    RequireEventually([&] { return producer_returned.load(std::memory_order_acquire); },
        "shutdown did not wake the active producer");
    Require(!producer_accepted.load(std::memory_order_acquire), "producer published after admission closed");
    Require(!shutdown_returned.load(std::memory_order_acquire), "shutdown returned before consumer release");
    release_consumer.Open();
    producer.join();
    consumer.join();
    shutdown.join();
    Require(shutdown_returned.load(std::memory_order_acquire), "shutdown never completed");
    Require(after_wait.load(std::memory_order_acquire) ==
            before_wait.load(std::memory_order_acquire),
        "full producer was not restored during shutdown");
}

void TestStopAndDiscardWakesWhenConsumerReleasesWithPendingItems()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 1; }, NoopWaitHooks{}),
        "first discard prefill failed");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 2; }, NoopWaitHooks{}),
        "second discard prefill failed");

    Gate release_consumer;
    std::atomic<bool> consumer_active{false};
    std::thread consumer([&] {
        Require(core.ConsumeOne([&](uint32_t slot) {
            (void)slots.at(slot);
            consumer_active.store(true, std::memory_order_release);
            release_consumer.Wait();
        }), "discard test consumer did not acquire a task");
    });
    RequireEventually([&] { return consumer_active.load(std::memory_order_acquire); },
        "discard test consumer never became active");

    std::vector<int> discarded;
    std::atomic<bool> shutdown_returned{false};
    std::atomic<int> before_wait{0};
    std::atomic<int> after_wait{0};
    CountingWaitHooks shutdown_hooks{&before_wait, &after_wait};
    std::thread shutdown([&] {
        core.StopAndDiscard(
            [&](uint32_t slot) { discarded.push_back(slots.at(slot)); },
            shutdown_hooks);
        shutdown_returned.store(true, std::memory_order_release);
    });
    RequireEventually([&] { return core.Snapshot().mode == QueueMode::Stopping; },
        "discard shutdown did not begin");
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire) >= 1; },
        "StopAndDiscard did not run wait hooks for the active consumer");
    release_consumer.Open();
    consumer.join();

    const auto deadline = std::chrono::steady_clock::now() + 250ms;
    while (!shutdown_returned.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    const bool returned_before_rescue = shutdown_returned.load(std::memory_order_acquire);
    if (!returned_before_rescue) {
        core.RequestStop();
    }
    shutdown.join();
    Require(returned_before_rescue,
        "StopAndDiscard was not notified when a consumer released one of several queued items");
    Require(discarded == std::vector<int>{2}, "StopAndDiscard did not discard the pending item");
    Require(after_wait.load(std::memory_order_acquire) ==
            before_wait.load(std::memory_order_acquire),
        "StopAndDiscard did not restore its wait hook");
}

void TestInspectHeadExcludesConsumerOwnedSlot()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 11; }, NoopWaitHooks{}),
        "first inspect prefill failed");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 22; }, NoopWaitHooks{}),
        "second inspect prefill failed");

    Gate release_consumer;
    std::atomic<bool> consumer_active{false};
    std::thread consumer([&] {
        Require(core.ConsumeOne([&](uint32_t slot) {
            slots.at(slot) = -999;
            consumer_active.store(true, std::memory_order_release);
            release_consumer.Wait();
        }), "inspect test consumer did not acquire a task");
    });
    RequireEventually([&] { return consumer_active.load(std::memory_order_acquire); },
        "inspect test consumer never became active");

    int inspected = -1;
    const bool exposed_consumer_slot =
        core.InspectHead([&](uint32_t slot) { inspected = slots.at(slot); });

    release_consumer.Open();
    consumer.join();
    Require(!exposed_consumer_slot, "InspectHead exposed a consumer-owned queue slot");
    const bool found_pending =
        core.InspectHead([&](uint32_t slot) { inspected = slots.at(slot); });
    Require(found_pending && inspected == 22, "InspectHead did not advance to the pending slot");
    core.StopAndDiscard();
}

void TestDiscardExceptionPreservesHeadAndAllowsNoopCleanup()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 31; }, NoopWaitHooks{}),
        "first discard-error prefill failed");
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 32; }, NoopWaitHooks{}),
        "second discard-error prefill failed");

    int discard_calls = 0;
    bool threw = false;
    try {
        core.StopAndDiscard([&](uint32_t) {
            ++discard_calls;
            throw std::runtime_error("discard failed once");
        });
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "discard failed once";
    }
    Require(threw, "discard exception was not propagated");

    const auto failed = core.Snapshot();
    Require(failed.mode == QueueMode::Failed && failed.failed,
        "discard exception did not put the queue in the failed state");
    Require(!failed.consumer_active, "discard exception left the consumer lease active");
    Require(failed.size == 2, "discard exception popped the failing head slot");
    int head_value = -1;
    Require(core.InspectHead([&](uint32_t slot) { head_value = slots.at(slot); }),
        "discard exception hid the preserved head slot");
    Require(head_value == 31, "discard exception advanced the queue head");

    core.StopAndDiscard();
    const auto stopped = core.Snapshot();
    Require(stopped.mode == QueueMode::Stopped && stopped.size == 0 && !stopped.consumer_active,
        "no-op cleanup did not discard failed queue metadata");
    Require(discard_calls == 1, "no-op cleanup retried the failing discard callback");

    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 33; }, NoopWaitHooks{}),
        "queue did not restart after no-op cleanup");
    int observed = -1;
    Require(core.ConsumeOne([&](uint32_t slot) { observed = slots.at(slot); }),
        "restarted queue did not consume its new task");
    Require(observed == 33, "restarted queue exposed an old failed-generation task");
    core.StopAndDrain(NoopWaitHooks{});
}

void TestCustomDiscardStopsOnConcurrentConsumerFailure()
{
    struct IdentifiedFailure final : std::runtime_error {
        IdentifiedFailure(const char* message, const void* identity_value)
            : std::runtime_error(message), identity(identity_value)
        {
        }

        const void* identity;
    };

    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    int failure_identity = 0;
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 41; }, NoopWaitHooks{}),
        "concurrent-failure prefill failed");

    Gate release_consumer;
    std::atomic<bool> consumer_active{false};
    const void* consumer_failure_identity = nullptr;
    std::string consumer_failure_message;
    std::thread consumer([&] {
        try {
            (void)core.ConsumeOne([&](uint32_t slot) {
                (void)slots.at(slot);
                consumer_active.store(true, std::memory_order_release);
                release_consumer.Wait();
                throw IdentifiedFailure("concurrent consumer failed once", &failure_identity);
            });
        } catch (const IdentifiedFailure& error) {
            consumer_failure_identity = error.identity;
            consumer_failure_message = error.what();
        } catch (...) {
            consumer_failure_message = "unexpected consumer exception type";
        }
    });
    RequireEventually([&] { return consumer_active.load(std::memory_order_acquire); },
        "concurrent-failure consumer never became active");

    std::atomic<int> discard_calls{0};
    std::atomic<int> before_wait{0};
    std::atomic<int> after_wait{0};
    CountingWaitHooks shutdown_hooks{&before_wait, &after_wait};
    const void* shutdown_failure_identity = nullptr;
    std::string shutdown_failure_message;
    std::thread shutdown([&] {
        try {
            core.StopAndDiscard(
                [&](uint32_t) { discard_calls.fetch_add(1, std::memory_order_relaxed); },
                shutdown_hooks);
        } catch (const IdentifiedFailure& error) {
            shutdown_failure_identity = error.identity;
            shutdown_failure_message = error.what();
        } catch (...) {
            shutdown_failure_message = "unexpected shutdown exception type";
        }
    });
    RequireEventually([&] { return core.Snapshot().mode == QueueMode::Stopping; },
        "concurrent-failure shutdown did not begin");
    RequireEventually([&] { return before_wait.load(std::memory_order_acquire) >= 1; },
        "custom discard did not wait for the active consumer");

    release_consumer.Open();
    consumer.join();
    shutdown.join();

    Require(consumer_failure_identity == &failure_identity,
        "consumer failure identity was not propagated");
    Require(shutdown_failure_identity == consumer_failure_identity,
        "custom discard observed a different failure identity");
    Require(consumer_failure_message == "concurrent consumer failed once" &&
            shutdown_failure_message == consumer_failure_message,
        "custom discard observed a different failure message");
    Require(discard_calls.load(std::memory_order_acquire) == 0,
        "custom discard ran against a consumer-failed head");
    Require(after_wait.load(std::memory_order_acquire) ==
            before_wait.load(std::memory_order_acquire),
        "custom discard did not restore its wait hook after failure");
    const auto failed = core.Snapshot();
    Require(failed.mode == QueueMode::Failed && failed.failed && failed.size == 1,
        "concurrent consumer failure did not preserve the failed head");

    core.StopAndDiscardMetadata();
    const auto stopped = core.Snapshot();
    Require(stopped.mode == QueueMode::Stopped && stopped.size == 0 && !stopped.consumer_active,
        "metadata-only cleanup did not stop the failed queue");
}

void TestStopStartDoesNotExecuteOldTask()
{
    OrderedMpscQueueCore core(2);
    std::vector<int> slots(core.Capacity(), -1);
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 11; }, NoopWaitHooks{}), "old enqueue failed");
    core.StopAndDiscard();
    core.Start();
    Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = 22; }, NoopWaitHooks{}), "new enqueue failed");
    std::vector<int> observed;
    Require(core.ConsumeOne([&](uint32_t slot) { observed.push_back(slots.at(slot)); }), "new task missing");
    core.StopAndDrain(NoopWaitHooks{});
    Require(observed == std::vector<int>{22}, "old task survived STOP-to-RUN cycle");
}

void TestOneThousandLifecycleCycles()
{
    for (int cycle = 0; cycle < 1000; ++cycle) {
        OrderedMpscQueueCore core(2);
        std::vector<int> slots(core.Capacity(), -1);
        core.Start();
        Require(core.Enqueue([&](uint32_t slot) { slots.at(slot) = cycle; }, NoopWaitHooks{}),
            "lifecycle enqueue failed");
        int observed = -1;
        Require(core.ConsumeOne([&](uint32_t slot) { observed = slots.at(slot); }),
            "lifecycle consume failed");
        Require(observed == cycle, "lifecycle payload corrupted");
        core.StopAndDrain(NoopWaitHooks{});
    }
}

} // namespace

int main()
{
    try {
        for (int producers : {1, 2, 4, 8, 16}) {
            TestExactlyOnceAndPerProducerOrder(producers);
        }
        TestCapacityTwoWakesEightWaiters();
        TestCopyExceptionDoesNotPublishSlot();
        TestWaitHooksRunOnlyForActualBlockingWaits();
        TestWaitHookRestoresOutsideQueueMutex();
        TestConsumerExceptionBroadcastsSameError();
        TestFourDrainWaitersWake();
        TestShutdownWakesFullWaiterAndWaitsForActiveConsumer();
        TestStopAndDiscardWakesWhenConsumerReleasesWithPendingItems();
        TestInspectHeadExcludesConsumerOwnedSlot();
        TestDiscardExceptionPreservesHeadAndAllowsNoopCleanup();
        TestCustomDiscardStopsOnConcurrentConsumerFailure();
        TestStopStartDoesNotExecuteOldTask();
        TestOneThousandLifecycleCycles();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "all ordered MPSC queue tests passed\n";
    return 0;
}
