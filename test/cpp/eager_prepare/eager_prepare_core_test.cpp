#include "torch_npu/csrc/core/npu/EagerPrepareQueueCore.h"

#include <atomic>
#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Core = c10_npu::queue::EagerPrepareQueueCore;
using namespace std::chrono_literals;

#define CHECK(x) do { if (!(x)) throw std::runtime_error("CHECK failed: " #x); } while (false)

struct Event {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    void Set() { std::lock_guard<std::mutex> lock(mutex); ready = true; cv.notify_all(); }
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, 5s, [&] { return ready; })) throw std::runtime_error("event timeout");
    }
};

struct TestJob final : Core::Job {
    std::function<void()> prepare = [] {};
    std::function<void()> execute = [] {};
    std::function<void()> cancel = [] {};
    std::function<void()> destroy = [] {};
    std::atomic<size_t> bytes{8};
    std::atomic<int> prepared{0}, executed{0}, cancelled{0};
    std::atomic<bool> cleanup_failed{false};
    ~TestJob() override { destroy(); }
    void Prepare() override { ++prepared; prepare(); }
    void Execute() override { ++executed; execute(); }
    void Cancel() noexcept override {
        ++cancelled;
        try { cancel(); if (!cleanup_failed) bytes = 0; } catch (...) { cleanup_failed = true; }
    }
    size_t RetainedBytes() const noexcept override { return bytes.load(); }
    bool CleanupFailed() const noexcept override { return cleanup_failed.load(); }
};

std::exception_ptr Error(const char* message) { return std::make_exception_ptr(std::runtime_error(message)); }

template <typename Fn> void Throws(Fn fn, const std::string& expected) {
    try { fn(); } catch (const std::exception& error) {
        CHECK(std::string(error.what()).find(expected) != std::string::npos); return;
    }
    throw std::runtime_error("expected exception: " + expected);
}

// This is the pre-existing FIFO: both legacy and prepare placeholders use it.
struct Fifo {
    std::vector<std::function<void()>> commands;
    Core::PublishReceipt Publish(Core& core, const Core::TicketPtr& ticket) {
        commands.emplace_back([&core, ticket] { core.Consume(ticket); });
        return {Core::PublishState::Committed, nullptr};
    }
    auto Publisher(Core& core) { return [this, &core](const Core::TicketPtr& ticket) { return Publish(core, ticket); }; }
    void Drain() { for (auto& command : commands) command(); commands.clear(); }
};

void ReversePrepareAndMixedLegacyOrder() {
    Core core({4, 128, 2}); Fifo fifo;
    Event first_started, second_done, release_first;
    std::vector<int> order;
    auto first = std::make_shared<TestJob>(); auto second = std::make_shared<TestJob>();
    first->prepare = [&] { first_started.Set(); release_first.Wait(); };
    first->execute = [&] { order.push_back(1); };
    second->prepare = [&] { second_done.Set(); };
    second->execute = [&] { order.push_back(3); };
    auto a = core.Submit(first, fifo.Publisher(core));
    fifo.commands.emplace_back([&] { order.push_back(2); });
    auto b = core.Submit(second, fifo.Publisher(core));
    first_started.Wait(); second_done.Wait();
    CHECK(first->executed == 0 && second->executed == 0);
    release_first.Set(); fifo.Drain();
    CHECK((order == std::vector<int>{1, 2, 3}));
    CHECK(core.Snapshot().slots_in_use == 2);
    core.Retire(a); core.Retire(b); core.Shutdown();
}

struct Hooks {
    Core* core = nullptr;
    Event* before = nullptr;
    std::atomic<int> detached{0}, attached{0};
    void BeforeWait() noexcept { ++detached; if (before) before->Set(); }
    void AfterWait() { if (core) (void)core->Snapshot(); ++attached; }
};

void K1BackpressureUntilRetirement() {
    Core core({1, 64, 1}); Fifo fifo;
    auto a = core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core));
    fifo.Drain();
    CHECK(core.Snapshot().slots_in_use == 1);
    Event waiting; Hooks hooks; hooks.core = &core; hooks.before = &waiting;
    auto next = std::async(std::launch::async, [&] { return core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core), hooks); });
    waiting.Wait(); CHECK(next.wait_for(0ms) == std::future_status::timeout);
    core.Retire(a);
    auto b = next.get(); fifo.Drain(); core.Retire(b); core.Shutdown();
    CHECK(hooks.detached == hooks.attached && hooks.attached > 0);
}

void CloseWakesBlockedSubmit() {
    Core core({1, 64, 1}); Fifo fifo;
    auto a = core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core));
    Event waiting; Hooks hooks; hooks.before = &waiting;
    auto next = std::async(std::launch::async, [&] {
        Throws([&] { core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core), hooks); }, "closed");
    });
    waiting.Wait(); core.Close(); next.get(); fifo.Drain(); core.Retire(a); core.Shutdown();
}

void PrepareFailurePreservesLogicalFirst() {
    Core core({4, 128, 2}); Fifo fifo;
    Event first_started, second_done, release_first;
    auto first = std::make_shared<TestJob>(); auto second = std::make_shared<TestJob>();
    first->prepare = [&] { first_started.Set(); release_first.Wait(); throw std::runtime_error("first-order"); };
    second->prepare = [&] { second_done.Set(); throw std::runtime_error("second-order"); };
    auto a = core.Submit(first, fifo.Publisher(core));
    auto b = core.Submit(second, fifo.Publisher(core));
    first_started.Wait(); second_done.Wait(); release_first.Set();
    Throws([&] { core.Consume(a); }, "first-order");
    Throws([&] { core.Consume(b); }, "first-order");
    CHECK(first->executed == 0 && second->executed == 0);
    CHECK(core.Snapshot().first_failed_sequence == a->sequence);
    core.Retire(a); core.Retire(b); core.Shutdown();
}

void EarlierPrefixExecutesDespiteLaterFailure() {
    Core core({3, 128, 2}); Fifo fifo;
    Event bad_done;
    auto first = std::make_shared<TestJob>(); auto bad = std::make_shared<TestJob>();
    bad->prepare = [&] { bad_done.Set(); throw std::runtime_error("later"); };
    auto a = core.Submit(first, fifo.Publisher(core)); auto b = core.Submit(bad, fifo.Publisher(core));
    bad_done.Wait(); core.Consume(a); CHECK(first->executed == 1);
    Throws([&] { core.Consume(b); }, "later"); core.Retire(a); core.Retire(b); core.Shutdown();
}

void PublishNotAcceptedWaitsForPrepareAck() {
    Core core({1, 64, 1}); Event started, release;
    auto job = std::make_shared<TestJob>(); std::atomic<bool> preparing{false};
    job->prepare = [&] { preparing = true; started.Set(); release.Wait(); preparing = false; };
    job->cancel = [&] { CHECK(!preparing); };
    auto ticket = core.Submit(job, [&](const Core::TicketPtr&) {
        started.Wait(); return Core::PublishReceipt{Core::PublishState::NotAccepted, Error("not accepted")};
    });
    CHECK(core.Publication(ticket) == Core::PublishState::NotAccepted);
    CHECK(core.Snapshot().slots_in_use == 1 && job->cancelled == 0);
    release.Set(); core.Shutdown();
    CHECK(job->cancelled == 1 && !job->cleanup_failed && core.Snapshot().slots_in_use == 0);
}

void CommittedPublishErrorKeepsOwnership() {
    Core core({1, 64, 1}); auto job = std::make_shared<TestJob>();
    auto ticket = core.Submit(job, [](const Core::TicketPtr&) {
        return Core::PublishReceipt{Core::PublishState::Committed, Error("committed error")};
    });
    Throws([&] { core.Consume(ticket); }, "committed error");
    CHECK(core.Snapshot().slots_in_use == 1 && job->executed == 0);
    core.Retire(ticket); core.Shutdown();
}

void ExecuteFailureHoldsUntilFenceAck() {
    Core core({1, 64, 1}); Fifo fifo; auto job = std::make_shared<TestJob>();
    job->execute = [] { throw std::runtime_error("execute failure"); };
    auto ticket = core.Submit(job, fifo.Publisher(core));
    Throws([&] { core.Consume(ticket); }, "execute failure");
    CHECK(core.Snapshot().slots_in_use == 1 && job->cancelled == 0);
    core.Retire(ticket); core.Shutdown(); CHECK(job->cancelled == 1);
}

void CancellationCannotRacePrepare() {
    Core core({1, 64, 1}); Fifo fifo; Event started, release;
    std::atomic<bool> preparing{false}; auto job = std::make_shared<TestJob>();
    job->prepare = [&] { preparing = true; started.Set(); release.Wait(); preparing = false; };
    job->cancel = [&] { CHECK(!preparing); };
    auto ticket = core.Submit(job, fifo.Publisher(core)); started.Wait(); core.Cancel(ticket);
    CHECK(job->cancelled == 0); release.Set();
    Throws([&] { core.Consume(ticket); }, "cancel");
    core.Retire(ticket); core.Shutdown(); CHECK(job->cancelled == 1 && !job->cleanup_failed);
}

void PostQueryBytesFailWithoutWaitingForLaterTicket() {
    Core core({2, 24, 2}); Fifo fifo; Event started, release;
    auto first = std::make_shared<TestJob>(); auto second = std::make_shared<TestJob>();
    first->prepare = [&] { started.Set(); release.Wait(); first->bytes = 32; };
    first->cancel = [&] { first->bytes = 8; };
    auto a = core.Submit(first, fifo.Publisher(core)); started.Wait();
    auto b = core.Submit(second, fifo.Publisher(core)); release.Set();
    Throws([&] { core.Consume(a); }, "byte");
    Throws([&] { core.Consume(b); }, "byte");
    CHECK(core.Snapshot().retained_bytes <= 24 && first->cancelled == 1);
    core.Retire(a); core.Retire(b); core.Shutdown();
}

void PostQueryBytesStayReservedThroughExecution() {
    Core core({2, 64, 1}); Fifo fifo; auto job = std::make_shared<TestJob>();
    job->prepare = [&] { job->bytes = 40; };
    auto ticket = core.Submit(job, fifo.Publisher(core)); fifo.Drain();
    CHECK(core.Snapshot().retained_bytes == 40);
    core.Retire(ticket); CHECK(core.Snapshot().retained_bytes == 0); core.Shutdown();
}

void InitialByteOverflowRejectedBeforeAdmission() {
    Core core({2, 4, 1}); Fifo fifo; auto job = std::make_shared<TestJob>();
    Throws([&] { core.Submit(job, fifo.Publisher(core)); }, "byte");
    CHECK(core.Snapshot().admitted == 0 && job->prepared == 0); core.Shutdown();
}

void IndeterminatePublicationRetainsOwners() {
    Core core({2, 64, 1}); auto job = std::make_shared<TestJob>();
    std::weak_ptr<TestJob> weak = job;
    auto ticket = core.Submit(job, [](const Core::TicketPtr&) {
        return Core::PublishReceipt{Core::PublishState::Indeterminate, Error("unknown ownership")};
    });
    Throws([&] { core.Shutdown(); }, "quarantined"); CHECK(core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1);
    job.reset(); ticket.reset(); CHECK(!weak.expired());
}

void PublishThrowIsIndeterminate() {
    Core core({1, 64, 1});
    auto ticket = core.Submit(std::make_shared<TestJob>(), [](const Core::TicketPtr&) -> Core::PublishReceipt {
        throw std::runtime_error("publish throw");
    });
    CHECK(core.Publication(ticket) == Core::PublishState::Indeterminate);
    Throws([&] { core.Shutdown(); }, "quarantined"); CHECK(core.Snapshot().quarantined);
}

void CleanupFailureQuarantines() {
    Core core({1, 64, 1}); Fifo fifo; auto job = std::make_shared<TestJob>();
    job->cancel = [] { throw std::runtime_error("cleanup failed"); };
    auto ticket = core.Submit(job, fifo.Publisher(core)); fifo.Drain(); core.Retire(ticket);
    CHECK(core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1);
    Throws([&] { core.Shutdown(); }, "quarantined");
}

void CallbacksAndReattachRunWithoutStateLock() {
    Core core({1, 64, 1}); Fifo fifo; Event started, release, waiting;
    auto job = std::make_shared<TestJob>();
    job->prepare = [&] { (void)core.Snapshot(); started.Set(); release.Wait(); };
    job->execute = [&] { (void)core.Snapshot(); };
    job->cancel = [&] { (void)core.Snapshot(); };
    Hooks submit_hooks; submit_hooks.core = &core;
    auto ticket = core.Submit(job, [&](const Core::TicketPtr& t) { (void)core.Snapshot(); return fifo.Publish(core, t); }, submit_hooks);
    started.Wait(); Hooks hooks; hooks.core = &core; hooks.before = &waiting;
    auto consuming = std::async(std::launch::async, [&] { core.Consume(ticket, hooks); });
    waiting.Wait(); release.Set(); consuming.get();
    CHECK(hooks.attached == hooks.detached && hooks.attached > 0);
    core.Retire(ticket); core.Shutdown();
}

void RejectDoubleConsumeAndEarlyRetire() {
    Core core({1, 64, 1}); Fifo fifo; Event started, release;
    auto job = std::make_shared<TestJob>(); job->prepare = [&] { started.Set(); release.Wait(); };
    auto ticket = core.Submit(job, fifo.Publisher(core)); started.Wait();
    Throws([&] { core.Retire(ticket); }, "retire"); release.Set(); core.Consume(ticket);
    Throws([&] { core.Consume(ticket); }, "consum"); core.Retire(ticket); core.Retire(ticket); core.Shutdown();
}

void DiscardWaitsForRollbackAck() {
    Core core({1, 64, 1}); Fifo fifo; Event started, release, waiting;
    auto job = std::make_shared<TestJob>(); std::atomic<bool> preparing{false};
    job->prepare = [&] { preparing = true; started.Set(); release.Wait(); preparing = false; };
    job->cancel = [&] { CHECK(!preparing); };
    auto ticket = core.Submit(job, fifo.Publisher(core)); started.Wait();
    Hooks hooks; hooks.core = &core; hooks.before = &waiting;
    auto discard = std::async(std::launch::async, [&] { core.CancelAndWait(ticket, hooks); core.Retire(ticket); });
    waiting.Wait(); CHECK(job->cancelled == 0); release.Set(); discard.get();
    CHECK(job->cancelled == 1 && !job->cleanup_failed && core.Snapshot().slots_in_use == 0); core.Shutdown();
}

void OwnerDestructionPrecedesCreditRelease() {
    Core core({1, 64, 1}); Fifo fifo; bool owner_destroyed = false; bool held_until_destroyed = false;
    auto job = std::make_shared<TestJob>();
    job->destroy = [&] { owner_destroyed = true; held_until_destroyed = core.Snapshot().slots_in_use == 1; };
    auto ticket = core.Submit(job, fifo.Publisher(core)); job.reset(); fifo.Drain(); core.Retire(ticket);
    CHECK(owner_destroyed && held_until_destroyed); core.Shutdown();
}

void FullOriginalFifoCannotBlockPrepareProgress() {
    Core core({2, 64, 2}); c10_npu::queue::OrderedMpscQueueCore fifo(1); fifo.Start();
    std::array<Core::TicketPtr, 1> slots; Event first_started, second_prepared;
    auto first = std::make_shared<TestJob>(); auto second = std::make_shared<TestJob>();
    first->prepare = [&] { first_started.Set(); second_prepared.Wait(); };
    second->prepare = [&] { second_prepared.Set(); };
    auto publish = [&](const Core::TicketPtr& ticket) {
        c10_npu::queue::NoopWaitHooks hooks;
        bool accepted = fifo.Enqueue([&](uint32_t slot) { slots[slot] = ticket; }, hooks);
        return Core::PublishReceipt{accepted ? Core::PublishState::Committed : Core::PublishState::NotAccepted, nullptr};
    };
    auto a = core.Submit(first, publish); first_started.Wait();
    auto pending = std::async(std::launch::async, [&] { return core.Submit(second, publish); });
    second_prepared.Wait();
    CHECK(fifo.ConsumeOne([&](uint32_t slot) { core.Consume(slots[slot]); }));
    auto b = pending.get();
    CHECK(fifo.ConsumeOne([&](uint32_t slot) { core.Consume(slots[slot]); }));
    core.Retire(a); core.Retire(b); fifo.StopAndDrain(); core.Shutdown();
}

void ConcurrentProducersUseRealOriginalFifo() {
    constexpr size_t total = 240, producers = 4;
    Core core({8, 256, 4}); c10_npu::queue::OrderedMpscQueueCore fifo(8); fifo.Start();
    std::array<Core::TicketPtr, 8> slots; std::vector<uint64_t> executed;
    auto consumer = std::async(std::launch::async, [&] {
        for (size_t i = 0; i < total; ++i) {
            CHECK(fifo.ConsumeOne([&](uint32_t slot) {
                auto ticket = slots[slot]; core.Consume(ticket); core.Retire(ticket); slots[slot].reset();
            }));
        }
    });
    std::vector<std::future<void>> producers_done;
    for (size_t p = 0; p < producers; ++p) producers_done.push_back(std::async(std::launch::async, [&] {
        for (size_t i = 0; i < total / producers; ++i) {
            auto job = std::make_shared<TestJob>();
            core.Submit(job, [&](const Core::TicketPtr& ticket) {
                job->execute = [&, sequence = ticket->sequence] { executed.push_back(sequence); };
                c10_npu::queue::NoopWaitHooks hooks;
                CHECK(fifo.Enqueue([&](uint32_t slot) { slots[slot] = ticket; }, hooks));
                return Core::PublishReceipt{Core::PublishState::Committed, nullptr};
            });
        }
    }));
    for (auto& producer : producers_done) producer.get(); consumer.get();
    CHECK(executed.size() == total);
    for (size_t i = 0; i < total; ++i) CHECK(executed[i] == i);
    CHECK(core.Snapshot().admitted == total && core.Snapshot().retired == total);
    fifo.StopAndDrain(); core.Shutdown();
}

struct TlsWitness {
    std::shared_ptr<std::atomic<bool>> destroyed;
    ~TlsWitness() { *destroyed = true; }
};
void QuarantineRetainsWorkerTlsAfterCoreDestruction() {
    auto destroyed = std::make_shared<std::atomic<bool>>(false);
    {
        Core core({1, 64, 1}); auto job = std::make_shared<TestJob>();
        job->prepare = [destroyed] { thread_local std::unique_ptr<TlsWitness> witness;
            witness.reset(new TlsWitness{destroyed}); };
        core.Submit(job, [](const Core::TicketPtr&) {
            return Core::PublishReceipt{Core::PublishState::Indeterminate, Error("uncertain")};
        });
        Throws([&] { core.Shutdown(); }, "quarantined"); CHECK(!*destroyed);
    }
    CHECK(!*destroyed);
}

void NormalShutdownRetainsTlsUntilRetirement() {
    Core core({1, 64, 1}); Fifo fifo; Event waiting;
    auto destroyed = std::make_shared<std::atomic<bool>>(false);
    auto job = std::make_shared<TestJob>();
    job->prepare = [destroyed] { thread_local std::unique_ptr<TlsWitness> witness;
        witness.reset(new TlsWitness{destroyed}); };
    auto ticket = core.Submit(job, fifo.Publisher(core)); fifo.Drain(); CHECK(!*destroyed);
    Hooks hooks; hooks.core = &core; hooks.before = &waiting;
    auto shutdown = std::async(std::launch::async, [&] { core.Shutdown(hooks); });
    waiting.Wait(); CHECK(!*destroyed && shutdown.wait_for(0ms) == std::future_status::timeout);
    core.Retire(ticket); shutdown.get(); CHECK(*destroyed);
}

void SubmitReattachCanReenterSubmission() {
    Core core({2, 64, 1}); Fifo fifo;
    struct ReenterHooks {
        Core& core; Fifo& fifo; Core::TicketPtr nested;
        void BeforeWait() noexcept {}
        void AfterWait() { nested = core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core)); }
    } hooks{core, fifo, nullptr};
    auto first = core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core), hooks);
    CHECK(hooks.nested && hooks.nested->sequence == first->sequence + 1);
    fifo.Drain(); core.Retire(first); core.Retire(hooks.nested); core.Shutdown();
}

void FailedRetirementRetainsDeviceOwners() {
    Core core({1, 64, 1}); Fifo fifo; auto job = std::make_shared<TestJob>();
    auto ticket = core.Submit(job, fifo.Publisher(core)); fifo.Drain(); core.Retire(ticket, false);
    CHECK(core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1 && job->cancelled == 0);
    Throws([&] { core.RethrowIfFailed(); }, "retirement failed");
    Throws([&] { core.Shutdown(); }, "quarantined");
}

void DiscardAfterExecuteStillNeedsDeviceRetirement() {
    Core core({1, 64, 1}); Fifo fifo; Event executing, release, waiting;
    auto job = std::make_shared<TestJob>();
    job->execute = [&] { executing.Set(); release.Wait(); };
    auto ticket = core.Submit(job, fifo.Publisher(core));
    auto consume = std::async(std::launch::async, [&] { core.Consume(ticket); }); executing.Wait();
    Hooks hooks; hooks.core = &core; hooks.before = &waiting;
    auto discard = std::async(std::launch::async, [&] { core.CancelAndWait(ticket, hooks); });
    waiting.Wait(); CHECK(job->cancelled == 0); release.Set(); consume.get(); discard.get();
    CHECK(job->cancelled == 0 && core.Snapshot().slots_in_use == 1);
    core.Retire(ticket); CHECK(job->cancelled == 1); core.Shutdown();
}

struct ResourceJob final : Core::Job {
    std::shared_ptr<std::atomic<size_t>> actual_bytes;
    std::unique_ptr<unsigned char[]> owned;
    bool release_on_cancel;
    explicit ResourceJob(std::shared_ptr<std::atomic<size_t>> actual, bool release)
        : actual_bytes(std::move(actual)), owned(new unsigned char[64]), release_on_cancel(release) { *actual_bytes += 64; }
    ~ResourceJob() override { Release(); }
    void Prepare() override {}
    void Execute() override {}
    void Cancel() noexcept override { if (release_on_cancel) Release(); }
    size_t RetainedBytes() const noexcept override { return owned ? 64 : 0; }
    void Release() noexcept { if (owned) { owned.reset(); *actual_bytes -= 64; } }
};

void NonzeroCleanupAckQuarantinesExternalOwner() {
    Core core({1, 64, 1}); Fifo fifo;
    auto actual_bytes = std::make_shared<std::atomic<size_t>>(0);
    auto externally_retained = std::make_shared<ResourceJob>(actual_bytes, false);
    auto ticket = core.Submit(externally_retained, fifo.Publisher(core)); fifo.Drain(); core.Retire(ticket);
    CHECK(*actual_bytes == 64);
    CHECK(core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1 && core.Snapshot().retained_bytes == 64);
    Throws([&] { core.Submit(std::make_shared<TestJob>(), fifo.Publisher(core)); }, "closed");
    Throws([&] { core.RethrowIfFailed(); }, "retained bytes");
    Throws([&] { core.Shutdown(); }, "quarantined");
}

void ExplicitResourceReleaseMakesExternalWrapperHarmless() {
    Core core({1, 64, 1}); Fifo fifo;
    auto actual_bytes = std::make_shared<std::atomic<size_t>>(0);
    auto externally_retained = std::make_shared<ResourceJob>(actual_bytes, true);
    auto first = core.Submit(externally_retained, fifo.Publisher(core)); fifo.Drain();
    CHECK(*actual_bytes == 64); core.Retire(first);
    CHECK(externally_retained && externally_retained->RetainedBytes() == 0 && *actual_bytes == 0);
    CHECK(core.Snapshot().slots_in_use == 0 && core.Snapshot().retained_bytes == 0);
    auto second_job = std::make_shared<ResourceJob>(actual_bytes, true);
    auto second = core.Submit(second_job, fifo.Publisher(core));
    CHECK(*actual_bytes == 64 && core.Snapshot().retained_bytes == 64);
    fifo.Drain(); core.Retire(second); CHECK(*actual_bytes == 0); core.Shutdown();
}

void WorkersStartedCountsActualWorkerEntry() {
    Core core({3, 64, 3}); Fifo fifo; Event all_preparing, release;
    std::atomic<size_t> entered{0}; std::vector<Core::TicketPtr> tickets;
    for (size_t i = 0; i < 3; ++i) {
        auto job = std::make_shared<TestJob>();
        job->prepare = [&] { if (++entered == 3) all_preparing.Set(); release.Wait(); };
        tickets.push_back(core.Submit(job, fifo.Publisher(core)));
    }
    all_preparing.Wait(); const auto actual_started = core.Snapshot().workers_started; release.Set();
    fifo.Drain(); for (const auto& ticket : tickets) core.Retire(ticket); core.Shutdown();
    CHECK(actual_started == 3 && core.Snapshot().workers_started == 3);
}

void ReservationPeaksIncludePrepareTopup() {
    Core core({2, 80, 2}); Fifo fifo;
    auto first = std::make_shared<TestJob>(); auto second = std::make_shared<TestJob>();
    first->prepare = [&] { first->bytes = 24; }; second->prepare = [&] { second->bytes = 32; };
    auto a = core.Submit(first, fifo.Publisher(core)); auto b = core.Submit(second, fifo.Publisher(core));
    fifo.Drain(); const auto snapshot = core.Snapshot(); core.Retire(a); core.Retire(b); core.Shutdown();
    CHECK(snapshot.peak_slots_in_use == 2 && snapshot.peak_retained_bytes == 56);
    CHECK(core.Snapshot().slots_in_use == 0 && core.Snapshot().retained_bytes == 0);
    CHECK(core.Snapshot().peak_slots_in_use == 2 && core.Snapshot().peak_retained_bytes == 56);
}

void SynchronousPublisherAcknowledgesBeforeWaitingForConsumer() {
    Core core({1, 64, 1}); c10_npu::queue::OrderedMpscQueueCore fifo(1); fifo.Start();
    Core::TicketPtr slot; Event consumer_done; bool completed_synchronously = true;
    auto consumer = std::async(std::launch::async, [&] {
        CHECK(fifo.ConsumeOne([&](uint32_t) { core.Consume(slot); core.Retire(slot); consumer_done.Set(); }));
    });
    auto job = std::make_shared<TestJob>();
    auto ticket = core.Submit(job, [&](const Core::TicketPtr& accepted) {
        c10_npu::queue::NoopWaitHooks hooks;
        CHECK(fifo.Enqueue([&](uint32_t) { slot = accepted; CHECK(core.MarkCommitted(accepted)); }, hooks));
        // Models NPUQueue INIT: Enqueue cannot return until its consumer drains.
        try { consumer_done.Wait(); } catch (...) { completed_synchronously = false; }
        return Core::PublishReceipt{Core::PublishState::Committed, nullptr};
    });
    consumer.get(); CHECK(core.Publication(ticket) == Core::PublishState::Committed);
    fifo.StopAndDrain(); core.Shutdown(); CHECK(job->executed == 1);
    CHECK(completed_synchronously);
}

void EarlyCommitCannotBeRetractedByLateFailure() {
    Core core({1, 64, 1}); auto job = std::make_shared<TestJob>();
    auto ticket = core.Submit(job, [&](const Core::TicketPtr& accepted) -> Core::PublishReceipt {
        CHECK(core.MarkCommitted(accepted)); throw std::runtime_error("late publication failure");
    });
    CHECK(core.Publication(ticket) == Core::PublishState::Committed);
    CHECK(!core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1);
    Throws([&] { core.Consume(ticket); }, "late publication failure");
    core.Retire(ticket); core.Shutdown();
}

void EarlyCommitCannotBeRetractedByNotAcceptedReceipt() {
    Core core({1, 64, 1}); auto job = std::make_shared<TestJob>();
    auto ticket = core.Submit(job, [&](const Core::TicketPtr& accepted) {
        CHECK(core.MarkCommitted(accepted));
        return Core::PublishReceipt{Core::PublishState::NotAccepted, nullptr};
    });
    CHECK(core.Publication(ticket) == Core::PublishState::Committed);
    CHECK(!core.Snapshot().quarantined && core.Snapshot().slots_in_use == 1);
    Throws([&] { core.Consume(ticket); }, "committed");
    core.Retire(ticket); core.Shutdown();
}

int main() {
    const std::pair<const char*, void(*)()> tests[] = {
        {"reverse prepare and mixed legacy FIFO order", ReversePrepareAndMixedLegacyOrder},
        {"K1 backpressure lasts until retirement", K1BackpressureUntilRetirement},
        {"close wakes blocked submit", CloseWakesBlockedSubmit},
        {"prepare failures preserve logical first", PrepareFailurePreservesLogicalFirst},
        {"earlier prefix executes despite later failure", EarlierPrefixExecutesDespiteLaterFailure},
        {"not accepted waits for prepare ACK", PublishNotAcceptedWaitsForPrepareAck},
        {"committed publish error keeps ownership", CommittedPublishErrorKeepsOwnership},
        {"execute failure holds until fence ACK", ExecuteFailureHoldsUntilFenceAck},
        {"cancel cannot race prepare", CancellationCannotRacePrepare},
        {"post-query bytes fail without waiting", PostQueryBytesFailWithoutWaitingForLaterTicket},
        {"post-query bytes remain reserved", PostQueryBytesStayReservedThroughExecution},
        {"initial byte overflow rejects before admission", InitialByteOverflowRejectedBeforeAdmission},
        {"indeterminate publication retains owners", IndeterminatePublicationRetainsOwners},
        {"publish throw is indeterminate", PublishThrowIsIndeterminate},
        {"cleanup failure quarantines", CleanupFailureQuarantines},
        {"callbacks and reattach run without state lock", CallbacksAndReattachRunWithoutStateLock},
        {"double consume and early retire rejected", RejectDoubleConsumeAndEarlyRetire},
        {"discard waits for rollback ACK", DiscardWaitsForRollbackAck},
        {"owner destruction precedes credit release", OwnerDestructionPrecedesCreditRelease},
        {"full original FIFO cannot block Prepare", FullOriginalFifoCannotBlockPrepareProgress},
        {"concurrent producers use real original FIFO", ConcurrentProducersUseRealOriginalFifo},
        {"quarantine retains worker TLS", QuarantineRetainsWorkerTlsAfterCoreDestruction},
        {"normal shutdown retains TLS until retirement", NormalShutdownRetainsTlsUntilRetirement},
        {"Submit reattach can reenter submission", SubmitReattachCanReenterSubmission},
        {"failed retirement retains device owners", FailedRetirementRetainsDeviceOwners},
        {"discard after Execute still needs device retirement", DiscardAfterExecuteStillNeedsDeviceRetirement},
        {"nonzero cleanup ACK quarantines external owner", NonzeroCleanupAckQuarantinesExternalOwner},
        {"explicit release makes external wrapper harmless", ExplicitResourceReleaseMakesExternalWrapperHarmless},
        {"workers started counts actual entry", WorkersStartedCountsActualWorkerEntry},
        {"reservation peaks include Prepare top-up", ReservationPeaksIncludePrepareTopup},
        {"synchronous publisher ACK precedes consumer wait", SynchronousPublisherAcknowledgesBeforeWaitingForConsumer},
        {"early commit cannot be retracted by late failure", EarlyCommitCannotBeRetractedByLateFailure},
        {"early commit cannot be retracted by NotAccepted", EarlyCommitCannotBeRetractedByNotAcceptedReceipt},
    };
    size_t passed = 0;
    for (const auto& test : tests) {
        try { test.second(); ++passed; std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) { std::cerr << "FAIL " << test.first << ": " << error.what() << '\n'; }
    }
    std::cout << passed << " host contract tests passed (no NPU execution).\n";
    return passed == sizeof(tests) / sizeof(tests[0]) ? 0 : 1;
}
