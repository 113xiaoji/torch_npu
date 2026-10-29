#include "torch_npu/csrc/core/npu/eager_parallel/DependencyScheduler.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>

using namespace c10_npu::eager_parallel;

// Exercise real std::allocator failure, including allocations in registration maps.
// Thread-local injection leaves the foreign-caller race test's allocations untouched.
// Keep the matched malloc/free replacement pair opaque to the optimizer: inlining
// replacement delete alone makes GCC diagnose ordinary STL new/delete as a mismatch.
#if defined(_MSC_VER)
#define R26_TEST_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define R26_TEST_NOINLINE __attribute__((noinline))
#else
#define R26_TEST_NOINLINE
#endif
thread_local std::ptrdiff_t fail_allocation_after = -1;
R26_TEST_NOINLINE void* operator new(std::size_t bytes)
{
    if (fail_allocation_after == 0) {
        fail_allocation_after = -1;
        throw std::bad_alloc();
    }
    if (fail_allocation_after > 0) --fail_allocation_after;
    void* result = std::malloc(bytes ? bytes : 1);
    if (!result) throw std::bad_alloc();
    return result;
}
R26_TEST_NOINLINE void operator delete(void* pointer) noexcept { std::free(pointer); }
R26_TEST_NOINLINE void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
#undef R26_TEST_NOINLINE

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error( \
    std::string(__func__) + ": " #expr); } while (false)

template<class F> void rejects(F f)
{
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    CHECK(caught);
}

// Only the external event service is simulated; all scheduling/lifetime code is real.
struct FakeBackend final : EventBackend {
    struct Event { bool published = false; bool complete = false; bool retired = false; };
    std::map<EventToken, Event> events;
    std::vector<std::pair<StreamId, EventToken>> waits;
    bool immediate_publication = true;
    bool fail_reserve = false;
    bool fail_record = false;
    bool fail_wait = false;
    bool fail_query = false;
    bool fail_publication = false;
    bool fail_retire = false;
    std::size_t record_calls = 0;
    std::size_t retire_calls = 0;
    std::function<void(const char*)> callback;
    void reserve_event(EventToken token, StreamId) override
    {
        if (callback) callback("reserve");
        if (fail_reserve) throw std::runtime_error("reserve failure");
        CHECK(events.emplace(token, Event{}).second);
    }
    void request_record(EventToken token, StreamId) override
    {
        if (callback) callback("record");
        ++record_calls;
        if (fail_record) throw std::runtime_error("record failure");
        events.at(token).published = immediate_publication;
    }
    Publication publication(EventToken token) override
    {
        if (callback) callback("publication");
        if (fail_publication) return Publication::Failed;
        return events.at(token).published ? Publication::Published : Publication::Pending;
    }
    void enqueue_wait(StreamId stream, EventToken token) override
    {
        if (callback) callback("wait");
        if (fail_wait) throw std::runtime_error("wait failure");
        CHECK(events.at(token).published);
        CHECK(!events.at(token).retired);
        waits.emplace_back(stream, token);
    }
    Completion device_complete(EventToken token) override
    {
        if (callback) callback("completion");
        if (fail_query) return Completion::Failed;
        CHECK(events.at(token).published);
        return events.at(token).complete ? Completion::Complete : Completion::Pending;
    }
    void retire_event(EventToken token) override
    {
        if (callback) callback("retirement");
        ++retire_calls;
        if (fail_retire) throw std::runtime_error("retire failure");
        CHECK(!events.at(token).retired);
        events.at(token).retired = true;
    }
    void finish_all()
    {
        for (auto& entry : events) { entry.second.published = true; entry.second.complete = true; }
    }
};

AllocationLease allocation(std::uint64_t identity, std::size_t bytes = 16,
                           std::uint64_t generation = 1)
{
    return {{identity, generation}, bytes, std::make_shared<int>(42), EntryReadiness::DeviceComplete};
}
FrozenCall call(std::uint64_t sequence, NodeKind kind, std::initializer_list<Access> accesses)
{
    FrozenCall result;
    result.logical_sequence = sequence;
    result.kind = kind;
    result.accesses = accesses;
    return result;
}
const auto launch = [](StreamId) { return LaunchResult::Submitted; };

void hazards_and_aliases()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    const auto a = allocation(1);
    const auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    const auto read1 = scheduler.admit(call(2, NodeKind::Producer, {{a, AccessMode::Read}}));
    const auto read2 = scheduler.admit(call(3, NodeKind::Consumer, {{a, AccessMode::Read}, {a, AccessMode::Read}}));
    const auto writer2 = scheduler.admit(call(4, NodeKind::Consumer, {{a, AccessMode::Write}}));
    CHECK(scheduler.inspect(read1).dependencies == std::vector<NodeId>{writer}); // RAW
    CHECK(scheduler.inspect(read2).dependencies == std::vector<NodeId>{writer}); // read/read alias
    CHECK(scheduler.inspect(writer2).dependencies == (std::vector<NodeId>{writer, read1, read2})); // WAR + WAW
    CHECK(scheduler.snapshot().retained_bytes == a.bytes); // whole allocation charged once
    CHECK(scheduler.submit(writer, launch) == SubmitResult::Submitted);
    CHECK(scheduler.submit(read1, launch) == SubmitResult::Submitted);
    CHECK(scheduler.submit(read2, launch) == SubmitResult::Submitted);
    CHECK(scheduler.submit(writer2, launch) == SubmitResult::Submitted);
    backend->finish_all();
    CHECK(scheduler.close() == CloseResult::Closed);
    CHECK(scheduler.snapshot().retained_bytes == 0);
    CHECK(backend->retire_calls == 4);
}

void producer_choice_and_fan_in()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto dense = allocation(10), a = allocation(11), b = allocation(12), c = allocation(13);
    auto d = scheduler.admit(call(1, NodeKind::Consumer, {{dense, AccessMode::Write}}));
    auto e1 = scheduler.admit(call(2, NodeKind::Producer, {{a, AccessMode::Write}}));
    auto e2 = scheduler.admit(call(3, NodeKind::Producer, {{b, AccessMode::Write}}));
    auto e3 = scheduler.admit(call(4, NodeKind::Producer, {{c, AccessMode::Write}}));
    CHECK(scheduler.inspect(e1).stream == 1);
    CHECK(scheduler.inspect(e2).stream == 2);
    CHECK(scheduler.inspect(e3).stream == 1);
    CHECK(scheduler.inspect(e1).dependencies.empty()); // no unrelated S0 tail
    for (auto id : {d, e1, e2, e3}) CHECK(scheduler.submit(id, launch) == SubmitResult::Submitted);
    auto cat = scheduler.admit(call(5, NodeKind::Consumer, {{dense, AccessMode::Read},
        {a, AccessMode::Read}, {b, AccessMode::Read}, {c, AccessMode::Read}}));
    CHECK(scheduler.submit(cat, launch) == SubmitResult::Submitted);
    CHECK(backend->waits.size() == 2);
    CHECK(backend->waits[0].second == scheduler.inspect(e3).event);
    CHECK(backend->waits[1].second == scheduler.inspect(e2).event);
    backend->finish_all();
    CHECK(scheduler.close() == CloseResult::Closed);
}

void delayed_publication_is_not_completion()
{
    auto backend = std::make_shared<FakeBackend>();
    backend->immediate_publication = false;
    DependencyScheduler scheduler(backend);
    auto a = allocation(20);
    std::weak_ptr<void> weak = a.owner;
    auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
    auto event = scheduler.inspect(writer).event;
    a.owner.reset();
    int launches = 0;
    auto counted = [&](StreamId) { ++launches; return LaunchResult::Submitted; };
    CHECK(scheduler.submit(writer, counted) == SubmitResult::Submitted);
    CHECK(scheduler.submit(reader, counted) == SubmitResult::PublicationPending);
    CHECK(backend->waits.empty());
    CHECK(launches == 1);
    CHECK(!weak.expired());
    backend->events.at(event).published = true;
    CHECK(scheduler.submit(reader, counted) == SubmitResult::Submitted);
    CHECK(backend->waits.size() == 1);
    CHECK(scheduler.submit(reader, counted) == SubmitResult::AlreadySubmitted);
    CHECK(launches == 2);
    CHECK(scheduler.close() == CloseResult::Pending);
    CHECK(!weak.expired());
    backend->events.at(event).complete = true;
    scheduler.poll();
    CHECK(!weak.expired()); // a second device reader still owns the allocation
    CHECK(!backend->events.at(event).retired); // consumer wait also pins producer event
    backend->finish_all();
    CHECK(scheduler.close() == CloseResult::Closed);
    CHECK(weak.expired());
}

void generation_and_late_ack()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto old = allocation(30);
    auto id = scheduler.admit(call(1, NodeKind::Producer, {{old, AccessMode::Write}}));
    auto old_event = scheduler.inspect(id).event;
    scheduler.submit(id, launch);
    rejects([&] { scheduler.admit(call(2, NodeKind::Producer, {{allocation(30, 16, 2), AccessMode::Write}})); });
    backend->finish_all(); scheduler.poll();
    auto next = scheduler.admit(call(2, NodeKind::Producer, {{allocation(30, 16, 2), AccessMode::Write}}));
    auto new_event = scheduler.inspect(next).event;
    CHECK(old_event.slot == new_event.slot);
    CHECK(old_event.generation != new_event.generation);
    backend->immediate_publication = false;
    scheduler.submit(next, launch);
    backend->events.at(old_event).published = true; // stale ACK cannot affect new generation
    backend->events.at(old_event).complete = true;
    scheduler.poll();
    CHECK(scheduler.snapshot().active_nodes == 1);
    CHECK(scheduler.snapshot().retained_bytes == 16);
    rejects([&] { scheduler.admit(call(3, NodeKind::Consumer, {{old, AccessMode::Read}})); });
    backend->finish_all(); CHECK(scheduler.close() == CloseResult::Closed);
}

void budgets_are_actual_nonzero_and_bounded()
{
    auto backend = std::make_shared<FakeBackend>();
    SchedulerConfig config; config.max_nodes = 2; config.max_retained_bytes = 40;
    DependencyScheduler scheduler(backend, config);
    rejects([&] { scheduler.admit(call(1, NodeKind::Producer, {{allocation(40, 0), AccessMode::Write}})); });
    auto a = allocation(40, 24);
    auto frozen = call(1, NodeKind::Producer, {{a, AccessMode::Read}});
    frozen.retained_bytes = 8;
    frozen.retained.push_back(std::make_shared<int>(1));
    scheduler.admit(frozen);
    CHECK(scheduler.snapshot().retained_bytes == 32);
    rejects([&] { scheduler.admit(call(2, NodeKind::Producer, {{allocation(41), AccessMode::Write}})); });
    scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
    rejects([&] { scheduler.admit(call(3, NodeKind::Consumer, {{a, AccessMode::Read}})); });
    CHECK(scheduler.close() == CloseResult::Closed); // unsubmitted cancellation needs no device wait
    CHECK(scheduler.snapshot().retained_bytes == 0);
}

void failure_and_cancel_keep_submitted_ownership()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(50);
    std::weak_ptr<void> weak = a.owner;
    auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
    a.owner.reset();
    backend->fail_record = true;
    CHECK(scheduler.submit(writer, launch) == SubmitResult::Failed);
    CHECK(scheduler.submit(reader, launch) == SubmitResult::Rejected);
    auto error = scheduler.snapshot().first_error;
    CHECK(error.find("record failure") != std::string::npos);
    scheduler.cancel("later cancellation");
    CHECK(scheduler.close() == CloseResult::Failed);
    CHECK(scheduler.snapshot().first_error == error);
    CHECK(!weak.expired());
    CHECK(scheduler.snapshot().quarantined_nodes == 1);
    // A late, positive ACK followed by real device completion can safely retire.
    backend->finish_all(); scheduler.poll();
    CHECK(weak.expired());
    CHECK(scheduler.snapshot().active_nodes == 0);
    CHECK(scheduler.close() == CloseResult::Failed); // first error remains sticky
}

void no_replay_and_close_late_completion()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto id = scheduler.admit(call(1, NodeKind::Producer, {{allocation(60), AccessMode::Write}}));
    int launches = 0;
    auto counted = [&](StreamId) { ++launches; return LaunchResult::Submitted; };
    scheduler.submit(id, counted);
    scheduler.cancel("cancel after launch");
    CHECK(scheduler.submit(id, counted) == SubmitResult::AlreadySubmitted);
    CHECK(launches == 1);
    CHECK(scheduler.close() == CloseResult::Failed);
    CHECK(scheduler.snapshot().active_nodes == 1);
    rejects([&] { scheduler.admit(call(2, NodeKind::Producer, {{allocation(61), AccessMode::Write}})); });
    backend->finish_all(); scheduler.poll();
    CHECK(scheduler.snapshot().active_nodes == 0);
    CHECK(backend->retire_calls == 1);
    scheduler.poll(); CHECK(backend->retire_calls == 1);
}

void reserve_and_launch_failures()
{
    for (int mode = 0; mode < 3; ++mode) {
        auto backend = std::make_shared<FakeBackend>();
        DependencyScheduler scheduler(backend);
        auto id = scheduler.admit(call(1, NodeKind::Producer, {{allocation(70), AccessMode::Write}}));
        backend->fail_reserve = mode == 0;
        int launches = 0;
        auto fail = [&](StreamId) {
            ++launches;
            return mode == 1 ? LaunchResult::RejectedBeforeSubmit : LaunchResult::Uncertain;
        };
        CHECK(scheduler.submit(id, fail) == SubmitResult::Failed);
        CHECK(launches == (mode == 0 ? 0 : 1));
        scheduler.poll();
        CHECK(scheduler.snapshot().active_nodes == (mode == 2 ? 1u : 0u));
        CHECK(scheduler.close() == CloseResult::Failed);
    }
}

void destroy_failure_never_retries_or_releases()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(80);
    std::weak_ptr<void> weak = a.owner;
    auto id = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    a.owner.reset(); scheduler.submit(id, launch);
    backend->finish_all(); backend->fail_retire = true;
    scheduler.poll(); scheduler.poll();
    CHECK(scheduler.close() == CloseResult::Failed);
    CHECK(backend->retire_calls == 1);
    CHECK(!weak.expired());
}

void single_caller_and_inference_only()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto training = call(1, NodeKind::Producer, {{allocation(90), AccessMode::Write}});
    training.inference = false;
    rejects([&] { scheduler.admit(training); });
    auto id = scheduler.admit(call(1, NodeKind::Producer, {{allocation(90), AccessMode::Write}}));
    scheduler.submit(id, launch);
    std::atomic<bool> rejected{false};
    std::thread foreign([&] {
        try { scheduler.poll(); } catch (const std::exception&) { rejected.store(true); }
    });
    foreign.join();
    CHECK(rejected.load());
    rejects([&] { scheduler.admit(call(2, NodeKind::Producer, {{allocation(91), AccessMode::Write}})); });
    backend->finish_all();
    CHECK(scheduler.close() == CloseResult::Failed);
    CHECK(scheduler.snapshot().first_error.find("owner") != std::string::npos);
}

void logical_sequence_and_access_strength()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(100);
    auto first = scheduler.admit(call(9, NodeKind::Producer, {{a, AccessMode::Read}, {a, AccessMode::Write}}));
    rejects([&] { scheduler.admit(call(9, NodeKind::Producer, {{a, AccessMode::Read}})); });
    auto second = scheduler.admit(call(10, NodeKind::Consumer, {{a, AccessMode::Read}}));
    CHECK(scheduler.inspect(second).dependencies == std::vector<NodeId>{first});
    CHECK(scheduler.submit(second, launch) == SubmitResult::PublicationPending);
    CHECK(scheduler.close() == CloseResult::Closed);
}

void entry_frontier_must_be_explicit()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(110);
    a.entry_readiness = EntryReadiness::NeedsFence;
    rejects([&] { scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Read}})); });
    auto entry = scheduler.admit(call(1, NodeKind::EntryFence, {{a, AccessMode::Write}}));
    scheduler.submit(entry, launch);
    auto reader = scheduler.admit(call(2, NodeKind::Producer, {{a, AccessMode::Read}}));
    scheduler.submit(reader, launch);
    CHECK(backend->waits.size() == 1);
    CHECK(backend->waits[0].second == scheduler.inspect(entry).event);
    backend->finish_all(); CHECK(scheduler.close() == CloseResult::Closed);
}

void reject_two_generations_in_one_call()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    rejects([&] { scheduler.admit(call(1, NodeKind::Producer, {
        {allocation(120, 16, 1), AccessMode::Write}, {allocation(120, 16, 2), AccessMode::Read}})); });
    CHECK(scheduler.snapshot().active_nodes == 0);
}

void completed_outputs_remain_ready_in_the_domain()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(130);
    a.entry_readiness = EntryReadiness::NewAllocation;
    auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    scheduler.submit(writer, launch); backend->finish_all(); scheduler.poll();
    auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
    CHECK(scheduler.inspect(reader).dependencies.empty());
    scheduler.submit(reader, launch); backend->finish_all();
    CHECK(scheduler.close() == CloseResult::Closed);
}

void wait_and_query_failures_are_conservative()
{
    for (int mode = 0; mode < 3; ++mode) {
        auto backend = std::make_shared<FakeBackend>();
        DependencyScheduler scheduler(backend);
        auto a = allocation(140);
        std::weak_ptr<void> weak = a.owner;
        auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
        scheduler.submit(writer, launch);
        if (mode == 0) {
            auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
            backend->fail_wait = true;
            CHECK(scheduler.submit(reader, launch) == SubmitResult::Failed);
            CHECK(scheduler.snapshot().submitted == 1); // reader launch never ran
        } else if (mode == 1) {
            backend->fail_query = true;
        } else {
            auto event = scheduler.inspect(writer).event;
            // A second node with delayed record reaches the publication failure path.
            backend->immediate_publication = false;
            auto id = scheduler.admit(call(2, NodeKind::Producer, {{allocation(141), AccessMode::Write}}));
            scheduler.submit(id, launch);
            backend->fail_publication = true;
            CHECK(backend->events.at(event).published);
        }
        a.owner.reset(); scheduler.poll();
        CHECK(scheduler.close() == CloseResult::Failed);
        CHECK(!weak.expired());
        CHECK(!scheduler.snapshot().first_error.empty());
        backend->fail_query = false;
        backend->fail_publication = false;
        backend->finish_all(); scheduler.poll();
        if (mode == 0) {
            CHECK(!weak.expired()); // ambiguous wait has no completion fence
            CHECK(scheduler.snapshot().active_nodes == 2);
        } else {
            CHECK(weak.expired());
            CHECK(scheduler.snapshot().active_nodes == 0);
        }
    }
}

void completion_before_ack_does_not_release_and_destructor_quarantines()
{
    auto backend = std::make_shared<FakeBackend>();
    backend->immediate_publication = false;
    auto a = allocation(150);
    std::weak_ptr<void> weak = a.owner;
    {
        DependencyScheduler scheduler(backend);
        auto id = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
        a.owner.reset(); scheduler.submit(id, launch);
        backend->events.at(scheduler.inspect(id).event).complete = true;
        CHECK(scheduler.close() == CloseResult::Pending);
        CHECK(!weak.expired());
        CHECK(backend->retire_calls == 0);
    }
    CHECK(!weak.expired()); // destroying the domain never substitutes for completion
    CHECK(backend->retire_calls == 0);
}

void completion_and_ack_interleavings()
{
    for (unsigned order = 0; order < 24; ++order) {
        auto backend = std::make_shared<FakeBackend>();
        backend->immediate_publication = false;
        DependencyScheduler scheduler(backend);
        auto a = allocation(160), b = allocation(161);
        auto left = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
        auto right = scheduler.admit(call(2, NodeKind::Producer, {{b, AccessMode::Write}}));
        auto consumer = scheduler.admit(call(3, NodeKind::Consumer, {{a, AccessMode::Read}, {b, AccessMode::Read}}));
        scheduler.submit(left, launch); scheduler.submit(right, launch);
        auto left_event = scheduler.inspect(left).event;
        auto right_event = scheduler.inspect(right).event;
        std::array<unsigned, 4> actions{{0, 1, 2, 3}};
        for (unsigned i = 0; i < order; ++i) std::next_permutation(actions.begin(), actions.end());
        bool consumer_submitted = false;
        for (unsigned action : actions) {
            if (action == 0) backend->events.at(left_event).published = true;
            if (action == 1) backend->events.at(right_event).published = true;
            if (action == 2) backend->events.at(left_event).complete = true;
            if (action == 3) backend->events.at(right_event).complete = true;
            scheduler.poll();
            auto result = scheduler.submit(consumer, launch);
            if (result == SubmitResult::Submitted) consumer_submitted = true;
            const bool both_published = backend->events.at(left_event).published && backend->events.at(right_event).published;
            CHECK(consumer_submitted == both_published);
            CHECK(scheduler.snapshot().retained_bytes == 32);
        }
        backend->finish_all(); CHECK(scheduler.close() == CloseResult::Closed);
        CHECK(backend->record_calls == 3);
        CHECK(backend->retire_calls == 3);
    }
}

void slot_reuse_and_bounded_history()
{
    auto backend = std::make_shared<FakeBackend>();
    SchedulerConfig config; config.max_nodes = 2; config.max_allocation_identities = 2;
    DependencyScheduler scheduler(backend, config);
    auto a = allocation(170);
    for (std::uint64_t i = 1; i <= 1000; ++i) {
        auto id = scheduler.admit(call(i, NodeKind::Producer, {{a, AccessMode::Write}}));
        CHECK(scheduler.inspect(id).event.slot == 1);
        CHECK(scheduler.inspect(id).event.generation == i);
        scheduler.submit(id, launch); backend->finish_all(); scheduler.poll();
        CHECK(scheduler.snapshot().active_nodes == 0);
    }
    scheduler.admit(call(1001, NodeKind::Producer, {{allocation(171), AccessMode::Write}}));
    rejects([&] { scheduler.admit(call(1002, NodeKind::Producer, {{allocation(172), AccessMode::Write}})); });
    CHECK(scheduler.close() == CloseResult::Closed);
}

void admission_is_transactional_under_allocation_failure()
{
    bool reached_success = false;
    std::size_t failures = 0;
    for (std::ptrdiff_t failpoint = 0; failpoint < 256; ++failpoint) {
        auto backend = std::make_shared<FakeBackend>();
        DependencyScheduler scheduler(backend);
        auto a = allocation(180), b = allocation(181);
        auto original = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
        scheduler.submit(original, launch);
        auto next = call(2, NodeKind::Producer, {{a, AccessMode::Read}, {b, AccessMode::Write}});
        bool threw = false;
        NodeId next_id = 0;
        fail_allocation_after = failpoint;
        try { next_id = scheduler.admit(std::move(next)); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_allocation_after = -1;
        if (threw) {
            ++failures;
            const auto state = scheduler.snapshot();
            CHECK(state.active_nodes == 1);
            CHECK(state.retained_bytes == 16);
            CHECK(state.admitted == 1 && state.submitted == 1);
            CHECK(state.accepting && state.first_error.empty());
            // Same logical sequence, ID, stream turn and event slot remain reusable.
            next_id = scheduler.admit(call(2, NodeKind::Producer, {{a, AccessMode::Read}, {b, AccessMode::Write}}));
        }
        CHECK(next_id == 2);
        const auto info = scheduler.inspect(next_id);
        CHECK(info.stream == 2);
        CHECK(info.event.slot == 2 && info.event.generation == 1);
        CHECK(info.dependencies == std::vector<NodeId>{original});
        CHECK(scheduler.submit(next_id, launch) == SubmitResult::Submitted);
        backend->finish_all(); CHECK(scheduler.close() == CloseResult::Closed);
        CHECK(scheduler.snapshot().retained_bytes == 0);
        if (!threw) { reached_success = true; break; }
    }
    CHECK(reached_success);
    CHECK(failures >= 10);
}

void retirement_commit_does_not_allocate()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    auto a = allocation(190);
    auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
    auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
    scheduler.submit(writer, launch); scheduler.submit(reader, launch);
    backend->finish_all();
    bool threw = false;
    fail_allocation_after = 0;
    try { scheduler.poll(); } catch (const std::bad_alloc&) { threw = true; }
    const bool allocated = fail_allocation_after != 0;
    fail_allocation_after = -1;
    CHECK(!threw);
    CHECK(!allocated);
    CHECK(scheduler.snapshot().active_nodes == 0);
    CHECK(scheduler.snapshot().retained_bytes == 0);
    CHECK(backend->retire_calls == 2);
    CHECK(scheduler.close() == CloseResult::Closed);
}

void mutate_from_callback(DependencyScheduler& scheduler, NodeId id, unsigned operation)
{
    switch (operation) {
        case 0: scheduler.admit(call(9, NodeKind::Producer, {{allocation(201), AccessMode::Write}})); break;
        case 1: scheduler.submit(id, launch); break;
        case 2: scheduler.poll(); break;
        case 3: scheduler.close(); break;
        case 4: scheduler.cancel("nested cancellation"); break;
        default: throw std::logic_error("invalid reentry operation");
    }
}

void launch_callbacks_reject_reentrant_mutation()
{
    for (unsigned operation = 0; operation < 5; ++operation) {
        for (bool swallow : {false, true}) {
            auto backend = std::make_shared<FakeBackend>();
            DependencyScheduler scheduler(backend);
            auto id = scheduler.admit(call(1, NodeKind::Producer, {{allocation(200), AccessMode::Write}}));
            const auto result = scheduler.submit(id, [&](StreamId) {
                CHECK(scheduler.snapshot().active_nodes == 1); // read-only diagnostics are permitted
                CHECK(scheduler.inspect(id).id == id);
                try { mutate_from_callback(scheduler, id, operation); }
                catch (const std::logic_error&) { if (!swallow) throw; }
                return LaunchResult::Submitted;
            });
            CHECK(result == SubmitResult::Failed);
            CHECK(scheduler.snapshot().active_nodes == 1);
            CHECK(scheduler.snapshot().admitted == 1);
            CHECK(scheduler.inspect(id).state == NodeState::Quarantined);
            CHECK(scheduler.snapshot().first_error.find("reentrant") != std::string::npos);
            CHECK(scheduler.close() == CloseResult::Failed);
        }
    }
}

void backend_callbacks_reject_reentrant_mutation()
{
    for (const std::string phase : {"reserve", "record", "publication", "wait", "completion", "retirement"}) {
        for (unsigned operation = 0; operation < 5; ++operation) {
            auto backend = std::make_shared<FakeBackend>();
            DependencyScheduler scheduler(backend);
            auto a = allocation(210);
            auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
            auto reader = scheduler.admit(call(2, NodeKind::Consumer, {{a, AccessMode::Read}}));
            bool callback_ran = false;
            backend->callback = [&](const char* invoked) {
                if (phase != invoked) return;
                callback_ran = true;
                // Deliberately swallow it: outer operation must still fail closed.
                try { mutate_from_callback(scheduler, writer, operation); }
                catch (const std::logic_error&) {}
            };
            scheduler.submit(writer, launch);
            scheduler.submit(reader, launch);
            backend->finish_all(); scheduler.poll();
            CHECK(callback_ran);
            CHECK(!scheduler.snapshot().accepting);
            CHECK(scheduler.snapshot().admitted == 2);
            const auto original_error = scheduler.snapshot().first_error;
            CHECK(original_error.find("reentrant") != std::string::npos);
            backend->callback = {};
            CHECK(scheduler.close() == CloseResult::Failed);
            CHECK(scheduler.snapshot().first_error == original_error);
        }
    }
}

struct FixedBackendFailure final : std::exception {
    const char* what() const noexcept override
    { return "first backend failure must survive exhausted memory without allocation"; }
};

void failure_latching_does_not_allocate()
{
    for (const std::string phase : {"uncertain", "record", "publication", "completion", "retirement"}) {
        auto backend = std::make_shared<FakeBackend>();
        DependencyScheduler scheduler(backend);
        auto a = allocation(220);
        std::weak_ptr<void> weak = a.owner;
        auto writer = scheduler.admit(call(1, NodeKind::Producer, {{a, AccessMode::Write}}));
        auto independent = scheduler.admit(call(2, NodeKind::Producer, {{allocation(221), AccessMode::Write}}));
        a.owner.reset();
        backend->callback = [&](const char* invoked) {
            if (phase != invoked) return;
            fail_allocation_after = 0;
            throw FixedBackendFailure{};
        };
        bool allocation_escaped = false;
        try {
            const auto receipt = scheduler.submit(writer, [&](StreamId) {
                if (phase == "uncertain") {
                    fail_allocation_after = 0;
                    return LaunchResult::Uncertain;
                }
                return LaunchResult::Submitted;
            });
            if (phase == "completion" || phase == "retirement") {
                CHECK(receipt == SubmitResult::Submitted);
                if (phase == "retirement") backend->finish_all();
                scheduler.poll();
            }
        } catch (const std::bad_alloc&) { allocation_escaped = true; }
        const bool allocated = fail_allocation_after != 0;
        fail_allocation_after = -1;
        CHECK(!allocation_escaped);
        CHECK(!allocated);
        const auto state = scheduler.snapshot();
        CHECK(!state.accepting);
        CHECK(state.quarantined_nodes == 1);
        CHECK(!weak.expired());
        CHECK(state.first_error == (phase == "uncertain" ? "launch submission uncertain" : FixedBackendFailure{}.what()));
        bool independent_launched = false;
        const auto independent_result = scheduler.submit(independent, [&](StreamId) {
            independent_launched = true;
            return LaunchResult::Submitted;
        });
        // poll may already have retired this cancelled, never-launched node.
        CHECK(independent_result == SubmitResult::Rejected || independent_result == SubmitResult::AlreadySubmitted);
        CHECK(!independent_launched);
        rejects([&] { scheduler.admit(call(3, NodeKind::Producer, {{allocation(222), AccessMode::Write}})); });

        // Diagnostic rendering is allowed to allocate/fail; it cannot undo the latch.
        bool snapshot_failed = false;
        fail_allocation_after = 0;
        try { scheduler.snapshot(); } catch (const std::bad_alloc&) { snapshot_failed = true; }
        fail_allocation_after = -1;
        CHECK(snapshot_failed);
        backend->callback = {};
        scheduler.cancel("later cancellation must not replace the original failure");
        CHECK(scheduler.close() == CloseResult::Failed);
        CHECK(scheduler.snapshot().first_error == state.first_error);
    }
}

void failure_text_is_bounded_and_cancel_does_not_allocate()
{
    auto backend = std::make_shared<FakeBackend>();
    DependencyScheduler scheduler(backend);
    const std::string long_error(4096, 'q');
    bool allocation_escaped = false;
    fail_allocation_after = 0;
    try { scheduler.cancel(long_error); } catch (const std::bad_alloc&) { allocation_escaped = true; }
    const bool allocated = fail_allocation_after != 0;
    fail_allocation_after = -1;
    CHECK(!allocation_escaped);
    CHECK(!allocated);
    CHECK(!scheduler.snapshot().accepting);
    CHECK(scheduler.snapshot().first_error == long_error.substr(0, 1024));
    scheduler.cancel("another error");
    CHECK(scheduler.snapshot().first_error == long_error.substr(0, 1024));
    CHECK(scheduler.close() == CloseResult::Failed);
}

int main()
{
    try {
        hazards_and_aliases();
        producer_choice_and_fan_in();
        delayed_publication_is_not_completion();
        generation_and_late_ack();
        budgets_are_actual_nonzero_and_bounded();
        failure_and_cancel_keep_submitted_ownership();
        no_replay_and_close_late_completion();
        reserve_and_launch_failures();
        destroy_failure_never_retries_or_releases();
        single_caller_and_inference_only();
        logical_sequence_and_access_strength();
        entry_frontier_must_be_explicit();
        reject_two_generations_in_one_call();
        completed_outputs_remain_ready_in_the_domain();
        wait_and_query_failures_are_conservative();
        completion_before_ack_does_not_release_and_destructor_quarantines();
        completion_and_ack_interleavings();
        slot_reuse_and_bounded_history();
        retirement_commit_does_not_allocate();
        admission_is_transactional_under_allocation_failure();
        launch_callbacks_reject_reentrant_mutation();
        backend_callbacks_reject_reentrant_mutation();
        failure_latching_does_not_allocate();
        failure_text_is_bounded_and_cancel_does_not_allocate();
        std::cout << "PASS R26 native protocol: hazards, fan-in, publication, completion, "
                     "generations, budgets, cancellation, errors, owner and inference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
