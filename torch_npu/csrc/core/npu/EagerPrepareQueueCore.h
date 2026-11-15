#pragma once

#include "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"
#include "torch_npu/csrc/core/npu/LockFreeMpscIngress.h"

#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#define C10_NPU_EAGER_PREPARE_MPSC_INGRESS 1

namespace c10_npu::queue {

// Native Prepare workers never access the original command FIFO. Only that
// FIFO's existing consumer calls Consume; Retire requires its device fence ACK.
// A Job must own all its arguments, and none of its callbacks may enter Python.
class EagerPrepareQueueCore final {
    struct State;

public:
    enum class IngressMode { SubmitMutex, LockFreeMpsc };
    struct Config {
        size_t slots = 64;
        size_t retained_bytes = 64 * 1024 * 1024;
        size_t workers = 2;
        IngressMode ingress = IngressMode::SubmitMutex;
    };
    struct Job {
        virtual ~Job() = default;
        virtual void Prepare() = 0;
        virtual void Execute() = 0;
        // Called after Prepare returns (without Execute), or after real device
        // retirement. Must release ALL budgeted owners even when the caller
        // keeps this shared Job wrapper. Successful cleanup must report zero
        // RetainedBytes; a destructor alone is not a resource-release ACK.
        virtual void Cancel() noexcept = 0;
        virtual size_t RetainedBytes() const noexcept = 0;
        virtual bool CleanupFailed() const noexcept { return false; }
    };
    enum class PublishState { NotAccepted, Committed, Indeterminate };
    struct PublishReceipt { PublishState state; std::exception_ptr error = nullptr; };
    struct Ticket {
        const uint64_t sequence;
    private:
        friend class EagerPrepareQueueCore;
        Ticket(uint64_t seq, std::shared_ptr<Job> owned, size_t bytes, const std::shared_ptr<State>& state)
            : sequence(seq), owner(state), job(std::move(owned)), retained_bytes(bytes) {}
        std::weak_ptr<State> owner;
        std::shared_ptr<Job> job;
        size_t retained_bytes;
        PublishState publication = PublishState::Indeterminate;
        bool published = false, prepare_ack = false, cancel_requested = false;
        bool consume_started = false, consumed = false, executing = false, execution_attempted = false;
        bool cleanup_active = false, cleanup_done = false, retiring = false, retired = false, quarantined = false;
        std::exception_ptr error;
    };
    using TicketPtr = std::shared_ptr<Ticket>;
    using PublishFn = std::function<PublishReceipt(const TicketPtr&)>;
    struct SnapshotData {
        size_t slots_in_use = 0, retained_bytes = 0, queued = 0, preparing = 0, ready = 0;
        uint64_t admitted = 0, retired = 0;
        uint64_t first_failed_sequence = std::numeric_limits<uint64_t>::max();
        bool closed = false, quarantined = false;
        size_t workers_started = 0;
        size_t peak_slots_in_use = 0, peak_retained_bytes = 0;
        size_t ingress_pending = 0, ingress_active_producers = 0, publisher_started = 0;
        bool ingress_atomic_lock_free = false, retained_core_mutex = true, submit_mutex_used = true;
    };

    explicit EagerPrepareQueueCore(Config config) : state_(std::make_shared<State>(config))
    {
        if (config.slots == 0 || config.retained_bytes == 0 || config.workers == 0) {
            throw std::invalid_argument("Eager Prepare configuration must be nonzero");
        }
        workers_->reserve(config.workers);
        try {
            for (size_t i = 0; i < config.workers; ++i) workers_->emplace_back([state = state_] { Worker(state); });
            if (config.ingress == IngressMode::LockFreeMpsc) *publisher_ = std::thread([state = state_] { Publisher(state); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(state_->state_mu); state_->stop_workers = true; state_->stop_publisher = true; state_->cv.notify_all(); }
            if (publisher_->joinable()) publisher_->join();
            for (auto& worker : *workers_) worker.join();
            throw;
        }
    }

    ~EagerPrepareQueueCore()
    {
        bool retain;
        {
            std::lock_guard<std::mutex> lock(state_->state_mu);
            state_->closed = true;
            state_->admission_gate.Close();
            retain = state_->quarantined || !state_->owners.empty() || state_->reserved_slots.load() != 0 ||
                state_->admission_gate.ActiveProducers() != 0 || state_->publisher_inflight != 0;
            if (retain) state_->quarantined = true;
            else { state_->stop_workers = true; state_->stop_publisher = true; }
            state_->cv.notify_all();
        }
        if (retain) {
            // SDK executors can borrow Prepare-worker TLS. Keep the threads
            // themselves alive, not just Jobs. Controller must end the process.
            // Storage was allocated during construction. Quarantine itself
            // needs no allocation and cannot destroy live std::thread objects.
            (void)workers_.release();
            (void)publisher_.release();
        } else {
            if (publisher_->joinable()) publisher_->join();
            for (auto& worker : *workers_) if (worker.joinable()) worker.join();
        }
    }
    EagerPrepareQueueCore(const EagerPrepareQueueCore&) = delete;
    EagerPrepareQueueCore& operator=(const EagerPrepareQueueCore&) = delete;

    TicketPtr Submit(std::shared_ptr<Job> job, PublishFn publish)
    {
        NoopWaitHooks hooks;
        return Submit(std::move(job), std::move(publish), hooks);
    }
    template <typename WaitHooks>
    TicketPtr Submit(std::shared_ptr<Job> job, PublishFn publish, WaitHooks& hooks)
    {
        // Detach before contending on submit_mu; reattach after releasing ALL
        // locks. Publish is native/detached; use noop or detach-aware FIFO hooks.
        hooks.BeforeWait();
        TicketPtr result;
        try { result = SubmitDetached(std::move(job), std::move(publish)); }
        catch (...) { const auto error = std::current_exception(); hooks.AfterWait(); std::rethrow_exception(error); }
        hooks.AfterWait();
        return result;
    }

    void Consume(const TicketPtr& ticket) { NoopWaitHooks hooks; Consume(ticket, hooks); }
    template <typename WaitHooks>
    void Consume(const TicketPtr& ticket, WaitHooks& hooks)
    {
        CheckTicket(ticket);
        std::shared_ptr<Job> job;
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(state_->state_mu);
            if (ticket->consume_started || ticket->retired) throw std::logic_error("Eager Prepare ticket already consumed");
            ticket->consume_started = true;
            while (!ticket->prepare_ack || !ticket->published || ticket->cleanup_active) WaitOnce(state_->cv, lock, hooks);
            if (ticket->cancel_requested && !ticket->error) FailLocked(*state_, ticket, MakeError("Eager Prepare ticket cancelled"));
            if (state_->first_error && state_->first_failed_sequence <= ticket->sequence) error = state_->first_error;
            else if (ticket->error) error = ticket->error;
            if (ticket->publication != PublishState::Committed || ticket->quarantined) {
                if (!error) error = MakeError("Eager Prepare publication quarantined or not committed");
            }
            if (error) { ticket->consumed = true; state_->cv.notify_all(); }
            else { ticket->executing = true; ticket->execution_attempted = true; job = ticket->job; }
        }
        if (error) std::rethrow_exception(error);
        try { job->Execute(); } catch (...) { error = std::current_exception(); }
        {
            std::lock_guard<std::mutex> lock(state_->state_mu);
            ticket->executing = false; ticket->consumed = true;
            if (error) FailLocked(*state_, ticket, error);
            state_->cv.notify_all();
        }
        if (error) std::rethrow_exception(error);
    }

    // Nonblocking request: a running Prepare owns its Job until its ACK.
    void Cancel(const TicketPtr& ticket)
    {
        CheckTicket(ticket);
        std::lock_guard<std::mutex> lock(state_->state_mu);
        if (!ticket->retired && !ticket->execution_attempted) ticket->cancel_requested = true;
        state_->cv.notify_all();
    }
    void CancelAndWait(const TicketPtr& ticket) { NoopWaitHooks hooks; CancelAndWait(ticket, hooks); }
    template <typename WaitHooks>
    void CancelAndWait(const TicketPtr& ticket, WaitHooks& hooks)
    {
        Cancel(ticket);
        bool cleanup = false;
        {
            std::unique_lock<std::mutex> lock(state_->state_mu);
            while (!ticket->prepare_ack || !ticket->published || ticket->executing || ticket->cleanup_active) {
                WaitOnce(state_->cv, lock, hooks);
            }
            if (ticket->quarantined) throw std::runtime_error("Eager Prepare ticket quarantined");
            if (ticket->retired) return;
            ticket->consume_started = true; ticket->consumed = true;
            if (!ticket->execution_attempted) {
                if (!ticket->error) FailLocked(*state_, ticket, MakeError("Eager Prepare ticket cancelled"));
                cleanup = ClaimCleanupLocked(ticket);
            }
        }
        if (cleanup) Cleanup(state_, ticket);
        { std::lock_guard<std::mutex> lock(state_->state_mu);
            if (ticket->quarantined) throw std::runtime_error("Eager Prepare cleanup quarantined"); }
    }

    // success=false means device retirement could not be proved. Keep every
    // owner, credit and worker TLS alive until the controller ends the process.
    void Retire(const TicketPtr& ticket, bool success = true)
    {
        CheckTicket(ticket);
        if (!success) { Quarantine(ticket, MakeError("Eager Prepare device retirement failed")); return; }
        bool cleanup;
        {
            std::lock_guard<std::mutex> lock(state_->state_mu);
            if (ticket->retired || ticket->retiring || ticket->quarantined) return;
            if (!ticket->prepare_ack || !ticket->published || !ticket->consumed || ticket->executing || ticket->cleanup_active) {
                throw std::logic_error("Eager Prepare cannot retire before host completion ACK");
            }
            cleanup = ClaimCleanupLocked(ticket);
        }
        if (cleanup) Cleanup(state_, ticket);
        ReleaseOwner(state_, ticket);
    }
    void Quarantine(const TicketPtr& ticket, std::exception_ptr error)
    {
        CheckTicket(ticket);
        std::lock_guard<std::mutex> lock(state_->state_mu);
        if (!ticket->retired && !ticket->retiring) QuarantineLocked(*state_, ticket, error ? error : MakeError("Eager Prepare quarantined"));
    }
    void Close()
    {
        state_->admission_gate.Close();
        std::lock_guard<std::mutex> lock(state_->state_mu);
        state_->closed = true; state_->cv.notify_all();
    }
    void Shutdown() { NoopWaitHooks hooks; Shutdown(hooks); }
    // Owner must serialize lifecycle calls. Quarantine refuses shutdown and
    // keeps worker TLS alive. Normal shutdown waits for every real retirement.
    template <typename WaitHooks>
    void Shutdown(WaitHooks& hooks)
    {
        Close();
        {
            std::unique_lock<std::mutex> lock(state_->state_mu);
            while ((!state_->owners.empty() || state_->pending.load() != 0 || state_->admission_gate.ActiveProducers() != 0 ||
                state_->publisher_inflight != 0) && !state_->quarantined) WaitOnce(state_->cv, lock, hooks);
            if (state_->quarantined) {
                while (!state_->work.empty() || state_->preparing != 0 || state_->pending.load() != 0 ||
                    state_->admission_gate.ActiveProducers() != 0 || state_->publisher_inflight != 0) WaitOnce(state_->cv, lock, hooks);
                throw std::runtime_error("Eager Prepare quarantined: live workers retained until process exit");
            }
            state_->stop_workers = true; state_->stop_publisher = true; state_->cv.notify_all();
        }
        hooks.BeforeWait();
        try {
            if (publisher_->joinable()) publisher_->join();
            for (auto& worker : *workers_) if (worker.joinable()) worker.join();
        }
        catch (...) { const auto error = std::current_exception(); hooks.AfterWait(); std::rethrow_exception(error); }
        hooks.AfterWait();
    }
    PublishState Publication(const TicketPtr& ticket) const
    {
        CheckTicket(ticket); std::lock_guard<std::mutex> lock(state_->state_mu); return ticket->publication;
    }
    // The original FIFO must call this at its ownership-copy ACK, before any
    // synchronous "wait until consumed" path. Waiting for PublishFn to return
    // would deadlock that consumer against this ticket's publication barrier.
    // This callback never takes submit_mu, invokes SDK code, or restores Python.
    bool MarkCommitted(const TicketPtr& ticket) noexcept
    {
        if (!ticket || ticket->owner.lock() != state_) return false;
        std::lock_guard<std::mutex> lock(state_->state_mu);
        if (ticket->published) return ticket->publication == PublishState::Committed;
        if (ticket->retired) return false;
        ticket->publication = PublishState::Committed;
        ticket->published = true;
        state_->cv.notify_all();
        return true;
    }
    void RethrowIfFailed() const
    {
        std::lock_guard<std::mutex> lock(state_->state_mu);
        if (state_->first_error) std::rethrow_exception(state_->first_error);
    }
    SnapshotData Snapshot() const
    {
        std::lock_guard<std::mutex> lock(state_->state_mu);
        size_t ready = 0;
        for (const auto& item : state_->owners) if (item.second->prepare_ack) ++ready;
        return {ReservedSlots(*state_), ReservedBytes(*state_), state_->work.size(), state_->preparing, ready,
            state_->admitted, state_->retired, state_->first_failed_sequence, state_->closed, state_->quarantined,
            state_->workers_started, state_->peak_slots_in_use, state_->peak_retained_bytes,
            state_->pending.load(), state_->admission_gate.ActiveProducers(), state_->publisher_started,
            state_->ingress && state_->ingress->IsLockFree() && state_->admission_gate.IsLockFree() &&
                state_->reserved_slots.is_lock_free() && state_->reserved_bytes.is_lock_free() && state_->pending.is_lock_free(),
            true, !state_->ingress};
    }

private:
    struct Request {
        std::shared_ptr<Job> job;
        PublishFn publish;
        size_t bytes = 0;
        TicketPtr ticket;
        std::exception_ptr error;
        bool transferred = false, done = false, quarantined = false;
    };
    using Ingress = LockFreeMpscIngress<Request>;
    struct State {
        explicit State(Config cfg) : config(cfg), ingress(cfg.ingress == IngressMode::LockFreeMpsc ? new Ingress(cfg.slots) : nullptr) {
            static_assert(std::atomic<size_t>::is_always_lock_free, "ingress requires lock-free credits");
        }
        const Config config;
        const std::unique_ptr<Ingress> ingress;
        std::atomic<size_t> reserved_slots{0}, reserved_bytes{0}, pending{0};
        LockFreeIngressGate admission_gate;
        std::mutex submit_mu;
        mutable std::mutex state_mu;
        std::condition_variable cv;
        std::deque<TicketPtr> work;
        std::unordered_map<uint64_t, TicketPtr> owners;
        size_t retained_bytes = 0, preparing = 0, workers_started = 0;
        size_t peak_slots_in_use = 0, peak_retained_bytes = 0;
        uint64_t admitted = 0, retired = 0;
        size_t publisher_started = 0, publisher_inflight = 0;
        uint64_t first_failed_sequence = std::numeric_limits<uint64_t>::max();
        std::exception_ptr first_error;
        bool closed = false, stop_workers = false, stop_publisher = false, quarantined = false;
    };
    static std::exception_ptr MakeError(const char* message) { return std::make_exception_ptr(std::runtime_error(message)); }
    static void RecordReservationPeakLocked(State& state)
    {
        const size_t slots = ReservedSlots(state), bytes = ReservedBytes(state);
        if (slots > state.peak_slots_in_use) state.peak_slots_in_use = slots;
        if (bytes > state.peak_retained_bytes) state.peak_retained_bytes = bytes;
    }
    void CheckTicket(const TicketPtr& ticket) const
    {
        if (!ticket || ticket->owner.lock() != state_) throw std::invalid_argument("foreign Eager Prepare ticket");
    }
    static void FailLocked(State& state, const TicketPtr& ticket, std::exception_ptr error)
    {
        if (!ticket->error) ticket->error = error;
        if (ticket->sequence < state.first_failed_sequence) {
            state.first_failed_sequence = ticket->sequence; state.first_error = ticket->error;
        }
        state.closed = true; state.admission_gate.Close(); state.cv.notify_all();
    }
    static void QuarantineLocked(State& state, const TicketPtr& ticket, std::exception_ptr error)
    {
        ticket->quarantined = true; state.quarantined = true; FailLocked(state, ticket, error);
    }
    static bool ClaimCleanupLocked(const TicketPtr& ticket)
    {
        if (ticket->cleanup_done || ticket->cleanup_active || ticket->quarantined) return false;
        ticket->cleanup_active = true; return true;
    }
    static void Cleanup(const std::shared_ptr<State>& state, const TicketPtr& ticket)
    {
        // No core/FIFO locks: native destructors may enter SDK allocators.
        ticket->job->Cancel();
        const bool cleanup_failed = ticket->job->CleanupFailed();
        const size_t remaining_bytes = ticket->job->RetainedBytes();
        const bool failed = cleanup_failed || remaining_bytes != 0;
        std::lock_guard<std::mutex> lock(state->state_mu);
        ticket->cleanup_active = false; ticket->cleanup_done = !failed;
        if (failed) QuarantineLocked(*state, ticket, MakeError(cleanup_failed ?
            "Eager Prepare cleanup failed" : "Eager Prepare cleanup retained bytes without a release ACK"));
        state->cv.notify_all();
    }
    static void ReleaseOwner(const std::shared_ptr<State>& state, const TicketPtr& ticket)
    {
        std::shared_ptr<Job> released;
        {
            std::lock_guard<std::mutex> lock(state->state_mu);
            if (ticket->retired || ticket->retiring || ticket->quarantined || !ticket->cleanup_done) return;
            ticket->retiring = true; released = std::move(ticket->job);
        }
        released.reset();
        {
            std::lock_guard<std::mutex> lock(state->state_mu);
            state->retained_bytes -= ticket->retained_bytes;
            if (state->ingress) {
                state->reserved_bytes.fetch_sub(ticket->retained_bytes);
                state->reserved_slots.fetch_sub(1);
            }
            ticket->retired = true; ticket->retiring = false;
            state->owners.erase(ticket->sequence); ++state->retired; state->cv.notify_all();
        }
    }
    static void ReleaseUnpublished(const std::shared_ptr<State>& state, const TicketPtr& ticket)
    {
        bool cleanup;
        {
            std::lock_guard<std::mutex> lock(state->state_mu);
            if (!ticket->published || ticket->publication != PublishState::NotAccepted ||
                !ticket->prepare_ack || ticket->quarantined || ticket->retired) return;
            ticket->consumed = true; cleanup = ClaimCleanupLocked(ticket);
        }
        if (cleanup) Cleanup(state, ticket);
        ReleaseOwner(state, ticket);
    }
    static size_t ReservedSlots(const State& state) {
        return state.ingress ? state.reserved_slots.load() : state.owners.size();
    }
    static size_t ReservedBytes(const State& state) {
        return state.ingress ? state.reserved_bytes.load() : state.retained_bytes;
    }
    static bool TryReserve(std::atomic<size_t>& counter, size_t amount, size_t limit) noexcept {
        size_t current = counter.load();
        do { if (amount > limit - current) return false; }
        while (!counter.compare_exchange_weak(current, current + amount));
        return true;
    }
    // Only lock-free atomics, bounded descriptor scanning, and noexcept moves.
    // All owners/callback allocations happened before this primitive. Separate
    // slot/byte reservations may transiently hold credits but never exceed caps.
    // Full/closed attempts complete; the blocking wrapper waits outside it.
    static typename Ingress::Node* TryEnqueue(const std::shared_ptr<State>& state,
        std::shared_ptr<Job>& job, PublishFn& publish, size_t bytes) noexcept
    {
        if (!state->admission_gate.TryEnter()) return nullptr;
        typename Ingress::Node* node = nullptr;
        node = state->ingress->TryClaim();
        if (node) {
            const bool slot = TryReserve(state->reserved_slots, 1, state->config.slots);
            const bool retained = slot && TryReserve(state->reserved_bytes, bytes, state->config.retained_bytes);
            if (retained) {
                auto& request = node->value;
                request.job = std::move(job); request.publish = std::move(publish); request.bytes = bytes;
                state->pending.fetch_add(1);
                state->ingress->Publish(node);
                state->admission_gate.Leave();
                return node;
            }
            if (retained) state->reserved_bytes.fetch_sub(bytes);
            if (slot) state->reserved_slots.fetch_sub(1);
            state->ingress->Release(node);
        }
        state->admission_gate.Leave();
        return nullptr;
    }
    static TicketPtr SubmitMpsc(const std::shared_ptr<State>& state, std::shared_ptr<Job> job,
        PublishFn publish, size_t bytes)
    {
        typename Ingress::Node* node;
        while (!(node = TryEnqueue(state, job, publish, bytes))) {
            std::unique_lock<std::mutex> lock(state->state_mu);
            state->cv.notify_all();
            if (state->admission_gate.Closed()) throw std::runtime_error("Eager Prepare admission closed");
            // Atomic credit returns and descriptor releases need not take this
            // mutex. A bounded fallback wait prevents a lost notification.
            state->cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return state->admission_gate.Closed() || (state->reserved_slots.load() < state->config.slots &&
                    bytes <= state->config.retained_bytes - state->reserved_bytes.load() && state->ingress->HasFree());
            });
        }
        TicketPtr result; std::exception_ptr error; bool quarantined;
        {
            std::unique_lock<std::mutex> lock(state->state_mu);
            RecordReservationPeakLocked(*state);
            state->cv.notify_all();
            // Wait for the whole publisher callback, preserving the old Submit
            // exception/lifetime contract. Copy ACK still independently unblocks
            // a synchronous original-FIFO consumer before callback return.
            state->cv.wait(lock, [&] { return node->value.done; });
            result = node->value.ticket; error = node->value.error; quarantined = node->value.quarantined;
        }
        if (!quarantined) {
            node->value = Request{}; // Native owners/destructors run without locks.
            state->ingress->Release(node);
            std::lock_guard<std::mutex> lock(state->state_mu); state->cv.notify_all();
        }
        if (error) std::rethrow_exception(error);
        return result;
    }
    static void Publisher(const std::shared_ptr<State>& state)
    {
        { std::lock_guard<std::mutex> lock(state->state_mu); ++state->publisher_started; state->cv.notify_all(); }
        for (;;) {
            auto* batch = state->ingress->TakeBatch();
            if (!batch) {
                std::unique_lock<std::mutex> lock(state->state_mu);
                if (state->stop_publisher && state->admission_gate.ActiveProducers() == 0 && state->ingress->Empty()) return;
                // Producers never take state_mu before publishing. In C++17 a
                // timed idle wait also handles preemption after the enqueue CAS
                // but before the wrapper's notification. No spinning SDK thread.
                state->cv.wait_for(lock, std::chrono::milliseconds(1));
                continue;
            }
            while (batch) {
                auto* node = batch; batch = batch->next; // Save before ACK permits reuse.
                auto& request = node->value;
                { std::lock_guard<std::mutex> lock(state->state_mu); ++state->publisher_inflight; }
                try {
                    request.ticket = AdmitAndPublish(state, request.job, request.publish, request.bytes, &request);
                    ReleaseUnpublished(state, request.ticket);
                } catch (...) {
                    request.error = std::current_exception();
                    if (!request.transferred) {
                        request.job->Cancel();
                        const bool failed = request.job->CleanupFailed() || request.job->RetainedBytes() != 0;
                        std::lock_guard<std::mutex> lock(state->state_mu);
                        state->pending.fetch_sub(1);
                        if (failed) { request.quarantined = true; state->quarantined = true; }
                        else {
                            state->reserved_bytes.fetch_sub(request.bytes);
                            state->reserved_slots.fetch_sub(1);
                        }
                        if (!state->first_error) {
                            state->first_error = request.error; state->first_failed_sequence = state->admitted;
                        }
                        state->closed = true; state->admission_gate.Close();
                    } else {
                        std::lock_guard<std::mutex> lock(state->state_mu);
                        QuarantineLocked(*state, request.ticket, request.error);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(state->state_mu);
                    request.done = true; --state->publisher_inflight; state->cv.notify_all();
                }
            }
        }
    }
    TicketPtr SubmitDetached(std::shared_ptr<Job> job, PublishFn publish)
    {
        if (!job || !publish) throw std::invalid_argument("Eager Prepare needs an owned Job and publisher");
        const size_t initial_bytes = job->RetainedBytes();
        if (initial_bytes > state_->config.retained_bytes) throw std::length_error("Eager Prepare initial byte limit exceeded");
        if (state_->ingress) return SubmitMpsc(state_, std::move(job), std::move(publish), initial_bytes);
        std::unique_lock<std::mutex> submit_lock(state_->submit_mu);
        auto ticket = AdmitAndPublish(state_, std::move(job), publish, initial_bytes, nullptr);
        submit_lock.unlock(); ReleaseUnpublished(state_, ticket); return ticket;
    }
    static TicketPtr AdmitAndPublish(const std::shared_ptr<State>& state, std::shared_ptr<Job> job,
        PublishFn& publish, size_t initial_bytes, Request* request)
    {
        TicketPtr ticket;
        {
            std::unique_lock<std::mutex> lock(state->state_mu);
            if (!request) {
                state->cv.wait(lock, [&] {
                    return state->closed || (state->owners.size() < state->config.slots &&
                        initial_bytes <= state->config.retained_bytes - state->retained_bytes);
                });
                if (state->closed) throw std::runtime_error("Eager Prepare admission closed");
            } else if (state->first_error) {
                std::rethrow_exception(state->first_error);
            }
            // Reserved requests admitted before Close are drained, not dropped.
            ticket.reset(new Ticket(state->admitted, std::move(job), initial_bytes, state));
            state->owners.emplace(ticket->sequence, ticket);
            try { state->work.push_back(ticket); }
            catch (...) { state->owners.erase(ticket->sequence); throw; }
            ++state->admitted; state->retained_bytes += initial_bytes;
            if (request) {
                request->ticket = ticket; request->transferred = true;
                state->pending.fetch_sub(1);
            }
            RecordReservationPeakLocked(*state); state->cv.notify_all();
        }
        // Work is visible BEFORE publication, so a full FIFO cannot prevent its
        // own head from finishing Prepare. Workers never enqueue or drain it.
        PublishReceipt receipt{PublishState::Indeterminate, nullptr};
        try { receipt = publish(ticket); } catch (...) { receipt.error = std::current_exception(); }
        {
            std::lock_guard<std::mutex> lock(state->state_mu);
            if (ticket->published && ticket->publication == PublishState::Committed) {
                // The FIFO's ACK is irreversible. Its consumer may even have
                // completed and retired this ticket while PublishFn was still
                // waiting. A later exception/receipt must not cancel, replay,
                // quarantine solely for unknown ownership, or revive its owner.
                if (receipt.state != PublishState::Committed && !receipt.error) {
                    receipt.error = MakeError("Eager Prepare publisher contradicted an already committed ownership ACK");
                }
                receipt.state = PublishState::Committed;
            }
            ticket->publication = receipt.state; ticket->published = true;
            if (receipt.state == PublishState::NotAccepted) {
                ticket->cancel_requested = true;
                FailLocked(*state, ticket, receipt.error ? receipt.error : MakeError("Eager Prepare publication not accepted"));
            } else if (receipt.state == PublishState::Indeterminate) {
                QuarantineLocked(*state, ticket, receipt.error ? receipt.error : MakeError("Eager Prepare publication indeterminate"));
            } else if (receipt.error) FailLocked(*state, ticket, receipt.error);
            state->cv.notify_all();
        }
        return ticket;
    }
    static void Worker(const std::shared_ptr<State>& state)
    {
        {
            std::lock_guard<std::mutex> lock(state->state_mu);
            ++state->workers_started;
            state->cv.notify_all();
        }
        for (;;) {
            TicketPtr ticket; bool skip;
            {
                std::unique_lock<std::mutex> lock(state->state_mu);
                state->cv.wait(lock, [&] { return state->stop_workers || !state->work.empty(); });
                if (state->stop_workers && state->work.empty()) return;
                ticket = std::move(state->work.front()); state->work.pop_front(); ++state->preparing;
                skip = ticket->cancel_requested;
            }
            std::exception_ptr error;
            if (!skip) { try { ticket->job->Prepare(); } catch (...) { error = std::current_exception(); } }
            const size_t actual_bytes = ticket->job->RetainedBytes();
            bool cleanup = false;
            {
                std::lock_guard<std::mutex> lock(state->state_mu);
                if (actual_bytes > ticket->retained_bytes) {
                    const size_t extra = actual_bytes - ticket->retained_bytes;
                    const bool reserved = state->ingress ? TryReserve(state->reserved_bytes, extra, state->config.retained_bytes) :
                        extra <= state->config.retained_bytes - state->retained_bytes;
                    if (!reserved) {
                        if (!error) error = MakeError("Eager Prepare post-query byte limit exceeded");
                    } else {
                        state->retained_bytes += extra; ticket->retained_bytes = actual_bytes;
                        RecordReservationPeakLocked(*state);
                    }
                }
                if (ticket->cancel_requested && !error) error = MakeError("Eager Prepare ticket cancelled");
                if (error) { FailLocked(*state, ticket, error); cleanup = ClaimCleanupLocked(ticket); }
            }
            if (cleanup) Cleanup(state, ticket);
            {
                std::lock_guard<std::mutex> lock(state->state_mu);
                ticket->prepare_ack = true; --state->preparing; state->cv.notify_all();
            }
            ReleaseUnpublished(state, ticket);
        }
    }
    template <typename WaitHooks>
    static void WaitOnce(std::condition_variable& cv, std::unique_lock<std::mutex>& lock, WaitHooks& hooks)
    {
        hooks.BeforeWait();
        std::exception_ptr error;
        try { cv.wait(lock); } catch (...) { error = std::current_exception(); }
        // Same runtime lock ordering as OrderedMpscQueueCore: never reattach
        // while holding state/submit/SDK/FIFO locks, including exceptional paths.
        lock.unlock();
        try { hooks.AfterWait(); } catch (...) { lock.lock(); throw; }
        lock.lock();
        if (error) std::rethrow_exception(error);
    }
    const std::shared_ptr<State> state_;
    std::unique_ptr<std::vector<std::thread>> workers_ = std::make_unique<std::vector<std::thread>>();
    std::unique_ptr<std::thread> publisher_ = std::make_unique<std::thread>();
};

} // namespace c10_npu::queue
