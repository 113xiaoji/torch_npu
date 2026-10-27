#pragma once

#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace c10_npu::prefix {

// The production owner uses direct ACL APIs. In particular its destructor must
// not enqueue onto the repository that may currently be stopping.
struct DeviceEvent {
    virtual ~DeviceEvent() = default;
    virtual void Wait() = 0;
};

struct QueueRunToken final {
    explicit QueueRunToken(uint64_t value) : generation(value) {}
    const uint64_t generation;
    std::atomic<bool> running{false};
};

class PrefixNotRunning final : public std::runtime_error {
public:
    PrefixNotRunning() : std::runtime_error("FIFO prefix H2D generation is not RUN") {}
};

class Registry;

class Completion final {
public:
    template <typename WaitHooks>
    void WaitDevice(WaitHooks& hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        while (state_ == State::Pending) {
            WaitOnce(lock, hooks);
        }
        if (state_ == State::Failed) {
            std::rethrow_exception(error_);
        }
        auto event = event_;
        lock.unlock();

        // Host receipt and actual device completion are deliberately distinct.
        hooks.BeforeWait();
        std::exception_ptr wait_error;
        try {
            event->Wait();
        } catch (...) {
            wait_error = std::current_exception();
        }
        hooks.AfterWait();
        if (wait_error) {
            std::rethrow_exception(wait_error);
        }
    }

    void Fail(std::exception_ptr error)
    {
        if (!error) {
            throw std::invalid_argument("prefix failure requires an exception");
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (state_ == State::Pending) {
            state_ = State::Failed;
            error_ = std::move(error);
            cv_.notify_all();
        }
    }

    uint64_t Generation() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return generation_;
    }

    std::shared_ptr<const QueueRunToken> QueueToken() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_token_;
    }

    bool Published() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return published_;
    }

private:
    friend class Registry;
    enum class State { Pending, Recorded, Failed };

    explicit Completion(std::shared_ptr<DeviceEvent> event) : event_(std::move(event)) {}

    void MarkPublished(uint64_t generation, const std::shared_ptr<QueueRunToken>& token)
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (published_) {
            throw std::logic_error("prefix event was published more than once");
        }
        generation_ = generation;
        queue_token_ = token;
        published_ = true;
    }

    bool RecordResult(std::exception_ptr error)
    {
        if (error) {
            Fail(std::move(error));
            return false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (!published_) {
            throw std::logic_error("prefix record receipt preceded publication");
        }
        if (state_ == State::Pending) {
            state_ = State::Recorded;
            cv_.notify_all();
        }
        return state_ == State::Recorded;
    }

    template <typename WaitHooks>
    void WaitOnce(std::unique_lock<std::mutex>& lock, WaitHooks& hooks)
    {
        hooks.BeforeWait();
        std::exception_ptr wait_error;
        try {
            cv_.wait(lock);
        } catch (...) {
            wait_error = std::current_exception();
        }
        // A language-runtime restore may take its own lock. Never do that
        // while owning either the completion or registry mutex.
        if (lock.owns_lock()) {
            lock.unlock();
        }
        try {
            hooks.AfterWait();
        } catch (...) {
            lock.lock();
            throw;
        }
        lock.lock();
        if (wait_error) {
            std::rethrow_exception(wait_error);
        }
    }

    const std::shared_ptr<DeviceEvent> event_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    State state_ = State::Pending;
    std::exception_ptr error_;
    uint64_t generation_ = 0;
    std::shared_ptr<const QueueRunToken> queue_token_;
    bool published_ = false;
};

// Independent from queue class layouts. A failed event stays owned here until
// the existing queue lifecycle proves there are no raw parameters referencing
// it. Acquire also gives the consumer a lease across record/release callbacks.
class Registry final {
public:
    using Ticket = std::shared_ptr<Completion>;

    Registry() : stopped_error_(std::make_exception_ptr(std::runtime_error("prefix registry stopped"))) {}
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    ~Registry()
    {
        for (auto& item : events_) {
            item.second.ticket->Fail(stopped_error_);
        }
    }

    void StartQueue(const void* queue)
    {
        if (!queue) {
            throw std::invalid_argument("prefix queue identity is null");
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto& state = queues_[queue];
        if (state.accepting) {
            throw std::logic_error("prefix generation can only start after quiescence");
        }
        for (const auto& item : events_) {
            if (item.second.queue == queue) {
                throw std::logic_error("prefix generation still owns old queue events");
            }
        }
        if (state.generation == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("prefix generation overflow");
        }
        auto token = std::make_shared<QueueRunToken>(state.generation + 1);
        ++state.generation;
        state.token = std::move(token);
        state.accepting = true;
    }

    Ticket Register(const void* key, std::shared_ptr<DeviceEvent> event)
    {
        if (!key || !event) {
            throw std::invalid_argument("prefix requires a unique nonnull event");
        }
        Ticket ticket(new Completion(std::move(event)));
        std::lock_guard<std::mutex> lock(mu_);
        if (!events_.emplace(key, Entry{ticket, nullptr}).second) {
            throw std::logic_error("prefix event is already registered");
        }
        return ticket;
    }

    // Call only in the existing queue's copy/commit callback, after payload
    // copy succeeds. No map allocation occurs on this publication hot path.
    bool Published(const void* key, const void* queue, const QueueRunToken* expected = nullptr)
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto event = events_.find(key);
        if (event == events_.end()) {
            return false; // an ordinary RECORD_EVENT is not a prefix marker
        }
        auto owner = queues_.find(queue);
        if (owner == queues_.end() || !owner->second.accepting) {
            throw std::logic_error("prefix queue generation is not accepting publication");
        }
        if (expected != nullptr &&
            (owner->second.token.get() != expected ||
             !owner->second.token->running.load(std::memory_order_acquire))) {
            throw PrefixNotRunning();
        }
        event->second.ticket->MarkPublished(owner->second.generation, owner->second.token);
        event->second.queue = queue;
        return true;
    }

    void SetQueueRunning(const void* queue, bool running)
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto owner = queues_.find(queue);
        if (owner != queues_.end() && owner->second.token) {
            owner->second.token->running.store(running && owner->second.accepting, std::memory_order_release);
        }
    }

    Ticket Acquire(const void* key) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto event = events_.find(key);
        return event == events_.end() ? Ticket{} : event->second.ticket;
    }

    void Recorded(const void* key, const Ticket& lease, std::exception_ptr error = {})
    {
        if (!lease) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto event = events_.find(key);
        if (event != events_.end() && event->second.ticket == lease) {
            if (lease->RecordResult(std::move(error))) {
                events_.erase(event);
            }
        }
    }

    void FailQueue(const void* queue, std::exception_ptr error)
    {
        if (!error) {
            error = stopped_error_;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto owner = queues_.find(queue);
        if (owner == queues_.end()) {
            return;
        }
        owner->second.accepting = false;
        if (owner->second.token) {
            owner->second.token->running.store(false, std::memory_order_release);
        }
        for (auto& item : events_) {
            if (item.second.queue == queue) {
                item.second.ticket->Fail(error);
            }
        }
    }

    void Quiescent(const void* queue)
    {
        for (;;) {
            Ticket retired;
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto owner = queues_.find(queue);
                if (owner != queues_.end() && owner->second.accepting) {
                    throw std::logic_error("cannot retire prefix events from an accepting queue");
                }
                auto item = events_.begin();
                while (item != events_.end() && item->second.queue != queue) {
                    ++item;
                }
                if (item == events_.end()) {
                    return;
                }
                retired = std::move(item->second.ticket);
                events_.erase(item);
            }
            // Retire one owner outside the registry lock without allocating
            // teardown bookkeeping. Lifecycle seals publication throughout.
        }
    }

    void Abandon(const void* key, std::exception_ptr error)
    {
        Ticket retired;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto event = events_.find(key);
            if (event == events_.end()) {
                return;
            }
            event->second.ticket->Fail(error ? error : stopped_error_);
            if (!event->second.ticket->Published()) {
                retired = std::move(event->second.ticket);
                events_.erase(event);
            }
        }
    }

private:
    struct Entry { Ticket ticket; const void* queue; };
    struct QueueState { uint64_t generation = 0; bool accepting = false; std::shared_ptr<QueueRunToken> token; };
    const std::exception_ptr stopped_error_;
    mutable std::mutex mu_;
    std::unordered_map<const void*, Entry> events_;
    std::unordered_map<const void*, QueueState> queues_;
};

} // namespace c10_npu::prefix
