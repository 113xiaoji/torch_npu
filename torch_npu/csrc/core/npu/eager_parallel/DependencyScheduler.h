#pragma once

// SDK-independent R26 protocol core. This file does not qualify any PTA route.
// All mutation belongs to the constructing caller; backend queries must be bounded,
// nonblocking. Backend and launch callbacks may inspect diagnostics but cannot mutate
// this scheduler; attempts throw before state changes and poison the external call,
// even if the callback swallows that exception. No scheduler lock is held across
// backend operations. Actual allocator/stream/SDK guarantees belong to an adapter.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace c10_npu {
namespace eager_parallel {

using StreamId = std::uint64_t;
using NodeId = std::uint64_t;

struct AllocationToken {
    std::uint64_t identity = 0;
    std::uint64_t generation = 0;
    bool operator<(const AllocationToken& other) const noexcept
    { return std::tie(identity, generation) < std::tie(other.identity, other.generation); }
};

struct EventToken {
    std::uint64_t domain = 0;
    std::uint64_t slot = 0;
    std::uint64_t generation = 0;
    bool operator<(const EventToken& other) const noexcept
    { return std::tie(domain, slot, generation) < std::tie(other.domain, other.slot, other.generation); }
    bool operator==(const EventToken& other) const noexcept
    { return domain == other.domain && slot == other.slot && generation == other.generation; }
};

enum class EntryReadiness { NeedsFence, DeviceComplete, NewAllocation };

struct AllocationLease {
    AllocationToken token;
    std::size_t bytes = 0; // full allocation size, never merely a view's logical nbytes
    // Must pin the actual allocation/DataPtr through mutations, not only StorageImpl.
    std::shared_ptr<void> owner;
    EntryReadiness entry_readiness = EntryReadiness::NeedsFence;
};

enum class AccessMode { Read, Write };
struct Access { AllocationLease lease; AccessMode mode = AccessMode::Read; };
enum class NodeKind { Producer, Consumer, EntryFence };

struct FrozenCall {
    std::uint64_t logical_sequence = 0;
    NodeKind kind = NodeKind::Consumer;
    std::vector<Access> accesses;
    // Workspace, frozen arguments and any other native owners not in accesses.
    std::size_t retained_bytes = 0;
    std::vector<std::shared_ptr<void>> retained;
    bool inference = true;
};

enum class Publication { Pending, Published, Failed };
enum class Completion { Pending, Complete, Failed };

class EventBackend {
public:
    virtual ~EventBackend() = default;
    // Reserve before launch. Throwing here MUST mean no event or device work exists.
    virtual void reserve_event(EventToken, StreamId) = 0;
    // Return may precede record's execution by a host submission queue.
    virtual void request_record(EventToken, StreamId) = 0;
    // Published means record actually reached the device stream, not copied to a queue.
    virtual Publication publication(EventToken) = 0;
    virtual void enqueue_wait(StreamId, EventToken) = 0;
    virtual Completion device_complete(EventToken) = 0;
    // Called once only, after completion AND all dependent device uses have retired,
    // or for an unused reserved event with a definite no-submit receipt.
    virtual void retire_event(EventToken) = 0;
};

enum class LaunchResult { Submitted, RejectedBeforeSubmit, Uncertain };
enum class SubmitResult { Submitted, PublicationPending, AlreadySubmitted, Rejected, Failed };
enum class CloseResult { Closed, Pending, Failed };
enum class NodeState { Admitted, Submitted, Published, DeviceComplete, Cancelled, Quarantined, Retired };

struct SchedulerConfig {
    StreamId consumer_stream = 0;
    std::array<StreamId, 2> producer_streams{{1, 2}};
    std::size_t max_nodes = 64;
    std::size_t max_retained_bytes = 512u * 1024u * 1024u;
    std::size_t max_allocation_identities = 4096;
};

struct NodeInfo {
    NodeId id = 0;
    std::uint64_t logical_sequence = 0;
    StreamId stream = 0;
    EventToken event;
    NodeState state = NodeState::Admitted;
    std::vector<NodeId> dependencies;
};

struct SchedulerSnapshot {
    std::size_t active_nodes = 0;
    std::size_t retained_bytes = 0;
    std::size_t quarantined_nodes = 0;
    std::uint64_t admitted = 0;
    std::uint64_t submitted = 0;
    std::uint64_t retired = 0;
    bool accepting = true;
    std::string first_error;
};

class DependencyScheduler {
public:
    explicit DependencyScheduler(std::shared_ptr<EventBackend> backend, SchedulerConfig config = {})
        : backend_(std::move(backend)), config_(config), owner_(std::this_thread::get_id()),
          domain_(next_domain_.fetch_add(1, std::memory_order_relaxed))
    {
        if (!backend_ || config_.max_nodes == 0 || config_.max_retained_bytes == 0 ||
            config_.max_allocation_identities == 0 ||
            config_.producer_streams[0] == config_.producer_streams[1] ||
            config_.consumer_stream == config_.producer_streams[0] ||
            config_.consumer_stream == config_.producer_streams[1] || domain_ == 0) {
            throw std::invalid_argument("invalid R26 scheduler configuration");
        }
        // Retirement must not allocate after a successful backend event destroy.
        // A slot is either live or free, so both vectors are bounded by max_nodes.
        event_generations_.reserve(config_.max_nodes);
        free_event_slots_.reserve(config_.max_nodes);
    }

    DependencyScheduler(const DependencyScheduler&) = delete;
    DependencyScheduler& operator=(const DependencyScheduler&) = delete;

    ~DependencyScheduler()
    {
        if (!nodes_.empty()) {
            // Destructor cannot establish SDK completion. Preserve all remaining native
            // owners and backend/event handles for process lifetime instead of unsafe free.
            // Explicit close/poll is required for ordinary retirement. A production adapter
            // may provide a separately proven service to recover this quarantine.
            auto retained = std::make_unique<Abandoned>(Abandoned{std::move(backend_), std::move(nodes_)});
            // Reachable process-lifetime storage keeps deliberate quarantine visible to
            // sanitizers. The short registry lock never encloses a backend call or wait.
            static auto* quarantine_mutex = new std::mutex;
            static auto* quarantines = new std::vector<std::unique_ptr<Abandoned>>;
            std::lock_guard<std::mutex> guard(*quarantine_mutex);
            quarantines->push_back(std::move(retained));
        }
    }

    NodeId admit(FrozenCall call)
    {
        check_mutation_entry();
        observe_foreign_caller();
        if (!accepting_) throw std::runtime_error("R26 admission stopped: " +
            std::string(first_error_.data(), first_error_size_));
        if (!call.inference) throw std::invalid_argument("R26 only admits qualified inference calls");
        if (call.logical_sequence == 0 || call.logical_sequence <= last_sequence_)
            throw std::invalid_argument("logical sequence must strictly increase");
        if (nodes_.size() >= config_.max_nodes) throw std::runtime_error("R26 node budget exhausted");
        if (!call.retained.empty() && call.retained_bytes == 0)
            throw std::invalid_argument("native retained owners require a nonzero byte charge");
        for (const auto& owner : call.retained)
            if (!owner) throw std::invalid_argument("null retained native owner");

        std::map<AllocationToken, Access> normalized;
        std::map<std::uint64_t, std::uint64_t> call_generations;
        for (const auto& access : call.accesses) {
            const auto& lease = access.lease;
            if (!lease.owner || lease.bytes == 0 || lease.token.identity == 0 || lease.token.generation == 0)
                throw std::invalid_argument("allocation lease requires identity, generation, bytes and actual owner");
            const auto call_generation = call_generations.emplace(lease.token.identity, lease.token.generation);
            if (!call_generation.second && call_generation.first->second != lease.token.generation)
                throw std::invalid_argument("one call cannot use two generations of one allocation identity");
            auto result = normalized.emplace(lease.token, access);
            if (!result.second) {
                if (result.first->second.lease.bytes != lease.bytes)
                    throw std::invalid_argument("views must report the same whole-allocation size");
                if (access.mode == AccessMode::Write) result.first->second.mode = AccessMode::Write;
            }
        }
        if (normalized.empty() && call.retained_bytes == 0)
            throw std::invalid_argument("device call has no accounted native resources");

        std::size_t extra = call.retained_bytes;
        std::set<NodeId> dependencies;
        std::set<std::uint64_t> new_identities;
        for (const auto& entry : normalized) {
            const auto& key = entry.first;
            const auto& access = entry.second;
            const auto generation = latest_generations_.find(key.identity);
            if (generation != latest_generations_.end() && key.generation < generation->second)
                throw std::invalid_argument("stale allocation generation");
            if (generation == latest_generations_.end()) new_identities.insert(key.identity);
            for (const auto& active : allocations_)
                if (active.first.identity == key.identity && active.first.generation != key.generation)
                    throw std::invalid_argument("allocation generation reused before completion");
            const auto allocation = allocations_.find(key);
            if (allocation == allocations_.end()) {
                // With no active references, the same tracked generation was already
                // positively completed in this accepting domain. Reusing its original
                // NewAllocation/NeedsFence descriptor does not erase that proof.
                const bool completed_in_domain = generation != latest_generations_.end() &&
                    generation->second == key.generation;
                if (!completed_in_domain && call.kind != NodeKind::EntryFence &&
                    access.lease.entry_readiness != EntryReadiness::DeviceComplete &&
                    !(access.mode == AccessMode::Write &&
                      access.lease.entry_readiness == EntryReadiness::NewAllocation))
                    throw std::invalid_argument("untracked allocation requires a domain-entry frontier");
                extra = checked_add(extra, access.lease.bytes);
            } else {
                if (allocation->second.bytes != access.lease.bytes)
                    throw std::invalid_argument("whole-allocation size changed within generation");
                add_dependency(dependencies, allocation->second.writer);
                if (access.mode == AccessMode::Write)
                    for (NodeId reader : allocation->second.readers) add_dependency(dependencies, reader);
            }
        }
        if (new_identities.size() > config_.max_allocation_identities - latest_generations_.size())
            throw std::runtime_error("R26 allocation identity budget exhausted; close the domain");
        if (extra > config_.max_retained_bytes - retained_bytes_)
            throw std::runtime_error("R26 retained byte budget exhausted");

        if (last_id_ == std::numeric_limits<NodeId>::max())
            throw std::overflow_error("node identity exhausted");
        const bool producer = call.kind == NodeKind::Producer;
        const auto sequence = call.logical_sequence;
        Node node;
        node.info.id = last_id_ + 1;
        node.info.logical_sequence = call.logical_sequence;
        node.info.stream = producer ?
            config_.producer_streams[producer_turn_ % config_.producer_streams.size()] : config_.consumer_stream;
        node.info.event = next_event();
        node.info.dependencies.assign(dependencies.begin(), dependencies.end());
        call.accesses.clear();
        for (const auto& entry : normalized) call.accesses.push_back(entry.second);
        node.call = std::move(call);
        const auto id = node.info.id;
        const auto event = node.info.event;

        // Build the complete registration transaction before changing live state.
        // Allocation failure preserves every existing node, dependency, owner,
        // budget, sequence, producer turn and event generation. Copying this bounded
        // graph is deliberately conservative; this CPU core makes no hot-path claim.
        auto next_nodes = nodes_;
        auto next_allocations = allocations_;
        auto next_generations = latest_generations_;
        next_nodes.emplace(id, std::move(node));
        for (NodeId dependency : dependencies) ++next_nodes.at(dependency).dependents;
        for (const auto& entry : normalized) {
            next_generations[entry.first.identity] = entry.first.generation;
            auto& allocation = next_allocations[entry.first];
            allocation.bytes = entry.second.lease.bytes;
            ++allocation.references;
            if (entry.second.mode == AccessMode::Write) {
                allocation.writer = id;
                allocation.readers.clear();
            } else {
                allocation.readers.insert(id);
            }
        }
        static_assert(noexcept(nodes_.swap(next_nodes)), "node commit must not throw");
        static_assert(noexcept(allocations_.swap(next_allocations)), "allocation commit must not throw");
        static_assert(noexcept(latest_generations_.swap(next_generations)), "generation commit must not throw");
        nodes_.swap(next_nodes);
        allocations_.swap(next_allocations);
        latest_generations_.swap(next_generations);
        commit_event(event);
        retained_bytes_ += extra;
        last_sequence_ = sequence;
        last_id_ = id;
        if (producer) ++producer_turn_;
        ++admitted_;
        return id;
    }

    SubmitResult submit(NodeId id, const std::function<LaunchResult(StreamId)>& launch)
    {
        check_mutation_entry();
        observe_foreign_caller();
        auto found = nodes_.find(id);
        if (found == nodes_.end()) {
            if (id > 0 && id <= last_id_) return SubmitResult::AlreadySubmitted;
            throw std::invalid_argument("unknown node id");
        }
        auto& node = found->second;
        if (node.launch_attempted) return SubmitResult::AlreadySubmitted;
        if (!accepting_ || node.info.state == NodeState::Cancelled) return SubmitResult::Rejected;
        std::map<StreamId, NodeId> waits;
        for (NodeId predecessor : node.info.dependencies) {
            auto& dependency = nodes_.at(predecessor);
            refresh_publication(dependency);
            if (has_error_) return SubmitResult::Rejected;
            if (!dependency.published) return SubmitResult::PublicationPending;
            if (dependency.info.stream == node.info.stream) continue;
            auto existing = waits.find(dependency.info.stream);
            if (existing == waits.end() || nodes_.at(existing->second).submission_order < dependency.submission_order)
                waits[dependency.info.stream] = predecessor;
        }
        try {
            external_call([&] { backend_->reserve_event(node.info.event, node.info.stream); });
            node.reserved = true;
        } catch (const std::exception& error) {
            if (callback_violation_) {
                // A backend could swallow reentry rejection and allocate an event
                // before returning. Its reservation state is then untrusted.
                node.may_have_device_work = true;
                quarantine(node, error.what());
            } else fail(error.what());
            return SubmitResult::Failed;
        } catch (...) {
            if (callback_violation_) {
                node.may_have_device_work = true;
                quarantine(node, "reentrant callback left event reservation uncertain");
            } else fail("event reserve failed");
            return SubmitResult::Failed;
        }
        try {
            for (const auto& wait : waits) {
                // A throwing SDK wait may already have queued work. Retain this node
                // and its predecessor events conservatively until proven completion.
                node.may_have_device_work = true;
                external_call([&] { backend_->enqueue_wait(node.info.stream, nodes_.at(wait.second).info.event); });
            }
        } catch (const std::exception& error) {
            quarantine(node, error.what()); return SubmitResult::Failed;
        } catch (...) {
            quarantine(node, "event wait failed"); return SubmitResult::Failed;
        }
        node.launch_attempted = true;
        LaunchResult receipt = LaunchResult::Uncertain;
        try {
            receipt = external_call([&] { return launch(node.info.stream); });
        } catch (const std::exception& error) {
            node.may_have_device_work = true;
            quarantine(node, error.what()); return SubmitResult::Failed;
        } catch (...) {
            node.may_have_device_work = true;
            quarantine(node, "launch threw without a definite no-submit receipt"); return SubmitResult::Failed;
        }
        if (receipt == LaunchResult::RejectedBeforeSubmit) {
            if (node.may_have_device_work) {
                // Successful queued waits still refer to predecessor event handles.
                // Without a recorded completion fence they cannot be retired here.
                quarantine(node, "launch rejected after stream waits");
            } else {
                fail("launch rejected before submission");
                node.info.state = NodeState::Cancelled;
            }
            return SubmitResult::Failed;
        }
        node.may_have_device_work = true;
        if (receipt != LaunchResult::Submitted) {
            quarantine(node, "launch submission uncertain"); return SubmitResult::Failed;
        }
        node.submission_order = ++submitted_;
        node.info.state = NodeState::Submitted;
        // Set before the call: a throwing/late record can still acquire a valid ACK.
        node.record_requested = true;
        try {
            external_call([&] { backend_->request_record(node.info.event, node.info.stream); });
        } catch (const std::exception& error) {
            quarantine(node, error.what()); return SubmitResult::Failed;
        } catch (...) {
            quarantine(node, "event record failed"); return SubmitResult::Failed;
        }
        refresh_publication(node);
        return has_error_ ? SubmitResult::Failed : SubmitResult::Submitted;
    }

    void poll()
    {
        check_mutation_entry();
        observe_foreign_caller();
        for (auto& entry : nodes_) {
            auto& node = entry.second;
            if (!node.record_requested || node.device_complete || node.retire_attempted) continue;
            refresh_publication(node);
            if (!node.published) continue;
            try {
                const auto completion = external_call([&] { return backend_->device_complete(node.info.event); });
                if (completion == Completion::Complete) {
                    node.device_complete = true;
                    node.info.state = NodeState::DeviceComplete;
                } else if (completion == Completion::Failed) {
                    quarantine(node, "device completion query failed");
                }
            } catch (const std::exception& error) {
                quarantine(node, error.what());
            } catch (...) {
                quarantine(node, "device completion query threw");
            }
        }
        retire_ready();
    }

    void cancel(std::string_view reason = "domain cancelled")
    {
        check_mutation_entry();
        observe_foreign_caller();
        fail(reason);
        retire_ready();
    }

    // One bounded pass only. Pending requires another owner poll/close after service
    // progress. Failed does NOT mean resources are safe to release: inspect active_nodes.
    CloseResult close()
    {
        check_mutation_entry();
        accepting_ = false;
        cancel_unsubmitted();
        poll();
        if (has_error_) return CloseResult::Failed;
        return nodes_.empty() ? CloseResult::Closed : CloseResult::Pending;
    }

    NodeInfo inspect(NodeId id) const
    {
        check_owner();
        const auto found = nodes_.find(id);
        if (found != nodes_.end()) return found->second.info;
        if (id == 0 || id > last_id_) throw std::invalid_argument("unknown node id");
        NodeInfo retired; retired.id = id; retired.state = NodeState::Retired;
        return retired;
    }

    SchedulerSnapshot snapshot() const
    {
        check_owner();
        SchedulerSnapshot result;
        result.active_nodes = nodes_.size();
        result.retained_bytes = retained_bytes_;
        result.admitted = admitted_;
        result.submitted = submitted_;
        result.retired = retired_;
        result.accepting = accepting_ && !foreign_caller_.load(std::memory_order_acquire);
        // Formatting may allocate; failure cannot undo the allocation-free error latch.
        result.first_error.assign(first_error_.data(), first_error_size_);
        for (const auto& entry : nodes_)
            if (entry.second.info.state == NodeState::Quarantined) ++result.quarantined_nodes;
        return result;
    }

private:
    struct AllocationState {
        std::size_t bytes = 0;
        std::size_t references = 0;
        NodeId writer = 0;
        std::set<NodeId> readers;
    };
    struct Node {
        NodeInfo info;
        FrozenCall call;
        std::size_t dependents = 0;
        std::uint64_t submission_order = 0;
        bool reserved = false;
        bool launch_attempted = false;
        bool may_have_device_work = false;
        bool record_requested = false;
        bool published = false;
        bool device_complete = false;
        bool retire_attempted = false;
    };
    struct Abandoned {
        std::shared_ptr<EventBackend> backend;
        std::map<NodeId, Node> nodes;
    };

    void check_owner() const
    {
        if (std::this_thread::get_id() != owner_) {
            // Foreign callers never touch owner-managed state, even on error paths.
            foreign_caller_.store(true, std::memory_order_release);
            throw std::runtime_error("R26 owner affinity violation; owner must drain before fallback");
        }
    }
    void check_mutation_entry()
    {
        check_owner();
        if (in_external_callback_) {
            callback_violation_ = true;
            throw std::logic_error("R26 reentrant scheduler mutation from backend or launch callback");
        }
    }
    struct CallbackScope {
        DependencyScheduler& scheduler;
        explicit CallbackScope(DependencyScheduler& value) noexcept : scheduler(value)
        {
            scheduler.in_external_callback_ = true;
            scheduler.callback_violation_ = false;
        }
        ~CallbackScope() noexcept { scheduler.in_external_callback_ = false; }
        CallbackScope(const CallbackScope&) = delete;
        CallbackScope& operator=(const CallbackScope&) = delete;
    };
    template<class Function> std::invoke_result_t<Function> external_call(Function&& function)
    {
        CallbackScope scope(*this);
        if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
            function();
            if (callback_violation_)
                throw std::logic_error("R26 reentrant scheduler mutation from backend or launch callback");
        } else {
            auto result = function();
            if (callback_violation_)
                throw std::logic_error("R26 reentrant scheduler mutation from backend or launch callback");
            return result;
        }
    }
    void observe_foreign_caller()
    {
        if (foreign_caller_.load(std::memory_order_acquire)) fail("R26 owner affinity violation");
    }
    static std::size_t checked_add(std::size_t a, std::size_t b)
    {
        if (b > std::numeric_limits<std::size_t>::max() - a)
            throw std::invalid_argument("retained byte count overflow");
        return a + b;
    }
    void add_dependency(std::set<NodeId>& dependencies, NodeId id) const
    {
        if (id == 0) return;
        const auto found = nodes_.find(id);
        if (found != nodes_.end() && !found->second.device_complete) dependencies.insert(id);
    }
    EventToken next_event() const
    {
        const auto slot = free_event_slots_.empty() ?
            static_cast<std::uint64_t>(event_generations_.size() + 1) : free_event_slots_.back();
        const auto generation = free_event_slots_.empty() ? 0 :
            event_generations_.at(static_cast<std::size_t>(slot - 1));
        if (generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("event generation exhausted");
        return {domain_, slot, generation + 1};
    }
    void commit_event(EventToken event) noexcept
    {
        if (free_event_slots_.empty()) event_generations_.push_back(event.generation);
        else {
            free_event_slots_.pop_back();
            event_generations_[static_cast<std::size_t>(event.slot - 1)] = event.generation;
        }
    }
    void cancel_unsubmitted() noexcept
    {
        for (auto& entry : nodes_) {
            auto& node = entry.second;
            if (!node.launch_attempted && !node.may_have_device_work &&
                node.info.state != NodeState::Quarantined)
                node.info.state = NodeState::Cancelled;
        }
    }
    void fail(std::string_view reason) noexcept
    {
        // Fail closed before handling diagnostic text. Neither the call-site argument
        // conversion nor this bounded copy may allocate, including inside a catch.
        accepting_ = false;
        const bool first = !has_error_;
        has_error_ = true;
        cancel_unsubmitted();
        if (first) {
            if (reason.empty()) reason = "R26 failure";
            first_error_size_ = std::min(reason.size(), first_error_.size());
            std::copy_n(reason.data(), first_error_size_, first_error_.data());
        }
    }
    void quarantine(Node& node, std::string_view reason) noexcept
    {
        node.info.state = NodeState::Quarantined;
        fail(reason);
    }
    void refresh_publication(Node& node)
    {
        if (!node.record_requested || node.published || node.retire_attempted) return;
        try {
            const auto publication = external_call([&] { return backend_->publication(node.info.event); });
            if (publication == Publication::Published) {
                node.published = true;
                if (node.info.state != NodeState::Quarantined) node.info.state = NodeState::Published;
            } else if (publication == Publication::Failed) {
                quarantine(node, "event publication failed");
            }
        } catch (const std::exception& error) {
            quarantine(node, error.what());
        } catch (...) {
            quarantine(node, "event publication query threw");
        }
    }
    void retire_ready()
    {
        bool progress;
        do {
            progress = false;
            for (auto it = nodes_.begin(); it != nodes_.end();) {
                auto& node = it->second;
                const bool cancelled_safe = node.info.state == NodeState::Cancelled && !node.may_have_device_work;
                if ((!node.device_complete && !cancelled_safe) || node.dependents != 0 || node.retire_attempted) {
                    ++it; continue;
                }
                node.retire_attempted = true;
                if (node.reserved) {
                    try {
                        external_call([&] { backend_->retire_event(node.info.event); });
                    } catch (const std::exception& error) {
                        quarantine(node, error.what()); ++it; continue;
                    } catch (...) {
                        quarantine(node, "event retirement failed"); ++it; continue;
                    }
                }
                for (NodeId dependency : node.info.dependencies) --nodes_.at(dependency).dependents;
                retained_bytes_ -= node.call.retained_bytes;
                for (const auto& access : node.call.accesses) {
                    auto allocation = allocations_.find(access.lease.token);
                    auto& state = allocation->second;
                    state.readers.erase(node.info.id);
                    if (state.writer == node.info.id) state.writer = 0;
                    if (--state.references == 0) {
                        retained_bytes_ -= state.bytes;
                        allocations_.erase(allocation);
                    }
                }
                free_event_slots_.push_back(node.info.event.slot);
                it = nodes_.erase(it);
                ++retired_;
                progress = true;
            }
        } while (progress);
    }

    std::shared_ptr<EventBackend> backend_;
    SchedulerConfig config_;
    const std::thread::id owner_;
    const std::uint64_t domain_;
    mutable std::atomic<bool> foreign_caller_{false};
    inline static std::atomic<std::uint64_t> next_domain_{1};
    std::map<NodeId, Node> nodes_;
    std::map<AllocationToken, AllocationState> allocations_;
    std::map<std::uint64_t, std::uint64_t> latest_generations_;
    std::vector<std::uint64_t> event_generations_;
    std::vector<std::uint64_t> free_event_slots_;
    std::size_t retained_bytes_ = 0;
    std::size_t producer_turn_ = 0;
    std::uint64_t last_sequence_ = 0;
    std::uint64_t last_id_ = 0;
    std::uint64_t admitted_ = 0;
    std::uint64_t submitted_ = 0;
    std::uint64_t retired_ = 0;
    bool accepting_ = true;
    bool in_external_callback_ = false;
    bool callback_violation_ = false;
    bool has_error_ = false;
    std::array<char, 1024> first_error_{};
    std::size_t first_error_size_ = 0;
};

} // namespace eager_parallel
} // namespace c10_npu
