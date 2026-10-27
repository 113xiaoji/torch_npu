#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c10_npu::host_regions {
// A queue slot borrows stable parameters. The packet itself has one publisher
// reference plus one outstanding-command count, never a shared_ptr per slot.
struct PacketCommandRef {
    void* packet = nullptr;
    uint32_t index = 0;
    uint64_t generation = 0;
};
bool AcquirePublishingPacketCommand(void* source, PacketCommandRef& output) noexcept;
bool IsPublishingPacketCommand(void* source) noexcept;
void* PacketCommandParameter(const PacketCommandRef&) noexcept;
void PacketCommandExecutionAttempted(const PacketCommandRef&) noexcept;
bool BeginPacketCommandRelease(const PacketCommandRef&) noexcept;
void FinishPacketCommandRelease(PacketCommandRef&) noexcept;
// Header must be detached first. Cleanup is attempted once; Finish runs even
// when a legacy ACL destructor throws, preserving original error propagation.
void ReleasePacketCommand(PacketCommandRef&);

namespace packet_detail {
struct Counters {
    std::atomic<uint64_t> parameter_allocations{0}, parameter_bytes_total{0}, parameter_bytes_live{0}, parameter_bytes_peak{0};
    std::atomic<uint64_t> owner_requests{0}, owner_hits{0}, owner_created{0}, owner_ids{0}, owner_full_guards{0};
    std::atomic<uint64_t> commands{0}, accepted{0}, retired{0}, blocks{0}, blocks_live{0};
    std::atomic<uint64_t> leases_acquired{0}, leases_retired{0}, suffix_reclaimed{0}, release_attempts{0}, execution_attempts{0};
};
inline Counters counters;
class AdmissionLease {
public:
    AdmissionLease(void* context, void (*destroy)(void*) noexcept) noexcept
        : context_(context), destroy_(destroy) {}
    AdmissionLease(const AdmissionLease&) = delete;
    AdmissionLease& operator=(const AdmissionLease&) = delete;
    bool accept(uint64_t generation) noexcept
    {
        if (!generation || !open_.load(std::memory_order_acquire)) return false;
        uint64_t previous = 0;
        generation_.compare_exchange_strong(previous, generation, std::memory_order_relaxed);
        if (generation_.load(std::memory_order_relaxed) != generation) return false;
        // Only the publisher calls accept/close. Its initial reference prevents
        // a concurrent fast consumer from destroying this object here.
        references_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    void close() noexcept
    {
        if (open_.exchange(false, std::memory_order_acq_rel)) retire();
    }
    void retire() noexcept
    {
        const auto before = references_.fetch_sub(1, std::memory_order_acq_rel);
        assert(before);
        if (before == 1) destroy_(context_); // Final destruction is caller-lock-free.
    }
private:
    void* context_;
    void (*destroy_)(void*) noexcept;
    std::atomic<size_t> references_{1};
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool> open_{true};
};

// Addresses do not move when another command exports pointer arrays or IDs.
// Counters include ACL pointer arrays and owner-ID spans; ACL objects retain
// their original creation/destruction APIs and never live in this arena.
class ParameterArena {
public:
    explicit ParameterArena(size_t chunk_size = 4096) noexcept : chunk_size_(chunk_size) {}
    ~ParameterArena() noexcept { counters.parameter_bytes_live.fetch_sub(allocated_, std::memory_order_release); }
    void* allocate(size_t bytes, size_t alignment)
    {
        if (sealed_) throw std::logic_error("sealed native parameter arena");
        if (!alignment || (alignment & (alignment - 1)) || alignment > alignof(std::max_align_t) ||
            bytes > std::numeric_limits<size_t>::max() - alignment) throw std::bad_alloc();
        bytes = std::max<size_t>(bytes, 1);
        if (!chunks_.empty()) {
            auto& chunk = chunks_.back();
            const size_t offset = (chunk.used + alignment - 1) & ~(alignment - 1);
            if (offset <= chunk.capacity && bytes <= chunk.capacity - offset) {
                chunk.used = offset + bytes;
                return chunk.data.get() + offset;
            }
        }
        const size_t capacity = std::max(chunk_size_, bytes + alignment);
        std::unique_ptr<unsigned char, Free> allocation(static_cast<unsigned char*>(std::malloc(capacity)));
        if (!allocation) throw std::bad_alloc();
        chunks_.emplace_back(std::move(allocation), capacity, bytes);
        allocated_ += capacity;
        counters.parameter_allocations.fetch_add(1, std::memory_order_relaxed);
        counters.parameter_bytes_total.fetch_add(capacity, std::memory_order_relaxed);
        auto current = counters.parameter_bytes_live.fetch_add(capacity, std::memory_order_relaxed) + capacity;
        auto peak = counters.parameter_bytes_peak.load(std::memory_order_relaxed);
        while (peak < current && !counters.parameter_bytes_peak.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {}
        return chunks_.back().data.get();
    }
    void seal() noexcept { sealed_ = true; }
private:
    struct Free { void operator()(void* p) const noexcept { std::free(p); } };
    struct Chunk {
        std::unique_ptr<unsigned char, Free> data;
        size_t capacity, used;
        Chunk(std::unique_ptr<unsigned char, Free> p, size_t capacity, size_t used) noexcept
            : data(std::move(p)), capacity(capacity), used(used) {}
    };
    std::vector<Chunk> chunks_;
    size_t chunk_size_;
    size_t allocated_ = 0;
    bool sealed_ = false;
};

// Immutable command use IDs borrow the same block arena as the ACL arrays.
// This trivial span owns no heap allocation or Tensor/Storage reference.
struct OwnerIdSpan {
    const uint32_t* values;
    size_t count;
    const uint32_t* begin() const noexcept { return values; }
    const uint32_t* end() const noexcept { return count ? values + count : values; }
    const uint32_t* data() const noexcept { return values; }
    size_t size() const noexcept { return count; }
    bool empty() const noexcept { return count == 0; }
};
template<class Arena, class Ids>
inline OwnerIdSpan CopyOwnerIds(Arena& arena, const Ids& ids)
{
    const auto count = ids.size();
    if (!count) return {};
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) throw std::bad_alloc();
    auto* data = static_cast<uint32_t*>(arena.allocate(count * sizeof(uint32_t), alignof(uint32_t)));
    std::copy(ids.begin(), ids.end(), data);
    return {data, count};
}

// Source eligibility is checked by Capture before EVERY lookup. A table value
// is the sole strong holder for its Storage; commands store only these IDs.
template<class Key, class Owner>
class BlockOwnerTable {
public:
    using OwnerId = uint32_t;
    template<class Factory>
    OwnerId retain(Key key, Factory&& factory, bool& hit)
    {
        if (sealed_) throw std::logic_error("sealed native owner table");
        const auto found = ids_.find(key);
        if (found != ids_.end()) { hit = true; return found->second; }
        hit = false;
        if (entries_.size() >= std::numeric_limits<OwnerId>::max()) throw std::bad_alloc();
        Owner owner = factory();
        const auto id = static_cast<OwnerId>(entries_.size());
        entries_.emplace_back(key, std::move(owner));
        try { ids_.emplace(key, id); }
        catch (...) { entries_.pop_back(); throw; }
        return id;
    }
    size_t checkpoint() const noexcept { return entries_.size(); }
    size_t size() const noexcept { return entries_.size(); }
    bool contains(Key key) const noexcept { return !sealed_ && ids_.find(key) != ids_.end(); }
    void rollback(size_t checkpoint) noexcept
    {
        assert(!sealed_);
        while (entries_.size() > checkpoint) {
            ids_.erase(entries_.back().key);
            entries_.pop_back();
        }
    }
    template<class Ids>
    void add_uses(const Ids& ids) noexcept
    {
        assert(!sealed_);
        // Only the producer mutates an unsealed table: do not replace deleted
        // Tensor retain RMWs with a per-reference atomic increment here.
        for (auto id : ids) {
            auto& uses = entries_[id].uses;
            uses.store(uses.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }
    }
    void seal() noexcept { sealed_ = true; }
    template<class Ids>
    void retire(const Ids& ids) noexcept
    {
        assert(sealed_);
        for (auto id : ids) {
            auto& entry = entries_[id];
            const auto before = entry.uses.fetch_sub(1, std::memory_order_acq_rel);
            assert(before);
            if (before == 1) entry.owner = Owner{};
        }
    }
private:
    struct Entry {
        Key key;
        Owner owner;
        std::atomic<size_t> uses{0};
        Entry(Key key, Owner value) noexcept : key(key), owner(std::move(value)) {}
        Entry(Entry&& other) noexcept : key(other.key), owner(std::move(other.owner)),
            uses(other.uses.load(std::memory_order_relaxed)) {}
        Entry& operator=(Entry&&) = delete;
    };
    std::vector<Entry> entries_;
    std::unordered_map<Key, OwnerId> ids_;
    bool sealed_ = false;
};
// Eligibility of the original source is never borrowed from another alias.
// Only a live, unsealed builder table can reuse the full storage certificate;
// a miss audits without constructing a holder during preflight.
template<class Table, class Key, class Audit>
inline bool CertifiedOwnerSourceAllowed(const Table* table, Key key, bool source_allowed, Audit&& audit)
{
    if (!source_allowed) return false;
    return (table && table->contains(key)) || audit();
}
// Collect source gates in original argument order. No full storage audit or
// holder factory runs until every input/output/temporary group has passed.
template<class Source, class Table, class Key, class Gate>
inline bool CollectOwnerSources(const Source* values, size_t count, const Table* certified,
    std::vector<Key>& keys, std::vector<std::pair<Key, const Source*>>& misses, Gate&& gate)
{
    for (size_t i = 0; i < count; ++i) {
        const auto key = gate(values[i]);
        if (!key) { misses.clear(); return false; }
        keys.push_back(key);
        if (!certified || !certified->contains(key)) misses.emplace_back(key, values + i);
    }
    return true;
}
template<class Key, class Source, class Audit>
inline bool AuditUniqueOwnerMisses(std::vector<std::pair<Key, const Source*>>& misses, Audit&& audit)
{
    struct ClearScratch {
        std::vector<std::pair<Key, const Source*>>& values;
        ~ClearScratch() noexcept { values.clear(); }
    } clear{misses};
    std::sort(misses.begin(), misses.end(), [](const auto& a, const auto& b) {
        return std::less<Key>{}(a.first, b.first);
    });
    misses.erase(std::unique(misses.begin(), misses.end(), [](const auto& a, const auto& b) {
        return a.first == b.first;
    }), misses.end());
    for (const auto& item : misses) if (!audit(*item.second)) return false;
    return true;
}
} // namespace packet_detail
} // namespace c10_npu::host_regions
