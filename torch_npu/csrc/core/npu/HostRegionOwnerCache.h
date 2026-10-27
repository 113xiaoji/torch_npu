#pragma once
#include <optional>
#include <unordered_map>
#include <utility>

namespace c10_npu::host_regions {
// Scope-owned cache only. The Owner must strongly hold the object identified by
// Key, so a pointer key cannot be recycled while an entry exists.
template<class Key, class Owner>
class BlockStorageOwnerCache {
public:
    BlockStorageOwnerCache() = default;
    BlockStorageOwnerCache(const BlockStorageOwnerCache&) = delete;
    BlockStorageOwnerCache& operator=(const BlockStorageOwnerCache&) = delete;
    template<class Factory>
    const Owner* get_or_create(Key key, bool source_allowed, Factory&& factory, bool& hit)
    {
        hit = false;
        // This gate precedes lookup: an unsafe alias cannot borrow a certificate.
        if (!source_allowed) return nullptr;
        const auto found = entries_.find(key);
        if (found != entries_.end()) { hit = true; return &found->second; }
        auto owner = factory();
        if (!owner) return nullptr; // Never cache negative eligibility decisions.
        const auto inserted = entries_.emplace(key, std::move(*owner));
        return &inserted.first->second;
    }
    size_t size() const noexcept { return entries_.size(); }
    void clear() noexcept { entries_.clear(); }
private:
    std::unordered_map<Key, Owner> entries_;
};
} // namespace c10_npu::host_regions
