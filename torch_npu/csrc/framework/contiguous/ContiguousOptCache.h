#pragma once

#include <c10/util/flat_hash_map.h>
#include <cstddef>
#include <mutex>

namespace at_npu::native {

// Callers execute optimizers after Find returns, using an owned value snapshot.
// No iterator or entry reference escapes the mutex. Existing keys are immutable
// so a colliding descriptor cannot replace an entry between two lookups.
template <class Value> class ContiguousOptCache final {
public:
    explicit ContiguousOptCache(std::size_t max_entries)
        : max_entries_(max_entries)
    {
    }

    bool Find(std::size_t key, Value& owned_snapshot) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return false;
        }
        owned_snapshot = it->second;
        return true;
    }

    bool Insert(std::size_t key, const Value& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.find(key) != entries_.end() || entries_.size() >= max_entries_) {
            return false;
        }
        return entries_.emplace(key, value).second;
    }

    bool AtCapacity() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size() >= max_entries_;
    }

private:
    const std::size_t max_entries_;
    mutable std::mutex mutex_;
    ska::flat_hash_map<std::size_t, Value> entries_;
};

} // namespace at_npu::native
