#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace op_api {
namespace workspace_r41 {
struct WorkspaceKey {
    int device;
    std::uintptr_t stream;
};
struct WorkspaceEligibility {
    bool enabled;
    unsigned task_queue_mode;
    bool internal_stream;
    bool sync_launch;
    bool graph_capture;
    bool host_capture;
    bool owned_or_prepare;
    bool current_stream_matches;
};
inline bool ReuseAllowed(const WorkspaceEligibility& value) {
    return value.enabled && value.task_queue_mode == 1 && value.internal_stream &&
        !value.sync_launch && !value.graph_capture && !value.host_capture &&
        !value.owned_or_prepare && value.current_stream_matches;
}
struct WorkspaceStats {
    std::uint64_t hits = 0, allocations = 0, growths = 0, evictions = 0, bypasses = 0;
    std::uint64_t retained_bytes = 0, entries = 0;
};
template <typename Buffer, std::size_t Capacity = 4>
class WorkspaceCache {
public:
    explicit WorkspaceCache(std::uint64_t byte_limit) : byte_limit_(byte_limit) {
        static_assert(Capacity != 0, "workspace cache must have at least one slot");
    }

    // This owns only reusable buffers; returned Buffer values independently own
    // their storage. The caller must retain that value through actual stream
    // submission, including when growth/eviction or producer exit drops a slot.
    // Buffer's allocator must then defer physical release/reuse appropriately
    // for the allocation stream. This helper never synchronizes a device.
    template <typename Factory>
    Buffer Acquire(WorkspaceKey key, std::uint64_t bytes, bool eligible, Factory&& factory) {
        if (bytes == 0) return {};
        if (!eligible || bytes > byte_limit_) {
            ++stats_.bypasses;
            auto result = factory(bytes);
            ++stats_.allocations;
            return result;
        }
        std::size_t existing = count_;
        for (std::size_t index = 0; index < count_; ++index) {
            if (entries_[index].key.device == key.device && entries_[index].key.stream == key.stream) {
                existing = index;
                if (entries_[index].bytes >= bytes) {
                    auto result = entries_[index].buffer;
                    auto touched = std::move(entries_[index]);
                    for (auto next = index + 1; next < count_; ++next)
                        entries_[next - 1] = std::move(entries_[next]);
                    entries_[count_ - 1] = std::move(touched);
                    ++stats_.hits;
                    return result;
                }
                break;
            }
        }
        // Allocate before modifying the working set: failure preserves its old
        // entries. No size rounding or Tensor shape change occurs here.
        auto result = factory(bytes);
        ++stats_.allocations;
        if (existing != count_) {
            Remove(existing);
            ++stats_.growths;
        }
        while (count_ == Capacity || stats_.retained_bytes > byte_limit_ - bytes) {
            Remove(0);
            ++stats_.evictions;
        }
        entries_[count_++] = Entry{key, bytes, result};
        stats_.retained_bytes += bytes;
        stats_.entries = count_;
        return result;
    }
    WorkspaceStats Stats() const { return stats_; }

private:
    struct Entry {
        WorkspaceKey key{};
        std::uint64_t bytes = 0;
        Buffer buffer{};
    };
    void Remove(std::size_t index) {
        stats_.retained_bytes -= entries_[index].bytes;
        for (auto next = index + 1; next < count_; ++next)
            entries_[next - 1] = std::move(entries_[next]);
        entries_[--count_] = Entry{};
        stats_.entries = count_;
    }
    const std::uint64_t byte_limit_;
    std::array<Entry, Capacity> entries_{};
    std::size_t count_ = 0;
    WorkspaceStats stats_{};
};
} // namespace workspace_r41
} // namespace op_api
