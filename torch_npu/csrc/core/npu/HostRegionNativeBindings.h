#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <utility>

namespace c10_npu::host_regions {
namespace native_binding_detail {
struct Counters {
    std::atomic<uint64_t> metadata_snapshots{0}, source_checks{0};
    std::atomic<uint64_t> contiguous_ledger_moves{0}, input_ledger_copies{0};
    std::atomic<uint64_t> input_duplicates_elided{0}, output_duplicates_elided{0};
    std::atomic<uint64_t> descriptors_created{0}, buffers_created{0};
    std::atomic<uint64_t> descriptors_cleanup_attempts{0}, buffers_cleanup_attempts{0};
    std::atomic<uint64_t> pairs_created{0}, pairs_transferred{0}, pairs_cleanup_attempts{0}, pairs_live{0};
    std::atomic<uint64_t> command_abort_attempts{0}, command_abort_pops{0}, command_abort_cleanup_failures{0};
};
inline Counters counters;

inline bool TrackStats() noexcept
{
    static const bool enabled = [] {
        const char* diagnostics = std::getenv("HDB_REGION_DIAGNOSTICS");
        const char* trace = std::getenv("HDB_REGION_TRACE");
        return (diagnostics && std::strcmp(diagnostics, "1") == 0) ||
            (trace && std::strcmp(trace, "1") == 0);
    }();
    return enabled;
}
inline void Increment(std::atomic<uint64_t>& value) noexcept
{
    if (TrackStats()) value.fetch_add(1, std::memory_order_relaxed);
}
inline void Decrement(std::atomic<uint64_t>& value) noexcept
{
    if (TrackStats()) value.fetch_sub(1, std::memory_order_release);
}

// Borrowed, immutable metadata for one synchronous ACL descriptor construction.
// No Tensor/Storage owner or metadata vector is copied into this snapshot.
template<class DataType, class Format, class Dimensions>
struct TensorMetadata {
    DataType data_type;
    Format origin_format, device_format;
    Dimensions logical_dims, storage_dims;
    void* data;
    size_t buffer_bytes;
    bool set_layout;
};

// A complete pair remains local until both destination arrays can accept it.
// Partial construction has the same RAII protection. Cleanup attempts never
// throw/retry ACL operations during exception unwinding.
template<class DescriptorPointer, class BufferPointer, class Api>
class DescriptorBufferPair {
public:
    DescriptorBufferPair(DescriptorPointer descriptor, Api api) noexcept
        : descriptor_(descriptor), api_(std::move(api))
    {
        static_assert(std::is_nothrow_move_constructible<Api>::value, "ACL API guard must not throw on move");
        if (descriptor_) Increment(counters.descriptors_created);
    }
    DescriptorBufferPair(const DescriptorBufferPair&) = delete;
    DescriptorBufferPair& operator=(const DescriptorBufferPair&) = delete;
    DescriptorBufferPair(DescriptorBufferPair&& other) noexcept
        : descriptor_(other.descriptor_), buffer_(other.buffer_), api_(std::move(other.api_)), complete_(other.complete_)
    {
        other.descriptor_ = nullptr; other.buffer_ = nullptr; other.complete_ = false;
    }
    ~DescriptorBufferPair() noexcept
    {
        if (descriptor_) {
            Increment(counters.descriptors_cleanup_attempts);
            try { api_.DestroyDescriptor(descriptor_); } catch (...) {}
        }
        if (buffer_) {
            Increment(counters.buffers_cleanup_attempts);
            try { api_.DestroyBuffer(buffer_); } catch (...) {}
        }
        if (complete_) {
            Increment(counters.pairs_cleanup_attempts);
            Decrement(counters.pairs_live);
        }
    }
    DescriptorPointer descriptor() const noexcept { return descriptor_; }
    BufferPointer buffer() const noexcept { return buffer_; }
    void SetBuffer(BufferPointer buffer) noexcept
    {
        assert(!buffer_ && !complete_);
        buffer_ = buffer;
        if (buffer_) Increment(counters.buffers_created);
        if (descriptor_ && buffer_) {
            complete_ = true;
            Increment(counters.pairs_created);
            Increment(counters.pairs_live);
        }
    }
    void Transfer() noexcept
    {
        assert(complete_ && descriptor_ && buffer_);
        descriptor_ = nullptr; buffer_ = nullptr; complete_ = false;
        Increment(counters.pairs_transferred);
        Decrement(counters.pairs_live);
    }
private:
    DescriptorPointer descriptor_ = nullptr;
    BufferPointer buffer_ = nullptr;
    Api api_;
    bool complete_ = false;
};

template<class Metadata, class Name, class Api>
inline auto PreparePair(const Metadata& metadata, const Name& name, Api api)
{
    using Descriptor = decltype(api.CreateDescriptor(metadata));
    using Buffer = decltype(api.CreateBuffer(metadata));
    DescriptorBufferPair<Descriptor, Buffer, Api> result(api.CreateDescriptor(metadata), api);
    if (!result.descriptor()) throw std::bad_alloc();
    if (metadata.set_layout) {
        api.SetFormat(result.descriptor(), metadata);
        api.SetShape(result.descriptor(), metadata);
    }
    if (!name.empty()) api.SetName(result.descriptor(), name);
    result.SetBuffer(api.CreateBuffer(metadata));
    if (!result.buffer()) throw std::bad_alloc();
    return result;
}

template<class Descriptors, class Buffers, class Pair>
inline void AdoptPair(Descriptors& descriptors, Buffers& buffers, Pair& pair)
{
    if (descriptors.size() != buffers.size() || descriptors.size() == std::numeric_limits<size_t>::max())
        throw std::length_error("invalid native ACL binding arrays");
    // Both pointer emplacements are allocation-free and nonthrowing after the
    // reserves. In particular, reserve failure cannot publish a half binding.
    descriptors.reserve(descriptors.size() + 1);
    buffers.reserve(buffers.size() + 1);
    descriptors.emplace_back(pair.descriptor());
    buffers.emplace_back(pair.buffer());
    pair.Transfer();
}

template<class Ledger, class Tensor>
inline bool OwnsBindingReference(const Ledger& ledger, const Tensor& tensor) noexcept
{
    for (const auto& entry : ledger) if (&entry == &tensor) return true;
    return false;
}

// OpCommand's existing destructor cannot clean its implementation stack.
// Newly introduced builder failures must finish that source while the object
// is still alive, before propagation. Existing pointer members mark terminal
// state, so nested guards cannot release/pop the same command twice.
template<class Command, class Stack, class... Ledgers>
class CommandFailureGuard {
public:
    CommandFailureGuard(bool enabled, Command*& command, Stack*& stack, Ledgers&... ledgers) noexcept
        : enabled_(enabled), exceptions_(enabled ? std::uncaught_exceptions() : 0), command_(command), stack_(stack), ledgers_(ledgers...) {}
    CommandFailureGuard(const CommandFailureGuard&) = delete;
    CommandFailureGuard& operator=(const CommandFailureGuard&) = delete;
    ~CommandFailureGuard() noexcept
    {
        if (!enabled_ || std::uncaught_exceptions() <= exceptions_ || !command_) return;
        auto* command = command_; auto* stack = stack_;
        command_ = nullptr; stack_ = nullptr;
        Increment(counters.command_abort_attempts);
        try { command->releaseSource(); }
        catch (...) {
            Increment(counters.command_abort_cleanup_failures);
            // A partially destroyed ACL prefix must never be retried. Detach
            // only its metadata before this implementation slot is reused.
            try { command->releaseSource(false); }
            catch (...) { Increment(counters.command_abort_cleanup_failures); }
        }
        try { std::apply([](auto&... ledger) { (ledger.clear(), ...); }, ledgers_); }
        catch (...) { Increment(counters.command_abort_cleanup_failures); }
        if (stack) {
            Increment(counters.command_abort_pops);
            try { stack->Pop(); }
            catch (...) { Increment(counters.command_abort_cleanup_failures); }
        }
    }
private:
    bool enabled_;
    int exceptions_;
    Command*& command_;
    Stack*& stack_;
    std::tuple<Ledgers&...> ledgers_;
};
template<class Command, class Stack, class... Ledgers>
inline auto GuardCommand(bool enabled, Command*& command, Stack*& stack, Ledgers&... ledgers)
{
    if (!command || !stack) throw std::logic_error("native builder command already terminated");
    return CommandFailureGuard<Command, Stack, Ledgers...>(enabled, command, stack, ledgers...);
}
} // namespace native_binding_detail

// These describe preparation, not queued ACL retirement. pairs_live counts
// complete, unadopted RAII pairs; descriptor/buffer cleanup counters count the
// local cleanup attempts, including a partial pair on construction failure.
inline void AppendNativeBindingStats(std::ostream& output)
{
    const auto& c = native_binding_detail::counters;
    output << ",\"native_binding_stats_enabled\":" << (native_binding_detail::TrackStats() ? "true" : "false");
#define HDB_BINDING_STAT(name) output << ",\"native_binding_" #name "\":" << c.name.load(std::memory_order_relaxed)
    HDB_BINDING_STAT(metadata_snapshots); HDB_BINDING_STAT(source_checks);
    HDB_BINDING_STAT(contiguous_ledger_moves); HDB_BINDING_STAT(input_ledger_copies);
    HDB_BINDING_STAT(input_duplicates_elided); HDB_BINDING_STAT(output_duplicates_elided);
    HDB_BINDING_STAT(descriptors_created); HDB_BINDING_STAT(buffers_created);
    HDB_BINDING_STAT(descriptors_cleanup_attempts); HDB_BINDING_STAT(buffers_cleanup_attempts);
    HDB_BINDING_STAT(pairs_created); HDB_BINDING_STAT(pairs_transferred);
    HDB_BINDING_STAT(pairs_cleanup_attempts); HDB_BINDING_STAT(pairs_live);
    HDB_BINDING_STAT(command_abort_attempts); HDB_BINDING_STAT(command_abort_pops);
    HDB_BINDING_STAT(command_abort_cleanup_failures);
#undef HDB_BINDING_STAT
}
} // namespace c10_npu::host_regions
