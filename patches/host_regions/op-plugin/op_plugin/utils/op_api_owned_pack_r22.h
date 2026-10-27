#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hdb_opapi {

// Only phase1 success transfers an executor. The reviewed ordinary/CommonRun
// phase2 consumes it on entry, including its error return; cancellation must
// never try to destroy that pointer a second time.
template<class Handle>
class ExecutorOwnership {
public:
    void Adopt(bool phase1_success, Handle value)
    {
        if (!phase1_success || !value || phase_ != Phase::Empty)
            throw std::logic_error("owned OpAPI executor cannot be adopted");
        value_ = value;
        phase_ = Phase::Prepared;
    }
    Handle TakeForExecute()
    {
        if (phase_ != Phase::Prepared) throw std::logic_error("owned OpAPI executor is not ready");
        phase_ = Phase::Submitted;
        return std::exchange(value_, {});
    }
    template<class Destroy>
    void Cancel(Destroy&& destroy)
    {
        if (phase_ != Phase::Prepared) return;
        phase_ = Phase::Cancelled;
        destroy(std::exchange(value_, {}));
    }
    bool MayHaveSubmitted() const noexcept { return phase_ == Phase::Submitted; }
    bool Prepared() const noexcept { return phase_ == Phase::Prepared; }
    ExecutorOwnership() = default;
    ExecutorOwnership(const ExecutorOwnership&) = delete;
    ExecutorOwnership& operator=(const ExecutorOwnership&) = delete;
private:
    enum class Phase { Empty, Prepared, Submitted, Cancelled };
    Handle value_{};
    Phase phase_ = Phase::Empty;
};

// One strong native owner per Storage identity. Each view still has a separate
// metadata snapshot; callers must audit every source before this deduplication.
template<class Owner>
class UniqueOwners {
public:
    void Reserve(size_t count) { entries_.reserve(count); }
    template<class Factory>
    void Retain(const void* identity, size_t bytes, Factory&& factory)
    {
        if (!identity) throw std::invalid_argument("owned OpAPI Storage identity is null");
        for (const auto& entry : entries_) if (entry.identity == identity) return;
        if (bytes > std::numeric_limits<size_t>::max() - bytes_)
            throw std::overflow_error("owned OpAPI retained bytes overflow");
        entries_.push_back(Entry{identity, factory()});
        bytes_ += bytes;
    }
    size_t Size() const noexcept { return entries_.size(); }
    size_t Bytes() const noexcept { return bytes_; }
    UniqueOwners() = default;
    UniqueOwners(const UniqueOwners&) = delete;
    UniqueOwners& operator=(const UniqueOwners&) = delete;
    UniqueOwners(UniqueOwners&&) = default;
    UniqueOwners& operator=(UniqueOwners&&) = default;
private:
    struct Entry { const void* identity; Owner owner; };
    std::vector<Entry> entries_;
    size_t bytes_ = 0;
};

// Braced tuple initialization evaluates snapshots in source order. Each
// completed snapshot owns its resources if a later snapshot throws.
template<class Snapshotter, class... Args>
auto SnapshotArgs(Snapshotter&& snapshot, Args&&... args)
{
    return std::tuple<std::decay_t<decltype(snapshot(std::forward<Args>(args)))>...>{
        snapshot(std::forward<Args>(args))...};
}

// Policies convert one frozen argument at a time and release its ACL type.
// The argument tuple must outlive this immovable pack. A payload stores the
// snapshots before the pack, so cleanup runs before snapshot destruction.
template<class Arguments, class Policy>
class ConvertedPack {
    template<size_t... I>
    static auto Types(std::index_sequence<I...>) -> std::tuple<
        std::decay_t<decltype(std::declval<Policy&>().Convert(std::get<I>(std::declval<const Arguments&>())))>...>;
public:
    static constexpr size_t size = std::tuple_size<Arguments>::value;
    using Values = decltype(Types(std::make_index_sequence<size>{}));

    ConvertedPack(const Arguments& arguments, Policy policy) : policy_(std::move(policy))
    {
        try { ConvertAll(arguments, std::make_index_sequence<size>{}); }
        catch (...) { Reset(); throw; }
    }
    ~ConvertedPack() noexcept { Reset(); }
    ConvertedPack(const ConvertedPack&) = delete;
    ConvertedPack& operator=(const ConvertedPack&) = delete;
    ConvertedPack(ConvertedPack&&) = delete;
    ConvertedPack& operator=(ConvertedPack&&) = delete;

    const Values& Get() const noexcept { return values_; }

    bool Reset() noexcept
    {
        bool success = true;
        ReleaseReverse<size>(success);
        return success;
    }

private:
    template<size_t I>
    void ConvertOne(const Arguments& arguments)
    {
        using Value = std::tuple_element_t<I, Values>;
        static_assert(std::is_nothrow_move_assignable<Value>::value,
            "ACL converted values must transfer without a throwing assignment");
        std::get<I>(values_) = policy_.Convert(std::get<I>(arguments));
        ++initialized_;
    }
    template<size_t... I>
    void ConvertAll(const Arguments& arguments, std::index_sequence<I...>)
    {
        (ConvertOne<I>(arguments), ...);
    }
    template<size_t I>
    void ReleaseReverse(bool& success) noexcept
    {
        if constexpr (I != 0) {
            if (initialized_ == I) {
                --initialized_; // An attempted cleanup must never be retried.
                auto value = std::exchange(std::get<I - 1>(values_), {});
                try { policy_.Release(value); }
                catch (...) { success = false; }
            }
            ReleaseReverse<I - 1>(success);
        }
    }
    Policy policy_;
    Values values_{};
    size_t initialized_ = 0;
};

// SDK 9.2 list creation copies child pointers; only successful outer creation
// transfers child ownership. Its public list-destroy API recursively releases
// children. Roll back every constructed child if conversion or creation fails.
template<class Range, class Policy, class Factory>
auto CreateOwnedList(const Range& arguments, Policy policy, Factory&& create)
{
    using Child = std::decay_t<decltype(policy.Convert(*arguments.begin()))>;
    std::vector<Child> children;
    children.reserve(arguments.size());
    try {
        for (const auto& argument : arguments) children.emplace_back(policy.Convert(argument));
        auto result = create(children);
        if (!result) throw std::bad_alloc();
        return result;
    } catch (...) {
        for (auto current = children.rbegin(); current != children.rend(); ++current) {
            try { policy.Release(*current); } catch (...) {}
        }
        throw;
    }
}
} // namespace hdb_opapi
