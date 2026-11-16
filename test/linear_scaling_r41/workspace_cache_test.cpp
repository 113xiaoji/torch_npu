#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "op_api_workspace_cache_r41.h"

namespace {
using namespace op_api::workspace_r41;
struct Buffer {
    std::size_t bytes;
    unsigned identity;
};
using Owner = std::shared_ptr<Buffer>;
struct Factory {
    unsigned calls = 0;
    Owner operator()(std::uint64_t bytes) {
        return std::make_shared<Buffer>(Buffer{static_cast<std::size_t>(bytes), ++calls});
    }
};
void require(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
}
const WorkspaceKey first{0, 0x1000};
const WorkspaceKey other_stream{0, 0x2000};
const WorkspaceKey other_device{1, 0x1000};

void repeated_smaller_request_reuses_same_buffer() {
    WorkspaceCache<Owner> cache(128);
    Factory factory;
    const auto large = cache.Acquire(first, 96, true, factory);
    const auto small = cache.Acquire(first, 32, true, factory);
    require(large == small && factory.calls == 1, "same stream must reuse sufficient capacity");
    require(large->bytes == 96, "cache must preserve original allocation size");
}
void streams_devices_and_cache_instances_do_not_alias() {
    WorkspaceCache<Owner> cache(512), other_thread_cache(512);
    Factory factory;
    auto a = cache.Acquire(first, 64, true, factory);
    auto b = cache.Acquire(other_stream, 64, true, factory);
    auto c = cache.Acquire(other_device, 64, true, factory);
    auto d = other_thread_cache.Acquire(first, 64, true, factory);
    require(a != b && a != c && a != d && b != c, "stream/device/TLS owners must stay isolated");
    require(cache.Acquire(first, 64, true, factory) == a, "isolated entry must remain reusable");
}
void growth_preserves_queued_owner() {
    WorkspaceCache<Owner> cache(256);
    Factory factory;
    auto old = cache.Acquire(first, 32, true, factory);
    std::weak_ptr<Buffer> old_lifetime = old;
    std::function<void()> queued = [owner = old] { require(owner->bytes == 32, "old queued buffer changed"); };
    old.reset();
    const auto grown = cache.Acquire(first, 128, true, factory);
    require(!old_lifetime.expired(), "growth freed queued workspace");
    queued();
    queued = {};
    require(old_lifetime.expired(), "cache must release displaced old capacity");
    require(cache.Acquire(first, 64, true, factory) == grown, "grown capacity must be reused");
}
void byte_budget_evicts_least_recently_used_and_keeps_leases() {
    WorkspaceCache<Owner> cache(128);
    Factory factory;
    auto a = cache.Acquire(first, 64, true, factory);
    auto b = cache.Acquire(other_stream, 64, true, factory);
    require(cache.Acquire(first, 64, true, factory) == a, "touch must reuse first stream");
    auto c = cache.Acquire(other_device, 64, true, factory);
    require(a->bytes == 64 && b->bytes == 64 && c->bytes == 64, "eviction invalidated outstanding lease");
    require(cache.Acquire(first, 64, true, factory) == a, "eviction discarded recently used stream");
    require(cache.Acquire(other_stream, 64, true, factory) != b, "byte budget must evict oldest stream");
    require(cache.Stats().retained_bytes <= 128, "retained byte limit exceeded");
}
void entry_limit_evicts_without_unbounded_metadata() {
    WorkspaceCache<Owner, 2> cache(1024);
    Factory factory;
    auto a = cache.Acquire(first, 8, true, factory);
    cache.Acquire(other_stream, 8, true, factory);
    cache.Acquire(other_device, 8, true, factory);
    require(cache.Stats().entries == 2, "entry limit exceeded");
    require(cache.Acquire(first, 8, true, factory) != a, "entry capacity must evict oldest key");
}
void disabled_and_capture_requests_use_fresh_fallback() {
    WorkspaceCache<Owner> cache(128);
    Factory factory;
    WorkspaceEligibility eligibility{true, 1, true, false, false, false, false, true};
    auto a = cache.Acquire(first, 32, ReuseAllowed(eligibility), factory);
    auto verify_fallback = [&](WorkspaceEligibility blocked) {
        auto x = cache.Acquire(first, 32, ReuseAllowed(blocked), factory);
        auto y = cache.Acquire(first, 32, ReuseAllowed(blocked), factory);
        require(x != a && y != a && x != y, "unsupported path reused cached workspace");
    };
    auto blocked = eligibility; blocked.enabled = false; verify_fallback(blocked);
    blocked = eligibility; blocked.graph_capture = true; verify_fallback(blocked);
    blocked = eligibility; blocked.host_capture = true; verify_fallback(blocked);
    blocked = eligibility; blocked.internal_stream = false; verify_fallback(blocked);
    blocked = eligibility; blocked.task_queue_mode = 2; verify_fallback(blocked);
    blocked = eligibility; blocked.task_queue_mode = 0; verify_fallback(blocked);
    blocked = eligibility; blocked.sync_launch = true; verify_fallback(blocked);
    blocked = eligibility; blocked.owned_or_prepare = true; verify_fallback(blocked);
    blocked = eligibility; blocked.current_stream_matches = false; verify_fallback(blocked);
    require(cache.Acquire(first, 16, true, factory) == a, "fallback changed admitted cache entry");
}
void zero_and_oversize_leave_working_set_intact() {
    WorkspaceCache<Owner> cache(128);
    Factory factory;
    auto a = cache.Acquire(first, 64, true, factory);
    const auto before = factory.calls;
    require(!cache.Acquire(first, 0, true, factory), "zero workspace must be null");
    require(factory.calls == before, "zero workspace allocated a buffer");
    auto huge = cache.Acquire(first, 129, true, factory);
    require(huge->bytes == 129, "oversize fallback must preserve request");
    require(cache.Acquire(first, 64, true, factory) == a, "oversize bypass evicted useful workspace");
}
void allocation_failure_keeps_old_cache_valid() {
    WorkspaceCache<Owner> cache(128);
    Factory factory;
    auto a = cache.Acquire(first, 32, true, factory);
    bool threw = false;
    try {
        cache.Acquire(first, 64, true, [](std::uint64_t) -> Owner { throw std::bad_alloc(); });
    } catch (const std::bad_alloc&) { threw = true; }
    require(threw, "factory failure must propagate");
    require(cache.Acquire(first, 32, true, factory) == a, "failed growth destroyed old workspace");
}
void queued_lease_survives_thread_exit() {
    std::function<void()> queued;
    std::weak_ptr<Buffer> lifetime;
    std::thread producer([&] {
        WorkspaceCache<Owner> cache(128);
        Factory factory;
        auto owner = cache.Acquire(first, 32, true, factory);
        lifetime = owner;
        queued = [owner] { require(owner->bytes == 32, "thread exit changed queued storage"); };
    });
    producer.join();
    require(!lifetime.expired(), "producer exit freed queued workspace");
    queued();
    queued = {};
    require(lifetime.expired(), "queue retirement must release final owner");
}
} // namespace

int main() {
    unsigned failures = 0;
    for (const auto& test : std::vector<std::pair<const char*, void(*)()>>{
            {"reuse", repeated_smaller_request_reuses_same_buffer},
            {"key isolation", streams_devices_and_cache_instances_do_not_alias},
            {"growth owner", growth_preserves_queued_owner},
            {"byte budget", byte_budget_evicts_least_recently_used_and_keeps_leases},
            {"entry limit", entry_limit_evicts_without_unbounded_metadata},
            {"eligibility fallback", disabled_and_capture_requests_use_fresh_fallback},
            {"zero and oversize", zero_and_oversize_leave_working_set_intact},
            {"allocation failure", allocation_failure_keeps_old_cache_valid},
            {"producer exit", queued_lease_survives_thread_exit}}) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL " << test.first << ": " << error.what() << '\n'; }
    }
    return failures ? 1 : 0;
}
