#include "torch_npu/csrc/core/npu/HostRegionCapture.h"
#include "torch_npu/csrc/core/npu/HostRegionNativeTls.h"
#include "torch_npu/csrc/core/npu/HostRegionNativeBindings.h"
#ifndef HDB_NATIVE_CORE_TEST
#include "torch_npu/csrc/core/npu/HostRegionRetirement.h"
#include "torch_npu/csrc/core/npu/HostRegionTensorOwner.h"
#include "torch_npu/csrc/core/npu/HostRegionOwnerCache.h"
#include "torch_npu/csrc/core/npu/HostRegionParameterRelease.h"
#include "third_party/op-plugin/op_plugin/ops/dvm/lazy_fusion_kernel.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifndef HDB_NATIVE_CORE_TEST
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif
#endif

#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#ifndef HDB_NATIVE_CORE_TEST
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#endif
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#ifndef HDB_NATIVE_CORE_TEST
namespace c10_npu::NPUCachingAllocator {
bool IsNativeCapturedRetirementDataOnStream(const c10::DataPtr&, const c10::Allocator*, int, aclrtStream) noexcept;
}
#endif

namespace {
using c10_npu::queue::QueueParas;
using namespace at_npu::native;
using namespace c10_npu::queue;
constexpr uint32_t max_batch_limit = 256;
// One fixed 26-op DLRM request reserves about 418.12 MiB of declared owners
// and NN workspaces before drain. This is a bounded ledger, not device peak use.
constexpr uint64_t owned_opapi_retained_byte_limit = 512ULL * 1024 * 1024;
struct Block;
void retire_block(Block*) noexcept;
struct RetireBlock { void operator()(Block* block) const noexcept { retire_block(block); } };
using BlockPtr = std::unique_ptr<Block, RetireBlock>;
struct RealCounters {
    std::atomic<uint64_t> runs_created{0}, runs_destroyed{0}, captured{0}, leases_acquired{0}, admission_acked{0};
    std::atomic<uint64_t> submit_attempts{0}, submit_returns{0}, device_retired{0}, cancelled{0}, payloads_live{0};
    std::atomic<uint64_t> device_pending{0}, retirement_failures{0}, eligible_fallback{0}, retained_bytes{0}, blocks_live{0};
    std::atomic<uint64_t> events_recorded{0}, events_completed{0};
    std::atomic<bool> failed{false};
} real_counts;
using CompletionHooks = c10_npu::host_regions::device_retirement_detail::CompletionHooks;
#ifdef HDB_NATIVE_CORE_TEST
CompletionHooks completion_hooks{};
void (*cleanup_context_hook_for_test)(void*) = nullptr;
void bind_owned_cleanup_context(void* context) { if (cleanup_context_hook_for_test) cleanup_context_hook_for_test(context); }
bool real_sdk_available() noexcept { return completion_hooks.create && completion_hooks.record && completion_hooks.wait && completion_hooks.destroy; }
const char* real_prepare_strategy() noexcept { return "worker"; } // Native boundary fixture; no SDK evidence.
#else
void check_real_context(void* expected) {
    aclrtContext actual = nullptr;
    NPU_CHECK_ERROR(aclrtGetCurrentContext(&actual));
    TORCH_CHECK(actual && actual == expected, "owned OpAPI requires the captured primary Context");
}
void bind_owned_cleanup_context(void* context) {
    NPU_CHECK_ERROR(c10_npu::SetDevice(0)); check_real_context(context);
}
void* create_completion(void* context) {
    c10_npu::NPUGuard guard(0); NPU_CHECK_ERROR(c10_npu::SetDevice(0)); check_real_context(context);
    aclrtEvent event = nullptr;
    const auto flags = c10_npu::acl::IsExistCreateEventExWithFlag() ? ACL_EVENT_SYNC : ACL_EVENT_DEFAULT;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtCreateEventWithFlag(&event, flags));
    return event;
}
void record_completion(void* event, void* stream, void* context) {
    c10_npu::NPUGuard guard(0); NPU_CHECK_ERROR(c10_npu::SetDevice(0)); check_real_context(context);
    NPU_CHECK_ERROR(aclrtRecordEvent(event, stream));
}
void wait_completion(void* event, void* context) {
    c10_npu::NPUGuard guard(0); NPU_CHECK_ERROR(c10_npu::SetDevice(0)); check_real_context(context);
    NPU_CHECK_ERROR(aclrtSynchronizeEvent(event));
}
void destroy_completion(void* event, void* context) noexcept {
    try {
        c10_npu::NPUGuard guard(0); NPU_CHECK_ERROR(c10_npu::SetDevice(0)); check_real_context(context);
        if (aclrtDestroyEvent(event) != ACL_ERROR_NONE) { real_counts.failed = true; ++real_counts.retirement_failures; }
    } catch (...) { real_counts.failed = true; ++real_counts.retirement_failures; }
}
CompletionHooks completion_hooks{create_completion, record_completion, wait_completion, destroy_completion};
bool real_sdk_available() noexcept { return c10_npu::host_regions::OwnedOpApiSdkContractAvailable(); }
const char* real_prepare_strategy() noexcept { return c10_npu::host_regions::OwnedOpApiPrepareStrategy(); }
#endif
struct Command {
    Command() : parameter(COMPILE_AND_EXECUTE, 0, nullptr) { legacy.attr = nullptr; }
    ~Command() noexcept;
    QueueParas parameter;
    ExecuteParas legacy;
    ExecuteParasOpApi opapi;
    std::unique_ptr<c10_npu::host_regions::OwnedOpApiPayload> owned_opapi;
    uint64_t token = 0, region = 0, local = 0;
    int device = -1;
    bool owns_acl = false, accepted = false, registered = false;
    bool packet = false, lease_acquired = false;
    Block* block = nullptr;
    c10_npu::host_regions::packet_detail::OwnerIdSpan owner_ids{};
    std::atomic<bool> execution_attempted{false}, release_attempted{false}, retired{false};
};
void arena_created() noexcept;
void arena_destroyed(uint64_t uncounted_slots) noexcept;
// One typed allocation per block. Only occupied slots are constructed, and
// their addresses never change while QueueParas points into the Command.
class CommandArena {
public:
    explicit CommandArena(size_t capacity)
        : data_(allocator_.allocate(capacity)), capacity_(capacity) { arena_created(); }
    ~CommandArena() noexcept
    {
        for (size_t i = 0; i < size_; ++i) data_[i].~Command();
        allocator_.deallocate(data_, capacity_);
        arena_destroyed(constructions_counted ? 0 : constructed);
    }
    CommandArena(const CommandArena&) = delete;
    CommandArena& operator=(const CommandArena&) = delete;
    Command& append()
    {
        if (size_ == capacity_) throw std::runtime_error("native command arena is full");
        auto* item = ::new (static_cast<void*>(data_ + size_)) Command();
        ++size_;
        ++constructed;
        return *item;
    }
    void pop_back() noexcept { data_[--size_].~Command(); }
    Command* begin() noexcept { return data_; }
    Command* end() noexcept { return data_ + size_; }
    const Command* begin() const noexcept { return data_; }
    const Command* end() const noexcept { return data_ + size_; }
    Command& operator[](size_t index) noexcept { return data_[index]; }
    Command& front() noexcept { return data_[0]; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    uint64_t constructed = 0;
    bool constructions_counted = false;
private:
    std::allocator<Command> allocator_;
    Command* data_;
    size_t capacity_, size_ = 0;
};
#ifndef HDB_NATIVE_CORE_TEST
class CapturedOwnerCache {
public:
    ~CapturedOwnerCache() noexcept { clear_and_report(); }
    at::Tensor retain(const at::Tensor& original, bool plain_acl)
    {
        ++requests_;
        auto* key = c10_npu::host_regions::CapturedStorageSourceKey(original, plain_acl);
        if (!key) { ++source_rejects_; return {}; }
        bool hit = false;
        const auto* owner = cache_.get_or_create(key, true, [&]() -> std::optional<at::Tensor> {
            if (!c10_npu::host_regions::CanCreateCapturedStorageOwner(original)) {
                ++factory_rejects_;
                return std::nullopt;
            }
            return c10_npu::host_regions::storage_owner_detail::Make(original.unsafeGetTensorImpl()->storage());
        }, hit);
        if (hit) ++hits_;
        else {
            ++misses_;
            if (owner) c10_npu::host_regions::storage_owner_detail::counters.cache_entries_live.fetch_add(1, std::memory_order_relaxed);
        }
        return owner ? *owner : at::Tensor{};
    }
    void clear_and_report() noexcept
    {
        auto& c = c10_npu::host_regions::storage_owner_detail::counters;
        const auto entries = cache_.size();
        cache_.clear(); // All storage destruction is outside run/region/queue locks.
        if (entries) {
            c.cache_entries_live.fetch_sub(entries, std::memory_order_release);
            c.cache_clears.fetch_add(1, std::memory_order_relaxed);
        }
        if (requests_) {
            c.requests.fetch_add(requests_, std::memory_order_relaxed);
            c.hits.fetch_add(hits_, std::memory_order_relaxed);
            c.misses.fetch_add(misses_, std::memory_order_relaxed);
            c.source_rejects.fetch_add(source_rejects_, std::memory_order_relaxed);
            c.factory_rejects.fetch_add(factory_rejects_, std::memory_order_relaxed);
            requests_ = hits_ = misses_ = source_rejects_ = factory_rejects_ = 0;
        }
    }
private:
    c10_npu::host_regions::BlockStorageOwnerCache<c10::StorageImpl*, at::Tensor> cache_;
    uint64_t requests_ = 0, hits_ = 0, misses_ = 0, source_rejects_ = 0, factory_rejects_ = 0;
};
#endif
struct Block {
    explicit Block(size_t capacity) : commands(capacity), lease(this, [](void* p) noexcept { retire_block(static_cast<Block*>(p)); }) {}
#ifndef HDB_NATIVE_CORE_TEST
    // Declared before commands: owners and pointer arrays outlive command ACL cleanup.
    c10_npu::host_regions::packet_detail::BlockOwnerTable<c10::StorageImpl*, at::Tensor> packet_owners;
#endif
    c10_npu::host_regions::packet_detail::ParameterArena parameter_arena;
    CommandArena commands;
    c10_npu::host_regions::packet_detail::AdmissionLease lease;
    bool packet = false, sealed = false, in_flight = false;
    std::unique_ptr<c10_npu::host_regions::device_retirement_detail::CompletionFence> device_fence;
    std::atomic<bool> may_have_submitted{false};
    std::atomic<bool> cleanup_failed{false};
    std::atomic_flag cleanup_failure_claimed = ATOMIC_FLAG_INIT;
    std::exception_ptr cleanup_failure;
    uint64_t retained_bytes = 0;
    void *execution_context = nullptr, *execution_stream = nullptr;
    Block* quarantined_next = nullptr;
#ifndef HDB_NATIVE_CORE_TEST
    CapturedOwnerCache owner_cache;
#endif
    uint64_t legacy = 0, opapi = 0, v2 = 0;
    uint64_t embedding_bag = 0, cast = 0, other_operators = 0;
    uint64_t capture_legacy_adopts = 0, capture_legacy_owner_moves = 0;
    uint64_t packet_owner_requests = 0, packet_owner_hits = 0, packet_owner_created = 0;
    uint64_t packet_owner_full_guards = 0;
    uint64_t packet_owner_ids = 0, packet_commands = 0;
    bool packet_counts_reported = false;
    void report_packet_counts() noexcept {
        if (!packet || packet_counts_reported) return;
        auto& c = c10_npu::host_regions::packet_detail::counters;
        c.owner_requests.fetch_add(packet_owner_requests, std::memory_order_relaxed);
        c.owner_hits.fetch_add(packet_owner_hits, std::memory_order_relaxed);
        c.owner_created.fetch_add(packet_owner_created, std::memory_order_relaxed);
        c.owner_full_guards.fetch_add(packet_owner_full_guards, std::memory_order_relaxed);
        c.owner_ids.fetch_add(packet_owner_ids, std::memory_order_relaxed);
        c.commands.fetch_add(packet_commands, std::memory_order_relaxed);
        packet_counts_reported = true;
    }
    ~Block() noexcept {
        report_packet_counts();
        if (device_fence) { --real_counts.blocks_live; real_counts.retained_bytes.fetch_sub(retained_bytes); }
        if (packet) c10_npu::host_regions::packet_detail::counters.blocks_live.fetch_sub(1, std::memory_order_release);
    }
};
void cancel_owned_payload(Command& command) noexcept
{
    if (!command.owned_opapi) return;
    auto& block = *command.block;
    if (block.cleanup_failed.load(std::memory_order_acquire)) return;
    try {
        if (command.owned_opapi->MayHaveSubmitted()) {
            // Host cleanup already ran on the original consumer. Inspect only
            // its saved result here; successful release needs no extra SDK bind.
            if (auto failure = command.owned_opapi->CleanupFailure()) std::rethrow_exception(failure);
            return; // The existing device fence still owns the payload.
        }
#ifndef HDB_NATIVE_CORE_TEST
        c10_npu::NPUGuard guard(0); // Keep the captured device through SDK cleanup.
#endif
        bind_owned_cleanup_context(block.execution_context);
        command.owned_opapi->Cancel();
        ++real_counts.cancelled;
        // A cancelled plan's noexcept cleanup can fail too. Both cases use the
        // existing block error path observed by this request's original release.
        if (auto failure = command.owned_opapi->CleanupFailure()) std::rethrow_exception(failure);
        command.owned_opapi.reset();
        --real_counts.payloads_live;
    }
    catch (...) {
        // The final lease observes every cancellation before reading the saved
        // exception. Never run SDK destructors after an unsuccessful bind.
        if (!block.cleanup_failure_claimed.test_and_set(std::memory_order_acq_rel)) block.cleanup_failure = std::current_exception();
        block.cleanup_failed.store(true, std::memory_order_release);
        return;
    }
}
void release_command_acl(Command& command) noexcept
{
    if (command.owned_opapi) { cancel_owned_payload(command); return; }
    if (!command.owns_acl || command.release_attempted.exchange(true, std::memory_order_acq_rel)) return;
    if (command.packet) c10_npu::host_regions::packet_detail::counters.release_attempts.fetch_add(1, std::memory_order_relaxed);
    try {
#ifndef HDB_NATIVE_CORE_TEST
        if (command.packet) ReleaseLegacyParameters(command.legacy,
            c10_npu::host_regions::ParameterArrayOwnership::PacketArena);
        else
#endif
            command.legacy.Release();
    } catch (...) { /* Original ACL release is attempted once, never retried. */ }
    command.owns_acl = false;
}
Command::~Command() noexcept {
    release_command_acl(*this);
    if (owned_opapi) { owned_opapi.reset(); --real_counts.payloads_live; }
}
// Unknown device completion is never treated as safe to free. Admission closes
// globally on the first fault, bounding quarantine to already admitted work.
std::atomic<Block*> quarantined_blocks{nullptr};
std::atomic<Block*> first_retirement_failure{nullptr};
[[noreturn]] void rethrow_retirement_failure() {
    auto* first = first_retirement_failure.load(std::memory_order_acquire);
    if (first && first->cleanup_failure) std::rethrow_exception(first->cleanup_failure);
    if (first && first->device_fence->Failure()) std::rethrow_exception(first->device_fence->Failure());
    throw std::runtime_error("real device completion/cleanup failed; admission is closed");
}
void retire_block(Block* block) noexcept {
#ifndef HDB_NATIVE_CORE_TEST
    std::optional<c10_npu::NPUGuard> context_guard;
#endif
    if (block->device_fence) {
        try {
#ifndef HDB_NATIVE_CORE_TEST
            context_guard.emplace(0); // Includes final native Storage/ACL destruction.
#endif
            bind_owned_cleanup_context(block->execution_context);
        } catch (...) {
            if (!block->cleanup_failure_claimed.test_and_set(std::memory_order_acq_rel)) block->cleanup_failure = std::current_exception();
            block->cleanup_failed.store(true, std::memory_order_release);
        }
        // Also covers an unpublished block destroyed by an arbitrary host
        // thread. Its prepared SDK objects require the captured Context too.
        for (auto& command : block->commands) cancel_owned_payload(command);
        const bool submitted = block->may_have_submitted.load(std::memory_order_acquire);
        const bool complete = !block->cleanup_failed.load(std::memory_order_acquire) && block->device_fence->Complete(submitted);
        if (submitted && block->device_fence->Recorded()) ++real_counts.events_recorded;
        if (!complete) {
            real_counts.failed.store(true, std::memory_order_release); ++real_counts.retirement_failures;
            Block* first = nullptr; first_retirement_failure.compare_exchange_strong(first, block);
            auto* head = quarantined_blocks.load(std::memory_order_relaxed);
            do { block->quarantined_next = head; } while (!quarantined_blocks.compare_exchange_weak(head, block));
            return;
        }
        if (submitted) {
            --real_counts.device_pending; ++real_counts.events_completed;
            for (const auto& command : block->commands) if (command.owned_opapi && command.owned_opapi->MayHaveSubmitted()) ++real_counts.device_retired;
        }
    }
    delete block;
}
enum class RegionPhase { New, Starting, Active, Ended, Done };
struct NativeRegion {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<BlockPtr> blocks;
    uint32_t queued = 0;
    RegionPhase phase = RegionPhase::New;
    bool entered = false, failed = false;
};
struct NativeRun {
    NativeRun(uint32_t count, uint32_t capacity, uint32_t batch, bool plain_builder, bool real_opapi)
        : caller(real_opapi), capacity(capacity), batch_limit(batch), plain_native_builder(plain_builder), real_opapi(real_opapi)
    {
        regions.reserve(count);
        for (uint32_t i = 0; i < count; ++i) regions.push_back(std::make_unique<NativeRegion>());
    }
    hdb_native::CallerTls caller; // Capture caller native TLS before any detach/lock.
    const uint32_t capacity, batch_limit;
    const bool plain_native_builder, real_opapi;
    std::vector<std::unique_ptr<NativeRegion>> regions;
    std::mutex control;
    std::atomic<bool> aborted{false};
    uint32_t active_tasks = 0;
    bool draining = false, drain_started = false, destroyed = false;
};
struct Scope {
    HdbRegionEmit emit = nullptr;
    HdbRegionBatchEmit batch_emit = nullptr;
    void* context = nullptr;
    uint64_t region = 0, local = 0;
    int device = -1;
    uint32_t batch_limit = 1;
    bool failed = false, emit_stopped = false;
    BlockPtr block;
    std::shared_ptr<NativeRun> run;
    std::unique_ptr<hdb_native::TaskTls> task_tls;
    std::vector<uint32_t> prepared_owner_ids;
#ifndef HDB_NATIVE_CORE_TEST
    std::vector<c10::StorageImpl*> prepared_owner_keys; // Reused scratch, never strong Tensor references.
    std::vector<std::pair<c10::StorageImpl*, const at::Tensor*>> prepared_owner_misses;
#endif
    size_t prepared_owner_checkpoint = 0;
    bool packet_prepared = false;
    bool open() const noexcept { return emit || batch_emit || run; }
};
thread_local Scope scope;
thread_local std::string last_error;
thread_local Command* publishing = nullptr;

struct Order { uint64_t token, region, local, ordinal; };
struct Counters {
    std::atomic<uint64_t> captured{0}, published{0}, discarded{0}, consumed{0};
    std::atomic<uint64_t> scopes_opened{0}, scopes_closed{0}, active_scopes{0}, max_active_scopes{0};
    std::atomic<uint64_t> failures{0}, unsupported{0}, max_live{0}, commands_live{0};
    std::atomic<uint64_t> legacy{0}, opapi{0}, v2{0}, fifo_errors{0};
    std::atomic<uint64_t> queue_copy_total{0}, queue_consume_total{0};
    std::atomic<uint64_t> queue_discarded{0}, queue_abandoned{0};
    std::atomic<uint64_t> queue_release_total{0}, queue_abandon_events{0};
    std::atomic<uint64_t> embedding_bag{0}, cast{0}, other_operators{0};
    std::atomic<uint64_t> batches_emitted{0}, max_batch_size{0};
    std::atomic<uint64_t> python_callbacks{0}, native_runs_created{0}, native_runs_destroyed{0};
    std::atomic<uint64_t> native_builder_runs_created{0}, native_builder_runs_destroyed{0};
    std::atomic<uint64_t> native_run_begins{0}, native_run_done{0}, native_run_drains{0};
    std::atomic<uint64_t> native_active_drains{0}, native_queued_blocks{0}, native_queued_commands{0};
    std::atomic<uint64_t> native_max_region_queued{0};
    std::atomic<uint64_t> native_command_arena_allocations{0}, native_command_arenas_live{0};
    std::atomic<uint64_t> native_command_slots_constructed{0};
    std::atomic<uint64_t> capture_legacy_adopts{0}, capture_legacy_owner_moves{0}, native_batch_publish_calls{0};
    uint64_t published_hash = 1469598103934665603ULL;
    uint64_t consumed_hash = 1469598103934665603ULL;
};
uint64_t real_worker_tid() noexcept {
#ifdef __linux__
    return static_cast<uint64_t>(::syscall(SYS_gettid));
#else
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())); // Host-only fixtures.
#endif
}
struct RealWorker { uint64_t tid, regions = 0, captured = 0; };
struct State {
    std::mutex mutex;
    std::vector<RealWorker> real_workers;
    bool real_diagnostics_incomplete = false;
    // The caller already publishes in source order. Direct C ABI users also
    // serialize whole blocks, with one claim/configuration check per block.
    std::mutex publish_mutex;
    std::unordered_map<uint64_t, BlockPtr> tokens;
    std::unordered_map<uint64_t, std::shared_ptr<NativeRun>> runs;
    std::unordered_map<uint64_t, Order> pending;
    uint64_t next_token = 1;
    uint64_t next_run = 1;
    Counters counts;
    std::atomic<bool> ever_enabled{false};
    bool trace = false, diagnostics = false;
    std::vector<Order> consumed_trace;
    std::vector<std::array<uint64_t, 2>> queue_published_trace, queue_consumed_trace;
    State()
    {
        const char* trace_value = std::getenv("HDB_REGION_TRACE");
        const char* diagnostic_value = std::getenv("HDB_REGION_DIAGNOSTICS");
        trace = trace_value && std::strcmp(trace_value, "1") == 0;
        diagnostics = trace || (diagnostic_value && std::strcmp(diagnostic_value, "1") == 0);
        if (diagnostics) real_workers.reserve(4);
        if (trace) {
            consumed_trace.reserve(4096);
            queue_published_trace.reserve(4096);
            queue_consumed_trace.reserve(4096);
        }
    }
};
State& state()
{
    // Original queue callbacks may run during Python module teardown.
    static State* result = new State();
    return *result;
}
void record_real_worker(bool begin) noexcept {
    auto& data = state();
    if (!data.diagnostics) return;
    try {
        std::lock_guard<std::mutex> lock(data.mutex);
        const auto tid = real_worker_tid();
        auto found = std::find_if(data.real_workers.begin(), data.real_workers.end(), [tid](const RealWorker& w){return w.tid == tid;});
        if (found == data.real_workers.end()) { data.real_workers.push_back({tid}); found = data.real_workers.end()-1; }
        if (begin) ++found->regions; else ++found->captured;
    } catch (...) { std::lock_guard<std::mutex> lock(data.mutex); data.real_diagnostics_incomplete = true; }
}
void arena_created() noexcept
{
    auto& c = state().counts;
    c.native_command_arena_allocations.fetch_add(1, std::memory_order_relaxed);
    c.native_command_arenas_live.fetch_add(1, std::memory_order_relaxed);
}
void arena_destroyed(uint64_t uncounted_slots) noexcept
{
    auto& c = state().counts;
    c.native_command_slots_constructed.fetch_add(uncounted_slots, std::memory_order_relaxed);
    c.native_command_arenas_live.fetch_sub(1, std::memory_order_relaxed);
}
void fail(const char* message) noexcept { try { last_error = message; } catch (...) { last_error.clear(); } }
void fail(const std::string& message) noexcept { fail(message.c_str()); }
void maximum(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
    auto old = target.load(std::memory_order_relaxed);
    while (old < value && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {}
}
uint64_t mix(uint64_t hash, const Order& order)
{
    for (uint64_t value : {order.token, order.region, order.local, order.ordinal}) {
        for (unsigned index = 0; index < 8; ++index) {
            hash ^= (value >> (index * 8)) & 255;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}
size_t copy_string(const std::string& text, char* output, size_t capacity)
{
    if (output && capacity) {
        const size_t count = std::min(capacity - 1, text.size());
        std::memcpy(output, text.data(), count);
        output[count] = '\0';
    }
    return text.size() + 1;
}
BlockPtr claim(uint64_t token, bool required = true)
{
    auto& data = state();
    std::lock_guard<std::mutex> lock(data.mutex);
    auto found = data.tokens.find(token);
    if (found == data.tokens.end()) {
        if (required) throw std::runtime_error("unknown or already consumed native region token");
        return nullptr;
    }
    auto block = std::move(found->second);
    data.tokens.erase(found);
    return block;
}
void discard_unaccepted(const Block& block)
{
    auto& data = state();
    uint64_t discarded = 0;
    for (const auto& command : block.commands) {
        if (!command.accepted) ++discarded;
    }
    if (data.diagnostics) {
        std::lock_guard<std::mutex> lock(data.mutex);
        for (const auto& command : block.commands) {
            if (!command.accepted && command.registered) data.pending.erase(command.parameter.correlation_id);
        }
    }
    data.counts.discarded.fetch_add(discarded, std::memory_order_relaxed);
    data.counts.commands_live.fetch_sub(discarded, std::memory_order_relaxed);
}
void count_block(Block& block)
{
    block.report_packet_counts();
    auto& c = state().counts;
    c.native_command_slots_constructed.fetch_add(block.commands.constructed, std::memory_order_relaxed);
    block.commands.constructions_counted = true;
#define HDB_ADD(field) c.field.fetch_add(block.field, std::memory_order_relaxed)
    HDB_ADD(legacy); HDB_ADD(opapi); HDB_ADD(v2);
    HDB_ADD(embedding_bag); HDB_ADD(cast); HDB_ADD(other_operators);
    HDB_ADD(capture_legacy_adopts); HDB_ADD(capture_legacy_owner_moves);
#undef HDB_ADD
    maximum(c.max_batch_size, block.commands.size());
}
std::shared_ptr<NativeRun> find_run(uint64_t id)
{
    hdb_native::WaitDetach detached;
    auto& data = state();
    std::lock_guard<std::mutex> lock(data.mutex);
    const auto found = data.runs.find(id);
    if (found == data.runs.end()) throw std::runtime_error("unknown or destroyed native run");
    return found->second;
}
void abort_run(const std::shared_ptr<NativeRun>& run)
{
    hdb_native::WaitDetach detached;
    run->aborted.store(true, std::memory_order_release);
    for (auto& region : run->regions) {
        std::lock_guard<std::mutex> lock(region->mutex);
        region->changed.notify_all();
    }
}
void enqueue_native_block(BlockPtr& block)
{
    hdb_native::WaitDetach detached;
    auto& run = *scope.run;
    auto& region = *run.regions.at(static_cast<size_t>(scope.region));
    const auto units = static_cast<uint32_t>(block->commands.size());
    std::unique_lock<std::mutex> lock(region.mutex);
    region.changed.wait(lock, [&] {
        return run.aborted.load(std::memory_order_acquire) || region.queued + units <= run.capacity;
    });
    if (run.aborted.load(std::memory_order_acquire)) throw std::runtime_error("native run aborted");
    if (region.phase != RegionPhase::Active) throw std::runtime_error("native region is not active");
    region.blocks.push_back(std::move(block));
    region.queued += units;
    auto& c = state().counts;
    ++c.native_queued_blocks;
    c.native_queued_commands.fetch_add(units, std::memory_order_relaxed);
    maximum(c.native_max_region_queued, region.queued);
    region.changed.notify_all();
}
void flush_block() noexcept
{
    if (!scope.block) return;
#ifndef HDB_NATIVE_CORE_TEST
    if (!scope.block->packet) scope.block->owner_cache.clear_and_report();
#endif
    if (scope.block->commands.empty()) return;
    auto block = std::move(scope.block);
    if (block->packet) {
        block->parameter_arena.seal();
#ifndef HDB_NATIVE_CORE_TEST
        block->packet_owners.seal();
#endif
        block->sealed = true;
    }
    count_block(*block);
    if (scope.emit_stopped) { discard_unaccepted(*block); return; }
    uint64_t token = 0;
    const auto command_count = static_cast<uint32_t>(block->commands.size());
    try {
        auto& data = state();
        if (scope.run) {
            {
                hdb_native::WaitDetach detached;
                std::lock_guard<std::mutex> lock(data.mutex);
                if (data.next_token == 0) throw std::runtime_error("native region token space exhausted");
                token = data.next_token++;
            }
            for (auto& command : block->commands) command.token = token;
            enqueue_native_block(block); // Direct unique ownership; no callback or token-map entry.
            data.counts.batches_emitted.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(data.mutex);
            if (data.next_token == 0) throw std::runtime_error("native region token space exhausted");
            token = data.next_token++;
            // Allocate the registry node before moving the owner. A map
            // allocation failure therefore still leaves a clear release owner.
            auto result = data.tokens.emplace(token, nullptr);
            for (auto& command : block->commands) command.token = token;
            result.first->second = std::move(block);
        }
        data.counts.batches_emitted.fetch_add(1, std::memory_order_relaxed);
        data.counts.python_callbacks.fetch_add(1, std::memory_order_relaxed);
        const int status = scope.batch_emit ? scope.batch_emit(token, command_count, scope.context) :
                                             scope.emit(token, scope.context);
        if (status == 0) return;
        fail("Python command emitter rejected native capture; scope must abort");
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown native region block emission failure"); }
    // Do not unwind an OpCommand before releaseSource(false) and Pop. The
    // emitter may already have discarded or transferred the token to a caller.
    scope.failed = scope.emit_stopped = true;
    if (!scope.run && !block && token) block = claim(token, false);
    if (block) discard_unaccepted(*block);
}
void check_configuration(int device, bool real_opapi = false)
{
    const char* sim = std::getenv("QWEN_ACL_HOST_SIM");
    if (real_opapi) {
#if !defined(HDB_NATIVE_CORE_TEST) && !defined(BUILD_LIBTORCH)
        if (c10_npu::impl::NPUTrace::getTrace()) throw std::runtime_error("real capture does not admit Python NPU trace hooks");
#endif
        if (real_counts.failed.load(std::memory_order_acquire)) rethrow_retirement_failure();
        if ((sim && std::strcmp(sim, "1") == 0) || !real_sdk_available())
            throw std::runtime_error("real owned OpAPI SDK lifecycle unavailable or retirement failed; HOSTSIM is not a real capability");
    } else if (!sim || std::strcmp(sim, "1") != 0) {
        throw std::runtime_error("native region capture requires QWEN_ACL_HOST_SIM=1");
    }
    if (device != 0 || c10_npu::current_device() != device) {
        throw std::runtime_error("native region capture supports current device 0 only");
    }
    if (!c10_npu::option::OptionsManager::GetTaskQueueEnable() ||
        c10_npu::option::OptionsManager::GetPerStreamQueue() ||
        c10_npu::option::OptionsManager::CheckBlockingEnable()) {
        throw std::runtime_error("native region capture requires default asynchronous task queue");
    }
    const auto stream = c10_npu::getCurrentNPUStream(device);
    if (stream != c10_npu::getDefaultNPUStream(device) || stream.isSyncLaunchStream()) {
        throw std::runtime_error("native region capture requires the default stream");
    }
    if (c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::Active) {
        throw std::runtime_error("native region capture cannot nest NPU graph capture");
    }
}
// Destination is a freshly constructed arena slot. Only plain legacy adopts;
// a custom function's copy can observe holder counts, so keep its original Copy
// order (including any throwing copy) before relinquishing the source owner.
bool adopt_captured_legacy(ExecuteParas& destination, ExecuteParas& source)
{
    const bool adopt = !source.customHandler;
    if (adopt) {
        std::strncpy(destination.opType, source.opType, sizeof(destination.opType) - 1);
        destination.isJitDisable = source.isJitDisable;
        destination.pta_correlation_id = source.pta_correlation_id;
        destination.hostMemory = std::move(source.hostMemory);
        destination.paras = source.paras;
        destination.constParams = source.constParams;
        destination.attr = source.attr;
    } else {
        destination.Copy(source);
    }
    source.paras = ACL_PARAMS{};
    source.constParams = CONST_PARAMS{};
    source.attr = nullptr;
    source.hostMemory.clear();
    source.customHandler = nullptr;
    return adopt;
}

void close_packet_publication(Block* block) noexcept
{
    // All accepted prefix commands keep their own outstanding count. Reclaim
    // only the unaccepted suffix before dropping the publisher protection.
    for (size_t index = 0; index < block->commands.size(); ++index) {
        auto& command = block->commands[index];
        if (command.lease_acquired) continue;
        c10_npu::host_regions::packet_detail::counters.suffix_reclaimed.fetch_add(1, std::memory_order_relaxed);
        release_command_acl(command);
        c10_npu::host_regions::PacketCommandRef ref{block, static_cast<uint32_t>(index), command.token};
        c10_npu::host_regions::FinishPacketCommandRelease(ref);
    }
    block->lease.close();
}
void publish_block(BlockPtr& owner)
{
    auto& block = *owner;
    // Declared before the publish lock so closure and all final destructors run
    // after the original queue/publisher lock has been released.
    struct PacketClose {
        Block* block = nullptr;
        bool success = false;
        ~PacketClose() noexcept {
            if (!block) return;
            if (!success) discard_unaccepted(*block);
            close_packet_publication(block);
        }
    } packet_close;
    if (block.packet) {
        block.in_flight = true;
        packet_close.block = owner.release();
    }
    auto& data = state();
    std::unique_lock<std::mutex> publish_lock(data.publish_mutex);
    check_configuration(block.commands.front().device, bool(block.device_fence));
    std::array<void*, max_batch_limit> items;
    for (size_t i = 0; i < block.commands.size(); ++i) items[i] = &block.commands[i].parameter;
    auto prepare = [](size_t index, void* context) {
        publishing = &static_cast<Block*>(context)->commands[index];
    };
    data.counts.native_batch_publish_calls.fetch_add(1, std::memory_order_relaxed);
    try {
        // Repository must recognize a packet before it chooses its batch path.
        // The original ordered batch producer lease then owns the entire block.
        if (block.packet) publishing = &block.commands.front();
        c10_npu::enCurrentNPUStreamBatch(items.data(), block.commands.size(),
                                       block.commands.front().device, prepare, &block);
    } catch (...) { publishing = nullptr; throw; }
    publishing = nullptr;
    for (const auto& command : block.commands) {
        if (!command.accepted) throw std::runtime_error("original NPU queue did not accept command");
    }
    packet_close.success = true;
    publish_lock.unlock();
    if (packet_close.block) {
        const bool real = bool(packet_close.block->device_fence);
        auto* completed = packet_close.block; packet_close.block = nullptr;
        close_packet_publication(completed);
        if (real && real_counts.failed.load(std::memory_order_acquire)) rethrow_retirement_failure();
    }
}

int drain_native_run(const std::shared_ptr<NativeRun>& run)
{
    for (size_t index = 0; index < run->regions.size(); ++index) {
        auto& region = *run->regions[index];
        for (;;) {
            BlockPtr block;
            bool failed = false;
            {
                std::unique_lock<std::mutex> lock(region.mutex);
                region.changed.wait(lock, [&] {
                    return run->aborted.load(std::memory_order_acquire) || !region.blocks.empty() ||
                           region.phase == RegionPhase::Done;
                });
                if (run->aborted.load(std::memory_order_acquire)) throw std::runtime_error("native run aborted");
                if (region.blocks.empty()) {
                    failed = region.failed;
                } else {
                    block = std::move(region.blocks.front());
                    region.blocks.pop_front();
                    const auto units = static_cast<uint32_t>(block->commands.size());
                    region.queued -= units;
                    --state().counts.native_queued_blocks;
                    state().counts.native_queued_commands.fetch_sub(units, std::memory_order_relaxed);
                    region.changed.notify_all();
                }
            }
            if (!block) {
                if (failed) { abort_run(run); return static_cast<int>(index + 1); }
                break;
            }
            try { publish_block(block); }
            catch (...) { if (block) discard_unaccepted(*block); throw; }
            // Both accepted owner release and failed-block destruction occur
            // outside every native run/region/publish lock.
        }
    }
    return 0;
}
int begin_scope(HdbRegionEmit emit, HdbRegionBatchEmit batch_emit, void* context,
                int device, uint64_t region, uint32_t batch_limit) noexcept
{
    try {
        last_error.clear();
        if ((!emit && !batch_emit) || scope.open()) throw std::runtime_error("invalid or nested native region scope");
        if (!batch_limit || batch_limit > max_batch_limit) throw std::runtime_error("native batch limit must be in 1..256");
        check_configuration(device);
        auto& data = state();
        data.ever_enabled.store(true, std::memory_order_release);
        scope.emit = emit; scope.batch_emit = batch_emit; scope.context = context;
        scope.device = device; scope.region = region; scope.batch_limit = batch_limit;
        data.counts.scopes_opened.fetch_add(1, std::memory_order_relaxed);
        maximum(data.counts.max_active_scopes, data.counts.active_scopes.fetch_add(1, std::memory_order_relaxed) + 1);
        return 0;
    } catch (const std::exception& error) { fail(error.what()); return -1; }
    catch (...) { fail("unknown native region begin failure"); return -1; }
}
} // namespace

namespace c10_npu {
namespace host_regions {
bool IsCapturing() noexcept { return scope.open(); }

bool UseCapturedNativeBindings() noexcept
{
    if (!scope.run || !scope.task_tls || !scope.run->plain_native_builder || scope.failed || scope.emit_stopped ||
        scope.run->aborted.load(std::memory_order_acquire)) return false;
    return true;
}

void initialize_real_block(Block& block) {
    if (!scope.run || !scope.run->real_opapi || block.device_fence) return;
#ifdef HDB_NATIVE_CORE_TEST
    block.execution_stream = reinterpret_cast<void*>(uintptr_t(1));
#else
    block.execution_stream = c10_npu::getDefaultNPUStream(scope.device).stream(false);
#endif
    block.execution_context = scope.run->caller.context();
    block.device_fence = std::make_unique<device_retirement_detail::CompletionFence>(completion_hooks, block.execution_context, block.execution_stream);
    ++real_counts.blocks_live;
}
void reserve_owned_bytes(size_t);
bool PreparedCapturedPacketCommand() noexcept { return scope.packet_prepared; }
void CancelCapturedPacketCommand() noexcept
{
    if (!scope.packet_prepared) return;
#ifndef HDB_NATIVE_CORE_TEST
    if (scope.block) scope.block->packet_owners.rollback(scope.prepared_owner_checkpoint);
#endif
    scope.prepared_owner_ids.clear();
    scope.packet_prepared = false;
}
void* AllocateCapturedPacketParams(size_t bytes, size_t alignment)
{
    if (!scope.packet_prepared || !scope.block || !scope.block->packet)
        throw std::logic_error("native packet parameters require a prepared command");
    return scope.block->parameter_arena.allocate(bytes, alignment);
}
bool PrepareCapturedPacketCommand(const at::Tensor* inputs, size_t ni, const at::Tensor* outputs, size_t no,
    const at::Tensor* temporaries, size_t nt)
{
    const char* enabled = std::getenv("HDB_REGION_PACKET");
    if (!scope.run || scope.failed || !enabled || std::strcmp(enabled, "1") != 0) return false;
    if (scope.packet_prepared) throw std::logic_error("nested native packet command preparation");
    scope.prepared_owner_ids.clear(); // Keep builder capacity across commands.
#ifdef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
    return false; // Packet admission requires the original ordered batch lease.
#elif !defined(HDB_NATIVE_CORE_TEST) || defined(HDB_NATIVE_PACKET_GATE_TEST)
    if (lazy_fusion::IsEnabled()) return false;
#endif
#ifndef HDB_NATIVE_CORE_TEST
    auto& keys = scope.prepared_owner_keys;
    auto& misses = scope.prepared_owner_misses;
    keys.clear();
    misses.clear();
    struct ClearMisses {
        decltype(scope.prepared_owner_misses)& values;
        ~ClearMisses() noexcept { values.clear(); }
    } clear_misses{misses};
    if (ni > std::numeric_limits<size_t>::max() - no || ni + no > std::numeric_limits<size_t>::max() - nt)
        throw std::bad_alloc();
    keys.reserve(ni + no + nt);
    misses.reserve(ni + no + nt);
    uint64_t full_guards = 0;
    const auto* certified = scope.block && scope.block->packet && !scope.block->sealed ?
        &scope.block->packet_owners : nullptr;
    const auto inspect = [&keys, &misses, certified](const at::Tensor* values, size_t count) {
        return packet_detail::CollectOwnerSources(values, count, certified, keys, misses,
            [](const at::Tensor& value) { return CapturedStorageSourceKey(value, true); });
    };
    // Full source qualification precedes any holder construction. Mixed owners
    // therefore keep their previous Tensor ownership and destruction timing.
    if (!inspect(inputs, ni) || !inspect(outputs, no) || !inspect(temporaries, nt)) return false;
    if (!packet_detail::AuditUniqueOwnerMisses(misses, [&full_guards](const at::Tensor& value) {
        ++full_guards;
        if (scope.run->real_opapi) ValidateOwnedOpApiTensor(value);
        return CanCreateCapturedStorageOwner(value);
    })) return false;
#else
    // The actual Capture TU is also compiled by narrow queue tests, where Tensor
    // is intentionally incomplete. Empty-owner plain commands exercise packets.
    if (ni || no || nt || inputs || outputs || temporaries) return false;
#endif
    if (scope.block && !scope.block->packet && !scope.block->commands.empty()) flush_block();
    if (!scope.block) scope.block = BlockPtr(new Block(scope.batch_limit));
    initialize_real_block(*scope.block);
    if (!scope.block->packet) {
        scope.block->packet = true;
        packet_detail::counters.blocks.fetch_add(1, std::memory_order_relaxed);
        packet_detail::counters.blocks_live.fetch_add(1, std::memory_order_relaxed);
    }
#ifndef HDB_NATIVE_CORE_TEST
    scope.prepared_owner_checkpoint = scope.block->packet_owners.checkpoint();
    scope.block->packet_owner_full_guards += full_guards;
#endif
    scope.packet_prepared = true;
    try {
#ifndef HDB_NATIVE_CORE_TEST
        scope.prepared_owner_ids.reserve(keys.size());
        size_t position = 0;
        const auto retain = [&](const at::Tensor* values, size_t count) {
            for (size_t i = 0; i < count; ++i, ++position) {
                bool hit = false;
                ++scope.block->packet_owner_requests;
                const auto id = scope.block->packet_owners.retain(keys[position], [&] {
                    auto owner = storage_owner_detail::Make(values[i].unsafeGetTensorImpl()->storage());
                    if (scope.run->real_opapi) {
                        const auto bytes = values[i].unsafeGetTensorImpl()->storage().nbytes();
                        reserve_owned_bytes(bytes); scope.block->retained_bytes += bytes;
                    }
                    return owner;
                }, hit);
                if (hit) ++scope.block->packet_owner_hits;
                else ++scope.block->packet_owner_created;
                // One use per command is sufficient, even repeated aliased args.
                if (std::find(scope.prepared_owner_ids.begin(), scope.prepared_owner_ids.end(), id) ==
                    scope.prepared_owner_ids.end()) scope.prepared_owner_ids.push_back(id);
            }
        };
        retain(inputs, ni); retain(outputs, no); retain(temporaries, nt);
#endif
    } catch (...) { CancelCapturedPacketCommand(); throw; }
    return true;
}

void RejectEffect(const char* effect)
{
    if (!IsCapturing()) return;
    scope.failed = true;
    state().counts.unsupported.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error(std::string("unsupported effect inside native region capture: ") + effect);
}

#ifndef HDB_NATIVE_CORE_TEST
at::Tensor TryRetainCapturedStorageOwner(const at::Tensor& original, bool plain_acl)
{
    if (!IsCapturing()) return {};
    if (scope.block && scope.block->packet) {
        flush_block();
        if (scope.block && scope.block->commands.empty()) scope.block.reset();
    }
    // OpCommand retains buffers before its first call to Capture. Allocation
    // failures here leave the original exported ACL/source ownership intact.
    if (!scope.block) scope.block = BlockPtr(new Block(scope.batch_limit));
    return scope.block->owner_cache.retain(original, plain_acl);
}
#endif

#ifdef HDB_NATIVE_CORE_TEST
void SetOwnedOpApiCompletionHooksForTest(device_retirement_detail::CompletionHooks hooks) noexcept { completion_hooks = hooks; }
#endif
bool UseCapturedOwnedOpApi() noexcept {
    return scope.run && scope.run->real_opapi && scope.task_tls && !scope.failed && !scope.emit_stopped &&
        !scope.run->aborted.load(std::memory_order_acquire) && !real_counts.failed.load(std::memory_order_acquire);
}
bool OwnedOpApiEnabled() noexcept { return scope.run && scope.run->real_opapi && scope.task_tls; }
#ifndef HDB_NATIVE_CORE_TEST
void ValidateOwnedOpApiTensor(const at::Tensor& value) {
    if (!UseCapturedOwnedOpApi()) throw std::runtime_error("owned OpAPI source outside certified real scope");
    auto* storage = CapturedStorageSourceKey(value, true);
    if (!storage || value.device().index() != 0 || !CanCreateCapturedStorageOwner(value))
        throw std::runtime_error("owned OpAPI requires native device0 Storage without Python metadata/hooks");
    if (!c10_npu::NPUCachingAllocator::IsNativeCapturedRetirementDataOnStream(
            storage->data_ptr(), storage->allocator(), 0, c10_npu::getDefaultNPUStream(0).stream(false)))
        throw std::runtime_error("owned OpAPI allocation stream differs from actual default execution stream");
}
#endif
void reserve_owned_bytes(size_t bytes) {
    auto retained = real_counts.retained_bytes.load(std::memory_order_relaxed);
    do {
        if (bytes > owned_opapi_retained_byte_limit || retained > owned_opapi_retained_byte_limit - bytes)
            throw std::runtime_error("owned OpAPI retained-byte budget exceeded");
    } while (!real_counts.retained_bytes.compare_exchange_weak(retained, retained+bytes));
}
void reserve_owned_plan() {
    auto live = real_counts.payloads_live.load(std::memory_order_relaxed);
    do {
        if (live >= 128) throw std::runtime_error("owned OpAPI prepared-plan budget exceeded");
    } while (!real_counts.payloads_live.compare_exchange_weak(live, live+1));
}
static int execute_owned(Command* command) {
    if (real_counts.failed.load(std::memory_order_acquire) || command->parameter.paramStream != command->block->execution_stream)
        throw std::runtime_error("device retirement failed or actual submission stream changed");
    ++real_counts.submit_attempts;
    auto observe = [&]() noexcept {
        if (command->owned_opapi->MayHaveSubmitted() && !command->block->may_have_submitted.exchange(true, std::memory_order_acq_rel))
            ++real_counts.device_pending;
    };
    try {
        const int result = command->owned_opapi->Execute(command->block->execution_stream);
        observe(); ++real_counts.submit_returns; return result;
    } catch (...) { observe(); throw; }
}
bool CaptureOwnedOpApi(const char* name, std::unique_ptr<OwnedOpApiPayload>& payload, int device) {
    if (!UseCapturedOwnedOpApi()) return false;
    if (!payload || !name || device != scope.device) throw std::runtime_error("invalid owned OpAPI command");
    if (scope.block && !scope.block->device_fence) {
        if (!scope.block->commands.empty()) flush_block(); else scope.block.reset();
    }
    if (!scope.block) {
        auto block = BlockPtr(new Block(scope.batch_limit));
        initialize_real_block(*block);
        block->packet = true; ++packet_detail::counters.blocks; ++packet_detail::counters.blocks_live;
        scope.block = std::move(block);
    }
    auto& block = *scope.block;
    size_t bytes = payload->RetainedBytes();
    reserve_owned_bytes(bytes);
    Command* appended = nullptr; bool reserved_plan = false;
    try {
        reserve_owned_plan(); reserved_plan = true;
        appended = &block.commands.append(); auto& command = *appended;
        command.device = device; command.region = scope.region; command.local = scope.local;
        command.block = &block; command.packet = true;
        std::strncpy(command.opapi.opType, name, sizeof(command.opapi.opType)-1);
        command.opapi.customHandler = [&command]() { return execute_owned(&command); };
        command.parameter.paramType = EXECUTE_OPAPI; command.parameter.paramLen = sizeof(ExecuteParasOpApi); command.parameter.paramVal = &command.opapi;
        command.parameter.paramStream = block.execution_stream;
        payload->Prepare(block.execution_stream);
        const auto prepared_bytes = payload->RetainedBytes();
        if (prepared_bytes > bytes) { reserve_owned_bytes(prepared_bytes-bytes); bytes = prepared_bytes; }
        command.owned_opapi = std::move(payload); // Last: all possibly throwing preparation precedes adoption.
        block.retained_bytes += bytes; ++scope.local;
        ++real_counts.captured; ++block.opapi; ++block.other_operators; ++block.packet_commands;
    } catch (...) { if (appended) block.commands.pop_back(); if (reserved_plan) --real_counts.payloads_live; real_counts.retained_bytes.fetch_sub(bytes); throw; }
    record_real_worker(false);
    auto& c = state().counts; ++c.captured; maximum(c.max_live, c.commands_live.fetch_add(1)+1);
    if (block.commands.size() == scope.batch_limit) flush_block();
    return true;
}

bool Capture(void* raw, int device)
{
    if (!IsCapturing()) return false;
    if (device != scope.device) RejectEffect("device change");
    auto* source = static_cast<QueueParas*>(raw);
    const auto type = source->paramType;
    if (scope.run && scope.run->real_opapi && type != COMPILE_AND_EXECUTE)
        RejectEffect("unowned/type-erased OpAPI callable in real capture");
    const bool packet_command = scope.packet_prepared;
    if (scope.run && scope.run->real_opapi && !packet_command) RejectEffect("uncertified legacy command in real capture");
    if (packet_command && (type != COMPILE_AND_EXECUTE ||
        static_cast<ExecuteParas*>(source->paramVal)->customHandler ||
        !static_cast<ExecuteParas*>(source->paramVal)->hostMemory.empty())) {
        CancelCapturedPacketCommand();
        throw std::logic_error("prepared native packet requires plain legacy without hostMemory");
    }
    if (type != COMPILE_AND_EXECUTE && type != EXECUTE_OPAPI && type != EXECUTE_OPAPI_V2) {
        RejectEffect("copy/event/unknown queue command");
    }
    if (scope.failed && !scope.emit_stopped) {
        // A caller may catch an unsupported-effect exception. Preserve only
        // commands preceding that error, then discard any continuation.
        flush_block();
        scope.emit_stopped = true;
    }
    // All vector allocation precedes the shallow ACL ownership transfer.
    if (scope.block && scope.block->packet != packet_command) {
        if (!scope.block->commands.empty()) flush_block();
        else scope.block.reset();
    }
    if (!scope.block) {
        scope.block = BlockPtr(new Block(scope.batch_limit));
    }
    auto& command = scope.block->commands.append();
    try {
        command.device = device;
        command.region = scope.region;
        command.local = scope.local++;
        command.parameter.paramStream = source->paramStream;
        command.parameter.correlation_id = source->correlation_id;
        if (type == COMPILE_AND_EXECUTE) {
            auto* original = static_cast<ExecuteParas*>(source->paramVal);
            const auto owners = original->hostMemory.size();
            // This is the last allocating step. On failure source still owns
            // all ACL objects, and the prepared holder checkpoint can cancel.
            if (packet_command) command.owner_ids = packet_detail::CopyOwnerIds(
                scope.block->parameter_arena, scope.prepared_owner_ids);
            const bool adopted = adopt_captured_legacy(command.legacy, *original);
            command.owns_acl = true;
            command.packet = packet_command;
            command.block = scope.block.get();
            if (packet_command) {
#ifndef HDB_NATIVE_CORE_TEST
                scope.block->packet_owners.add_uses(command.owner_ids);
#endif
                ++scope.block->packet_commands;
                scope.block->packet_owner_ids += command.owner_ids.size();
                scope.prepared_owner_ids.clear();
                scope.packet_prepared = false;
            }
            command.parameter.paramType = COMPILE_AND_EXECUTE;
            command.parameter.paramLen = sizeof(ExecuteParas);
            command.parameter.paramVal = &command.legacy;
            ++scope.block->legacy;
            if (adopted) {
                ++scope.block->capture_legacy_adopts;
                scope.block->capture_legacy_owner_moves += owners;
            }
        } else {
            if (type == EXECUTE_OPAPI_V2) {
                command.opapi.Copy(*static_cast<ExecuteParasOpApiV2*>(source->paramVal));
                ++scope.block->v2;
            } else {
                command.opapi.Copy(*static_cast<ExecuteParasOpApi*>(source->paramVal));
                ++scope.block->opapi;
            }
            command.parameter.paramType = EXECUTE_OPAPI;
            command.parameter.paramLen = sizeof(ExecuteParasOpApi);
            command.parameter.paramVal = &command.opapi;
        }
    } catch (...) {
        scope.block->commands.pop_back();
        throw;
    }
    const char* name = type == COMPILE_AND_EXECUTE ? command.legacy.opType : command.opapi.opType;
    if (std::strcmp(name, "EmbeddingBag") == 0) ++scope.block->embedding_bag;
    else if (std::strcmp(name, "Cast") == 0) ++scope.block->cast;
    else ++scope.block->other_operators;
    auto& c = state().counts;
    c.captured.fetch_add(1, std::memory_order_relaxed);
    maximum(c.max_live, c.commands_live.fetch_add(1, std::memory_order_relaxed) + 1);
    if (scope.block->commands.size() == scope.batch_limit || scope.emit_stopped) flush_block();
    return true;
}

bool IsPublishingCaptured(void* raw) noexcept
{
    return publishing && raw == &publishing->parameter && !publishing->accepted &&
           !publishing->packet && publishing->owns_acl && publishing->parameter.paramType == COMPILE_AND_EXECUTE &&
           !publishing->legacy.customHandler;
}

bool IsPublishingPacketCommand(void* raw) noexcept
{
    return publishing && raw == &publishing->parameter && publishing->packet;
}
bool AcquirePublishingPacketCommand(void* raw, PacketCommandRef& output) noexcept
{
    if (!IsPublishingPacketCommand(raw) || publishing->lease_acquired || output.packet) return false;
    auto* block = publishing->block;
    if (!block || !block->sealed || !block->in_flight || !block->lease.accept(publishing->token)) return false;
    const auto index = static_cast<uint32_t>(publishing - block->commands.begin());
    output = {block, index, publishing->token};
    publishing->lease_acquired = true;
    if (publishing->owned_opapi) ++real_counts.leases_acquired;
    packet_detail::counters.leases_acquired.fetch_add(1, std::memory_order_relaxed);
    return true;
}
static Command* packet_command(const PacketCommandRef& ref) noexcept
{
    if (!ref.packet || !ref.generation) return nullptr;
    auto* block = static_cast<Block*>(ref.packet);
    if (!block->packet || !block->sealed || ref.index >= block->commands.size()) return nullptr;
    auto& command = block->commands[ref.index];
    return command.packet && command.token == ref.generation ? &command : nullptr;
}
void* PacketCommandParameter(const PacketCommandRef& ref) noexcept
{
    auto* command = packet_command(ref);
    return command ? (command->owned_opapi ? static_cast<void*>(&command->opapi) : static_cast<void*>(&command->legacy)) : nullptr;
}
void PacketCommandExecutionAttempted(const PacketCommandRef& ref) noexcept
{
    if (auto* command = packet_command(ref)) {
        if (!command->execution_attempted.exchange(true, std::memory_order_acq_rel)) {
            packet_detail::counters.execution_attempts.fetch_add(1, std::memory_order_relaxed);
            if (command->block->device_fence && !command->owned_opapi &&
                !command->block->may_have_submitted.exchange(true, std::memory_order_acq_rel)) ++real_counts.device_pending;
        }
    }
}
bool BeginPacketCommandRelease(const PacketCommandRef& ref) noexcept
{
    auto* command = packet_command(ref);
    if (!command || command->release_attempted.exchange(true, std::memory_order_acq_rel)) return false;
    packet_detail::counters.release_attempts.fetch_add(1, std::memory_order_relaxed);
    return true;
}
void FinishPacketCommandRelease(PacketCommandRef& ref) noexcept
{
    auto* command = packet_command(ref);
    auto* block = static_cast<Block*>(ref.packet);
    ref = {};
    if (!command || command->retired.exchange(true, std::memory_order_acq_rel)) return;
    command->owns_acl = false;
#ifndef HDB_NATIVE_CORE_TEST
    if (!block->device_fence) block->packet_owners.retire(command->owner_ids);
#endif
    // Only accepted commands took a count. Suffix cancellation shares the same
    // owner-use retirement but never consumes a queue admission it did not own.
    if (command->lease_acquired) {
        packet_detail::counters.retired.fetch_add(1, std::memory_order_relaxed);
        packet_detail::counters.leases_retired.fetch_add(1, std::memory_order_relaxed);
        block->lease.retire();
    }
}

void ReleasePacketCommand(PacketCommandRef& ref) {
    auto* command = packet_command(ref);
    if (!command) { ref = {}; return; }
    const bool real = bool(command->block->device_fence);
    const auto cleanup = [&] {
        struct Finish { PacketCommandRef& ref; ~Finish() noexcept { FinishPacketCommandRelease(ref); } } finish{ref};
        if (!BeginPacketCommandRelease(ref)) return;
        if (command->owned_opapi) { cancel_owned_payload(*command); return; }
#ifndef HDB_NATIVE_CORE_TEST
        ReleaseLegacyParameters(command->legacy, ParameterArrayOwnership::PacketArena);
#else
        command->legacy.Release();
#endif
    };
    cleanup();
    if (real && real_counts.failed.load(std::memory_order_acquire)) rethrow_retirement_failure();
}

void CheckCapturedPublishAccepted(void* raw)
{
    if (publishing && raw == &publishing->parameter && !publishing->accepted)
        throw std::runtime_error("original NPU queue did not accept command");
}

void BeforeQueuePublish(void* raw)
{
    if (!publishing || raw != &publishing->parameter) return;
    auto& data = state();
    if (!data.diagnostics) return;
    auto* parameter = static_cast<QueueParas*>(raw);
    std::lock_guard<std::mutex> lock(data.mutex);
    // Allocate optional metadata before original queue ownership acceptance.
    data.pending.emplace(parameter->correlation_id,
        Order{publishing->token, publishing->region, publishing->local,
              data.counts.published.load(std::memory_order_relaxed) + 1});
    publishing->registered = true;
}

void QueueCopyAccepted(void* raw) noexcept
{
    const bool captured = publishing && raw == &publishing->parameter;
    // Functional ACK always precedes diagnostic/activation tests. It runs in
    // the original queue copy callback before consumer visibility, even if a
    // later queue wait or wrapper throws after this point.
    if (captured) {
        publishing->accepted = true;
        if (publishing->owned_opapi) ++real_counts.admission_acked;
        if (!publishing->packet) publishing->owns_acl = false;
        else packet_detail::counters.accepted.fetch_add(1, std::memory_order_relaxed);
    }
    auto& data = state();
    if (!data.ever_enabled.load(std::memory_order_acquire)) return;
    auto& c = data.counts;
    c.queue_copy_total.fetch_add(1, std::memory_order_relaxed);
    if (captured) {
        c.published.fetch_add(1, std::memory_order_relaxed);
        c.commands_live.fetch_sub(1, std::memory_order_relaxed);
    }
    if (!data.diagnostics) return;
    const auto* parameter = static_cast<QueueParas*>(raw);
    std::lock_guard<std::mutex> lock(data.mutex);
    if (data.trace && data.queue_published_trace.size() < 4096) {
        const auto normalized = parameter->paramType == EXECUTE_OPAPI_V2 ? EXECUTE_OPAPI : parameter->paramType;
        data.queue_published_trace.push_back({parameter->correlation_id, static_cast<uint64_t>(normalized)});
    }
    if (!captured) return;
    const auto found = data.pending.find(parameter->correlation_id);
    if (found != data.pending.end()) c.published_hash = mix(c.published_hash, found->second);
}

void QueueConsumed(void* raw, int result) noexcept
{
    auto& data = state();
    if (!data.ever_enabled.load(std::memory_order_acquire)) return;
    auto& c = data.counts;
    c.queue_consume_total.fetch_add(1, std::memory_order_relaxed);
    if (result != 0) c.failures.fetch_add(1, std::memory_order_relaxed);
    if (!data.diagnostics) return;
    const auto* parameter = static_cast<QueueParas*>(raw);
    std::lock_guard<std::mutex> lock(data.mutex);
    if (data.trace && data.queue_consumed_trace.size() < 4096) {
        data.queue_consumed_trace.push_back({parameter->correlation_id, static_cast<uint64_t>(parameter->paramType)});
    }
    auto found = data.pending.find(parameter->correlation_id);
    if (found == data.pending.end()) return;
    const auto order = found->second;
    if (order.ordinal != c.consumed.load() + c.queue_discarded.load() + c.queue_abandoned.load() + 1) ++c.fifo_errors;
    ++c.consumed;
    c.consumed_hash = mix(c.consumed_hash, order);
    if (data.trace && data.consumed_trace.size() < 4096) data.consumed_trace.push_back(order);
    data.pending.erase(found);
}

void QueueReleased(void* raw) noexcept
{
    auto& data = state();
    if (!data.ever_enabled.load(std::memory_order_acquire)) return;
    data.counts.queue_release_total.fetch_add(1, std::memory_order_relaxed);
    if (!data.diagnostics) return;
    const auto* parameter = static_cast<QueueParas*>(raw);
    std::lock_guard<std::mutex> lock(data.mutex);
    if (data.pending.erase(parameter->correlation_id)) ++data.counts.queue_discarded;
}

void QueueAbandoned(int device) noexcept
{
    if (device != 0) return;
    auto& data = state();
    if (!data.ever_enabled.load(std::memory_order_acquire)) return;
    data.counts.queue_abandon_events.fetch_add(1, std::memory_order_relaxed);
    if (!data.diagnostics) return;
    std::lock_guard<std::mutex> lock(data.mutex);
    // Metadata abandonment after a poisoned queue is weaker than clean release.
    data.counts.queue_abandoned += data.pending.size();
    data.pending.clear();
}
} // namespace host_regions
} // namespace c10_npu

extern "C" int hdb_region_activate() noexcept
{
    try { state().ever_enabled.store(true, std::memory_order_release); return 0; }
    catch (...) { fail("native region accounting activation failed"); return -1; }
}

extern "C" int hdb_region_begin(HdbRegionEmit emit, void* context, int device, uint64_t region) noexcept
{
    return begin_scope(emit, nullptr, context, device, region, 1);
}

extern "C" int hdb_region_begin_batch(HdbRegionBatchEmit emit, void* context, int device,
                                      uint64_t region, uint32_t batch_limit) noexcept
{
    return begin_scope(nullptr, emit, context, device, region, batch_limit);
}

extern "C" int hdb_region_end() noexcept
{
    if (!scope.open()) { fail("native region scope is not open"); return -1; }
    // Body/unsupported-effect errors keep their successful prefix. Only an
    // emitter failure prevents further callbacks and discards its suffix.
    flush_block();
    const bool failed = scope.failed;
    auto run = scope.run;
    const auto region_index = scope.region;
    auto task_tls = std::move(scope.task_tls);
    scope = Scope{};
    task_tls.reset(); // Restore worker TLS after capture closes and outside all channel locks.
    if (run) {
        hdb_native::WaitDetach detached;
        std::lock_guard<std::mutex> control(run->control);
        auto& region = *run->regions[static_cast<size_t>(region_index)];
        std::lock_guard<std::mutex> lock(region.mutex);
        region.failed = region.failed || failed;
        region.phase = RegionPhase::Ended;
        --run->active_tasks;
        region.changed.notify_all();
    }
    auto& c = state().counts;
    c.scopes_closed.fetch_add(1, std::memory_order_relaxed);
    c.active_scopes.fetch_sub(1, std::memory_order_relaxed);
    if (failed) {
        ++c.failures;
        if (last_error.empty()) fail("native region scope rejected an unsupported effect");
    }
    return failed ? -1 : 0;
}

extern "C" int hdb_region_publish(uint64_t token) noexcept
{
    BlockPtr block;
    try {
        last_error.clear();
        if (scope.open() || publishing) throw std::runtime_error("cannot publish inside native capture");
        block = claim(token);
        publish_block(block);
        return 0;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown native region publication failure"); }
    publishing = nullptr;
    if (block) {
        discard_unaccepted(*block);
        ++state().counts.failures;
    }
    return -1;
}

extern "C" int hdb_region_discard(uint64_t token) noexcept
{
    try {
        last_error.clear();
        auto block = claim(token);
        discard_unaccepted(*block);
        return 0;
    } catch (const std::exception& error) { fail(error.what()); return -1; }
    catch (...) { fail("unknown native region discard failure"); return -1; }
}

namespace {
uint64_t create_native_run(uint32_t regions, uint32_t capacity, uint32_t batch_limit, bool plain_builder, bool real_opapi = false) noexcept
{
    try {
        last_error.clear();
        if (real_opapi) {
            const char *real = std::getenv("HDB_REGION_REAL_OPAPI"), *packet = std::getenv("HDB_REGION_PACKET"), *builder = std::getenv("HDB_REGION_NATIVE_BUILDER");
            if (!real || std::strcmp(real,"1") || !packet || std::strcmp(packet,"1") || (builder && !std::strcmp(builder,"1")))
                throw std::runtime_error("real OpAPI requires HDB_REGION_REAL_OPAPI=1, HDB_REGION_PACKET=1 and legacy builder disabled");
            check_configuration(0, true);
#ifdef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
            throw std::runtime_error("real OpAPI requires the original ordered queue");
#elif !defined(HDB_NATIVE_CORE_TEST) || defined(HDB_NATIVE_PACKET_GATE_TEST)
            if (lazy_fusion::IsEnabled()) throw std::runtime_error("real OpAPI requires lazy fusion disabled");
#endif
        }
        if (plain_builder) {
            const char* builder = std::getenv("HDB_REGION_NATIVE_BUILDER");
            if (!builder || std::strcmp(builder, "1") != 0)
                throw std::runtime_error("native builder creation requires HDB_REGION_NATIVE_BUILDER=1");
            const char* packet = std::getenv("HDB_REGION_PACKET");
            if (!packet || std::strcmp(packet, "1") != 0)
                throw std::runtime_error("native builder creation requires HDB_REGION_PACKET=1");
#ifdef TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE
            throw std::runtime_error("native builder creation requires the ordered batch queue");
#elif !defined(HDB_NATIVE_CORE_TEST) || defined(HDB_NATIVE_PACKET_GATE_TEST)
            if (lazy_fusion::IsEnabled())
                throw std::runtime_error("native builder creation requires lazy fusion disabled");
#endif
        }
        if (scope.open() || publishing || !regions || regions > 256 || !capacity || capacity > 8192 ||
            !batch_limit || batch_limit > max_batch_limit || batch_limit > capacity)
            throw std::runtime_error("invalid native run bounds or active capture");
        auto run = std::make_shared<NativeRun>(regions, capacity, batch_limit, plain_builder, real_opapi);
        if (real_opapi && !run->caller.context()) throw std::runtime_error("real capture requires an initialized primary Context");
#ifndef HDB_NATIVE_CORE_TEST
        if (real_opapi && run->caller.context() != c10_npu::GetDeviceContext(0))
            throw std::runtime_error("real capture does not support an external/nonprimary Context");
        if (real_opapi) c10_npu::host_regions::PrepareOwnedOpApiCallerState();
#endif
        auto& data = state();
        uint64_t id;
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> lock(data.mutex);
            if (data.next_run == 0) throw std::runtime_error("native run ID space exhausted");
            id = data.next_run++;
            data.runs.emplace(id, run);
        }
        data.ever_enabled.store(true, std::memory_order_release);
        ++data.counts.native_runs_created;
        if (plain_builder) ++data.counts.native_builder_runs_created;
        if (real_opapi) ++real_counts.runs_created;
        return id;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown native run creation failure"); }
    return 0;
}
} // namespace

extern "C" uint64_t hdb_region_run_create(uint32_t regions, uint32_t capacity, uint32_t batch_limit) noexcept
{
    return create_native_run(regions, capacity, batch_limit, false);
}

extern "C" int hdb_region_run_create_out(uint32_t regions, uint32_t capacity, uint32_t batch_limit,
                                       uint64_t* output) noexcept
{
    if (!output) { fail("native run creation requires an output slot"); return -1; }
    *output = 0;
    const auto id = hdb_region_run_create(regions, capacity, batch_limit);
    if (!id) return -1;
    *output = id; // Caller-owned storage survives a Python signal while returning from FFI.
    return 0;
}

extern "C" int hdb_region_run_create_native_builder_out(uint32_t regions, uint32_t capacity, uint32_t batch_limit,
                                                      uint64_t* output) noexcept
{
    if (!output) { fail("native run creation requires an output slot"); return -1; }
    *output = 0;
    const auto id = create_native_run(regions, capacity, batch_limit, true);
    if (!id) return -1;
    *output = id;
    return 0;
}

extern "C" int hdb_region_run_create_real_opapi_out(uint32_t regions, uint32_t capacity, uint32_t batch_limit, uint64_t* output) noexcept
{
    if (!output) { fail("real native run requires an output slot"); return -1; }
    *output = 0;
    const auto id = create_native_run(regions, capacity, batch_limit, false, true);
    if (!id) return -1;
    *output = id; return 0;
}

extern "C" int hdb_region_run_begin(uint64_t id, uint32_t index) noexcept
{
    std::shared_ptr<NativeRun> run;
    bool reserved = false;
    try {
        last_error.clear();
        if (scope.open() || publishing) throw std::runtime_error("nested native region begin");
        run = find_run(id);
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> control(run->control);
            if (run->destroyed || run->aborted.load(std::memory_order_acquire) || index >= run->regions.size())
                throw std::runtime_error("invalid, destroyed or aborted native run region");
            auto& region = *run->regions[index];
            std::lock_guard<std::mutex> lock(region.mutex);
            if (region.phase != RegionPhase::New) throw std::runtime_error("native region was already begun or completed");
            region.phase = RegionPhase::Starting;
            ++run->active_tasks;
            reserved = true;
        }
        auto task = std::make_unique<hdb_native::TaskTls>(run->caller, run->real_opapi);
        check_configuration(run->caller.device(), run->real_opapi);
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> control(run->control);
            auto& region = *run->regions[index];
            std::lock_guard<std::mutex> lock(region.mutex);
            if (run->aborted.load(std::memory_order_acquire)) throw std::runtime_error("native run aborted during begin");
            region.phase = RegionPhase::Active;
            region.entered = true;
        }
        scope.run = run;
        scope.region = index;
        scope.device = run->caller.device();
        scope.batch_limit = run->batch_limit;
        scope.task_tls = std::move(task);
        if (run->real_opapi) record_real_worker(true);
        auto& c = state().counts;
        ++c.native_run_begins;
        c.scopes_opened.fetch_add(1, std::memory_order_relaxed);
        maximum(c.max_active_scopes, c.active_scopes.fetch_add(1, std::memory_order_relaxed) + 1);
        return 0;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown native run begin failure"); }
    if (reserved) {
        hdb_native::WaitDetach detached;
        std::lock_guard<std::mutex> control(run->control);
        auto& region = *run->regions[index];
        std::lock_guard<std::mutex> lock(region.mutex);
        region.phase = RegionPhase::Ended;
        region.failed = true;
        --run->active_tasks;
        region.changed.notify_all();
    }
    return -1;
}

extern "C" int hdb_region_run_done(uint64_t id, uint32_t index, int failed) noexcept
{
    try {
        last_error.clear();
        auto run = find_run(id);
        hdb_native::WaitDetach detached;
        std::lock_guard<std::mutex> control(run->control);
        if (run->destroyed || index >= run->regions.size() || (failed != 0 && failed != 1))
            throw std::runtime_error("invalid native region completion");
        auto& region = *run->regions[index];
        std::lock_guard<std::mutex> lock(region.mutex);
        if (region.phase == RegionPhase::Starting || region.phase == RegionPhase::Active || region.phase == RegionPhase::Done ||
            (!failed && (!region.entered || region.phase != RegionPhase::Ended)))
            throw std::runtime_error("native region completion must follow end exactly once");
        region.failed = region.failed || failed;
        region.phase = RegionPhase::Done;
        ++state().counts.native_run_done;
        region.changed.notify_all();
        return 0;
    } catch (const std::exception& error) { fail(error.what()); return -1; }
    catch (...) { fail("unknown native region completion failure"); return -1; }
}

extern "C" int hdb_region_run_drain(uint64_t id) noexcept
{
    std::shared_ptr<NativeRun> run;
    bool reserved = false;
    int result = -1;
    hdb_native::WaitDetach detached; // Covers all waits/publication; destroyed only after every lock.
    try {
        last_error.clear();
        if (scope.open() || publishing) throw std::runtime_error("cannot drain inside native capture/publication");
        run = find_run(id);
        {
            std::lock_guard<std::mutex> lock(run->control);
            if (run->destroyed || run->drain_started) throw std::runtime_error("native drain already started or run destroyed");
            run->drain_started = run->draining = true;
            reserved = true;
            ++state().counts.native_active_drains;
            ++state().counts.native_run_drains;
        }
        result = drain_native_run(run);
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown native run drain failure"); }
    if (reserved) {
        if (result < 0) { abort_run(run); ++state().counts.failures; }
        std::lock_guard<std::mutex> lock(run->control);
        run->draining = false;
        --state().counts.native_active_drains;
    }
    return result;
}

extern "C" int hdb_region_run_abort(uint64_t id) noexcept
{
    try {
        last_error.clear();
        auto run = find_run(id);
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> lock(run->control);
            if (run->destroyed) throw std::runtime_error("native run already destroyed");
        }
        abort_run(run);
        return 0;
    } catch (const std::exception& error) { fail(error.what()); return -1; }
    catch (...) { fail("unknown native run abort failure"); return -1; }
}

extern "C" int hdb_region_run_destroy(uint64_t id) noexcept
{
    try {
        last_error.clear();
        auto run = find_run(id);
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> lock(run->control);
            if (run->destroyed || run->active_tasks || run->draining)
                throw std::runtime_error("cannot destroy active or already destroyed native run");
            run->destroyed = true;
        }
        auto& data = state();
        {
            hdb_native::WaitDetach detached;
            std::lock_guard<std::mutex> lock(data.mutex);
            data.runs.erase(id); // Local shared owner prevents destruction under registry lock.
        }
        for (auto& region : run->regions) {
            // No scope/drain can start after destroyed; every owner dies outside locks.
            while (!region->blocks.empty()) {
                auto block = std::move(region->blocks.front());
                region->blocks.pop_front();
                const auto units = static_cast<uint32_t>(block->commands.size());
                region->queued -= units;
                --data.counts.native_queued_blocks;
                data.counts.native_queued_commands.fetch_sub(units, std::memory_order_relaxed);
                discard_unaccepted(*block);
            }
        }
        ++data.counts.native_runs_destroyed;
        if (run->plain_native_builder) ++data.counts.native_builder_runs_destroyed;
        if (run->real_opapi) ++real_counts.runs_destroyed;
        return 0;
    } catch (const std::exception& error) { fail(error.what()); return -1; }
    catch (...) { fail("unknown native run destroy failure"); return -1; }
}

extern "C" int hdb_region_run_destroy_out(uint64_t* output) noexcept
{
    if (!output) { fail("native run destruction requires an output slot"); return -1; }
    if (!*output) { last_error.clear(); return 0; }
    if (hdb_region_run_destroy(*output) != 0) return -1;
    *output = 0; // Destruction ACK is visible even if FFI return raises in Python.
    return 0;
}

extern "C" size_t hdb_region_error(char* output, size_t size) noexcept
{
    return copy_string(last_error, output, size);
}

extern "C" size_t hdb_region_stats(char* output, size_t size) noexcept
{
    try {
        auto& data = state();
        std::lock_guard<std::mutex> lock(data.mutex);
        const auto& c = data.counts;
        std::ostringstream text;
        text << "{\"abi\":1,\"batch_abi\":1,\"native_run_abi\":1,\"native_builder_abi\":1,\"backend\":\"torch_npu_hostsim_capture\"";
#define HDB_STAT(field) text << ",\"" #field "\":" << c.field.load(std::memory_order_relaxed)
        text << ",\"real_capture_abi\":1,\"opapi_owned_abi\":1,\"real_prepare_strategy\":\"" << real_prepare_strategy()
             << "\",\"real_capture_supported\":" << (real_sdk_available() && !real_counts.failed.load() ? "true" : "false");
#define REAL_STAT(field, name) text << ",\"" name "\":" << real_counts.field.load(std::memory_order_relaxed)
        REAL_STAT(runs_created,"real_runs_created"); REAL_STAT(runs_destroyed,"real_runs_destroyed");
        text << ",\"real_runs_live\":" << real_counts.runs_created.load()-real_counts.runs_destroyed.load();
        REAL_STAT(captured,"owned_opapi_captured"); REAL_STAT(leases_acquired,"owned_opapi_leases_acquired");
        REAL_STAT(admission_acked,"owned_opapi_admission_acked"); REAL_STAT(submit_attempts,"owned_opapi_submit_attempts");
        REAL_STAT(submit_returns,"owned_opapi_submit_returns"); REAL_STAT(device_retired,"owned_opapi_device_retired");
        REAL_STAT(cancelled,"owned_opapi_cancelled"); REAL_STAT(payloads_live,"owned_opapi_payloads_live");
        REAL_STAT(device_pending,"owned_opapi_device_pending"); REAL_STAT(retirement_failures,"owned_opapi_retirement_failures");
        REAL_STAT(eligible_fallback,"owned_opapi_eligible_fallback"); REAL_STAT(retained_bytes,"owned_opapi_retained_bytes");
        REAL_STAT(events_recorded,"owned_opapi_device_events_recorded"); REAL_STAT(events_completed,"owned_opapi_device_events_completed");
#undef REAL_STAT
        text << ",\"owned_opapi_diag_incomplete\":" << (data.real_diagnostics_incomplete ? "true" : "false") << ",\"real_worker_threads\":[";
        bool real_comma = false;
        for (const auto& worker : data.real_workers) {
            if (real_comma) text << ','; real_comma = true;
            text << "{\"tid\":" << worker.tid << ",\"regions\":" << worker.regions << ",\"owned_opapi_captured\":" << worker.captured << '}';
        }
        text << ']';
        HDB_STAT(captured); HDB_STAT(published); HDB_STAT(discarded);
        HDB_STAT(scopes_opened); HDB_STAT(scopes_closed); HDB_STAT(active_scopes); HDB_STAT(max_active_scopes);
        HDB_STAT(failures); HDB_STAT(unsupported); HDB_STAT(max_live); HDB_STAT(commands_live);
        HDB_STAT(legacy); HDB_STAT(opapi); HDB_STAT(v2);
        HDB_STAT(queue_copy_total); HDB_STAT(queue_consume_total);
        HDB_STAT(queue_release_total); HDB_STAT(queue_abandon_events);
        HDB_STAT(embedding_bag); HDB_STAT(cast); HDB_STAT(other_operators);
        HDB_STAT(batches_emitted); HDB_STAT(max_batch_size);
        HDB_STAT(python_callbacks); HDB_STAT(native_runs_created); HDB_STAT(native_runs_destroyed);
        HDB_STAT(native_builder_runs_created); HDB_STAT(native_builder_runs_destroyed);
        text << ",\"native_builder_runs_live\":" << std::count_if(data.runs.begin(), data.runs.end(),
            [](const auto& entry) { return entry.second->plain_native_builder; });
        HDB_STAT(native_run_begins); HDB_STAT(native_run_done); HDB_STAT(native_run_drains);
        HDB_STAT(native_active_drains); HDB_STAT(native_queued_blocks); HDB_STAT(native_queued_commands);
        HDB_STAT(native_max_region_queued);
        HDB_STAT(native_command_arena_allocations); HDB_STAT(native_command_arenas_live);
        HDB_STAT(native_command_slots_constructed); HDB_STAT(capture_legacy_adopts);
        HDB_STAT(capture_legacy_owner_moves); HDB_STAT(native_batch_publish_calls);
        const auto& pc = c10_npu::host_regions::packet_detail::counters;
#define HDB_PACKET_STAT(field) text << ",\"packet_" #field "\":" << pc.field.load(std::memory_order_relaxed)
        HDB_PACKET_STAT(parameter_allocations); HDB_PACKET_STAT(parameter_bytes_total);
        HDB_PACKET_STAT(parameter_bytes_live); HDB_PACKET_STAT(parameter_bytes_peak);
        HDB_PACKET_STAT(owner_requests); HDB_PACKET_STAT(owner_hits); HDB_PACKET_STAT(owner_created);
        HDB_PACKET_STAT(owner_full_guards);
        HDB_PACKET_STAT(owner_ids); HDB_PACKET_STAT(commands); HDB_PACKET_STAT(accepted); HDB_PACKET_STAT(retired);
        HDB_PACKET_STAT(blocks); HDB_PACKET_STAT(blocks_live);
        HDB_PACKET_STAT(leases_acquired); HDB_PACKET_STAT(leases_retired); HDB_PACKET_STAT(suffix_reclaimed);
        HDB_PACKET_STAT(release_attempts); HDB_PACKET_STAT(execution_attempts);
#undef HDB_PACKET_STAT
        c10_npu::host_regions::AppendNativeBindingStats(text);
#ifndef HDB_NATIVE_CORE_TEST
        const auto packet = c10_npu::host_regions::GetStorageOwnerPacketSnapshot();
        text << ",\"storage_owner_cache_requests\":" << packet.requests
             << ",\"storage_owner_cache_hits\":" << packet.hits
             << ",\"storage_owner_cache_misses\":" << packet.misses
             << ",\"storage_owner_source_rejects\":" << packet.source_rejects
             << ",\"storage_owner_factory_rejects\":" << packet.factory_rejects
             << ",\"storage_owner_packets_created\":" << packet.created
             << ",\"storage_owner_packets_live\":" << packet.live
             << ",\"storage_owner_cache_entries_live\":" << packet.cache_entries_live
             << ",\"storage_owner_cache_clears\":" << packet.cache_clears;
#else
        text << ",\"storage_owner_cache_requests\":0,\"storage_owner_cache_hits\":0,\"storage_owner_cache_misses\":0"
             << ",\"storage_owner_source_rejects\":0,\"storage_owner_factory_rejects\":0"
             << ",\"storage_owner_packets_created\":0,\"storage_owner_packets_live\":0"
             << ",\"storage_owner_cache_entries_live\":0,\"storage_owner_cache_clears\":0";
#endif
#ifndef HDB_NATIVE_CORE_TEST
        const auto retirement = c10_npu::host_regions::GetCapturedRetirementSnapshot();
        text << ",\"native_retirement_pending\":" << retirement.pending
             << ",\"native_retirement_retired\":" << retirement.retired
             << ",\"native_retirement_peak\":" << retirement.peak
             << ",\"native_retirement_transferred\":" << retirement.transferred
             << ",\"native_retirement_fallback\":" << retirement.fallback
             << ",\"native_retirement_handoffs\":" << retirement.handoffs;
#else
        text << ",\"native_retirement_pending\":0,\"native_retirement_retired\":0"
             << ",\"native_retirement_peak\":0,\"native_retirement_transferred\":0,\"native_retirement_fallback\":0"
             << ",\"native_retirement_handoffs\":0";
#endif
#ifndef HDB_NATIVE_CORE_TEST
        const auto queue_batch = c10_npu::GetNativeQueueBatchSnapshot();
        text << ",\"native_queue_batch_fast_calls\":" << queue_batch.fast_calls
             << ",\"native_queue_batch_committed_bursts\":" << queue_batch.committed_bursts
             << ",\"native_queue_batch_committed_items\":" << queue_batch.committed_items
             << ",\"native_queue_batch_fallback_calls\":" << queue_batch.fallback_calls;
#else
        // This narrow test compiles the real Capture TU, not the real queue core.
        text << ",\"native_queue_batch_fast_calls\":0,\"native_queue_batch_committed_bursts\":0"
             << ",\"native_queue_batch_committed_items\":0,\"native_queue_batch_fallback_calls\":0";
#endif
        if (data.diagnostics) {
            HDB_STAT(consumed); HDB_STAT(fifo_errors); HDB_STAT(queue_discarded); HDB_STAT(queue_abandoned);
            text << ",\"published_hash\":" << c.published_hash << ",\"consumed_hash\":" << c.consumed_hash
                 << ",\"pending_consumption\":" << data.pending.size();
        } else {
            text << ",\"consumed\":null,\"fifo_errors\":null,\"queue_discarded\":null,\"queue_abandoned\":null"
                 << ",\"published_hash\":null,\"consumed_hash\":null,\"pending_consumption\":null";
        }
#undef HDB_STAT
        const auto consumed = c.queue_consume_total.load(std::memory_order_relaxed);
        const auto copied = c.queue_copy_total.load(std::memory_order_relaxed);
        text << ",\"queue_pending_total\":" << (copied >= consumed ? copied - consumed : 0)
             << ",\"queue_pending_semantics\":\"copy_minus_consume_includes_error_suffix\""
             << ",\"diagnostics_enabled\":" << (data.diagnostics ? "true" : "false")
             << ",\"captured_consumption_tracked\":" << (data.diagnostics ? "true" : "false")
             << ",\"queue_tracking_enabled\":" << (data.ever_enabled.load() ? "true" : "false")
             << ",\"tokens_live\":" << data.tokens.size()
             << ",\"native_runs_live\":" << data.runs.size() << ",\"trace\":[";
        bool comma = false;
        for (const auto& item : data.consumed_trace) {
            if (comma) text << ',';
            comma = true;
            text << '[' << item.token << ',' << item.region << ',' << item.local << ',' << item.ordinal << ']';
        }
        text << "],\"queue_published_trace\":[";
        comma = false;
        for (const auto& item : data.queue_published_trace) {
            if (comma) text << ',';
            comma = true;
            text << '[' << item[0] << ',' << item[1] << ']';
        }
        text << "],\"queue_consumed_trace\":[";
        comma = false;
        for (const auto& item : data.queue_consumed_trace) {
            if (comma) text << ',';
            comma = true;
            text << '[' << item[0] << ',' << item[1] << ']';
        }
        text << "]}";
        return copy_string(text.str(), output, size);
    } catch (...) { return copy_string("{\"error\":\"native stats failed\"}", output, size); }
}
