#pragma once

#include <cstddef>
#include <cstdint>
#include "torch_npu/csrc/core/npu/HostRegionPacket.h"
#include "torch_npu/csrc/core/npu/HostRegionOpApiCommand.h"

namespace at { class Tensor; }

namespace c10_npu {
namespace host_regions {
// The first backend deliberately admits the host simulator only. These hooks
// are dormant until a Python region explicitly opens a thread-local scope.
bool IsCapturing() noexcept;
// Only an explicitly certified plain-intrinsic native run may stage bindings.
// Old run creation APIs never inherit this capability from environment flags.
bool UseCapturedNativeBindings() noexcept;
// The caller prechecks plain legacy/no exported hostMem before preparing.
// Source eligibility is audited for every Tensor, including table hits.
bool PrepareCapturedPacketCommand(const at::Tensor* inputs, size_t input_count,
    const at::Tensor* outputs, size_t output_count, const at::Tensor* temporaries, size_t temporary_count);
bool PreparedCapturedPacketCommand() noexcept;
void* AllocateCapturedPacketParams(size_t bytes, size_t alignment);
void CancelCapturedPacketCommand() noexcept;
// Undefined means use the original r12 retainer; only the current block caches.
at::Tensor TryRetainCapturedStorageOwner(const at::Tensor& original, bool plain_acl);
void RejectEffect(const char* effect);
bool Capture(void* queue_parameter, int device);
// Only the current, not-yet-ACKed plain legacy capture may transfer its owners.
bool IsPublishingCaptured(void* queue_parameter) noexcept;
// Batch fallbacks must stop at the first captured item without a functional ACK.
// Ordinary batch items (no current matching capture publication) are unaffected.
void CheckCapturedPublishAccepted(void* queue_parameter);
void BeforeQueuePublish(void* queue_parameter);
void QueueCopyAccepted(void* queue_parameter) noexcept;
void QueueConsumed(void* queue_parameter, int result) noexcept;
void QueueReleased(void* queue_parameter) noexcept;
void QueueAbandoned(int device) noexcept;
} // namespace host_regions
} // namespace c10_npu

#if defined(_WIN32)
#define HDB_REGION_EXPORT __declspec(dllexport)
#else
#define HDB_REGION_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
using HdbRegionEmit = int (*)(uint64_t token, void* context);
using HdbRegionBatchEmit = int (*)(uint64_t token, uint32_t command_count, void* context);
// Enable whole-queue accounting before provider setup publishes ordinary work.
// This does not initialize a device or change its queue/stream configuration.
HDB_REGION_EXPORT int hdb_region_activate() noexcept;
HDB_REGION_EXPORT int hdb_region_begin(HdbRegionEmit emit, void* context,
                                      int device, uint64_t region) noexcept;
// A bounded block costs command_count original-command credits. The old begin
// remains the size-one compatibility ABI. Valid batch_limit is 1..256.
HDB_REGION_EXPORT int hdb_region_begin_batch(HdbRegionBatchEmit emit, void* context,
                                          int device, uint64_t region,
                                          uint32_t batch_limit) noexcept;
HDB_REGION_EXPORT int hdb_region_end() noexcept;
// Tokens are single-use. Every successful capture must be published or
// discarded; publication consumes the token even when the original queue errs.
// A block publication error preserves its queue-accepted prefix and releases
// only the never-accepted suffix; no member of a consumed token may be retried.
HDB_REGION_EXPORT int hdb_region_publish(uint64_t token) noexcept;
HDB_REGION_EXPORT int hdb_region_discard(uint64_t token) noexcept;
// Per-invocation native handoff. Blocks never enter the callback token map.
// run_create captures audited caller native TLS; begin/end install/restore it.
HDB_REGION_EXPORT uint64_t hdb_region_run_create(uint32_t regions, uint32_t capacity, uint32_t batch_limit) noexcept;
// Cold out-slot form preserves a recoverable handle across Python FFI-return interrupts.
// Non-null output is zeroed before creation and receives the handle before success return.
HDB_REGION_EXPORT int hdb_region_run_create_out(uint32_t regions, uint32_t capacity, uint32_t batch_limit,
                                             uint64_t* output) noexcept;
// Provider contract established before any operand: plain native intrinsics,
// immutable bindings/configuration, and no late custom/host/scalar/sync effects.
// Requires HDB_REGION_NATIVE_BUILDER=1 and HDB_REGION_PACKET=1. Out-slot failure
// and interrupt recovery are identical to the original create_out operation.
HDB_REGION_EXPORT int hdb_region_run_create_native_builder_out(uint32_t regions, uint32_t capacity,
    uint32_t batch_limit, uint64_t* output) noexcept;
// Explicit real-device, typed-owned OpAPI capability. Legacy creation APIs do
// not inherit it from flags; absence of an exact SDK lifecycle contract rejects.
HDB_REGION_EXPORT int hdb_region_run_create_real_opapi_out(uint32_t regions, uint32_t capacity,
    uint32_t batch_limit, uint64_t* output) noexcept;
HDB_REGION_EXPORT int hdb_region_run_begin(uint64_t run, uint32_t region) noexcept;
// Python stores its body/end error before marking done. failed must be 0 or 1.
HDB_REGION_EXPORT int hdb_region_run_done(uint64_t run, uint32_t region, int failed) noexcept;
// 0 success, source region+1 for worker failure after its prefix, -1 native error.
HDB_REGION_EXPORT int hdb_region_run_drain(uint64_t run) noexcept;
HDB_REGION_EXPORT int hdb_region_run_abort(uint64_t run) noexcept;
// Caller must join every submitted worker first. Active tasks/drain reject.
HDB_REGION_EXPORT int hdb_region_run_destroy(uint64_t run) noexcept;
// Successful destruction clears the caller-owned slot before return; zero is idempotent.
// Unknown/active nonzero IDs remain failures and do not clear the slot.
HDB_REGION_EXPORT int hdb_region_run_destroy_out(uint64_t* output) noexcept;
HDB_REGION_EXPORT size_t hdb_region_stats(char* output, size_t size) noexcept;
HDB_REGION_EXPORT size_t hdb_region_error(char* output, size_t size) noexcept;
}
