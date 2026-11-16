# R41 internal OpAPI workspace reuse candidate

Status: host helper tests passed; native OpAPI rebuild, A3 correctness and performance qualification remain required. This is an opt-in candidate, not a measured speedup.

## Existing behavior and scope

`op_api_common.h::hit_cache`, `HDB_ORIGINAL_EXEC_NPU_CMD_V1`, `HDB_ORIGINAL_EXEC_NPU_NO_FORMAT_CHECK_CMD_V1`, and `op_api_common.cpp::ExecuteCachedOp` allocate a Tensor using `OpPreparation::unsafe_empty_workspace(bytes)` and pass its raw storage address into an `OpCommand::RunOpApiV2` callback. The original callbacks capture only that address. TQ1 prepares on the caller thread and submits phase2 through the queue. `ExecuteParasOpApi::Copy(ExecuteParasOpApiV2&)` moves the callback into the queued command; `ExecuteParasOpApi::Release()` clears it after consumption.

The one-argument allocation creates a `StorageImpl` and `NPUTensorImpl` around `NPUCachingAllocator::allocate(bytes)`. Its repeated Tensor construction and allocator metadata work are the candidate's target. The two-argument overload uses the separate `NPUWorkspaceAllocator`; TQ2 already reuses stream-local device storage, and those call sites stay unchanged. Raw-pointer-only `GetWorkSpaceAddr`/`GetWorkSpaceAddrV2`, synchronous APIs, owned SDK execution, and eager-prepare execution also stay unchanged.

## Ordering and ownership argument

1. ACL defines workspace as temporary Device memory for operator computation, separate from inputs and outputs. Workspace is an argument to phase2; phase1 receives no workspace pointer and cannot access the new cache's storage. [CANN 9.1 two-stage API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/common/two_phase_api.md), [aclnnMm workspace and stream arguments](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnMm.md).
2. ACL documents that asynchronous tasks on one stream execute in submission order, and `CommonOpExecutorRun` executes the executor's operator tasks on the supplied stream. Reusing scratch storage across successive submissions on that same stream preserves that order without a device/stream wait. [AscendCL call flow](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000005.html), [CommonOpExecutorRun contract](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_1129.html). Applying these contracts to scratch reuse is an inference, not evidence that every SDK kernel has been tested.
3. The existing allocator independently documents the same rule in `NPUCachingAllocator.cpp`: allocations are stream-associated and freed blocks may be reused on their allocation stream. It handles block disposal, physical release and cross-stream safety. The candidate keeps that allocator and its original exact allocation request on cache misses.
4. The candidate is thread-local and keyed by `(device index, already-known raw ACL stream)`. Distinct streams never share a cached buffer; two host threads also have independent buffers. It never changes an SDK executor, cache key, model, input, phase1 call, or phase2 stream.
5. Each admitted callback additionally owns the returned Tensor. Growth, LRU eviction, or producer thread exit can remove a cache entry, but cannot release a buffer before its queued callback submits phase2. After submission, normal allocation-stream allocator lifetime rules apply. A C++ ownership test cannot prove Device completion safety; A3 correctness testing must exercise this path.
6. Cache admission requires TQ1, a torch_npu-managed nonsynchronous current stream matching the supplied raw stream, no graph capture, no host capture, and no owned/eager-prepare mode. Capture checks use `stream(false)` and do not drain queues. Unsupported paths use the original allocation.
7. Admission additionally requires an initialized runtime, exact builtin native allocator identity, force-uncached mode off, no allocator-private-pool scope on the device, no recovery unsafe flag, and runtime resource epoch zero. `use_mem_pool()` enters the allocator's pool tracking without SDK graph capture, so SDK capture status alone is insufficient. The allocator-pool gate conservatively bypasses even when another thread owns the pool scope.
8. The resource epoch is invalidated before existing managed stream destruction/device reset, stream rebuilding, stop/restart-device repair, and marking native allocator blocks unsafe. After any such transition, this candidate permanently bypasses workspace reuse for the rest of the process; clearing an unsafe flag or encountering the same raw stream address does not re-enable it. Start a fresh process to qualify reuse again. This is fail-closed cache admission, not synchronization or new support for concurrent reset/inflight work. Existing callers must quiesce work and observe existing live-Tensor/storage lifetime requirements before recovery/finalization. Direct external ACL stream/context/device destruction or reset bypassing these managed entry points remains unsupported; this candidate cannot detect such external mutations.

There is no synchronization, ACL/CANN modification, zero initialization, Tensor shape rewrite, model-specific allowlist or skipped computation. An execution error propagates through the original `NPU_CHECK_ERROR`; any partially submitted work retains the same stream ordering. Allocation failure preserves the previous cache entries and propagates instead of silently retrying with different behavior.

## Bounds and controls

`TORCH_NPU_WORKSPACE_CACHE=1` enables reuse; any other value or unset disables it. The flag is read once per process, so run off/on in separate processes using the same binary and configuration. The cache retains at most four entries and 64 MiB of requested capacity **per caller thread**. Requests larger than 64 MiB bypass reuse and keep the current working set. Cache misses allocate the exact requested bytes, without changing `unsafe_empty_workspace` globally. LRU eviction and growth release only the cache's owner.

The 64 MiB bound covers the retained reusable working set, not the underlying allocator's rounded reservations or old buffers still held by queued callbacks. Allocation happens before eviction to preserve the old cache on failure, so growth can temporarily hold both old and new buffers. Queue depth and normal allocator behavior still bound/retire outstanding command storage; this feature introduces no independent deferred-retirement queue. Keeping Tensors resident can increase allocator active-memory accounting and reduce memory available to other operators. High-pressure/OOM behavior needs measurement.

The only feature controls are the enable flag and the fixed limits above. Counters are thread-local, with no atomic/global aggregation on the hot path. `extern "C" OpApiWorkspaceCacheStatsR41(uint64_t* output, size_t count)` copies up to eight fields from the calling host thread:

| Index | Meaning |
| --- | --- |
| 0 | effective enable flag |
| 1 | sufficient-capacity reuse hits |
| 2 | successful factory allocations, including enabled-path bypasses |
| 3 | growths replacing a smaller entry |
| 4 | LRU evictions for entry/byte limits |
| 5 | nonzero enabled-path requests bypassing admission or size limit |
| 6 | requested bytes retained in reusable entries |
| 7 | reusable entry count |

Disabled returns zeros for counters 1–7 and invokes the original allocator directly, without constructing a TLS cache. The new allocator/pool/unsafe/runtime-epoch gates also return the original allocation before touching the TLS cache; those early bypasses are not included in counters 1–7. Other helper bypasses are counted. Counters describe only calls reaching the helper, not every workspace allocation in torch_npu. Read the export from the same application thread that launches the operations; querying only the main thread will miss worker-thread activity. The enabled field reports the requested cached flag; positive reuse-hit deltas are still needed to prove an effective treatment.

## Overlay and validation

Overlay the modified R34 `op_api_common.h` and `.cpp`, and copy `op-plugin/op_plugin/utils/op_api_workspace_cache_r41.h` from this directory beside them in the actual op-plugin source before rebuilding. Neither TQ2 nor raw workspace helper call sites need an overlay change.

The fail-closed production gate also changes `NPUCachingAllocator.{h,cpp}`, `NPUFunctions.cpp`, `NPUStream.cpp`, and `npu/Module.cpp`, and adds `core/npu/RuntimeResourceEpoch.{h,cpp}`. Reconfigure CMake so its existing `core/CMakeLists.txt` source glob includes the new source, then rebuild all changed objects and header dependents. Additive C++ exports are `c10_npu::NPUCachingAllocator::isNativeAllocator() noexcept`, `c10_npu::GetRuntimeResourceEpoch() noexcept`, and `c10_npu::InvalidateRuntimeResources() noexcept`; existing type layouts are unchanged by these three interfaces.

Host TDD: `cmd /c test\linear_scaling_r41\workspace_cache_tests.cmd` compiles with MSVC C++17 `/W4 /WX`; the test writes binaries/objects to `%TEMP%`. The baseline implementation produced eight expected runtime failures; the implemented helper passed all nine cases. Linux equivalent:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  -Ipatches/linear_scaling_r41/op-plugin/op_plugin/utils \
  test/linear_scaling_r41/workspace_cache_test.cpp -o /tmp/workspace_cache_r41
/tmp/workspace_cache_r41
```

Also run `python test/linear_scaling_r41/workspace_admission_test.py` on Windows/MSVC or Linux/C++17. This compiles the actual admission function extracted verbatim from `op_api_common.cpp` with CPU-only dependency stubs and links the real `RuntimeResourceEpoch.cpp`. It covers custom allocator, private pool, force-uncached, unsafe storage flag, uninitialized runtime, scope exit, and permanent bypass after resource generation changes. It checks the cold lifecycle hooks precede their resource mutations and that native identity uses the actual builtin allocator address. Before the fix it compiled successfully and produced six expected runtime failures. These wiring checks supplement helper tests; the full production translation units and ACL behavior still require the normal ARM build and hardware qualification.

Hardware qualification must preserve the existing model, shapes/inputs, task queue configuration, executor/cache settings and stream topology. Use the same rebuilt library for both flag states, with profiling instrumentation identical. At minimum:

- Verify effective flag 0/1 via the export and record library SHA/build identity. On enabled worker threads, after warmup expect positive hits and substantially fewer successful allocations than eligible nonzero calls; otherwise classify the result as no effective workspace treatment.
- Run the existing full forward/backward/optimizer correctness oracle; compare complete outputs, gradients and parameter updates under established tolerances, with baseline on the same binary. Include repeated steps and threads/stream topology used in the real experiment.
- Exercise nonzero workloads across two streams, producer exit before queue drain, increasing/decreasing workspace demand, and enough keys/bytes for eviction. Ensure queues are drained through existing synchronization at the verification boundary. Explicit graph/external/TQ2 fallback qualification must not report this feature as active reuse.
- Measure repeated off/on runs with identical warmup, inputs and ROI. Report median/p95 and sample count, plus helper counter deltas. Separate allocation-hotspot reduction from end-to-end latency and scaling. Do not attribute an improvement to workspace reuse without the effective-hit proof.
- Record peak allocator active/reserved memory and any OOM/recovery behavior. A speedup or A3 correctness claim is unsupported until these checks complete.
