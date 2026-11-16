# R41 Eager concurrency candidates

These are experimental, independently selectable PTA changes. They preserve
the original model, inputs, op executors, device streams and Eager semantics.
CPU mechanism tests do not establish NPU speedup or linear throughput scaling.

| Variable | Default | Treatment |
|---|---|---|
| `TORCH_NPU_EVENT_HOST_WAIT` | unset / `spin` | `condition` waits on the existing pending-record predicate with a condition variable; ACL Event synchronization still runs |
| `TORCH_NPU_WORKSPACE_CACHE` | unset / `0` | `1` enables a bounded caller-thread cache for audited internal-stream TQ1 opAPI workspace sites |

The Event setting is cached at first use; invalid or empty values are rejected.
The workspace setting is cached at first use. Use a fresh process for each arm.
The Event change does not add timeout/cancellation or repair pre-existing
enqueue-failure rollback behavior. All users must finish before manager
destruction. It retains the existing pending-record count and failure cleanup.

Workspace entries are keyed by device and internal raw stream, with at most
four entries and 64 MiB of retained cache capacity per caller thread. Queued
callbacks independently own their buffers through actual submission, so total
in-flight memory can exceed the retained-cache bound. See
[workspace_audit.md](workspace_audit.md) for the ownership and SDK ordering
argument and explicit scope limitations.

Workspace reuse bypasses external/synchronous streams, TQ0/TQ2, graph/host
capture, private allocator pools, custom allocators, force-uncached allocation,
owned/prepare modes, uninitialized runtime and unsafe resources. A reset,
stop/restart, stream recovery or mark-unsafe transition permanently disables
reuse for that process. This observation is not a barrier and does not make
concurrent reset with in-flight work supported.

## Build

Apply `patches/eager_mpsc_r34/op-plugin/op_plugin/utils/op_api_common.{h,cpp}`
to the matching op-plugin utils directory and put
`op-plugin/op_plugin/utils/op_api_workspace_cache_r41.h` beside them. The
torch-npu changes are in the normal source tree. Ensure the new
`torch_npu/csrc/core/npu/RuntimeResourceEpoch.cpp` is included in the native
target; reconfigure CMake if necessary.

Rebuild all translation units depending on the changed headers. In particular,
`NPUEventManager` now has a condition variable and a different class size;
replacing only two old objects is not a complete ABI-safe build. Preserve the
original package/library and publish a separate candidate artifact.

## Verification

From the repository root:

```text
python test/linear_scaling_r41/event_wait_native.py
test\linear_scaling_r41\workspace_cache_tests.cmd
python test/linear_scaling_r41/workspace_admission_test.py
python -m unittest discover -s test/linear_scaling_r41 -p test_native_policy.py -v
```

The CPU tests compile production Event/admission code with explicit dependency
stubs and exercise the standalone cache. They cover wakeups, pending counts,
ownership, cache bounds, fallback gates and resource invalidation. They do not
substitute for real SDK execution, numerical validation or shutdown/recovery
qualification on an NPU.

`benchmarks/linear_scaling_r41/real_service.py` retains the R38 closed-loop
request engine, original DLRM/model/input identity, full output checking,
natural GIL-off checks, private streams, topology proof and provider pins. R41
adds an explicit expected native SHA, native mode getters and per-caller TLS
cache receipts outside the timed request loop. Enabled-cache performance cells
fail qualification if any caller has no actual cache hits.

Use four treatments: `spin/0`, `condition/0`, `spin/1`, `condition/1`. Run
1/2/4 callers and all correctness cells before interleaved performance rounds.
Compare absolute throughput, scaling, p99, forward/completion CPU and wall time,
and memory; also calibrate the rebuilt `spin/0` baseline against the frozen
library. Keep the model, inputs, batch, CPU budget and runtime fixed.

`model_sanity.py` separately checks real concurrent MLP, CNN and Attention
outputs and immutable weights. Its receipts are correctness evidence only.
Source/provider/runtime identities and observed shared-device interference
must accompany any A3 performance conclusion.
