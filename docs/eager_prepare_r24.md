# R24 native eager Prepare queue

This opt-in experiment intercepts the op-plugin SDK invocation macros. It does
not replace model methods, AST-transform a model loop, or run Python on Prepare
workers. Importing `torch_npu._eager_prepare` is inert; `configure()` defaults to
off. The original eager FIFO and execution stream remain authoritative.

## Current capability boundary

The supported execution domain is a fixed-shape, no-grad/inference workload,
one native caller thread, device0, the default stream, asynchronous task queue1,
and no Python dispatch modes, trace callback, framework profiler, concurrent
metadata/configuration mutation or R22 host-region capture. Configuration and
close may be called across threads; model admissions belong to the configuring
caller. CPU affinity/core budget belongs to the unchanged benchmark/controller.
Finish model/input setup and its original synchronization before enabling,
then call configure inside the admitted inference/no-grad scope. Ordinary
module construction and migration are not silently brought into the queue.

| SDK schema | typed registration | asynchronous SDK qualification |
|---|---|---|
| aclnnEmbeddingBag | yes | existing exact NN-provider profile, restricted domain |
| aclnnAdd | yes | **not qualified** |
| aclnnRelu | yes | **not qualified** |
| aclnnMm | yes | **not qualified** |

The queued EmbeddingBag domain is FP32 weight, int64 indices/offsets, nonempty
rank2/rank1/rank1 inputs, sum, sparse, no per-sample weights, no last offset,
negative padding, native device0 storage and disjoint input/output storage.
Other valid arguments (mean/max, FP16, int32, empty, etc.) retain the original
route through an explicitly counted fallback fence. An SDK/allocator exception
after choosing the queue is terminal; it is never caught to rerun the operator.

`allowed_ops` requests schemas; `qualified_ops` is a separate evidence-backed
subset. Presence of a symbol or a typed function registration cannot qualify an
operator. New operators require their own exact provider identity, typed query
ABI, metadata-only phase1, executor consumption/rollback, cache and cleanup
evidence before their qualification is enabled. In particular, the existing
EmbeddingBag function pointers are not used as another operator's implementation.

The macro wrapper evaluates typed on-path arguments once with `std::apply` and
`std::forward_as_tuple`. Temporaries remain alive for the complete expression;
the original fallback copies or owned queue snapshots are made before it ends.
Off invokes the complete original macro without changing its arguments,
publication, synchronization or argument evaluation. The actual source header
is kept separately from the frozen R22 overlay.
The synchronous SDK macro retains its original expression and move-only
converted-parameter return contract. It only adds the enabled-domain fallback
fence before returning the complete original synchronous expression; it never
takes the void asynchronous admission path.

Intercepted unsupported/unqualified SDK macros drain the original FIFO and synchronize the
same default stream **before original SDK phase1**, outside core/state locks.
These fences are counted and included in end-to-end timings. They do not cover
data-dependent host metadata reads which happened earlier in the operator
kernel. First model cohorts therefore still require per-operator caller-meta
review or existing synchronization evidence. `.item()`, `.cpu()`, resize,
in-place/aliasing, RNG, training, collectives and arbitrary dynamic-shape eager
programs are not automatically certified by preserving Execute order.
Compatibility branches which enter legacy aclop before these macros are also
outside automatic coverage and need their own caller-meta/synchronization audit.

## Public interface

```python
from torch_npu import _eager_prepare

configuration = _eager_prepare.configure(
    enabled=True, workers=4, capacity=8, byte_limit=536870912,
    allowed_ops=("aclnnEmbeddingBag", "aclnnAdd", "aclnnRelu", "aclnnMm"),
    diagnostic_trace=False, trace_limit=256,
)
# Run the original model in its admitted inference/no-grad scope.
# Preserve the original benchmark's synchronization and iteration boundaries.
before_close = _eager_prepare.snapshot()
closed = _eager_prepare.drain_close()
```

Enabling initializes the existing NPU runtime. Workers/capacity must be positive,
workers<=capacity, workers<=8, capacity<=1024, bytes<=512MiB; the experimental
cohort retains workers4 and CPU0..7. A failed model/close must preserve the first
exception. Do not rerun the model, force-release quarantined resources, or
pretend that an exception is a retirement acknowledgement.

## Ownership, publication and retirement

Caller code retains the original output metadata/allocation and argument
validation. It creates native Storage owners plus immutable metadata and freezes
device/stream/context, dispatch/inference state and CacheParams. Native workers
reconstruct only admitted native TLS, recheck SDK cleanup poison/configuration,
and perform Prepare. The entire PTA hash/cache operation stays on one worker;
HugeMem UnInit detaches TLS but does not return the executor's pool index.
R24 storage eligibility is activated explicitly by this runtime, independently
of the old process-cached `FTLAB_HOST_OWNER_RETIREMENT` flag. It retains the
concrete allocator/deleter/device/stream, NPUStorage type/Python-slot/lazy-fusion,
instrumentation, history/context/tracker, graph/private-pool, neighbor context
and stream-use checks. The old R22 predicate and its cached-env behavior remain
unchanged; the benchmark must not secretly enable that older region feature.

A stable job is reserved and dispatched before its deferred callback is
published to the original FIFO. Repository's successful deep-copy tail records
the exact source parameter's explicit ACK and immediately marks its Core ticket
committed. This acknowledgement does not wait for enqueue/publisher to return:
Repository INIT may synchronously wait for the original consumer to finish.
Later publisher errors cannot revoke the accepted prefix or rerun the operator.
The original consumer waits for its ticket, then directly calls SDK Execute in
original FIFO/default-stream order.
Workers never enqueue into or drain that FIFO. The core has an independent work
queue and never waits for a later task to unblock an earlier one.

The FIFO closure owns a retirement token. Its final release waits for Prepare/
Execute ACK and the actual same-stream completion fence before returning credit.
SDK cleanup failure retains its ticket, owner and worker TLS. `OwnedQueueJob`
explicitly releases payload/Storage/workspace before core credit is returned,
even when a Submit local or a retirement closure still holds the Job wrapper.
Nonaccepted publication gets one core rollback; an uncertain publication is
quarantined. There is no retry or fallback for an already accepted prefix.

The retained-byte guard is conservative logical Storage+declared-workspace
reservation, not total process/device memory. Initial reservations and successful
Prepare topups feed core high-water counters. Post-query topup never waits for
credits held by later prepared jobs: exceeding the bound fails and rolls back.

## Shutdown and failure

Lifecycle lock acquisition first detaches the calling interpreter; destruction
unlocks before reattachment. Snapshot only detaches and takes short state/core
locks, so returning a snapshot while holding the lifecycle lock cannot self-lock.
Nested detach is a no-op when no interpreter state is attached. Waits do not
hold a GIL while another thread must reacquire it to release a lifecycle lock.

Cold close closes admission, finishes active native submitters, drains the
original healthy FIFO, performs the original device0 synchronization, waits for
all real retirement ACKs, then stops/joins the Prepare pool. It leaves the
original Repository running. After successful close, later original CPU copies
and eager calls follow the off route. Failure preserves the first error and
workers; a repeated close rethrows the latched cold-close failure without
touching SDK, events or retirement again. NPU shutdown entry points use the same
cold-close gate before original synchronization/finalization.

Quarantine does not claim that SDK resources were reclaimed. An external bounded
test controller must terminate a stuck failed process; preserving the Job object
alone would not preserve the producer's private SDK cache TLS.
The original runtime Finalize entry has a cold gate before its SDK/init locks.
If the same cold close failed, it emits
`R24_EAGER_PREPARE_FINALIZE_FAILSTOP` and waits while keeping the complete domain
alive. It does not retry SDK work, enter forced cleanup or use `os._exit`.
The owned outer controller must TERM/KILL and reap that failed process. Forced
termination is not a graceful close or an SDK resource acknowledgement.

## Snapshot and diagnostic trace

Schema1 includes mode/enabled, workers/capacity/byte_limit, caller_tid,
allowed_ops/qualified_ops, per-schema `operators`, `queue`, `peaks`, `closure`,
`first_error`, and `diagnostic_trace`.

- `seen`: an enabled SDK macro invocation.
- `eligible`: qualified and admitted arguments chosen for the owned path.
- `queued`: original FIFO deep-copy ACK.
- `prepared`: successful phase1 or legal cache hit.
- `executed`: SDK phase2 entry attempt.
- `returned`: SDK phase2 returned, including nonzero status; **not success**.
- `fallback`: original route selected after its fence; `fallback_reasons` and
  `fallback_fences` are recorded per schema.

Active snapshots are not linearizable. Only stable post-close counters and
closure evidence can be used for acceptance. Numerical correctness, eligible-op
coverage, full/partial/none model-core coverage and performance validity remain
separate decisions. No model family becomes fully covered because workers4 is
configured or one unrelated pointwise op was queued.

`queue.workers_started` counts real native worker entry, not configuration.
`peaks` are actual core reservations, not opaque SDK temporary memory peaks.
Cold tracing is explicitly enabled, pre-reserved and capped at 256 records.
It records Prepare sequence/op, native/caller/consumer TIDs, monotonic start/end,
thread CPU start/end, success and a bounded dtype/shape/stride/offset signature
from native snapshots; it never reads tensor data. A missing signature is
explicitly `shape_unknown=true`. Dropped/allocation-failed observations are
visible; completeness requires successful close and matching Prepare records.
Formal performance runs leave tracing off and retain only counters.

Worker wall overlap, actual worker birth, and thread-CPU overlap are different
claims. None by itself proves SDK-internal parallelism, useful work scaling,
lock wait, kernel overlap or speedup. There is no new native query/Execute/device
duration attribution in the ordinary counters.

## Reproducible overlay and current verification limit

```text
python tools/host_regions/apply_eager_prepare_overlay_r24.py
python tools/host_regions/apply_eager_prepare_overlay_r24.py --apply
```

The installer pins submodule commit312921458e7d396c54a1e92e90385d7d5f02dede,
checks all eight committed R24 payloads, and accepts only a complete pristine
base, the complete pinned seven-file R22 parent, or the complete exact R24
overlay. Mixed/dirty destinations are refused; no fetch/checkout/index update
or SDK/build operation is performed. Frozen R22 payloads/manifest/installer are
retained unmodified.

Host tests cover the actual C++ owned-job adapter with the actual queue core,
external strong-Job ownership across retirement, single argument evaluation and
temporary lifetimes, valid-unadmitted domain policy, public bridge/error
contracts, manifest tampering, and a compiled CPython3.13 extension exercising
real GIL lifecycle concurrency/nested detach/no cold retry. The old lock order
is reproduced only in a disposable child with a strict timeout and kill/reap.
The actual native publication marker/copy-ACK callback is also extracted and
compiled against the real OrderedMpscQueueCore and EagerPrepareQueueCore. A
bounded synchronous publisher reproduces INIT waiting for consumption; late
publisher errors preserve committed ownership and are not replayed.
The exact synchronous macro is compiled as an assignment expression with a
move-only converted-parameter owner and checked for single evaluation and no
retry after SDK/fence failure.
These are host contracts, not SDK mocks passed off as device qualification.
The full PTA/CANN native translation units, actual device cases and paired
model performance still require the separate build/device gates.
