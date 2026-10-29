# R24 model-independent Eager Prepare validation

This directory is a benchmark/validation harness, not a new queue implementation.
Default commands only print plans. It does not patch model methods, recognize
model ASTs, import a simulator, install packages, or enable an unqualified SDK op.
R22/R23 evidence remains historical and is never pooled with these results.

## Current evidence

Local qualification is CPU-only: CPython 3.13.12 and Torch 2.7.1+cpu. The tests
exercise real Torch models, deterministic initialization, every output value,
statistics, receipt validation and the classification of queue coverage. They
do **not** establish target CPython3.14.6/std/FT, NPU correctness, actual worker
overlap, SDK qualification, or speedup. Those require the fresh target protocol.

The explicit native adapter is the separately implemented
`torch_npu._eager_prepare` schema1 API. Typed/registered is not qualified. Initial
qualification covers only the SDK contracts independently established by the
native owner; the on harness reads `qualified_ops` rather than assuming all schemas
are usable. Current Add/Relu/Mm declarations alone do not qualify those SDK calls.
Fresh off has no active queue, empty operator counters and `qualified_ops=[]`;
its `qualification.status=no_active_qualification` does not borrow the artifact's
on qualification. Both modes remain bound to the same approved artifact.

## Four actual inference workloads

These are new deterministic **synthetic** inference models, not downloaded
trained models and not the original nine-file R22 DLRM. All parameters are FP32,
non-grad, eval mode; there is no dropout/training or graph compilation. Every
Linear is bias-free and is called with a 2-D input. Weights and inputs are
generated from a fixed CPU RNG state without consuming the caller's RNG state.

| Workload | Fixed input / output | Full core target per forward |
|---|---|---|
| recommendation | dense128x13;26x1024x16 sum/sparse EmbeddingBag,2 indices/bag; bottom13-64-32-16, dot interaction, top367-64-32-1; output128x1 | EmbeddingBag26, Mm6, Bmm1 |
| branch_mlp | input32x256;two independent256-128-128 branches, merge-add,128-16 head; output32x16 | Mm5 |
| branch_cnn | input4x3x32x32;3x3/3x3 branch plus1x1 shortcut, concat/pool and4096-64-10 head; output4x10 | Conv3, Mm2 |
| transformer | input4x16x64;one explicit unfused4-head attention/FFN block, residuals and two LayerNorms; output4x16x64 | Mm4, Bmm2, Softmax1, LayerNorm2 |

The CPU tests verify that these module/function calls really execute; merely
declaring unused layers is not sufficient. Actual target SDK schema lowering,
view/stride eligibility and qualified provider identity still come from native
operator evidence. In particular, ordinary Linear weight transposes are not
silently made contiguous to hide unsupported admission. New SDK family names
must be explicitly mapped by a reviewed artifact receipt; known Add/Relu entries
cannot be relabeled as Mm to manufacture core coverage.

## Four independent result axes

- `numerical_pass`: all calibration and final output values agree with the CPU
  reference at fixed rtol1e-4/atol1e-5; finite/shape/dtype checks are mandatory.
  Full actual/reference output bytes, size, byte order and SHA are preserved.
  CPU state/input hashes before and copied-back state/input hashes after must
  agree. No `.cpu()`, hash or metadata observer is inserted in the timed loop.
- `eligible_operator_coverage_pass`: at least one eligible op really entered the
  queue, and eligible/queued/prepared/executed/returned counts agree after a
  successful run. Counter rollback, unqualified queued entries, native first
  error and failed closure invalidate the mechanism evidence. `returned` means
  the SDK phase2 call returned, including a nonzero status; it is **not** by
  itself a success proof. First-error/closure/numeric checks must all pass.
- `model_core_coverage`: `full`, `partial`, or `none`, using the declared core
  families/counts above. Unsupported core families may run correctly through
  the original FIFO and still yield a valid mixed-model numerical result.
  A partial result is not full; all fallback or only small pointwise hits gives
  core `none`. A full core result can still have ancillary fallback, recorded
  separately as `mixed_execution` and per-op fallback reasons. Unregistered
  ops are not necessarily visible in the initial registered-op counters;
  missing target-family hits must not be represented as observed zero cost.
- `performance_valid`: a correctly qualified, uninstrumented real-NPU
  benchmark completed the fixed exposure. It can be true for legitimate
  partial/none coverage; that does not prove a cross-model queue benefit.
  CPU smoke and cold traced qualification always set this false.

`passed` only means the workload's numerical/resource/measurement protocol
completed. It does not combine these axes into a claim of complete operator
coverage or acceleration. `prepare_parallelism_proven` is never inferred from
`workers=4`, queue counts, process CPU/wall, or a model name.

## Native adapter and ownership units

The native owner's explicit API is:

```python
configure(enabled=False, workers=4, capacity=8, byte_limit=536870912,
          allowed_ops=(...), diagnostic_trace=False, trace_limit=256)
snapshot()
drain_close()
```

All return a schema1 native snapshot. The runner stores the configure snapshot
under its own `configuration` result key; the public API itself has no extra
configuration container. Required evidence includes:

- mode/enabled/workers/capacity/byte_limit/allowed_ops/qualified_ops;
- operators by **actual SDK schema**: seen,eligible,queued,prepared,executed,
  returned,fallback and reason counters;
- queue slots/retained bytes, closed/quarantined state, peaks and first_error;
- explicit admission_closed/fifo_drained/device_synchronized/all_retired/
  workers_stopped/passed closure.

Queued means original-FIFO ACK; prepared means successful phase1/legal cache
hit; executed means phase2 entry. These are different from device completion
and resource retirement. The reported retained-byte peak is the queue's
accounting boundary, not whole-device HBM or arbitrary SDK-private allocation.
Normal active-on close requires slots/retained bytes zero, peaks within8/512MiB, no first
error/quarantine, and every closure flag true. A failed/uncertain close is
attempted only once; the first failure is preserved. A reviewed outer process
owner must kill/reap a hung or quarantined process; the harness does not add an
SDK failure hook, replay an accepted op, call `os._exit`, or retry cleanup.
Fresh off is checked separately for no domain: modeoff/disabled, no qualification,
empty operator counters, zero queue counts/peaks/workers_started and no first
error/quarantine. Its raw `queue.closed`, `fifo_drained` and
`device_synchronized` defaults are preserved; they do not certify an active queue
was retired. Each completed original model synchronize remains the off device
completion boundary. `closure_kind` distinguishes these two cases.

## Required independent queue tests

The generic runner tests validate receipts, not the queue implementation. The
native/core owner must independently cover the following, preferably using
deterministic Events/barriers and real threads rather than timing sleeps:

| Unit/fault/order scenario | Required observation |
|---|---|
| prepare2 finishes before prepare1 | original consumer executes exactly1 then2, once each; no publish bypass |
| eligible op / unsupported op / eligible op | one original FIFO preserves dependencies and side effects across fallback; a separate consumer must not overtake it |
| capacity2 plus bounded retained bytes | both slot and byte credit block admission safely; credit is not released at dequeue if owners remain in flight; peaks never exceed limits |
| blocked producer during close/failure | waiters wake, no queue-lock/SDK-call lock cycle, no accepted work lost |
| phase1 failure before/after reservation | rollback only the ownership actually acquired; preserve first error; no hidden fallback replay after admission |
| phase2 nonzero/throw or cleanup poison | first error latches, subsequent admission is refused, uncertain owners/TLS remain held/quarantined; returned-count alone cannot pass |
| early Python drop, asynchronous producer input | strong owners survive until the real completion boundary; actual in-flight/reuse window must be observed to claim coverage |
| aliases/views/in-place/out/host reads | admitted schema preserves sizes/strides/offset/storage and dependency order; otherwise explicit safe ordered fallback/rejection |
| internal allocation/reentrant dispatch | internal SDK/framework work is not recaptured recursively or counted as model-core hits |
| dtype/device/stream/Tensor subclass | legal cases retain original semantics; unsupported scope is not silently enabled; no model-name/source whitelist |
| graceful close/natural process teardown | every accepted sequence is retired or accounted for; workers stop or explicit quarantine remains; resource closure is not fabricated |

Host fake callbacks can test bookkeeping and failure order, but are not real
CANN/TLS lifetime proof. The normal benchmark synchronizes each request and
cannot certify an unconsumed-close/async window that it never observed.

## Target artifact receipt

Real execution requires `--artifact FILE --artifact-sha256 EXACT_SHA`. This is
data prepared after the reviewed R24 std/FT build/install; no sample receipt is
pre-marked successful. Schema1 requires:

| Fields | Meaning |
|---|---|
| abi,python,prefix,boot_id | fixed std/FT identity, **lexical venv executable** and actual prefix, same observed host boot |
| torch_version,torch_npu_version | exact installed versions; real mode also requires CPython3.14.6 and natural ABI/GIL |
| wheel:{path,sha256},native:{path,sha256} | exact new candidate wheel and installed libtorch_npu; all wheel torch_npu payload files are compared against installed bytes |
| cann_prefix,sdk_files:[{path,sha256}] | approved resolved/canonical absolute SDK/provider paths and hashes, including real AscendCL and all actually mapped private-CANN files |
| harness_manifest:{path,sha256} | absolute path of this deployment's SOURCE_PINS.json and its externally approved exact byte digest |
| device_name | actual NPU model; device properties/UUID string and fresh topology are recorded |
| allowed_ops,qualified_ops | requested registered schemas and exact independently qualified subset |
| operator_families | additional reviewed SDK-schema→semantic-family mappings; it cannot override known Add/Relu/Mm identities |

The runner checks the venv **without resolving it away into its base binary**,
then retains the validated sklearn.metrics→torch→torch_npu AArch64 static-TLS
import order. Before loading `models.py`, `contracts.py` or the topology helper,
pure-stdlib bootstrap verifies the artifact digest, its approved manifest digest,
and every size/hash in the complete11-file source/test/doc closure. Deploy all11
files at the manifest's repository-relative paths plus `SOURCE_PINS.json`.
Locally re-signing a changed manifest does not satisfy the artifact's fixed digest.
The outer reviewed launcher must pin the runner and artifact receipt themselves;
the process's already-running bootstrap cannot establish its own initial trust.

It verifies all installed wheel package bytes and mapped native/SDK identities,
actual NPU tensors, ABI/GIL before/after timing, and the entire approved manifest,
source closure and artifact hashes at completion. SDK observations after
calibration and at completion include all mappings under `cann_prefix`, any
alternate copy of an approved SDK basename, and ACLNN provider names beginning
libascendcl/libopapi/libcust_opapi/libnnopbase/libaclnn anywhere. Each actual path
must be approved and match its exact file hash; an unused approved provider cannot
hide a different loaded copy. Legitimate lazy loading or mode-specific subsets
are allowed only within that same approved set. Both actual observations are
kept in the result and independently checked by the pair analyzer; the subsets
need not be identical across modes. No guessed Python APIs are treated as
native evidence. Real mode cannot be enabled by a host-test fake adapter.

Before model/input setup, the runner freezes the actual public
`torch.npu.matmul.allow_hf32=False` and `torch.npu.conv.allow_hf32=False` in both
modes. This is a declared R24 FP32 protocol; Conv's default may otherwise permit
HF32. It records and checks both flags before setup, immediately before timing
and after the run, and the pair analyzer requires all three observations.
Tolerance remains rtol1e-4/atol1e-5. Native inactivity is observed before setup;
the copied model and inputs are migrated and synchronized and the CPU reference
is computed before queue configuration. Cold configure runs inside actual
`torch.inference_mode()` with no-grad, followed by calibration/warmup/timing
inside inference mode. Setup transfers/reference do not enter native queue
counters or diagnostic trace.

The source-pinned portable topology contract is reused read-only from
`benchmarks/host_regions/measurement_contracts_r22.py` (SHA5c62...). It freshly
proves exactly CPU0–7 are eight physical cores in the selected domain. All
caller/prepare/consumer/SDK threads share that total budget. Intra/inter-op and
OMP/MKL/OpenBLAS stay1; natural std GIL on / FT GIL off is required with no
PYTHON_GIL override, HOSTSIM, preload/audit or legacy region engine. Eager execution
uses the declared HF32-disabled FP32 policy; CANN/JIT defaults otherwise stay as
provided by the reviewed environment.

## Minimal commands and fixed campaign

From the repository/worktree root, this is host-only and needs no NPU import:

```sh
python -I -B benchmarks/eager_prepare/runner.py --model branch_mlp
python -I -B -m unittest discover -s test/eager_prepare -p 'test_*.py' -v
python -I -B benchmarks/eager_prepare/runner.py --model branch_mlp \
  --host-smoke --samples 3 --warmup 1 --output /absolute/new/HOST.json
```

Target execution is a fresh process in the receipt's exact interpreter and a
reviewed, rebuilt SDK environment, not the operator shell's arbitrary inherited
environment. Preserve TASK_QUEUE_ENABLE1, PER_STREAM_QUEUE0,
ASCEND_LAUNCH_BLOCKING0, TORCH_DEVICE_BACKEND_AUTOLOAD0 and numeric-library
threads1; set TORCH_NPU_EXPERIMENTAL_FREE_THREADED to0/std or1/FT. Root's outer
controller first records boot/process/NPU health and exclusive idle state.

1. **Fresh-process qualification:** std/FT ×4 workloads ×off/on, one fresh process each,
   fixed calibration1+warmup0+sample1. Native diagnostic trace is enabled only
   for on, bounded at256 events. No load/retry is added if overlap is unproven.

   ```sh
   "$EXACT_PYTHON" -I -B benchmarks/eager_prepare/runner.py --execute \
     --model recommendation --abi std --mode on --purpose qualification \
     --artifact /absolute/STD_ARTIFACT.json --artifact-sha256 "$STD_SHA" \
     --output /absolute/new/QUAL-std-recommendation-on.json
   ```

   Here “cold” means a fresh-process diagnostic: the traced sample is the second
   forward, after one completed full calibration. Snapshot differences select
   only the subsequent sample's newly observed Prepare sequences. This does not
   claim first-call SDK queries, all cache misses or cold-cache performance. Actual
   TIDs/CPU/wall are reported separately. Event `native_tid` is the actual Prepare
   worker OS TID, distinct from `caller_tid` and `consumer_tid`; at least2 distinct Prepare TIDs and
   summed threadCPU/span>1.02 establishes observed native overlap under the
   source-pinned clock contract. It is not pure query duration, four-way CPU
   saturation, or GPU parallelism. Missing/invalid/overflow trace cannot pass
   that proof. A configuration count is never substituted for actual TIDs.

2. **Benchmark:** same artifact and model/seed/input bytes,20 warmup+2000 timed
   model-entry→unchanged original torch.npu.synchronize samples, with one full
   calibration before them. Diagnostic trace stays off. Each process records
   all durations, mean/P50/P95/P99/min/max, process/caller CPU and loop wall,
   per-schema hot counter deltas, full numeric bytes and closure. Validation,
   CPU transfers/hashes and final close are outside latency. Quantiles use the
   declared nearest-rank rule; no stage quantiles are summed.

   ```sh
   "$EXACT_PYTHON" -I -B benchmarks/eager_prepare/runner.py --execute \
     --model recommendation --abi std --mode off --round 1 \
     --artifact /absolute/STD_ARTIFACT.json --artifact-sha256 "$STD_SHA" \
     --output /absolute/new/std-recommendation-round1-off.json
   ```

3. `campaign_plan.py` prints the fixed48-process grid: 4 models×2 ABI×2 modes×
   3 rounds, adjacent paired modes with predeclared reversed order in the next
   round and counterbalanced starting modes. It imports no Torch and launches
   no process. A reviewed outer owner (the existing R23 WNOWAIT/group-empty
   helper can be reused) executes each slot sequentially with1800s timeout,
   captures raw/log/exit and reaps its complete group before continuing.

   ```sh
   python -I -B benchmarks/eager_prepare/campaign_plan.py \
     --std-artifact /absolute/STD_ARTIFACT.json \
     --ft-artifact /absolute/FT_ARTIFACT.json --output-root /absolute/new/R24_RUN
   ```

4. `analyze.py` compares one exact-hash off/on pair. It rejects different
   input/weight identity, ABI/runtime/device, artifact-bound approved source
   manifest or artifact; both mapping observations must match that artifact's
   approved provider inventory. It permits
   valid mixed results but preserves their core classification. Collect all
   three declared paired ratios per ABI/model before describing a gain or
   regression; do not pool ABIs, use R21 as the same-artifact control, or
   select/retry rounds from the observed result. Stability is a separate,
   predeclared phase; it is not silently added or mixed into short throughput.

   ```sh
   python -I -B benchmarks/eager_prepare/analyze.py \
     --off /absolute/OFF.json --off-sha256 "$OFF_SHA" \
     --on /absolute/ON.json --on-sha256 "$ON_SHA" \
     --output /absolute/new/PAIR.json
   ```

No target build, SDK qualification, device workload or performance result has
been executed by this directory's local authoring/tests. These commands become
real NPU evidence only after their actual receipts, exit statuses, operator
coverage and exact artifact identities have been collected and independently
reviewed.
