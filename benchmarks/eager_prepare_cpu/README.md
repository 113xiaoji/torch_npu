# Real-operator CPU simulation

This benchmark simulates parallel preparation and ordered consumption for a
captured eager operator sequence. Every numerical operator is a real CPU PyTorch
`aten.OpOverload`. Native workers run controlled C++ integer busy-work on copied
operator/tensor metadata, using the production `EagerPrepareQueueCore` for
Submit/Consume/Retire. There is no NPU execution, CANN preparation, live transparent
CPU eager backend, or Python 3.14 free-threading measurement. Receipts explicitly
say `simulated_prepare=true` and `real_npu=false`.

The four models are imported by explicit path from the separately frozen
`benchmarks/eager_prepare/models.py`. Both its SHA256 and the SHA256 of its source
pin manifest are checked before import. Their source, weights, inputs, and RNG
construction remain unchanged. They are synthetic inference models, not trained
accuracy workloads.

This directory's separate `SOURCE_PINS.json` freezes the CPU driver, tests,
documentation, and native source/build dependencies. The campaign verifies every
listed byte count/SHA256 before loading the native bridge and records the pin
manifest's SHA256 in its JSON receipt. It does not edit the original model pins.

## Reproduction

Use an installed CPU PyTorch and C++17 toolchain. No additional dependencies are
installed. On Windows the bridge uses the installed Visual Studio Build Tools;
`VSDEVCMD` can select an existing `VsDevCmd.bat`.

```powershell
python tools/eager_prepare_cpu/build_native.py
python -m unittest discover -s test/eager_prepare_cpu -v
python benchmarks/eager_prepare_cpu/run.py --output benchmarks/eager_prepare_cpu/results/full.json
```

The default bounded campaign runs all four models, controlled preparation targets
0/10/50/200 microseconds, five warmups and thirty timed forwards per arm, and three
paired rounds whose arm order alternates forward/reverse/forward. Queue capacity
is eight tickets and the native metadata byte limit is 1 MiB. A small
integration run uses the same real paths with fewer samples:

```powershell
python benchmarks/eager_prepare_cpu/run.py --output benchmarks/eager_prepare_cpu/results/mini.json --cost-us 0 200 --warmup 1 --samples 3 --rounds 1
```

The process receives the lowest eight CPUs in its actual allowed affinity set
(or the smaller available set), shared by Python producer/consumer, all native
workers, and Torch. Torch intra/inter-op and OMP/MKL/OpenBLAS/BLIS thread settings
are one. The report records the actual affinity, library parallel configuration,
Python version/GIL state, source hashes, and native core hash.

## Capture, dependencies, and supported domain

`TorchDispatchMode` captures one execution of the original eager model without
FX, `torch.compile`, or model edits. It records actual `OpOverload` objects, nested
argument/output pytrees, scalar attributes, tensor value IDs, and shape/stride/
dtype/storage-offset metadata. Strong references prevent Python tensor-ID reuse
during capture; only original input/parameter/buffer tensors survive in the tape.
All captured intermediate Tensor values are discarded.

Every replay starts with a fresh value environment. Each actual CPU call resolves
its arguments from outputs produced during that replay. View/alias behavior comes
from the actual aten operators. Intermediate references are removed after their
last use. A test substitutes different initial input values and checks propagated
outputs, preventing a tape of precomputed answers from satisfying correctness.

The supported domain is read-only, static-shape CPU inference with aten operators.
Mutable schemas, seeded RNG, custom operators, data-dependent scalar extraction,
and operators with data-dependent/dynamic output shapes are rejected before
invocation. Fixed-shape integer `aten.index.Tensor` is allowed after checking the
actual index dtypes; boolean indexing is rejected. Unknown tensor constants outside the initial inputs,
parameters, and buffers are rejected. Arbitrary Python side effects outside Torch
dispatch are not a supported model contract and cannot be discovered generally by
this recorder. This harness is not a general compiler or sandbox.

## Serial and queue arms

`direct_eager` executes the original model and supplies context for replay
overhead. `replay_serial` runs native `prepare_inline` and the actual aten call in
the caller for every op. `queue_w1/w2/w4` use the same tape, copied metadata, fixed
work iterations, and actual aten calls. The main Python producer submits bounded
metadata; one persistent Python consumer obtains the next production-core ticket,
executes the CPU op, and retires the ticket after the CPU op returns. Native
workers touch no Python objects, Torch tensors, or tensor data.

The captured sequences contain 70/10/11/32 aten calls for recommendation/branch
MLP/branch CNN/transformer respectively. The simulated preparation cost is applied
to **every captured aten call**, including views, transposes, unsqueezes, and other
metadata operations. This is not a claim that every such call performs real CANN
preparation or corresponds one-for-one with an NPU kernel. The recorder's
operator granularity and Python replay overhead are part of the CPU experiment.

One Session and one Python consumer persist across the warmups and timed forwards
of an arm block. Native sequence numbers increase across forwards. Creation and
destruction, trace capture, source hashing, correctness checks, and calibration
are outside the timed forward region; startup and capture times are recorded
separately. The timed region includes fresh replay environments, metadata
submission/copying, prepare work, backpressure, ordered waiting, actual aten calls,
and retirement. It does not hide queue management overhead.

The byte limit bounds copied native metadata, not Torch tensor memory. Capacity
remains occupied through CPU-op completion and retirement. Native faults or CPU
consumer exceptions abort the session and wake blocked admission. A consumer's
original CPU exception takes priority over the consequent producer error.

## Evidence and interpretation

Before timing, a separate execution of each original model provides independent
eager callsites and complete per-call/final numerical outputs. Every model,
controlled cost, and serial/queue arm must match them. Model/input full data hashes
are compared before and after capture, validation, each timed arm, and the whole
campaign. Each arm's last timed output is also checked against direct eager.

One serial native calibration selects fixed work-iteration counts; those exact
counts are shared across every model and arm at that cost. Zero cost means zero
additional busy-loop iterations; metadata hashing and bridge overhead remain.
The artificial cost does not estimate CANN GetWorkspaceSize work. No sleeping is
used as preparation work.

Separate cold traces record actual native TIDs, native thread CPU time, and native
prepare wall intervals. Windows `GetThreadTimes` can return zero for short
bodies, so calibration uses native body wall time and those zero CPU-time
readings remain visible. Raw hardware thread cycles, where available, are CPU
activity evidence without conversion to invented nanoseconds. CPU consume intervals use the same native monotonic
clock. The report derives observed distinct-thread prepare overlap and
prepare/consume overlap from intersecting intervals. One separately labeled
approximately 50ms/op cold run on the first model supplies positive native thread
CPU time despite coarse Windows accounting. Its intervals, CPU-time/wall-span
ratio, correctness, and hashes are retained, and it is excluded from all formal
warmups, samples, costs, and performance ratios. Formal performance runs have
native tracing disabled. All raw wall and process-CPU samples, mean, P50, P95,
P99, min, and max are retained, including slower arms and outliers. Counters,
prepare checksums, capacity/byte peaks, and retirement totals are checked.

Queue/serial ratios measure this controlled CPU simulation. Direct eager/replay
overhead is reported separately. No CPU-model aggregate is an NPU speedup, a CANN
SDK benefit, or evidence of Python 3.14t benefit.

`serial_over_queue_p50_speedup` is serial replay P50 divided by queue replay P50,
so values above one mean faster queue replay. `queue_over_serial_p50_latency_ratio`
is the inverse, so values below one mean lower queue latency. Both definitions
are recorded in the JSON. `replay_serial_vs_direct_p50_overhead_ratio` is the
separate replay-serial/direct-eager comparison.

## Test-first record

The initial capture/replay suite produced eight expected missing-driver failures
before implementation. Six real CPU capture/replay tests then passed, including
all four models, per-call/full-output comparison, unchanged hashes, intermediate
release, replacement-input propagation, view aliases, multioutputs, unsupported
operations, and wrong-order detection. The persistent-pool test failed for the
missing `ReplayRunner` before its implementation. Three campaign-contract tests
failed for the missing campaign runner before implementation and then passed.
Native-dependent test and campaign commands above provide the integration gate;
their actual results belong in the run evidence, not inferred from unit results.
