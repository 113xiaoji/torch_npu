# Actual E/EI native execution and host-sim validation

**Goal:** Execute the unchanged original model through native Embedding and
Interaction side streams, verify CPU simulation numerics, and compare E2E off/E/EI.

**Architecture:** Generated native wrappers invoke the original body immediately
on the caller, under a selected internal stream. A bounded begin/complete protocol
pre-reserves events and owner slots; input allocation dependencies select waits,
and unsupported operations or public observation/mutation boundaries drain the
group. No model rewrite, Python dispatcher replacement, or host Prepare worker
pool is introduced.

**Tech stack:** PTA C++17, existing NPU stream/allocator APIs, native FT Python
3.14.6/Torch 2.13, isolated CPU numerical ACL simulation on node04/55.

This implements the previously approved PTA-only joint design under a narrower
initial execution profile: `TASK_QUEUE_ENABLE=0`, single caller, inference,
device 0/default external stream, base formats/native allocator, no graph capture
or callback instrumentation. This makes host submission receipts immediate while
device/simulated stream execution remains asynchronous. The existing frozen-call
DependencyScheduler stays separate because generic eager outputs do not exist
before the original body returns.

- [x] Confirm existing shadow/NoOp limitations and remote build/source identities.
- [x] Add real Tensor container visitor and Python e/ei configuration tests.
- [x] Connect ImmediateProtocol/ImmediateRuntime to generated Invoke; validate
  exact schemas/scalars, allocation retention, alias handling and first errors.
- [x] Add direct stream/event, copy, mutation and observation boundary hooks;
  do not add throwing hooks beneath noexcept stream/device exchange functions.
- [x] Build isolated numerical host-sim with lazy FIFO CPU workers and real event
  dependencies. Test missing wait, event reuse, asynchronous failures and math.
- [x] Compile exact native candidate with production entry wrappers and fresh _C.
- [x] Run actual e/ei numerical and lifetime scenarios, including shared inputs,
  metadata aliases, mutation, explicit sync, second caller and failed operations.
- [x] Run original DLRM on the same candidate/backend with fresh off/e/ei process
  arms, rotating order, fixed affinity/data/threads and multiple paired rounds.
- [x] Independently review safety, raw evidence, statistical calculation and
  source/artifact mapping before publishing results and code.

Result: 36 normal scenarios, 16 fault processes, and all 39 cold/performance
processes passed. The 108,000 timed forwards show no E2E benefit: paired median
latency increases 43.25% for E and 43.18% for EI. See `docs/R27_HOST_E2E.md`.

## Validation contract

`test/cpp/eager_parallel/immediate_protocol_test.cpp` exercises actual protocol
dependency decisions with a controllable backend. The Tensor visitor uses actual
ATen containers. A faulting guard fixture verifies that all original result
tensors stay alive while production InvokeImmediate recovers an exception.

`tools/hostsim_r27/` implements CPU arithmetic behind the actual SDK ABI, preserving
the R25 provider. Kernels capture descriptor metadata before cleanup and run in
stream FIFO workers. Events capture specific record generations. Failed work
must make query/synchronize fail; unknown legacy compute fails explicitly.
Explicit scalar bag/Gram oracles check simulator arithmetic independently.

`benchmarks/eager_parallel_r27/host_e2e.py` uses the frozen original model loader,
same state/input identities, eight correctness forwards plus a final timed-output
comparison, actual native scheduled counters and SDK operator counts. Timing is
original `model(*inputs)` through synchronization, excluding setup, correctness
checks and snapshots. The simulator performs real CPU work without artificial
delay; these times cannot establish real NPU speedup.

Fresh process arms use the same native binary, simulator artifacts, eight physical
CPUs, single-thread ATen kernels and natural GIL-off interpreter. Report per-round
median/p95 and median paired ratios, all raw samples, CPU time/context switches,
loaded-library hashes, exact operation counts and process-group closure.

## Explicit limits

Native unchecked raw stream/device operations, raw DataPtr writes, IPC sharing,
allocator checkpoints, custom allocators and unqualified callback domains remain
outside this experimental scope. Public hooks and conservative fallback cover
the admitted native paths; this is not a claim that all possible C++ extensions
are intercepted. CANN/NPU numeric, device overlap and performance qualification
remain false until tested on an actual device.
