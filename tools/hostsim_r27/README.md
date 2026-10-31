# R27 CPU numerical host simulator

This isolated SDK runs the original native TorchNPU operator path with CPU
arithmetic. It is for validating the native E/EI scheduling adapter without an
NPU. It does not qualify CANN, NPU numerical behavior, device overlap, or device
performance. R25 sources and binaries remain unchanged.

`runtime.cpp` supplies one bounded FIFO per ACL stream. A worker starts only
on the first enqueue, so creating the normal 32-stream pool creates no threads.
Independent streams have independent workers. An event records an exact FIFO
ticket; a wait captures that event generation and delays only its consumer
stream. Query and synchronize propagate asynchronous failures. Synchronous
copies and allocation release drain the device; asynchronous copies enter the
selected stream FIFO. The pause APIs are deterministic test gates and are not
used by model or performance runs. No artificial delay is added.

`opapi.cpp` captures tensor/scalar metadata by value at preparation and transfers
the executor callback to its stream at launch. Destroying ACL descriptors after
launch does not invalidate queued metadata. Backing allocations must remain
alive through completion, as with a real asynchronous API. CPU ATen views honor
shape, strides, dtype, and storage offset. Results are copied into the native
output allocations. Supported numerical APIs are EmbeddingBag sum (including
the inference `sparse` flag), Add, Relu, InplaceRelu, Mm, Addmm, Cat,
BatchMatMul, Index, Sigmoid, Cast, and InplaceCopy. Sum-mode auxiliary outputs
copy CPU values when shapes match and otherwise are zeroed; this is an explicit
simulator policy, not CANN auxiliary-value qualification.

`build.py` verifies the existing ACL stub's exact hash, then generates a new
`libascendcl.so` overlay and a fail-closed `libacl_op_compiler.so`. Legacy compute
and raw kernel launches return unsupported. A missing OpAPI therefore cannot
silently return success through the legacy fallback. All generated sources,
compiler commands, input hashes, and output hashes are recorded in the fresh
SDK directory. The C++ runtime is shared by `libascendcl.so` and `libopapi.so`.

Example on native55:

```sh
taskset -c 160,162 python3 build.py \
  --output /home/r27-pta-ei-20261031/sdk-NEW \
  --source /home/r26-pta-joint-20261029/attempt-001-entry-shadow-ft/src/torch-npu \
  --base-sdk /home/r26-pta-joint-20261029/sdk/provider-002 \
  --torch /root/ft-pytorch-repro/envs/py314t/lib/python3.14t/site-packages/torch
```

For execution, put the new SDK directory first in `LD_LIBRARY_PATH`, use
`QWEN_ACL_HOST_SIM=1`, `TASK_QUEUE_ENABLE=0`, and keep
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Bind the entire process
to the declared CPU budget so simulator workers inherit it. Set Torch intra-op
and inter-op thread counts to one before model execution. Compare arms using
the same numerical SDK and worker configuration. Measured latency includes CPU
math and simulator queues; it is not an estimate of NPU performance.

Verification helpers:

* `test_provider.py SDK` checks actual async copy, event ordering, a missing-wait
  negative control, event-generation capture, event-query enum values,
  descriptor cleanup before completion, explicit Python bag sums, and a
  strided/offset Gram product against explicit Python dot products.
* `test_failure.py SDK` queues an invalid output shape and requires asynchronous
  errors to remain errors through both event and stream queries.
* `test_faults.py SDK` checks before/after-enqueue failure effects, an exact
  relative call ordinal, and persistent query failure/reset.
* `test_index.py SDK` distinguishes empty Int/Long indices from the empty FP32
  tensor OpPlugin uses as a None sentinel, including empty output shapes.
* `smoke_original.py` loads the existing frozen original-model loader and checks
  a complete native NPU forward against its CPU original-model output.

The runtime diagnostics report submitted/completed tasks, worker native TIDs,
per-stream counts, event records/waits, and errors. OpAPI diagnostics report
descriptor balance, per-operator counts, active/peak simultaneous kernel
callbacks, and live handles. Peak callbacks establish host concurrency only.
Each operator also records `stream_executions`, keyed by the actual ACL stream
handle. This independently confirms where successful numerical callbacks ran.

For native-adapter lifecycle tests only,
`R27HostSimInjectFault(operation, nth_call, after_enqueue, repetitions)` arms
`event_record`, `event_wait`, `stream_query`, or `event_query`. The ordinal is
relative to activation; a positive repetition count injects that many failures
starting at the ordinal, and `-1` persists until `R27HostSimClearFault()`.
Query operations admit only `after_enqueue=false`. An after-enqueue record or
wait really submits its effect before returning failure. Set/clear injection
from the single test owner while no API invocation is in progress. Normal runs
must report `injected_faults=0` and `fault_active=false`; they add no test work
or delay. Fault runs are separate fresh processes and cannot enter performance
cohorts.
