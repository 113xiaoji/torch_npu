# Independent CPU simulation review — 2026-10-29

Verdict: no confirmed P1/P2 finding in the implemented, documented CPU simulation domain. The completed campaign is usable evidence for the production queue core's bounded parallel metadata preparation and FIFO CPU consumption with real aten operators. It is not validation of live TorchNPU dispatch, CANN/ACLNN preparation, real device retirement, NPU performance, SDK thread safety, or Python 3.14 free-threading.

This review independently inspected the capture/replay driver, campaign runner, native bridge/builder, production `EagerPrepareQueueCore.h`, frozen model source, tests, and completed JSON. A second read-only audit independently reconstructed the workloads and checked numerical/statistical/provenance fields. Only this review document was added; no source or formal result was changed.

## Reviewed identities

- Formal report: `results/full-20261029-01.json`, 1,262,908 bytes, SHA256 `c32a46fb5b61911740386097f235f8ce2895bc4e6e369b9d2601e273c875c2d9`.
- CPU source manifest: SHA256 `2b89dd773477e6480beaf2b46238e4ab4759f8a1b7ba0be1b75b6772efcfca7e`; all 14 listed byte counts and hashes match disk and the receipt.
- Directly included production core: `704f99c5cfd8faeb8640dd52965f8e5b3d85a1a9069a26ab68d957d0b6fdb23d`; the loaded extension's embedded `core_info()` agrees with both the source and formal receipt.
- Native bridge: `2a99c55763847ec406e21faa6f780ef43dd6010d13b43c95eca183d729a5c744`; builder: `1d0d02cf0e3d834076bd00f7bc8fe4794a290179599c74e8cc26b40e43e092f9`.
- Matching build cache: `tools/eager_prepare_cpu/.cache/fc7fec70450b62e7f4ae`; its build manifest source/Python identity matches the current dependencies. Current `.pyd` SHA256 is `043d9b00e37657bf380bd53216d740264727e64831ca6689cc2aa63d8321ceb7`. This current artifact check supplements the campaign's recorded source and embedded-core identities; it is not a separately signed campaign binary receipt.

## Correctness and lifecycle assessment

Capture records actual aten overloads and nested value references from an original eager execution. Its tape retains initial inputs/parameters/buffers, but does not retain intermediate tensor answers. Each replay creates a fresh environment and resolves dependencies from that replay's actual aten outputs. Independent eager per-call outputs and the complete final output are compared. Existing tests and an additional native-queue changing-input experiment prevent a tape of precomputed answers from satisfying the checks. Views and multioutputs execute through their actual aten implementations.

The native workers own copied integers, do real arithmetic work, and never access Python objects, Torch tensors, or tensor data. Inline and worker arms call the same `RunPrepareWork` with globally fixed iterations. Besides reproducing all model/cost aggregate native checksums, the second auditor independently implemented its wrapping-integer/floating-point recurrence in Python and matched first-op traced checksums at all four formal costs.

The bridge uses the actual production core for Submit, Consume, and Retire. Its ordered native marker precedes the real CPU operator; the Python consumer acknowledges retirement only after that operator returns. The bridge disallows a second `next()` across an unretired CPU operation. Capacity/metadata bytes stay reserved through this acknowledgment. Prepare failures, CPU exceptions, and abort paths wake waiting producers/consumers, release native metadata and join workers. The CPU consumer's original exception takes priority over a resulting producer-admission exception. The deterministic reversed-ready, K=1, gated-abort and failure tests all passed freshly.

The bridge's publication callback commits by returning immediately; this CPU campaign does not exercise the live TorchNPU FIFO's early `MarkCommitted` integration path or real device-fence acknowledgment. Those require their separate integration/host/NPU validation gates.

## Formal campaign audit

The receipt records Windows x64, CPython 3.13.12 with the GIL enabled, Torch 2.7.1+cpu, process affinity CPU 0–7, and Torch intra/inter-op and BLAS/OpenMP environment thread settings of one. All producer/consumer/native threads share that process CPU budget.

All four frozen models and all 16 model/cost combinations are present. Operator counts are recommendation 70, branch MLP 10, branch CNN 11, and transformer 32. Costs 0/10/50/200 µs use exactly 0/4,572/23,812/95,915 common iterations across every model and applicable arm. Each model/cost has three rounds in forward/reverse/forward arm order, five warmups and 30 retained samples per arm. Pools persist across each arm block's warmups/samples. Capture, calibration and pool construction/destruction are explicitly excluded; the measured forward includes replay, submission/copying, work, waiting, CPU execution and retirement.

Independently checked:

- All 64 pre-timing correctness records, per-call counts/full-output success, model/input full-data hashes, metadata byte arrays, and all 16 model/cost work checksums.
- All 7,200 raw wall samples and 7,200 raw process-CPU samples, every per-arm and pooled mean/P50/P95/P99/min/max, all 48 queue speedup/inverse pairs and 16 direct-eager overhead ratios. All recomputed values agree.
- All 193 native snapshots: 48 correctness, 144 formal arm blocks and one separate proof. Counts agree with the operator counts and, for each formal block, 35 forwards. Peaks respect K=8 and 1 MiB; retained slots/bytes, queued/preparing/ready work, live workers and cancellations are zero after shutdown. Formal trace collection is disabled.
- All cold trace sequence sets and original consume order. Each ticket's prepare interval ends before its own consume interval, consume intervals are ordered, and recorded overlap fields recompute from actual intersecting intervals on distinct native TIDs.

For context, the verified pooled `replay_serial P50 / queue_w4 P50` values are:

| Model | 0 µs | 10 µs | 50 µs | 200 µs |
|---|---:|---:|---:|---:|
| Recommendation | 0.4062 | 0.7262 | 2.5591 | 3.9979 |
| Branch MLP | 0.5055 | 0.7688 | 2.0286 | 3.1290 |
| Branch CNN | 0.7208 | 0.8901 | 1.3298 | 3.1383 |
| Transformer | 0.3606 | 0.7119 | 2.5607 | 3.6993 |

Values below one retain the observed queue slowdown at low synthetic costs. These are pooled simulation ratios, not a CANN cost prediction or an estimate of NPU gains. Applying work to every recorded aten call, including metadata/view calls, is explicit and materially limits transfer to real NPU workloads.

The separate approximately 50 ms/op cold proof has four actual worker TIDs, 70 prepare records, 3,625,000,000 ns summed native thread CPU over a 941,664,300 ns prepare wall span: ratio 3.849567197. It records 7,722,837,383 raw thread cycles and no zero CPU-time records, with native/native and prepare/consume overlap. This establishes native CPU activity and concurrency. It is excluded from every formal cost, warmup, sample and ratio. Short formal-cost cold traces honestly retain zero GetThreadTimes deltas where Windows accounting is coarse; cycles were not converted into invented nanoseconds.

## Fresh verification performed by this reviewer

```text
python -m unittest discover -s test/eager_prepare_cpu -v
Ran 17 tests in 5.063s — OK

python -m unittest discover -s test/eager_prepare_cpu_native -v
Ran 13 tests in 0.007s — OK
```

A separate in-memory branching/view/multioutput model was captured once, then replayed with five different input tensors through one persistent W4/K1 runner, comparing every call and final output against a new original-eager reference for each input. All five forwards passed, base sequences increased by five operators each forward, 25/25 tickets prepared/executed/retired, peak slots stayed one, and all resources were zero after shutdown. No full formal timing campaign was rerun during review.

Remaining scope limits are appropriately disclosed: read-only static-shape CPU aten inference, no supported arbitrary Python side effects, no custom operators/RNG/data-dependent scalar control or dynamic output shapes, and no claim that this is a general transparent eager backend. Timed arms check their final timed output against direct eager; exhaustive per-call checking happens separately before performance timing. Linux-specific native code was not executed on this Windows host.
