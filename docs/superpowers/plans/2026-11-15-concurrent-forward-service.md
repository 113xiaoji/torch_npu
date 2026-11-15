# Concurrent forward service implementation plan

**Goal:** Measure actual full-model request concurrency under the same free-threaded runtime with controlled GIL mode and explicit completion.

**Architecture:** Python closed-loop clients share one eval model but own input slots, thread-local inference/stream contexts and completion events. A model-independent engine captures request timing and outputs; adapters provide full DLRM CPU/NPU inference. A serial controller runs fresh processes and audits workload/runtime identities.

**Tech stack:** Python3.14.6t, Torch2.13, frozen D894 torch_npu/CANN9.1, C++ production unchanged.

- [x] `benchmarks/forward_concurrency_r35/engine.py`: deterministic barrier/closed-loop tests observed RED then GREEN; unique request IDs, fixed aggregate work, warmup/start synchronization, bounded error propagation and timing implemented.
- [x] `contracts.py`: RED/GREEN throughput/latency/overlap/identity and CPU/NPU separation; native snapshots/mappings/close gates independently recomputed.
- [x] `real_service.py`: frozen model/request-bank references, per-worker context/input slots, shared/private streams and per-request completion; validated outputs outside timing. Close/report failure paths tested.
- [x] `campaign.py`: fresh bounded processes, retained invalid initial mapping-audit artifact; no CPU/shadow/NPU conflation.
- [x] CPU full six-case GIL/caller matrix and predeclared A3 subset (three correctness/seven performance cases) completed; 39000 performance requests validated. Single-round pilot limits retained.
- [x] Independent code/data review, README/RESULTS and exact source closures archived. Unrelated worktrees and production C++ sources unchanged.
