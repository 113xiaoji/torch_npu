# Actual E/EI host-sim measurement

The original model runs through the native `ImmediateRuntime` adapter. E sends
EmbeddingBag calls to two producer streams. EI also schedules Cat/BMM/Index on
a consumer stream, with allocation dependencies and event waits. The model's
Python forward and original native operator bodies are retained.

Use a fresh process with `TASK_QUEUE_ENABLE=0`. The public native scope is:

```python
with torch.inference_mode(), torch_npu._eager_parallel.scope(mode="ei"):
    result = model(*inputs)
    torch.npu.synchronize()
```

The experimental profile requires one caller, device 0/default caller stream,
native allocations and base formats. Embedding admits FP32 sum without sample
weights or scale-grad-by-frequency; EI additionally admits out-of-place Cat,
BMM and integral Index. Other operators drain the group before their original
body. Public mutation/observation boundaries and explicit synchronization also
drain. Raw native stream/device changes, custom allocator/deleter callbacks,
instrumentation, graph capture and IPC are outside the admitted profile.

The independent SDK in `tools/hostsim_r27` performs CPU numerical work in real
FIFO worker threads. It replaces the earlier NoOp validation backend. Its CPU
math, simulator queue costs and host scheduling costs all enter E2E timing;
these are not NPU kernel times or predictions of NPU speedup.

`host_e2e.py` takes a pinned experiment configuration and writes a new result
directory. The configuration identifies the exact package, source helpers,
original model tree, SDK libraries, file hashes and eight-CPU affinity. It
rejects drift, shadow substitution, missing scheduled operators, one-stream
fallback, resource errors and active fault injection.

```sh
python host_e2e.py --config CONFIG.json --output NEW_RESULT \
  --arm ei --kind perf --round 0 --warmup 50 --samples 3000
```

Supported arms are `off`, `e`, `ei`. Cold measures the first NPU forward with
zero warmup and one sample, then checks its numerical result. Perf checks eight
full original forwards before its warmup, measures only forward-through-sync,
and checks the final timed output afterward. Startup, snapshots, CPU reference
construction and numerical comparisons are excluded from each sample.

SDK counts must show exactly 26 EmbeddingBag, two Cat, one BMM, one Index, six
Addmm, five Relu and one Sigmoid per timed original forward. Actual SDK stream
counts are differenced across the timed window: E/EI must execute 13 Embedding
calls per producer per forward, and EI must use a distinct consumer for the
Interaction chain. Native intent counters alone cannot pass this gate.

Use the same candidate and SDK for all arms, rotate all six arm permutations,
and repeat complete rounds. Report per-process median and nearest-rank p95,
paired ratios and paired latency changes. Do not pool samples from different
processes or compare this numerical backend directly with earlier NoOp timings.
Keep cold results separate. Retain raw samples, file identities, loaded-library
mapping, CPU topology and process closure receipts with the report.
