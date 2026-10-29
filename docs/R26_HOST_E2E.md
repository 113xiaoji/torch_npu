# R26 original DLRM host end-to-end screen

The deployed R26 shadow mode did not improve host latency. It was slower in all
ten paired rounds: median paired latency increased **2.1011%**, with **1.9897%**
higher p95. This measures native entry diagnostic cost. The production E/EI
scheduling adapter is not connected, so this experiment does not measure the
proposed parallel optimizer or real NPU performance.

## Result

Each row is the median of ten fresh-process metrics. The percentages above come
from same-round off/shadow ratios, not from dividing these aggregate rows.

| Mode | Median latency, ms | p95 latency, ms | Process CPU per batch, ms | Voluntary switches per batch |
| --- | ---: | ---: | ---: | ---: |
| same-package off | 0.812633 | 0.829705 | 1.077572 | 84.045 |
| same-package shadow | 0.828575 | 0.846550 | 1.094676 | 84.010 |

| Round | Off median, ms | Shadow median, ms | Off / shadow |
| --- | ---: | ---: | ---: |
| 0 | 0.811560 | 0.832850 | 0.974437 |
| 1 | 0.801035 | 0.839195 | 0.954528 |
| 2 | 0.810070 | 0.827610 | 0.978806 |
| 3 | 0.818465 | 0.825560 | 0.991406 |
| 4 | 0.809525 | 0.825090 | 0.981135 |
| 5 | 0.813705 | 0.830280 | 0.980037 |
| 6 | 0.814360 | 0.827900 | 0.983645 |
| 7 | 0.814480 | 0.832280 | 0.978613 |
| 8 | 0.814280 | 0.824595 | 0.987491 |
| 9 | 0.805630 | 0.829250 | 0.971516 |

Median paired off/shadow ratio: `0.9794216488424812`. Its round range is
`0.954528–0.991406`. Median paired p95 ratio: `0.9804915881527159`.
All ten median and p95 comparisons favor off. Extra cumulative CPU is observed,
while voluntary context switches are essentially unchanged. This is consistent
with diagnostic bookkeeping cost, without attributing measured percentages to
individual locks or functions. No formal significance claim is made; individual
loop samples are not treated as independent experiments.

## Fixed protocol

- Host: node04 / the existing 55 host-sim machine; same boot as prior validation.
- Eight physical CPUs 144,146,148,150,152,154,156,158; NUMA node1, one LLC.
  These CPUs were idle in the cold availability observation. The host is shared;
  exclusive CPU ownership or a noise-free entire window is not claimed.
- Natural FT CPython3.14.6, Torch git
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, NumPy2.4.4, GIL=false throughout.
- Same candidate package and NoOp SDK in both arms. Same original DLRM source,
  weights, input and FP32 batch128/26-table configuration; forward unchanged.
  The frozen loader omits only the audited unused SummaryWriter import.
- Two independent one-forward cold gates, then ten matched fresh-process rounds.
  Round order alternates off/shadow and shadow/off. Every perf process warms up
  50 forwards and times 3000: **60,000 timed forwards** in total.
- Each sample starts before `model(*inputs)` and ends after
  `torch.npu.synchronize()`. Import, construction, transfer, configuration,
  warmup, snapshots, rusage reads and close are outside individual samples.
  All measured windows exceed two seconds (2.408–2.525 seconds).
- Torch intra/inter-op and OMP/MKL/OpenBLAS/NumExpr threads are one. No synthetic
  Prepare work/delay, SDK trace or Python profiler is enabled. Native shadow
  counters are intentionally the candidate behavior being measured.

## Validation and identities

Every process passed source/artifact/mapping/model identity, natural GIL and
affinity checks. Each perf process independently records exactly 78,000 timed
SDK Embedding Prepare/Execute calls and 79,300 including warmup. Shadow also has
78,000 timed native Embedding entries with 312,000 outputs, 6000 Cats with 87,000
inputs, 3000 BMMs and 3000 Index calls. Off has no native shadow admissions.
Timed deltas and warmup-inclusive totals are checked separately.

SDK descriptors/executors/HugeMem gauges close to zero, counters balance and
native errors/active calls are zero. All 22 owned processes exit0 and their
process groups close. NoOp output is checked only for shape `[128,1]`, FP32 and
NPU0 metadata. This is not a numerical golden or a device overlap result.

| Identity | SHA256 |
| --- | --- |
| Main candidate | `84a0ac2b98a05c041ed7b8537caf849f3c1bf4d25a53eb916ffdb76606c6def3` |
| Fresh FT `_C` | `7b498e65b46fd508132786c639dc121b65997e3fabfbc9675c83e29e33a4ddcb` |
| NoOp OpAPI provider | `9f96e344e7f72698551c179dc0901277b06dc66e7b6a0e7fb16188dbbded5878` |
| E2E worker | `e7870d37abdd21eb98d5804c62819188ca845e6ae8a2699a1f5ba1c822ca2d9c` |
| Private evidence archive | `59e284dd7dd1f372a73217304e8d85d8e72c6b95943ec4b6248484ba92cf19af` |

The archive has 116 pinned files plus its member manifest, including the exact
controller, argv, preflight, config, lifecycle and raw samples. Independent
review recomputed all rounds from raw samples and replayed identity/count/
resource checks with no failure. Raw logs and binaries are not committed.

## Interpretation

The public runtime currently supports off and original-order shadow diagnostics.
The result is approximately 2.1% diagnostic overhead, not failed E/EI acceleration.
Measuring that optimization requires its device adapter, allocation leases and
complete effect coverage first. The older R25 four-worker run used a different
runtime path; comparing it with this table cannot establish R26 optimizer gain.
