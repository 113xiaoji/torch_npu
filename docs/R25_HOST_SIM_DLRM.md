# R25 original DLRM host-sim first screen

The native Eager Prepare queue does not improve this host-sim workload. Every
candidate regressed in all three paired rounds. Four workers increased paired
median host latency by 68.2%. This is simulated host-path evidence; the provider
does not compute tensor values, and no real NPU speedup or numerical claim is made.

## Measured result

Run `screen-004-ft`, natural free-threaded CPython 3.14.6, Torch git
`cf30153c4c131c8164ee7798e5022d810682e2cb`, NumPy 2.4.4. Eight physical CPUs
144,146,148,150,152,154,156,158 share NUMA node 1 and one LLC. Torch/BLAS threads
are one. The host is shared; the eight CPUs were idle in the cold availability
sample, with no exclusive ownership claim.

| Arm | Round 0 median ms | Round 1 median ms | Round 2 median ms | Median paired speedup | Paired latency increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 0.814865 | 0.825350 | 0.817300 | 1.0000x | baseline |
| queue_w1 | 1.176920 | 1.161130 | 1.153700 | 0.7084x | 41.2% |
| queue_w2 | 1.249375 | 1.240385 | 1.241775 | 0.6582x | 51.9% |
| queue_w4 | 1.367040 | 1.418945 | 1.374390 | 0.5947x | 68.2% |

Speedup is the same round's off median divided by the candidate median; the
table reports the median of three such ratios. W4's three ratios are 0.596080,
0.581665 and 0.594664. P95 ratios are also below one in every round. Samples
within a process are not treated as independent A/B experiments, and this small
screen makes no formal significance claim.

Four separate cold processes passed before twelve performance processes. Each
performance process warmed up 20 forwards and timed 200, for 2400 timed samples.
Timing covers the original model call through `torch.npu.synchronize()` return.
Imports, model construction/transfer, pool creation and warmup are excluded.
Native and SDK diagnostic tracing are disabled during performance. Every arm
uses the same new package and the same simulator libraries, with no injected
preparation CPU loops, sleeps, or delay knobs.
Enabled arms include the queue's normal caller-state preparation and fallback
policies; this measures the net enabled feature, not an isolated GetWorkspace call.

Mean cumulative process CPU per batch, taking the median of three windows,
increased from 1.087 ms off to 1.985/2.299/2.917 ms for W1/W2/W4. More workers
therefore consumed more CPU as well as increasing wall latency. This simulator's
GetWorkspace performs metadata validation and bookkeeping with zero workspace;
the queue adds ownership checks, handoffs and synchronization. Each enabled
forward also records five fallback fences for unsupported Relu preparation.
These observations identify overhead that must be considered; they do not
separately quantify each overhead's contribution or predict real CANN behavior.

## Identity and mechanism

The original DLRM source, model functions and MLP bias are preserved: FP32,
batch 128, 26 sparse tables, 1024 rows/table, width 16, two indices per bag,
bottom MLP [13,64,32,16], top MLP [367,64,32,1], dot interaction and 455153
parameters. CPU source/state/input identities match the fixed R22 assets.
The inference loader omits only the previously audited unused SummaryWriter
import; it does not rewrite the model forward. Historical dependency snapshot
integrity is verified separately from actual native NumPy import path/SHA.

All 16 processes passed identity and lifecycle gates. Enabled arms execute
exactly 26 native EmbeddingBag admissions and SDK Prepare/Execute calls per
forward; each timed process contributes exactly 5200, for 46800 timed native
admissions across the nine enabled performance processes. Actual native worker
TIDs match W1/W2/W4 and are distinct from caller and FIFO consumer TIDs. Native
admission/Execute sequence counts and hashes agree. Queue credits/retained bytes
and SDK live descriptor/executor/HugeMem gauges are zero after close; every
owned child process group closes. This is partial operator coverage, not an
all-ATen parallelization or numerical validation result.

| Input/artifact | SHA256 |
| --- | --- |
| original DLRM source | `9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5` |
| original state tree | `6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19` |
| original input tree | `239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6` |
| final full-build main library | `c0ec43faf52367df3cb1d999b71d98ba8a0be37056b9184d59c39f59987449aa` |
| rebuilt FT `_C` | `2b1b0fbd64b4414c1a84e2dc4b0d5c7285589e84e7a14922011444b82df915b9` |
| isolated typed `libopapi.so` | `9f96e344e7f72698551c179dc0901277b06dc66e7b6a0e7fb16188dbbded5878` |
| runner worker source | `2bf916e9cc11a85d20815c43e7c4550309cbca79a68360fe3bec4b27e5c3b8d4` |
| exact-count contracts source | `d4a155977918a02e3fbb0ac93d5b4a1484ebd8b06bf7545cc1eca3766dc4311c` |
| retained evidence archive | `7ef1c8489a4eeb0bd9e1543c2e9f5b73956d3e1b3dba760d8235e740427956fe` |

The archive has 124 hash-pinned evidence files plus its member manifest; it
contains no native binary. Raw evidence remains with the task rather than in
Git. The final candidate used a fresh full AArch64 FT build and new `_C`, not
object reuse. All 7141 materialized source files passed build postchecks.

## Changes needed to test the real eager path

The separate HostSim build option is off by default. It attests the explicit
typed provider and reports `host_sim`, `real_sdk_qualified=false`, empty real
`qualified_ops` and a separate EmbeddingBag simulation qualification. The real
CANN 9.2 qualification and frozen R24 overlay are unchanged.

The R25 overlay checks the full thirteen-argument tuple type before compiling
EmbeddingBag-specific expressions. It also treats `nullopt` and an engaged
undefined Tensor as absent optional sample weights; a defined empty Tensor is
still present and rejected. The latter fixes original DLRM admission because
Torch lowers high-level None through an undefined Tensor before the optional
backend argument. Existing ordinary and owned conversion both map the two
absent forms to SDK nullptr. Independent RED/GREEN domain and actual conversion
fixtures cover this without weakening storage, TLS or SDK guards.

Local profile tests pass 6/6, Linux runner fixtures 29/29, SDK native contracts
and 7/7 ctypes tests, and build/rebind fixtures 11/11. Independent review closed
the exact-count and historical source-pin findings. See the
[runner protocol](../benchmarks/hostsim_r25/README.md) and
[isolated build instructions](../tools/hostsim_r25/build/README.md) for reproduction.
