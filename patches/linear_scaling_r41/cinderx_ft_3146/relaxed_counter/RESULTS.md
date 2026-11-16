# FT exact call-counter relaxed RMW experiment

The one-line FT counter candidate is correct under the tested3.14.6t configuration and shows a small four-caller improvement relative to the original full-class CinderX artifact. One/two-caller results do not show consistent gains. This does not establish linear scaling or improvement over plain CPython.

| Callers | Original CinderX median TPS | Candidate median TPS | Paired TPS ratios, rounds0/1/2 | Paired median / mean |
|---:|---:|---:|---|---|
| 1 | 517.674 | 515.248 | .98452 / 1.00507 / 1.00104 | 1.00104 / .99688 |
| 2 | 955.716 | 959.051 | .99762 / .97543 / 1.00349 | .99762 / .99218 |
| 4 | 1783.517 | 1787.095 | 1.01984 / 1.02466 / 1.00201 | 1.01984 / 1.01550 |

The TPS columns are window medians; paired ratios are computed within matching rounds and then summarized. They differ from a ratio of medians.

| Arm / callers | Pooled mean latency μs | Pooled p99 latency μs | Mean forward caller CPU μs | Mean response caller CPU μs |
|---|---:|---:|---:|---:|
| Original /1 | 1919.336 | 2012.267 | 1911.045 | 1914.227 |
| Candidate /1 | 1925.831 | 2020.998 | 1914.831 | 1918.008 |
| Original /2 | 2067.247 | 2207.446 | 2054.548 | 2057.863 |
| Candidate /2 | 2084.303 | 2206.455 | 2075.080 | 2078.288 |
| Original /4 | 2234.657 | 2380.123 | 2222.350 | 2225.169 |
| Candidate /4 | 2207.002 | 2347.342 | 2196.708 | 2199.477 |

PSS was not sampled. Caller clocks are not whole-process or background-thread CPU.

## Minimal change and evidence

`candidate.patch` changes only FT `Ci_code_extra_incr_calls` from `_Py_atomic_add_uint64` (seq_cst) to `__atomic_fetch_add(...,__ATOMIC_RELAXED)`, with corrected comments. Each original increment, exact count, cold gate, non-FT branch and ownership/frame operation remains intact. `consumer-audit.md` records all readers, producers, threshold users and lifetime/publication paths; root independently reviewed it before tests/build. All count readers already use relaxed loads; compiled entry, classification and ROI state publish through distinct atomics/locks.

The original CPU FP profile observed the call-count block at ELF0x37e7d4 (15/c1 and31/c4 tip samples); the surrounding ARM instructions retain acquire `skey_word` load and cold-bit test. At0x37e7d0, the original SO calls `__aarch64_ldadd8_acq_rel`, and the candidate calls `__aarch64_ldadd8_relax`; neighboring instructions match. `counter-asm-baseline.txt` and `counter-asm-candidate.txt` preserve the disassembly. This sampled block is a small contributor, not an explanation of the entire earlier3%-5% regression.

A full isolated source copy compared3308 files to the frozen source: only `cinderx/Common/code_extra.h` changed. The actual FT header tests passed for both headers: four writers exact800000, four monotonic readers, threshold checks and modulo-uint64 rollover. Layout56/offset0 and the non-FT branch match. Existing CountCalls Auto/All, AutoJIT policy and gate/ROI regressions passed in both artifacts:52 unittest executions,36 plugin-child native-SHA/GIL attestations. Tests retained their original bodies/applicability wrappers. A test-only compatibility launcher and actual `_cinderx_auto` bootstrap select the private artifacts without changing the immutable prefix. Initial bootstrap and build-path validation failures are retained in the evidence.

The build reused228 unchanged objects, recompiled all64 header dependencies identified from the original292 dependency files, recreated four affected archives and linked a separate SO. An independent command comparison verifies unchanged optimization flags; only source/output paths and dependency-file bookkeeping differ. Original objects/archives/SO stayed unchanged. No default package or user dirty checkout was replaced.

## Native qualification and formal validation

The same new worker source closure is used for both full-class artifacts: `e388d3f9d9774f904b9988b8b999743c6f5e88c9e493cf40a420566da745e02f`. Original model, weights, all bank8 bytes, batch128, engine, requests and CPU64..71 / Torch1/1 remain fixed. CPython3.14.6t SHA79db6884..., Torch2.13+cpu and NumPy2.4.4 run naturally GIL off.

Each artifact has new1/2/4-caller native qualification: all10 selected generic user methods remain compiled; each of the five independently inventory-hot methods has positive exact native active-frame samples for every caller; warmup/ROI/validation deopts are empty. Other five methods are compilation-only qualified. Inclusive frames overlap and prove neither entry counts nor full native coverage. Qualifications do not sample formal timing ROIs.

Six smoke cells precede18 interleaved performance cells, each3000 requests and warm24/caller, with no observer. Independent raw audit passed24 receipts /54,192 worker-validated outputs, maximum absolute error0; all metrics were recomputed. It independently verifies the complete identity, derived required set, four compilation stages, every caller GIL/TID, nonempty critical Torch/CinderX mappings, all raw native artifact hashes and CPU-only evidence. No Ascend/torch_npu provider mapping appears. The exact devices=[] owned CPU container is recorded before/after.

## Artifacts

- Original SO: `b0809aac744059cfcf7069fae65ec2f1c6041231a81f1d4094b94384017d9315`.
- Candidate SO: `75504d191304c5e188efee703c6a08f6d4e9eda4f3a21553b0bd5a6e8aceb5d1`, included in the archive and retained at `/tools/r42-cinderx-relaxed-counter-20261117/pylib/_cinderx.so` in the owned CPU container.
- Evidence archive `evidence-ab-001.tar.gz`:32,099,118 bytes /332 files; SHA `5b12d90374b2d243f3a76a62eea466c9b8c87aeb48796e7add18e442558eec3f`.
- Full file manifest: `evidence-ab-001.sha256`.
- Raw metrics audit: `independent-audit-ab-001.json`; command/CPU-only checks: `independent-build-and-cpu-only-review.json`.
- Reproducer/build plan, canonical dependency validation, test logs, six raw native diagnostics and24 formal raw receipts are in the archive. The final report and ARM disassembly were added as external review artifacts after sealing the raw archive; its SHA is unchanged.

This is an experimental patch ready for root review, not a default replacement or a full explanation of the CinderX regression.
