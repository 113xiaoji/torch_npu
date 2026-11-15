# R34 A3 verification status — 2026-11-15

**No model performance result is available. Global lock freedom is not achieved.**
The optional bounded MPSC ingress and dedicated FIFO publisher are implemented;
the runtime still uses core/state mutexes, the original PTA FIFO and allocators.
The original mutex ingress remains the default and the comparison uses one
candidate binary. Exact CANN9.1 v1 qualification is complete for the reviewed
fixed shapes and inputs. The Eager-only profile verifies every required
symbol's mapped binary size/SHA; R22 and the older9.2 capability stay unchanged.

## Observed evidence

| Check | Observed result | Scope |
| --- | --- | --- |
| Windows C++17/MSVC warning-as-error | 78 contract invocations; 30 repeat processes passed | CPU only |
| A3 host GCC10 warning-as-error | 78 contract invocations passed | CPU only |
| A3 GCC11 AddressSanitizer + UBSan | 78 passed, exit0 | CPU only; intentional quarantine leaks excluded from leak detection |
| A3 Python3.14t, explicit GIL-on lifecycle fixture | 14 integration tests passed | CPU only; separate from natural GIL-off performance runtime |
| Offline campaign contracts | 12 tests passed | No device execution |
| A3 native build | Three replacement translation units compiled and linked | Preserved R32 empty-SUM object source and original route |
| Natural GIL-off CPU runtime check | Exact9.1 EmbeddingBag capability=true; Relu=false; fixed profile=true | NPU was not initialized; this is not application qualification |
| Uniform real SDK probe | 192 cases:144 numerical executions,48 cancellations; all passed,max error0 | Four Prepare threads and a separate consumer; six lengths |
| Original ragged SDK probe | 800 cases:600 numerical executions,200 cancellations; all passed,max error0 | Original25 nonempty MetaSyn26 tables;50850indices unchanged |
| Cache replay | Fresh-address hit, worker-local miss then hit, hit cancellation and later hit passed | Cache remained enabled; v2 unqualified |
| Original off-route model preflight | Full forward and all intermediate stage comparisons passed | Correctness only; not an on-route or comparative performance result |
| R34 synchronous fallback regression | Old capability lookup reproduced failure; corrected lookup passed | Eager-only9.1 qualification stays stable during fallback |
| ThreadSanitizer | Compiled; runtime aborted before tests because container denies personality(ADDR_NO_RANDOMIZE) | No race-clean result claimed; container policy unchanged |

Windows denied execution of one old temporary sync-expression fixture, and the
host Python3.9 lacks development headers. The full integration suite was then
run successfully using the existing A3 Python3.14t runtime with explicit
`-X gil=1` for its GIL deadlock regression. This setting does not enter model
performance measurements.

The SDK's executor-cache manager is thread local. A first lookup on a different
Prepare worker misses; same-worker reuse can hit. SDK disassembly also shows
executor destruction restores its HugeMem pool index before consumer cleanup.
Every reviewed ragged table had a wave with four real SDK queries overlapping.
This establishes SDK preparation concurrency, not model-level speedup.

## Current resource block

An initial occupancy interruption cleared and SDK verification completed. At
17:52 +08 the runtime container stopped with exit137, OOMKilled=false, before
the new candidate's application qualification. The17:55 inventory then showed
sglang compute processes on all16 devices, including physical8/14/15. The last
group held approximately59–60% HBM despite utilization0; those devices were
not free. No other jobs were stopped or modified. No comparative performance
samples were collected or accepted from these changing resource conditions.

Candidate native SHA256:
`d8943e56a5b234b2dc7cb0091861087a611437661b0ddd6479d5ca610a7ccaa4`

The SDK profile, source hashes and both qualification receipts are described
in [the overlay documentation](../../patches/eager_mpsc_r34/README.md).
Raw results, resource inventories and launch recipes remain in the private
evidence archive, outside the public repository.
Archive002 has109 payloads,243381 bytes, SHA256
`8dcb37235a2949c7894f2dccb7eec968b6db719e1a13fd1d9ef3f37d6d09f36a`.

## Remaining acceptance work

1. Obtain one uncontended device and a stable runtime container. Run the new
   candidate's real queue qualification, including the full unchanged model.
2. Run `multi_caller.py` for both ingress modes: four concurrent callers,
   correct outputs, actual worker/publisher/consumer TIDs, matching sequence
   receipts and complete retirement. Join all callers before cold close;
   concurrent model/fallback and lifecycle calls are outside the contract.
3. Run the bounded DLRM diagnostic to measure actual Prepare overlap, then
   seven correctness arms and the matched off(TQ1)/mutex/mpsc workers1/2/4
   E2E campaign. Keep the original model, mixed inputs, cache and precision.
4. Report medians, tails and CPU cost from accepted raw samples. A faster
   queue microbenchmark or an SDK overlap interval is not model speedup.

The CAS ingress removes `submit_mu` from that route. It does not remove
`state_mu`, integration counters' mutex, blocking publication ACKs or downstream
locks. Snapshot explicitly reports `global_lock_free=false` and the remaining
core mutex. Any future claim of a globally lock-free path needs a separate
implementation and proof covering those stages.
