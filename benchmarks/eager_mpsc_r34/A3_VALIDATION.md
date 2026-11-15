# R34 A3 verification status — 2026-11-15

**No model performance result is available. Global lock freedom is not achieved.**
The optional bounded MPSC ingress and dedicated FIFO publisher are implemented;
the runtime still uses core/state mutexes, the original PTA FIFO and allocators.
The original mutex ingress remains the default and the comparison uses one
candidate binary. CANN9.1 qualification remains closed; no SDK gate was bypassed.

## Observed evidence

| Check | Observed result | Scope |
| --- | --- | --- |
| Windows C++17/MSVC warning-as-error | 78 contract invocations; 30 repeat processes passed | CPU only |
| A3 host GCC10 warning-as-error | 78 contract invocations passed | CPU only |
| A3 GCC11 AddressSanitizer + UBSan | 78 passed, exit0 | CPU only; intentional quarantine leaks excluded from leak detection |
| A3 Python3.14t, explicit GIL-on lifecycle fixture | 14 integration tests passed | CPU only; separate from natural GIL-off performance runtime |
| Offline campaign contracts | 12 tests passed | No device execution |
| A3 native build and import | Linked candidate; import succeeded with natural GIL=false | No new model forward |
| Independent uncached CANN probe | 24 numerical cases across L1/2/8/64/128/256; max absolute error0; six cancellation cases | One Prepare thread, distinct Execute thread; bypasses queue solely for SDK qualification |
| Independent four-worker CANN probe | Actual query intervals overlapped4; first shape execute/cancel correct | SDK qualification only |
| Cache replay | First shape miss then fresh-address hit4/4 correct | Partial; complete shape/cancellation coverage pending |
| ThreadSanitizer | Compiled; runtime aborted before tests because container denies personality(ADDR_NO_RANDOMIZE) | No race-clean result claimed; container policy unchanged |

Windows denied execution of one old temporary sync-expression fixture, and the
host Python3.9 lacks development headers. The full integration suite was then
run successfully using the existing A3 Python3.14t runtime with explicit
`-X gil=1` for its GIL deadlock regression. This setting does not enter model
performance measurements.

The SDK's executor-cache manager is thread local. A first lookup on a different
Prepare worker misses; same-worker reuse can hit. SDK disassembly also shows
executor destruction restores its HugeMem pool index before consumer cleanup.
These observations do not replace the remaining real lifecycle tests.

## Current resource block

The initially idle physical14 became occupied while SDK validation was running.
The latest recorded inventory at 17:21 +08 shows compute jobs on all16 devices,
HBM99%, utilization20–100%. The failed probe saw only7.08MiB free while its own
allocator held32MiB. Only our probe was terminated; other jobs were untouched.
No performance samples from this contention are accepted.

Candidate native SHA256:
`5182cb707fc80e55ecb7a1634248d54abbbe0a3db46bc0756d16009e119c92e2`

Private evidence archive:64 payloads,114695 bytes,
SHA256 `b174750d32e20ce52cb7f57e75f7c79dcb4435b280d6ed5052bd12166f14e7e9`.
Raw paths and launch recipes remain outside the public repository.

## Remaining acceptance work

1. Obtain one uncontended device and complete actual SDK cache replay,
   cache-hit cancellation and all six shapes. Only then consider an exact
   CANN9.1 binary-profile extension with its recorded evidence.
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
