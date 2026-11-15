# R34 A3 verification status — 2026-11-15

**Performance comparison is now available; global lock freedom is not achieved.**
The completed seven-arm pilot and reversed key-group repeat show no speedup for
the tested D894 fixed-input warm-cache workload. See the measured tables and
same-binary cache/phase evidence in [PERFORMANCE_20261115.md](PERFORMANCE_20261115.md).
The earlier resource interruptions below are historical; they do not invalidate
the later ten accepted uninstrumented performance windows.
The new queue has now passed actual A3 DLRM correctness and four-caller order /
retirement qualification. [The machine-readable results](evidence/real_queue_qualification_20261115.json)
contain the unchanged model/input identities, counters and diagnostic spans.
That qualification JSON is an earlier immutable snapshot: its zero performance
windows apply to the interrupted19:32 retry. Current measured windows are in
[performance_pilot_20261115.json](evidence/performance_pilot_20261115.json).
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
| Actual MPSC4 DLRM | Eight full forwards plus all intermediate stages passed; full output max error8.94e-8 | Original B128/T26/V1024/D16 and50850IDs unchanged |
| Actual ordered queue | Timed correctness delta75 real SDK calls; whole process300 tickets executed and retired in order, zero retained owners | Real CANN9.1, original PTA FIFO; not shadow |
| Four concurrent callers | MPSC and mutex each passed32 real calls; four Prepare workers, one consumer; publisher identity checked | Correct outputs, complete trace/order/retirement ACKs |
| Two-forward cold diagnostic | Both routes reached Prepare-span peak4;50 real calls each | Inclusive cold Prepare spans; not steady-state E2E |
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

## Earlier resource interruptions

The19:12 retry initially saw resident SGLang processes. When compute processes
cleared, the first independent-container attempt still failed at aclInit507008:
kernel logs showed a UDA device-namespace conflict. After that namespace owner
exited, a fresh process passed actual model qualification and all four auxiliary
checks above. No driver/sharing/security setting or other service was changed.

The new test container then received Docker SIGTERM and SIGKILL at19:32 +08,
exiting137 with OOMKilled=false. Another TP16 namespace subsequently reserved
all16 devices. The seven-arm cohort failed before its first correctness worker;
**zero valid performance windows were collected**. The event record does not
identify who requested the stop, and no attribution is made.

Read-only resource monitoring ran19:29:09–19:36:45,114 samples at approximately
4-second intervals, and observed foreign processes in35 samples. Qualification
had completed at19:26:21, before monitoring began; this monitor cannot establish
an exclusive qualification interval. Numerical/order correctness remains valid,
but the cold spans below cannot establish a performance improvement.

| Cold Prepare diagnostic | Calls | Span peak | Wall p50 | Total thread CPU |
| --- | ---: | ---: | ---: | ---: |
| MPSC, workers4 | 50 | 4 | 48.632µs | 9.081ms |
| Mutex, workers4 | 50 | 4 | 43.412µs | 9.631ms |

Interaction and other unqualified operators retain the original fenced path.
In the model correctness window, each forward had25 prepared EmbeddingBag
calls and18 conservative fallbacks, including the unchanged empty table. These
are coverage/synchronization counts, not a measured bottleneck attribution.

Candidate native SHA256:
`d8943e56a5b234b2dc7cb0091861087a611437661b0ddd6479d5ca610a7ccaa4`

The SDK profile, source hashes and both qualification receipts are described
in [the overlay documentation](../../patches/eager_mpsc_r34/README.md).
Raw results, resource inventories and launch recipes remain in the private
evidence archive, outside the public repository.
Archive002 has109 payloads,243381 bytes, SHA256
`8dcb37235a2949c7894f2dccb7eec968b6db719e1a13fd1d9ef3f37d6d09f36a`.
The retry archive has58 payloads,164986 bytes, SHA256
`5c40d2fa894cd56a4c7d9711543e967843a66c5b006a9b423d0ea346a039d3b3`.

## Remaining acceptance work

The seven correctness arms, seven performance windows, three reversed key
repeats and two independent instrumented diagnostics have now completed.
Further acceptance for a production optimization requires validating a new
candidate that reduces these costs; the measured candidate did not accelerate
this workload. A full balanced campaign and other workloads remain separate
follow-up work.

Join all callers before cold close; concurrent model/fallback and lifecycle
calls remain outside the contract.

The CAS ingress removes `submit_mu` from that route. It does not remove
`state_mu`, integration counters' mutex, blocking publication ACKs or downstream
locks. Snapshot explicitly reports `global_lock_free=false` and the remaining
core mutex. Any future claim of a globally lock-free path needs a separate
implementation and proof covering those stages.
