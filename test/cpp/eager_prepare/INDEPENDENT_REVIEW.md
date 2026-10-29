# Independent portable-core review — 2026-10-29

Reviewer: independent local source and contract review. No production source,
test source, Git index, browser, SDK, or NPU operation was changed or executed.
Only this review document was added; the existing host build directory was
rebuilt and its tests rerun.

## Reviewed identities

- `torch_npu/csrc/core/npu/EagerPrepareQueueCore.h`:
  `fa69c4fc1e200d2eef1dfeb4c1bd439323b6915cd4b81da8547300f9b1251d37`
- `test/cpp/eager_prepare/eager_prepare_core_test.cpp`:
  `9a9c898c1bb2b6df28ae69c15c2dbf8d2d76b49c47e656c186fa673816159e66`
- Included `OrderedMpscQueueCore.h`:
  `49334fe4afb07fd4a8c504e00e0f664c5e2a2c23500c465e54295148437d007a`

The SDK integration is being edited independently and is not approved by this
portable-core review. Its references below describe the observed ownership
shape rather than freezing or approving that integration.

## Finding

**P2 — Retiring the core's shared reference does not prove that owned arguments
have been destroyed before their credits are returned.**

`ReleaseOwner` (core lines 298–311) moves and resets `ticket->job`, then erases
the owner and decrements retained-byte accounting. However, `Submit` accepts
`shared_ptr<Job>` and the public contract does not forbid another strong
reference. Therefore `released.reset()` need not run the Job destructor. The
existing `OwnerDestructionPrecedesCreditRelease` check explicitly resets the
caller's reference before retirement and covers only that special case.

A fresh standalone C++17/MSVC reproduction using the unchanged header and a
Job which owns 64 bytes, has no-op Prepare/Execute/Cancel, and releases its
owned allocation in its destructor produced:

```text
Core configuration: slots=1, retained_bytes=64, workers=1
after_first_retire slots=0 reserved=0 actual_alive=1
after_second_admit reserved=64 actual_owned_bytes=128
```

The first ticket was consumed and successfully retired; the caller retained
its original `shared_ptr<Job>`. The second job was admitted normally. The
reproduction is in the local temporary directory
`C:/Users/zhouyongzhe/AppData/Local/Temp/r24-independent-credit-iw31zen1`
(`credit.cpp`, `run.cmd`, `compile.log`), compiled with `/std:c++17 /EHsc /W4 /WX`,
exit 0. No repository source was patched for this experiment.

This is a resource-accounting/ownership contract gap, not evidence of an
early device free or UAF. Device retirement still occurs before the core
returns its own reservation. It nevertheless prevents a general claim that
the actual Job or all retained arguments are destroyed before credit release.
It is also distinct from the documented temporary opaque-SDK allocation
exception: these are completed, persistently owned Jobs after retirement.

The inspected SDK integration has both a `Retirement::job` strong reference
and a producer-local `job`; `OwnedQueueJob::Cancel` delegates to payload
Cancel rather than destroying the payload. Those references must be included
when resolving this obligation. Either make release of the accounted owned
resources an explicit, verified ACK before returning credits, or enforce an
exclusive ownership transfer and make all remaining references non-owning.
If the intended contract is only outstanding-ticket reservation accounting,
state that limitation explicitly and withdraw the broader destruction/bounded
actual-retention claim. Add the externally retained-owner case to the chosen
contract's regression tests. This finding was sent to the root agent before
this document was written; no implementation change was made by the reviewer.

## Other examined boundaries

No additional P1/P2 was identified in the reviewed portable core under its
documented native-Job and externally serialized lifecycle contract.

- Publication: `submit_mu` serializes admission/publication order. Work becomes
  visible before the original-FIFO publisher can block. Consume waits for both
  Prepare completion and the publication receipt. NotAccepted sets cancellation
  and releases only after Prepare/cleanup ACK; Committed plus an error retains
  the ticket; an exception without a receipt stays Indeterminate and quarantines.
  Correct FIFO ownership receipts remain the integration's obligation.
- Prepare/cancel: cancellation is a request while Prepare runs. Worker cleanup
  follows return/throw from Prepare, and `prepare_ack` is published afterwards.
  `cleanup_active` excludes concurrent cleanup/Consume. `CancelAndWait` waits
  through an attempted Execute and leaves cleanup/credits to actual retirement.
- Completion and credits: early Retire rejects missing Prepare/publication/
  consume completion; failed retirement and cleanup failure retain the owner
  and worker TLS. A successful device completion ACK cannot be proved by this
  portable API; the caller supplies that assertion to Retire.
- Locks and hooks: Prepare, Execute, Cancel, publisher, and Job-reference release
  occur without `state_mu`; Submit reattaches only after releasing `submit_mu`.
  CV hooks unlock before `AfterWait` and recheck predicates after relocking.
  The tests exercise reentrant Snapshot and Submit hooks. They do not instantiate
  a real interpreter/GIL or prove that an integration's hooks are balanced.
- TLS/lifecycle: quarantine deliberately leaks the preallocated thread-vector
  owner; each worker retains State, its owners, and its thread-local SDK state.
  Normal Shutdown waits for outstanding owners before stopping/joining workers.
  Close is not a cancellation or device-fence ACK. A stuck opaque Prepare may
  still require the external controller's process termination; Shutdown has no
  timeout and does not establish such an ACK.
- First failure: recorded failure is selected by the lowest admitted sequence,
  while an already accepted earlier prefix can still execute. A globally poisoned
  SDK must be rejected by the native Job's Prepare entry; the core does not
  itself implement an SDK poison check.
- Bytes: initial retained bytes are reserved before work publication; successful
  post-query growth is reserved without waiting for later tickets. Overflow
  rolls back the unexecuted job; cleanup failure quarantines. This is neither a
  hard temporary SDK allocation cap nor, as the finding shows, an enforced bound
  on arbitrary externally retained shared ownership.

## Fresh verification

Environment: Windows x64, Visual Studio Build Tools 2022/MSVC 19.44, C++17.

```text
cmd /d /c test\cpp\eager_prepare\run_host_tests.cmd
```

Fresh clean compile and link passed with warnings as errors. The executable
reported **26/26 host contract checks**; CTest **1/1**, exit 0 (0.02 s test time).
The final build's `Testing/Temporary/LastTest.log` was read, not only the prior
author's evidence file.

```text
ctest --test-dir test/cpp/eager_prepare/build --repeat until-fail:30 --output-on-failure
```

Fresh **30/30 process runs** passed, exit 0, 0.50 s total. Each process includes
the four-producer/four-worker, 240-original-FIFO-command test. Core and test-source
hashes were rechecked after these runs and matched the identities above.

These are host contract tests. They establish no SDK executor thread safety,
real NPU completion behavior, actual GIL behavior, numerical correctness,
device error recovery, or performance improvement. No sanitizer/TSan run was
performed. SDK integration and device qualification remain separate gates.


## Targeted P2 follow-up — resource-release ACK (2026-10-29)

**The P2 above is closed for the following reviewed source identities.** This
appendix preserves the initial finding and its original evidence unchanged.

- Core: `11ec62bc817ce2406291a37f4236072c07a965b61d8a9d2850698c1610da04dd`
- Portable test: `3b45306983e37f265325f7ab0a86d633abd6487bf21321474eedb9dfec8aa75c`
- `EagerPrepareOwnedJob.h`:
  `dd6d82317ef1d2b4805b344077105245f8e1500622b0cbf8e89604b6710a8478`
- `test/host_regions/eager_prepare_integration/owned_job_test.cpp`:
  `11b8e1d464d2162108d6ac668523d772df1a0a4541d76882616dbc62f490c299`

The core now requires a successful explicit cleanup ACK: after native Cancel,
`CleanupFailed()` must be false **and** `RetainedBytes()` must equal zero.
Otherwise cleanup does not complete, the ticket/domain is quarantined, the
owner and all its original credits remain, and admission is closed. Thus the
old no-op Cancel Job cannot turn an internal shared-reference release into a
false resource-release ACK.

The actual OwnedQueueJob adapter now captures MayHaveSubmitted, performs native
Cancel and checks both context/payload cleanup state. Only successful cleanup
resets its payload and context before reporting zero retained bytes. A strong
reference held by a FIFO Retirement closure or the producer therefore retains
an empty wrapper, rather than budgeted native arguments. On failure the owned
payload/context remain available for quarantine. Cached MayHaveSubmitted and
cleanup state preserve queries needed after successful payload destruction.

Fresh independent validation:

1. Clean build via `cmd /d /c test\cpp\eager_prepare\run_host_tests.cmd`:
   **30/30 portable checks**, CTest 1/1, exit 0. This includes the new retained
   external-owner negative and successful explicit-release positive cases.
   The earlier 30-repeat evidence belongs to the original reviewed version;
   this follow-up does not relabel it as a new-version stress run.
2. A new standalone heap-owning 64-byte reproduction against the unchanged
   current header compiled with MSVC `/std:c++17 /EHsc /W4 /WX` and returned 0:

   ```text
   bad_release quarantine=1 slots=1 reserved=64 second_admission_rejected=1
   good_release wrapper_alive=1 owns_storage=0 slots=0 reserved=0 second_admission_completed=1
   ```

   Local reproduction directory:
   `C:/Users/zhouyongzhe/AppData/Local/Temp/r24-independent-release-ack-sgkwpyn9`.
   Both branches allocate an actual `new unsigned char[64]`; the successful
   branch explicitly frees that allocation while keeping its shared wrapper.
3. Fresh actual-adapter fixture:

   ```text
   python -I -B test/host_regions/test_eager_prepare_integration.py NativeOwnedJobTests.test_actual_owned_job_bridge_without_sdk_or_python_worker_objects -v
   ```

   **1 test passed**, 1.675 s, exit 0. It compiles the real OwnedQueueJob header
   with MSVC, retains the external `held` wrapper, confirms the payload's alive
   count drops to zero during Retire, and confirms slots/bytes are returned only
   afterwards. It also checks cached submission state and failure propagation.

An initial read during the integration author's concurrent RED-to-GREEN window
observed fixture `Context::Failed override` before the corresponding base method
was installed and reproduced MSVC C3668. That transient mismatch was reported;
the current exact header/test identities above contain the callback and were
freshly compiled and executed successfully. The mismatched intermediate pair
is not the basis of this closure.

No additional important finding was identified in this targeted resource-release
fix. This closes the portable ownership/credit P2, not the whole still-changing
SDK integration. The original no-NPU/no-SDK-test/no-performance-claim limitations
continue to apply. No production or test source was edited by this reviewer.
