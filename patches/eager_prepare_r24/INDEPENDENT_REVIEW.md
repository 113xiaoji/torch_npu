# Independent native integration review

Review date: 2026-10-29. Scope: native R24 integration, its pinned op-plugin overlay, public bridge, allocator admission, lifecycle/finalization and host fixtures. This review did not edit implementation files, install software, access a device/browser, stage or commit changes.

## Current disposition

**Changes required: one open P1 build blocker.** The INIT publication deadlock found in the first frozen candidate is closed by the independently tested v2 early-copy-ACK change. No other P1/P2 finding was confirmed in the reviewed lifetime, fallback, TLS, allocator, retirement or finalization paths.

Reviewed v2 pins: `INTEGRATION_SOURCE_PINS-v2.json`, SHA256 `ccdb6c271d09860b8dc126d2fcb8c135fe76d3eb9de690cf9959f512994e3bc8`. All 35 declared inputs matched bytes and SHA256 before the v2 review. Native `EagerPrepare.cpp`: `ab6f183f4b6357bf8608999296a9c1b8d7a5aca0a8ceb138559e126e4b8058c5`; core: `704f99c5cfd8faeb8640dd52965f8e5b3d85a1a9069a26ab68d957d0b6fdb23d`.

The original pins document SHA256 `044c50bfa66030c69022ab4ddabbfc2813fc202c26034b37e31be71eb4ff24ec` and all 34 original inputs also matched at initial review. Its native CPP/core hashes were `6627be903b51b3de720ec18afb6126029df5a10873725d669967c44e1bbf0297` / `11ec62bc817ce2406291a37f4236072c07a965b61d8a9d2850698c1610da04dd`. That snapshot is superseded by v2.

## P1 — preserve the synchronous macro's return-valued expression (OPEN)

Location: `patches/eager_prepare_r24/op-plugin/op_plugin/utils/op_api_common.h:906-912`.

`EXEC_NPU_CMD_SYNC` now expands to `do { ... } while(false)`. The original macro at lines 718 onward is an immediately invoked lambda returning the converted ACL parameter owner. Existing pinned op-plugin callers use the macro as an initializer, including:

- `third_party/op-plugin/op_plugin/utils/AdvancedIndex.cpp:82`: `auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnNonzeroV2, self, out);`
- `third_party/op-plugin/op_plugin/ops/opapi/NonzeroKernelNpuOpApi.cpp:33`.
- `third_party/op-plugin/op_plugin/ops/opapi/MaskedSelectKernelNpuOpApi.cpp:52`.
- The Unique, Unique2, UniqueDim and UniqueConsecutive kernels likewise assign and then query the returned parameter owner.

A statement macro cannot appear on the right side of these initializers. This breaks compilation even when the experimental runtime is disabled; the runtime condition cannot hide a syntax error. This is a concrete source/compiler defect, distinct from the separately documented absence of a full target build.

Independent bounded compiler reproduction: extracted the exact v2 `EXEC_NPU_CMD_SYNC` definition into a temporary C++17 source, supplied native admission stubs and `#define HDB_ORIGINAL_EXEC_NPU_CMD_SYNC(api, ...) 17`, and compiled the real assignment form above with MSVC `/std:c++17 /EHsc /W4 /WX /c`. Result: exit 2, `C2059` at `do`, `C2143`, and `C3536` for the uninitialized result. The same fixture using the original-expression alias compiled with exit 0. Temporary files were removed by their owning controller.

Required correction: keep the fence/admission operation inside an expression that returns the original result (and preserves its ownership/move semantics). Add a host compile/run regression which actually assigns the result, exercises off/on paths, and retains the converted owner until after its caller-side metadata use. Do not close this finding on the existing asynchronous dispatch fixture alone.

## P1 — INIT FIFO waits for an unpublished ticket (CLOSED in v2)

Original locations: `EagerPrepare.cpp:330-346`, `EagerPrepareQueueCore.h:135,365-368`, and unchanged `NPUStream.cpp:852-855`.

`InitRepo` stores `RepoStatus::INIT` (`NPUQueue.cpp:1474`), and default-stream creation calls it (`NPUStream.cpp:278`). On its first publication, `enCurrentNPUStream` enqueues the deferred callback and synchronously calls `MakeSureQueueEmpty` before returning. In the original candidate, the consumer called `core->Consume` and waited for `ticket->published`; `SubmitDetached` set that flag only after the publisher returned. The publisher could not return until that consumer finished. Setup synchronization alone does not establish RUN: `emptyAllNPUStream` drains at `NPUStream.cpp:661-665` without the INIT-to-RUN transition.

Initial independent reproduction compiled both actual core headers. Its publisher executed `fifo.Enqueue(copy)` and then actual `fifo.WaitDrained(hooks)` before returning the receipt, matching the INIT order. The RUN control returned normally and reached `EXECUTE_REACHED` / `CLOSED`; the INIT case printed `INIT_BRANCH_WAIT_DRAINED`, `PREPARE_ACK`, `CONSUMER_ENTERED`, then hit its three-second external timeout and was killed/reaped without executing.

The v2 production exact-parameter callback now calls `MarkCommitted` before the original enqueue can enter its synchronous drain (`EagerPrepare.cpp:307-316`, marker binding at 356). The core publishes and notifies at the ownership-copy ACK (`EagerPrepareQueueCore.h:242-255`) and refuses to downgrade that committed ownership when a late publisher return disagrees (`383-399`). It does not acquire the submit mutex, call SDK/Python, or acknowledge a foreign parameter.

Independent v2 verification:

- Fresh `python -I -B -m unittest discover -s test/host_regions -p test_eager_prepare_integration.py -v`: **13/13 PASS, 0 skips, 14.803 seconds**. This includes the actual extracted marker/callback paired with the actual ordered FIFO, synchronous consumption before return, duplicate/foreign ACK checks, and late throw/NotAccepted/error ownership cases.
- Recompiled the original two-core reproduction against the v2 core. With `MarkCommitted(ticket)` in the FIFO copy callback, its actual `WaitDrained` INIT path exited 0 and reached `EXECUTE_REACHED` / `CLOSED`. Removing only the early ACK still reproduced the three-second timeout and kill/reap. Compiler exit was 0 with `/std:c++17 /EHsc /W4 /WX`.

Compact reproduction shape (actual core types; a zero-byte native Job supplies no-op Prepare/Cancel and observable Execute):

```cpp
EagerPrepareQueueCore eager({2, 64, 1});
OrderedMpscQueueCore fifo(2);
fifo.Start();
EagerPrepareQueueCore::TicketPtr slot;
NoopWaitHooks hooks;
std::thread consumer([&] {
    fifo.ConsumeOne([&](uint32_t) {
        eager.Consume(slot);
        eager.Retire(slot);
    }, hooks);
});
eager.Submit(job, [&](const auto& ticket) {
    fifo.Enqueue([&](uint32_t) {
        slot = ticket;
        eager.MarkCommitted(ticket); // Removing only this line reproduces INIT hang.
    }, hooks);
    fifo.WaitDrained(hooks);         // Original enCurrentNPUStream INIT ordering.
    return EagerPrepareQueueCore::PublishReceipt{
        EagerPrepareQueueCore::PublishState::Committed, {}};
});
consumer.join();
eager.Shutdown();
```

## Review evidence and limits

The earlier integration fixture also passed independently at 12/12, 0 skips, 13.095 seconds before the ACK change. The author-reported full host suites are not relabeled as this reviewer's independent runs.

Source review covered exact-copy ACK wiring into the original FIFO; release-lane closure ownership; Prepare/Execute/Cancel sequencing; payload reset before credit; retained native Storage and workspace lifetime; caller/worker context and cache-state capture; explicit R24 allocator eligibility; unsupported-argument fallback fences; default-off routing; cold-close once semantics; interpreter detach/lock order; and fail-stop entry before runtime SDK finalization locks. Add/Relu/Mm remain unqualified. No wider model/alias/DAG or SDK-parallelism certification follows from this review.

No complete PTA/CANN translation unit was compiled, no NPU execution was performed, and no device/model performance was measured. The compiler reproduction above checks the actual extracted macro's C++ expression contract; it does not claim a target build. CPU trace simulation and later Hostsim execution remain separate evidence domains and cannot establish native SDK qualification, actual device retirement, numerical device correctness or NPU speedup.

## V3 closure addendum — synchronous expression P1 CLOSED

This addendum supersedes the v2 disposition and the sync finding's OPEN status above. **Both confirmed P1 findings are now closed in the reviewed source; no confirmed P1/P2 finding remains open.** The full current-host-suite receipt is tracked separately below and is not inferred from the old 64-test run.

Verified `INTEGRATION_SOURCE_PINS-v3.json` SHA256 `c34a2510b5178758ef0029ad6dd13f812d4f029ebc0a3467f865bb73d8d76ea4`; all 36 declared inputs match their byte and SHA256 pins. The changed op-plugin header is `306fe9ed42c4acaf921ee26f14a6ba399b64ce97f58577fd1ff525cae2376c95`. Native CPP `ab6f183f4b6357bf8608999296a9c1b8d7a5aca0a8ceb138559e126e4b8058c5` and core `704f99c5cfd8faeb8640dd52965f8e5b3d85a1a9069a26ab68d957d0b6fdb23d` are unchanged from the independently tested INIT fix.

At `op_api_common.h:906-912`, the wrapper is now an immediately invoked `[&]() -> decltype(auto)` expression, performs the existing enabled-only admission/fence, and `return`s the original synchronous macro result. No catch/retry was added. Reversing exactly the three changed macro lines independently reproduced the v2 header SHA256 `746e7f4cf3e3c60d99327eaaed924c0b778a61ec7fb931de05a2020f59d2340d`.

Independent verification for this closure:

- Repeated the original exact-wrapper/real-assignment compiler reproduction. The reconstructed v2 statement wrapper still fails with exit 2 and `C2059` / `C2143` / `C3536`; the current v3 expression compiles with exit 0 and its executable exits 0. Compiler flags remain `/std:c++17 /EHsc /W4 /WX`.
- Fresh `python -I -B test/host_regions/test_eager_prepare_integration.py NativeOwnedJobTests.test_actual_sync_macro_preserves_converted_parameter_expression -v`: **1/1 PASS, 0 skips, 1.534 seconds**. The fixture extracts the actual current macro, assigns its move-only converted-parameter result to `auto npuAclParams`, uses `Get<1>()` while its owner is alive, verifies off has no fence and on fences once, counts one argument evaluation, and verifies SDK/fence failures do not replay the original operation. A fence failure leaves argument evaluation and original-call counts unchanged.

At the time this closure was appended, v3 honestly recorded the fresh full current suite as pending; the v2 64/64 receipt is historical, not a current v3 full-suite pass. Full PTA/CANN compilation and device execution remain unperformed and are not implied by closing this C++ expression defect.

## Final current-source host receipt verified

The root subsequently completed the fresh full current suite. This reviewer read `.cache/root-final-host/VERIFICATION.json` (SHA256 `7475266131cd74a0687a0886adafdf79b9c35040d6eca9dba19bb5e9a09e1f16`) and checked its bound stderr file directly. The receipt binds the exact v3 manifest SHA256 `c34a2510b5178758ef0029ad6dd13f812d4f029ebc0a3467f865bb73d8d76ea4`, records exit 0 and 65 tests/0 skips, and states the host-only scope. The log `.cache/root-final-host/HOST_TESTS.stderr` is 12,535 bytes with SHA256 `d1ab1ef34bdfcecbe53890342ce6c42393a487a831df17ba8c260e583ee4e7b2`; it contains 65 `... ok` result lines, no failure/error/skip lines, and the summary **65 tests in 42.189 seconds, OK**.

The reviewer independently rechecked all 36 current source/dependency byte/hash pins after reading that receipt: no mismatch. The v3 manifest's earlier pending field remains unchanged as historical metadata; this later external receipt establishes completion of its full current-source rerun. This was a receipt review of the root's full run, separate from this reviewer's independent focused runs described above.

**Final review disposition: both confirmed P1 defects closed; no confirmed P1/P2 finding remains open in the reviewed v3 source.** No full PTA/CANN translation-unit build, NPU/device run, new SDK qualification or NPU/model speedup claim is established by these host results.
