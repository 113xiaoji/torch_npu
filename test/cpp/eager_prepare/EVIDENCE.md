# Host verification evidence — 2026-10-29

Environment: Windows x64, Visual Studio Build Tools 2022, MSVC
19.44.35225.0, C++17, Ninja/CMake. No new dependencies were installed.

## Red / green

The initial 18 tests were written before core behavior. The interface-only
scaffold compiled and all 18 failed at runtime (`Eager Prepare core not
implemented`, or the expected byte-overflow assertion). Full local output is
retained in `red.log` (ignored build evidence).

After implementation, those 18 passed. Additional lifecycle tests found a real
retirement bug: `owner destruction precedes credit release` failed because
credits were released before the Job destructor ran. The full failing output
is `retirement_red.log`: 21 passed, 1 failed. The fix keeps the owner counted
while destroying the Job outside locks, and releases credits afterwards.

Initial verification command:

```text
cmd /c test\cpp\eager_prepare\run_host_tests.cmd
```

Result: compilation passed with `/W4 /WX`; all 26 host contract checks passed;
CTest 1/1 passed, exit 0. Complete verbose compiler and test output: `green.log`.
The runner forces a fresh compile to avoid the local Chinese MSVC
`showIncludes`/Ninja dependency-discovery issue.

Concurrency repeat:

```text
ctest --test-dir test/cpp/eager_prepare/build --repeat until-fail:30 --output-on-failure
```

Result: 30/30 process runs passed, exit 0, 0.49 seconds. Full output:
`stress.log`. Each process includes four producers, four Prepare workers,
and 240 operations through the actual OrderedMpscQueueCore.

## Initial review identity

- Core SHA-256: `FA69C4FC1E200D2EEF1DFEB4C1BD439323B6915CD4B81DA8547300F9B1251D37`
- Test source SHA-256: `9A9C898C1BB2B6DF28AE69C15C2DBF8D2D76B49C47E656C186FA673816159E66`

These hashes identify the 26-check version examined by INDEPENDENT_REVIEW.md.

## Independent review P2 correction

The independent review's external-shared-owner finding was reproduced with a
real 64-byte heap owner retained by the caller after Retire. A new regression
requires the core to retain its 64-byte credit and quarantine when Job::Cancel
leaves that resource alive. Before the fix, this check failed exactly at the
quarantine/slot/byte assertion. `external_owner_red.log` records 27 checks
passing and two failing: this regression and the newly requested actual
worker-birth counter (whose interface scaffold returned zero).

The core now accepts cleanup only when both CleanupFailed is false and
RetainedBytes is zero after Cancel. Otherwise it quarantines, retains its
owners and credits, and refuses new admission. The Job contract explicitly
requires releasing budgeted resources rather than relying on wrapper
destruction. A companion test deliberately retains the shared wrapper after
successful explicit resource release; actual live bytes fall to zero before
the next 64-byte job is admitted. Execute-completion tests still require the
separate device-retirement assertion before any cleanup or credit release.

The native integration's OwnedQueueJob now resets its payload/context after
successful cancellation to supply this zero-byte ACK even while its FIFO
Retirement closure retains the wrapper. Its own native-payload regression and
build remain integration-owner verification, not a result claimed here.

Snapshot::workers_started now increments at actual Worker entry. Three gated
Prepare calls prove three workers actually entered; configuration alone is
never reported as an observed birth count.

Fresh command: `cmd /c test\cpp\eager_prepare\run_host_tests.cmd`.
Result: **29/29 host checks**, warnings-as-errors clean rebuild, CTest 1/1,
exit 0. Complete output: `external_owner_green.log`.

Fresh repeat command:
`ctest --test-dir test/cpp/eager_prepare/build --repeat until-fail:30 --output-on-failure`.
Result: **30/30 process runs**, exit 0, 0.52 seconds. Complete output:
`external_owner_stress.log`.

Verified identities immediately after the P2 and birth-counter correction:

- Core SHA-256: `822D9ED7E6F1A378065129329941F13AB838652BB5575C155BAEE02A5214F95E`
- Test source SHA-256: `1BD67E70994C9E2A857180765B4D80E5419EEAE34093716CCE5B974255947D58`

## Actual reservation peak counters

The integration requested true reservation peaks rather than values sampled
when Submit returns. Snapshot now reports peak_slots_in_use and
peak_retained_bytes, updated under the reservation lock at initial admission
and every successful Prepare top-up. A new regression verifies two jobs grow
from 8 bytes each to 24 and 32 bytes, producing a 56-byte peak that survives
retirement. Its zero-valued interface scaffold failed first:
`peak_counters_red.log` (29 pass, 1 fail).

Fresh clean build using the same runner: **30/30 host checks**, CTest 1/1,
exit 0; complete output `peak_counters_green.log`. Fresh same 30-process repeat:
**30/30**, exit 0, 0.51 seconds; complete output `peak_counters_stress.log`.

Verified source identities before the synchronous-publisher correction:

- Core SHA-256: `11EC62BC817CE2406291A37F4236072C07A965B61D8A9D2850698C1610DA04DD`
- Test source SHA-256: `3B45306983E37F265325F7AB0A86D633ABD6487BF21321474EEDB9DFEC8AA75C`

## Synchronous publisher / early commit P1 correction

The native INIT path can wait for its original FIFO consumer before PublishFn
returns. The previous publication barrier waited for that return, producing a
cycle: publisher waits for consumer, consumer waits for published. A regression
using the actual OrderedMpscQueueCore, its copy callback, and a condition-variable
consumer-completed gate failed at the five-second deadlock diagnostic deadline.
Two additional checks showed that a late exception or NotAccepted receipt could
overwrite the earlier ownership ACK. `early_commit_red.log` records 30 pass,
3 fail, 5.15 seconds, exit 8.

The new `bool MarkCommitted(const TicketPtr&) noexcept` publishes that irreversible
ownership ACK immediately. It only takes state_mu and notifies waiters; it never
takes submit_mu or calls SDK/runtime hooks. A later contradictory receipt or
exception remains a committed failure; it cannot cancel/replay that ticket or
reclassify its established owner as indeterminate. Its consumer can finish and
retire before the publisher returns.

Fresh runner result: **33/33 host checks**, warnings-as-errors clean compile,
CTest 1/1, exit 0, 0.02 seconds (`early_commit_green.log`). Fresh 30-process
repeat: **30/30**, exit 0, 0.56 seconds (`early_commit_stress.log`).

Current verified identities:

- Core SHA-256: `704F99C5CFD8FAEB8640DD52965F8E5B3D85A1A9069A26AB68D957D0B6FDB23D`
- Test source SHA-256: `40F5E0D0FFA2EC9FDD9BE69E508719DF12781A2C00E0C940C508D7DB1FE1A7D5`

## Scope and residual obligations

This is host contract evidence, not NPU/SDK proof or performance evidence.
The integration must prove original-FIFO ownership ACK, real completion fences,
SDK cleanup behavior, and its own Prepare-time poison check. The core allows
already accepted earlier jobs to Prepare after a later failure; globally
poisoned SDK state must be rejected by Job::Prepare itself. The native
integration owner confirmed this contract.

Post-query byte overflow never waits for later tickets. It rolls back the
unexecuted job; opaque SDK temporary allocations remain outside the core's
observable reservation bound. If cleanup or ownership is uncertain, Shutdown
fails closed and the actual Prepare threads and their TLS survive until process
termination. Quarantine retention storage is allocated at construction, so the
destructor does not allocate it during failure handling.
