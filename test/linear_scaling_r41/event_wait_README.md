# R41 event host wait candidate

Baseline source: `7d5507972f6468c63e62ef3221dfb73c439f1913`.

The candidate changes only the host pending-record wait in
`NPUEvent::synchronize()`. Set `TORCH_NPU_EVENT_HOST_WAIT=spin` or `condition`
before importing/initializing torch_npu. Unset defaults to `spin`. Other values,
including an empty string, are rejected during event-manager initialization.
The process configuration is cached once; changing the environment later does
not change the effective mode.

The additive C symbol `torch_npu_event_host_wait_mode()` returns a static
`const char*` containing `spin`, `condition`, or `invalid`, and is `noexcept`.
It can be queried outside the measured region to qualify the selected mode.
For example, load the candidate `libtorch_npu.so` with `ctypes.CDLL`, set this
function's `restype` to `ctypes.c_char_p`, and require the expected value before
starting the benchmark. The getter does not prove device initialization or
correct candidate library selection; those still need independent qualification.

In condition mode, one manager-wide condition variable waits with the existing
pending-count mutex released. Its predicate remains absence from the existing
`ska::flat_hash_map<aclrtEvent, int>`, checked while holding that mutex. A final
decrement and `ClearUnrecordedCount()` notify all waiters. A notification is not
a completion latch: another pending record or a re-record before mutex
reacquisition keeps the waiter blocked. Notification before waiting is safe
because the predicate is checked before sleeping.

The device SDK synchronization, event flags, query behavior, block/IPC polling,
queue admission, record/dequeue ordering, and SDK error handling are unchanged.
This is a mechanism candidate; no DLRM throughput or linear-scaling benefit is
established by the CPU-only tests.

## Existing error and lifetime semantics

- `EventTask::LaunchRecordTask` increases the pending count before queue
  publication. Generic enqueue exceptions/rejections have no automatic
  rollback in this baseline; a stranded count can block either wait mode
  until existing decrement/clear cleanup occurs. This patch adds no timeout
  or new cancellation/error channel.
- `RecordEventFunc` and `ValueWriteFunc` in `framework/OpParamMaker.cpp`
  decrement after the SDK attempt even if that attempt returns an error.
  Therefore an absent count establishes the existing host-dequeue predicate,
  not successful SDK record or device completion. `aclrtSynchronizeEvent`
  remains responsible for its existing device wait/error behavior.
- Consumer error paths in `NPUQueue.cpp` call `ClearUnrecordedCount()`.
  Clearing retains its global absent-means-recorded semantics and now wakes
  condition waiters, which continue to the existing SDK synchronization.
- Private prefix-event destruction already rolls back an unpublished count
  with `DecreaseUnrecordedCount`; that final decrement now also notifies.
- All callers must finish before the singleton is destroyed. This patch does
  not introduce shutdown cancellation or permit destroying a condition
  variable with active waiters. Tests join every native waiter before exit.
- The global condition variable can wake callers for unrelated events; each
  rechecks its own map predicate. Contention/notification cost requires actual
  NPU qualification and the original 1/2/4-caller DLRM A/B workload.

## CPU-native verification

Run from the repository root on Windows with MSVC Build Tools or Linux with a
C++17 compiler:

```text
python test/linear_scaling_r41/event_wait_native.py
```

The runner builds the real `NPUEventManager.cpp` and header in a temporary
directory. Only external ACL, logging, c10 pool, exception, option, and hash-map
dependencies are stubbed; the map substitute is `std::unordered_map`. Thus the
tests exercise the production manager's actual counter, mutex, configuration,
condition wait, and notifications, without a Torch extension or NPU. They do
not validate ACL or c10 container implementation, Python bindings, or the full
`NPUEvent.cpp` translation unit.

Cases cover unset/default and explicit modes, invalid/empty input, cached
configuration, notify-before-wait, mutex availability while waiting, multiple
pending records, independent events, re-record/reuse, multiple waiters, global
queue-error clear, explicit rollback, invalid decrement, stranded unpublished
counts, and joined-before-destruction lifetime. A 120 ms pending-record window
also checks that condition mode consumes less than 40 ms of waiter thread CPU;
its timing is mechanism evidence, not an application performance benchmark.

TDD evidence: the baseline production manager compiled successfully against the
test dependencies, then the executable failed with
`FAIL: NPUEventManager is missing native WaitForEventRecorded` before the
production changes were applied.

## Header and rebuild scope

`NPUEvent` layout is unchanged. `NPUEventManager` gains an out-of-line public
method and a `std::condition_variable` appended after all existing members.
Existing member order and offsets are retained, but manager size/destruction
change; this is not a claim of binary layout compatibility. Rebuild the manager,
event translation unit, and every translation unit depending on the changed
manager header using the candidate's normal dependency-aware build. Do not
reuse stale objects merely because existing member offsets were retained.

The C getter is an additive exported symbol using `TORCH_NPU_API`. There are no
new source files to add to the build configuration and no new runtime library
dependency beyond the project's existing standard C++ threading facilities.
