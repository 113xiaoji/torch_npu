# Eager Prepare host contract tests

These tests compile the actual portable `EagerPrepareQueueCore.h` as C++17.
They do not load PyTorch, Python, the Ascend SDK, or an NPU. The full-FIFO and
multi-producer tests also use the real `OrderedMpscQueueCore`; the mixed legacy
command fixture simulates the original FIFO's callbacks.

From an initialized C++ compiler environment:

```sh
cmake -S test/cpp/eager_prepare -B test/cpp/eager_prepare/build
cmake --build test/cpp/eager_prepare/build
ctest --test-dir test/cpp/eager_prepare/build --verbose
```

On the validation Windows host, `test\cpp\eager_prepare\run_host_tests.cmd`
initializes the installed Visual Studio Build Tools environment and runs those
commands. It rebuilds from scratch because localized MSVC include diagnostics
can prevent Ninja from detecting header dependencies correctly.

Tests use condition-variable gates, futures, and five-second diagnostic
timeouts. They do not rely on arbitrary sleeps. CTest also imposes a 45-second
process timeout. Compilation enables warnings as errors (`/W4 /WX` or
`-Wall -Wextra -Werror`).

Coverage includes reversed Prepare completion with ordered Execute, interleaved
legacy callbacks, four concurrent producers and 240 original-FIFO commands,
full-FIFO progress, slot and retained-byte ownership through retirement,
rollback acknowledgement, first failure by logical sequence, all three publish
ownership outcomes, cleanup failure, and worker TLS lifetime. The quarantine
tests deliberately retain live worker threads until the test process exits.

The core only measures bytes reported by Job::RetainedBytes. A Job must report
owned inputs before admission and updated executor/workspace ownership after
Prepare. Successful Job::Cancel must explicitly release every accounted owner
and report zero retained bytes, even if other shared_ptr references keep the
Job wrapper alive. The core checks that zero-byte release ACK and quarantines
without returning credits if any bytes remain. Destroying only the core's
shared reference is not proof of resource release.

A post-query top-up never waits for a later ticket: an over-budget job
fails and rolls back its unexecuted resources. This is not a hard cap on
temporary allocations made *inside* an opaque SDK query. Cleanup failure closes
admission and quarantines resources and worker TLS; it cannot establish a new
successful byte reservation.

The integration must return an accurate PublishReceipt, keep SDK calls native,
check SDK-wide poison at each Prepare entry, and call Retire only after a real
device completion ACK. CancelAndWait authorizes rollback of an unexecuted job;
an attempted Execute still requires device completion before Retire. No host
test establishes NPU correctness, SDK thread safety, or performance benefit.

If a publisher can wait for its consumer before returning (for example the
original queue's INIT path), its ownership-copy callback must call
MarkCommitted(ticket) immediately. A successful ownership ACK cannot be revoked
by a later PublishReceipt or exception. Asynchronous publishers may continue to
return a receipt directly. Waiting until the synchronous publisher returns to
publish the ACK would deadlock its consumer.

Snapshot::workers_started counts entries into the actual Worker function.
It is a cumulative birth counter, not the configured worker count, and can be
less than the configuration immediately after construction while OS threads
are still starting.

Snapshot::peak_slots_in_use and peak_retained_bytes record the actual high-water
marks under the reservation lock, including successful Prepare byte top-ups.
They remain available after retirement. These are reservation peaks, not sampled
producer-return values or opaque SDK allocation peaks.
