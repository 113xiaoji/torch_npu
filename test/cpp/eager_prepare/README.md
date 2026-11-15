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

Snapshot::peak_slots_in_use and peak_retained_bytes record reservations observed
under the state lock, including successful Prepare byte top-ups. They remain
available after retirement. In MPSC mode a producer can reserve atomic credits
before reaching that lock; a transient pending peak can therefore be missed.
The atomic slot/byte caps still apply to those pending reservations. These
counters do not measure opaque SDK allocation peaks.

## Optional MPSC ingress (R34)

The same binary runs both configurations. Existing three-field aggregates keep
the mutex route; the fourth field selects the experimental ingress:

```cpp
Core::Config baseline{8, 536870912, 4};
Core::Config candidate{8, 536870912, 4, Core::IngressMode::LockFreeMpsc};
```

The test executable runs the original contract suite with both configurations.
The MPSC route preallocates fixed descriptors and uses atomic descriptor claims,
slot/byte reservations, and a CAS producer stack. A dedicated native publisher
detaches and reverses batches, assigns sequences on dequeue, dispatches Prepare,
and then publishes to the original FIFO. A producer paused before its enqueue
CAS exposes no hole. Descriptors are reused only after the publisher's full
completion ACK and caller cleanup; the consumer saves its successor before ACK.
Producers never dereference existing head nodes, so recycled head addresses do
not require hazard pointers for this push/exchange algorithm.

Close atomically closes a combined admission-bit/active-producer-count gate.
A producer holding a permit from before Close may still link afterwards; it
belongs to the accepted prefix. Close does not drain that prefix. Shutdown waits
for active permits, pending reservations, publisher callbacks, worker jobs, and
real retirements. Production callers must stop and join all submitters before
draining the original FIFO. A first failure cancels unpublished pending jobs;
cleanup failure retains descriptors, credits, owners, and live thread TLS until
process exit.

This is an experiment in a lock-free ingress primitive. The blocking Submit
wrapper still takes state_mu for its completion wait. The publisher, worker
queue, readiness, consumption, accounting transitions, original PTA FIFO,
allocators, and SDK are not globally lock-free. Snapshot reports
ingress_atomic_lock_free, retained_core_mutex, and submit_mutex_used explicitly.
Both compile-time requirements and runtime is_lock_free checks cover the
primitive's atomics. Payload/callback construction occurs before the primitive;
ticket/map/deque allocation still occurs on the publisher.

Submit waits for the entire publisher function to finish, preserving lifetime
and late-error behavior. The existing early ownership-copy MarkCommitted ACK
still breaks the synchronous INIT consumer/publication dependency. The native
publisher requires owned callback captures and captured caller TLS/context.
In C++17 its empty-queue wait has a one-millisecond fallback to handle a producer
paused after its enqueue CAS and before notification. The extra native thread,
handoff, waits, and notifications are part of candidate performance.

The host suite additionally covers preempted producer permits at Close,
unlinked-producer progress, cross-batch descriptor reuse, pending slot/byte
credits, Shutdown waiting for pending work, pending failure cleanup/quarantine,
and publisher TLS retirement/quarantine. No host result proves NPU numerical
correctness, SDK parallel-query qualification, or a performance gain.
