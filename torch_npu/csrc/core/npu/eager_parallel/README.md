# R26 native dependency/completion core

`DependencyScheduler.h` is a C++17, Torch-independent protocol engine. It has one
constructing caller, two producer stream IDs and one consumer stream ID. It is
not a qualified production NPU adapter. No host Prepare workers are created.

The caller supplies a monotonically increasing logical sequence and complete
whole-allocation read/write effects. Duplicate aliases are collapsed, with write
dominating read. The engine records RAW, WAR and WAW edges; shared reads have no
mutual edge. Independent producers alternate between two internal streams. A
consumer waits only for its actual predecessors, combining each foreign stream's
dependencies into the latest *submitted* event on that stream.

Every allocation carries allocator identity, generation, actual allocation bytes,
and an owner that pins its real address. Untracked operands default to
`EntryReadiness::NeedsFence`. The adapter must either prove device completion,
declare a newly allocated output, or create a `NodeKind::EntryFence` on the caller
stream with write effects for the imported allocations. Its no-op launch receipt
followed by event record establishes the domain-entry frontier. This imports only
the already-existing predecessor; later independent embedding calls do not wait
for an unrelated current consumer-stream tail.

`submit(id, launch)` calls the native launch function at most once. It returns
`PublicationPending` without enqueueing any wait when a predecessor record is
not known to be submitted. The adapter must make bounded publication progress
outside scheduler/global locks. A pending result is not permission to fall back
and execute the operation again. `AlreadySubmitted` also covers a retired ID and
means it must not be replayed, not that a cancelled ID necessarily launched.

Backend and launch callbacks may call `snapshot()` or `inspect()` on the owner,
but cannot reenter `admit`, `submit`, `poll`, `close` or `cancel`. A scoped callback
guard rejects every such mutation before changing the graph. The rejection also
marks that external call as invalid if the callback catches the exception. The
outer operation then preserves the error and quarantines any possibly submitted
work or uncertain reservation; it cannot retain a reference across a reentrant
registration's table replacement. Callback state resets on both return and throw.

Publication and completion are independent. Publication authorizes device wait;
only positive device completion authorizes retiring submitted resources. Event
handles additionally remain held while admitted dependent nodes may use them.
`poll()` and `close()` each make one bounded query pass and never sleep or wait
for device completion. `close()` cancels unsubmitted work and rejects future
admission. A failed close can still have active/quarantined nodes; consult the
snapshot instead of treating the failure return as a drain acknowledgment.

Record/query errors preserve the first error and retain unknown work. A later
positive publication and completion can retire it safely. Ambiguous launch/wait
errors without a recorded completion fence remain quarantined. Event retirement
is attempted once: a throwing destroy is never retried. Destroying a domain
with active nodes moves its native owners and backend into reachable process
lifetime quarantine; it does not silently release them or synchronize the device.
Error state and quarantine are latched without allocation before diagnostic text
is handled. The first reason is copied into a fixed 1024-byte buffer and later
errors cannot replace it. `snapshot()` may allocate its returned string and fail
under memory exhaustion, but that cannot reopen admission or clear quarantine.
The `cancel` reason is a non-owning `std::string_view` used only during that call.

The foreign-caller path only sets an atomic poison flag and throws. It never
touches the owner-managed DAG. The owner stops admission and drains after seeing
the flag. A production entry layer must hold a second caller before any metadata
read/mutation until the owner acknowledges drain. This core does not authorize
the second caller to catch the exception and immediately run the baseline path.

Adapter guarantees still required before any real route is enabled:

| Guarantee | Required evidence |
| --- | --- |
| Actual allocation pin | Allocator generation and DataPtr lifetime, plus intercepted metadata/storage mutations; StorageImpl ownership alone is insufficient. |
| Complete effects | All inputs, four EmbeddingBag outputs, auxiliaries, workspace, copies, aliases and native descriptor owners accounted; no uncharged retained owner. |
| Domain entry | Every prior writer/reader outside the domain is complete or correctly represented by the imported caller-stream frontier. |
| Stream identity and FIFO | Streams belong to the correct device/context and preserve enqueue order; numeric IDs in this core carry no SDK proof. |
| Event reservation | All event resources reserved before launch; a reserve exception guarantees no retained backend event or device side effect. |
| Record ACK | `Published` means the record has actually reached the stream, not merely been copied into a host queue. ACKs are keyed by domain, slot and generation. |
| Submission receipt | `RejectedBeforeSubmit` is definite; partial or unknown submissions return `Uncertain` or throw and cannot replay. |
| Bounded service | Every backend method returns promptly, is non-reentrant, and never depends on progress from this blocked caller; publication deadlines are enforced by the adapter. |
| Completion and cleanup | Positive completion is reliable for that exact generation; retained CPU borrows are finished, native descriptors remain owned until safe, and event destruction has the stated contract. |
| Qualification and fallback | Inference, route, shape/layout, TLS, SDK artifacts, stream restrictions, unsupported operations and host observations are qualified at the entry boundary before host reads. |

Node count, live allocation/native bytes and allocation identity history are
bounded independently. Exhaustion rejects admission before submission; it does
not silently drop effects. Identity history intentionally persists until the
domain closes so a stale generation cannot be readmitted after resource reuse.
Normal retirement erases node payloads and reuses event slots with an incremented
generation. Admission prepares the entire bounded graph transaction before no-throw
commit, so allocation failure preserves existing state and all identities. Event slot
storage is reserved at construction so successful retirement needs no allocation.
This conservative registration implementation copies bounded bookkeeping tables and
makes no production scheduling overhead claim. The fake-backend tests do not establish
any of the SDK guarantees.
