# R26 native protocol tests

These tests compile the actual `DependencyScheduler.h` state engine. Only the event
service is simulated. They establish CPU protocol behavior, not CANN qualification,
kernel numerics, actual allocator safety, device overlap or performance.

From the repository root, Windows MSVC or Linux C++:

```text
python -m unittest discover -s test/eager_parallel -p test_native_core.py -v
cmake -S test/cpp/eager_parallel -B /tmp/r26-core
cmake --build /tmp/r26-core --config Release
ctest --test-dir /tmp/r26-core -C Release --output-on-failure
```

Separate Linux sanitizer builds:

```text
cmake -S test/cpp/eager_parallel -B /tmp/r26-asan -DR26_ENABLE_ASAN=ON
cmake --build /tmp/r26-asan
ctest --test-dir /tmp/r26-asan --output-on-failure
cmake -S test/cpp/eager_parallel -B /tmp/r26-tsan -DR26_ENABLE_TSAN=ON
cmake --build /tmp/r26-tsan
ctest --test-dir /tmp/r26-tsan --output-on-failure
```

The independent native target includes RAW/WAR/WAW, read/read aliases, whole allocation
byte accounting, two side streams, stream-tail fan-in, 24 publication/completion order
permutations, 1000 event-slot generations, stale allocation/ACK rejection, an explicit
entry frontier, exactly-once launch/retirement, reserve/record/wait/query/destroy failure,
late completion after cancel/close, fail-closed owner affinity and inference gating.
Real allocation-failure injection traverses every allocation in admission and checks
strong exception safety; retirement after a successful event destroy does not allocate.
Reentry tests attempt all five mutating public APIs from launch and each of the six
event backend phases, including callbacks that swallow the rejection. Every attempt
fails closed before graph mutation; read-only owner diagnostics remain permitted.
Failure-path OOM tests arm the next allocation failure during an uncertain launch
and record/publication/query/destroy errors. Latching/quarantine never allocate,
independent admission/submission stays blocked, later snapshot allocation failure
does not alter the latch, and the original diagnostic is bounded and sticky.

`feature_pipeline_test.cpp` drives the same scheduler through 26 embedding producers,
two real emulated stream-tail waits, Cat, Gram/BMM, ordered lower-triangle Index and
final Cat. Four fixed random seeds change device progress order while preserving each
stream's FIFO. An independent scalar oracle computes the answer directly from the
original weights, indices and offsets. It covers an empty bag, repeated indices, the
read-only transpose alias and dense input's second consumer. This is CPU numerical and
protocol evidence only, and no host-simulation timing is reported as device performance.

Deliberately unresolved event/launch failures enter process-lifetime quarantine. This
storage remains reachable for sanitizers; hiding a leak by releasing an unconfirmed
allocation would violate the tested protocol. A successful test run does not imply the
vendor SDK was instrumented by either sanitizer.


R31 selective scheduling adds metadata mean-length boundaries (63/64/65,
include-last-offset, ragged, empty and overflow), storage byte-interval aliases,
and unique caller-frontier generations to `immediate_protocol_test.cpp`.
`test_immediate_selective.py` compiles the actual lean caller begin/complete/fail
bodies with deterministic metadata, body, poisoned-completion and failed-sync
faults. Its second test compiles the production Invoke template against CPU Torch
and checks all four results stay alive during recovery, with zero output visits
on the successful caller path. These tests do not establish NPU kernel numerics.

The isolated Linux `run_immediate_linux_validation.py` runner includes
`--case selective_caller` for Release/ASan/TSan. Copy both the Python test module
and `immediate_selective_fixture.cpp.in` into its explicit source tree, alongside
the existing R27 fixtures. Use a fresh entry probe for the updated
`immediate_invoke_lifetime_test.cpp`; preexisting probe binaries are insufficient.
