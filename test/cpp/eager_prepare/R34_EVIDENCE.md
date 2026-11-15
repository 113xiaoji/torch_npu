# R34 optional ingress host evidence — 2026-11-15

Environment: Windows x64; MSVC 19.44.35225.0, C++17, Ninja/CMake,
`/W4 /WX`. No SDK, PyTorch, or NPU execution was involved.

Tests preceded the new behavior. With the mutex implementation, the first new
runtime regression failed at `publisher != caller`; pending requests also
failed the accepted-prefix/Close requirement. The existing 33 checks passed in
both test configurations, for 68 successful invocations and two failures.
After implementation, 70/70 invocations passed. A second RED iteration added
the atomic admission-gate requirement: 73 passed and its missing-gate assertion
failed. Implementing the combined closed bit and producer count made 74 pass.

Further verification covers pending cleanup success/failure, actual Shutdown
waiting on pending tickets, and publisher TLS destruction after successful
retirement versus retention after quarantine. The final local source currently
runs 78 successful invocations: the original suite executes with both routes;
six MPSC-specific cases skip their bodies in the mutex iteration.

Commands:

```text
cmd /d /c test\cpp\eager_prepare\run_host_tests.cmd
ctest --test-dir test/cpp/eager_prepare/build --repeat until-fail:30 --output-on-failure
```

Observed result: warning-as-error compilation passed; CTest 1/1 passed;
78/78 contract invocations passed. All 30 repeat processes passed. Each process
includes the real ordered FIFO producer/consumer stress case for both routes.
Local ignored evidence is in `build-green-final.log`, `build-red-gate.log`, and
`build-stress-r34.log`. Root owns independent A3 compilation and real SDK runs;
those results are not asserted here.

The primitive uses fixed descriptor storage and no allocation while claiming,
reserving, or linking. CAS publication determines queue order, not permit or
credit acquisition. Batch reversal preserves that order. Close admits no new
producer permits; preexisting permits may finish linking. The owner must join
submitters before its external FIFO drain. The preserved state mutex and all
downstream locks mean this implementation does not reach global lock freedom.

The candidate intentionally waits for full publisher completion and includes an
additional native publisher thread. A one-millisecond timed idle/full fallback
prevents lost notifications without taking the state mutex before enqueue CAS.
Reservation peaks observed under the state lock can miss transient pending
peaks; actual atomic credit caps remain enforced. Opaque SDK temporary memory is
outside the reported retained-byte cap, as in the baseline.
