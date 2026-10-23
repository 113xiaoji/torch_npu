# Measured free-threaded drain-wake-r2 snapshot

This branch preserves the production source used by the measured
`drain-wake-r2` ordinary/free-threaded packages. The upstream base is
`46936d18a18ba219fc5ed310fe2cc6409f94114b` (TorchNPU 2.13.0).
`drain_r2_source_pins.json` records all 31 changed/added production files.
Upstream submodule gitlinks are unchanged; initialize them recursively and
use the repository's existing code generators for generated interfaces.

The snapshot includes the experimental free-threaded build gate, attached
Python thread-state handling, ordered multi-producer queue and lifecycle
handling, and the contiguous-cache concurrency fix. The drain change only
notifies drain waiters on actual drain while Running; non-Running lifecycle
notifications remain. It keeps publication FIFO, drain predicates, error
propagation, and per-request device-wide synchronization.

Default-off sequence options and Inductor compatibility code remain at the
measured source version. They are not independent performance results.
The later Noop wait fast path and prefix-completion experiments are absent.

## Measurement and bounded stability

The clean interleaved comparison used reduced DLRM, batch128, 455153
parameters, Eager per-op submission, an Ordered queue, default stream,
and forward plus device-wide synchronization. Both modes used CPython
3.14.6 and PyTorch 2.13.0+cpu at
`cf30153c4c131c8164ee7798e5022d810682e2cb` on Kunpeng950/openEuler24.03 SP4.
All process threads shared the stated physical-core budget. The original
8-core placement crossed LLCs; this result is not the later placement gain.

| FT, 8 Python threads / 8 physical cores | Cache-fixed baseline | drain-r2 |
|---|---:|---:|
| Throughput, batch/s | 1010.35 | 1213.62 |
| Full-request P99, ms | 15.94763 | 7.65254 |

The ratio of three process medians gives +20.12% throughput and -52.01%
P99. Paired throughput changes were +13.4595%, +20.1183%, +18.8096%.
The complete comparison campaign passed 80 starts and 481800 requests;
the candidate accounted for 56 starts and 391800 requests. There were no
failed or replacement attempts. This is bounded exposure, not a guarantee.

Other cells are retained in the evidence: ordinary four-thread throughput
was -2.64% in the initial campaign, and FT single-thread was -1.30%.
Do not describe this as a universal 20% gain. The simulator does not run
real NPU kernels; real CANN/driver compatibility and device numerical
correctness were not established by this experiment.

## Native regression

On Linux with a C++17 compiler, run from the repository root:

```sh
python3 test/cpp/experimental_free_threaded/run_candidate.py --source-root .
```

This compiles the real queue header and runs correctness, nonempty-drain
wake, metadata lifecycle, and discard lifecycle ten times each. It writes
all compiler/execution records under the ignored test results directory.
The historical run passed both compilations and all 40 executions.
It does not require NPU hardware and is not a performance benchmark.

The contiguous fix additionally passed its isolated seven-case native
cache regression and ThreadSanitizer check in the original experiment;
the full TorchNPU library was not ThreadSanitizer-instrumented.

## Build and wheel identity

Use matching ordinary `cp314` and free-threaded `cp314t` environments.
The measured build used C++17, GCC12.3.1, CMake3.31.12, Release with
the existing O2/g1 flags and C++11 ABI=1, without PGO/LTO. These are
CMake cache options, passed with `-D`:

```text
TORCH_VERSION=2.13.0
GLIBCXX_USE_CXX11_ABI=1
TORCH_NPU_EXPERIMENTAL_FREE_THREADED=ON
TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE=OFF
```

When using `setup.py`, set the environment variable
`TORCH_NPU_EXPERIMENTAL_FREE_THREADED=1`; its environment parser accepts
`0` or `1`, rather than CMake's `ON`/`OFF` notation.

Exact compiler/link recipes and the full performance report were retained
with the experiment. The delivered `2drainr2` wheels wrap the tested
package files without recompiling, exclude bytecode caches, and preserve
the snapshot's backend/CLI entry points. A fresh isolated pip install
matched all 1326 package files per ABI; backend autoload, CLI loading and
1/8-thread DLRM smokes passed (360 requests), including FT GIL-off checks.

| Artifact | SHA256 |
|---|---|
| FT libtorch_npu.so | `e4418b133d7e1504fbcccd1b45a9d1c2b26377ba569f356211a181db999bc426` |
| Ordinary libtorch_npu.so | `f7f8ef8081bf8294845ba6ec383af51cb8054f2e268bf72c293729fe74c98fbd` |
| FT wheel, cp314-cp314t-linux_aarch64 | `acc9f63576e1b6c8bec3412ebe5f2611c11cd8f38a93b4d593117a4da504f0e0` |
| Ordinary wheel, cp314-cp314-linux_aarch64 | `d57bfdf943c5c22c42d1d4fb07b9c0b5b9df885fa81e91deeec8d6d30c5afa79` |

The measured runtime needs the matching ACL host simulator libraries and
NumPy2.4.4. The generic linux_aarch64 wheel tag does not assert manylinux
portability. Neither generated code, copied include trees, binaries,
profiling logs, nor model data are committed on this branch.
