# CPU-only native Prepare simulation

This CPython extension includes the production `EagerPrepareQueueCore.h`
directly. It neither copies/reimplements that core nor links PyTorch, CANN, or
an NPU library. Native workers own copied `int64` metadata and execute synthetic
integer/floating-point work. A single Python consumer executes the real CPU
Torch operator **after** `next()` and acknowledges completion with `retire()`.
Every result explicitly labels `real_npu=false`, `simulated_prepare=true`.

```python
from tools.eager_prepare_cpu import load_native
native = load_native()
session = native.Session(workers=4, capacity=8, byte_limit=1024 * 1024,
                         work_iterations=10000, trace=False)
```

The producer calls `session.submit(seq, tuple_of_int64)`. Submission releases
the GIL before taking native locks or waiting for a retirement credit. The
separate consumer repeatedly does:

```python
seq = session.next()             # waits detached; native Execute is a marker
if seq is not None:
    result = original_cpu_op(...)  # owned by the caller, not the native bridge
    session.retire(seq)           # actual CPU-op completion acknowledgement
```

Sequences must be nonnegative and strictly increasing throughout one persistent
Session, including across warmup/sample rounds. `next()` rejects a second
consumer or any attempt to get another ticket before the previous CPU operation
is retired. `retire()` checks the consumer's actual OS thread ID. It is not a
device fence or proof that an NPU operation completed.

`close()` stops admission and wakes empty consumers; existing FIFO entries
remain available, and `next()` returns `None` after they drain. After the consumer
joins and all tickets retire, `shutdown()` joins the native pool and returns a
snapshot. Calling shutdown with outstanding tickets raises and requires the
caller to finish them or abort. `abort(reason)` cancels native metadata, wakes
blocked producers/consumers, and joins the pool. It never owns or destroys Torch
tensors. Save and rethrow the original CPU operator exception rather than a
secondary producer admission error. Aborted shutdown is idempotent.

The byte guard and peak counters apply to **admitted native metadata
reservations**, not the full Torch model, tensor storage, Python objects,
pending caller argument copies, or the independently bounded trace buffer.
`retired` counts resource retirement (including cancelled metadata), while
`cpu_completed` counts only explicit successful consumer acknowledgements.
`execute_markers` must never be described as successfully executed Torch ops.

`prepare_inline(metadata, iterations, trace=False)` calls exactly the same C++
work function without a pool. Its checksum equals that metadata's worker
checksum at the same iteration count; `snapshot()['checksum_xor']` aggregates
all successful Prepare checksums. Use one calibrated iteration count for both
paths and all workers. Zero iterations still hashes the metadata; it omits the
synthetic busy-work loop.

Cold diagnostics use `trace=True` with a bounded `trace_limit` (default 4096).
`trace_records()` returns completion-order rows containing `seq`, actual OS
`tid`, `start_ns`, `end_ns`, `thread_cpu_ns`, raw `thread_cycles`, and `checksum`.
`now_ns()` shares the native steady-clock timebase. Hot mode takes no per-op
clocks and stores no trace rows; it retains counters and observed worker IDs.

On Windows, thread CPU ns are the actual GetThreadTimes counter. Short tasks
can legitimately return zero because accounting is coarse. QueryThreadCycleTime
supplies independent raw cycles; these are **not converted to nanoseconds**.
Linux uses CLOCK_THREAD_CPUTIME_ID/gettid; raw cycles are `None`. Calibration
should use measured inline wall time and retain the actual CPU readings. The
Linux path is supplied but has not been executed in this Windows validation.

Build/test with the installed compiler and Python; no package installation:

```text
python tools/eager_prepare_cpu/build_native.py
python -m unittest discover -s test/eager_prepare_cpu_native -v
```

The Windows builder uses the installed VS 2022 Build Tools (`VSDEVCMD` can
override its location), C++17, `/O2 /W4 /WX`. It writes versioned binaries,
compiler logs, generated build identity, and manifests only under `.cache/`.
`core_info()['core_sha256']` is embedded at compile time. The loader rejects a
cached binary if that hash differs from the current source, and detects source
changes during compilation. Restart Python after changing native sources.

Diagnostic-only `test_gate_seq`/`fail_prepare_seq` constructor arguments and
`wait_gate_entered`, `release_test_gate`, `wait_prepared` create deterministic
test schedules without sleeping. They are disabled in ordinary sessions.

These experiments test queue contracts and potential overlap with real CPU
operators. They establish no NPU/SDK performance, SDK thread safety, real device
completion, or Python 3.14 free-threading benefit.
