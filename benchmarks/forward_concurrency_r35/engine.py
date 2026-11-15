"""Model-independent closed-loop callers for complete forward requests.

The session factory runs inside each caller thread. It owns that thread's
inference context, stream and completion event. ``complete(output)`` must wait
for this request's device work before returning. No accelerator API is used by
this module, and no engine lock surrounds ``forward`` or ``complete``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Protocol


class ForwardSession(Protocol):
    def forward(self, request_id: int) -> Any: ...

    def complete(self, output: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: int
    worker_index: int
    thread_ident: int
    native_thread_id: int
    arrival_ns: int
    forward_return_ns: int
    response_ns: int
    thread_cpu_start_ns: int
    forward_thread_cpu_end_ns: int
    thread_cpu_end_ns: int


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    worker_index: int
    thread_ident: int
    native_thread_id: int
    thread_cpu_ns: int
    completed_requests: int


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    concurrency: int
    total_requests: int
    warmup_per_worker: int
    start_ns: int
    end_ns: int
    requests: tuple[RequestRecord, ...]
    outputs: tuple[Any, ...]
    workers: tuple[WorkerRecord, ...]
    forward_overlap_ns: int
    max_forward_overlap: int


class ClosedLoopError(RuntimeError):
    """A caller or controller failed; ``__cause__`` preserves its exception."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        worker_index: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.worker_index = worker_index
        self.cause = cause
        self.unfinished_worker_indices: tuple[int, ...] = ()


class ClosedLoopTimeout(ClosedLoopError, TimeoutError):
    """The run expired; blocked daemon callers may still require process exit."""


def _forward_overlap(records: tuple[RequestRecord, ...]) -> tuple[int, int]:
    """Return union time with >=2 forward spans, and maximum span overlap.

    These host dispatch intervals do not establish simultaneous Python bytecode
    execution or simultaneous accelerator kernels. Half-open intervals prevent
    adjacent forwards from counting as overlap.
    """
    edges = []
    for record in records:
        if record.forward_return_ns > record.arrival_ns:
            edges.append((record.arrival_ns, 1))
            edges.append((record.forward_return_ns, -1))
    active = maximum = overlap_ns = 0
    previous_ns = 0
    for timestamp_ns, delta in sorted(edges):
        if active >= 2:
            overlap_ns += timestamp_ns - previous_ns
        active += delta
        maximum = max(maximum, active)
        previous_ns = timestamp_ns
    return overlap_ns, maximum


def run_closed_loop(
    *,
    concurrency: int,
    total_requests: int,
    warmup_per_worker: int,
    session_factory: Callable[[int], ContextManager[ForwardSession]],
    before_timing: Callable[[], None] | None = None,
    timeout_seconds: float = 60.0,
) -> ClosedLoopResult:
    """Run fixed-total requests with one outstanding request per caller.

    Each caller warms up with IDs ``0..warmup_per_worker-1``, waits while the
    controller calls ``before_timing``, then runs IDs
    ``range(worker_index, total_requests, concurrency)``. Outputs are retained
    in request-ID order for validation after timing. Wall time begins just
    before the common start gate opens and ends at the last request completion;
    session startup, warmup, callback, cleanup and output validation are excluded.

    The timeout covers startup, warmup, timed requests and cleanup. A synchronous
    controller callback is checked when it returns. Python cannot stop a thread
    stuck in native code: failures wake waiting callers, join for at most 250 ms,
    and report unfinished daemon callers. Use a process timeout for hard hangs.
    """
    for name, value, minimum in (
        ("concurrency", concurrency, 1),
        ("total_requests", total_requests, 1),
        ("warmup_per_worker", warmup_per_worker, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a finite positive number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    if not callable(session_factory):
        raise TypeError("session_factory must be callable")
    if before_timing is not None and not callable(before_timing):
        raise TypeError("before_timing must be callable or None")

    deadline = time.monotonic() + timeout_seconds
    state_changed = threading.Condition()
    start_gate = threading.Event()
    cancelled = threading.Event()
    ready_count = done_count = 0
    failures: list[ClosedLoopError] = []
    # Each caller is the sole writer of its slots. Publication and result reads
    # are ordered by the condition used for caller completion.
    caller_records: list[list[RequestRecord]] = [[] for _ in range(concurrency)]
    caller_outputs: list[list[tuple[int, Any]]] = [[] for _ in range(concurrency)]
    worker_records: list[WorkerRecord | None] = [None] * concurrency
    threads: list[threading.Thread] = []

    def worker(worker_index: int) -> None:
        nonlocal ready_count, done_count
        phase = "session"
        try:
            with session_factory(worker_index) as session:
                for warmup_id in range(warmup_per_worker):
                    if cancelled.is_set():
                        return
                    phase = "warmup_forward"
                    output = session.forward(warmup_id)
                    phase = "warmup_complete"
                    session.complete(output)
                    del output
                phase = "ready"
                with state_changed:
                    ready_count += 1
                    state_changed.notify_all()
                start_gate.wait()
                if cancelled.is_set():
                    return

                thread_ident = threading.get_ident()
                native_thread_id = threading.get_native_id()
                worker_cpu_start_ns = time.thread_time_ns()
                records = caller_records[worker_index]
                outputs = caller_outputs[worker_index]
                for request_id in range(worker_index, total_requests, concurrency):
                    if cancelled.is_set():
                        break
                    phase = "forward"
                    arrival_ns = time.perf_counter_ns()
                    thread_cpu_start_ns = time.thread_time_ns()
                    output = session.forward(request_id)
                    forward_return_ns = time.perf_counter_ns()
                    forward_thread_cpu_end_ns = time.thread_time_ns()
                    phase = "complete"
                    # Even if another caller fails, complete an already issued
                    # request before releasing its output or session resources.
                    session.complete(output)
                    thread_cpu_end_ns = time.thread_time_ns()
                    response_ns = time.perf_counter_ns()
                    records.append(RequestRecord(
                        request_id=request_id,
                        worker_index=worker_index,
                        thread_ident=thread_ident,
                        native_thread_id=native_thread_id,
                        arrival_ns=arrival_ns,
                        forward_return_ns=forward_return_ns,
                        response_ns=response_ns,
                        thread_cpu_start_ns=thread_cpu_start_ns,
                        forward_thread_cpu_end_ns=forward_thread_cpu_end_ns,
                        thread_cpu_end_ns=thread_cpu_end_ns,
                    ))
                    outputs.append((request_id, output))
                worker_cpu_end_ns = time.thread_time_ns()
                worker_records[worker_index] = WorkerRecord(
                    worker_index=worker_index,
                    thread_ident=thread_ident,
                    native_thread_id=native_thread_id,
                    thread_cpu_ns=worker_cpu_end_ns - worker_cpu_start_ns,
                    completed_requests=len(records),
                )
                phase = "cleanup"
        except BaseException as exc:
            error = ClosedLoopError(
                f"caller {worker_index} failed during {phase}: {exc}",
                worker_index=worker_index, phase=phase, cause=exc,
            )
            with state_changed:
                failures.append(error)
                cancelled.set()
                start_gate.set()
                state_changed.notify_all()
        finally:
            with state_changed:
                done_count += 1
                state_changed.notify_all()

    def wait_for(predicate: Callable[[], bool], phase: str) -> None:
        with state_changed:
            while not predicate():
                if failures:
                    raise failures[0]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClosedLoopTimeout("closed-loop run timed out", phase=phase)
                state_changed.wait(timeout=remaining)
            if failures:
                raise failures[0]
            if time.monotonic() >= deadline:
                raise ClosedLoopTimeout("closed-loop run timed out", phase=phase)

    try:
        for index in range(concurrency):
            thread = threading.Thread(
                target=worker, args=(index,), name=f"forward-caller-{index}", daemon=True,
            )
            thread.start()
            threads.append(thread)
        wait_for(lambda: ready_count == concurrency, "ready")
        if before_timing is not None:
            try:
                before_timing()
            except BaseException as exc:
                raise ClosedLoopError(
                    f"controller failed during before_timing: {exc}",
                    phase="before_timing", cause=exc,
                ) from exc
        if time.monotonic() >= deadline:
            raise ClosedLoopTimeout("closed-loop run timed out", phase="before_timing")
        start_ns = time.perf_counter_ns()
        start_gate.set()
        wait_for(lambda: done_count == concurrency, "run")
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise ClosedLoopTimeout("caller did not exit", phase="cleanup")
    except BaseException as exc:
        cancelled.set()
        start_gate.set()
        abort_deadline = time.monotonic() + 0.25
        for thread in threads:
            thread.join(timeout=max(0.0, abort_deadline - time.monotonic()))
        if isinstance(exc, ClosedLoopError):
            exc.unfinished_worker_indices = tuple(
                index for index, thread in enumerate(threads) if thread.is_alive()
            )
            if exc.cause is not None:
                raise exc from exc.cause
        raise

    records = tuple(sorted(
        (record for per_caller in caller_records for record in per_caller),
        key=lambda record: record.request_id,
    ))
    if len(records) != total_requests or any(record is None for record in worker_records):
        raise ClosedLoopError("callers did not complete every request", phase="result")
    outputs = tuple(output for _, output in sorted(
        (item for per_caller in caller_outputs for item in per_caller),
        key=lambda item: item[0],
    ))
    overlap_ns, maximum_overlap = _forward_overlap(records)
    return ClosedLoopResult(
        concurrency=concurrency,
        total_requests=total_requests,
        warmup_per_worker=warmup_per_worker,
        start_ns=start_ns,
        end_ns=max(record.response_ns for record in records),
        requests=records,
        outputs=outputs,
        workers=tuple(record for record in worker_records if record is not None),
        forward_overlap_ns=overlap_ns,
        max_forward_overlap=maximum_overlap,
    )
