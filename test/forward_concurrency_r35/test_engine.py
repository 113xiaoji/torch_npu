"""Deterministic, accelerator-free tests of the concurrent caller engine."""

import contextlib
import importlib
import queue
import threading
import time
import unittest
import weakref


try:
    engine = importlib.import_module("benchmarks.forward_concurrency_r35.engine")
except ModuleNotFoundError as exc:
    if exc.name not in {
        "benchmarks.forward_concurrency_r35",
        "benchmarks.forward_concurrency_r35.engine",
    }:
        raise
    engine = None


class Output:
    def __init__(self, worker_index, request_id):
        self.worker_index = worker_index
        self.request_id = request_id


class EngineTests(unittest.TestCase):
    def run_engine(self, **kwargs):
        self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
        return engine.run_closed_loop(**kwargs)

    def start_run(self, **kwargs):
        self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
        results = queue.Queue()

        def run():
            try:
                results.put((True, engine.run_closed_loop(**kwargs)))
            except BaseException as exc:
                results.put((False, exc))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, results

    def finish_run(self, thread, results):
        succeeded, value = results.get(timeout=5)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive(), "engine controller did not return")
        if not succeeded:
            raise value
        return value

    def test_all_callers_enter_forward_before_any_is_released(self):
        for concurrency in (2, 4):
            with self.subTest(concurrency=concurrency):
                entered = set()
                entered_lock = threading.Lock()
                all_entered = threading.Event()
                release = threading.Event()

                @contextlib.contextmanager
                def factory(worker_index):
                    class Session:
                        def forward(self, request_id):
                            if request_id == worker_index:
                                with entered_lock:
                                    entered.add(worker_index)
                                    if len(entered) == concurrency:
                                        all_entered.set()
                                if not release.wait(3):
                                    raise AssertionError("callers were serialized")
                            return Output(worker_index, request_id)

                        def complete(self, output):
                            pass

                    yield Session()

                thread, results = self.start_run(
                    concurrency=concurrency,
                    total_requests=8,
                    warmup_per_worker=0,
                    session_factory=factory,
                    timeout_seconds=4,
                )
                try:
                    self.assertTrue(all_entered.wait(2), "all callers must enter together")
                    self.assertEqual(entered, set(range(concurrency)))
                    self.assertTrue(results.empty(), "response preceded forward completion")
                finally:
                    release.set()
                result = self.finish_run(thread, results)
                self.assertEqual(result.max_forward_overlap, concurrency)
                self.assertGreater(result.forward_overlap_ns, 0)
                self.assertEqual(len({r.thread_ident for r in result.requests}), concurrency)
                self.assertEqual(len({r.native_thread_id for r in result.requests}), concurrency)

    def test_total_and_exact_strided_request_ids_are_independent_of_concurrency(self):
        for concurrency in (1, 2, 4):
            with self.subTest(concurrency=concurrency):
                seen = [[] for _ in range(concurrency)]
                completed = [[] for _ in range(concurrency)]
                references = {}
                cleaned = set()
                guard = threading.Lock()

                @contextlib.contextmanager
                def factory(worker_index):
                    class Session:
                        def forward(self, request_id):
                            seen[worker_index].append(request_id)
                            output = Output(worker_index, request_id)
                            with guard:
                                references[request_id] = weakref.ref(output)
                            return output

                        def complete(self, output):
                            completed[worker_index].append(output.request_id)

                    try:
                        yield Session()
                    finally:
                        with guard:
                            cleaned.add(worker_index)

                result = self.run_engine(
                    concurrency=concurrency,
                    total_requests=13,
                    warmup_per_worker=0,
                    session_factory=factory,
                    timeout_seconds=4,
                )
                self.assertEqual(result.concurrency, concurrency)
                self.assertEqual(result.total_requests, 13)
                self.assertEqual([r.request_id for r in result.requests], list(range(13)))
                self.assertEqual(cleaned, set(range(concurrency)))
                self.assertEqual(len(result.outputs), 13)
                for index in range(concurrency):
                    expected = list(range(index, 13, concurrency))
                    self.assertEqual(seen[index], expected)
                    self.assertEqual(completed[index], expected)
                    worker = result.workers[index]
                    self.assertEqual(worker.worker_index, index)
                    self.assertEqual(worker.completed_requests, len(expected))
                    self.assertGreaterEqual(worker.thread_cpu_ns, 0)
                for record in result.requests:
                    self.assertEqual(record.worker_index, record.request_id % concurrency)
                    self.assertIs(result.outputs[record.request_id], references[record.request_id]())
                    self.assertLessEqual(result.start_ns, record.arrival_ns)
                    self.assertLessEqual(record.arrival_ns, record.forward_return_ns)
                    self.assertLessEqual(record.forward_return_ns, record.response_ns)
                    self.assertLessEqual(record.response_ns, result.end_ns)
                    self.assertLessEqual(record.thread_cpu_start_ns, record.forward_thread_cpu_end_ns)
                    self.assertLessEqual(record.forward_thread_cpu_end_ns, record.thread_cpu_end_ns)
                self.assertEqual(result.end_ns, max(r.response_ns for r in result.requests))
                if concurrency == 1:
                    self.assertEqual(result.max_forward_overlap, 1)
                    self.assertEqual(result.forward_overlap_ns, 0)

    def test_warmup_finishes_before_callback_and_is_not_retained_or_counted(self):
        concurrency = 4
        warmup = 24
        calls = [[] for _ in range(concurrency)]
        completions = [[] for _ in range(concurrency)]
        timing_enabled = threading.Event()
        callback_finished_ns = []
        warmup_refs = []
        guard = threading.Lock()

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    is_warmup = len(calls[worker_index]) < warmup
                    if not is_warmup and not timing_enabled.is_set():
                        raise AssertionError("timed request entered before callback")
                    calls[worker_index].append(request_id)
                    output = Output(worker_index, request_id)
                    if is_warmup:
                        with guard:
                            warmup_refs.append(weakref.ref(output))
                    return output

                def complete(self, output):
                    completions[worker_index].append(output.request_id)

            yield Session()

        def before_timing():
            for index in range(concurrency):
                self.assertEqual(calls[index], list(range(warmup)))
                self.assertEqual(completions[index], list(range(warmup)))
            callback_finished_ns.append(time.perf_counter_ns())
            timing_enabled.set()

        result = self.run_engine(
            concurrency=concurrency,
            total_requests=12,
            warmup_per_worker=warmup,
            session_factory=factory,
            before_timing=before_timing,
            timeout_seconds=4,
        )
        self.assertEqual(result.warmup_per_worker, warmup)
        self.assertEqual(len(result.requests), 12)
        self.assertEqual(len(result.outputs), 12)
        self.assertLessEqual(callback_finished_ns[0], result.start_ns)
        self.assertTrue(all(reference() is None for reference in warmup_refs))
        for index in range(concurrency):
            expected = list(range(warmup)) + list(range(index, 12, concurrency))
            self.assertEqual(calls[index], expected)
            self.assertEqual(completions[index], expected)

    def test_response_and_wall_end_follow_request_completion(self):
        entered_completion = threading.Event()
        release_completion = threading.Event()
        completion_return_ns = []

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    return Output(worker_index, request_id)

                def complete(self, output):
                    entered_completion.set()
                    if not release_completion.wait(3):
                        raise AssertionError("test did not release request completion")
                    completion_return_ns.append(time.perf_counter_ns())

            yield Session()

        thread, results = self.start_run(
            concurrency=1,
            total_requests=1,
            warmup_per_worker=0,
            session_factory=factory,
            timeout_seconds=4,
        )
        try:
            self.assertTrue(entered_completion.wait(2))
            self.assertTrue(results.empty())
        finally:
            release_completion.set()
        result = self.finish_run(thread, results)
        record = result.requests[0]
        self.assertLessEqual(record.forward_return_ns, completion_return_ns[0])
        self.assertLessEqual(completion_return_ns[0], record.response_ns)
        self.assertEqual(result.end_ns, record.response_ns)

    def test_cleanup_is_outside_measured_wall_interval(self):
        cleanup_started_ns = []

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    return Output(worker_index, request_id)

                def complete(self, output):
                    pass

            try:
                yield Session()
            finally:
                cleanup_started_ns.append(time.perf_counter_ns())

        result = self.run_engine(
            concurrency=1, total_requests=1, warmup_per_worker=0,
            session_factory=factory, timeout_seconds=4,
        )
        self.assertLessEqual(result.end_ns, cleanup_started_ns[0])

    def test_forward_failure_never_completes_that_request_and_cleans_sessions(self):
        cleaned = set()
        completed = []
        guard = threading.Lock()
        both_in_forward = threading.Barrier(2)
        failure = ValueError("forward failed")

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    both_in_forward.wait(timeout=2)
                    if worker_index == 0:
                        raise failure
                    return Output(worker_index, request_id)

                def complete(self, output):
                    with guard:
                        completed.append(output.worker_index)

            try:
                yield Session()
            finally:
                with guard:
                    cleaned.add(worker_index)

        self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
        with self.assertRaises(engine.ClosedLoopError) as caught:
            self.run_engine(
                concurrency=2, total_requests=2, warmup_per_worker=0,
                session_factory=factory, timeout_seconds=4,
            )
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(caught.exception.worker_index, 0)
        self.assertEqual(caught.exception.phase, "forward")
        self.assertNotIn(0, completed)
        self.assertEqual(cleaned, {0, 1})

    def test_warmup_failure_wakes_other_callers_waiting_at_start_gate(self):
        cleaned = set()
        guard = threading.Lock()
        both_in_warmup = threading.Barrier(2)
        failure = ValueError("warmup failed")

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    both_in_warmup.wait(timeout=2)
                    if worker_index == 0:
                        raise failure
                    return Output(worker_index, request_id)

                def complete(self, output):
                    pass

            try:
                yield Session()
            finally:
                with guard:
                    cleaned.add(worker_index)

        self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
        with self.assertRaises(engine.ClosedLoopError) as caught:
            self.run_engine(
                concurrency=2, total_requests=2, warmup_per_worker=1,
                session_factory=factory, timeout_seconds=4,
            )
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(caught.exception.phase, "warmup_forward")
        self.assertEqual(cleaned, {0, 1})

    def test_callback_failure_cancels_waiting_workers_and_preserves_cause(self):
        cleaned = set()
        guard = threading.Lock()
        failure = ValueError("snapshot failed")

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    raise AssertionError("callback failure must prevent timed forward")

                def complete(self, output):
                    raise AssertionError("no request should complete")

            try:
                yield Session()
            finally:
                with guard:
                    cleaned.add(worker_index)

        def before_timing():
            raise failure

        self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
        with self.assertRaises(engine.ClosedLoopError) as caught:
            self.run_engine(
                concurrency=2, total_requests=2, warmup_per_worker=0,
                session_factory=factory, before_timing=before_timing,
                timeout_seconds=4,
            )
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIsNone(caught.exception.worker_index)
        self.assertEqual(caught.exception.phase, "before_timing")
        self.assertEqual(cleaned, {0, 1})

    def test_timeout_returns_while_a_forward_is_blocked(self):
        entered = threading.Event()
        release = threading.Event()
        cleaned = threading.Event()

        @contextlib.contextmanager
        def factory(worker_index):
            class Session:
                def forward(self, request_id):
                    entered.set()
                    release.wait()
                    return Output(worker_index, request_id)

                def complete(self, output):
                    pass

            try:
                yield Session()
            finally:
                cleaned.set()

        thread, results = self.start_run(
            concurrency=1, total_requests=1, warmup_per_worker=0,
            session_factory=factory, timeout_seconds=0.5,
        )
        try:
            self.assertTrue(entered.wait(2))
            succeeded, failure = results.get(timeout=3)
            self.assertFalse(succeeded)
            self.assertIsInstance(failure, engine.ClosedLoopTimeout)
            self.assertEqual(failure.unfinished_worker_indices, (0,))
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertFalse(cleaned.is_set())
        finally:
            release.set()
        self.assertTrue(cleaned.wait(2))

    def test_session_entry_and_completion_failures_are_reported(self):
        for phase in ("session", "complete", "cleanup"):
            with self.subTest(phase=phase):
                failure = ValueError(phase)

                @contextlib.contextmanager
                def factory(worker_index):
                    class Session:
                        def forward(self, request_id):
                            return Output(worker_index, request_id)

                        def complete(self, output):
                            if phase == "complete":
                                raise failure

                    if phase == "session":
                        raise failure
                    yield Session()
                    if phase == "cleanup":
                        raise failure

                self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
                with self.assertRaises(engine.ClosedLoopError) as caught:
                    self.run_engine(
                        concurrency=1, total_requests=1, warmup_per_worker=0,
                        session_factory=factory, timeout_seconds=4,
                    )
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(caught.exception.phase, phase)

    def test_invalid_arguments_fail_before_sessions_start(self):
        def factory(worker_index):
            raise AssertionError("invalid arguments started a worker")

        defaults = dict(
            concurrency=1, total_requests=1, warmup_per_worker=0,
            session_factory=factory, timeout_seconds=4,
        )
        cases = (
            {"concurrency": 0}, {"concurrency": True}, {"concurrency": 1.5},
            {"total_requests": 0}, {"total_requests": -1},
            {"warmup_per_worker": -1}, {"warmup_per_worker": False},
            {"timeout_seconds": 0}, {"timeout_seconds": float("inf")},
            {"timeout_seconds": float("nan")}, {"session_factory": None},
            {"before_timing": 42},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                self.assertIsNotNone(engine, "the closed-loop engine is not implemented")
                with self.assertRaises((TypeError, ValueError)):
                    self.run_engine(**(defaults | invalid))


if __name__ == "__main__":
    unittest.main()
