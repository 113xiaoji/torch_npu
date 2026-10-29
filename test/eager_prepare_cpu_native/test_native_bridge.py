"""Real CPython bridge contract tests; every worker executes native code only."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import threading
import unittest
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
try:
    native = importlib.import_module("tools.eager_prepare_cpu").load_native()
except (ImportError, FileNotFoundError):
    native = None


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        if native is None and self._testMethodName != "test_bridge_available":
            self.skipTest("bridge missing: availability test records RED")

    def test_bridge_available(self):
        self.assertIsNotNone(native, "CPU native bridge is missing")

    def test_core_identity_and_explicit_simulation_label(self):
        info = native.core_info()
        source = ROOT / "torch_npu/csrc/core/npu/EagerPrepareQueueCore.h"
        self.assertEqual(info["core_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertFalse(info["real_npu"])
        self.assertTrue(info["simulated_prepare"])

    def test_loader_rejects_stale_binary_core_identity(self):
        builder = importlib.import_module("tools.eager_prepare_cpu.build_native")
        with tempfile.TemporaryDirectory() as temporary:
            changed_header = Path(temporary) / "changed_core.h"
            changed_header.write_text("// deliberately different test source\n", encoding="utf-8")
            with patch.object(builder, "CORE", changed_header):
                with self.assertRaisesRegex(RuntimeError, "compiled core SHA"):
                    builder.load_native()

    def test_serial_parallel_checksum_and_zero_work(self):
        for iterations in (0, 100, 1000):
            metadata = [(1, 2, -3, 17), (), (2**63 - 1, -2**63)]
            expected = 0
            session = native.Session(3, 4, 4096, iterations, trace=True)
            for seq, values in enumerate(metadata):
                expected ^= native.prepare_inline(values, iterations, trace=True)["checksum"]
                session.submit(seq, values)
            for seq in range(len(metadata)):
                self.assertEqual(session.next(), seq)
                session.retire(seq)
            session.close()
            self.assertIsNone(session.next())
            snapshot = session.shutdown()
            self.assertEqual(snapshot["checksum_xor"], expected)
            self.assertEqual(snapshot["cpu_completed"], len(metadata))
            self.assertEqual(snapshot["slots_in_use"], 0)
            self.assertTrue(all(row["thread_cpu_ns"] >= 0 for row in session.trace_records()))

    def test_reversed_ready_still_consumes_in_order(self):
        session = native.Session(2, 2, 4096, 100, trace=True, test_gate_seq=0)
        session.submit(0, (2, 3))
        session.wait_gate_entered()
        session.submit(1, (5, 7))
        session.wait_prepared(1)
        self.assertEqual([row["seq"] for row in session.trace_records()], [1])
        session.release_test_gate()
        self.assertEqual(session.next(), 0)
        session.retire(0)
        self.assertEqual(session.next(), 1)
        session.retire(1)
        session.shutdown()

    def test_k1_backpressure_until_real_retire_ack(self):
        session = native.Session(2, 1, 4096, 100)
        session.submit(0, (1,))
        self.assertEqual(session.next(), 0)
        started, submitted = threading.Event(), threading.Event()
        errors = []
        def producer():
            started.set()
            try:
                session.submit(1, (2,))
                submitted.set()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=producer)
        thread.start()
        self.assertTrue(started.wait(5))
        self.assertFalse(submitted.is_set())
        session.retire(0)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(submitted.is_set())
        self.assertEqual(session.next(), 1)
        session.retire(1)
        session.shutdown()

    def test_no_next_can_cross_unretired_cpu_operation(self):
        session = native.Session(2, 2, 4096, 0)
        session.submit(0, (1,))
        session.submit(1, (2,))
        self.assertEqual(session.next(), 0)
        with self.assertRaisesRegex(RuntimeError, "retire"):
            session.next()
        session.retire(0)
        self.assertEqual(session.next(), 1)
        session.retire(1)
        session.shutdown()

    def test_close_wakes_empty_consumer(self):
        session = native.Session(1, 1, 4096, 0)
        started = threading.Event()
        result = []
        def consumer():
            started.set()
            result.append(session.next())
        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertTrue(started.wait(5))
        session.close()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        session.shutdown()

    def test_cpu_consumer_failure_aborts_backpressured_producer(self):
        session = native.Session(2, 1, 4096, 100)
        session.submit(0, (1,))
        self.assertEqual(session.next(), 0)
        started = threading.Event()
        producer_errors = []
        def producer():
            started.set()
            try:
                session.submit(1, (2,))
            except BaseException as exc:
                producer_errors.append(str(exc))
        thread = threading.Thread(target=producer)
        thread.start()
        self.assertTrue(started.wait(5))
        original_cpu_error = ValueError("actual CPU operator failure")
        session.abort(str(original_cpu_error))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(producer_errors)
        self.assertIn(str(original_cpu_error), producer_errors[0])
        with self.assertRaisesRegex(RuntimeError, "actual CPU operator failure"):
            session.next()
        snapshot = session.snapshot()
        self.assertEqual(snapshot["cpu_completed"], 0)
        self.assertEqual(snapshot["slots_in_use"], 0)
        self.assertEqual(snapshot["live_workers"], 0)

    def test_prepare_failure_wakes_consumer_and_abort_cleans_pool(self):
        session = native.Session(1, 2, 4096, 100, fail_prepare_seq=0)
        session.submit(0, (1,))
        with self.assertRaisesRegex(RuntimeError, "injected Prepare failure"):
            session.next()
        session.abort("Prepare failed")
        self.assertEqual(session.snapshot()["live_workers"], 0)
        self.assertEqual(session.snapshot()["cpu_completed"], 0)

    def test_abort_releases_gated_prepare_and_waiting_consumer(self):
        session = native.Session(1, 1, 4096, 100, test_gate_seq=0)
        session.submit(0, (1,))
        session.wait_gate_entered()
        entered = threading.Event()
        errors = []
        def consumer():
            entered.set()
            try:
                session.next()
            except BaseException as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertTrue(entered.wait(5))
        session.abort("cancel gated work")
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertEqual(session.snapshot()["cpu_completed"], 0)
        self.assertEqual(session.shutdown()["live_workers"], 0)

    def test_persistent_sequences_trace_bound_and_no_hot_trace(self):
        session = native.Session(2, 2, 4096, 0, trace=True, trace_limit=2)
        for seq in range(4):
            session.submit(seq, (seq,))
            self.assertEqual(session.next(), seq)
            session.retire(seq)
        with self.assertRaisesRegex(ValueError, "increasing"):
            session.submit(0, (0,))
        self.assertEqual(len(session.trace_records()), 2)
        self.assertEqual(session.snapshot()["trace_dropped"], 2)
        session.shutdown()
        hot = native.Session(1, 1, 4096, 0)
        hot.submit(0, (1,))
        hot.next()
        hot.retire(0)
        self.assertEqual(hot.trace_records(), [])
        self.assertIsNone(native.prepare_inline((1,), 0)["wall_ns"])
        hot.shutdown()

    @unittest.skipUnless(sys.platform == "win32", "QueryThreadCycleTime is a Windows counter")
    def test_native_thread_cycles_are_observed_without_ns_fabrication(self):
        row = native.prepare_inline((1, 2, 3), 100000, trace=True)
        self.assertIn("thread_cycles", row)
        self.assertGreater(row["thread_cycles"], 0)
        self.assertGreaterEqual(row["thread_cpu_ns"], 0)
        session = native.Session(1, 1, 4096, 100000, trace=True)
        session.submit(0, (1, 2, 3))
        session.next()
        session.retire(0)
        session.shutdown()
        trace = session.trace_records()[0]
        self.assertGreater(trace["thread_cycles"], 0)
        self.assertNotEqual(trace["tid"], threading.get_native_id())


if __name__ == "__main__":
    unittest.main(verbosity=2)
