import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/eager_prepare_cpu/run.py"


def campaign():
    if not RUNNER.exists():
        raise AssertionError("bounded CPU campaign runner is not implemented")
    spec = importlib.util.spec_from_file_location("r24_cpu_campaign", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CampaignContractTests(unittest.TestCase):
    def test_frozen_cpu_source_pin_rejects_changed_content(self):
        runner = campaign()
        self.assertTrue(hasattr(runner, "verify_cpu_source_pins"), "CPU source verification is missing")
        results = ROOT / "benchmarks/eager_prepare_cpu/results"
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-pins-", dir=results) as directory:
            source, manifest = Path(directory) / "source.txt", Path(directory) / "pins.json"
            source.write_bytes(b"frozen source")
            relative = source.relative_to(ROOT).as_posix()
            manifest.write_text(json.dumps({"files": {relative: {
                "bytes": 13, "sha256": hashlib.sha256(b"frozen source").hexdigest()}}}), encoding="utf-8")
            runner.verify_cpu_source_pins(manifest)
            source.write_bytes(b"changed source")
            with self.assertRaisesRegex(RuntimeError, "source pin mismatch"):
                runner.verify_cpu_source_pins(manifest)

    def test_cold_cpu_proof_is_separate_from_formal_performance_samples(self):
        runner = campaign()
        self.assertTrue(hasattr(runner, "cold_cpu_proof"), "separate heavy native CPU proof is missing")

    def test_paired_arm_orders_reverse_every_round_and_keep_all_results(self):
        runner = campaign()
        orders = [runner.arm_order(index) for index in range(3)]
        self.assertEqual(orders[1], list(reversed(orders[0])))
        self.assertEqual(orders[2], orders[0])
        self.assertEqual(set(orders[0]), {"direct_eager", "replay_serial", "queue_w1", "queue_w2", "queue_w4"})

    def test_summary_uses_actual_samples_and_reports_tail_quantiles(self):
        runner = campaign()
        summary = runner.summarize_ns([1000, 2000, 3000, 4000, 5000])
        self.assertEqual(summary["samples"], 5)
        self.assertEqual(summary["mean_us"], 3.0)
        self.assertEqual(summary["p50_us"], 3.0)
        self.assertGreaterEqual(summary["p99_us"], summary["p95_us"])
        self.assertEqual(summary["min_us"], 1.0)
        self.assertEqual(summary["max_us"], 5.0)

    def test_overlap_requires_real_distinct_native_threads_and_intersecting_intervals(self):
        runner = campaign()
        records = [
            {"seq": 0, "tid": 101, "start_ns": 10, "end_ns": 40, "thread_cpu_ns": 25},
            {"seq": 1, "tid": 202, "start_ns": 20, "end_ns": 60, "thread_cpu_ns": 30},
            {"seq": 2, "tid": 101, "start_ns": 61, "end_ns": 70, "thread_cpu_ns": 8},
        ]
        overlap = runner.overlap_evidence(records, [{"start_ns": 30, "end_ns": 55}])
        self.assertTrue(overlap["native_prepare_overlap"])
        self.assertTrue(overlap["prepare_consume_overlap"])
        self.assertEqual(overlap["native_worker_tids"], [101, 202])
        records[1]["tid"] = 101
        self.assertFalse(runner.overlap_evidence(records, [])["native_prepare_overlap"])


if __name__ == "__main__":
    unittest.main()
