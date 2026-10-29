"""Real CPU operators and the production-core native bridge; no queue mocks."""
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DRIVER = ROOT / "benchmarks/eager_prepare_cpu/simulation.py"


def driver():
    if not DRIVER.exists():
        raise AssertionError("CPU capture/replay driver has not been implemented")
    name = "eager_prepare_cpu_simulation"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DRIVER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class CaptureReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def test_00_driver_available(self):
        self.assertTrue(DRIVER.exists(), "CPU capture/replay driver has not been implemented")

    def test_dependency_uses_this_replay_values_and_releases_capture_intermediates(self):
        sim, torch = driver(), self.torch
        refs = []

        class Chain(torch.nn.Module):
            def forward(self, x):
                intermediate = x + 3
                refs.append(weakref.ref(intermediate))
                return intermediate.square() - x

        x = torch.tensor([2.0, 4.0])
        tape = sim.capture(Chain(), (x,))
        gc.collect()
        self.assertIsNone(refs[0](), "capture must not retain intermediate Tensor values")
        replacement = torch.tensor([5.0, 7.0])
        output, audit = sim.replay(tape, initial_overrides={tape.input_ids[0]: replacement})
        torch.testing.assert_close(output, (replacement + 3).square() - replacement)
        self.assertEqual(audit["consume_order"], list(range(len(tape.ops))))

    def test_multiple_outputs_nested_args_and_real_view_aliases(self):
        sim, torch = driver(), self.torch

        class Views(torch.nn.Module):
            def forward(self, x):
                a, b = x.chunk(2, dim=1)
                return {"first": a.transpose(0, 1), "rest": (b + a, x.view(-1))}

        x = torch.arange(24.0).reshape(4, 6)
        model = Views()
        tape = sim.capture(model, (x,))
        actual, _ = sim.replay(tape)
        sim.assert_tree_close(torch, actual, model(x))
        self.assertEqual(actual["first"].untyped_storage().data_ptr(), x.untyped_storage().data_ptr())
        self.assertEqual(actual["rest"][1].untyped_storage().data_ptr(), x.untyped_storage().data_ptr())
        self.assertTrue(any(len(op.output_ids) > 1 for op in tape.ops))

    def test_mutation_rng_custom_and_scalar_data_dependency_are_rejected(self):
        sim, torch = driver(), self.torch

        class Mutable(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)

        class Random(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand_like(x)

        class DataDependent(torch.nn.Module):
            def forward(self, x):
                return x + x.sum().item()

        class BooleanIndex(torch.nn.Module):
            def forward(self, x):
                return x[x > 0]

        for model in (Mutable(), Random(), DataDependent(), BooleanIndex()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(sim.UnsupportedCapture, "mutation|RNG|data-dependent"):
                    sim.capture(model, (torch.ones(2),))

    def test_custom_operator_is_rejected_before_its_side_effect_runs(self):
        sim, torch = driver(), self.torch
        calls = []
        library = torch.library.Library("r24_cpu_capture_test", "DEF")
        library.define("side_effect(Tensor input) -> Tensor")

        def implementation(x):
            calls.append("side effect")
            return x + 1

        library.impl("side_effect", implementation, "CPU")

        class Custom(torch.nn.Module):
            def forward(self, x):
                return torch.ops.r24_cpu_capture_test.side_effect.default(x)

        with self.assertRaisesRegex(sim.UnsupportedCapture, "custom operator/side effects"):
            sim.capture(Custom(), (torch.ones(2),))
        self.assertEqual(calls, [])

    def test_all_four_pinned_models_match_independent_eager_calls_and_hashes(self):
        sim, torch = driver(), self.torch
        models, pins = sim.load_pinned_models()
        self.assertEqual(len(models.MODEL_NAMES), 4)
        self.assertEqual(pins["models_sha256"], "d67d1f49708c154cabf7b04f4af1eda2f368b03d450a9536928e2cadf50d11db")
        for name in models.MODEL_NAMES:
            with self.subTest(model=name):
                workload = models.build_workload(torch, name)
                before = sim.fingerprint(workload.model, workload.inputs)
                tape = sim.capture(workload.model, workload.inputs)
                reference = sim.reference_run(workload.model, workload.inputs)
                actual, audit = sim.replay(tape, reference=reference)
                sim.assert_tree_close(torch, actual, reference.output)
                self.assertEqual(audit["validated_calls"], len(tape.ops))
                self.assertEqual(before, sim.fingerprint(workload.model, workload.inputs))
                self.assertGreaterEqual(len(tape.ops), 10)

    def test_wrong_operator_order_is_visible(self):
        sim, torch = driver(), self.torch
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tape = sim.capture(model, (torch.ones(2, 3),))
        with self.assertRaisesRegex(sim.ReplayError, "order|sequence"):
            sim.replay(tape, sequence_order=list(reversed(range(len(tape.ops)))))


class NativeReplayTests(unittest.TestCase):
    def test_cli_all_four_model_campaign_emits_real_scoped_receipt_and_exits(self):
        results = ROOT / "benchmarks/eager_prepare_cpu/results"
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-cli-", dir=results) as directory:
            output = Path(directory) / "receipt.json"
            completed = subprocess.run([sys.executable,
                str(ROOT / "benchmarks/eager_prepare_cpu/run.py"), "--output", str(output),
                "--cost-us", "0", "200", "--warmup", "0", "--samples", "1", "--rounds", "1"],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(receipt["real_npu"])
            self.assertTrue(receipt["simulated_prepare"])
            self.assertEqual(len(receipt["models"]), 4)
            self.assertLessEqual(len(receipt["environment"]["actual_allowed_cpus"]), 8)
            self.assertEqual(receipt["environment"]["torch_intra_threads"], 1)
            self.assertEqual(receipt["environment"]["torch_inter_threads"], 1)
            proof = receipt["cold_cpu_proof"]
            self.assertTrue(proof["overlap"]["native_prepare_overlap"])
            self.assertGreater(proof["overlap"]["prepare_cpu_ns_sum"], 0)
            self.assertEqual(proof["before_hash"], proof["after_hash"])
            for model in receipt["models"].values():
                self.assertEqual(model["campaign_before_hash"], model["campaign_after_hash"])
                self.assertEqual(len(model["correctness"]["records"]), 8)
                for cost in model["performance"]:
                    self.assertEqual(len(cost["rounds"][0]["arms"]), 5)
                    for arm in cost["rounds"][0]["arms"]:
                        self.assertEqual(len(arm["durations_ns"]), 1)
                        self.assertTrue(arm["last_timed_output_matches_direct_eager"])
                        self.assertFalse(arm["trace_enabled"])

    def test_real_aten_consumer_exception_preserves_its_original_type(self):
        sim = driver()
        from tools.eager_prepare_cpu import load_native
        import torch
        model = torch.nn.Embedding(4, 3).eval()
        model.weight.requires_grad_(False)
        tape = sim.capture(model, (torch.tensor([0, 1]),))
        with self.assertRaises(IndexError):
            sim.replay(tape, native=load_native(), workers=2, capacity=1,
                initial_overrides={tape.input_ids[0]: torch.tensor([1000, 1])})

    def test_persistent_pool_uses_monotonic_sequences_across_forwards(self):
        sim = driver()
        self.assertTrue(hasattr(sim, "ReplayRunner"), "persistent replay pool is not implemented")
        from tools.eager_prepare_cpu import load_native
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False), torch.nn.ReLU()).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inputs = (torch.ones(2, 3),)
        tape = sim.capture(model, inputs)
        runner = sim.ReplayRunner(tape, native=load_native(), workers=2, capacity=2, iterations=1000)
        try:
            for index in range(3):
                output, audit = runner.run()
                sim.assert_tree_close(torch, output, model(*inputs))
                self.assertEqual(audit["base_seq"], index * len(tape.ops))
            snap = runner.snapshot()
            self.assertEqual(snap["workers_started"], 2)
            self.assertEqual(snap["retired"], 3 * len(tape.ops))
        finally:
            runner.close()

    def test_native_queue_all_models_checksums_bounds_retirement_and_overlap(self):
        sim = driver()
        from tools.eager_prepare_cpu import load_native
        native = load_native()
        import torch
        models, _ = sim.load_pinned_models()
        for name in models.MODEL_NAMES:
            workload = models.build_workload(torch, name)
            tape = sim.capture(workload.model, workload.inputs)
            reference = sim.reference_run(workload.model, workload.inputs)
            for workers in (1, 2, 4):
                with self.subTest(model=name, workers=workers):
                    output, audit = sim.replay(tape, native=native, workers=workers,
                        capacity=4, byte_limit=65536, iterations=10000, trace=True,
                        reference=reference)
                    sim.assert_tree_close(torch, output, reference.output)
                    snap = audit["native"]
                    self.assertEqual(snap["admitted"], len(tape.ops))
                    self.assertEqual(snap["retired"], len(tape.ops))
                    self.assertEqual(snap["prepare_calls"], len(tape.ops))
                    self.assertLessEqual(snap["peak_slots_in_use"], 4)
                    self.assertLessEqual(snap["peak_retained_bytes"], 65536)
                    self.assertEqual(snap["checksum_xor"], sim.expected_checksum(native, tape, 10000))
                    self.assertFalse(snap["real_npu"])
                    self.assertTrue(audit["trace"])
                    self.assertTrue(all(record["thread_cpu_ns"] >= 0 for record in audit["trace"]))
                    self.assertTrue(all(record["end_ns"] > record["start_ns"] for record in audit["trace"]))

    def test_native_consumer_failure_aborts_producer_without_deadlock(self):
        sim = driver()
        from tools.eager_prepare_cpu import load_native
        import torch

        class Chain(torch.nn.Module):
            def forward(self, x):
                for _ in range(15):
                    x = torch.relu(x + 1)
                return x

        tape = sim.capture(Chain(), (torch.ones(2),))
        with self.assertRaisesRegex(sim.ReplayError, "injected consumer failure"):
            sim.replay(tape, native=load_native(), workers=2, capacity=2,
                iterations=1000, fail_consumer_at=1)


if __name__ == "__main__":
    unittest.main()
