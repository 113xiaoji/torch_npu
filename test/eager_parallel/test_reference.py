"""Independent arithmetic references; these tests make no NPU claim."""
import importlib.util
import os
from pathlib import Path
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "benchmarks/eager_parallel_r26/reference.py"
    if not path.is_file():
        raise AssertionError("R26 explicit numerical reference is not implemented")
    spec = importlib.util.spec_from_file_location("r26_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.r = load()

    def test_sum_preserves_last_empty_bag_and_repeated_indices(self):
        self.assertEqual(self.r.embedding_bag_sum_fp64(
            [[1, 2], [3, 1], [2, 4]], [0, 1, 2, 2], [0, 2, 4]),
            [[4.0, 3.0], [4.0, 8.0], [0.0, 0.0]])

    def test_padding_none_negative_and_last_offset(self):
        weights = [[1, 2], [3, 1], [2, 4]]
        self.assertEqual(self.r.embedding_bag_sum_fp64(weights, [0, 2], [0, 2],
            include_last_offset=True, padding_idx=None), [[3.0, 6.0]])
        self.assertEqual(self.r.embedding_bag_sum_fp64(weights, [0, 2], [0, 2],
            include_last_offset=True, padding_idx=-1), [[1.0, 2.0]])
        self.assertEqual(self.r.embedding_bag_sum_fp64(weights, [], [0, 0]),
            [[0.0, 0.0], [0.0, 0.0]])

    def test_optional_sample_weights_is_not_truthiness(self):
        self.assertEqual(self.r.embedding_bag_sum_fp64([[1.0]], [], [0], per_sample_weights=[]), [[0.0]])
        with self.assertRaisesRegex(ValueError, "sample weights"):
            self.r.embedding_bag_sum_fp64([[1.0]], [0], [0], per_sample_weights=[])
        self.assertEqual(self.r.embedding_bag_sum_fp64([[2.0]], [0, 0], [0],
            per_sample_weights=[0.5, 2.0]), [[5.0]])

    def test_invalid_offsets_and_indices_rejected(self):
        for indices, offsets in [([0], [1]), ([0], [0, 2]), ([0], [0, -1]), ([-1], [0])]:
            with self.subTest(indices=indices, offsets=offsets), self.assertRaises(ValueError):
                self.r.embedding_bag_sum_fp64([[1.0]], indices, offsets)

    def test_interaction_original_pair_order_self_and_cat(self):
        result = self.r.interaction_fp64([[1, 2]], [[[3, 4]], [[5, 6]]])
        self.assertEqual(result["li"], [1, 2, 2])
        self.assertEqual(result["lj"], [0, 0, 1])
        self.assertEqual(result["R"], [[1.0, 2.0, 11.0, 17.0, 39.0]])
        self.assertEqual(self.r.interaction_fp64([[1, 2]], [[[3, 4]]], itself=True)["R"],
            [[1.0, 2.0, 5.0, 11.0, 25.0]])
        self.assertEqual(self.r.interaction_fp64([[1, 2]], [[[3, 4]]], mode="cat")["R"],
            [[1.0, 2.0, 3.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "interaction"):
            self.r.interaction_fp64([[1]], [[[2]]], mode="unknown")

    def test_cpu_matrix_matches_independent_reference(self):
        report = self.r.cpu_reference_report()
        self.assertTrue(report["cpu_reference_pass"])
        self.assertFalse(report["real_device_numerical_claim"])
        self.assertEqual({case["tables"] for case in report["cases"]}, {1, 2, 7, 26, 39})
        self.assertTrue(all(case["embedding_pass"] and case["interaction_pass"] for case in report["cases"]))

    def test_four_output_contract_does_not_read_undefined_auxiliary_values(self):
        import torch
        result = (torch.tensor([[1.0]]), torch.tensor([999]), torch.tensor([float("nan")]), torch.tensor([42]))
        reference = (torch.tensor([[1.0]]), torch.tensor([-1]), torch.tensor([float("nan")]), torch.tensor([-42]))
        contract = {1: [], 2: [], 3: []}
        self.assertTrue(self.r.compare_four_outputs(result, reference, contract)["passed"])
        contract[1] = [(0,)]
        self.assertFalse(self.r.compare_four_outputs(result, reference, contract)["passed"])
        with self.assertRaisesRegex(ValueError, "four"):
            self.r.compare_four_outputs(result[:1], reference, contract)
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            self.r.compare_four_outputs(result, reference, {})

    def test_auxiliary_shape_mismatch_fails_before_value_reads(self):
        import torch
        actual = (torch.ones(1, 1), torch.ones(2), torch.empty(0), torch.empty(0))
        baseline = (torch.ones(1, 1), torch.empty(0), torch.empty(0), torch.empty(0))
        report = self.r.compare_four_outputs(actual, baseline, {1: [(0,)], 2: [], 3: []})
        self.assertFalse(report["passed"])
        self.assertFalse(report["metadata_pass"])

    def test_frozen_loader_and_original_model_pins(self):
        helper = self.r.load_frozen_helper()
        self.assertEqual(helper.SOURCE_SHA["dlrm_s_pytorch.py"], self.r.ORIGINAL_MODEL_SHA256)
        self.assertEqual(helper.STATE_SHA, self.r.ORIGINAL_STATE_SHA256)
        self.assertEqual(helper.INPUT_SHA, self.r.ORIGINAL_INPUT_SHA256)

    def test_original_loader_rejects_unpinned_source_before_import(self):
        self.assertTrue(callable(getattr(self.r, "load_original_model", None)), "original model loader missing")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.r.load_original_model(ROOT / "test/eager_parallel")

    def test_npu_test_entry_uses_actual_public_scope_contract(self):
        # A CPU harness test of the entry's API usage, not a simulated NPU pass.
        import torch
        path = ROOT / "test/npu/test_eager_parallel_feature_pipeline.py"
        spec = importlib.util.spec_from_file_location("r26_npu_entry_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []
        case = module.RealFeaturePipelineTests()

        class Runtime:
            def configure(self, *, enabled, mode):
                calls.append(("configure", enabled, mode))

            @contextmanager
            def scope(self, *, mode, streams):
                if not torch.is_inference_mode_enabled():
                    raise RuntimeError("scope requires inference_mode")
                calls.append(("scope", mode, streams))
                yield

        case.runtime = Runtime()
        case.torch = SimpleNamespace(inference_mode=torch.inference_mode, no_grad=torch.no_grad,
            npu=SimpleNamespace(synchronize=lambda: calls.append(("sync",))))
        case.model, case.inputs = torch.nn.Identity(), (torch.tensor([3.0]),)
        self.assertEqual(case.run_original(False, "shadow").tolist(), [3.0])
        self.assertEqual(case.run_original(True, "shadow").tolist(), [3.0])
        self.assertEqual(calls, [("configure", False, "shadow"), ("sync",), ("scope", "shadow", 2), ("sync",)])

    @unittest.skipUnless(os.environ.get("R26_DLRM_SOURCE"), "R26_DLRM_SOURCE absent: original pinned model not executed")
    def test_original_model_cpu_reference_and_identity(self):
        report = self.r.cpu_original_report(Path(os.environ["R26_DLRM_SOURCE"]))
        self.assertTrue(report["cpu_reference_pass"])
        self.assertEqual(report["identity"]["state_sha256"], self.r.ORIGINAL_STATE_SHA256)
        self.assertEqual(report["identity"]["input_sha256"], self.r.ORIGINAL_INPUT_SHA256)
        self.assertTrue(report["model_loader_adapter"]["all_other_ast_nodes_unchanged"])
        self.assertFalse(report["real_device_numerical_claim"])


if __name__ == "__main__":
    unittest.main()
