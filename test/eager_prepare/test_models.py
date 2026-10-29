import importlib.util
import os
from collections import Counter
from pathlib import Path
import unittest
from unittest import mock

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
SOURCE = Path(__file__).parents[2] / "benchmarks/eager_prepare/models.py"


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        cls.torch = torch

    def load(self):
        self.assertTrue(SOURCE.is_file(), "R24 workload factory is not implemented")
        spec = importlib.util.spec_from_file_location("r24_model_tests", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_four_actual_model_families_have_all_finite_full_outputs(self):
        m = self.load()
        expected = {"recommendation": (128, 1), "branch_mlp": (32, 16),
                    "branch_cnn": (4, 10), "transformer": (4, 16, 64)}
        for name, shape in expected.items():
            with self.subTest(name=name):
                work = m.build_workload(self.torch, name, seed=123)
                with self.torch.inference_mode():
                    output = work.model(*work.inputs)
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(self.torch.isfinite(output).all().item())
                self.assertTrue(work.required_core)
                self.assertFalse(work.model.training)
                self.assertTrue(all(not p.requires_grad for p in work.model.parameters()))

    def test_same_seed_preserves_all_weights_inputs_and_outputs(self):
        m = self.load()
        for name in m.MODEL_NAMES:
            a = m.build_workload(self.torch, name, seed=91)
            b = m.build_workload(self.torch, name, seed=91)
            for key, value in a.model.state_dict().items():
                self.assertTrue(self.torch.equal(value, b.model.state_dict()[key]))
            with self.torch.inference_mode():
                self.assertTrue(self.torch.equal(a.model(*a.inputs), b.model(*b.inputs)))

    def test_factory_does_not_consume_callers_cpu_rng_state(self):
        m = self.load()
        before = self.torch.get_rng_state().clone()
        m.build_workload(self.torch, "transformer", seed=99)
        self.assertTrue(self.torch.equal(before, self.torch.get_rng_state()))

    def test_declared_core_layers_are_actually_called_not_dead_model_members(self):
        m = self.load(); torch = self.torch
        for name in m.MODEL_NAMES:
            work = m.build_workload(torch, name)
            hits = Counter(); hooks = []
            for module in work.model.modules():
                family = ("embedding_bag" if isinstance(module, torch.nn.EmbeddingBag) else
                    "mm" if isinstance(module, torch.nn.Linear) else
                    "conv" if isinstance(module, torch.nn.Conv2d) else
                    "layer_norm" if isinstance(module, torch.nn.LayerNorm) else None)
                if family:
                    hooks.append(module.register_forward_hook(
                        lambda mod, args, output, key=family: hits.update([key])))
            try:
                with torch.inference_mode(), mock.patch.object(torch, "bmm", wraps=torch.bmm) as bmm, \
                        mock.patch.object(torch, "softmax", wraps=torch.softmax) as softmax:
                    work.model(*work.inputs)
                    hits["bmm"], hits["softmax"] = bmm.call_count, softmax.call_count
                for family, minimum in work.required_core.items():
                    self.assertEqual(hits[family], minimum, (name, family))
            finally:
                for hook in hooks:
                    hook.remove()

    def test_nonfinite_output_is_failed_but_remains_json_serializable(self):
        import json
        path = SOURCE.with_name("contracts.py")
        spec = importlib.util.spec_from_file_location("r24_numeric_tests", path)
        c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
        row = c.validate_output(self.torch, self.torch.tensor([float("nan")]), self.torch.tensor([1.0]))
        self.assertFalse(row["passed"])
        json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
