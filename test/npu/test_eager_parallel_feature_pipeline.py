"""Real NPU gate entry; unavailable/unqualified hardware is SKIP, never PASS.

Run with R26_DLRM_SOURCE pointing at the complete frozen upstream checkout.
No NoOp fallback, fabricated capability, or rewritten model is installed here.
"""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]


class RealFeaturePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch_npu
        except (ImportError, OSError) as error:
            raise unittest.SkipTest("real NPU runtime unavailable: " + str(error)) from error
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("real NPU device unavailable; CPU/NoOp is not an NPU numerical PASS")
        try:
            from torch_npu import _eager_parallel
        except ImportError as error:
            raise unittest.SkipTest("R26 native binding unavailable") from error
        snapshot = _eager_parallel.snapshot()
        if snapshot.get("real_sdk_qualified") is not True:
            raise unittest.SkipTest("actual native SDK qualification absent; shadow/NoOp does not qualify a device gate")
        source = os.environ.get("R26_DLRM_SOURCE")
        if not source:
            raise unittest.SkipTest("R26_DLRM_SOURCE absent; original model must remain source-pinned")
        path = ROOT / "benchmarks/eager_parallel_r26/reference.py"
        spec = importlib.util.spec_from_file_location("r26_npu_reference", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        cpu, inputs, cls.provenance = reference.load_original_model(source)
        cls.torch, cls.runtime, cls.reference = torch, _eager_parallel, reference
        with torch.no_grad():
            cls.cpu_expected = cpu(*inputs)
        cls.model = copy.deepcopy(cpu).to("npu:0")
        cls.inputs = (inputs[0].to("npu:0"), [value.to("npu:0") for value in inputs[1]],
                      [value.to("npu:0") for value in inputs[2]])

    def run_original(self, enabled, mode):
        with self.torch.inference_mode():
            if not enabled:
                self.runtime.configure(enabled=False, mode="shadow")
            context = self.runtime.scope(mode=mode, streams=2) if enabled else nullcontext()
            with context:
                result = self.model(*self.inputs)
        self.torch.npu.synchronize()
        return result.detach().cpu()

    def test_original_forward_shadow_matches_eager_off_and_cpu(self):
        baseline = self.run_original(False, "shadow")
        candidate = self.run_original(True, "shadow")
        self.torch.testing.assert_close(baseline, self.cpu_expected, rtol=1e-4, atol=1e-5)
        self.torch.testing.assert_close(candidate, baseline, rtol=1e-4, atol=1e-5)
        self.assertIsNone(self.runtime.snapshot().get("first_error"))

    def test_original_forward_parallel_requires_native_ei_capability(self):
        snapshot = self.runtime.snapshot()
        if snapshot.get("device_parallel_qualified") is not True:
            self.skipTest("device parallel scheduling not qualified; shadow correctness is not EI qualification")
        # A future backend must explicitly advertise its mode; never silently
        # run shadow after asking for a parallel numerical qualification.
        self.assertIn("EI", snapshot.get("qualified_modes", []))
        baseline = self.run_original(False, "shadow")
        candidate = self.run_original(True, "EI")
        self.torch.testing.assert_close(candidate, baseline, rtol=1e-4, atol=1e-5)
        self.assertIsNone(self.runtime.snapshot().get("first_error"))


if __name__ == "__main__":
    unittest.main()
