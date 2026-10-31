"""Execute the same bounded synchronous-submit protocol used by the native adapter."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ImmediateProtocolTests(unittest.TestCase):
    def test_immediate_group_ordering_and_failures(self):
        self.assertTrue((ROOT / "torch_npu/csrc/core/npu/eager_parallel/ImmediateProtocol.h").is_file(),
                        "actual immediate-submit group protocol is missing")
        spec = importlib.util.spec_from_file_location("r27_compiler", ROOT / "test/eager_parallel/test_native_core.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.NativeCoreTests().compile_run("immediate_protocol_test.cpp", b"PASS R27 immediate protocol")


if __name__ == "__main__":
    unittest.main()
