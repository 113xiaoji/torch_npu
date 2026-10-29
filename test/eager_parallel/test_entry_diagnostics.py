"""Compile and exercise the actual native shadow bookkeeping implementation."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class EntryDiagnosticsTests(unittest.TestCase):
    def test_native_shadow_lifecycle_and_foreign_thread(self):
        header=ROOT/'torch_npu/csrc/core/npu/eager_parallel/EntryDiagnostics.h'
        self.assertTrue(header.is_file(),'native entry diagnostics has not been implemented')
        spec=importlib.util.spec_from_file_location('r26_native_test_compiler',
            ROOT/'test/host_regions/test_eager_prepare_integration.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        helper.NativeOwnedJobTests().compile_run(Path(__file__).with_name('entry_diagnostics_fixture.cpp'))


if __name__=='__main__':unittest.main()
