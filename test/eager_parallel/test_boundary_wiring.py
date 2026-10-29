"""Central bypasses must report before reading or replacing storage metadata."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class BoundaryWiringTests(unittest.TestCase):
    def test_storage_resize_precedes_metadata_and_data_ptr_replacement(self):
        source=(ROOT/'torch_npu/csrc/aten/common/ResizeNpu.h').read_text(encoding='utf-8')
        body=source[source.index('static void storage_resize_npu('):]
        self.assertIn('eager_parallel::Boundary("storage.resize")',body)
        self.assertLess(body.index('eager_parallel::Boundary'),body.index('storage.resizable()'))

    def test_set_storage_precedes_reading_old_storage(self):
        source=(ROOT/'torch_npu/csrc/aten/common/SetNpu.cpp').read_text(encoding='utf-8')
        body=source[source.index('void set_storage_nd_npu('):]
        self.assertIn('eager_parallel::Boundary("tensor.set_storage")',body)
        self.assertLess(body.index('eager_parallel::Boundary'),body.index('self.storage()'))

    def test_host_scalar_precedes_host_value_read(self):
        source=(ROOT/'torch_npu/csrc/aten/common/LocalScalarDenseNpu.cpp').read_text(encoding='utf-8')
        body=source[source.index('NPUNativeFunctions::_local_scalar_dense('):]
        self.assertIn('eager_parallel::Boundary("tensor.item")',body)
        self.assertLess(body.index('eager_parallel::Boundary'),body.index('self.numel()'))


if __name__=='__main__':unittest.main()
