"""Central bypasses must report before reading or replacing storage metadata."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class BoundaryWiringTests(unittest.TestCase):
    def assert_boundary_precedes(self, path, function, reason, first_effect):
        source = (ROOT / path).read_text(encoding='utf-8')
        body = source[source.index(function):]
        hook = 'eager_parallel::Boundary("' + reason + '")'
        self.assertTrue(hook in body, function + ' must join before ' + first_effect)
        self.assertLess(body.index(hook), body.index(first_effect))

    def test_public_stream_and_event_effects_join_owned_work_first(self):
        stream = 'torch_npu/csrc/core/npu/NPUStream.cpp'
        event = 'torch_npu/csrc/core/npu/NPUEvent.cpp'
        functions = 'torch_npu/csrc/core/npu/NPUFunctions.cpp'
        guard = 'torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp'
        cases = [
            (stream, 'bool NPUStream::query() const', 'stream.query', 'host_regions::RejectEffect'),
            (stream, 'void NPUStream::synchronize() const', 'stream.synchronize', 'TORCH_CHECK'),
            (stream, 'bool npuSynchronizeDevice(bool check_error)', 'device.synchronize', 'host_regions::RejectEffect'),
            (stream, 'bool npuSynchronizeUsedDevices(bool check_error)', 'device.synchronize_used', 'host_regions::RejectEffect'),
            (functions, 'void device_synchronize()', 'device.synchronize', 'NPU_CHECK_ERROR'),
            (functions, 'aclError SynchronizeUsedDevices()', 'device.synchronize_used', 'int32_t cur_device'),
            (event, 'bool NPUEvent::query() const', 'event.query', 'if (!is_created_)'),
            (event, 'void NPUEvent::record(const NPUStream& stream)', 'event.record', 'checkNotExternalStream'),
            (event, 'void NPUEvent::block(const NPUStream& stream)', 'event.wait', 'checkNotExternalStream'),
            (event, 'void NPUEvent::synchronize() const', 'event.synchronize', 'if (is_created_)'),
            (event, 'float NPUEvent::elapsed_time(', 'event.elapsed_time', 'TORCH_CHECK'),
            (event, 'uint64_t NPUEvent::recorded_time() const', 'event.recorded_time', 'TORCH_CHECK'),
            (event, 'void NPUEvent::reset(', 'event.reset', 'if (!c10_npu::acl'),
            (event, 'void NPUEvent::ipc_handle(', 'event.ipc_handle', 'if (!is_created_)'),
            (guard, 'void NPUGuardImpl::record(', 'event.record', 'TORCH_CHECK'),
            (guard, 'void NPUGuardImpl::block(', 'event.wait', 'if (!event)'),
            (guard, 'bool NPUGuardImpl::queryEvent(', 'event.query', 'if (!event)'),
            (guard, 'void NPUGuardImpl::synchronizeEvent(', 'event.synchronize', 'if (!event)'),
        ]
        for case in cases:
            with self.subTest(path=case[0], function=case[1]):
                self.assert_boundary_precedes(*case)

    def test_direct_mutation_and_cpu_fallback_precede_tensor_access(self):
        cases = [
            ('torch_npu/csrc/aten/common/ResizeNpu.h', 'inline at::TensorImpl* resize_impl_npu_(', 'tensor.resize', 'if (self->sizes()'),
            ('torch_npu/csrc/aten/common/CopyKernel.cpp', 'at::Tensor& NPUNativeFunctions::copy_(', 'tensor.copy', 'if (self.numel()'),
            ('torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp', 'at::Tensor& NPUNativeOpApiFunctions::copy_(', 'tensor.copy', 'DO_COMPATIBILITY'),
            ('torch_npu/csrc/aten/common/CopyMemoryKernel.cpp', 'at::Tensor& NPUNativeFunctions::copy_memory_(', 'tensor.copy_memory', 'c10_npu::NPUGuard'),
            ('torch_npu/csrc/aten/common/ChangeDataPtr.cpp', 'int64_t NPUNativeFunctions::npu_change_data_ptr(', 'tensor.change_data_ptr', 'TORCH_CHECK'),
            ('torch_npu/csrc/aten/common/TensorShape.cpp', 'const at::Tensor& NPUNativeFunctions::as_strided__symint(', 'tensor.as_strided_', 'auto ks = self.key_set()'),
            ('torch_npu/csrc/aten/ops/StreamAndEventKernelNpu.cpp', 'void NPUNativeFunctions::record_stream(', 'tensor.record_stream', 'struct c10::StreamData3'),
            ('torch_npu/csrc/aten/VariableFallbackKernel.cpp', 'void npu_cpu_fallback(', 'dispatch.cpu_fallback', 'if (!has_op_name_warned'),
        ]
        for case in cases:
            with self.subTest(path=case[0]):
                self.assert_boundary_precedes(*case)

    def test_python_setters_and_raw_stream_handle_join_before_effect(self):
        module = 'torch_npu/csrc/npu/Module.cpp'
        cases = [
            (module, 'PyObject* THNPModule_setStream_wrap(', 'stream.set_current', 'auto stream ='),
            (module, 'PyObject* THNPModule_setDevice_wrap(', 'device.set', 'torch_npu::utils::npu_lazy_init'),
            ('torch_npu/csrc/npu/Stream.cpp', 'static PyObject* THNPStream_get_npu_stream(', 'stream.raw_handle', 'return PyLong_FromVoidPtr'),
        ]
        for case in cases:
            with self.subTest(path=case[0], function=case[1]):
                self.assert_boundary_precedes(*case)

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
