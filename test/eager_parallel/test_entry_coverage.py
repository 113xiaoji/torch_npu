"""Host code-generation contract tests; these do not qualify NPU execution."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EntryCoverageTests(unittest.TestCase):
    def test_separate_r26_overlay_checks_pinned_inputs_before_writing(self):
        import subprocess
        import sys
        installer = ROOT / 'tools/eager_parallel/apply_overlay.py'
        self.assertTrue(installer.is_file(), 'R26 standalone overlay installer is missing')
        result = subprocess.run([sys.executable, str(installer), '--repo-root', str(ROOT), '--check'],
                                text=True, capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('R26 overlay check passed', result.stdout)

    def test_op_plugin_overlay_enters_before_route_selection(self):
        path = ROOT / 'patches/eager_parallel_r26/op-plugin/torchnpugen/gen.py'
        self.assertTrue(path.is_file(), 'R26 op-plugin entry generator is missing')
        source = path.read_text(encoding='utf-8')
        self.assertIn('wrap_entry_body(', source)
        template = ROOT / 'patches/eager_parallel_r26/op-plugin/torchnpugen/templates/OpInterface.cpp'
        self.assertIn('eager_parallel/EntryGuard.h', template.read_text(encoding='utf-8'))

    def test_actual_op_plugin_generator_preserves_four_embedding_outputs(self):
        import subprocess
        import sys
        script = '''
from torchnpugen.gen import gen_return
from torchnpugen.model import NativeFunction
f, _ = NativeFunction.from_yaml({
    'func': '_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)',
    'impl_ns': 'op_api, acl_op'})
output = gen_return(f, {})[0]
assert output.index('eager_parallel::Invoke') < output.index('CheckJitDisable')
assert output.index('eager_parallel::Invoke') < output.index('FormatHelper::')
assert '-> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>' in output
assert 'mode, sparse, per_sample_weights, include_last_offset, padding_idx);' in output
print('actual op-plugin generator contract passed')
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
                                cwd=ROOT / 'third_party/op-plugin', capture_output=True,
                                text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_overlay_rejects_unrelated_target_edits_without_replacing_files(self):
        import json
        import tempfile
        path = ROOT / 'tools/eager_parallel/apply_overlay.py'
        spec = importlib.util.spec_from_file_location('r26_overlay_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        overlay = ROOT / 'patches/eager_parallel_r26'
        manifest = json.loads((overlay / 'op-plugin-manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp).resolve()
            for name in module.FILES:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((overlay / 'op-plugin' / name).read_bytes())
            altered = target / 'torchnpugen/gen.py'
            altered.write_text('user edit\n', encoding='utf-8')
            before = {name: (target / name).read_bytes() for name in module.FILES}
            with self.assertRaisesRegex(ValueError, 'unrelated modifications'):
                module.plan(overlay / 'op-plugin', target, manifest)
            self.assertEqual(before, {name: (target / name).read_bytes() for name in module.FILES})

    def test_wrapper_enters_before_device_metadata_and_host_hook(self):
        path = ROOT / 'torchnpugen/eager_parallel.py'
        self.assertTrue(path.is_file(), 'R26 native entry generator is missing')
        spec = importlib.util.spec_from_file_location('r26_codegen', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = module.wrap_entry_body('aten::index.Tensor', 'dispatcher',
                                       'at::Tensor', ['self', 'indices'],
                                       'device_check();\nhost_hook();\nreturn kernel();')
        self.assertLess(output.index('eager_parallel::Invoke'), output.index('device_check()'))
        self.assertLess(output.index('eager_parallel::Invoke'), output.index('host_hook()'))
        self.assertIn('"aten::index.Tensor", "dispatcher"', output)
        self.assertIn('[&]() -> at::Tensor', output)
        self.assertIn('}, self, indices);', output)

    def test_generator_handles_void_and_no_arguments(self):
        path = ROOT / 'torchnpugen/eager_parallel.py'
        self.assertTrue(path.is_file(), 'R26 native entry generator is missing')
        spec = importlib.util.spec_from_file_location('r26_codegen_void', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = module.wrap_entry_body('aten::unknown', 'dispatcher', 'void', [], 'kernel();')
        self.assertIn('[&]() -> void', output)
        self.assertTrue(output.endswith('});'))

    def test_all_dispatch_generators_use_entry_and_include_header(self):
        for path in ('torchnpugen/utils.py', 'torchnpugen/custom_functions.py'):
            self.assertIn('wrap_entry_body(', (ROOT / path).read_text(encoding='utf-8'), path)
        for path in ('torchnpugen/gen_backend_stubs.py',
                     'torchnpugen/templates/CustomRegisterSchema.cpp'):
            self.assertIn('eager_parallel/EntryGuard.h', (ROOT / path).read_text(encoding='utf-8'), path)

    def test_shared_native_and_structured_paths_also_wrap_before_the_body(self):
        native = (ROOT / 'torchnpugen/utils.py').read_text(encoding='utf-8')
        shared = native[native.index('def get_target_native_registration('):]
        self.assertIn('wrap_entry_body(', shared)
        structured = (ROOT / 'torchnpugen/gen_backend_stubs.py').read_text(encoding='utf-8')
        self.assertIn('sig_body_str = wrap_entry_body(', structured)

    def test_actual_unstructured_generator_enters_before_backend_call(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from torchgen.model import BackendIndex, BackendMetadata, DispatchKey, Location, NativeFunction
        from torchgen.dest.register_dispatch_key import RegisterDispatchKey
        from torchgen.selective_build.selector import SelectiveBuilder
        from torchgen.utils import Target
        from torchnpugen.utils import gen_unstructured
        function, _ = NativeFunction.from_yaml(
            {'func': 'r26_unknown(Tensor self) -> Tensor'}, Location('test.yaml', 1), set())
        index = BackendIndex(DispatchKey.PrivateUse1, False, True, True,
                             {function.func.name: BackendMetadata('r26_unknown', False, 'at_npu::native')})
        generator = RegisterDispatchKey(index, Target.ANONYMOUS_DEFINITION,
            SelectiveBuilder.get_nop_selector(), False, False, None, False)
        output = gen_unstructured(generator, function)
        self.assertLess(output.index('eager_parallel::Invoke'), output.index('check_and_update_common_device'))
        self.assertLess(output.index('eager_parallel::Invoke'), output.index('PreHook('))
        self.assertLess(output.index('eager_parallel::Invoke'), output.index('return at_npu::native::r26_unknown('))

    def test_header_keeps_disabled_route_allocation_free(self):
        path = ROOT / 'torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h'
        self.assertTrue(path.is_file(), 'native entry contract is missing')
        source = path.read_text()
        self.assertLess(source.index('if (!Enabled())'), source.index('EntryGuard guard('))
        self.assertIn('std::current_exception()', source)
        self.assertIn('void fail(std::exception_ptr error) noexcept', source)

    def test_shadow_collection_is_count_only_and_cannot_replace_operator_errors(self):
        source = (ROOT / 'torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h').read_text(encoding='utf-8')
        self.assertIn('std::size_t input_count) noexcept', source)
        self.assertIn('void complete(std::size_t output_count) noexcept', source)
        self.assertIn('NoThrowTensorCount', source)
        self.assertNotIn('std::unique_ptr', source)
        self.assertNotIn('std::vector<at::Tensor> inputs', source)
        self.assertNotIn('guard.complete(detail::tensors(', source)


if __name__ == '__main__':
    unittest.main()
