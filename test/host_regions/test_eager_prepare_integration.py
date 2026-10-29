"""Host-only public/native bridge contracts; never imports torch or an SDK."""
import importlib.util
from pathlib import Path
import sys
import os
import shutil
import subprocess
import tempfile
import sysconfig
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'torch_npu/_eager_prepare.py'


class PublicBridgeTests(unittest.TestCase):
    def load(self, backend):
        self.assertTrue(PUBLIC.is_file(), 'R24 eager-prepare public bridge is missing')
        spec = importlib.util.spec_from_file_location('_r24_public_bridge', PUBLIC)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch_npu': SimpleNamespace(_C=backend)}):
            spec.loader.exec_module(module)
        return module

    def test_default_is_off_and_configuration_is_explicit(self):
        calls = []
        backend = SimpleNamespace(_npu_eager_prepare_configure=lambda *a: calls.append(a) or '{"schema":1,"enabled":false}')
        module = self.load(backend)
        self.assertEqual(calls, [], 'import must not configure workers or initialize the device')
        result = module.configure()
        self.assertFalse(result['enabled'])
        self.assertIs(calls[0][0], False)
        self.assertEqual(calls[0][1:4], (4, 8, 536870912))

    def test_snapshot_and_close_route_to_native_without_model_callbacks(self):
        calls = []
        backend = SimpleNamespace(
            _npu_eager_prepare_snapshot=lambda: calls.append('snapshot') or '{"schema":1,"enabled":false}',
            _npu_eager_prepare_drain_close=lambda: calls.append('close') or '{"schema":1,"closure":{"passed":true}}')
        module = self.load(backend)
        self.assertEqual(module.snapshot()['schema'], 1)
        self.assertTrue(module.drain_close()['closure']['passed'])
        self.assertEqual(calls, ['snapshot', 'close'])

    def test_invalid_config_is_rejected_before_native_side_effect(self):
        calls = []
        module = self.load(SimpleNamespace(_npu_eager_prepare_configure=lambda *a: calls.append(a)))
        for kwargs in ({'enabled': 1}, {'workers': True}, {'capacity': 0}, {'byte_limit': -1},
                       {'allowed_ops': ['aclnnMm', 'aclnnMm']}, {'allowed_ops': ['anything']},
                       {'allowed_ops': 'aclnnMm'}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                module.configure(**kwargs)
        self.assertEqual(calls, [])

    def test_first_native_failure_is_not_replaced_by_cleanup_or_fallback(self):
        error = RuntimeError('unique-native-retirement-failure')
        def fail():
            raise error
        module = self.load(SimpleNamespace(_npu_eager_prepare_drain_close=fail))
        with self.assertRaises(RuntimeError) as caught:
            module.drain_close()
        self.assertIs(caught.exception, error)

    def test_native_result_requires_schema_one_object(self):
        for value in ('[]', '{"schema":2}', 'null'):
            module = self.load(SimpleNamespace(_npu_eager_prepare_snapshot=lambda: value))
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                module.snapshot()

    def test_cold_trace_is_explicit_and_hard_bounded(self):
        calls = []
        module = self.load(SimpleNamespace(_npu_eager_prepare_configure=lambda *a: calls.append(a) or '{"schema":1}'))
        module.configure(diagnostic_trace=True, trace_limit=32)
        self.assertEqual(calls[0][-2:], (True, 32))
        for kwargs in ({'diagnostic_trace': 1}, {'trace_limit': 257}, {'trace_limit': 0}, {'trace_limit': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                module.configure(**kwargs)
        self.assertEqual(len(calls), 1)


class NativeOwnedJobTests(unittest.TestCase):
    def test_actual_sync_macro_preserves_converted_parameter_expression(self):
        text = (ROOT / 'patches/eager_prepare_r24/op-plugin/op_plugin/utils/op_api_common.h').read_text(encoding='utf-8')
        start = text.index('#define EXEC_NPU_CMD_SYNC(')
        macro = text[start:text.index('\n\n', start)]
        template = (ROOT / 'test/host_regions/eager_prepare_integration/sync_expression_fixture.cpp.in').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory(prefix='r24-native-sync-expression-') as folder:
            source = Path(folder) / 'sync_expression.cpp'
            source.write_text(template.replace('/*MACRO*/', macro), encoding='utf-8')
            self.compile_run(source)

    def test_actual_copy_ack_unblocks_synchronous_publisher_and_keeps_late_commit(self):
        text = (ROOT / 'torch_npu/csrc/core/npu/EagerPrepare.cpp').read_text(encoding='utf-8')
        start = text.index('thread_local const void* publication_parameter')
        marker = text[start:text.index('thread_local bool native_job_active', start)]
        start = text.index('void QueueCopyAccepted(const void* parameter) noexcept')
        end = text.index('\n}\n', start) + 2
        callback = text[start:end]
        start = text.index('PublicationMarker marker(')
        construction = text[start:text.index(';', start) + 1]
        template = (ROOT / 'test/host_regions/eager_prepare_integration/publication_ack_fixture.cpp.in').read_text(encoding='utf-8')
        for key, value in {'MARKER': marker, 'CALLBACK': callback, 'CONSTRUCTION': construction}.items():
            template = template.replace('/*' + key + '*/', value)
        with tempfile.TemporaryDirectory(prefix='r24-native-copy-ack-') as folder:
            source = Path(folder) / 'publication_ack.cpp'; source.write_text(template, encoding='utf-8')
            self.compile_run(source)

    def test_actual_owned_job_bridge_without_sdk_or_python_worker_objects(self):
        header = ROOT / 'torch_npu/csrc/core/npu/EagerPrepareOwnedJob.h'
        self.assertTrue(header.is_file(), 'native owned job adapter is missing')
        source = ROOT / 'test/host_regions/eager_prepare_integration/owned_job_test.cpp'
        self.compile_run(source)

    def test_single_evaluation_and_valid_unadmitted_argument_fallback(self):
        header = ROOT / 'torch_npu/csrc/core/npu/EagerPrepareDispatch.h'
        self.assertTrue(header.is_file(), 'single-evaluation native dispatch is missing')
        self.compile_run(ROOT / 'test/host_regions/eager_prepare_integration/dispatch_test.cpp')

    def test_actual_storage_entry_keeps_safety_predicates_without_legacy_activation(self):
        text = (ROOT / 'torch_npu/csrc/core/npu/NPUCachingAllocator.cpp').read_text(encoding='utf-8')
        self.assertTrue('bool IsNativeEagerRetirementData(' in text, 'R24 independent allocator eligibility is missing')
        def function(signature):
            start = text.index(signature); brace = text.index('{', start); depth = 1; end = brace + 1
            while depth:
                depth += (text[end] == '{') - (text[end] == '}'); end += 1
            return text[start:end]
        template = (ROOT / 'test/host_regions/eager_prepare_integration/storage_gate_fixture.cpp.in').read_text(encoding='utf-8')
        values = {'LEAF':function('bool capturedRetirementEligible(const Block* block)'),
            'ON_STREAM':function('bool IsNativeCapturedRetirementDataOnStream('),
            'OLD_ENTRY':function('bool IsNativeCapturedRetirementData('),
            'EAGER_ENTRY':function('bool IsNativeEagerRetirementData(')}
        for key, value in values.items(): template = template.replace('/*'+key+'*/', value)
        with tempfile.TemporaryDirectory(prefix='r24-native-storage-facts-') as folder:
            source = Path(folder)/'storage_gate.cpp'; source.write_text(template,encoding='utf-8')
            self.compile_run(source)

    def compile_run(self, source):
        with tempfile.TemporaryDirectory(prefix='r24-owned-job-') as folder:
            build = Path(folder)
            if os.name == 'nt':
                vc = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')
                if not vc.is_file():
                    self.skipTest('MSVC C++ toolchain unavailable')
                script = build / 'compile.cmd'
                script.write_text('@echo off\ncall "' + str(vc) + '" -arch=x64 -host_arch=x64 >nul\n'
                    'if errorlevel 1 exit /b 1\ncl /nologo /std:c++17 /EHsc /W4 /WX /I"' + str(ROOT) +
                    '" "' + str(source) + '" /Fe:"' + str(build/'owned_job.exe') + '"\n', encoding='utf-8')
                command = ['cmd', '/d', '/c', str(script)]
                executable = build/'owned_job.exe'
            else:
                compiler = shutil.which('c++')
                if compiler is None:
                    self.skipTest('C++ compiler unavailable')
                executable = build/'owned_job'
                command = [compiler, '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror', '-I'+str(ROOT),
                           str(source), '-o', str(executable)]
            result = subprocess.run(command, cwd=build, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode('utf-8', errors='replace'))
            result = subprocess.run([str(executable)], cwd=build, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode('utf-8', errors='replace'))


class R24OverlayTests(unittest.TestCase):
    def module(self):
        tool = ROOT / 'tools/host_regions/apply_eager_prepare_overlay_r24.py'
        self.assertTrue(tool.is_file(), 'R24 independent overlay installer is missing')
        spec = importlib.util.spec_from_file_location('_r24_overlay', tool)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_exact_r24_payload_and_tampering_rejection(self):
        tool = self.module()
        values = tool.sources(ROOT)
        self.assertEqual(len(values), 8)
        self.assertIn('op_plugin/utils/op_api_eager_prepare_r24.h', values)
        import json
        with tempfile.TemporaryDirectory(prefix='r24-overlay-source-') as folder:
            copy_root = Path(folder).resolve()
            manifest = ROOT / tool.MANIFEST
            target_manifest = copy_root / tool.MANIFEST
            target_manifest.parent.mkdir(parents=True)
            target_manifest.write_bytes(manifest.read_bytes())
            raw = json.loads(manifest.read_bytes())
            for name, value in values.items():
                target = target_manifest.parent / raw['payload_root'] / name
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
            self.assertEqual(tool.sources(copy_root), values)
            name = next(iter(values))
            (target_manifest.parent / raw['payload_root'] / name).write_bytes(b'changed')
            with self.assertRaises(tool.OverlayError):
                tool.sources(copy_root)

    def test_only_clean_base_complete_parent_or_complete_r24_is_accepted(self):
        tool = self.module()
        values = {'a': b'new-a', 'b': b'new-b', 'c': b'new-c'}
        base = {'a': b'base-a'}; parent = {'a': b'parent-a', 'b': b'parent-b'}
        self.assertEqual(tool.baseline_state(values, parent, base, {'a':b'base-a','b':None,'c':None}), 'base')
        self.assertEqual(tool.baseline_state(values, parent, base, {'a':b'parent-a','b':b'parent-b','c':None}), 'parent')
        self.assertEqual(tool.baseline_state(values, parent, base, values), 'applied')
        for before in ({'a':b'parent-a','b':None,'c':None}, {'a':b'new-a','b':b'parent-b','c':None}):
            with self.assertRaises(tool.OverlayError):
                tool.baseline_state(values, parent, base, before)


class LifecycleGILTests(unittest.TestCase):
    def test_actual_interpreter_detach_and_close_once_with_bounded_old_order_reproduction(self):
        header = ROOT / 'torch_npu/csrc/core/npu/EagerPrepareLifecycle.h'
        self.assertTrue(header.is_file(), 'native lifecycle/GIL guard is missing')
        source = ROOT / 'test/host_regions/eager_prepare_integration/lifecycle_fixture.cpp'
        client = r'''
import importlib.util,sys,threading,time
spec=importlib.util.spec_from_file_location('_r24_lifecycle_fixture',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.reset(False)
a=threading.Thread(target=m.close);a.start()
deadline=time.monotonic()+2
while m.snapshot()[0]!=1 and time.monotonic()<deadline:time.sleep(.001)
assert m.snapshot()[0]==1
b=threading.Thread(target=m.configure);b.start();time.sleep(.05);m.release()
a.join(2);b.join(2);assert not a.is_alive() and not b.is_alive()
assert m.snapshot()==(1,1) and m.nested()
m.close();assert m.snapshot()==(1,1)
m.reset(True);m.release()
for index in range(2):
 try:m.close()
 except RuntimeError as e:assert str(e)=='unique-cold-ack-failure'
 else:raise AssertionError('failure close returned success')
assert m.snapshot()==(1,0) and m.nested()
print('PASS real-GIL lock order, nested detach and no retry after failed cold ACK')
'''
        with tempfile.TemporaryDirectory(prefix='r24-real-gil-') as folder:
            build = Path(folder)
            suffix = sysconfig.get_config_var('EXT_SUFFIX')
            for broken in (True, False):
                output = build / ('broken' if broken else 'fixed')
                output.mkdir()
                extension = output / ('_r24_lifecycle_fixture' + suffix)
                if os.name == 'nt':
                    vc = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')
                    if not vc.is_file(): self.skipTest('MSVC C++ toolchain unavailable')
                    script = output / 'compile.cmd'
                    script.write_text('@echo off\ncall "' + str(vc) + '" -arch=x64 -host_arch=x64 >nul\n'
                        'if errorlevel 1 exit /b 1\ncl /nologo /std:c++17 /EHsc /W4 /WX /LD /MD ' +
                        ('/DR24_REPRODUCE_OLD_LOCK_ORDER ' if broken else '') + '/I"' + str(ROOT) + '" /I"' +
                        sysconfig.get_path('include') + '" "' + str(source) + '" /Fe:"' + str(extension) +
                        '" /link /LIBPATH:"' + str(Path(sys.base_prefix)/'libs') + '"\n', encoding='utf-8')
                    command = ['cmd','/d','/c',str(script)]
                else:
                    compiler = shutil.which('c++')
                    if compiler is None: self.skipTest('C++ compiler unavailable')
                    command = [compiler,'-std=c++17','-shared','-fPIC','-pthread','-I'+str(ROOT),
                        '-I'+sysconfig.get_path('include'),str(source),'-o',str(extension)]
                    if broken: command.insert(1,'-DR24_REPRODUCE_OLD_LOCK_ORDER')
                result = subprocess.run(command,cwd=output,capture_output=True,timeout=90)
                self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode('utf-8',errors='replace'))
                argv = [sys.executable,'-I','-B','-c',client,str(extension)]
                if broken:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        subprocess.run(argv,capture_output=True,timeout=3)
                else:
                    result = subprocess.run(argv,capture_output=True,timeout=8)
                    self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode('utf-8',errors='replace'))
                    self.assertIn(b'PASS real-GIL',result.stdout)
                    finalizer = r'''
import importlib.util,sys
spec=importlib.util.spec_from_file_location('_r24_lifecycle_fixture',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.reset(True);m.release()
try:m.close()
except RuntimeError as e:assert str(e)=='unique-cold-ack-failure'
else:raise AssertionError('close failure was hidden')
m.finalize()
raise AssertionError('failed Finalize returned')
'''
                    with self.assertRaises(subprocess.TimeoutExpired) as stopped:
                        subprocess.run([sys.executable,'-I','-B','-c',finalizer,str(extension)],capture_output=True,timeout=2)
                    stderr = stopped.exception.stderr or b''
                    self.assertIn(b'SDK_CALLBACK_CALLS=1',stderr)
                    self.assertIn(b'R24_EAGER_PREPARE_FINALIZE_FAILSTOP',stderr)
                    self.assertNotIn(b'ORIGINAL_FINALIZE_REACHED',stderr)


if __name__ == '__main__':
    unittest.main()
