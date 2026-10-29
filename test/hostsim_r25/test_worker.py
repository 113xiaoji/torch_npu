"""Pure host fixtures only; no torch/NPU import is used by these tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import subprocess
import sys
import ast
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


class WorkerTests(unittest.TestCase):
    def setUp(self):
        source=ROOT/'benchmarks/hostsim_r25/worker.py'
        self.assertTrue(source.exists(),'dedicated isolated worker must exist')
        import sys
        sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location('_hostsim_r25_worker',source)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_actual_worker_configure_and_forward_share_inference_caller_scope(self):
        # Execute the actual worker cohort's try body, stopping at the first
        # original forward. No Torch, NPU, imported model, or SDK is involved.
        tree=ast.parse((ROOT/'benchmarks/hostsim_r25/worker.py').read_text(encoding='utf-8'))
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        cohort=next(node for node in function.body if isinstance(node,ast.Try))
        body=ast.Module(body=cohort.body,type_ignores=[])
        code=compile(ast.fix_missing_locations(body),'<actual R25 worker cohort>','exec')
        events=[];state=dict(inference=False,scope=None)
        class ForwardReached(Exception):pass
        class InferenceMode:
            def __enter__(self):
                state.update(inference=True,scope=self);events.append(('enter',self))
            def __exit__(self,*args):
                events.append(('exit',self));state.update(inference=False,scope=None)
        def configure(**kwargs):
            if not state['inference']:raise RuntimeError('native regions require no-grad caller')
            events.append(('configure',state['scope']));return {}
        def model(*inputs):
            if not state['inference']:raise RuntimeError('forward is outside inference mode')
            events.append(('forward',state['scope']));raise ForwardReached()
        values=dict(torch=SimpleNamespace(inference_mode=InferenceMode),
            _eager_prepare=SimpleNamespace(configure=configure),time=SimpleNamespace(perf_counter_ns=lambda:1),
            row=dict(workers=1,diagnostic_trace=True,warmup=1),report={},
            c=SimpleNamespace(sim_identity=lambda snapshot:None),model=model,inputs=())
        with self.assertRaises(ForwardReached):exec(code,values)
        self.assertEqual([name for name,_ in events],['enter','configure','forward','exit'])
        self.assertTrue(all(scope is events[0][1] for _,scope in events))
        self.assertFalse(state['inference'])

    def test_native_dependency_precedence_with_duplicate_legacy_startup_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            candidate=root/'candidate';native=root/'python314t-site-packages'
            source=root/'original-dlrm';deps=root/'legacy-deps-ft'
            for path in (candidate,native,source,deps):path.mkdir()
            (candidate/'torch_npu.py').write_text('origin="isolated candidate"\n')
            (native/'torch_npu.py').write_text('raise ImportError("old installed candidate")\n')
            (native/'numpy.py').write_text('__version__="2.4.4"\norigin="native ABI"\n')
            (deps/'numpy.py').write_text('raise ImportError("cpython314 extension cannot load in 314t")\n')
            (deps/'original_model_dependency.py').write_text('origin="pinned fallback"\n')
            runtime=dict(torch_npu_package=str(candidate/'torch_npu'),pythonpath=[str(candidate)])
            manifest=dict(source_dir=str(source),import_deps=str(deps))
            # Simulate the broken worker's duplicated legacy PYTHONPATH entries.
            initial=[str(root),str(candidate),str(deps),str(native),str(deps),str(source)]
            paths=self.m.import_search_path(runtime,manifest,initial)
            self.assertEqual(paths[0],str(candidate))
            self.assertLess(paths.index(str(native)),paths.index(str(deps)))
            self.assertEqual(paths.count(str(deps)),1)
            self.assertEqual(initial.count(str(deps)),2,'helper must not mutate its input')
            script='''import json,sys
sys.path[:]=json.loads(sys.argv[1])
import numpy,torch_npu,original_model_dependency
assert numpy.__version__=="2.4.4" and numpy.origin=="native ABI"
assert torch_npu.origin=="isolated candidate"
assert original_model_dependency.origin=="pinned fallback"
'''
            result=subprocess.run([sys.executable,'-I','-B','-c',script,json.dumps(paths)],capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,(result.stdout+result.stderr).decode(errors='replace'))

    def test_import_manifest_dependency_path_is_explicitly_rejected_in_runtime_pythonpath(self):
        runtime=dict(torch_npu_package='/candidate/torch_npu',pythonpath=['/candidate','/original/deps-ft'])
        manifest=dict(source_dir='/original/dlrm',import_deps='/original/deps-ft')
        with self.assertRaisesRegex(ValueError,'manifest import_deps'):
            self.m.import_search_path(runtime,manifest,['/native/site-packages'])

    def test_loaded_numpy_artifact_requires_the_pinned_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();native=root/'native';legacy=root/'legacy'
            native.mkdir();legacy.mkdir()
            path=native/'_multiarray_umath.cpython-314t.so';path.write_bytes(b'native FT extension')
            pins={str(path):self.m.c.sha(path)}
            result=self.m.loaded_artifact(SimpleNamespace(__file__=str(path)),pins,'NumPy extension')
            self.assertEqual(result,dict(path=str(path),sha256=pins[str(path)]))
            foreign=legacy/path.name;foreign.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError,'pinned path'):
                self.m.loaded_artifact(SimpleNamespace(__file__=str(foreign)),pins,'NumPy extension')
            path.write_bytes(b'changed extension')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                self.m.loaded_artifact(SimpleNamespace(__file__=str(path)),pins,'NumPy extension')

    def test_manifest_fallback_snapshot_still_verifies_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';deps=root/'deps'
            source.mkdir();deps.mkdir()
            (source/'dlrm.py').write_bytes(b'original source')
            (deps/'numpy.py').write_bytes(b'legacy numpy snapshot remains pinned')
            manifest=dict(asset_identity=dict(source_sha256=self.m.c.SOURCE_SHA),
                weight_verification=dict(initialization=dict(state_sha256=self.m.c.STATE_SHA)),
                input_spec=dict(tensor_tree_sha256=self.m.c.INPUT_SHA,batch_size=128,sparse_dtype='int64',indices_per_bag=2),
                source_dir=str(source),source_files={'dlrm.py':self.m.c.sha(source/'dlrm.py')},
                import_deps=str(deps),import_deps_files={'numpy.py':self.m.c.sha(deps/'numpy.py')})
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            pin=dict(path=str(path),sha256=self.m.c.sha(path))
            # Only the Linux lexical-path check is replaced for this local host
            # fixture; actual manifest/source/dependency SHA comparisons run.
            def verify(value,label):
                self.m.c.require(self.m.c.sha(value['path'])==value['sha256'],label+' SHA mismatch')
            with patch.object(self.m.c,'verify_pin',side_effect=verify):
                _,verified=self.m.prepare_manifest(pin)
                self.assertEqual(verified['import_deps']['verified_files'],1)
                (deps/'numpy.py').write_bytes(b'tampered legacy snapshot')
                with self.assertRaisesRegex(ValueError,'pinned DLRM import source differs'):
                    self.m.prepare_manifest(pin)

    def test_mappings_deduplicate_elf_segments_but_not_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'libopapi.so';p.write_bytes(b'newsim')
            maps='100-200 r--p 0 00:00 1 '+str(p)+'\n200-300 r-xp 0 00:00 1 '+str(p)+'\n'
            rows=self.m.mapped_libraries(maps,{'libopapi.so'})
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['sha256'],self.m.c.sha(p))
            with self.assertRaises(ValueError):self.m.mapped_libraries(maps+'300-400 r-xp 0 00:00 1 '+str(p)+' (deleted)\n',{'libopapi.so'})

    def test_result_metadata_never_reads_or_hashes_stub_output(self):
        class Output:
            shape=(128,1);dtype='torch.float32';device='npu:0'
            def cpu(self):raise AssertionError('NoOp output is not a numerical golden')
        self.assertEqual(self.m.output_metadata(Output()),dict(shape=[128,1],dtype='torch.float32',device='npu:0',numeric_validated=False))

    def test_failure_cleanup_preserves_first_error(self):
        events=[]
        def bad():events.append('close');raise RuntimeError('second cleanup error')
        error=ValueError('first forward error')
        result=self.m.cleanup_preserving_error(error,bad)
        self.assertIs(result[0],error)
        self.assertEqual(events,['close'])
        self.assertIn('second cleanup error',result[1])


if __name__=='__main__':unittest.main()
