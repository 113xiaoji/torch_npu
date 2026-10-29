"""R25 host simulation identity gates; no Torch, SDK, or remote runtime required."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import hashlib
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_r24_integration_tests', ROOT / 'test/host_regions/test_eager_prepare_integration.py')
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


class HostSimProfileTests(unittest.TestCase):
    def test_actual_undefined_optional_snapshot_and_conversion_preserve_null_sdk_argument(self):
        payload = ROOT / 'patches/eager_prepare_r25/op-plugin/op_plugin/utils'
        source = (payload / 'op_api_common.cpp').read_text(encoding='utf-8')
        def function(signature):
            start=source.index(signature);brace=source.index('{',start);depth=1;end=brace+1
            while depth:
                depth+=(source[end]=='{')-(source[end]=='}');end+=1
            return source[start:end]
        header=(payload / 'op_api_eager_prepare_r24.h').read_text(encoding='utf-8')
        start=header.index('struct EagerSnapshotter');end=header.index('\n};',start)+3
        owned=(payload / 'op_api_owned_r22.h').read_text(encoding='utf-8')
        begin=owned.index('    template<class T, class Converted');finish=owned.index('    TORCH_NPU_API void Release(',begin)
        values=dict(SNAPSHOT_FUNCTION=function('OwnedTensor SnapshotEagerOwnedTensor('),
            EAGER_SNAPSHOTTER=header[start:end],OPTIONAL_CONVERT=owned[begin:finish],
            TENSOR_CONVERT=function('aclTensor* OwnedConvertPolicy::Convert(const OwnedTensor& value) const'))
        template=(ROOT / 'test/host_regions/eager_prepare_integration/undefined_optional_fixture.cpp.in').read_text(encoding='utf-8')
        for key,value in values.items():template=template.replace('/*'+key+'*/',value)
        with tempfile.TemporaryDirectory(prefix='r25-undefined-optional-') as folder:
            path=Path(folder)/'undefined_optional.cpp';path.write_text(template,encoding='utf-8')
            integration.NativeOwnedJobTests().compile_run(path)

    def test_actual_domain_template_compiles_only_exact_embedding_signature(self):
        payload = ROOT / 'patches/eager_prepare_r25/op-plugin/op_plugin/utils/op_api_eager_prepare_r24.h'
        header = payload.read_text(encoding='utf-8')
        start = header.index('template<class Tuple')
        end = header.index('\ntemplate<class... Args>', start)
        template = (ROOT / 'test/host_regions/eager_prepare_integration/embedding_signature_fixture.cpp.in').read_text(encoding='utf-8')
        template = template.replace('/*DOMAIN*/', header[start:end])
        for key, name, api in (
            ('DYNAMIC_CALL', 'AddRmsNormDynamicMxQuantKernelNpuOpApi.cpp', 'aclnnAddRmsNormDynamicMxQuant'),
            ('QUANT_CALL', 'AddRmsNormQuantKernelOpApi.cpp', 'aclnnAddRmsNormQuant'),
        ):
            path = ROOT / 'third_party/op-plugin/op_plugin/ops/opapi' / name
            if not path.is_file():
                self.skipTest('pinned op-plugin sources required for the exact failing callsites')
            source = path.read_text(encoding='utf-8')
            begin = source.index('EXEC_NPU_CMD(' + api + ',')
            call = source[begin:source.index(');', begin) + 2]
            template = template.replace('/*' + key + '*/', call)
        with tempfile.TemporaryDirectory(prefix='r25-embedding-signature-') as folder:
            path = Path(folder) / 'embedding_signature.cpp'; path.write_text(template, encoding='utf-8')
            integration.NativeOwnedJobTests().compile_run(path)

    def test_cmake_build_profile_binds_exact_provider_and_is_off_by_default(self):
        cmake = shutil.which('cmake')
        if cmake is None:
            self.skipTest('CMake unavailable')
        with tempfile.TemporaryDirectory(prefix='r25-build-profile-') as folder:
            root = Path(folder); source = root / 'source'; source.mkdir()
            (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.18)\nproject(Profile NONE)\ninclude("' +
                (ROOT / 'cmake/EagerPrepareHostSim.cmake').as_posix() + '")\n', encoding='utf-8')
            def configure(name, *args):
                return subprocess.run([cmake, '-S', str(source), '-B', str(root / name), *args], capture_output=True, timeout=20)
            result = configure('off'); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / 'off/r25_hostsim/EagerPrepareHostSimProfile.h').exists())
            result = configure('missing', '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM=ON')
            self.assertNotEqual(result.returncode, 0)
            provider = root / 'isolated libopapi.so'; provider.write_bytes(b'exact-provider-profile')
            result = configure('on', '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM=ON', '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM_LIBRARY=' + str(provider))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            header = (root / 'on/r25_hostsim/EagerPrepareHostSimProfile.h').read_text(encoding='utf-8')
            self.assertIn(hashlib.sha256(provider.read_bytes()).hexdigest(), header)
            self.assertIn(provider.resolve().as_posix(), header)
            self.assertIn('22ULL', header)

    def test_order_evidence_detects_reordering_and_schema_changes(self):
        header = ROOT / 'torch_npu/csrc/core/npu/EagerPrepareOrder.h'
        self.assertTrue(header.is_file(), 'native bounded FIFO order evidence is missing')
        source = r'''
#include "torch_npu/csrc/core/npu/EagerPrepareOrder.h"
#include <cassert>
int main() {
 c10_npu::eager_prepare::SequenceOrderEvidence a,b,c,d;
 a.Record(0,"aclnnEmbeddingBag"); a.Record(1,"aclnnRelu");
 b.Record(0,"aclnnEmbeddingBag"); b.Record(1,"aclnnRelu");
 c.Record(1,"aclnnRelu"); c.Record(0,"aclnnEmbeddingBag");
 d.Record(0,"aclnnEmbeddingBag"); d.Record(1,"aclnnMm");
 assert(a.count==2 && a.ordered && a.hash==b.hash);
 assert(!c.ordered && c.hash!=a.hash); assert(d.hash!=a.hash);
}
'''
        with tempfile.TemporaryDirectory(prefix='r25-order-') as folder:
            path = Path(folder) / 'order.cpp'; path.write_text(source, encoding='utf-8')
            integration.NativeOwnedJobTests().compile_run(path)

    def test_explicit_identity_gate_and_simulation_labels(self):
        header = ROOT / 'patches/eager_prepare_r25/op-plugin/op_plugin/utils/op_api_eager_hostsim_r25.h'
        self.assertTrue(header.is_file(), 'isolated R25 HostSim identity gate is missing')
        source = r'''
#include "patches/eager_prepare_r25/op-plugin/op_plugin/utils/op_api_eager_hostsim_r25.h"
#include "torch_npu/csrc/core/npu/EagerPrepareBackend.h"
#include <cassert>
#include <cstring>
int abi_value=1; bool cache_enabled=false; const char* backend="HOST_SIM_R25";
int abi(){return abi_value;} const char* kind(){return backend;}
bool cache(const char*){return cache_enabled;} void placeholder(){}
int main(){
  const char* missing=""; const char* rejected=""; int checked=0;
  auto resolve=[&](const char* name)->void* {
    if(std::strcmp(name,missing)==0)return nullptr;
    if(std::strcmp(name,"R25HostSimAbi")==0)return reinterpret_cast<void*>(&abi);
    if(std::strcmp(name,"R25HostSimBackendKind")==0)return reinterpret_cast<void*>(&kind);
    if(std::strcmp(name,"CanUsePTACache")==0)return reinterpret_cast<void*>(&cache);
    return reinterpret_cast<void*>(&placeholder);
  };
  auto verify=[&](const char* name,void*){++checked;return std::strcmp(name,rejected)!=0;};
  auto accepted=[&](const char* enabled){return hdb_opapi::VerifyHostSimIdentity(enabled,resolve,verify);};
  assert(!accepted(nullptr)); assert(!accepted("0")); assert(!accepted("true")); assert(checked==0);
  assert(accepted("1")); assert(checked>20);
  abi_value=2;assert(!accepted("1"));abi_value=1;
  backend="CANN9.2";assert(!accepted("1"));backend="HOST_SIM_R25";
  cache_enabled=true;assert(!accepted("1"));cache_enabled=false;
  missing="aclDestroyAclOpExecutor";assert(!accepted("1"));missing="";
  rejected="aclnnEmbeddingBag";assert(!accepted("1"));rejected="";
  assert(accepted("1"));
#ifdef TORCH_NPU_EAGER_PREPARE_HOST_SIM
  assert(c10_npu::eager_prepare::BuildUsesHostSimulation());
  assert(std::strcmp(c10_npu::eager_prepare::BackendName(),"host_sim")==0);
#else
  assert(!c10_npu::eager_prepare::BuildUsesHostSimulation());
  assert(std::strcmp(c10_npu::eager_prepare::BackendName(),"real_sdk")==0);
#endif
}
'''
        runner = integration.NativeOwnedJobTests()
        with tempfile.TemporaryDirectory(prefix='r25-profile-') as folder:
            path = Path(folder) / 'profile.cpp'
            for simulated in (False, True):
                prefix = '#define TORCH_NPU_EAGER_PREPARE_HOST_SIM 1\n' if simulated else ''
                path.write_text(prefix + source, encoding='utf-8')
                runner.compile_run(path)

    def test_versioned_overlay_keeps_real_profile_and_r24_frozen(self):
        path = ROOT / 'tools/host_regions/apply_eager_prepare_overlay_r25.py'
        self.assertTrue(path.is_file(), 'versioned R25 installer is missing')
        spec = importlib.util.spec_from_file_location('_r25_overlay', path)
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        values = tool.sources(ROOT)
        self.assertEqual(len(values), 9)
        r24 = integration.R24OverlayTests().module()
        old = r24.sources(ROOT)
        self.assertEqual(values['op_plugin/utils/op_api_sdk_profile_r22.h'], old['op_plugin/utils/op_api_sdk_profile_r22.h'])
        self.assertEqual(tool.baseline_state(values, old, {}, {name: old.get(name) for name in values}), 'parent')
        mixed = dict(values); mixed['op_plugin/utils/op_api_common.cpp'] = old['op_plugin/utils/op_api_common.cpp']
        with self.assertRaises(tool.OverlayError):
            tool.baseline_state(values, old, {}, mixed)


if __name__ == '__main__':
    unittest.main()
