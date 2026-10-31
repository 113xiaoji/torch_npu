"""Build an isolated CPU numerical SDK; do not mutate any existing SDK/source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ACL_SHA='bd3d6f1d27707b2b5e85ba750e8e6587c0013acf3c3b21382421b099d8019b00'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_body(source,name,body):
    found=re.search(r'\b'+re.escape(name)+r'\s*\([^)]*\)\s*\{',source)
    if found is None: raise ValueError('missing source definition: '+name)
    start=found.end()-1; end=start+1; depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[:start]+'{'+body+'}'+source[end:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--base-sdk',type=Path,required=True);p.add_argument('--torch',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    acl=a.source/'third_party/acl/libs/acl.cpp';assert sha(acl)==ACL_SHA,'unreviewed ACL stub source'
    source=acl.read_text()
    replacements={
      'aclrtCreateStream':'return R27StreamCreate(stream);',
      'aclrtCreateStreamWithConfig':'return R27StreamCreate(stream);',
      'aclrtDestroyStream':'return R27StreamDestroy(stream);',
      'aclrtDestroyStreamForce':'return R27StreamDestroy(stream);',
      'aclrtSynchronizeStream':'return R27StreamSync(stream);',
      'aclrtSynchronizeStreamWithTimeout':'return R27StreamSync(stream);',
      'aclrtSynchronizeDevice':'return R27DeviceSync();',
      'aclrtCreateEvent':'return R27EventCreate(event);',
      'aclrtCreateEventWithFlag':'return R27EventCreate(event);',
      'aclrtCreateEventExWithFlag':'return R27EventCreate(event);',
      'aclrtDestroyEvent':'return R27EventDestroy(event);',
      'aclrtRecordEvent':'return R27EventRecord(event,stream);',
      'aclrtStreamWaitEvent':'return R27EventWait(stream,event);',
      'aclrtSynchronizeEvent':'return R27EventSync(event);',
      'aclrtQueryEvent':'int value=0;auto result=R27EventQuery(event,&value);if(status)*status=static_cast<aclrtEventStatus>(value?0:1);else return 100001;return result;',
      'aclrtQueryEventStatus':'int value=0;auto result=R27EventQuery(event,&value);if(status)*status=static_cast<aclrtEventRecordedStatus>(value?1:0);else return 100001;return result;',
      'aclrtStreamQuery':'int value=0;auto result=R27StreamQuery(stream,&value);if(status)*status=static_cast<aclrtStreamStatus>(value);else return 100001;return result;',
      'aclrtResetEvent':'return R27EventRecord(event,stream);',
      'aclrtEventElapsedTime':'if(ms)*ms=0;return 100004;',
      'aclrtMemcpy':'return R27Copy(dst,destMax,src,count);',
      'aclrtMemcpyAsync':'return R27CopyAsync(dst,destMax,src,count,stream);',
      'aclrtMemcpyAsyncWithCondition':'return R27CopyAsync(dst,destMax,src,count,stream);',
      'aclrtFree':'auto result=R27DeviceSync();if(result)return result;std::free(devPtr);return 0;',
      'aclrtFreeHost':'auto result=R27DeviceSync();if(result)return result;std::free(hostPtr);return 0;',
      'aclrtMemset':'auto result=R27DeviceSync();if(result)return result;if(count>maxCount||(!devPtr&&count))return 100001;if(count)std::memset(devPtr,value,count);return 0;',
      'rtKernelLaunch':'return static_cast<rtError_t>(100004);',
    }
    for name,body in replacements.items():source=replace_body(source,name,body)
    # Relative source-only runtime header is made absolute in the generated file.
    source=source.replace('"../../../torch_npu/csrc/inductor/mlir/hacl_rt.h"',json.dumps(str(a.source/'torch_npu/csrc/inductor/mlir/hacl_rt.h')))
    source='#include "runtime.h"\n'+source
    (out/'acl_r27.cpp').write_text(source)
    legacy=(a.source/'third_party/acl/libs/acl_op_compiler.cpp').read_text()
    for name in ('aclopCompileAndExecute','aclopCompileAndExecuteV2','aclopExecuteV2'):
        legacy=replace_body(legacy,name,'return 100004;')
    (out/'acl_op_compiler_r27.cpp').write_text(legacy)
    for path in a.base_sdk.glob('*.so'):
        if path.name not in ('libascendcl.so','libacl_op_compiler.so','libopapi.so'):shutil.copyfile(path,out/path.name)
    common=['c++','-std=c++20','-O2','-fPIC','-shared','-pthread','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(HERE)]
    rtlink=['-L'+str(out),'-lr27_hostsim_runtime','-Wl,-rpath,$ORIGIN']
    commands=[common+[str(HERE/'runtime.cpp'),'-o',str(out/'libr27_hostsim_runtime.so')],
      common+['-I'+str(a.source/'third_party/acl/inc'),str(out/'acl_r27.cpp')]+rtlink+['-o',str(out/'libascendcl.so')],
      common+['-I'+str(a.source/'third_party/acl/inc'),str(out/'acl_op_compiler_r27.cpp'),'-o',str(out/'libacl_op_compiler.so')],
      common+['-I'+str(a.torch/'include'),'-I'+str(a.torch/'include/torch/csrc/api/include'),str(HERE/'opapi.cpp')]+rtlink+
      ['-L'+str(a.torch/'lib'),'-ltorch_cpu','-lc10','-Wl,-rpath,'+str(a.torch/'lib'),'-o',str(out/'libopapi.so')]]
    manifest=dict(schema=1,backend_kind='HOST_SIM_R27_CPU',numeric_supported=True,real_sdk_qualified=False,
                  numeric_scope='CPU emulation of the frozen DLRM inference operator closure; EmbeddingBag sum only, no scale_grad_by_freq or per-sample weights',
                  embedding_auxiliary_policy='Copy CPU auxiliary values only when shapes match; otherwise fill zero; auxiliary CANN semantics unqualified',
                  auxiliary_numeric_qualified=False,
                  source_pins={str(path):sha(path) for path in HERE.glob('*') if path.is_file()},acl_source_sha256=ACL_SHA,commands=commands)
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for index,command in enumerate(commands):
        with (out/('compile-'+str(index)+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('compile failed: '+str(out/('compile-'+str(index)+'.log')))
    manifest['outputs']={path.name:sha(path) for path in out.glob('*.so')}
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(passed=True,output=str(out))))


if __name__=='__main__':main()
