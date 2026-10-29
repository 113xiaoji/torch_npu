"""Verify actual original DLRM entry coverage in a pinned NoOp PTA build.

This is not a device-parallel or performance runner. It never reads NoOp values.
"""
import argparse
import ctypes
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def validate_shadow(snapshot,forwards):
    require(type(forwards) is int and forwards>0,'positive forward count required')
    require(snapshot.get('schema')==1 and snapshot.get('backend')=='shadow','native shadow identity missing')
    for flag in ('device_parallel_qualified','real_sdk_qualified','coverage_incomplete'):
        require(snapshot.get(flag) is False,'unexpected native qualification or incomplete counters: '+flag)
    require(snapshot.get('active_calls')==0 and snapshot.get('foreign_thread_calls')==0
        and snapshot.get('first_error') is None,'native entries did not close cleanly')
    require(snapshot.get('closure',{}).get('passed') is True
        and snapshot['closure'].get('admission_closed') is True,'native shadow close missing')
    ops=snapshot.get('operator_counts',{})
    embeddings=[ops[name] for name in ('aten::_embedding_bag','aten::_embedding_bag_forward_only') if name in ops]
    require(sum(r['completed'] for r in embeddings)==26*forwards,'original 26 Embedding calls not observed')
    require(sum(r['outputs'] for r in embeddings)==104*forwards,'four Embedding outputs not observed')
    for name,count in [('aten::cat',2),('aten::bmm',1),('aten::index.Tensor',1)]:
        row=ops.get(name,{})
        require(row.get('entered')==row.get('completed')==count*forwards and row.get('failed')==0,
            'original feature pipeline entry missing or duplicated: '+name)
    require(ops['aten::cat']['inputs']==29*forwards,'Cat input TensorList count differs')
    require(all(r['entered']==r['completed'] and r['failed']==0 for r in ops.values()),'a native entry failed')
    return dict(passed=True,forwards=forwards,embedding_calls=26*forwards,cat_calls=2*forwards,
        bmm_calls=forwards,index_calls=forwards,device_parallel_qualified=False,
        full_entry_coverage_qualified=False,numerical_claim=False,performance_claim=False)


def mapped(pins):
    maps=Path('/proc/self/maps').read_text()
    result=[]
    for name,pin in pins.items():
        found=set()
        for line in maps.splitlines():
            fields=line.split(None,5)
            if len(fields)==6 and fields[5].startswith('/') and Path(fields[5]).name==name:
                found.add(str(Path(fields[5]).resolve(strict=True)))
        require(found=={str(Path(pin['path']).resolve(strict=True))},'loaded library identity differs: '+name)
        require(sha(pin['path'])==pin['sha256'],'loaded artifact bytes changed')
        result.append(dict(name=name,**pin))
    return result


def run(config,output,report):
    require(sys.platform=='linux' and sys.version_info[:3]==(3,14,6),'target Linux Python3.14.6 required')
    require(os.environ.get('PYTHON_GIL') is None and sys._is_gil_enabled() is False,'natural FT required')
    require(os.environ.get('QWEN_ACL_HOST_SIM')=='1','explicit host simulation required')
    require(config['compute_backend']=='host_sim_noop','this runner supports NoOp metadata only')
    require(type(config['forwards']) is int and 1<=config['forwards']<=64,'bounded forward count required')
    require(len(config['cpus'])==len(set(config['cpus']))==8,'eight-CPU budget required')
    require(set(os.sched_getaffinity(0))==set(config['cpus']),'externally bound CPU budget differs')
    for path,digest in config['artifact_pins'].items():require(sha(path)==digest,'artifact pin mismatch: '+path)
    for path,digest in config['source_pins'].items():require(sha(path)==digest,'source pin mismatch: '+path)
    package=Path(config['torch_npu_package']).resolve(strict=True)
    require(os.environ.get('PYTHONPATH')==str(package.parent),'candidate-only PYTHONPATH required')
    sys.path.insert(0,str(package.parent))
    # The native interpreter's NumPy wins; old model dependencies remain fallback.
    sys.path.append(config['legacy_deps'])
    import torch
    import numpy
    import torch_npu
    from torch_npu import _eager_parallel
    require(Path(torch_npu.__file__).resolve().parent==package,'import escaped new candidate')
    require(torch.version.git_version=='cf30153c4c131c8164ee7798e5022d810682e2cb'
        and numpy.__version__=='2.4.4' and sys._is_gil_enabled() is False,'target runtime identity changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference_path=Path(__file__).with_name('reference.py')
    require(str(reference_path) in config['source_pins'],'reference source must be pinned')
    spec=importlib.util.spec_from_file_location('r26_shadow_reference',reference_path)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    model,inputs,provenance=reference.load_original_model(config['original_source'])
    report['model']=provenance
    torch.npu.set_device(0)
    model=model.to('npu:0')
    inputs=inputs[0].to('npu:0'),[v.to('npu:0') for v in inputs[1]],[v.to('npu:0') for v in inputs[2]]
    torch.npu.synchronize()
    library=ctypes.CDLL(config['libraries']['libopapi.so']['path'])
    library.R25HostSimBackendKind.restype=ctypes.c_char_p
    library.R25HostSimDiagnosticsJson.restype=ctypes.c_char_p
    require(library.R25HostSimBackendKind()==b'HOST_SIM_R25','unexpected simulation provider')
    sdk=lambda:json.loads(library.R25HostSimDiagnosticsJson().decode())
    stages=[]
    with torch.inference_mode():
        for enabled in (False,True):
            before=sdk()
            if enabled:
                context=_eager_parallel.scope(mode='shadow')
            else:
                _eager_parallel.configure(enabled=False,mode='shadow')
                context=nullcontext()
            with context:
                for _ in range(config['forwards']):
                    result=model(*inputs)
                    torch.npu.synchronize()
                after=_eager_parallel.snapshot()
            closed=_eager_parallel.snapshot()
            metadata=dict(shape=list(result.shape),dtype=str(result.dtype),device=str(result.device))
            require(metadata==dict(shape=[128,1],dtype='torch.float32',device='npu:0'),'output metadata differs')
            after_sdk=sdk()
            operation=after_sdk['operations']['aclnnEmbeddingBag']
            old=before['operations']['aclnnEmbeddingBag']
            require(operation['prepared']-old['prepared']==operation['executed']-old['executed']==26*config['forwards'],
                'actual SDK Embedding count differs')
            require(after_sdk['errors']==0 and after_sdk['live_descriptors']==after_sdk['live_executors']==0
                and after_sdk['huge_pending']==after_sdk['huge_active_threads']==0,'SDK resources not closed')
            if enabled:gate=validate_shadow(closed,config['forwards'])
            else:
                require(after['entered']==after['completed']==0 and not after['enabled'],'disabled native gate activated')
                gate=dict(passed=True,disabled=True)
            stages.append(dict(enabled=enabled,native=closed,sdk_before=before,sdk_after=after_sdk,
                metadata=metadata,gate=gate))
    report.update(stages=stages,mapping=mapped(config['libraries']),gil_enabled=sys._is_gil_enabled(),
        native_module=str(torch_npu._C.__file__),compute_backend='host_sim_noop',schedule_backend='shadow',
        native_header_sha256=config.get('native_header_sha256'),real_device_numerical_claim=False,
        device_overlap_claim=False,performance_claim=False,full_entry_coverage_qualified=False)
    require(report['gil_enabled'] is False,'GIL changed during actual model execution')
    for path,digest in config['artifact_pins'].items():require(sha(path)==digest,'artifact changed during run')
    report['passed']=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());report=dict(passed=False,worker_sha256=sha(__file__))
    dump(args.output/'config.json',config)
    try:run(config,args.output,report)
    except BaseException as error:
        report['passed']=False
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        dump(args.output/'result.json',report);raise
    dump(args.output/'result.json',report)
    print(json.dumps(dict(passed=True,stages=len(report['stages']),forwards_per_stage=config['forwards'],
        real_device_numerical_claim=False,device_overlap_claim=False)))


if __name__=='__main__':main()
