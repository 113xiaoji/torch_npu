"""One isolated actual DLRM host-sim process; no installation or runtime edits."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import threading
import time
import traceback

import contracts as c


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def load_helper(pin,name):
    c.verify_pin(pin,name)
    spec=importlib.util.spec_from_file_location(name,pin['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def mapped_libraries(text,names):
    paths=set()
    for line in text.splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not Path(fields[5]).is_absolute():
            continue
        raw=fields[5]
        if Path(raw.removesuffix(' (deleted)')).name in names:
            c.require(not raw.endswith(' (deleted)'),'deleted simulator mapping')
            paths.add(raw)
    return [dict(name=Path(p).name,path=str(Path(p).resolve(strict=True)),sha256=c.sha(p)) for p in sorted(paths)]


def output_metadata(output):
    result=dict(shape=list(output.shape),dtype=str(output.dtype),device=str(output.device),numeric_validated=False)
    c.require(result['shape']==[128,1] and result['dtype']=='torch.float32' and result['device']=='npu:0',
              'original DLRM host-sim output shape/dtype/device mismatch')
    return result


def cleanup_preserving_error(first,cleanup):
    secondary=None
    try:cleanup()
    except BaseException as error:
        if first is None:first=error
        else:secondary=repr(error)
    return first,secondary


def sdk_api(path):
    lib=ctypes.CDLL(path)
    lib.R25HostSimDiagnosticsJson.argtypes=[];lib.R25HostSimDiagnosticsJson.restype=ctypes.c_char_p
    lib.R25HostSimSetTracing.argtypes=[ctypes.c_bool];lib.R25HostSimSetTracing.restype=ctypes.c_int
    return lib


def sdk_snapshot(lib):
    return json.loads(lib.R25HostSimDiagnosticsJson().decode('utf-8'))


def thread_affinities(cpus):
    rows=[]
    for task in sorted(Path('/proc/self/task').iterdir()):
        try:mask=sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:continue
        c.require(set(mask)<=set(cpus),'process thread escaped eight-CPU budget')
        rows.append(dict(tid=int(task.name),cpus=mask))
    return rows


def prepare_manifest(pin):
    c.verify_pin(pin,'DLRM asset manifest')
    manifest=json.loads(Path(pin['path']).read_text())
    c.require(manifest['asset_identity']['source_sha256']==c.SOURCE_SHA
        and manifest['weight_verification']['initialization']['state_sha256']==c.STATE_SHA
        and manifest['input_spec']['tensor_tree_sha256']==c.INPUT_SHA
        and manifest['input_spec']['batch_size']==128 and manifest['input_spec']['sparse_dtype']=='int64'
        and manifest['input_spec']['indices_per_bag']==2,'original DLRM asset identity differs')
    verified={}
    for directory,files in [('source_dir','source_files'),('import_deps','import_deps_files')]:
        root=Path(manifest[directory]).resolve(strict=True)
        for relative,digest in manifest[files].items():
            path=(root/relative).resolve(strict=True)
            c.require(path.is_relative_to(root) and c.sha(path)==digest,'pinned DLRM import source differs: '+relative)
        verified[directory]=dict(path=str(root),verified_files=len(manifest[files]))
    verified['import_deps']['scope']='complete historical snapshot integrity; fallback imports only'
    for name,value in manifest.get('runtime_dependencies',{}).items():c.verify_pin(value,'DLRM runtime dependency '+name)
    return manifest,verified


def import_search_path(runtime,manifest,current):
    """Candidate first, interpreter packages next, original model imports last."""
    key=lambda value:Path(value).resolve()
    candidate=str(Path(runtime['torch_npu_package']).parent)
    source=str(Path(manifest['source_dir']).resolve())
    deps=str(Path(manifest['import_deps']).resolve())
    c.require(all(key(value)!=key(deps) for value in runtime['pythonpath']),
        'manifest import_deps must remain fallback, never runtime pythonpath')
    c.require([key(value) for value in runtime['pythonpath']]==[key(candidate)],
        'candidate-only pythonpath required; native interpreter dependencies take precedence')
    delayed={key(candidate),key(source),key(deps)}
    paths=[candidate]+[value for value in current if key(value) not in delayed]+[source,deps]
    result=[];seen=set()
    for value in paths:
        identity=key(value)
        if identity not in seen:result.append(value);seen.add(identity)
    return result


def loaded_artifact(module,pins,label):
    path=Path(module.__file__).resolve(strict=True)
    matches=[digest for name,digest in pins.items() if Path(name).resolve()==path]
    c.require(len(matches)==1,label+' actual import must match exactly one pinned path')
    digest=c.sha(path)
    c.require(digest==matches[0],label+' actual import SHA mismatch')
    return dict(path=str(path),sha256=digest)


def run(config,row,output,report):
    runtime=next(r for r in config['runtimes'] if r['name']==row['runtime'])
    c.require(platform.system()=='Linux' and sys.version_info[:3]==(3,14,6),'native Linux CPython 3.14.6 required')
    c.require(os.environ.get('PYTHON_GIL') is None,'PYTHON_GIL override forbidden')
    c.require(sys._is_gil_enabled() is runtime['expected_gil'],'natural interpreter GIL differs before imports')
    c.require(Path(sys.executable).absolute()==Path(runtime['python']).absolute(),'lexical interpreter path differs')
    os.sched_setaffinity(0,set(config['cpus']))
    c.require(sorted(os.sched_getaffinity(0))==sorted(config['cpus']),'worker CPU binding differs')
    for path,digest in runtime['artifact_pins'].items():c.verify_pin(dict(path=path,sha256=digest),'candidate package artifact')
    for name,pin in config['sdk']['libraries'].items():c.verify_pin(pin,name)
    for path,digest in config['sdk']['source_pins'].items():c.verify_pin(dict(path=path,sha256=digest),'SDK source')
    manifest,verified=prepare_manifest(runtime['model_manifest'])
    legacy=load_helper(config['helpers']['r22'],'_r25_pinned_inference_helpers')
    sys.path[:]=import_search_path(runtime,manifest,sys.path)
    report['import_resolution']=dict(policy='candidate; native interpreter paths; original source/dependency fallback',
        search_path=list(sys.path),historical_dependency_snapshot_is_import_closure=False)
    import torch
    import torch_npu
    import numpy as np
    from numpy._core import _multiarray_umath
    from torch_npu import _eager_prepare
    c.require(sys._is_gil_enabled() is runtime['expected_gil'],'native package import changed natural GIL')
    c.require(torch.version.git_version==c.TORCH_GIT and np.__version__=='2.4.4','original torch/NumPy source identity differs')
    numpy_artifacts=dict(package=loaded_artifact(np,runtime['artifact_pins'],'NumPy package'),
        extension=loaded_artifact(_multiarray_umath,runtime['artifact_pins'],'NumPy native extension'))
    package=Path(torch_npu.__file__).resolve().parent
    c.require(package==Path(runtime['torch_npu_package']).resolve(),'candidate package import fell back to installed package')
    c.require(Path(torch_npu._C.__file__).resolve().parent==package,'candidate native Python extension import escaped isolated package')
    for path,digest in runtime['artifact_pins'].items():c.require(c.sha(path)==digest,'artifact changed across import')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    c.require(torch.get_num_threads()==torch.get_num_interop_threads()==1,'Torch thread budget differs')
    c.require(sys.gettrace() is None and sys.getprofile() is None,'Python profiler/trace active')
    official=legacy.load_inference_module(manifest['source_dir'],report)
    for relative in legacy.SOURCE_SHA:
        loaded=sys.modules.get(relative[:-3].replace('/','.'))
        c.require(loaded is not None and Path(loaded.__file__).resolve()==Path(manifest['source_dir'])/relative,
                  'official DLRM dependency imported outside original pinned tree')
    c.require(official.ext_dist.my_size<=1,'original single-process DLRM required')
    np.random.seed(20260917);torch.manual_seed(20260917)
    model=official.DLRM_Net(m_spa=16,ln_emb=np.array([1024]*26),ln_bot=np.array([13,64,32,16]),
        ln_top=np.array([367,64,32,1]),arch_interaction_op='dot',arch_interaction_itself=False,
        sigmoid_top=2,ndevices=1).eval()
    model.requires_grad_(False)
    generator=torch.Generator(device='cpu').manual_seed(20260918)
    dense=torch.rand((128,13),generator=generator,dtype=torch.float32)
    offsets=[torch.arange(0,256,2,dtype=torch.int64) for _ in range(26)]
    indices=[torch.randint(1024,(256,),generator=generator,dtype=torch.int64) for _ in range(26)]
    inputs=dense,offsets,indices
    state=legacy.tensor_tree_sha256(dict(model.state_dict()),torch);input_sha=legacy.tensor_tree_sha256(inputs,torch)
    c.require(state==c.STATE_SHA and input_sha==c.INPUT_SHA,'original state/input hash differs')
    c.require(sum(p.numel() for p in model.parameters())==455153,'original parameter count differs')
    c.require(all(layer.bias is not None for layer in model.modules() if isinstance(layer,torch.nn.Linear)),
              'original MLP bias was removed')
    sparse=legacy.sparse_proof(model,torch);eager=legacy.eager_proof(model,torch)
    torch.npu.set_device(0)
    model=model.to('npu:0')
    inputs=dense.to('npu:0'),[v.to('npu:0') for v in offsets],[v.to('npu:0') for v in indices]
    torch.npu.synchronize()
    legacy.device_proof(model,inputs);legacy.sparse_proof(model,torch)
    # Never read back NoOp output, model, or input values as numerical evidence.
    lib=sdk_api(config['sdk']['libraries']['libopapi.so']['path'])
    c.require(lib.R25HostSimSetTracing(row['diagnostic_trace'])==0,'SIM trace toggle failed')
    c.validate_sdk(sdk_snapshot(lib))
    configured=False;first=None;secondary=None
    try:
        with torch.inference_mode():
            startup=time.perf_counter_ns()
            configured=True
            initial=_eager_prepare.configure(enabled=row['workers']>0,workers=max(1,row['workers']),capacity=8,
                byte_limit=536870912,allowed_ops=('aclnnEmbeddingBag','aclnnAdd','aclnnRelu','aclnnMm'),
                diagnostic_trace=row['diagnostic_trace'],trace_limit=256)
            startup=time.perf_counter_ns()-startup
            c.sim_identity(initial)
            report['pool_startup_ns']=startup
            report['pool_startup_included_in_samples']=False
            output_tensor=None
            for _ in range(row['warmup']):
                output_tensor=model(*inputs)
                torch.npu.synchronize()
            before=_eager_prepare.snapshot();sdk_before=sdk_snapshot(lib)
            mapping=mapped_libraries(Path('/proc/self/maps').read_text(),set(config['sdk']['libraries']))
            c.validate_mapping(mapping,config['sdk']['libraries'])
            native_name='libtorch_npu.so';native_path=str(package/'lib'/native_name)
            native_mapping=mapped_libraries(Path('/proc/self/maps').read_text(),{native_name})
            c.validate_mapping(native_mapping,{native_name:dict(path=native_path,sha256=runtime['artifact_pins'][runtime['torch_npu_package']+'/lib/'+native_name])})
            report['ready']=dict(python=sys.version,executable=sys.executable,abi=sysconfig.get_config_var('SOABI'),
                gil_enabled=sys._is_gil_enabled(),torch_git=torch.version.git_version,numpy_version=np.__version__,
                torch_npu=str(package),mapping=mapping,artifact_pins=runtime['artifact_pins'],
                native_mapping=native_mapping,python_executable_sha256=c.sha(sys.executable),
                torch_source_path=torch.__file__,numpy_source_path=np.__file__,
                numpy_import_artifacts=numpy_artifacts,import_resolution=report['import_resolution'],
                extension=dict(path=torch_npu._C.__file__,sha256=c.sha(torch_npu._C.__file__)),
                sdk_source_pins=config['sdk']['source_pins'],thread_affinity=thread_affinities(config['cpus']),
                cpu_identity=dict(state_sha256=state,input_tensor_tree_sha256=input_sha),
                source_verification=verified,sparse=sparse,eager=eager,prepared_snapshot=before)
            dump(output/'ready.json',report['ready'])
            # Parent controller has already validated the same source/artifact pins.
            # Process startup, imports, transfer, configure and warmup are excluded.
            durations=[];caller=threading.get_native_id()
            cpu_start=time.process_time_ns();start=time.perf_counter_ns()
            for _ in range(row['samples']):
                tick=time.perf_counter_ns()
                output_tensor=model(*inputs)
                torch.npu.synchronize()
                durations.append(time.perf_counter_ns()-tick)
            wall=time.perf_counter_ns()-start;cpu=time.process_time_ns()-cpu_start
            c.require(threading.get_native_id()==caller,'original DLRM caller TID changed')
            after=_eager_prepare.snapshot();sdk_after=sdk_snapshot(lib)
            metadata=output_metadata(output_tensor)
            c.require(legacy.eager_proof(model,torch)==eager,'original eager forward changed')
            report.update(durations_ns=durations,timing=c.distribution(durations),window_wall_ns=wall,
                window_process_cpu_ns=cpu,forward_calls=row['warmup']+row['samples'],
                output=metadata,native_before=before,native_after=after,sdk_before=sdk_before,sdk_after=sdk_after)
    except BaseException as error:
        first=error
        dump(output/'error-before-cleanup.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        def close():
            if configured:report['native_closed']=_eager_prepare.drain_close()
            torch.npu.synchronize()
            report['sdk_closed']=sdk_snapshot(lib)
            report['thread_affinity_closed']=thread_affinities(config['cpus'])
        first,secondary=cleanup_preserving_error(first,close)
    if secondary:report['cleanup_error']=secondary
    if first is not None:raise first
    report['mechanism']=c.validate_mechanism(report['native_before'],report['native_after'],report['native_closed'],
        forwards=row['samples'],workers=row['workers'])
    report['sdk_execution_gate']=c.validate_sdk_delta(report['sdk_before'],report['sdk_after'],forwards=row['samples'],
        queued_embeddings=report['mechanism']['embedding_commands_per_forward']*row['samples'])
    c.validate_sdk(report['sdk_closed'],closed=True)
    if row['kind']=='cold':report['cold_gate']=c.validate_cold(report['native_closed'],report['sdk_closed'],row['workers'])
    else:
        c.require(report['sdk_after'].get('tracing_enabled') is False,'SDK trace enabled in performance process')
    c.validate_mapping(mapped_libraries(Path('/proc/self/maps').read_text(),set(config['sdk']['libraries'])),config['sdk']['libraries'])
    c.require(sys._is_gil_enabled() is runtime['expected_gil'],'natural GIL changed during run')
    report.update(passed=True,performance_eligible=row['kind']=='perf',real_sdk_qualified=False,
        numeric_golden_claim=False,true_device_kernel_claim=False,
        boundary='actual original DLRM host forward through synchronize; simulator kernels have no numerical meaning')


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--row',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);args.output.mkdir(exist_ok=False)
    report=dict(passed=False,worker_sha256=c.sha(__file__),contracts_sha256=c.sha(c.__file__))
    try:
        config=c.validate_config(json.loads(args.config.read_text()));row=json.loads(args.row.read_text())
        report.update(row)
        run(config,row,args.output,report)
    except BaseException as error:
        report.update(error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        dump(args.output/'result.json',report)
        raise
    dump(args.output/'result.json',report)


if __name__=='__main__':main()
