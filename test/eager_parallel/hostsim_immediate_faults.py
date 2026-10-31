"""Fault actual native E/EI event operations in an isolated fresh process."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

CASES={
    'first_record_before':('event_record',1,False,1,0),
    'first_record_after':('event_record',1,True,1,0),
    'post_body_record_before':('event_record',2,False,1,1),
    'post_body_record_after':('event_record',2,True,1,1),
    'first_wait_before':('event_wait',1,False,1,0),
    'first_wait_after':('event_wait',1,True,1,0),
    'query_quarantine':('stream_query',1,False,-1,1),
    'prior_output_cleanup':('event_record',1,False,1,0),
}


def require(value,message):
    if not value:raise AssertionError(message)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--arm',choices=('e','ei'),required=True)
    parser.add_argument('--case',choices=tuple(CASES),required=True)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get('TASK_QUEUE_ENABLE')=='0' and not sys._is_gil_enabled(),'fresh FT native process required')
    import torch
    import torch.nn.functional as F
    import torch_npu
    from torch_npu import _eager_parallel as parallel
    require(Path(torch_npu.__file__).resolve().parent==args.package.resolve(),'wrong candidate')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sdk=ctypes.CDLL(str(args.sdk/'libr27_hostsim_runtime.so'))
    sdk.R27HostSimInjectFault.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_bool,ctypes.c_int64]
    sdk.R27HostSimClearFault.argtypes=[]
    sdk.R27DeviceSync.argtypes=[]
    sdk.R27HostSimDiagnosticsJson.restype=ctypes.c_char_p
    opapi=ctypes.CDLL(str(args.sdk/'libopapi.so'));opapi.R27HostSimOpDiagnosticsJson.restype=ctypes.c_char_p
    def snapshot():
        return dict(native=parallel.snapshot(),runtime=json.loads(sdk.R27HostSimDiagnosticsJson()),
                    opapi=json.loads(opapi.R27HostSimOpDiagnosticsJson()))
    def executed(state):return state['opapi']['operations'].get('aclnnEmbeddingBag',{}).get('executed',0)
    report=dict(passed=False,arm=args.arm,case=args.case,real_npu_claim=False)
    def identity():
        paths=[Path(torch_npu._C.__file__),Path(torch._C.__file__),args.package/'lib/libtorch_npu.so',args.package/'__init__.py',args.package/'_eager_parallel.py',
               *(args.sdk/name for name in ('libascendcl.so','libacl_op_compiler.so','libopapi.so','libr27_hostsim_runtime.so'))]
        pins={}
        for path in paths:
            with path.open('rb') as stream:pins[str(path.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
        mappings={};lines=Path('/proc/self/maps').read_text().splitlines()
        for path in paths:
            if path.suffix=='.so':
                found={str(Path(line.split()[-1]).resolve()) for line in lines if line.split()[-1].endswith('/'+path.name)}
                allowed={str(p.resolve()) for p in paths if p.name==path.name}
                require(found==allowed,'unexpected loaded mapping: '+path.name)
                mappings[str(path.resolve())]=sorted(found)
        return dict(pins=pins,mappings=mappings)
    report['identity_before']=identity()
    with Path(__file__).open('rb') as stream:report['worker_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    try:
        with torch.inference_mode():
            weight=torch.arange(48,dtype=torch.float32).view(12,4).to('npu:0')
            indices=torch.tensor([2,3,3,2],dtype=torch.long).to('npu:0')
            offsets=torch.tensor([0,2],dtype=torch.long).to('npu:0')
            torch.npu.synchronize()
            operation,nth,after,repetitions,expected_calls=CASES[args.case]
            original=None
            try:
                with parallel.scope(mode=args.arm):
                    if args.case=='prior_output_cleanup':
                        prior=F.embedding_bag(indices,weight,offsets,mode='sum',sparse=True)
                        torch.npu.synchronize()
                        report['prior_output']=dict(address=prior.data_ptr(),shape=list(prior.shape))
                        explicit_event=torch.npu.Event()
                    report['before']=snapshot()
                    require(sdk.R27HostSimInjectFault(operation.encode(),nth,after,repetitions)==0,'fault activation failed')
                    try:
                        F.embedding_bag(indices,weight,offsets,mode='sum',sparse=True)
                    except RuntimeError:
                        if args.case=='prior_output_cleanup':
                            records_before=json.loads(sdk.R27HostSimDiagnosticsJson())['event_records']
                            try:explicit_event.record()
                            except RuntimeError:
                                report['public_event_rejected_while_poisoned']=True
                            else:raise AssertionError('public event bypassed poisoned-domain admission')
                            require(json.loads(sdk.R27HostSimDiagnosticsJson())['event_records']==records_before,
                                    'rejected public event reached SDK record')
                            # This successful previous-group output has no State
                            # or guard owner. Its allocator cleanup must remain
                            # safe while the current domain is still poisoned.
                            del prior
                            report['prior_output_deleted_while_poisoned']=True
                        raise
                    # The persistent-query case fails at the real drain boundary.
                    torch.npu.synchronize()
            except RuntimeError as error:
                original=error
                report['original_error']=dict(type=type(error).__name__,message=str(error),notes=list(getattr(error,'__notes__',[])))
            require(original is not None,'requested fault did not fail native execution')
            report['failed']=snapshot()
            native=report['failed']['native']
            require(native['first_error'] is not None,'native first error missing')
            require('failed' in native['first_error'],'native first error lost failure context')
            if args.case=='query_quarantine':
                require(native['quarantined'] and native['retained_owners']>0,'unproven completion released owners')
                require(not native['closure']['admission_closed'],'unproven completion falsely closed domain')
                require(report['original_error']['notes'],'failed close did not annotate the original error')
            else:
                require(not native['quarantined'] and native['retained_owners']==0,'recoverable event failure leaked owners')
                require(native['closure']['admission_closed'] and not native['closure']['passed'],'failed domain falsely qualified')
            require(sdk.R27HostSimClearFault()==0,'fault reset failed')
            # An explicitly cleared diagnostic fault allows a later real
            # completion proof; only then may a quarantined owner set retire.
            parallel.drain_close()
            require(sdk.R27DeviceSync()==0,'allocator completion bookkeeping failed')
            report['recovered']=snapshot()
            recovered=report['recovered']
            require(recovered['native']['first_error']==native['first_error'],'recovery replaced the first error')
            require(recovered['native']['retained_owners']==recovered['native']['owned_event_handles']==0,'recovery retained completed owners/events')
            require(executed(recovered)-executed(report['before'])==expected_calls,'original kernel was replayed or improperly submitted')
            require(recovered['runtime']['submitted']==recovered['runtime']['completed'],'recovery left SDK work pending')
            require(recovered['opapi']['live_objects']==recovered['opapi']['active_kernels']==0,'recovery left SDK resources active')
            require(not recovered['runtime']['fault_active'] and recovered['runtime']['injected_faults']>0,'fault evidence missing')
        require(not sys._is_gil_enabled(),'GIL enabled during native exception handling')
        report['identity_after']=identity()
        require(report['identity_after']==report['identity_before'],'artifacts changed during fault case')
        report.update(passed=True,gil_enabled=False,expected_embedding_executions=expected_calls)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        sdk.R27HostSimClearFault()
        try:report['final']=snapshot()
        except BaseException as error:report['snapshot_error']=str(error)
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,arm=args.arm,case=args.case,output=str(args.output))))


if __name__=='__main__':main()
