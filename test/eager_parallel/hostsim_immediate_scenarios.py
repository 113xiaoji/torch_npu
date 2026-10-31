"""Native E/EI numerical and lifecycle cases on the isolated R27 CPU SDK.

This uses real TorchNPU dispatch and actual stream callbacks. It does not alter
any model methods, install Python operator interception, or use shadow mode.
Run one arm per fresh free-threaded process with TASK_QUEUE_ENABLE=0.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--arm',choices=('e','ei'),required=True)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=('all','shapes','mutation','boundaries','foreign','errors','allocator'),default='all')
    args=parser.parse_args()
    require(os.environ.get('TASK_QUEUE_ENABLE')=='0','fresh TASK_QUEUE_ENABLE=0 process required')
    require(not sys._is_gil_enabled(),'free threading required')
    import torch
    import torch.nn.functional as F
    import torch_npu
    from torch_npu import _eager_parallel as parallel
    require(Path(torch_npu.__file__).resolve().parent==args.package.resolve(),'wrong candidate package')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    runtime=ctypes.CDLL(str(args.sdk/'libr27_hostsim_runtime.so'))
    runtime.R27HostSimDiagnosticsJson.restype=ctypes.c_char_p
    runtime.R27HostSimSetStreamPaused.argtypes=[ctypes.c_void_p,ctypes.c_bool]
    runtime.R27DeviceSync.argtypes=[]
    opapi=ctypes.CDLL(str(args.sdk/'libopapi.so'))
    opapi.R27HostSimOpDiagnosticsJson.restype=ctypes.c_char_p
    report=dict(passed=False,arm=args.arm,case=args.case,rows=[],gil_enabled=sys._is_gil_enabled(),
                real_npu_numeric_claim=False,host_sim_numeric_validated=False)

    def identity():
        paths=[Path(torch_npu._C.__file__),Path(torch._C.__file__),args.package/'lib/libtorch_npu.so',args.package/'__init__.py',args.package/'_eager_parallel.py',
               *(args.sdk/name for name in ('libascendcl.so','libacl_op_compiler.so','libopapi.so','libr27_hostsim_runtime.so'))]
        pins={}
        for path in paths:
            with path.open('rb') as stream:pins[str(path.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
        mappings={}
        lines=Path('/proc/self/maps').read_text().splitlines()
        for path in paths:
            if path.suffix=='.so':
                found={str(Path(line.split()[-1]).resolve()) for line in lines if line.split()[-1].endswith('/'+path.name)}
                allowed={str(p.resolve()) for p in paths if p.name==path.name}
                require(found==allowed,'unexpected loaded artifact mapping: '+path.name)
                mappings[str(path.resolve())]=sorted(found)
        return dict(pins=pins,mappings=mappings)
    report['identity_before']=identity()
    with Path(__file__).open('rb') as stream:report['worker_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()

    def snapshots():
        return dict(native=parallel.snapshot(),runtime=json.loads(runtime.R27HostSimDiagnosticsJson()),
                    opapi=json.loads(opapi.R27HostSimOpDiagnosticsJson()))

    def closed(label, **extra):
        state=snapshots();native=state['native']
        require(native.get('actual_stream_scheduling') is True,label+': native adapter inactive')
        require(native.get('first_error') is None and native['closure']['passed'],label+': native closure failed')
        require(native['active_calls']==native['retained_owners']==native['owned_event_handles']==0,label+': native resources live')
        # Tensor destruction after the scope can enqueue allocator free-event
        # markers. Check native closure first, then separately finish this SDK
        # bookkeeping; it is not additional numerical or native-owned work.
        closure_runtime=state['runtime']
        require(runtime.R27DeviceSync()==0,label+': allocator bookkeeping failed')
        state=snapshots()
        require(state['runtime']['submitted']==state['runtime']['completed'] and state['runtime']['errors']==0,label+': SDK work/errors')
        require(state['opapi']['live_objects']==state['opapi']['active_kernels']==0,label+': SDK objects live')
        report['rows'].append(dict(name=label,passed=True,closure_runtime=closure_runtime,**extra,**state))
        return native

    def equal(actual, expected, label):
        actual=actual.detach().cpu()
        require(actual.shape==expected.shape and bool(torch.isfinite(actual).all()),label+': invalid output')
        require(torch.allclose(actual,expected,rtol=1e-4,atol=1e-5),label+': numerical mismatch')
        return float((actual-expected).abs().max()) if actual.numel() else 0.0

    def inputs(batch=3,dim=4):
        weight=torch.arange(12*dim,dtype=torch.float32).view(12,dim)/13-2
        # Repeated indices and an empty first bag exercise real sum semantics.
        indices=torch.tensor([2,2]*(batch-1),dtype=torch.long)
        offsets=torch.tensor([0]+[2*(i-1) for i in range(1,batch)],dtype=torch.long)
        expected=[]
        for bag in range(batch):
            end=int(offsets[bag+1]) if bag+1<batch else len(indices)
            expected.append([sum(float(weight[int(indices[p]),d]) for p in range(int(offsets[bag]),end)) for d in range(dim)])
        return weight.to('npu:0'),indices.to('npu:0'),offsets.to('npu:0'),torch.tensor(expected,dtype=torch.float32)

    def embedding(values):
        weight,indices,offsets,_=values
        return F.embedding_bag(indices,weight,offsets,mode='sum',sparse=True)

    def shape_cases():
        for batch,dim,tables in ((1,1,1),(3,3,2),(8,7,5),(3,16,26)):
            data=inputs(batch,dim)
            dense=torch.arange(batch*dim,dtype=torch.float32).view(batch,dim)/19
            cpu_bags=[data[3] for _ in range(tables)]
            packed=torch.cat([dense]+cpu_bags,dim=1).view(batch,tables+1,dim)
            gram=torch.bmm(packed,packed.transpose(1,2))
            li,lj=torch.tril_indices(tables+1,tables+1,offset=-1)
            expected=torch.cat([dense,gram[:,li,lj]],dim=1)
            native_dense=dense.to('npu:0')
            with parallel.scope(mode=args.arm,capacity=4 if tables==5 else 64):
                # Every table deliberately shares the same weight/storage.
                bags=[embedding(data) for _ in range(tables)]
                native_packed=torch.cat([native_dense]+bags,dim=1).view(batch,tables+1,dim)
                native_gram=torch.bmm(native_packed,native_packed.transpose(1,2))
                actual=torch.cat([native_dense,native_gram[:,li,lj]],dim=1)
                error=equal(actual,expected,'shape pipeline')
            state=closed(f'shapes-b{batch}-d{dim}-t{tables}',max_abs_error=error)
            require(state['scheduled_embedding']==tables,'embedding shape case escaped native scheduling')
            if args.arm=='ei':require(state['scheduled_cat']==2 and state['scheduled_bmm']==state['scheduled_index']==1,'EI pipeline escaped native scheduling')
            if tables==5:require(state['budget_drains']>0,'capacity backpressure was not exercised')
        data=inputs()
        empty_index=torch.empty(0,dtype=torch.long)
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            selected=output[:,empty_index]
            equal(selected,torch.empty((3,0),dtype=torch.float32),'empty integral index')
        state=closed('empty-integral-index')
        require(state['scheduled_embedding']==1,'empty-index producer was not scheduled')
        if args.arm=='ei':require(state['scheduled_index']==1,'empty Index did not use native consumer')
        cpu_weight=torch.arange(24,dtype=torch.float32).view(6,4)
        weight=cpu_weight.to('npu:0')
        indices=torch.tensor([2,3,3,2],dtype=torch.long).to('npu:0')
        offsets=torch.tensor([0,2,4],dtype=torch.long).to('npu:0')
        expected=torch.stack([cpu_weight[3],cpu_weight[3]])
        with parallel.scope(mode=args.arm):
            output=F.embedding_bag(indices,weight,offsets,mode='sum',sparse=True,include_last_offset=True,padding_idx=2)
            error=equal(output,expected,'padding and terminal offset')
        state=closed('padding-terminal-offset',max_abs_error=error)
        require(state['scheduled_embedding']==1,'padding Embedding did not schedule')

    def mutation_cases():
        data=inputs()
        expected=torch.relu(data[3])
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            alias=output.view(3,4)
            alias.relu_()
            error=equal(output,expected,'inplace alias relu')
        closed('inplace-alias-relu',max_abs_error=error)
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            output.resize_(12)
            error=equal(output,data[3].reshape(12),'resize pending output')
        closed('resize-pending-output',max_abs_error=error)
        replacement=torch.arange(6,dtype=torch.float32).view(2,3)
        native_replacement=replacement.to('npu:0')
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            output.set_(native_replacement)
            error=equal(output,replacement,'set pending output')
        closed('set-pending-output',max_abs_error=error)
        weight,indices,offsets,expected=data
        cpu_weight=weight.cpu().relu()
        cpu_indices=indices.cpu();cpu_offsets=offsets.cpu()
        changed=F.embedding_bag(cpu_indices,cpu_weight,cpu_offsets,mode='sum',sparse=True)
        with parallel.scope(mode=args.arm):
            first=embedding(data)
            weight.relu_()
            second=embedding(data)
            first_error=equal(first,expected,'write-after-read first result')
            second_error=equal(second,changed,'write-after-read next result')
        closed('shared-weight-mutation',first_error=first_error,second_error=second_error)

    def boundary_cases():
        data=inputs(1,1)
        event=torch.npu.Event();other=torch.npu.Stream()
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            require(output.item()==float(data[3].item()),'item did not wait for producer')
        closed('scalar-item-boundary')
        data=inputs()
        with parallel.scope(mode=args.arm):
            output=embedding(data)
            event.record()
            other.wait_event(event)
            event.synchronize();other.synchronize()
            require(event.query(),'completed explicit event remained pending')
            error=equal(output,data[3],'explicit event boundary')
        closed('explicit-event-stream-sync',max_abs_error=error)
        with torch.npu.stream(other):
            try:
                with parallel.scope(mode=args.arm):pass
            except RuntimeError as error:
                require('default caller stream' in str(error),'wrong rejection for nondefault stream')
            else:raise AssertionError('nondefault caller stream admitted')
        report['rows'].append(dict(name='nondefault-stream-config-rejected',passed=True))

    def foreign_case():
        data=inputs();failures=[]
        def foreign():
            try:
                with torch.inference_mode():embedding(data)
            except BaseException as error:failures.append(str(error))
        with parallel.scope(mode=args.arm):
            before=snapshots()['opapi']['operations'].get('aclnnEmbeddingBag',{}).get('prepared',0)
            worker=threading.Thread(target=foreign);worker.start();worker.join(timeout=10)
            require(not worker.is_alive(),'foreign caller deadlocked')
            require(len(failures)==1 and 'foreign caller rejected before effects' in failures[0],'foreign thread not rejected')
            after=snapshots()['opapi']['operations'].get('aclnnEmbeddingBag',{}).get('prepared',0)
            require(after==before,'foreign caller issued an SDK prepare')
            equal(embedding(data),data[3],'owner continues after foreign rejection')
        closed('foreign-caller-rejected-before-effects')

    def error_cases():
        data=inputs()
        marker=RuntimeError('first user body error')
        try:
            with parallel.scope(mode=args.arm):
                embedding(data)
                raise marker
        except RuntimeError as error:require(error is marker,'scope close replaced original body error')
        else:raise AssertionError('body error swallowed')
        closed('first-body-error-preserved')
        original=None
        try:
            with parallel.scope(mode=args.arm):
                output=embedding(data)
                torch.bmm(output,output)  # 2-D inputs: native shape failure.
        except (RuntimeError,IndexError,ValueError) as error:original=str(error)
        require(original is not None,'invalid native BMM did not fail')
        state=snapshots();native=state['native']
        require(native['first_error'] is not None and native['failed']>=1,'native failure not retained')
        require(native['retained_owners']==0 and not native['quarantined'],'recoverable native failure leaked owners')
        require(native['closure']['admission_closed'] and not native['closure']['passed'],'failed domain falsely passed closure')
        report['rows'].append(dict(name='native-shape-error-preserved',passed=True,error=original,**state))
        with parallel.scope(mode=args.arm):equal(embedding(data),data[3],'fresh domain after closed failure')
        closed('fresh-domain-after-recoverable-error')

    def allocator_case():
        data=inputs()
        expected=torch.relu(data[3])
        caller=torch.npu.current_stream()
        caller_raw=caller.npu_stream
        probes=[]
        try:
            with parallel.scope(mode=args.arm):
                output=embedding(data)
                torch.npu.synchronize()  # Finish the owned group before fallback.
                address=output.data_ptr()
                producer_id=parallel.snapshot()['stream_ids'][1]
                require(runtime.R27HostSimSetStreamPaused(caller_raw,True)==0,'caller pause failed')
                consumed=torch.relu(output)  # Unknown schema: caller-stream fallback.
                del output
            # Scope closure is allowed: the fallback's queue work belongs to the
            # caller. The allocator must keep its side-stream input unavailable.
            state=json.loads(runtime.R27HostSimDiagnosticsJson())
            caller_state=next(row for row in state['streams'] if row['handle']==caller_raw)
            require(caller_state['submitted']>caller_state['completed'],'caller kernel was not held pending')
            blocks=[block for segment in torch.npu.memory_snapshot() for block in segment['blocks']
                    if block['address']<=address<block['address']+block['size']]
            report['allocator_handoff']=dict(address=address,caller_stream=caller_raw,
                producer_stream_id=producer_id,matching_blocks=blocks,runtime_while_paused=state)
            require(len(blocks)==1 and blocks[0]['state']=='active_pending_free',
                    'side-stream output was not protected from reuse while caller consumed it')
            producer=torch.npu.Stream(stream_id=producer_id,device_index=0,device_type=caller.device_type)
            with torch.npu.stream(producer):
                for _ in range(64):
                    value=torch.empty((3,4),dtype=torch.float32,device='npu:0')
                    require(value.data_ptr()!=address,'side-stream allocation reused pending caller input')
                    probes.append(value)
        finally:
            runtime.R27HostSimSetStreamPaused(caller_raw,False)
        torch.npu.synchronize()
        error=equal(consumed,expected,'caller handoff result')
        probes.clear()
        closed('caller-allocation-handoff',max_abs_error=error,probe_allocations=64)

    try:
        with torch.inference_mode():
            for name,function in [('shapes',shape_cases),('mutation',mutation_cases),('boundaries',boundary_cases),('foreign',foreign_case),('errors',error_cases),('allocator',allocator_case)]:
                if args.case in ('all',name):function()
        require(not sys._is_gil_enabled(),'GIL enabled during native scenarios')
        report['identity_after']=identity()
        require(report['identity_after']==report['identity_before'],'artifacts changed during scenarios')
        report.update(passed=True,host_sim_numeric_validated=True,final=snapshots(),gil_enabled=False)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        try:report['final']=snapshots()
        except BaseException as snapshot_error:report['snapshot_error']=str(snapshot_error)
        raise
    finally:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,arm=args.arm,cases=len(report['rows']),output=str(args.output))))


if __name__=='__main__':main()
