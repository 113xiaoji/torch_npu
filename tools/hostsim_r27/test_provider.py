"""Numerical and stream-order contracts; deliberately rejects the old NoOp SDK."""
import argparse
import ctypes as C
import json
from pathlib import Path


def run(directory):
    runtime = C.CDLL(str(directory / 'libascendcl.so'), mode=C.RTLD_GLOBAL)
    op = C.CDLL(str(directory / 'libopapi.so'), mode=C.RTLD_GLOBAL)
    ptr = C.c_void_p
    i64 = C.c_int64
    u64 = C.c_uint64
    def bind(lib, name, args, result=C.c_int):
        f = getattr(lib, name); f.argtypes = args; f.restype = result; return f
    create = bind(runtime, 'aclrtCreateStream', [C.POINTER(ptr)])
    sync = bind(runtime, 'aclrtSynchronizeStream', [ptr])
    pause = bind(runtime, 'R27HostSimSetPaused', [C.c_bool])
    pause_stream = bind(runtime, 'R27HostSimSetStreamPaused', [ptr,C.c_bool])
    stream_query = bind(runtime, 'aclrtStreamQuery', [ptr,C.POINTER(C.c_int)])
    event_query = bind(runtime, 'aclrtQueryEventStatus', [ptr,C.POINTER(C.c_int)])
    legacy_query = bind(runtime, 'aclrtQueryEvent', [ptr,C.POINTER(C.c_int)])
    event_create = bind(runtime, 'aclrtCreateEvent', [C.POINTER(ptr)])
    record = bind(runtime, 'aclrtRecordEvent', [ptr, ptr])
    wait = bind(runtime, 'aclrtStreamWaitEvent', [ptr, ptr])
    copy = bind(runtime, 'aclrtMemcpyAsync', [ptr, C.c_size_t, ptr, C.c_size_t, C.c_int, ptr])
    tensor = bind(op, 'aclCreateTensor', [C.POINTER(i64),u64,C.c_int,C.POINTER(i64),i64,C.c_int,C.POINTER(i64),u64,ptr],ptr)
    destroy = bind(op, 'aclDestroyTensor', [ptr])
    descriptors = []
    def desc(data, shape, strides=None, offset=0):
        if strides is None:
            strides=[]; stride=1
            for size in reversed(shape): strides.insert(0,stride); stride*=size
        dims=(i64*len(shape))(*shape); step=(i64*len(shape))(*strides)
        dtype = 9 if data._type_ is i64 else 0
        result=tensor(dims,len(shape),dtype,step,offset,2,dims,len(shape),C.cast(data,ptr))
        assert result, 'tensor creation failed'; descriptors.append(result); return result
    producer, consumer, event = ptr(),ptr(),ptr()
    assert create(C.byref(producer)) == create(C.byref(consumer)) == event_create(C.byref(event)) == 0
    source=(C.c_float*4)(1,2,3,4); middle=(C.c_float*4)(-9,-9,-9,-9); output=(C.c_float*4)(-7,-7,-7,-7)
    assert pause(True)==0
    assert copy(middle,16,source,16,0,producer)==0
    deferred=list(middle)==[-9]*4
    assert record(event,producer)==wait(consumer,event)==0
    assert copy(output,16,middle,16,0,consumer)==pause(False)==sync(consumer)==0
    assert list(output)==[1,2,3,4], 'event wait did not order producer copy before consumer'
    assert deferred, 'async copy executed immediately: stream scheduling is not simulated'
    # A deliberately omitted wait must expose stale data, not be accidentally
    # repaired by a global host serialization barrier.
    assert pause_stream(producer,True)==0
    stale=(C.c_float*4)(-11,-11,-11,-11); readback=(C.c_float*4)()
    assert copy(stale,16,source,16,0,producer)==0
    assert record(event,producer)==0
    status=C.c_int(-123)
    assert stream_query(producer,C.byref(status))==0 and status.value==1
    assert event_query(event,C.byref(status))==0 and status.value==0
    assert legacy_query(event,C.byref(status))==0 and status.value==1
    assert copy(readback,16,stale,16,0,consumer)==sync(consumer)==0
    assert list(readback)==[-11]*4, 'negative control silently serialized independent streams'
    assert wait(consumer,event)==copy(readback,16,stale,16,0,consumer)==0
    assert pause_stream(producer,False)==sync(consumer)==0
    assert list(readback)==[1,2,3,4]
    assert event_query(event,C.byref(status))==0 and status.value==1
    assert legacy_query(event,C.byref(status))==0 and status.value==0
    # Waiting captures the old record generation even after event re-record.
    other=ptr();assert create(C.byref(other))==pause_stream(other,True)==0
    assert wait(consumer,event)==record(event,other)==0
    assert sync(consumer)==0, 'wait retargeted to a later event generation'
    assert pause_stream(other,False)==sync(other)==0
    # Explicit bag-sum oracle, independent of ATen embedding_bag.
    weights=(C.c_float*12)(*range(1,13)); indices=(i64*5)(2,0,2,1,1); offsets=(i64*3)(0,2,2)
    result=(C.c_float*9)(*([-123]*9)); aux=(i64*5)(); bags=(i64*3)(); maximum=(i64*3)()
    args=[desc(weights,[4,3]),desc(indices,[5]),desc(offsets,[3]),False,0,False,None,False,-1,
          desc(result,[3,3]),desc(aux,[5]),desc(bags,[3]),desc(maximum,[3])]
    prep=bind(op,'aclnnEmbeddingBagGetWorkspaceSize',[ptr,ptr,ptr,C.c_bool,i64,C.c_bool,ptr,C.c_bool,i64,ptr,ptr,ptr,ptr,C.POINTER(u64),C.POINTER(ptr)])
    execute=bind(op,'aclnnEmbeddingBag',[ptr,u64,ptr,ptr])
    size,executor=u64(),ptr(); assert prep(*args,C.byref(size),C.byref(executor))==0
    assert execute(None,size,executor,producer)==0
    # Cleanup before completion must not invalidate captured metadata.
    for item in descriptors: assert destroy(item)==0
    descriptors.clear()
    assert sync(producer)==0
    expected=[]
    for first,last in [(0,2),(2,2),(2,5)]:
        expected.extend(sum(weights[indices[p]*3+k] for p in range(first,last)) for k in range(3))
    assert list(result)==expected, ('embedding arithmetic mismatch',list(result),expected)
    # Gram product includes a non-contiguous transposed input.
    padded=(C.c_float*10)(999,*list(result))
    gram=(C.c_float*9)(*([-123]*9)); left=desc(result,[1,3,3]); right=desc(padded,[1,3,3],[9,1,3],offset=1); out=desc(gram,[1,3,3])
    prep=bind(op,'aclnnBatchMatMulGetWorkspaceSize',[ptr,ptr,ptr,C.c_int8,C.POINTER(u64),C.POINTER(ptr)])
    execute=bind(op,'aclnnBatchMatMul',[ptr,u64,ptr,ptr])
    assert prep(left,right,out,0,C.byref(size),C.byref(executor))==0
    assert execute(None,size,executor,consumer)==sync(consumer)==0
    expected_gram=[sum(expected[i*3+k]*expected[j*3+k] for k in range(3)) for i in range(3) for j in range(3)]
    assert list(gram)==expected_gram, 'strided Gram arithmetic mismatch'
    for item in descriptors: assert destroy(item)==0
    event_destroy=bind(runtime,'aclrtDestroyEvent',[ptr]);stream_destroy=bind(runtime,'aclrtDestroyStream',[ptr])
    assert event_destroy(event)==stream_destroy(other)==stream_destroy(producer)==stream_destroy(consumer)==0
    diagnostics=bind(runtime,'R27HostSimDiagnosticsJson',[],C.c_char_p)
    summary=json.loads(diagnostics());assert summary['errors']==0 and summary['submitted']==summary['completed']
    op_diagnostics=bind(op,'R27HostSimOpDiagnosticsJson',[],C.c_char_p)
    op_summary=json.loads(op_diagnostics());assert op_summary['live_objects']==0 and op_summary['active_kernels']==0
    embedding_streams=op_summary['operations']['aclnnEmbeddingBag']['stream_executions']
    bmm_streams=op_summary['operations']['aclnnBatchMatMul']['stream_executions']
    assert embedding_streams=={str(producer.value):1} and bmm_streams=={str(consumer.value):1}
    print(json.dumps(dict(passed=True,deferred_copy=True,event_order=True,missing_wait_negative=True,event_generation=True,
                         query_out_parameters=True,embedding=True,strided_offset_gram=True,runtime=summary,opapi=op_summary)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
    run(parser.parse_args().directory.resolve())
