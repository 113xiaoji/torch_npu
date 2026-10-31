"""Empty integer indices are not the OpPlugin empty-FP32 None sentinel."""
import ctypes as C
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]);runtime=C.CDLL(str(root/'libascendcl.so'),mode=C.RTLD_GLOBAL)
op=C.CDLL(str(root/'libopapi.so'),mode=C.RTLD_GLOBAL)
ptr=C.c_void_p;i64=C.c_int64;u64=C.c_uint64
def bind(lib,name,args,result=C.c_int):
    f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
tensor=bind(op,'aclCreateTensor',[C.POINTER(i64),u64,C.c_int,C.POINTER(i64),i64,C.c_int,C.POINTER(i64),u64,ptr],ptr)
create_list=bind(op,'aclCreateTensorList',[C.POINTER(ptr),u64],ptr)
destroy_tensor=bind(op,'aclDestroyTensor',[ptr]);destroy_list=bind(op,'aclDestroyTensorList',[ptr])
create=bind(runtime,'aclrtCreateStream',[C.POINTER(ptr)]);sync=bind(runtime,'aclrtSynchronizeStream',[ptr])
destroy_stream=bind(runtime,'aclrtDestroyStream',[ptr])
prepare=bind(op,'aclnnIndexGetWorkspaceSize',[ptr,ptr,ptr,C.POINTER(u64),C.POINTER(ptr)])
execute=bind(op,'aclnnIndex',[ptr,u64,ptr,ptr]);stream=ptr();assert create(C.byref(stream))==0
def desc(data,shape,dtype):
    dims=(i64*len(shape))(*shape);steps=[];stride=1
    for length in reversed(shape):steps.insert(0,stride);stride*=length
    strides=(i64*len(shape))(*steps)
    result=tensor(dims,len(shape),dtype,strides,0,2,dims,len(shape),data)
    assert result;return result
for dtype,ctype,values in [(9,i64,[]),(3,C.c_int,[]),(9,i64,[2,0])]:
    data=(C.c_float*6)(0,1,2,3,4,5);sentinel=(C.c_float*0)();index=(ctype*len(values))(*values)
    output=(C.c_float*(2*len(values)))()
    x=desc(data,[2,3],0);none=desc(sentinel,[0],0);ix=desc(index,[len(values)],dtype)
    y=desc(output,[2,len(values)],0);list_handle=create_list((ptr*2)(none,ix),2)
    size,executor=u64(),ptr();assert prepare(x,list_handle,y,C.byref(size),C.byref(executor))==0
    assert execute(None,size,executor,stream)==0
    assert destroy_list(list_handle)==destroy_tensor(x)==destroy_tensor(y)==0
    assert sync(stream)==0,'valid empty integral index was conflated with the FP32 None sentinel'
    assert list(output)==[data[row*3+col] for row in range(2) for col in values]
assert destroy_stream(stream)==0
print(json.dumps(dict(passed=True,empty_int64=True,empty_int32=True,none_sentinel_with_nonempty_index=True)))
