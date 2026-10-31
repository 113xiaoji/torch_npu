"""An asynchronous kernel failure must remain an error through event queries."""
import ctypes as C
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]);runtime=C.CDLL(str(root/'libascendcl.so'),mode=C.RTLD_GLOBAL)
op=C.CDLL(str(root/'libopapi.so'),mode=C.RTLD_GLOBAL)
ptr=C.c_void_p;i64=C.c_int64;u64=C.c_uint64
def bind(lib,name,args,result=C.c_int):
    f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
create=bind(runtime,'aclrtCreateStream',[C.POINTER(ptr)])
event_create=bind(runtime,'aclrtCreateEvent',[C.POINTER(ptr)])
record=bind(runtime,'aclrtRecordEvent',[ptr,ptr])
sync=bind(runtime,'aclrtSynchronizeStream',[ptr])
query_event=bind(runtime,'aclrtQueryEventStatus',[ptr,C.POINTER(C.c_int)])
query_stream=bind(runtime,'aclrtStreamQuery',[ptr,C.POINTER(C.c_int)])
pause=bind(runtime,'R27HostSimSetPaused',[C.c_bool])
tensor=bind(op,'aclCreateTensor',[C.POINTER(i64),u64,C.c_int,C.POINTER(i64),i64,C.c_int,C.POINTER(i64),u64,ptr],ptr)
prep=bind(op,'aclnnReluGetWorkspaceSize',[ptr,ptr,C.POINTER(u64),C.POINTER(ptr)])
execute=bind(op,'aclnnRelu',[ptr,u64,ptr,ptr])
stream,event=ptr(),ptr();assert create(C.byref(stream))==event_create(C.byref(event))==0
x=(C.c_float*2)(-1,2);y=(C.c_float*3)();two=(i64*1)(2);three=(i64*1)(3);stride=(i64*1)(1)
a=tensor(two,1,0,stride,0,2,two,1,x);b=tensor(three,1,0,stride,0,2,three,1,y)
size,executor=u64(),ptr();assert prep(a,b,C.byref(size),C.byref(executor))==0
assert pause(True)==execute(None,size,executor,stream)==record(event,stream)==pause(False)==0
assert sync(stream)!=0,'asynchronous bad output shape was swallowed'
status=C.c_int(-123)
assert query_event(event,C.byref(status))!=0,'failed event reported successful completion'
assert query_stream(stream,C.byref(status))!=0,'failed stream reported successful completion'
print(json.dumps(dict(passed=True,async_failure_propagated=True,event_query_failed=True,stream_query_failed=True)))
