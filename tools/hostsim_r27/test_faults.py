"""Before/after-enqueue fault controls preserve the actual runtime effects."""
import ctypes as C
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]);lib=C.CDLL(str(root/'libascendcl.so'),mode=C.RTLD_GLOBAL)
assert hasattr(lib,'R27HostSimInjectFault'),'explicit simulator fault-injection ABI missing'
ptr=C.c_void_p
def bind(name,args,result=C.c_int):
    function=getattr(lib,name);function.argtypes=args;function.restype=result;return function
inject=bind('R27HostSimInjectFault',[C.c_char_p,C.c_uint64,C.c_bool,C.c_int64])
clear=bind('R27HostSimClearFault',[])
create=bind('aclrtCreateStream',[C.POINTER(ptr)])
event_create=bind('aclrtCreateEvent',[C.POINTER(ptr)])
record=bind('aclrtRecordEvent',[ptr,ptr]);wait=bind('aclrtStreamWaitEvent',[ptr,ptr])
sync=bind('aclrtSynchronizeStream',[ptr]);query=bind('aclrtStreamQuery',[ptr,C.POINTER(C.c_int)])
pause=bind('R27HostSimSetStreamPaused',[ptr,C.c_bool])
copy=bind('aclrtMemcpyAsync',[ptr,C.c_size_t,ptr,C.c_size_t,C.c_int,ptr])
diagnostics=bind('R27HostSimDiagnosticsJson',[],C.c_char_p)
def state():return json.loads(diagnostics())
producer,consumer,event=ptr(),ptr(),ptr()
assert create(C.byref(producer))==create(C.byref(consumer))==event_create(C.byref(event))==0
before=state()['submitted'];assert inject(b'event_record',1,False,1)==0
assert record(event,producer)!=0 and state()['submitted']==before,'before-record failure submitted work'
assert inject(b'event_record',2,False,1)==0
assert record(event,producer)==0 and record(event,producer)!=0 and record(event,producer)==0
assert sync(producer)==0
source=(C.c_int*1)(42);middle=(C.c_int*1)(-1);output=(C.c_int*1)(-2)
assert pause(producer,True)==copy(middle,4,source,4,0,producer)==0
before=state()['submitted'];assert inject(b'event_record',1,True,1)==0
assert record(event,producer)!=0 and state()['submitted']==before+1,'after-record failure lost queued marker'
assert wait(consumer,event)==copy(output,4,middle,4,0,consumer)==0
assert pause(producer,False)==sync(consumer)==0
assert output[0]==42,'after-record failure lost event dependency identity'
for after in (False,True):
    before=state()['submitted'];assert inject(b'event_wait',1,after,1)==0
    assert wait(consumer,event)!=0 and state()['submitted']==before+int(after)
assert sync(consumer)==0
assert inject(b'stream_query',1,False,-1)==0
status=C.c_int(-1)
assert query(producer,C.byref(status))!=0 and query(producer,C.byref(status))!=0
assert clear()==query(producer,C.byref(status))==0 and status.value==0
destroy_event=bind('aclrtDestroyEvent',[ptr]);destroy_stream=bind('aclrtDestroyStream',[ptr])
assert destroy_event(event)==destroy_stream(producer)==destroy_stream(consumer)==0
summary=state();assert summary['injected_faults']==7 and summary['fault_active'] is False
assert summary['submitted']==summary['completed']
assert summary['workers']==summary['live_events']==0 and summary['live_streams']==1
print(json.dumps(dict(passed=True,before_and_after_effects=True,nth_record=True,persistent_query=True,runtime=summary)))
