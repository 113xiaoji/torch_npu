"""Cold retirement ordering with real Python workers; no SDK/device claims."""
import atexit
from contextvars import ContextVar
import gc
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import test_native_submit_lifetime_r22 as base

runtime,contracts=base.runtime,base.contracts
provider=importlib.import_module(base.NAME+'.torch_npu_provider')
HERE=Path(__file__).resolve().parent
CLOSURE_ZERO=('owned_opapi_device_pending','owned_opapi_payloads_live','owned_opapi_retirement_failures',
              'owned_opapi_retained_bytes','queue_pending_total','native_retirement_pending',
              'packet_blocks_live','native_runs_live','failures')


class DelayedSession(base.LifetimeSession):
    def __init__(self,regions,capacity,*,block=False,error=None,log=None):
        super().__init__(regions,capacity)
        self.cold_calls=0;self.pending=True;self.error=error;self.log=log
        self.cold_entered=threading.Event();self.cold_release=threading.Event()
        if not block:self.cold_release.set()
        self.retire_worker_tls=self.cold_retire

    def cold_retire(self):
        self.cold_calls+=1
        if not all(ref() is not None for ref in self.borrowed):raise AssertionError('SDK-shaped TLS died before cold ACK')
        self.cold_entered.set()
        if self.log:self.log('COLD_BEGIN')
        if not self.cold_release.wait(3):raise RuntimeError('fixture cold ACK was not released')
        if self.error is not None:
            if self.log:self.log('COLD_FAILED')
            raise self.error
        self.pending=False
        if self.log:self.log('COLD_ACK')
        return True


def make_case(*,required=True,block=False,error=None,log=None):
    sessions=[]
    def create(regions,capacity):
        session=DelayedSession(regions,capacity,block=block,error=error,log=log)
        sessions.append(session);return session
    options={}
    if 'native_retirement_required' in contracts.RegionBackend.__dataclass_fields__:
        options['native_retirement_required']=required
    backend=contracts.RegionBackend(lambda *_:None,lambda _:True,native_run=create,**options)
    rt=runtime.RegionRuntime(contracts.RegionConfig(enabled=True,workers=4,regions=4,commands_per_region=8))
    return rt,backend,sessions


def dispose_fixture(rt):
    """Only these fakes have no SDK/real accepted queue work; never production recovery."""
    executor=getattr(rt,'_executor',None)
    if executor is not None:
        executor.shutdown(wait=True,cancel_futures=True);rt._executor=None
    owners=getattr(runtime,'_native_retirement_owners',{})
    owners.pop(id(rt),None)


def child_scenario(path,mode):
    def log(event):
        with Path(path).open('a',encoding='utf-8') as stream:stream.write(event+'\n');stream.flush()
    class Marker:
        def __del__(self):log('TLS_EXIT')
    rt,backend,sessions=make_case(error=RuntimeError('fixture cold failure') if mode in ('exit-failure','already-quarantined') else None,log=log)
    def prepare(index,value):
        session=sessions[-1]
        if not hasattr(session.local,'cache'):session.local.cache=Marker()
        with session.condition:
            session.borrowed.append(weakref.ref(session.local.cache));session.prepared+=1
        return value
    rt.map(prepare,range(4),backend=backend)
    atexit.register(log,'ORDINARY_ATEXIT')
    if mode=='already-quarantined':
        try:rt.close()
        except BaseException:log('EXPLICIT_CLOSE_FAILED')
    if mode=='hook-interrupt':
        class InterruptedLock:
            def __enter__(self):raise KeyboardInterrupt('fixture pre-thread-exit registry interruption')
            def __exit__(self,*_args):return False
        runtime._native_retirement_owners_lock=InterruptedLock()
    del rt;gc.collect();log('USER_END')


class NativeCloseTests(unittest.TestCase):
    def setUp(self):
        self.previous_quarantine=runtime._native_cleanup_quarantine
        self.runtimes=[]
    def tearDown(self):
        for rt in self.runtimes:dispose_fixture(rt)
        runtime._native_cleanup_quarantine=self.previous_quarantine
    def start(self,**kwargs):
        rt,backend,sessions=make_case(**kwargs);self.runtimes.append(rt)
        self.assertEqual(rt.map(lambda i,v:sessions[-1].prepare(i,v),range(4),backend=backend),list(range(4)))
        return rt,sessions[0]

    def test_normal_close_keeps_all_worker_tls_alive_while_cold_ack_waits(self):
        rt,session=self.start(block=True);errors=[]
        def close():
            try:rt.close()
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=close);thread.start()
        try:
            self.assertTrue(session.cold_entered.wait(.5),'normal close skipped its cold retirement barrier')
            self.assertTrue(thread.is_alive());self.assertTrue(all(ref() is not None for ref in session.borrowed))
            self.assertIsNotNone(rt._executor)
        finally:session.cold_release.set();thread.join(2)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[]);self.assertFalse(session.pending)
        self.assertTrue(all(ref() is None for ref in session.borrowed));self.assertTrue(rt.closed)

    def test_failed_or_interrupted_ack_quarantines_without_shutdown_or_retry(self):
        for failure in (RuntimeError('cold ACK failed'),KeyboardInterrupt('cold ACK interrupted')):
            rt,session=self.start(error=failure)
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)) as caught:rt.close()
                self.assertIs(caught.exception,failure);self.assertEqual(session.cold_calls,1)
                self.assertTrue(all(ref() is not None for ref in session.borrowed));self.assertFalse(rt.closed)
                self.assertIsNotNone(rt._native_failed_session)
                with self.assertRaisesRegex(RuntimeError,'quarantin'):rt.close()
                self.assertEqual(session.cold_calls,1)
            dispose_fixture(rt);runtime._native_cleanup_quarantine=self.previous_quarantine

    def test_idempotent_close_only_runs_one_cold_barrier(self):
        rt,session=self.start();rt.close();rt.close();self.assertEqual(session.cold_calls,1)

    def test_interrupt_publishing_cold_ack_state_keeps_tls_and_quarantines(self):
        rt,session=self.start()
        class InterruptedBarriers(list):
            def clear(self):super().clear();raise KeyboardInterrupt('ACK state publication interrupted')
        rt._native_retirement_barriers=InterruptedBarriers(rt._native_retirement_barriers)
        with self.assertRaises(KeyboardInterrupt):rt.close()
        self.assertIsNotNone(rt._native_failed_session,'cold ACK publication interruption escaped quarantine')
        self.assertTrue(all(ref() is not None for ref in session.borrowed));self.assertEqual(session.cold_calls,1)

    def test_cold_executor_replacement_requires_ack_before_stopping_workers(self):
        rt,session=self.start();rt._shutdown_native_executor()
        self.assertEqual(session.cold_calls,1);self.assertFalse(session.pending)
        self.assertTrue(all(ref() is None for ref in session.borrowed))

    def test_idle_pool_worker_cannot_reenter_close_outside_region_binding(self):
        rt,session=self.start();future=rt._executor.submit(rt.close)
        with self.assertRaisesRegex(RuntimeError,'worker'):future.result(2)
        self.assertFalse(rt.closed);self.assertEqual(session.cold_calls,0)
        self.assertTrue(all(ref() is not None for ref in session.borrowed))

    def test_active_and_reentrant_close_reject_before_any_second_barrier(self):
        rt,session=self.start();rt._gate.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'active'):rt.close()
        finally:rt._gate.release()
        original=session.retire_worker_tls
        # The provider's stable callback itself may trigger harmless Python cleanup.
        def barrier():
            with self.assertRaisesRegex(RuntimeError,'active'):rt.close()
            return original()
        # Use a new runtime: callbacks are retained before any SDK-shaped open.
        rt.close();other,backend,sessions=make_case();self.runtimes.append(other)
        create=backend.native_run
        def native_run(*args):
            value=create(*args);cold=value.retire_worker_tls
            def reentrant():
                with self.assertRaisesRegex(RuntimeError,'active'):other.close()
                return cold()
            value.retire_worker_tls=reentrant;return value
        from dataclasses import replace
        other.map(lambda i,v:v,range(4),backend=replace(backend,native_run=native_run));other.close()

    def test_gc_retains_real_pool_until_ordered_pre_thread_exit_barrier(self):
        rt,backend,sessions=make_case()
        rt.map(lambda i,v:sessions[-1].prepare(i,v),range(4),backend=backend)
        reference=weakref.ref(rt);session=sessions[0];del rt;gc.collect()
        try:
            self.assertIsNotNone(reference(),'GC dropped the real runtime before accepted leases retired')
            self.assertTrue(all(ref() is not None for ref in session.borrowed))
            self.assertTrue(hasattr(runtime,'_before_native_thread_shutdown'),'ordered pre-thread-exit hook missing')
            runtime._before_native_thread_shutdown()
            self.assertEqual(session.cold_calls,1);self.assertFalse(session.pending)
        finally:
            held=reference()
            if held is not None:dispose_fixture(held)

    def test_generic_native_fixture_does_not_claim_or_run_sdk_retirement_ack(self):
        rt,session=self.start(required=False);rt.close();self.assertEqual(session.cold_calls,0)

    def test_generic_optional_session_attribute_is_not_an_sdk_retirement_capability(self):
        rt,backend,sessions=make_case(required=False);self.runtimes.append(rt)
        create=backend.native_run
        def generic(*args):
            value=create(*args);value.native_retirement_required=True;return value
        from dataclasses import replace
        rt.map(lambda i,v:v,range(4),backend=replace(backend,native_run=generic));rt.close()
        self.assertEqual(sessions[0].cold_calls,0)

    def test_real_backend_without_cold_barrier_rejects_before_open(self):
        rt,backend,sessions=make_case();self.runtimes.append(rt)
        create=backend.native_run
        def missing(*args):
            session=create(*args);session.retire_worker_tls=None;return session
        from dataclasses import replace
        with self.assertRaisesRegex(RuntimeError,'retirement'):
            rt.map(lambda i,v:v,range(4),backend=replace(backend,native_run=missing))
        self.assertNotIn('open',sessions[0].events,'real session acquired owners before retirement capability')
        self.assertEqual(sessions[0].borrowed,[])

    def test_interpreter_shutdown_rejects_new_real_ownership_even_if_hook_already_registered(self):
        rt,backend,sessions=make_case();self.runtimes.append(rt)
        with patch.object(runtime,'_native_thread_exit_registered',True),patch.object(runtime.threading,'_SHUTTING_DOWN',True):
            with self.assertRaisesRegex(RuntimeError,'shutdown|interpreter'):
                rt.map(lambda i,v:v,range(4),backend=backend)
        self.assertNotIn('open',sessions[0].events)

    def test_retry_close_removes_acknowledged_owner_after_post_join_registry_interrupt(self):
        rt,session=self.start();original=runtime._native_retirement_owners_lock
        class InterruptedOnce:
            calls=0
            def __enter__(self):
                self.calls+=1
                if self.calls==1:raise KeyboardInterrupt('post-join registry removal interrupted')
                return original.__enter__()
            def __exit__(self,*args):return original.__exit__(*args)
        with patch.object(runtime,'_native_retirement_owners_lock',InterruptedOnce()):
            with self.assertRaises(KeyboardInterrupt):rt.close()
            self.assertIsNone(rt._executor);self.assertFalse(rt.closed)
            self.assertIs(runtime._native_retirement_owners[id(rt)],rt);self.assertEqual(session.cold_calls,1)
            rt.close()
            self.assertTrue(rt.closed);self.assertNotIn(id(rt),runtime._native_retirement_owners)
            self.assertEqual(session.cold_calls,1,'registry-only retry repeated SDK retirement')


class ProviderColdBarrierTests(unittest.TestCase):
    def make_provider(self):
        stream=SimpleNamespace(stream_id=0,device_index=0,device_type=20)
        calls=[];stats=dict.fromkeys(CLOSURE_ZERO,0)
        npu=SimpleNamespace(default_stream=lambda device:stream,synchronize=lambda device:calls.append(device),current_device=lambda:7)
        obj=provider.NativeCaptureProvider.__new__(provider.NativeCaptureProvider)
        obj.torch_npu=SimpleNamespace(npu=npu);obj.native=SimpleNamespace(stats=lambda:dict(stats))
        obj.real_opapi_enabled=True;obj.batch_size=32;obj._caller=ContextVar('cold_provider_case',default=None)
        def session():
            obj._caller.set(provider._CallerContext(0,stream,None,None,True))
            return obj._create_native_run(4,8)
        return obj,session,npu,stream,calls,stats

    def test_provider_binds_device_zero_and_stable_default_domain_not_closing_thread_device(self):
        obj,create,npu,stream,calls,stats=self.make_provider();first=create();second=create()
        self.assertTrue(callable(getattr(first,'retire_worker_tls',None)),'real provider did not supply a cold retirement barrier')
        self.assertIs(first.retire_worker_tls,second.retire_worker_tls,'one stable barrier per domain avoids per-map retained growth')
        self.assertIs(first.retire_worker_tls(),True);self.assertEqual(calls,[0])

    def test_provider_domain_drift_or_unclosed_native_state_never_acknowledges(self):
        obj,create,npu,stream,calls,stats=self.make_provider();session=create()
        barrier=getattr(session,'retire_worker_tls',None);self.assertTrue(callable(barrier),'cold barrier missing')
        stream.stream_id=3
        with self.assertRaisesRegex(RuntimeError,'stream|domain'):barrier()
        self.assertEqual(calls,[]);stream.stream_id=0;stats['owned_opapi_payloads_live']=1
        with self.assertRaisesRegex(RuntimeError,'retir|pending|closure'):barrier()

    def test_original_sync_returning_normally_cannot_hide_host_queue_fault_or_pending(self):
        for name in CLOSURE_ZERO:
            obj,create,npu,stream,calls,stats=self.make_provider();barrier=create().retire_worker_tls;stats[name]=1
            with self.subTest(field=name),self.assertRaisesRegex(RuntimeError,'closure'):barrier()

    def real_bridge(self):
        from test_real_provider_r22 import FakeLibrary,REAL_ENV
        stream=SimpleNamespace(stream_id=0,device_index=0,device_type=20);calls=[]
        npu=SimpleNamespace(default_stream=lambda _device:stream,synchronize=lambda device:calls.append(device))
        library=FakeLibrary(dict.fromkeys(CLOSURE_ZERO,0))
        with patch.dict(provider.os.environ,REAL_ENV,clear=True):
            obj=provider.NativeCaptureProvider(SimpleNamespace(),SimpleNamespace(npu=npu),
                library=library,native_run=True,real_opapi=True)
        return obj,library,stream,calls

    def test_low_level_real_session_refuses_missing_or_generic_barrier_before_open(self):
        obj,library,stream,calls=self.real_bridge()
        for barrier in (None,lambda:True):
            with self.subTest(barrier=barrier),self.assertRaisesRegex(provider.UnsupportedRegion,'retirement'):
                provider.NativeRunSession(obj.native,4,8,8,retire_worker_tls=barrier)
        self.assertNotIn('run_create_real_opapi_out',library.calls)

    def test_real_session_from_unflagged_handwritten_backend_still_requires_cold_ack(self):
        obj,library,stream,calls=self.real_bridge()
        def create(*args):
            obj._caller.set(provider._CallerContext(0,stream,None,None,True));return obj._create_native_run(*args)
        backend=contracts.RegionBackend(lambda *_:None,lambda _:True,native_run=create)
        rt=runtime.RegionRuntime(contracts.RegionConfig(enabled=True,workers=2,regions=2,commands_per_region=8))
        try:
            rt.map(lambda i,v:v,range(2),backend=backend);rt.close();self.assertEqual(calls,[0])
        finally:dispose_fixture(rt)

    def test_real_session_revalidates_cold_capability_before_native_open(self):
        obj,library,stream,calls=self.real_bridge();obj._caller.set(provider._CallerContext(0,stream,None,None,True))
        session=obj._create_native_run(4,8);session.retire_worker_tls=None
        with self.assertRaisesRegex(provider.UnsupportedRegion,'retirement'):session.open()
        self.assertNotIn('run_create_real_opapi_out',library.calls)


class ExitOrderingTests(unittest.TestCase):
    def child(self,mode):
        parent=Path(os.environ.get('R22_CLOSE_TEST_ARTIFACTS',tempfile.gettempdir())).resolve()
        folder=Path(tempfile.mkdtemp(prefix='r22-cold-exit-',dir=parent));log=folder/'events.txt'
        source='import sys\nsys.path.insert(0,'+repr(str(HERE))+')\nfrom test_native_close_retirement_r22 import child_scenario\nchild_scenario('+repr(str(log))+','+repr(mode)+')\n'
        script=folder/'child.py';script.write_text(source,encoding='utf-8')
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen([sys.executable,'-I','-B',str(script)],stdout=stdout,stderr=stderr,start_new_session=os.name=='posix')
            timed_out=False
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                timed_out=True
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait(timeout=2)
        result=dict(mode=mode,exit_code=process.returncode,timed_out=timed_out,events=log.read_text().splitlines() if log.exists() else [],stderr=(folder/'stderr.log').read_text(errors='replace'))
        (folder/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result

    def test_normal_exit_retirement_precedes_worker_tls_exit_and_ordinary_atexit(self):
        result=self.child('normal');self.assertFalse(result['timed_out']);self.assertEqual(result['exit_code'],0,result)
        self.assertIn('COLD_ACK',result['events'],'process exit skipped cold retirement')
        self.assertLess(result['events'].index('COLD_ACK'),result['events'].index('TLS_EXIT'))
        self.assertLess(result['events'].index('TLS_EXIT'),result['events'].index('ORDINARY_ATEXIT'))

    def test_exit_ack_failure_or_existing_quarantine_blocks_futures_shutdown_until_parent_kill(self):
        for mode in ('exit-failure','already-quarantined'):
            with self.subTest(mode=mode):
                result=self.child(mode);self.assertTrue(result['timed_out'],'unsafe exit continued into futures worker shutdown')
                self.assertNotIn('TLS_EXIT',result['events']);self.assertEqual(result['events'].count('COLD_FAILED'),1)
                self.assertIn('fail-stop',result['stderr'])

    def test_exit_registry_snapshot_interrupt_also_blocks_futures_shutdown(self):
        result=self.child('hook-interrupt')
        self.assertTrue(result['timed_out']);self.assertNotIn('TLS_EXIT',result['events'])
        self.assertNotIn('COLD_BEGIN',result['events']);self.assertIn('fail-stop',result['stderr'])


if __name__=='__main__':unittest.main()
