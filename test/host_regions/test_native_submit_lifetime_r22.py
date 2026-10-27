"""Real Python worker/TLS lifetimes at an SDK-shaped boundary; no SDK/device."""
from contextlib import contextmanager
import contextvars
import gc
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import unittest
import weakref
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('HOST_REGIONS_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
SOURCE_ROOT = REPO_ROOT / 'torch_npu' / '_host_regions'
NAME = 'r22_submit_lifetime_tests'
spec = importlib.util.spec_from_file_location(NAME, SOURCE_ROOT / '__init__.py',
    submodule_search_locations=[str(SOURCE_ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[NAME] = package
spec.loader.exec_module(package)
runtime = sys.modules[NAME + '.runtime']
contracts = sys.modules[NAME + '.contracts']


class SubmissionFailure(RuntimeError):
    pass


class CacheOwner:
    """Thread-local owner; queued native executors borrow it through weakrefs."""
    pass


class LifetimeSession:
    def __init__(self, regions, capacity, *, delay=False, close_failure=False):
        self.condition = threading.Condition()
        self.terminals = [False] * regions
        self.local = threading.local()
        self.borrowed = []
        self.prepared = 0
        self.active = 0
        self.aborted = False
        self.closed = False
        self.unsafe_retire = False
        self.close_failure = close_failure
        self.delay = delay
        self.release = threading.Event()
        self.events = []
        self.close_calls = 0

    def open(self):
        self.events.append('open')

    def begin(self, index):
        self.local.region = index
        with self.condition:
            self.active += 1

    def prepare(self, index, value):
        if not hasattr(self.local, 'cache'):
            self.local.cache = CacheOwner()
        with self.condition:
            self.borrowed.append(weakref.ref(self.local.cache))
            self.prepared += 1
            self.condition.notify_all()
        if self.delay:
            if not self.release.wait(3):
                raise RuntimeError('test timed out waiting for abort')
        return value

    def end(self):
        with self.condition:
            self.active -= 1

    def done(self, index, failed):
        with self.condition:
            self.terminals[index] = True
            self.condition.notify_all()

    def drain(self):
        with self.condition:
            if not self.condition.wait_for(lambda: all(self.terminals), timeout=3):
                raise RuntimeError('test native drain timed out')
        return 0

    def abort(self):
        self.aborted = True
        self.release.set()
        self.events.append('abort')

    def close(self):
        self.close_calls += 1
        self.events.append('close')
        self.unsafe_retire = self.active != 0 or any(ref() is None for ref in self.borrowed)
        if self.close_failure:
            raise RuntimeError('native close not acknowledged')
        self.closed = True


class SubmitLifetimeTests(unittest.TestCase):
    def setUp(self):
        prior = runtime._native_cleanup_quarantine
        self.addCleanup(setattr, runtime, '_native_cleanup_quarantine', prior)

    def _close_fixture(self, rt):
        # Boundary fakes have no queued SDK work. Production has no operation
        # that clears quarantine or retries native destruction.
        rt._native_failed_session = None
        rt.close()

    def setup_case(self, *, after_enqueue=True, delay=False, close_failure=False):
        sessions = []
        def create(regions, capacity):
            session = LifetimeSession(regions, capacity, delay=delay, close_failure=close_failure)
            sessions.append(session)
            return session
        def forbidden(*args, **kwargs):
            raise AssertionError('legacy handoff used')
        backend = contracts.RegionBackend(forbidden, lambda _: True, native_run=create)
        rt = runtime.RegionRuntime(contracts.RegionConfig(enabled=True, workers=4, regions=4, commands_per_region=8))
        self.addCleanup(self._close_fixture, rt)
        base = runtime.ThreadPoolExecutor
        primary = SubmissionFailure('lost Future after queue insertion' if after_enqueue else 'not queued')
        holder = {}
        class Executor:
            def __init__(self, **kwargs):
                self.real = base(**kwargs)
                self.calls = 0
                self.inject = True
                self.shutdown_calls = 0
                holder['pool'] = self
            def submit(self, *args, **kwargs):
                native = len(args) > 1 and args[1] is runtime._native_work
                if native:
                    self.calls += 1
                    if self.inject and self.calls == 3 and not after_enqueue:
                        raise primary
                future = self.real.submit(*args, **kwargs)
                if native and self.inject and self.calls == 3:
                    session = sessions[-1]
                    with session.condition:
                        if not session.condition.wait_for(lambda: session.prepared >= 3, timeout=3):
                            raise RuntimeError('test did not reach three prepared leases')
                    raise primary
                return future
            def shutdown(self, **kwargs):
                self.shutdown_calls += 1
                return self.real.shutdown(**kwargs)
        return rt, backend, sessions, Executor, holder, primary

    def test_lost_future_retires_current_session_while_cache_tls_is_alive(self):
        rt, backend, sessions, executor, holder, primary = self.setup_case(after_enqueue=True)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            with self.assertRaises(SubmissionFailure) as caught:
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(caught.exception, primary)
        self.assertTrue(sessions[0].closed)
        self.assertFalse(sessions[0].unsafe_retire, 'worker TLS died before queued executor cancellation')
        self.assertIs(rt._executor, holder['pool'])
        self.assertEqual(holder['pool'].shutdown_calls, 0)

    def test_normal_call_keeps_order_and_closes_pool_only_on_explicit_close(self):
        rt, backend, sessions, executor, holder, _primary = self.setup_case()
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            rt._prepare_native_executor()
            holder['pool'].inject = False
            self.assertEqual(rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend), list(range(4)))
        self.assertTrue(sessions[0].closed)
        self.assertEqual(sessions[0].close_calls, 1)
        self.assertIsNone(runtime._native_cleanup_quarantine)
        self.assertIsNone(rt._native_failed_session)
        self.assertEqual(holder['pool'].shutdown_calls, 0)
        rt.close()
        self.assertEqual(holder['pool'].shutdown_calls, 1)

    def test_running_unreturned_task_finishes_before_session_close_without_thread_exit(self):
        rt, backend, sessions, executor, holder, primary = self.setup_case(after_enqueue=True, delay=True)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            with self.assertRaises(SubmissionFailure) as caught:
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(caught.exception, primary)
        self.assertEqual(sessions[0].active, 0)
        self.assertFalse(sessions[0].unsafe_retire)
        self.assertEqual(holder['pool'].shutdown_calls, 0)

    def test_failed_submit_preserves_prior_accepted_lease_and_refuses_pool_shutdown(self):
        rt, backend, sessions, executor, holder, primary = self.setup_case(after_enqueue=True)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            rt._prepare_native_executor()
            pool = holder['pool']
            pool.inject = False
            self.assertEqual(rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend), list(range(4)))
            prior_borrowed = tuple(sessions[0].borrowed)
            pool.calls, pool.inject = 0, True
            with self.assertRaises(SubmissionFailure):
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(rt._executor, pool)
        self.assertTrue(sessions[1].closed, 'current session destruction was not acknowledged')
        self.assertTrue(all(ref() is not None for ref in prior_borrowed))
        for operation in (lambda: rt.map(lambda i, value: value, range(4), backend=backend), rt.close):
            with self.assertRaisesRegex(RuntimeError, 'quarantin|cleanup'):
                operation()
        rt.__del__()
        self.assertEqual(pool.shutdown_calls, 0)
        self.assertTrue(all(ref() is not None for ref in prior_borrowed), 'accepted prior lease lost its cache TLS owner')

    def test_submit_before_queue_has_no_hidden_task_or_thread_shutdown(self):
        rt, backend, sessions, executor, holder, primary = self.setup_case(after_enqueue=False)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            with self.assertRaises(SubmissionFailure) as caught:
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(caught.exception, primary)
        self.assertTrue(sessions[0].closed)
        self.assertFalse(sessions[0].unsafe_retire)
        self.assertEqual(holder['pool'].shutdown_calls, 0)

    def test_close_without_ack_retains_session_and_tls_and_refuses_map_and_close(self):
        rt, backend, sessions, executor, holder, primary = self.setup_case(close_failure=True)
        prior_quarantine = getattr(runtime, '_native_cleanup_quarantine', None)
        def fixture_cleanup():
            # Test-only release: no SDK or native owner exists in this fixture.
            # Production deliberately has no quarantine recovery/retry API.
            if hasattr(runtime, '_native_cleanup_quarantine'):
                runtime._native_cleanup_quarantine = prior_quarantine
            rt._native_failed_session = None
        self.addCleanup(fixture_cleanup)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            with self.assertRaises(SubmissionFailure) as caught:
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(caught.exception, primary)
        session, pool = sessions[0], holder['pool']
        self.assertIs(getattr(rt, '_native_failed_session', None), session, 'unacknowledged native session was not retained')
        self.assertIs(runtime._native_cleanup_quarantine, rt, 'failed runtime has no strong process-lifetime owner')
        self.assertFalse(session.closed)
        self.assertEqual(session.close_calls, 1)
        self.assertTrue(all(ref() is not None for ref in session.borrowed))
        for operation in (lambda: rt.map(lambda i, value: value, range(4), backend=backend), rt.close):
            with self.assertRaisesRegex(RuntimeError, 'quarantin|cleanup'):
                operation()
        self.assertEqual(pool.shutdown_calls, 0)
        self.assertEqual(session.close_calls, 1, 'failed native destroy was blindly retried')
        rt.__del__()  # Even an explicit finalizer cannot kill quarantined TLS.
        self.assertEqual(pool.shutdown_calls, 0)
        other = runtime.RegionRuntime(rt.config)
        self.addCleanup(self._close_fixture, other)
        with self.assertRaisesRegex(RuntimeError, 'quarantin|cleanup'):
            other.map(lambda i, value: value, range(4), backend=backend)
        self.assertIsNone(other._executor)

    def test_late_unreturned_pending_queue_entry_cannot_enter_closed_session(self):
        sessions, holder = [], {}
        release_late = threading.Event()
        base = runtime.ThreadPoolExecutor
        primary = SubmissionFailure('pending Future lost')
        def create(regions, capacity):
            session = LifetimeSession(regions, capacity, delay=True)
            old_close = session.close
            def close():
                old_close()
                release_late.set()
            session.close = close
            sessions.append(session)
            return session
        backend = contracts.RegionBackend(lambda *args: self.fail('legacy handoff'), lambda _: True, native_run=create)
        rt = runtime.RegionRuntime(contracts.RegionConfig(enabled=True, workers=2, regions=4, commands_per_region=8))
        self.addCleanup(self._close_fixture, rt)
        class Executor:
            def __init__(self, **kwargs):
                self.real, self.calls, self.late = base(**kwargs), 0, None
                holder['pool'] = self
            def submit(self, *args):
                if len(args) > 1 and args[1] is runtime._native_work:
                    self.calls += 1
                    if self.calls == 3:
                        def late():
                            if not release_late.wait(3):
                                raise RuntimeError('late-entry fixture timed out')
                            return args[0](*args[1:])
                        self.late = self.real.submit(late)
                        raise primary
                    future = self.real.submit(*args)
                    # Occupy both workers before queuing the unknown third.
                    with sessions[-1].condition:
                        if not sessions[-1].condition.wait_for(lambda: sessions[-1].prepared >= self.calls, timeout=3):
                            raise RuntimeError('known workers did not prepare')
                    return future
                return self.real.submit(*args)
            def shutdown(self, **kwargs):
                return self.real.shutdown(**kwargs)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=Executor):
            with self.assertRaises(SubmissionFailure) as caught:
                rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
        self.assertIs(caught.exception, primary)
        holder['pool'].late.result(timeout=3)
        self.assertEqual(sessions[0].prepared, 2, 'revoked late task still entered native preparation')
        self.assertTrue(sessions[0].closed)
        self.assertFalse(sessions[0].unsafe_retire)

    def test_close_ack_then_return_interrupt_never_retries_sdk_but_retains_pool(self):
        rt, backend, sessions, executor, holder, _primary = self.setup_case(after_enqueue=True)
        original_create = backend.native_run
        def create(*args):
            session = original_create(*args)
            old_close = session.close
            def close():
                old_close()
                raise KeyboardInterrupt('after native destruction acknowledgement')
            session.close = close
            return session
        backend = contracts.RegionBackend(backend.scope, backend.guard, native_run=create)
        with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor):
            rt._prepare_native_executor()
            holder['pool'].inject = False
            with self.assertRaises(KeyboardInterrupt):
                rt.map(lambda i, value: value, range(4), backend=backend)
        self.assertTrue(sessions[0].closed)
        self.assertEqual(sessions[0].close_calls, 1)
        self.assertIs(rt._native_failed_session, sessions[0])
        self.assertIs(runtime._native_cleanup_quarantine, rt)
        with self.assertRaisesRegex(RuntimeError, 'quarantin|cleanup'):
            rt.close()
        self.assertEqual(holder['pool'].shutdown_calls, 0)

    def test_pending_ticket_drops_context_and_input_owners_and_late_work_is_noop(self):
        owner = CacheOwner()
        reference = weakref.ref(owner)
        context_value = contextvars.ContextVar('captured-owner')
        token = context_value.set(owner)
        task = runtime._NativeSubmission(contextvars.copy_context(), owner)
        context_value.reset(token)
        del owner
        task.cancel_pending()
        gc.collect()
        self.assertIsNone(reference())
        task.run_context(lambda value: self.fail('revoked task entered native work'))
        task.wait(RuntimeError('primary'))

    def test_quarantine_registration_completes_after_lock_or_partial_publish_interrupt(self):
        for stage in ('before_lock', 'after_session_store', 'after_head_publish'):
            with self.subTest(stage=stage):
                rt, backend, sessions, executor, holder, primary = self.setup_case(close_failure=True)
                prior = getattr(runtime, '_native_cleanup_quarantine', None)
                original_lock = runtime._native_cleanup_quarantine_lock
                injected = []
                class Lock:
                    def __enter__(self):
                        if stage == 'before_lock' and sessions and sessions[0].close_calls and not injected:
                            injected.append(stage)
                            raise KeyboardInterrupt(stage)
                        return original_lock.__enter__()
                    def __exit__(self, *args):
                        result = original_lock.__exit__(*args)
                        if stage == 'after_head_publish' and runtime._native_cleanup_quarantine is rt and not injected:
                            injected.append(stage)
                            raise InterruptedError(stage)
                        return result
                def assign(instance, name, value):
                    object.__setattr__(instance, name, value)
                    if (stage == 'after_session_store' and instance is rt and name == '_native_failed_session'
                            and value is not None and not injected):
                        injected.append(stage)
                        raise KeyboardInterrupt(stage)
                observed = None
                try:
                    with patch.object(runtime, 'ThreadPoolExecutor', side_effect=executor), \
                            patch.object(runtime, '_native_cleanup_quarantine_lock', Lock()), \
                            patch.object(runtime.RegionRuntime, '__setattr__', assign):
                        try:
                            rt.map(lambda i, value: sessions[-1].prepare(i, value), range(4), backend=backend)
                        except BaseException as error:
                            observed = error
                    self.assertTrue(injected, 'fault injector did not reach quarantine registration')
                    self.assertIs(observed, primary, 'cleanup interruption replaced the original failure')
                    self.assertIs(rt._native_failed_session, sessions[0])
                    self.assertIs(runtime._native_cleanup_quarantine, rt)
                    self.assertIs(rt._native_quarantine_next, prior, 'idempotent registration created a self-cycle')
                    self.assertEqual(sessions[0].close_calls, 1)
                    with self.assertRaisesRegex(RuntimeError, 'quarantin|cleanup'):
                        rt.close()
                    self.assertEqual(holder['pool'].shutdown_calls, 0)
                finally:
                    # Fixture only, after simulated leases have no SDK effect.
                    runtime._native_cleanup_quarantine = prior
                    rt._native_failed_session = None

    def test_running_ack_waits_for_context_cleanup_and_repeated_interrupts(self):
        body_done, cleanup_entered, release, acknowledged = (threading.Event() for _ in range(4))
        class Context:
            def run(self, function, *args):
                try:
                    return function(*args)
                finally:
                    cleanup_entered.set()
                    if not release.wait(3):
                        raise RuntimeError('test context cleanup timed out')
        task = runtime._NativeSubmission(Context())
        worker = threading.Thread(target=task.run_context, args=(body_done.set,))
        worker.start()
        self.assertTrue(cleanup_entered.wait(2))
        self.assertTrue(body_done.is_set())
        task.cancel_pending()  # Cannot revoke a task that is already running.
        primary = RuntimeError('primary')
        old_wait = task._changed.wait
        calls = []
        def interrupted_wait(*args, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise KeyboardInterrupt('cleanup interrupt')
            return old_wait(*args, **kwargs)
        def wait():
            task.wait(primary)
            acknowledged.set()
        with patch.object(task._changed, 'wait', side_effect=interrupted_wait):
            waiter = threading.Thread(target=wait)
            waiter.start()
            try:
                self.assertFalse(acknowledged.wait(0.03), 'session.done was mistaken for complete context exit')
            finally:
                release.set()
                worker.join(3)
                waiter.join(3)
        self.assertFalse(worker.is_alive() or waiter.is_alive())
        self.assertTrue(acknowledged.is_set())
        self.assertEqual(sum('KeyboardInterrupt' in note for note in primary.__notes__), 2)


if __name__ == '__main__':
    unittest.main()
