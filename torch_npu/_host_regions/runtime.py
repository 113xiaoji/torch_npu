"""Bounded coarse Python regions with source-order command publication.

This layer schedules already-admitted bodies. The frontend must call the
backend guard before map(), and the backend must intercept every device effect.
An arbitrary body cannot be forcibly cancelled: failure wakes blocked emitters
and joins running bodies before releasing their owners. No performance or
automatic Torch interception is implied by this portable implementation.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import contextvars
import sys
import threading
import weakref

from .contracts import RegionBackend, RegionCancelled, RegionCommand, RegionConfig, _NativeRetirementSession


_local = threading.local()
# A failed invocation cannot acknowledge retirement of earlier accepted queue
# leases. Keep its SDK worker TLS and failed owners strongly reachable, close
# native admission, and never retry SDK destruction from Python cleanup.
# This is quarantine, not resource retirement.
_native_cleanup_quarantine = None
_native_cleanup_quarantine_lock = threading.Lock()
# Successful publication is not a host-consumption ACK. Strongly retain each
# real runtime once, before its first native open, until a cold barrier and
# worker shutdown finish. This also makes GC safe without SDK calls in __del__.
_native_retirement_owners = {}
_native_retirement_owners_lock = threading.Lock()
_native_thread_exit_registered = False
_native_thread_exit_stop = threading.Event()


def _retain_native_retirement_owner(runtime):
    global _native_thread_exit_registered
    with _native_retirement_owners_lock:
        if getattr(threading, '_SHUTTING_DOWN', True) is not False:
            raise RuntimeError("interpreter shutdown has closed native retirement admission")
        if not _native_thread_exit_registered:
            # ThreadPoolExecutor was imported above, so its _python_exit hook
            # is already registered. CPython calls these hooks in reverse:
            # this cold barrier runs before futures can stop worker TLS.
            threading._register_atexit(_before_native_thread_shutdown)
            _native_thread_exit_registered = True
        _native_retirement_owners[id(runtime)] = runtime


def _before_native_thread_shutdown():
    try:
        with _native_retirement_owners_lock:
            owners = tuple(_native_retirement_owners.values())
        for owner in owners:
            owner.close()
    except BaseException as error:
        # Continuing into futures._python_exit would destroy SDK TLS with
        # no accepted-lease retirement proof. This includes interruption while
        # snapshotting the registry, before an individual close has started.
        # Never retry an uncertain SDK barrier. The external controller must
        # terminate this failed process; no normal retirement/exit is claimed.
        try:
            cause = getattr(error, '__cause__', None)
            sys.stderr.write(f"native worker retirement fail-stop: {type(error).__name__}: {error}; cause={cause!r}\n")
            sys.stderr.flush()
        except BaseException:
            pass
        while True:
            try:
                _native_thread_exit_stop.wait()
            except BaseException:
                pass


def in_region(runtime=None) -> bool:
    """Whether this thread is inside a region invocation or its callbacks.

    Frontends use this before transformation to execute nested original Python
    in the existing capture scope. With runtime, require that exact instance.
    It does not report other threads' activity.
    """
    active = getattr(_local, "runtime", None)
    return active is not None and (runtime is None or active is runtime)


def emit_command(command: RegionCommand) -> None:
    """Transfer one owned command to the current region, once.

    The command is published on the coordinator or discarded after abort.
    Outside a region this rejects without taking ownership. A publish exception
    is an uncertain submission: the backend owns it and it is never discarded.
    """
    emit = getattr(_local, "emit", None)
    if emit is None:
        raise RuntimeError("emit_command requires an active region body or backend scope")
    emit(command)


@contextmanager
def _binding(runtime, emitter, emitter_token=None):
    previous = (getattr(_local, "runtime", None), getattr(_local, "emit", None),
                getattr(_local, "emitter_token", None))
    _local.runtime, _local.emit, _local.emitter_token = runtime, emitter, emitter_token
    try:
        yield
    finally:
        _local.runtime, _local.emit, _local.emitter_token = previous


class _Region:
    def __init__(self, index, start, stop):
        self.index, self.start, self.stop = index, start, stop
        self.condition = threading.Condition()
        self.commands = deque()
        self.queued_units = 0
        self.waiting_units = 0
        self.results = []
        self.done = False
        self.error = None


class _NativeRegion:
    __slots__ = ("index", "start", "stop", "results", "error")

    def __init__(self, index, start, stop):
        self.index, self.start, self.stop = index, start, stop
        self.results, self.error = [], None


class _NativeSubmission:
    """Own a queued call until one worker claims it or the caller revokes it.

    submit() can enqueue work and then raise without returning its Future.
    The executor therefore receives only this ticket and the worker function;
    a revoked queue entry retains no native session, context or input owners.
    """
    def __init__(self, context, *args):
        self._changed = threading.Condition()
        self._call = (context, args)
        self._running = self._done = False

    def run_context(self, function):
        with self._changed:
            call, self._call = self._call, None
            if call is None:
                return
            self._running = True
        try:
            return call[0].run(function, *call[1])
        finally:
            # Include context.run restoration and drop captured owners before
            # acknowledging quiescence; session.done alone is too early.
            call = None
            with self._changed:
                self._done = True
                self._changed.notify_all()

    def cancel_pending(self):
        call = None
        with self._changed:
            if not self._running and not self._done:
                call, self._call = self._call, None
                self._done = True
                self._changed.notify_all()
        # User/context owners can run destructors. Never release under a lock.
        call = None

    def wait(self, primary):
        while True:
            try:
                with self._changed:
                    while not self._done:
                        self._changed.wait()
                return
            except (KeyboardInterrupt, InterruptedError) as error:
                primary.add_note(f"native task acknowledgement interrupted by {type(error).__name__}; waiting for completion")


def _abort_native(session, primary):
    while True:
        try:
            session.abort()
            return
        except (KeyboardInterrupt, InterruptedError) as error:
            primary.add_note(f"native abort interrupted by {type(error).__name__}; retrying control acknowledgement")
        except BaseException as error:
            primary.add_note(f"native abort failed: {type(error).__name__}: {error}")
            return


def _native_work(runtime, session, region, body, items):
    """Python owns original bodies/errors; native owns capture/TLS/handoff."""
    error = None
    try:
        with _binding(runtime, None):
            session.begin(region.index)
            try:
                for index in range(region.start, region.stop):
                    region.results.append(body(index, items[index]))
            except BaseException as caught:
                error = caught
            finally:
                try:
                    session.end()
                except BaseException as caught:
                    if error is None:
                        error = caught
                    else:
                        error.add_note(f"native capture end/TLS restoration failed: {type(caught).__name__}: {caught}")
    except BaseException as caught:
        if error is None:
            error = caught
        elif caught is not error:
            error.add_note(f"native worker cleanup failed: {type(caught).__name__}: {caught}")
    # This slot is visible before native done releases the caller's head wait.
    # It retains no worker traceback; native stores only the failure bit.
    region.error = error.with_traceback(None) if error is not None else None
    try:
        session.done(region.index, error is not None)
    except BaseException as caught:
        if region.error is None:
            region.error = caught.with_traceback(None)
        else:
            region.error.add_note(f"native terminal notification failed: {type(caught).__name__}: {caught}")
        # A failed done may not have marked terminal; wake drain before return.
        _abort_native(session, caught)
        raise


class _Run:
    def __init__(self, runtime, regions):
        self.runtime, self.regions = runtime, regions
        self.aborted = threading.Event()
        self.registry_lock = threading.Lock()
        # Weak entries detect duplicate emission without retaining owners after
        # publication. The caller must not reuse a command after transferring it.
        self.commands = weakref.WeakValueDictionary()
        self.cleanup_errors = []

    def claim(self, command):
        if not isinstance(command, RegionCommand):
            raise TypeError("emit expects RegionCommand")
        with self.registry_lock:
            if id(command) in self.commands:
                raise ValueError("a RegionCommand may be emitted only once")
            self.commands[id(command)] = command

    def discard(self, command):
        try:
            with _binding(self.runtime, None):
                command.discard()
        except BaseException as error:
            # Continue releasing the rest; the original error remains primary.
            with self.registry_lock:
                self.cleanup_errors.append(f"command discard failed: {type(error).__name__}: {error}")

    def emit(self, region, command, emitter_token):
        if (getattr(_local, "runtime", None) is not self.runtime
                or getattr(_local, "emitter_token", None) is not emitter_token):
            raise RuntimeError("backend emitter is only valid in its active region")
        self.claim(command)
        accepted = False
        try:
            if command.units > self.runtime.config.commands_per_region:
                raise ValueError("command block exceeds region capacity")
            with region.condition:
                while (region.queued_units + command.units > self.runtime.config.commands_per_region
                       and not self.aborted.is_set() and not region.done):
                    region.waiting_units = command.units
                    try:
                        region.condition.wait()
                    finally:
                        region.waiting_units = 0
                if self.aborted.is_set() or region.done:
                    raise RegionCancelled("region invocation has stopped")
                region.commands.append(command)
                region.queued_units += command.units
                accepted = True
                if len(region.commands) == 1:
                    region.condition.notify()
        finally:
            if not accepted:
                self.discard(command)

    def abort(self):
        self.aborted.set()
        for region in self.regions:
            with region.condition:
                region.condition.notify_all()

    def release(self):
        """Called after joining workers; user callbacks never run under locks."""
        for region in self.regions:
            while region.commands:
                command = region.commands.popleft()
                region.queued_units -= command.units
                self.discard(command)
                del command
            region.results.clear()
            region.error = None


def _work(run, region, body, items, backend):
    # A token preserves active-emitter identity without a self-referential
    # closure retaining the completed run until a cyclic GC collection.
    emitter_token = object()
    emitter = lambda command: run.emit(region, command, emitter_token)
    error = None
    body_error = None
    try:
        if run.aborted.is_set():
            return
        with _binding(run.runtime, emitter, emitter_token), backend.scope(region.index, emitter):
            try:
                for index in range(region.start, region.stop):
                    if run.aborted.is_set():
                        raise RegionCancelled("region invocation has stopped")
                    region.results.append(body(index, items[index]))
            except BaseException as caught:
                body_error = caught
                raise
        # An adapter's __exit__ may suppress exceptions; it cannot turn a
        # failed body into a successful region with missing results.
        error = body_error
    except BaseException as caught:
        # Avoid keeping the entire worker stack (body, inputs, backend scope)
        # alive through a saved error. The exception type and message survive.
        error = body_error if body_error is not None else caught
        if error is not caught:
            error.add_note(f"backend scope restoration also failed: {type(caught).__name__}: {caught}")
    finally:
        if error is not None:
            error = error.with_traceback(None)
        with region.condition:
            region.error = error
            region.done = True
            region.condition.notify_all()


class RegionRuntime:
    """Persistent workers; at most one invocation per runtime.

    Concurrent/reentrant map or close calls reject immediately. close is
    idempotent once idle. A backend must itself reject unsupported effects and
    preserve tensor/storage owners until actual device retirement. Python
    contextvars are copied for each region; backend TLS belongs to scope().
    """

    def __init__(self, config: RegionConfig | None = None):
        self.config = RegionConfig.from_env() if config is None else config
        if not isinstance(self.config, RegionConfig):
            raise TypeError("config must be RegionConfig")
        self._gate = threading.Lock()
        self._closed = False
        self._executor = None
        self._native_executor_ready = False
        self._native_failed_session = None
        self._native_quarantine_next = None
        self._native_retirement_barriers = []
        self._native_worker_ids = set()
        self._native_cold_failure = None

    @property
    def closed(self) -> bool:
        """Closed-state snapshot for preflight; map still checks atomically."""
        return self._closed

    def map(self, body, items, *, backend: RegionBackend) -> list:
        """Run body(index, item) in contiguous regions and return ordered values.

        items are frozen before scheduling. No guard is guessed here: explicit
        callers must perform their own admission check before calling map.
        Body exceptions are surfaced after their source-order command prefix;
        there is no retry or fallback after body execution begins.
        """
        if getattr(_local, "runtime", None) is not None:
            raise RuntimeError("cannot enter a runtime from an active region invocation")
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("runtime already has an active invocation")
        try:
            if self._closed:
                raise RuntimeError("runtime is closed")
            if self._native_failed_session is not None:
                raise RuntimeError("native cleanup is unacknowledged; runtime is quarantined")
            if not callable(body) or not isinstance(backend, RegionBackend):
                raise TypeError("map requires callable body and RegionBackend")
            items = tuple(items)
            if not items:
                return []
            count = min(self.config.regions, len(items))
            if self.config.enabled and backend.native_run is not None:
                regions = [_NativeRegion(i, i * len(items) // count, (i + 1) * len(items) // count)
                           for i in range(count)]
                with _binding(self, None):
                    return self._native_parallel(body, items, backend, regions)
            regions = [_Region(i, i * len(items) // count, (i + 1) * len(items) // count)
                       for i in range(count)]
            with _binding(self, None):
                if not self.config.enabled:
                    return self._serial(body, items, backend, regions)
                return self._parallel(body, items, backend, regions)
        finally:
            self._gate.release()

    def _serial(self, body, items, backend, regions):
        run, results = _Run(self, regions), []
        publish_error = None
        body_error = None
        emitter_token = object()

        def emit(command):
            nonlocal publish_error
            if (getattr(_local, "runtime", None) is not self
                    or getattr(_local, "emitter_token", None) is not emitter_token):
                raise RuntimeError("backend emitter is only valid in its active region")
            run.claim(command)
            if command.units > self.config.commands_per_region:
                run.discard(command)
                raise ValueError("command block exceeds region capacity")
            if run.aborted.is_set():
                run.discard(command)
                raise RegionCancelled("region invocation has stopped")
            try:
                with _binding(self, None):
                    command.publish()
            except BaseException as caught:
                publish_error = caught.with_traceback(None)
                run.abort()
                raise

        try:
            for region in regions:
                with _binding(self, emit, emitter_token), backend.scope(region.index, emit):
                    try:
                        for index in range(region.start, region.stop):
                            results.append(body(index, items[index]))
                            if publish_error is not None:
                                raise publish_error
                    except BaseException as caught:
                        body_error = caught
                        raise
                if publish_error is not None:
                    raise publish_error
                if body_error is not None:
                    raise body_error
            return results
        except BaseException as error:
            run.abort()
            results.clear()
            primary = publish_error if publish_error is not None else body_error
            primary = error if primary is None else primary
            if primary is not error:
                primary.add_note(f"backend scope restoration also failed: {type(error).__name__}: {error}")
            for note in run.cleanup_errors:
                primary.add_note(note)
            if primary is not error:
                raise primary from None
            raise

    def _parallel(self, body, items, backend, regions):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.config.workers,
                                                thread_name_prefix="ftlab-region")
        run, futures, results = _Run(self, regions), [], []
        command = None
        try:
            # Submit the earliest region first. Every region has independent
            # queue credits; later workers cannot exhaust the head's buffer.
            for region in regions:
                context = contextvars.copy_context()
                futures.append(self._executor.submit(context.run, _work, run, region,
                                                     body, items, backend))
            for region in regions:
                while True:
                    with region.condition:
                        while not region.commands and not region.done:
                            region.condition.wait()
                        if region.commands:
                            # A producer may need several credits for one
                            # block. Wake only when that whole block fits.
                            needed = region.waiting_units or 1
                            lacked_space = region.queued_units + needed > self.config.commands_per_region
                            command = region.commands.popleft()
                            region.queued_units -= command.units
                            if lacked_space and region.queued_units + needed <= self.config.commands_per_region:
                                region.condition.notify()
                        else:
                            if region.error is not None:
                                raise region.error
                            break
                    # Ownership has left the runtime before calling publish.
                    # If publication raises, it must never be retried/discarded.
                    try:
                        command.publish()
                    finally:
                        command = None
                results.extend(region.results)
                region.results.clear()
            for future in futures:
                future.result()
            return results
        except BaseException as error:
            run.abort()
            # Let queued tasks observe abort and return, then join all of them.
            # Future.cancel() alone would leave executor work items retaining
            # body/input owners after map returned.
            for future in futures:
                while True:
                    try:
                        future.result()
                        break
                    except BaseException as wait_error:
                        if future.done():
                            # _work records body errors in source order. A
                            # finished task cannot supersede the primary error.
                            break
                        # A second interrupt while waiting is not evidence the
                        # task stopped. Keep the gate/owners and resume waiting.
                        error.add_note(f"cleanup wait interrupted by {type(wait_error).__name__}; waiting for worker completion")
            run.release()
            results.clear()
            for note in run.cleanup_errors:
                error.add_note(note)
            raise
        finally:
            # No future keeps the completed call's bodies/results/errors alive.
            futures.clear()
            run.release()

    def _native_parallel(self, body, items, backend, regions):
        with _native_cleanup_quarantine_lock:
            if _native_cleanup_quarantine is not None:
                raise RuntimeError("native cleanup is unacknowledged; native runtimes are quarantined")
        # Preallocate tracking before creating native owners or submitting work.
        futures = [None] * len(regions)
        submissions = [None] * len(regions)
        results, primary, submit_failed, session = [], None, False, None
        self._prepare_native_executor()
        try:
            session = backend.native_run(len(regions), self.config.commands_per_region)
            if backend.native_retirement_required or (
                    isinstance(session, _NativeRetirementSession) and session.native_retirement_required):
                barrier = getattr(session, 'retire_worker_tls', None)
                if not callable(barrier):
                    raise RuntimeError("native worker retirement capability is missing before session open")
                if not any(saved is barrier for saved in self._native_retirement_barriers):
                    self._native_retirement_barriers.append(barrier)
                _retain_native_retirement_owner(self)
            session.open()
            for region in regions:
                context = contextvars.copy_context()
                submission = _NativeSubmission(context, self, session, region, body, items)
                submissions[region.index] = submission
                try:
                    futures[region.index] = self._executor.submit(submission.run_context, _native_work)
                except BaseException:
                    submit_failed = True
                    raise
            status = session.drain()
            if type(status) is not int or not 0 <= status <= len(regions):
                raise RuntimeError("native drain returned an invalid status")
            if status:
                error = regions[status - 1].error
                if error is None:
                    raise RuntimeError("native drain reported failure without a Python exception")
                raise error
            for future in futures:
                future.result()
            for region in regions:
                if region.error is not None:
                    raise region.error
                results.extend(region.results)
        except BaseException as error:
            primary = error
            if session is not None:
                _abort_native(session, primary)
            if submit_failed:
                # Revoke calls that never started, then join running calls by
                # their pre-held tickets, including a task without a Future.
                # SDK cache state belongs to persistent worker TLS: shutting
                # the pool down here could invalidate queued executors before
                # session.close cancels them. Earlier accepted FIFO leases also
                # cannot be inferred retired merely from this run's close.
                for submission in submissions:
                    if submission is None:
                        continue
                    while True:
                        try:
                            submission.cancel_pending()
                            break
                        except (KeyboardInterrupt, InterruptedError) as cancel_error:
                            primary.add_note(f"native pending cancellation interrupted by {type(cancel_error).__name__}; retrying acknowledgement")
                for submission in submissions:
                    if submission is not None:
                        submission.wait(primary)
            # Submitted bodies may still own native scopes or be credit-blocked.
            # Never destroy the session or clear their owners until all joined.
            for future in futures:
                if future is None:
                    continue
                while True:
                    try:
                        future.result()
                        break
                    except BaseException as wait_error:
                        if future.done():
                            if wait_error is not primary:
                                primary.add_note(f"native worker also failed: {type(wait_error).__name__}: {wait_error}")
                            break
                        primary.add_note(f"native cleanup wait interrupted by {type(wait_error).__name__}; waiting for completion")
        finally:
            try:
                if session is not None:
                    session.close()
            except BaseException as close_error:
                if primary is None:
                    primary = close_error
                else:
                    primary.add_note(f"native run destruction failed: {type(close_error).__name__}: {close_error}")
            if primary is not None:
                # Even acknowledged current-session destruction does not retire
                # leases accepted by earlier invocations on the original queue.
                # No global queue retirement ACK exists here, so error unwinding
                # (including __exit__/GC) must not shut down this pool.
                self._quarantine_native_session(session, primary)
            futures.clear()
            submissions.clear()
            for region in regions:
                region.results.clear()
                region.error = None
        if primary is not None:
            results.clear()
            raise primary
        return results

    def _quarantine_native_session(self, session, primary):
        global _native_cleanup_quarantine
        while True:
            try:
                with _native_cleanup_quarantine_lock:
                    if self._native_failed_session is None:
                        # Creation can fail before returning a session; the
                        # runtime still owns TLS borrowed by prior queue leases.
                        self._native_failed_session = session if session is not None else self
                    owner = _native_cleanup_quarantine
                    while owner is not None:
                        if owner is self:
                            return
                        owner = owner._native_quarantine_next
                    # Retry safely if interrupted after storing the session,
                    # after linking next, or after publishing the global head.
                    self._native_quarantine_next = _native_cleanup_quarantine
                    _native_cleanup_quarantine = self
                return
            except (KeyboardInterrupt, InterruptedError) as error:
                primary.add_note(f"native quarantine registration interrupted by {type(error).__name__}; retaining worker TLS before returning")

    def _shutdown_native_executor(self, primary=None):
        executor = self._executor
        if executor is None:
            # ACK/join may have completed before registry removal was
            # interrupted. A retry must finish only that bookkeeping.
            with _native_retirement_owners_lock:
                _native_retirement_owners.pop(id(self), None)
            return
        if threading.get_ident() in self._native_worker_ids:
            raise RuntimeError("cannot stop a runtime from its own worker thread")
        if self._native_failed_session is not None:
            raise RuntimeError("quarantined worker TLS has no retirement acknowledgement") from self._native_cold_failure
        try:
            for barrier in self._native_retirement_barriers:
                if barrier() is not True:
                    raise RuntimeError("native worker retirement barrier did not acknowledge its domain")
            self._native_retirement_barriers.clear()  # Never repeat SDK work during join.
        except BaseException as error:
            self._native_cold_failure = error
            self._quarantine_native_session(None, error)
            raise
        interrupted = None
        while True:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
                break
            except (KeyboardInterrupt, InterruptedError) as error:
                if primary is not None:
                    primary.add_note(f"native executor shutdown interrupted by {type(error).__name__}; waiting for all threads")
                elif interrupted is None:
                    interrupted = error
                else:
                    interrupted.add_note(f"native executor shutdown interrupted again by {type(error).__name__}")
        self._executor = None
        self._native_executor_ready = False
        self._native_worker_ids.clear()
        with _native_retirement_owners_lock:
            _native_retirement_owners.pop(id(self), None)
        if interrupted is not None:
            raise interrupted

    def _prepare_native_executor(self):
        if self._native_executor_ready:
            return
        # An old callback pool can carry idle tokens and inherited contexts.
        # Replace it cold so prestart reliably occupies every new worker.
        if self._executor is not None:
            self._shutdown_native_executor()
        release, warm = threading.Event(), []
        worker_ids, worker_ids_lock = set(), threading.Lock()
        try:
            # Hold each dummy task so every persistent worker is started. Start
            # threads under empty Python context, before creating any NativeRun.
            def prepare():
                self._executor = ThreadPoolExecutor(max_workers=self.config.workers,
                                                    thread_name_prefix="ftlab-region")
                def prestart():
                    with worker_ids_lock:
                        worker_ids.add(threading.get_ident())
                    return release.wait()
                for _ in range(self.config.workers):
                    warm.append(self._executor.submit(prestart))
            contextvars.Context().run(prepare)
            release.set()
            for future in warm:
                future.result()
            self._native_worker_ids = worker_ids
            self._native_executor_ready = True
        except BaseException as error:
            release.set()
            self._shutdown_native_executor(error)
            raise

    def close(self):
        if getattr(_local, "runtime", None) is not None:
            raise RuntimeError("cannot close a runtime from an active region invocation")
        if threading.get_ident() in self._native_worker_ids:
            raise RuntimeError("cannot close a runtime from its own worker thread")
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("cannot close runtime during an active invocation")
        try:
            if self._closed:
                return
            if self._native_failed_session is not None:
                raise RuntimeError("native cleanup is unacknowledged; quarantined worker TLS must remain alive") from self._native_cold_failure
            self._shutdown_native_executor()
            self._closed = True
        finally:
            self._gate.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("runtime is closed")
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):
        # Explicit close is preferred. Never wait or reenter from destruction.
        executor = getattr(self, "_executor", None)
        if executor is not None and getattr(self, "_native_retirement_barriers", None):
            # Normally the strong retirement registry prevents reaching GC.
            # This defensive path must still never stop borrowed SDK TLS.
            self._quarantine_native_session(None, RuntimeError("real runtime reached GC without a retirement ACK"))
        elif executor is not None and getattr(self, "_native_failed_session", None) is None:
            executor.shutdown(wait=False, cancel_futures=True)
