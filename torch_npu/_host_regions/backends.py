"""Fail-closed adapters for explicit, audited native command capture providers.

Importing this module does not patch TorchNPU, import Torch, or install a no-op
capture. An environment flag cannot turn an unaudited backend into a safe one.
Providers must own every deferred payload and reject unsupported device effects
before they happen. The adapter cannot establish those facts from Python.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from threading import Lock, get_ident
from typing import Any, Protocol

from .contracts import RegionBackend, RegionCommand, UnsupportedRegion


@dataclass(frozen=True)
class CaptureCapabilities:
    """Provider audit declaration, not a runtime proof of native correctness."""

    abi_version: int = 1
    backend: str = ""
    audit_id: str = ""
    owns_commands: bool = False
    captures_all_effects: bool = False
    restores_thread_state: bool = False
    rejects_unsupported_before_effect: bool = False
    preserves_owner_retirement: bool = False
    operation_policy: bool = False
    audited: bool = False

    def validate(self) -> None:
        if type(self.abi_version) is not int or self.abi_version != 1:
            raise ValueError("capture provider abi_version must be 1")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("capture provider backend must be named")
        if not isinstance(self.audit_id, str) or not self.audit_id.strip():
            raise ValueError("capture provider audit_id must identify its audit")
        for name in ("owns_commands", "captures_all_effects", "restores_thread_state",
                     "rejects_unsupported_before_effect", "preserves_owner_retirement",
                     "operation_policy", "audited"):
            if getattr(self, name) is not True:
                raise ValueError(f"capture provider requires {name}=True")


class CapturedCommand:
    """One owned publication handle, with mutually exclusive terminal actions.

    publish must transfer ALL resources to the original queue and its correct
    retirement mechanism. The owners tuple keeps native handles alive before
    that transfer; it does not extend device memory lifetime on its own. If
    publication throws, the provider owns any uncertain/partial submission.
    There is never a retry or a call to discard after publication starts.
    """

    def __init__(self, publish: Callable[[], None], discard: Callable[[], None],
                 *, owners: tuple[Any, ...] = (), label: str = "", units: int = 1):
        if not callable(publish) or not callable(discard):
            raise TypeError("publish and discard must be callable")
        if not isinstance(owners, tuple) or not isinstance(label, str):
            raise TypeError("owners must be tuple and label must be str")
        if type(units) is not int or units < 1:
            raise ValueError("units must be a positive exact integer")
        self._publish = publish
        self._discard = discard
        self._owners = owners
        self._state = "pending"
        self._emitted = False
        self._lock = Lock()
        self.label = label
        self.units = units

    def publish(self) -> None:
        with self._lock:
            if self._state != "pending":
                raise RuntimeError(f"command already {self._state}")
            self._state = "publishing"
            callback = self._publish
            held = self._discard, self._owners
            self._publish = self._discard = None
            self._owners = ()
        try:
            callback()
        except BaseException:
            with self._lock:
                self._state = "indeterminate"
            raise
        else:
            with self._lock:
                self._state = "published"
        finally:
            # Any native finalizer or Python owner destructor runs outside lock.
            del held

    def discard(self) -> None:
        with self._lock:
            if self._state != "pending":
                return
            self._state = "discarded"
            callback = self._discard
            held = self._publish, self._owners
            self._publish = self._discard = None
            self._owners = ()
        try:
            callback()
        finally:
            del held

    def as_region_command(self) -> RegionCommand:
        with self._lock:
            if self._emitted:
                raise RuntimeError("capture command already emitted")
            if self._state != "pending":
                raise RuntimeError(f"command already {self._state}")
            self._emitted = True
        return RegionCommand(self.publish, self.discard, owners=(self,), label=self.label, units=self.units)


class CaptureProvider(Protocol):
    capabilities: CaptureCapabilities

    def capture(self, region: int,
                sink: Callable[[CapturedCommand], None]) -> AbstractContextManager:
        """Install worker TLS, route all admitted effects, then restore TLS.

        Calls to sink are synchronous on the worker that entered this context.
        A successful sink call transfers command ownership to the scheduler.
        A failing sink call must not cause provider-side retry or direct enqueue.
        """
        ...


def make_capture_backend(provider: CaptureProvider, *,
                         guard: Callable[[Mapping[str, Any]], bool]) -> RegionBackend:
    """Adapt an explicitly supplied provider; no automatic native integration.

    guard must validate the actual callable/op policy, TLS mode, inputs, alias
    and effects before any worker starts. False means serial fallback there;
    unsupported effects discovered after entry abort the invocation instead.
    """
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, CaptureCapabilities):
        raise TypeError("capture provider requires CaptureCapabilities")
    capabilities.validate()
    if not callable(getattr(provider, "capture", None)) or not callable(guard):
        raise TypeError("capture and guard must be callable")

    @contextmanager
    def scope(region, emit):
        if type(region) is not int or region < 0 or not callable(emit):
            raise ValueError("region must be nonnegative and emit callable")
        worker = get_ident()
        opened = True

        def sink(command):
            if not isinstance(command, CapturedCommand):
                raise TypeError("provider must emit a CapturedCommand")
            # A duplicate emission must not discard the first accepted handle.
            wrapped = command.as_region_command()
            try:
                if not opened:
                    raise RuntimeError("capture scope is closed")
                if get_ident() != worker:
                    raise RuntimeError("capture sink called from a different thread")
                emit(wrapped)
            except BaseException:
                # emit owns accepted handles; idempotence also covers a runtime
                # which discarded before raising on cancellation.
                command.discard()
                raise

        try:
            manager = provider.capture(region, sink)
            manager.__enter__()
            try:
                yield
            except BaseException as exc:
                # Providers may clean up but cannot suppress a failed region.
                manager.__exit__(type(exc), exc, exc.__traceback__)
                raise
            else:
                manager.__exit__(None, None, None)
        finally:
            opened = False

    return RegionBackend(scope, guard,
                         name=f"capture:{capabilities.backend}:{capabilities.audit_id}",
                         native_run=getattr(provider, "native_run", None),
                         native_retirement_required=getattr(provider, "native_retirement_required", False))


def backend_available(provider: CaptureProvider | None, *, target: str | None = None) -> bool:
    """Inspect declarations without installing capture, importing Torch or starting workers."""
    caps = getattr(provider, "capabilities", None)
    if not isinstance(caps, CaptureCapabilities):
        return False
    try:
        caps.validate()
    except ValueError:
        return False
    return (target is None or caps.backend == target) and callable(getattr(provider, "capture", None))


def make_torch_npu_backend(provider: CaptureProvider | None = None, *,
                           guard: Callable[[Mapping[str, Any]], bool]) -> RegionBackend:
    """Require a real integrated TorchNPU provider; no shipped provider exists yet."""
    if provider is None:
        raise UnsupportedRegion("TorchNPU requires an integrated, audited capture provider")
    if not backend_available(provider, target="torch_npu"):
        raise UnsupportedRegion("TorchNPU requires complete audited backend='torch_npu' capabilities")
    return make_capture_backend(provider, guard=guard)
