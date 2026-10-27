"""Contracts for opt-in coarse Python regions. No Torch import is required."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class RegionConfig:
    enabled: bool = False
    workers: int = 4
    regions: int = 4
    commands_per_region: int = 32

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        for name, limit in (("workers", 64), ("regions", 256), ("commands_per_region", 8192)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in 1..{limit}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None):
        env = os.environ if env is None else env
        flag = env.get("FTLAB_HOST_REGIONS", "0")
        if flag not in ("0", "1"):
            raise ValueError("FTLAB_HOST_REGIONS must be 0 or 1")
        return cls(enabled=flag == "1",
                   workers=int(env.get("FTLAB_HOST_REGION_WORKERS", "4")),
                   regions=int(env.get("FTLAB_HOST_REGION_COUNT", "4")),
                   commands_per_region=int(env.get("FTLAB_HOST_REGION_COMMANDS", "32")))


@dataclass(frozen=True)
class RegionCommand:
    """An owned command; publish transfers ownership to the original backend.

    publish must preserve all owners until the backend's actual retirement.
    A successful return means ownership was transferred, not device completion.
    discard releases only commands which were never published. Both callbacks
    run outside scheduler locks. The backend owns uncertain/partial submissions;
    this runtime never retries publish or falls back after any worker has begun.
    """
    publish: Callable[[], None]
    discard: Callable[[], None] = lambda: None
    owners: tuple[Any, ...] = ()
    label: str = ""
    units: int = 1

    def __post_init__(self):
        if not callable(self.publish) or not callable(self.discard):
            raise TypeError("publish and discard must be callable")
        if not isinstance(self.owners, tuple) or not isinstance(self.label, str):
            raise TypeError("owners must be tuple and label must be str")
        if type(self.units) is not int or self.units < 1:
            raise ValueError("units must be a positive exact integer")


@dataclass(frozen=True)
class RegionBackend:
    """Backend contract, not an automatic proof that arbitrary Python is safe.

    scope(region_index, emit) must capture and restore backend TLS and route
    *all* device effects of admitted bodies to owned RegionCommands. It must
    reject unsupported synchronization/escape paths before they execute.
    guard runs once on the caller, before scheduling. A false guard causes the
    frontend to execute its original function, before any parallel work begins.
    Pure host fixtures can explicitly use nullcontext; Torch requires a real
    capture provider and an operation/callable capability policy.
    native_run, when set, must return an unopened lightweight session: only its
    open() acquires native owners, after runtime retains it for finally cleanup.
    A backend whose native plans borrow persistent worker TLS must declare
    native_retirement_required=True and return sessions with a stable callable
    retire_worker_tls. It is a cold original-queue/device retirement barrier,
    returning exact True only after its domain is safe to stop these workers.
    Pure host/native fixtures without that declaration provide no SDK ACK.
    """
    scope: Callable[[int, Callable[[RegionCommand], None]], AbstractContextManager]
    guard: Callable[[Mapping[str, Any]], bool]
    name: str = "explicit"
    native_run: Callable[[int, int], Any] | None = None
    native_retirement_required: bool = False

    def __post_init__(self):
        if not callable(self.scope) or not callable(self.guard):
            raise TypeError("backend scope and guard must be callable")
        if not isinstance(self.name, str):
            raise TypeError("backend name must be str")
        if self.native_run is not None and not callable(self.native_run):
            raise TypeError("native_run must be callable or None")
        if type(self.native_retirement_required) is not bool:
            raise TypeError("native_retirement_required must be bool")
        if self.native_retirement_required and self.native_run is None:
            raise ValueError("native retirement requires a native session factory")

    @classmethod
    def pure_host(cls, guard):
        """Explicit opt-in for reviewed pure host callables, never Torch auto-detection."""
        return cls(lambda _region, _emit: nullcontext(), guard, "reviewed-pure-host")


class _NativeRetirementSession:
    """Nominal marker for the built-in native session, not an optional duck flag.

    Its concrete implementation must bind the actual bridge's real mode and
    validate a real cold barrier before its open acquires native ownership.
    """
    @property
    def native_retirement_required(self):
        raise NotImplementedError


class RegionCancelled(RuntimeError):
    pass


class UnsupportedRegion(RuntimeError):
    pass
