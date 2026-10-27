"""Explicit framework bootstrap for environment-controlled loop installation.

Importing this module never patches Torch. A framework calls install_from_env
once with a backend factory that certifies each accepted plan. Missing native
capture must yield None; a null capture backend is never substituted.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import threading
import types
import weakref
from dataclasses import replace

from .contracts import RegionBackend, RegionConfig
from .frontend import compile_loop
from .runtime import RegionRuntime, in_region

_hooks_lock = threading.RLock()
_hooks = weakref.WeakKeyDictionary()


def _own_dict(instance):
    if type(instance).__getattribute__ is not object.__getattribute__:
        return None
    descriptor = inspect.getattr_static(type(instance), "__dict__", None)
    if not isinstance(descriptor, types.GetSetDescriptorType):
        return None
    value = object.__getattribute__(instance, "__dict__")
    return value if type(value) is dict else None


def _boolean_assumptions(fn, instance, own):
    """Discover only simple boolean If tests; frontend rechecks them per call."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        parameters = list(inspect.signature(fn).parameters)
    except (OSError, TypeError, SyntaxError, ValueError):
        return {}
    if not parameters:
        return {}
    parameter = parameters[0]
    constants = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == parameter):
            name = node.test.attr
            descriptor = inspect.getattr_static(type(instance), name, None)
            if hasattr(type(descriptor), "__set__") or hasattr(type(descriptor), "__delete__"):
                continue
            value = own.get(name)
            if type(value) is bool:
                constants[parameter + "." + name] = value
    return constants


class InstanceInstallation:
    def __init__(self, instance, replacements=(), reports=()):
        self._instance = weakref.ref(instance)
        self._replacements = list(replacements)
        self.reports = tuple(reports)

    @property
    def installed_count(self):
        return len(self._replacements)

    def close(self):
        instance = self._instance()
        if instance is not None:
            own = _own_dict(instance)
            if own is not None:
                for name, wrapper in self._replacements:
                    current = own.get(name)
                    if isinstance(current, types.MethodType) and current.__func__ is wrapper:
                        del own[name]
        self._replacements.clear()


def install_instance(instance, backend_factory, runtime):
    """Install accepted loop methods on one instance, preserving its class/source.

    backend_factory(plan) must certify the original callable, its call/effect
    set, and dynamic arguments through backend.guard. Returning None declines
    the method. Existing instance overrides and custom attribute access are
    preserved. The caller must not mutate a model concurrently with execution.
    """
    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    own = _own_dict(instance)
    if own is None or not runtime.config.enabled:
        return InstanceInstallation(instance)
    methods = {}
    for cls in type(instance).__mro__:
        for name, value in vars(cls).items():
            if name not in methods:
                methods[name] = value
    replacements = []
    reports = []
    # Complete compilation/certification before modifying any instance method.
    for name, fn in methods.items():
        if name.startswith("__") or name in own or not isinstance(fn, types.FunctionType):
            continue
        constants = _boolean_assumptions(fn, instance, own)
        plan = compile_loop(fn, constants=constants)
        if not plan.accepted:
            continue
        backend = backend_factory(plan)
        reports.append({"method": name, "plan": plan.report, "backend_available": backend is not None})
        if backend is None:
            continue
        if not isinstance(backend, RegionBackend):
            raise TypeError("backend_factory must return RegionBackend or None")
        replacements.append((name, plan.bind(runtime, backend)))
    token = InstanceInstallation(instance, replacements, reports)
    for name, wrapper in replacements:
        own[name] = types.MethodType(wrapper, instance)
    return token


class FrameworkInstallation:
    def __init__(self, module_class, backend_factory, config):
        self.enabled = config.enabled
        self.reason = "enabled with explicit backend factory" if config.enabled else "disabled by environment"
        self._module_class = module_class
        self._factory = backend_factory
        self._runtime = RegionRuntime(config)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._instances = {}  # id -> (weakref, token); never invokes model __hash__/__eq__.
        self._closed = False
        self._closing = False
        self._active_calls = 0
        self._original = None
        self._wrapper = None
        if self.enabled:
            def call(instance, *args, **kwargs):
                # map() holds this exact runtime's gate through all workers and
                # callbacks. close() must acquire that gate before restoring
                # hooks, so nested calls need no second active-call lock. A
                # foreign runtime provides no such protection for this hook.
                if in_region(self._runtime):
                    return self._original(instance, *args, **kwargs)
                with self._lock:
                    if self._closing:
                        raise RuntimeError("host-regions framework installation is closing")
                    active = not self._closed
                    if active:
                        self._active_calls += 1
                try:
                    if active and not in_region() and not getattr(self._local, "installing", False):
                        self._ensure(instance)
                    return self._original(instance, *args, **kwargs)
                finally:
                    if active:
                        with self._lock:
                            self._active_calls -= 1
            self._wrapper = call
            with _hooks_lock:
                existing = _hooks.get(module_class)
                if existing is not None and not existing._closed:
                    self._runtime.close()
                    raise RuntimeError("a host-regions bootstrap is already installed for this Module class")
                self._original = module_class._call_impl
                _hooks[module_class] = self
                module_class._call_impl = call

    def _ensure(self, instance):
        with self._lock:
            if self._closed or self._closing:
                return
            key = id(instance)
            previous = self._instances.get(key)
            if previous is not None and previous[0]() is instance:
                return
            self._local.installing = True
            try:
                token = install_instance(instance, self._factory, self._runtime)
                def gone(reference, key=key):
                    with self._lock:
                        entry = self._instances.get(key)
                        if entry is not None and entry[0] is reference:
                            self._instances.pop(key, None)
                reference = weakref.ref(instance, gone)
                self._instances[key] = (reference, token)
            finally:
                self._local.installing = False

    @property
    def installed_count(self):
        with self._lock:
            return sum(token.installed_count for _, token in self._instances.values())

    @property
    def reports(self):
        with self._lock:
            return tuple(report for _, token in self._instances.values() for report in token.reports)

    def close(self):
        # Reserve close under the same lock used to admit calls/installations.
        # Do not allow a new installation to escape restoration after close.
        with self._lock:
            if self._closed:
                return
            if self._closing or self._active_calls:
                raise RuntimeError("cannot close framework installation during an active call")
            self._closing = True
        try:
            self._runtime.close()
        except BaseException:
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            tokens = [token for _, token in self._instances.values()]
            self._instances.clear()
        cleanup_error = None
        for token in tokens:
            try:
                token.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(f"additional instance cleanup error: {type(error).__name__}: {error}")
        with _hooks_lock:
            if self._wrapper is not None and self._module_class._call_impl is self._wrapper:
                self._module_class._call_impl = self._original
            if _hooks.get(self._module_class) is self:
                del _hooks[self._module_class]
        with self._lock:
            self._closed = True
            self._closing = False
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def install_from_env(torch_module, backend_factory, *, env=None):
    """One framework bootstrap; environment alone cannot supply a missing backend.

    It can be called from an explicitly integrated TorchNPU package initializer.
    No automatic sitecustomize, installed-package edit or implicit fake backend.
    """
    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    return FrameworkInstallation(torch_module.nn.Module, backend_factory, RegionConfig.from_env(env))


def install_torch_npu_from_env(torch_module, provider, *, certify, env=None):
    """Framework-owned bootstrap with a complete native provider and effect policy.

    certify(plan, bound_arguments) must approve the original function, its
    reachable operations, thread state and storage/alias contract. There is no
    default policy accepting arbitrary Eager code. Native source prototypes in
    this repository deliberately do not advertise whole-framework capability.
    """
    from .backends import backend_available, make_torch_npu_backend
    config = RegionConfig.from_env(env)
    if not callable(certify):
        raise TypeError("certify must be callable")
    if not config.enabled or not backend_available(provider, target="torch_npu"):
        token = FrameworkInstallation(torch_module.nn.Module, lambda _plan: None,
                                      replace(config, enabled=False))
        if config.enabled:
            token.reason = "TorchNPU capture provider unavailable or incomplete; original execution retained"
        return token

    def factory(plan):
        return make_torch_npu_backend(provider, guard=lambda arguments: certify(plan, arguments))

    return FrameworkInstallation(torch_module.nn.Module, factory, config)
