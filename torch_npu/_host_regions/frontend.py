"""Opt-in extraction of one independent Python loop, without rewriting a model.

Structural acceptance is deliberately not a purity proof.  The backend's guard
must certify every admitted callable/operator, input iteration/snapshot, TLS,
mutation and exception boundary for this exact plan.  In particular, a passing
``eval()`` flag alone does not certify a Torch module.  A wrapper falls back only
before any transformed work starts; failures after that point are never retried.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import functools
import hashlib
import inspect
import textwrap
import types
from typing import Any, Callable, Mapping

from .contracts import RegionBackend
from . import runtime as region_runtime


class _Rejected(ValueError):
    pass


def _names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for item in target.elts:
            result.update(_names(item))
        return result
    raise _Rejected("assignment targets must be local names; state mutation is unsupported")


def _parameters(arguments: ast.arguments) -> set[str]:
    result = {node.arg for node in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
    for node in (arguments.vararg, arguments.kwarg):
        if node is not None:
            result.add(node.arg)
    return result


def _reject_private_names(function: Callable, tree: ast.FunctionDef) -> None:
    """Generated code has no class scope, so class-mangled names cannot move."""
    compiled_names = set(function.__code__.co_varnames + function.__code__.co_names
                         + function.__code__.co_freevars + function.__code__.co_cellvars)
    for node in ast.walk(tree):
        name = (node.attr if isinstance(node, ast.Attribute) else
                node.id if isinstance(node, ast.Name) else
                node.arg if isinstance(node, ast.arg) else "")
        if name.startswith("__") and not name.endswith("__"):
            # Top-level local double-underscore names need no mangling and are
            # preserved. Private attributes are conservatively always rejected.
            if isinstance(node, ast.Attribute) or name not in compiled_names:
                raise _Rejected("private class identifiers need original name-mangling scope")


_EXPRESSION_NODES = (
    ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Slice,
    ast.Call, ast.keyword, ast.Tuple, ast.List, ast.Set, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Load, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)


def _expression(node: ast.AST, *, output: str, locals_: set[str], defined: set[str]) -> None:
    for part in ast.walk(node):
        if not isinstance(part, _EXPRESSION_NODES):
            raise _Rejected(f"unsupported expression: {type(part).__name__}")
        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
            if part.id == output:
                raise _Rejected("the output collection may only be used by its terminal append")
            if part.id in locals_ and part.id not in defined:
                raise _Rejected(f"loop-carried or conditionally undefined local: {part.id}")


def _append_value(statement: ast.stmt, output: str) -> ast.expr | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id == output and call.func.attr == "append"):
        if len(call.args) != 1 or call.keywords:
            raise _Rejected("append must receive exactly one positional value")
        return call.args[0]
    return None


def _statements(statements: list[ast.stmt], *, output: str, locals_: set[str],
                defined: set[str]) -> tuple[set[str], bool]:
    """Check definite assignment and exactly one terminal append on every path."""
    defined = set(defined)
    terminal = False
    for statement in statements:
        if terminal:
            raise _Rejected("no statements may follow an append on a completed path")
        value = _append_value(statement, output)
        if value is not None:
            _expression(value, output=output, locals_=locals_, defined=defined)
            terminal = True
        elif isinstance(statement, ast.Assign):
            _expression(statement.value, output=output, locals_=locals_, defined=defined)
            assigned = set().union(*(_names(target) for target in statement.targets))
            if output in assigned:
                raise _Rejected("output collection reassignment is unsupported")
            defined.update(assigned)
        elif isinstance(statement, ast.If):
            _expression(statement.test, output=output, locals_=locals_, defined=defined)
            left, left_done = _statements(statement.body, output=output, locals_=locals_, defined=defined)
            right, right_done = _statements(statement.orelse, output=output, locals_=locals_, defined=defined)
            if left_done != right_done:
                raise _Rejected("append must terminate both branches at the same structural boundary")
            defined = left & right
            terminal = left_done
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            # Explicit capabilities must certify these calls; no implicit purity assumption.
            _expression(statement.value, output=output, locals_=locals_, defined=defined)
        elif isinstance(statement, ast.Pass):
            pass
        else:
            raise _Rejected(f"unsupported loop statement: {type(statement).__name__}")
    return defined, terminal


class _Specialize(ast.NodeTransformer):
    def __init__(self, constants: Mapping[str, bool]):
        self.constants = constants
        self.used: set[str] = set()

    def visit_If(self, node):
        if isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name):
            key = f"{node.test.value.id}.{node.test.attr}"
            if key in self.constants:
                self.used.add(key)
                chosen = node.body if self.constants[key] else node.orelse
                result = []
                for statement in chosen:
                    value = self.visit(statement)
                    result.extend(value if isinstance(value, list) else [value])
                return result
        return self.generic_visit(node)


class _AppendToReturn(ast.NodeTransformer):
    def __init__(self, output):
        self.output = output

    def visit_Expr(self, node):
        value = _append_value(node, self.output)
        if value is not None:
            return ast.copy_location(ast.Return(value=value), node)
        return node


def _type_member(cls: type, name: str, default=None):
    # Avoid invoking user descriptors or metaclass attribute lookup during guards.
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return default


def _constant_matches(arguments: Mapping[str, Any], key: str, value: bool) -> bool:
    parameter, attribute = key.split(".")
    if parameter not in arguments:
        return False
    obj = arguments[parameter]
    cls = type(obj)
    if _type_member(cls, "__getattribute__") is not object.__getattribute__:
        return False
    descriptor = _type_member(cls, attribute)
    if descriptor is not None and (
            _type_member(type(descriptor), "__set__") is not None
            or _type_member(type(descriptor), "__delete__") is not None):
        return False
    dictionary_descriptor = _type_member(cls, "__dict__")
    if not isinstance(dictionary_descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)):
        return False
    dictionary = object.__getattribute__(obj, "__dict__")
    return type(dictionary) is dict and type(dictionary.get(attribute)) is bool and dictionary[attribute] is value


def _fresh(names: set[str], stem: str) -> str:
    name = stem
    while name in names:
        name += "_"
    names.add(name)
    return name


def _build_factory(function: Callable, tree: ast.FunctionDef, loop_position: int,
                   output: str) -> Callable:
    tree = copy.deepcopy(tree)
    loop = tree.body[loop_position]
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(_parameters(tree.args))
    names.update(function.__globals__)
    runtime_name = _fresh(names, "__ft_regions_runtime")
    backend_name = _fresh(names, "__ft_regions_backend")
    body_name = _fresh(names, "__ft_regions_body")
    index_name = _fresh(names, "__ft_regions_index")
    item_name = _fresh(names, "__ft_regions_item")
    factory_name = _fresh(names, "__ft_regions_factory")
    # An original function-name read must remain a live global read, not bind
    # to the transformed nested function in this generated factory scope.
    tree.name = _fresh(names, "__ft_regions_transformed")
    body = [_AppendToReturn(output).visit(statement) for statement in loop.body]
    worker = ast.FunctionDef(
        name=body_name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=index_name), ast.arg(arg=item_name)],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Assign(targets=[loop.target], value=ast.Name(id=item_name, ctx=ast.Load()))] + body,
        decorator_list=[],
    )
    dispatch = ast.Assign(targets=[ast.Name(id=output, ctx=ast.Store())], value=ast.Call(
        func=ast.Attribute(value=ast.Name(id=runtime_name, ctx=ast.Load()), attr="map", ctx=ast.Load()),
        args=[ast.Name(id=body_name, ctx=ast.Load()), loop.iter],
        keywords=[ast.keyword(arg="backend", value=ast.Name(id=backend_name, ctx=ast.Load()))],
    ))
    ast.copy_location(worker, loop)
    ast.copy_location(dispatch, loop)
    # Replace the empty output initialization as well as the loop.
    tree.body[loop_position - 1:loop_position + 1] = [worker, dispatch]
    tree.decorator_list = []
    tree.returns = None
    tree.type_comment = None
    if hasattr(tree, "type_params"):
        tree.type_params = []
    # Defaults and annotations have already been evaluated by the original def.
    # Never reevaluate them as part of extraction.
    tree.args.defaults = []
    tree.args.kw_defaults = [None for _ in tree.args.kwonlyargs]
    for argument in tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs:
        argument.annotation = None
        argument.type_comment = None
    for argument in (tree.args.vararg, tree.args.kwarg):
        if argument is not None:
            argument.annotation = None
            argument.type_comment = None
    factory = ast.FunctionDef(
        name=factory_name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=runtime_name), ast.arg(arg=backend_name)],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[tree, ast.Return(value=ast.Name(id=tree.name, ctx=ast.Load()))], decorator_list=[],
    )
    ast.copy_location(factory, tree)
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, inspect.getsourcefile(function) or "<host-region>", "exec"), namespace)
    # Use original live globals; copying globals would silently freeze rebinding.
    return types.FunctionType(namespace[factory_name].__code__, function.__globals__)


@dataclass(frozen=True)
class LoopPlan:
    """Analysis and guarded wrapper factory for one original Python function."""
    original: Callable
    accepted: bool
    source_sha256: str
    _report: dict[str, Any]
    _function: Callable | None = None
    _factory: Callable | None = None
    _constants: tuple[tuple[str, bool], ...] = ()
    _code: types.CodeType | None = None

    @property
    def report(self) -> dict[str, Any]:
        return copy.deepcopy(self._report)

    def bind(self, runtime, backend: RegionBackend | None = None) -> Callable:
        """Bind once; check code/default identities before any capability guard.

        Keyword defaults also have an item-identity snapshot because their
        dictionary is mutable. No user equality is invoked by these checks.
        Concurrent mutation while a guarded invocation runs remains unsupported.
        """
        function = self._function
        @functools.wraps(self.original)
        def original_only(*args, **kwargs):
            return self.original(*args, **kwargs)

        if (not self.accepted or function.__code__ is not self._code
                or "__wrapped__" in function.__dict__ or "__signature__" in function.__dict__):
            return original_only
        defaults, keyword_defaults = function.__defaults__, function.__kwdefaults__
        if ((defaults is not None and type(defaults) is not tuple)
                or (keyword_defaults is not None and type(keyword_defaults) is not dict)):
            return original_only
        keyword_snapshot = tuple(keyword_defaults.items()) if keyword_defaults is not None else ()
        transformed = self._factory(runtime, backend)
        transformed.__defaults__ = defaults
        transformed.__kwdefaults__ = keyword_defaults
        signature = inspect.signature(self.original)
        raw_signature = inspect.signature(function) if function is not None else signature
        bound_instance = self.original.__self__ if inspect.ismethod(self.original) else None

        @functools.wraps(self.original)
        def wrapper(*args, **kwargs):
            if (not self.accepted or backend is None
                    or not getattr(getattr(runtime, "config", None), "enabled", False)
                    or getattr(runtime, "closed", False) or region_runtime.in_region()):
                return self.original(*args, **kwargs)
            if (function.__code__ is not self._code or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not keyword_defaults
                    or "__wrapped__" in function.__dict__ or "__signature__" in function.__dict__):
                return self.original(*args, **kwargs)
            current_keywords = tuple(keyword_defaults.items()) if keyword_defaults is not None else ()
            if (len(current_keywords) != len(keyword_snapshot)
                    or any(key is not old_key or value is not old_value
                           for (key, value), (old_key, old_value)
                           in zip(current_keywords, keyword_snapshot))):
                return self.original(*args, **kwargs)
            # Preserve the original callable's argument validation and default binding.
            signature.bind(*args, **kwargs)
            raw_args = (bound_instance,) + args if bound_instance is not None else args
            bound = raw_signature.bind(*raw_args, **kwargs)
            bound.apply_defaults()
            if not all(_constant_matches(bound.arguments, key, value) for key, value in self._constants):
                return self.original(*args, **kwargs)
            if not backend.guard(dict(bound.arguments)):
                return self.original(*args, **kwargs)
            # No except/fallback here: after entering transformed code, effects may exist.
            return transformed(*raw_args, **kwargs)

        return wrapper


def compile_loop(fn: Callable, *, constants: Mapping[str, bool] | None = None) -> LoopPlan:
    """Analyze canonical ``out=[]; for ...: out.append(...); return out``.

    A straight local-assignment prelude is allowed.  All loop paths must append
    once, and only locally assigned values may be mutated.  Optional constants
    specialize direct ``if parameter.attribute`` tests only; each exact boolean
    must remain in the instance's own dictionary on every invocation.  The
    backend additionally certifies no concurrent mutation during execution.
    Rejections remain callable through ``bind`` and execute the original code.
    """
    if not callable(fn):
        raise TypeError("compile_loop expects a callable")
    function = fn.__func__ if inspect.ismethod(fn) else fn
    source_sha256 = ""
    assumptions = dict(constants or {})
    report: dict[str, Any] = {"accepted": False, "reasons": [], "calls": [],
                              "assumptions": assumptions, "requires_backend_guard": True,
                              "source_sha256": source_sha256}
    try:
        if not inspect.isfunction(function):
            raise _Rejected("only inspectable Python functions and bound methods are supported")
        if "__wrapped__" in function.__dict__:
            raise _Rejected("wrapped functions may expose another callable's source")
        if "__signature__" in function.__dict__:
            raise _Rejected("custom function signatures cannot certify original argument binding")
        source_code = function.__code__
        source_lines, source_start = inspect.getsourcelines(function)
        source = "".join(source_lines)
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        report["source_sha256"] = source_sha256
        if function.__closure__:
            raise _Rejected("functions with unspecified closure state are unsupported")
        parsed = ast.parse(textwrap.dedent(source))
        ast.increment_lineno(parsed, source_start - 1)
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
            raise _Rejected("expected one synchronous function definition")
        tree = parsed.body[0]
        _reject_private_names(function, tree)
        parameters = _parameters(tree.args)
        for key, value in assumptions.items():
            parts = key.split(".") if isinstance(key, str) else []
            if (type(value) is not bool or len(parts) != 2 or parts[0] not in parameters
                    or not parts[1].isidentifier()):
                raise _Rejected("constants must map a direct parameter.attribute to an exact bool")
        specialization = _Specialize(assumptions)
        tree = specialization.visit(tree)
        if specialization.used != set(assumptions):
            raise _Rejected("each constant must specialize a direct boolean if test")
        loops = [i for i, statement in enumerate(tree.body) if isinstance(statement, ast.For)]
        if len(loops) != 1:
            raise _Rejected("expected exactly one top-level for loop")
        position = loops[0]
        loop = tree.body[position]
        if position == 0 or position != len(tree.body) - 2 or loop.orelse:
            raise _Rejected("loop must follow output initialization and precede return output")
        initialization, returned = tree.body[position - 1], tree.body[-1]
        if not (isinstance(initialization, ast.Assign) and len(initialization.targets) == 1
                and isinstance(initialization.targets[0], ast.Name)
                and isinstance(initialization.value, ast.List) and not initialization.value.elts):
            raise _Rejected("the statement preceding the loop must initialize an empty output list")
        output = initialization.targets[0].id
        if not (isinstance(returned, ast.Return) and isinstance(returned.value, ast.Name)
                and returned.value.id == output):
            raise _Rejected("function must return its output collection directly")
        prefix = tree.body[:position - 1]
        if prefix and isinstance(prefix[0], ast.Expr) and isinstance(prefix[0].value, ast.Constant) and isinstance(prefix[0].value.value, str):
            prefix = prefix[1:]
        defined = set(parameters)
        for statement in prefix:
            if not isinstance(statement, ast.Assign):
                raise _Rejected("only local assignments are supported before the output list")
            _expression(statement.value, output=output, locals_=set(), defined=defined)
            defined.update(set().union(*(_names(target) for target in statement.targets)))
        if output in defined:
            raise _Rejected("output collection name must be fresh")
        local_names = {node.id for node in ast.walk(loop) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        if local_names & defined:
            raise _Rejected("loop writes a parameter or prelude local; possible loop-carried state")
        _expression(loop.iter, output=output, locals_=local_names, defined=set())
        targets = _names(loop.target)
        if output in targets:
            raise _Rejected("loop target aliases its output collection")
        _, terminal = _statements(loop.body, output=output, locals_=local_names, defined=targets)
        if not terminal:
            raise _Rejected("each loop path must end in exactly one output.append")
        report["calls"] = sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
                                   and not (isinstance(node.func, ast.Attribute)
                                            and isinstance(node.func.value, ast.Name)
                                            and node.func.value.id == output and node.func.attr == "append")})
        report["output_name"] = output
        report["iteration"] = ast.unparse(loop.iter)
        report["accepted"] = True
        factory = _build_factory(function, tree, position, output)
        return LoopPlan(fn, True, source_sha256, report, function, factory, tuple(assumptions.items()), source_code)
    except (_Rejected, OSError, TypeError, SyntaxError) as error:
        report["accepted"] = False
        report["reasons"] = [str(error)]
        return LoopPlan(fn, False, source_sha256, report, function if inspect.isfunction(function) else None)
