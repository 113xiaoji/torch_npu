"""Restricted intrinsic policy for independently certified EmbeddingBag regions.

This module imports neither Torch nor TorchNPU. Its injected modules must be the
source-audited build selected by the provider. AST acceptance is not an arbitrary
Python purity proof. The admitted operator family has read-only sum semantics;
the native provider separately owns capture, TLS and retirement. Real CANN/device
safety is outside this policy's certificate. External concurrent mutation of the
model, containers, functions or tensor metadata during a call is unsupported.
"""
from __future__ import annotations

import ast
import builtins
from collections import OrderedDict
import hashlib
import inspect
import os
import textwrap
from typing import Any

from .frontend import _Specialize, _constant_matches


class _Decline(ValueError):
    pass


def _need(value, reason):
    if not value:
        raise _Decline(reason)


def logical_stream_identity(stream):
    """Read logical identity without asking for the draining raw stream handle.

    The audited TorchNPU Stream binding exposes these three read-only integer
    members directly. Its npu_stream property calls NPUStream::stream(), which
    drains the task queue; simulator raw handles may also alias distinct streams.
    """
    identity = (stream.stream_id, stream.device_index, stream.device_type)
    if not all(type(value) is int for value in identity):
        raise ValueError('logical stream identity requires exact integers')
    return identity


def _same_tree(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


class _InactiveGather(ast.NodeTransformer):
    """Recognize only the canonical optional[index] is not None gather guard."""
    def __init__(self):
        self.containers = set()
        self.accesses = []

    def visit_If(self, node):
        test = node.test
        _need(isinstance(test, ast.Compare) and len(test.ops) == 1
              and isinstance(test.ops[0], ast.IsNot) and len(test.comparators) == 1
              and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
              and isinstance(test.left, ast.Subscript) and isinstance(test.left.value, ast.Name)
              and isinstance(test.left.slice, ast.Name),
              'unsupported runtime conditional')
        _need(len(node.body) == len(node.orelse) == 1, 'unsupported optional gather branch')
        yes, no = node.body[0], node.orelse[0]
        _need(isinstance(yes, ast.Assign) and isinstance(no, ast.Assign)
              and len(yes.targets) == len(no.targets) == 1
              and isinstance(yes.targets[0], ast.Name) and _same_tree(yes.targets[0], no.targets[0])
              and isinstance(no.value, ast.Constant) and no.value.value is None,
              'optional gather must assign one local or None')
        call = yes.value
        _need(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
              and call.func.attr == 'gather' and _same_tree(call.func.value, test.left)
              and len(call.args) == 2 and not call.keywords
              and isinstance(call.args[0], ast.Constant) and type(call.args[0].value) is int
              and call.args[0].value == 0 and isinstance(call.args[1], ast.Name),
              'only an inactive optional indexed gather is admitted')
        self.containers.add(test.left.value.id)
        self.accesses.append(test.left)
        return no


class NpuLoopPolicy:
    """Policy for a structural LoopPlan; check must precede native TLS capture.

    Parameter/local names and iteration counts are inferred from the plan. No
    model/function names or source line numbers participate in acceptance.
    """
    def __init__(self, torch, torch_npu, plan, *, real_opapi=False):
        if type(real_opapi) is not bool:
            raise TypeError('real_opapi must be bool')
        self.real_opapi = real_opapi
        self.torch, self.torch_npu, self.plan = torch, torch_npu, plan
        self.accepted = False
        self.reason = self.last_reason = 'policy not analyzed'
        self._builtins = set()
        self._pins = []
        try:
            _need(plan.accepted, 'frontend declined this loop')
            self._function = plan.original.__func__ if inspect.ismethod(plan.original) else plan.original
            self._code = self._function.__code__
            _need(self._code is plan._code, 'original function code differs from structural plan')
            source = inspect.getsource(self._function)
            _need(hashlib.sha256(source.encode('utf-8')).hexdigest() == plan.source_sha256,
                  'original source differs from structural plan')
            self._constants = dict(plan.report['assumptions'])
            tree = ast.parse(textwrap.dedent(source)).body[0]
            tree = _Specialize(self._constants).visit(tree)
            optional = _InactiveGather()
            tree = optional.visit(tree)
            self._none_containers = optional.containers
            self._optional_accesses = optional.accesses
            self._parameters = set(inspect.signature(self._function).parameters)
            self._output = plan.report['output_name']
            loops = [part for part in tree.body if isinstance(part, ast.For)]
            _need(len(loops) == 1, 'one independent loop is required')
            self._loop = loops[0]
            self._prefix = tree.body[:tree.body.index(self._loop) - 1]
            if self._prefix and isinstance(self._prefix[0], ast.Expr) and isinstance(self._prefix[0].value, ast.Constant):
                self._prefix = self._prefix[1:]
            self._aliases = {part.targets[0].id for part in ast.walk(self._loop)
                             if isinstance(part, ast.Assign) and len(part.targets) == 1
                             and isinstance(part.targets[0], ast.Name)
                             and isinstance(part.value, ast.Subscript)
                             and isinstance(part.value.value, ast.Name)
                             and part.value.value.id in self._parameters}
            self._validate_syntax(tree)
            self._check_builtins()
            # Structural work happens once. Invocation checks use these direct
            # access actions, never walk or dispatch on AST node types.
            self._prefix_actions = tuple(self._prepare_statement(node) for node in self._prefix)
            self._items = self._prepare_expr(self._loop.iter)
            self._optional_checks = tuple(self._prepare_expr(node) for node in self._optional_accesses)
            self._body_actions = tuple(self._prepare_statement(node) for node in self._loop.body)
            target = self._loop.target
            self._unpack = not isinstance(target, ast.Name)
            _need(not self._unpack or isinstance(target, (ast.Tuple, ast.List)),
                  'unsupported iteration unpacking')
            targets = target.elts if self._unpack else (target,)
            _need(all(isinstance(variable, ast.Name) for variable in targets), 'unsupported iteration target')
            self._targets = tuple(variable.id for variable in targets)
            for obj, name in ((torch.nn.EmbeddingBag, 'forward'), (torch.nn.Module, '__call__'),
                              (torch.nn.Module, '_call_impl'), (torch.nn.functional, 'embedding_bag'),
                              (torch, 'embedding_bag')):
                value = getattr(obj, name)
                self._pins.append((obj, name, value, getattr(value, '__code__', None)))
            for cls, names in ((torch.nn.Module, ('__getattribute__', '__getattr__', '_slow_forward')),
                               (torch.nn.ModuleList, ('__getitem__', '__len__', '__iter__'))):
                for name in names:
                    if hasattr(cls, name):
                        value = getattr(cls, name)
                        self._pins.append((cls, name, value, getattr(value, '__code__', None)))
            self.accepted = True
            self.reason = 'supported intrinsic loop; invocation metadata guards still required'
        except (ValueError, TypeError, AttributeError, OSError, SyntaxError, KeyError) as error:
            self.reason = str(error)
        self.last_reason = self.reason

    @property
    def evidence(self):
        return {'accepted': self.accepted, 'reason': self.reason,
                'source_sha256': self.plan.source_sha256,
                'operator_family': 'EmbeddingBag', 'mode': 'sum',
                'scope': 'audited host simulator, no real CANN safety claim',
                'contract': 'read-only model/inputs; external concurrent mutation is unsupported',
                'inactive_optional_containers': sorted(getattr(self, '_none_containers', ())),
                'builtin_guards': sorted(self._builtins)}

    def _validate_syntax(self, tree):
        allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.For, ast.Return,
                   ast.Expr, ast.Name, ast.Constant, ast.Subscript, ast.Call, ast.Attribute,
                   ast.keyword, ast.List, ast.Tuple, ast.Load, ast.Store)
        local_names = self._parameters | {part.id for part in ast.walk(tree)
                                         if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
        for part in ast.walk(tree):
            _need(isinstance(part, allowed), 'unsupported intrinsic AST: ' + type(part).__name__)
            if isinstance(part, ast.Assign):
                _need(len(part.targets) == 1 and isinstance(part.targets[0], ast.Name),
                      'only one local assignment is admitted')
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
                _need(part.id in local_names or part.id in ('range', 'len', 'enumerate'),
                      'unknown global or callable: ' + part.id)
            if isinstance(part, ast.Attribute):
                _need(isinstance(part.value, ast.Name) and part.value.id == self._output
                      and part.attr == 'append', 'unknown attribute or operator')
            if isinstance(part, ast.Call):
                if isinstance(part.func, ast.Attribute):
                    _need(len(part.args) == 1 and not part.keywords, 'invalid output append')
                elif isinstance(part.func, ast.Name) and part.func.id in ('range', 'len', 'enumerate'):
                    _need(part.func.id not in local_names and len(part.args) == 1 and not part.keywords,
                          'shadowed or unsupported builtin invocation')
                    self._builtins.add(part.func.id)
                else:
                    _need(isinstance(part.func, ast.Name) and part.func.id in self._aliases,
                          'only indexed module aliases may be called')
                    _need(len(part.args) == 2 and len(part.keywords) <= 1
                          and all(k.arg == 'per_sample_weights' for k in part.keywords),
                          'unsupported EmbeddingBag call signature')
        _need(isinstance(self._loop.iter, ast.Call) and isinstance(self._loop.iter.func, ast.Name)
              and self._loop.iter.func.id in ('range', 'enumerate'), 'unsupported iteration source')
        _need(self._none_containers <= self._parameters, 'optional gather source must be a parameter')

    def _check_builtins(self):
        namespace = self._function.__globals__
        builtin_namespace = self._function.__builtins__
        for name in self._builtins:
            _need(namespace.get(name, builtin_namespace.get(name)) is getattr(builtins, name),
                  'shadowed builtin: ' + name)

    def _sequence(self, value):
        if type(value) in (list, tuple):
            return tuple(value)
        if type(value) is self.torch.nn.ModuleList:
            state = object.__getattribute__(value, '__dict__')
            modules = state.get('_modules')
            _need(type(modules) is dict and all(type(key) is str for key in modules)
                  and tuple(modules) == tuple(str(i) for i in range(len(modules))),
                  'invalid ModuleList container')
            return tuple(modules.values())
        raise _Decline('iteration/indexing requires a plain list, tuple or exact ModuleList')

    def _tensor(self, value, *, weight=False):
        types = (self.torch.Tensor, self.torch.nn.Parameter) if weight else (self.torch.Tensor,)
        _need(type(value) in types, 'tensor subclass or non-tensor is unsupported')
        _need(value.device.type == 'npu' and value.device.index == 0, 'tensor must use npu:0')
        _need(value.layout == self.torch.strided and value.is_contiguous()
              and value.storage_offset() == 0 and value.numel() > 0,
              'tensor must be nonempty dense contiguous with zero storage offset')
        _need(value.requires_grad is False and not value.is_conj() and not value.is_neg(),
              'tensor requires gradients or conjugate/negative resolution')
        _need(value.dim() == (2 if weight else 1), 'unsupported tensor rank')
        _need(value.dtype == self.torch.float32 if weight else
              value.dtype in (self.torch.int32, self.torch.int64), 'unsupported tensor dtype')

    def _module(self, module, values, offsets, weights):
        _need(type(module) is self.torch.nn.EmbeddingBag, 'only exact audited EmbeddingBag is admitted')
        state = object.__getattribute__(module, '__dict__')
        _need(not set(state).intersection({'forward', '_call_impl', '_slow_forward', '_wrapped_call_impl', '__call__'})
              and state.get('_compiled_call_impl') is None,
              'instance forward or compiled module is unsupported')
        _need(all(type(state.get(name)) is dict and not state[name] for name in ('_modules', '_buffers')),
              'unexpected module state')
        _need(all((name == '_is_full_backward_hook' and value is None)
                  or (type(value) in (dict, OrderedDict) and len(value) == 0)
                  for name, value in state.items() if 'hook' in name), 'module hooks are unsupported')
        # The audited sparse.py/F.embedding_bag sum forward passes sparse
        # unchanged; it selects weight-gradient layout. With no grad recording
        # and a non-grad weight, either exact bool uses the same read-only host
        # path. The native adapter retains the original sparse command attribute.
        _need(state.get('max_norm') is None and type(state.get('mode')) is str and state['mode'] == 'sum'
              and type(state.get('sparse')) is bool and state.get('scale_grad_by_freq') is False
              and state.get('include_last_offset') is False and state.get('padding_idx') is None,
              'EmbeddingBag state requires an unsupported operation')
        parameters = state.get('_parameters')
        _need(type(parameters) is dict and set(parameters) == {'weight'}, 'unexpected module parameters')
        self._tensor(parameters['weight'], weight=True)
        self._tensor(values)
        self._tensor(offsets)
        if self.real_opapi:
            # Mixed index types can introduce an auxiliary Cast before the
            # audited EmbeddingBag SDK schema is reached.
            _need(values.dtype == offsets.dtype, 'real OpAPI requires matching indices/offsets dtypes')
        _need(weights is None, 'per-sample weights are unsupported')
        return _RESULT

    def _prepare_expr(self, node):
        """Compile the admitted expression subset to small metadata checks."""
        if isinstance(node, ast.Name):
            name = node.id
            return lambda env, sequence: env[name]
        if isinstance(node, ast.Constant):
            _need(node.value is None or type(node.value) in (int, bool), 'unsupported literal')
            value = node.value
            return lambda env, sequence: value
        if isinstance(node, ast.Subscript):
            container, subscript = self._prepare_expr(node.value), self._prepare_expr(node.slice)

            def indexed(env, sequence):
                items = sequence(container(env, sequence))
                index = subscript(env, sequence)
                _need(type(index) is int and 0 <= index < len(items), 'unsafe/mismatched indexed container')
                return items[index]

            return indexed
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = tuple(self._prepare_expr(arg) for arg in node.args)
            first = args[0]
            if node.func.id == 'len':
                return lambda env, sequence: len(sequence(first(env, sequence)))
            if node.func.id == 'range':
                def counted(env, sequence):
                    count = first(env, sequence)
                    _need(type(count) is int and 0 < count <= 4096, 'unsupported region count')
                    return range(count)
                return counted
            if node.func.id == 'enumerate':
                def enumerated(env, sequence):
                    items = sequence(first(env, sequence))
                    _need(0 < len(items) <= 4096, 'unsupported region count')
                    return enumerate(items)
                return enumerated
            name, second = node.func.id, args[1]
            weights = self._prepare_expr(node.keywords[0].value) if node.keywords else None

            def module(env, sequence):
                values, offsets = first(env, sequence), second(env, sequence)
                optional = weights(env, sequence) if weights is not None else None
                return self._module(env[name], values, offsets, optional)

            return module
        raise _Decline('unsupported guarded expression')

    def _prepare_statement(self, node):
        if isinstance(node, ast.Assign):
            return node.targets[0].id, self._prepare_expr(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            _need(isinstance(call.func, ast.Attribute) and call.func.attr == 'append', 'unknown expression effect')
            return None, self._prepare_expr(call.args[0])
        raise _Decline('unsupported guarded statement')

    @staticmethod
    def _apply(actions, env, sequence):
        for name, expression in actions:
            value = expression(env, sequence)
            if name is None:
                _need(value is _RESULT, 'each result must be an admitted module output')
            else:
                env[name] = value

    def check(self, bound_arguments: dict[str, Any]) -> bool:
        if not self.accepted:
            return False
        try:
            _need(type(bound_arguments) is dict, 'bound arguments must be a plain dict')
            admitted_mode = ((os.environ.get('HDB_REGION_REAL_OPAPI') == '1'
                              and os.environ.get('QWEN_ACL_HOST_SIM') != '1') if self.real_opapi else
                             (os.environ.get('QWEN_ACL_HOST_SIM') == '1'
                              and os.environ.get('HDB_REGION_REAL_OPAPI', '0') == '0'))
            _need(admitted_mode and os.environ.get('TASK_QUEUE_ENABLE') == '1'
                  and os.environ.get('ASCEND_LAUNCH_BLOCKING', '0') in ('', '0'),
                  'requires the provider-certified capture mode and asynchronous task queue 1')
            _need(self._function.__code__ is self._code, 'original function code changed')
            _need(all(_constant_matches(bound_arguments, key, value) for key, value in self._constants.items()),
                  'specialization assumption changed')
            self._check_builtins()
            for obj, name, expected, code in self._pins:
                current = getattr(obj, name)
                _need(current is expected and getattr(current, '__code__', None) is code,
                      'audited Torch callable changed: ' + name)
            _need(not self.torch.is_grad_enabled(), 'grad mode is enabled')
            _need(not self.torch._C._is_torch_function_mode_enabled()
                  and self.torch.utils._python_dispatch._get_current_dispatch_mode() is None,
                  'Python dispatch or torch-function mode is active')
            _need(not self.torch.is_autocast_enabled() and not self.torch.is_autocast_enabled('npu'),
                  'autocast is enabled')
            npu = self.torch_npu.npu
            device = npu.current_device()
            _need(type(device) is int and device == 0, 'current device must be npu:0')
            current_identity = logical_stream_identity(npu.current_stream())
            default_identity = logical_stream_identity(npu.default_stream(0))
            _need(current_identity == default_identity and current_identity[1] == 0,
                  'current stream must be the default stream')
            module_globals = getattr(getattr(self.torch.nn, 'modules', None), 'module', None)
            if module_globals is not None:
                hooks = [getattr(module_globals, name, {}) for name in (
                    '_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks',
                    '_global_backward_pre_hooks')]
                _need(all(type(value) in (dict, OrderedDict) and len(value) == 0 for value in hooks),
                      'global module hooks are unsupported')
            snapshots = {}

            def sequence(value):
                key = id(value)
                if key not in snapshots:
                    # Keep the container alive alongside its snapshot: keys
                    # cannot be reused, and no user __hash__/__eq__ is invoked.
                    snapshots[key] = value, self._sequence(value)
                return snapshots[key][1]

            for name in self._none_containers:
                _need(all(value is None for value in sequence(bound_arguments[name])),
                      'optional gather is active')
            env = dict(bound_arguments)
            self._apply(self._prefix_actions, env, sequence)
            items = self._items(env, sequence)
            checked_iterations = 0
            for item in items:
                iteration = dict(env)
                if not self._unpack:
                    iteration[self._targets[0]] = item
                else:
                    _need(type(item) is tuple and len(self._targets) == len(item),
                          'unsupported iteration unpacking')
                    iteration.update(zip(self._targets, item))
                for access in self._optional_checks:
                    _need(access(iteration, sequence) is None, 'optional gather is active or unbounded')
                self._apply(self._body_actions, iteration, sequence)
                checked_iterations += 1
            _need(checked_iterations > 0, 'no admitted module calls')
            self.last_reason = 'accepted invocation under read-only host simulator contract'
            return True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as error:
            self.last_reason = str(error)
            return False


_RESULT = object()
