"""Experimental native entry diagnostics for the R26 eager runtime.

Disabled by default. Shadow mode executes original kernels on the original
stream and reports native entry coverage. It does not qualify parallel execution.
Importing this module does not initialize an NPU or create worker threads.
"""
import json
from contextlib import contextmanager

import torch as _torch

import torch_npu as _torch_npu


def _call(name, *args):
    function = getattr(_torch_npu._C, '_npu_eager_parallel_' + name, None)
    if function is None:
        raise RuntimeError('native eager-parallel binding is unavailable; rebuild torch_npu with R26 sources')
    result = json.loads(function(*args))
    if type(result) is not dict or type(result.get('schema')) is not int or result['schema'] != 1:
        raise RuntimeError('native eager-parallel snapshot schema differs')
    return result


def configure(*, enabled=False, mode='shadow', streams=2, capacity=64, byte_limit=536870912):
    """Configure actual native diagnostics; unsupported execution modes fail closed."""
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if type(mode) is not str:
        raise TypeError('mode must be str')
    if mode != 'shadow':
        raise ValueError('only shadow mode is currently qualified for entry diagnostics')
    for name, value in (('streams', streams), ('capacity', capacity), ('byte_limit', byte_limit)):
        if type(value) is not int:
            raise TypeError(name + ' must be int')
    if streams != 2 or not 1 <= capacity <= 1024 or not 1 <= byte_limit <= 536870912:
        raise ValueError('configuration exceeds admitted stream/capacity/byte bounds')
    return _call('configure', enabled, mode, streams, capacity, byte_limit)


def snapshot():
    """Read the native diagnostic state without claiming device completion."""
    return _call('snapshot')


def drain_close():
    """Close native admission and join any work owned by the runtime."""
    return _call('drain_close')


@contextmanager
def scope(*, mode='shadow', streams=2, capacity=64, byte_limit=536870912):
    """Run native shadow diagnostics inside caller-owned ``torch.inference_mode``.

    Original model code, gradient state, kernels, and stream choice are retained.
    On an exceptional exit, a later close failure never replaces the body error.
    """
    if not _torch.is_inference_mode_enabled():
        raise RuntimeError('eager-parallel scope requires caller-owned torch.inference_mode()')
    state = configure(enabled=True, mode=mode, streams=streams, capacity=capacity, byte_limit=byte_limit)
    try:
        yield state
    except BaseException as error:
        try:
            drain_close()
        except BaseException as close_error:
            try:
                if hasattr(error, 'add_note'):
                    error.add_note('native eager-parallel close also failed: ' + repr(close_error))
            except BaseException:
                pass  # Diagnostic formatting must not replace the first body error.
        raise
    else:
        drain_close()
