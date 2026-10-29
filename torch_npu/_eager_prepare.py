"""Experimental, model-independent native eager Prepare queue (R24).

Import has no device/worker side effect. Only operators whose native SDK
contracts are qualified may be queued; a typed schema alone is insufficient.
Use ``drain_close`` before leaving a configured inference run.
"""
import json

import torch_npu as _torch_npu

_OPERATORS = ('aclnnEmbeddingBag', 'aclnnAdd', 'aclnnRelu', 'aclnnMm')


def _result(value):
    result = json.loads(value)
    if type(result) is not dict or type(result.get('schema')) is not int or result['schema'] != 1:
        raise RuntimeError('native eager-prepare snapshot schema differs')
    return result


def configure(*, enabled=False, workers=4, capacity=8, byte_limit=536870912,
              allowed_ops=_OPERATORS, diagnostic_trace=False, trace_limit=256):
    """Configure the cold queue; enabling initializes the existing NPU runtime.

    ``allowed_ops`` requests eligibility. ``qualified_ops`` in the returned
    snapshot identifies the independently qualified subset; unsupported calls
    keep the original eager route and record a fallback reason.
    """
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if type(diagnostic_trace) is not bool or type(trace_limit) is not int:
        raise TypeError('diagnostic_trace must be bool and trace_limit must be int')
    if not 1 <= trace_limit <= 256:
        raise ValueError('cold diagnostic trace is limited to at most 256 records')
    for name, value in (('workers', workers), ('capacity', capacity), ('byte_limit', byte_limit)):
        if type(value) is not int:
            raise TypeError(name + ' must be int')
        if value <= 0:
            raise ValueError(name + ' must be positive')
    if workers > capacity or workers > 8 or capacity > 1024 or byte_limit > 536870912:
        raise ValueError('configuration exceeds the admitted worker/queue/byte bounds')
    if type(allowed_ops) not in (tuple, list):
        raise TypeError('allowed_ops must be a plain tuple or list')
    if (not allowed_ops or any(type(name) is not str or name not in _OPERATORS for name in allowed_ops)
            or len(set(allowed_ops)) != len(allowed_ops)):
        raise ValueError('allowed_ops must contain unique registered schema names')
    if enabled:
        _torch_npu.npu._lazy_init()
    return _result(_torch_npu._C._npu_eager_prepare_configure(
        enabled, workers, capacity, byte_limit, tuple(allowed_ops), diagnostic_trace, trace_limit))


def snapshot():
    """Return native counters and closure state; no device synchronization."""
    return _result(_torch_npu._C._npu_eager_prepare_snapshot())


def drain_close():
    """Close admission, drain the original FIFO and retire before joining workers.

    A failure propagates unchanged and retains the quarantined worker domain;
    it is not an acknowledgement that SDK resources were reclaimed.
    """
    return _result(_torch_npu._C._npu_eager_prepare_drain_close())
