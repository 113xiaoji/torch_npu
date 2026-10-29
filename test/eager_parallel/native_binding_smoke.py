"""Run the private probe linked to actual production native binding sources.

No NPU/model/backend-dispatch claim is made by this CPU smoke test.
"""
import importlib
import json
import sys

import torch


def main():
    before = sys._is_gil_enabled()
    assert before is False, 'run under the natural free-threaded target interpreter'
    native = importlib.import_module('_r26_native_entry_probe')
    assert sys._is_gil_enabled() is False
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = torch.cat([a @ b, a @ b], dim=0)
    assert torch.equal(native.invoke_tensors(a, b), expected)
    initial = json.loads(native._npu_eager_parallel_snapshot())
    assert initial['enabled'] is False and initial['entered'] == 0
    enabled = json.loads(native._npu_eager_parallel_configure(True, 'shadow', 2, 64, 536870912))
    assert enabled['backend'] == 'shadow' and enabled['enabled'] is True
    assert enabled['real_sdk_qualified'] is False and enabled['device_parallel_qualified'] is False
    with torch.inference_mode():
        assert torch.equal(native.invoke_tensors(a, b), expected)
    recorded = json.loads(native._npu_eager_parallel_snapshot())
    assert recorded['entered'] == 1 and recorded['completed'] == 1
    assert recorded['nested_entries'] == 2 and recorded['active_calls'] == 0
    assert recorded['operator_counts']['probe::outer']['inputs'] == 2
    assert recorded['operator_counts']['probe::outer']['outputs'] == 1
    try:
        native.invoke_failure()
    except RuntimeError as error:
        assert str(error) == 'original operator body failure'
    else:
        raise AssertionError('operator body failure was swallowed')
    final = json.loads(native._npu_eager_parallel_drain_close())
    assert final['enabled'] is False and final['failed'] == 1
    assert final['first_error'] == 'original operator body failure'
    assert final['closure']['passed'] is True and final['closure']['device_synchronized'] is False
    assert sys._is_gil_enabled() is False
    print(json.dumps({'scope': 'production native-binding CPU smoke; no NPU/model qualification',
                      'natural_gil_enabled_before': before, 'natural_gil_enabled_after': sys._is_gil_enabled(),
                      'tensor_results_equal': True, 'state': final}, indent=2))


if __name__ == '__main__':
    main()
