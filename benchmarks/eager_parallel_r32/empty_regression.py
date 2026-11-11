"""Allocator-poisoned empty SUM regression for the actual native candidate.

This calls both real ATen entries, compares complete outputs with CPU, and
also checks nonempty inputs containing empty bags. It changes no model input.
"""
import argparse
import json
from pathlib import Path


def run():
    import torch
    import torch_npu
    from torch_npu import _eager_parallel as ep
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    with torch.inference_mode():
        ep.configure(enabled=False, mode='shadow')
        cpu_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        weight = cpu_weight.npu()
        for entry in ('_embedding_bag', '_embedding_bag_forward_only'):
            op = getattr(torch.ops.aten, entry).default
            for dtype in (torch.int32, torch.int64):
                for include_last in (False, True):
                    for bags in (1, 3, 128):
                        for empty in (False, True):
                            ids = torch.tensor([] if empty else [1, 2], dtype=dtype)
                            offsets = torch.zeros(bags + int(include_last), dtype=dtype)
                            if include_last:
                                offsets[-1] = ids.numel()
                            args = (cpu_weight, ids, offsets, False, 0, False, None, include_last, -1)
                            expected = op(*args)[0]
                            ni, no = ids.npu(), offsets.npu()
                            for repeat in range(3):
                                poison = torch.full((bags, 4), 77., device='npu:0')
                                torch.npu.synchronize()
                                del poison
                                output = op(weight, ni, no, False, 0, False, None, include_last, -1)
                                torch.npu.synchronize()
                                actual = output[0].cpu()
                                passed = actual.shape == expected.shape and torch.equal(actual, expected)
                                rows.append(dict(entry=entry, dtype=str(dtype), include_last_offset=include_last,
                                    bags=bags, empty=empty, repeat=repeat, passed=bool(passed),
                                    auxiliary_shapes=[list(x.shape) for x in output[1:]],
                                    max_abs_error=float((actual - expected).abs().max())))
        assert all(row['passed'] for row in rows), [row for row in rows if not row['passed']]
    return dict(passed=True, device=torch.npu.get_device_name(0), cases=rows,
                original_model_input_changed=False, backend='real_cann_native_aten')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], cases=len(result['cases']))), flush=True)
