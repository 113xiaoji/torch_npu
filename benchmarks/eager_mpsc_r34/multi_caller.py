"""Real native multi-caller qualification; separate from frozen DLRM E2E."""
import argparse
import json
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingress', choices=('mutex', 'mpsc'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['TORCH_NPU_EAGER_PREPARE_INGRESS'] = args.ingress
    os.environ['TORCH_NPU_EAGER_PREPARE_MULTI_CALLER'] = '1'
    import torch
    import torch_npu
    from torch_npu import _eager_prepare as queue
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    barrier = threading.Barrier(4)
    torch.manual_seed(1702)
    with torch.inference_mode():
        weight = torch.randn(1024, 16)
        cases = []
        for length in (2, 8, 64, 128):
            ids = torch.randint(1024, (128 * length,), dtype=torch.int64)
            offsets = torch.arange(128, dtype=torch.int64) * length
            expected = torch.nn.functional.embedding_bag(ids, weight, offsets,
                                                         mode='sum', sparse=True)
            cases.append((ids.npu(), offsets.npu(), expected))
        weight_n = weight.npu()
        torch.npu.synchronize()
        configured = queue.configure(enabled=True, workers=4, capacity=8,
                                     allowed_ops=('aclnnEmbeddingBag',),
                                     diagnostic_trace=True, trace_limit=64)
        assert configured['real_sdk_qualified'] and configured['multiple_callers']

        def caller(index):
            with torch.inference_mode():
                torch.npu.set_device(0)
                ids, offsets, _ = cases[index]
                barrier.wait(timeout=30)
                values = [torch.nn.functional.embedding_bag(
                    ids, weight_n, offsets, mode='sum', sparse=True) for _ in range(8)]
                return dict(index=index, tid=threading.get_native_id(), values=values)

        with ThreadPoolExecutor(4) as callers:
            futures = [callers.submit(caller, i) for i in range(4)]
            results = [future.result(timeout=90) for future in futures]
        torch.npu.synchronize()
        errors = []
        for row in results:
            for value in row.pop('values'):
                actual = value.cpu()
                expected = cases[row['index']][2]
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                errors.append(float((actual-expected).abs().max()))
        closed = queue.drain_close()
    counts = closed['operators']['aclnnEmbeddingBag']
    assert all(counts[name] == 32 for name in ('queued', 'prepared', 'executed', 'returned', 'sdk_prepare_calls'))
    assert closed['queue']['admitted'] == closed['queue']['retired'] == 32
    assert closed['closure']['passed'] and closed['order_evidence']['passed']
    trace = closed['diagnostic_trace']
    assert trace['complete'] and not trace['incomplete'] and len(trace['events']) == 32
    assert {event['caller_tid'] for event in trace['events']} == {row['tid'] for row in results}
    publisher_tids = {event['publisher_tid'] for event in trace['events']}
    if args.ingress == 'mpsc':
        assert len(publisher_tids) == 1 and publisher_tids.isdisjoint(row['tid'] for row in results)
    else:
        assert publisher_tids == {row['tid'] for row in results}
    assert len({event['consumer_tid'] for event in trace['events']}) == 1
    result = dict(schema=1, passed=True, ingress=args.ingress, callers=results,
                  max_abs_error=max(errors), configured=configured, closed=closed,
                  performance_evidence=False)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, ingress=args.ingress, max_abs_error=max(errors))), flush=True)


if __name__ == '__main__':
    main()
