"""Bounded two-forward real preparation trace; excluded from performance cohort."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ingress', choices=('mutex', 'mpsc'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    os.environ['TORCH_NPU_EAGER_PREPARE_INGRESS'] = args.ingress
    os.sched_setaffinity(0, config['cpus'])
    import torch
    import torch_npu
    from torch_npu import _eager_prepare as prepare
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    spec = importlib.util.spec_from_file_location('r34_frozen_workload', config['workload'])
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    with torch.inference_mode():
        model, inputs, provenance = workload.load_original_model(config['upstream'])
        expected = model(*inputs).clone()
        model = model.to('npu:0')
        inputs = (inputs[0].npu(), [v.npu() for v in inputs[1]], [v.npu() for v in inputs[2]])
        torch.npu.synchronize()
        configured = prepare.configure(enabled=True, workers=4, capacity=8,
                                       allowed_ops=('aclnnEmbeddingBag',),
                                       diagnostic_trace=True, trace_limit=128)
        for _ in range(2):
            actual = model(*inputs)
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
        closed = prepare.drain_close()
    assert closed['real_sdk_qualified'] and closed['closure']['passed']
    trace = closed['diagnostic_trace']
    assert trace['complete'] and not trace['incomplete'] and len(trace['events']) == 50
    changes = []
    for event in trace['events']:
        changes.extend(((event['start_ns'], 1), (event['end_ns'], -1)))
    concurrency = peak = 0
    for _, change in sorted(changes):
        concurrency += change
        peak = max(peak, concurrency)
    result = dict(schema=1, passed=True, performance_evidence=False, ingress=args.ingress,
                  provenance=provenance, configured=configured, closed=closed,
                  peak_concurrent_prepare=peak,
                  prepare_wall_ns=[e['end_ns']-e['start_ns'] for e in trace['events']],
                  prepare_thread_cpu_ns=[e['end_thread_cpu_ns']-e['start_thread_cpu_ns']
                                         for e in trace['events']])
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, ingress=args.ingress, peak_concurrent_prepare=peak)), flush=True)


if __name__ == '__main__':
    main()
