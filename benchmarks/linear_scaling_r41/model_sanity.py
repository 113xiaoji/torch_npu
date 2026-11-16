"""Real-NPU numerical concurrency qualification across three Eager structures.

This is correctness evidence only. The original DLRM performance experiment is
in real_service.py; these models do not replace or alter that workload.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
import traceback


def load_worker():
    spec = importlib.util.spec_from_file_location('r41_sanity_worker',
        Path(__file__).with_name('real_service.py'))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def run(config):
    W = load_worker()
    W.validate_gil('off', sys._is_gil_enabled(), sysconfig.get_config_var('Py_GIL_DISABLED'),
                   sys._xoptions, os.environ)
    W.P.validate_environment(config, dict(device='npu', gil_mode='off', backend='off',
                                        stream_mode='private'), os.environ)
    W.N.expected_treatment(config, os.environ)
    W.verify_pins(config['pins'])
    source_sha = W.source_identity(config)
    assert W.sha(Path(sys.executable).resolve()) == config['python_sha256']
    assert sys.version_info[:3] == (3, 14, 6)
    os.sched_setaffinity(0, range(8, 16))
    import torch
    import torch_npu
    assert not sys._is_gil_enabled(), 'extension import enabled the GIL'
    assert torch.__version__ == '2.13.0+cpu'
    package = Path(torch_npu.__file__).resolve().parent
    assert package == Path(config['package']).resolve()
    native = package / 'lib/libtorch_npu.so'
    assert W.sha(native) == config['native_sha256']
    proof = W.N.NativeTreatment(native, config, os.environ)
    treatment = proof.read()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(32, 32)
            self.k = torch.nn.Linear(32, 32)
            self.v = torch.nn.Linear(32, 32)
        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (32 ** -0.5)
            return torch.matmul(torch.softmax(scores, dim=-1), v)

    torch.manual_seed(5173)
    builders = (
        ('mlp', lambda: torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32)), (8, 64)),
        ('cnn', lambda: torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AvgPool2d(2), torch.nn.Flatten(), torch.nn.Linear(8*8*8, 16)), (2, 3, 16, 16)),
        ('attention', Attention, (2, 16, 32)),
    )
    receipts = []
    for name, builder, shape in builders:
        model = builder().eval()
        original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bank = [torch.randn(shape) for _ in range(8)]
        with torch.inference_mode():
            references = [model(x).clone() for x in bank]
        model = model.to('npu')
        streams = [torch.npu.Stream() for _ in range(4)]
        inputs = [[x.clone().to('npu') for x in bank] for _ in streams]
        torch.npu.synchronize()
        with torch.inference_mode():
            warm_output = model(inputs[0][0])
        torch.npu.synchronize()
        torch.testing.assert_close(warm_output.cpu(), references[0], rtol=2e-4, atol=2e-5)
        del warm_output
        mappings_before = W.native_mappings(config)

        def caller(index):
            torch.npu.set_device(0)
            outputs = []
            with torch.inference_mode(), torch.npu.stream(streams[index]):
                event = torch.npu.Event(enable_timing=False)
                for request in range(16):
                    output = model(inputs[index][request % 8])
                    event.record(torch.npu.current_stream(0))
                    event.synchronize()
                    outputs.append(output)
                counters = proof.counters()
            return outputs, counters

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(caller, i) for i in range(4)]
            results = [future.result(timeout=120) for future in futures]
        torch.npu.synchronize()
        maximum = 0.0
        for outputs, _ in results:
            for request, output in enumerate(outputs):
                actual = output.cpu()
                assert output.device.type == 'npu'
                assert bool(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, references[request % 8], rtol=2e-4, atol=2e-5)
                maximum = max(maximum, float((actual - references[request % 8]).abs().max()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), original_state[key], rtol=0, atol=0)
        mappings_after = W.native_mappings(config)
        assert mappings_before == mappings_after
        receipts.append(dict(model=name, callers=4, requests=64, passed=True,
            output_shape=list(references[0].shape), max_abs_error=maximum,
            weights_unchanged=True, workspace_caller_counters=[row[1] for row in results],
            provider_mappings=mappings_after))
        del model, inputs, results, outputs, futures
    assert proof.read() == treatment
    assert not sys._is_gil_enabled()
    W.verify_pins(config['pins'])
    return dict(passed=True, performance_evidence=False, natural_gil_off=True,
                native_sha256=config['native_sha256'], treatment=treatment,
                source_sha256=source_sha, model_sanity_sha256=W.sha(Path(__file__)),
                total_requests=192, models=receipts)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, performance_evidence=False)
    try:
        result = run(json.loads(args.config.read_text()))
    except BaseException as error:
        result['error'] = dict(message=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, requests=result['total_requests'])), flush=True)


if __name__ == '__main__':
    main()
