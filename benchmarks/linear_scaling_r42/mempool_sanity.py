"""Real NPU ordered private-pool/graph qualification; never performance evidence."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
import threading
import traceback


def run(config):
    spec = importlib.util.spec_from_file_location('_r42_pool_worker', Path(__file__).with_name('real_service.py'))
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    W.validate_gil('off', sys._is_gil_enabled(), sysconfig.get_config_var('Py_GIL_DISABLED'), sys._xoptions, os.environ)
    W.N.expected_treatment(config, os.environ)
    W.P.validate_environment(config, dict(device='npu',gil_mode='off',backend='off',stream_mode='private'), os.environ)
    W.verify_pins(config['pins'])
    source_sha = W.source_identity(config)
    assert W.sha(Path(sys.executable).resolve()) == config['python_sha256']
    os.sched_setaffinity(0, range(8,16))
    import torch
    import torch_npu
    assert sys.version_info[:3] == (3,14,6) and torch.__version__ == '2.13.0+cpu'
    assert not sys._is_gil_enabled()
    package = Path(torch_npu.__file__).resolve().parent
    assert package == Path(config['package']).resolve()
    native = package / 'lib/libtorch_npu.so'
    assert W.sha(native) == config['native_sha256']
    proof = W.N.NativeTreatment(native,config,os.environ)
    treatment = proof.read()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.manual_seed(6207)
    a_cpu, b_cpu = torch.randn(256,256), torch.randn(256,256)
    reference = torch.mm(a_cpu,b_cpu)
    a,b = a_cpu.to('npu'),b_cpu.to('npu')
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    mappings = W.native_mappings(config)

    def execute(repetitions=8):
        outputs=[]
        with torch.inference_mode(), torch.npu.stream(stream):
            for _ in range(repetitions):
                outputs.append(torch.mm(a,b))
            event = torch.npu.Event(enable_timing=False)
            event.record(stream)
            event.synchronize()
        return outputs

    def validate(outputs):
        for output in outputs:
            actual=output.cpu()
            assert output.device.type == 'npu' and bool(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5)

    before=proof.counters()
    warm=execute(12)
    validate(warm)
    after=proof.counters()
    assert after['hits'] > before['hits'] and after['allocations'] > before['allocations'], 'warm-cache path not reached'
    del warm
    scopes=[]
    for options in ({},dict(no_split=True),dict(use_on_oom=True)):
        pool=torch.npu.MemPool(**options)
        first=proof.counters()
        with torch.npu.use_mem_pool(pool):
            outputs=execute()
            inside=proof.counters()
            snapshot=pool.snapshot()
            assert snapshot, 'actual private pool has no segments'
            assert all(inside[k] == first[k] for k in ('hits','allocations','growths')), 'active pool unexpectedly used TLS cache'
        validate(outputs)
        resumed=execute()
        final=proof.counters()
        assert final['hits'] > inside['hits'], 'cache did not resume after endAllocateToPool'
        validate(resumed)
        scopes.append(dict(options=options,pool_id=list(pool.id),before=first,inside=inside,after=final,
                           private_segments=snapshot,validated_outputs=len(outputs)+len(resumed)))
        del outputs,resumed,pool

    entered,released=threading.Event(),threading.Event()
    other={}
    def holder():
        torch.npu.set_device(0)
        private_stream=torch.npu.Stream()
        pool=torch.npu.MemPool()
        try:
            with torch.npu.stream(private_stream),torch.npu.use_mem_pool(pool):
                allocation=torch.empty(1024,device='npu')
                private_stream.synchronize()
                other.update(pool_id=list(pool.id),private_segments=pool.snapshot())
                assert other['private_segments']
                entered.set()
                assert released.wait(60), 'ordered pool handoff release timeout'
        finally:
            entered.set()
        return True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(holder)
        try:
            assert entered.wait(60) and not future.done(), 'other-thread pool not active'
            first=proof.counters()
            outputs=execute()
            inside=proof.counters()
            assert all(inside[k] == first[k] for k in ('hits','allocations','growths')), 'ordered other-thread active pool unexpectedly used TLS cache'
            validate(outputs)
        finally:
            released.set()
        assert future.result(timeout=60)
    resumed=execute()
    final=proof.counters()
    assert final['hits'] > inside['hits']
    validate(resumed)
    other.update(before=first,inside=inside,after=final,validated_outputs=len(outputs)+len(resumed),handoff='threading.Event entered/released')
    del outputs,resumed

    graph=torch.npu.NPUGraph()
    graph_stream=torch.npu.Stream()
    graph_warm_before=proof.counters()
    with torch.npu.stream(graph_stream),torch.inference_mode():
        for _ in range(4):
            graph_output=torch.mm(a,b)
    graph_stream.synchronize()
    graph_before=proof.counters()
    assert graph_before['hits'] > graph_warm_before['hits'], 'graph stream was not warm-cache qualified'
    with torch.npu.graph(graph,stream=graph_stream),torch.inference_mode():
        graph_output=torch.mm(a,b)
    graph_after=proof.counters()
    assert all(graph_after[k] == graph_before[k] for k in ('hits','allocations','growths')), 'graph pool unexpectedly used TLS cache'
    for _ in range(8):
        graph.replay()
        torch.npu.synchronize()
        validate([graph_output])
    graph_pool=list(graph.pool())
    graph.reset()
    assert W.native_mappings(config)==mappings and proof.read()==treatment
    assert not sys._is_gil_enabled()
    W.verify_pins(config['pins'])
    W.source_identity(config)
    return dict(passed=True,performance_evidence=False,private_pool_scope_proven=True,natural_gil_off=True,
        native_sha256=config['native_sha256'],treatment=treatment,source_sha256=source_sha,
        provider_mappings=mappings,mempool_sanity_sha256=W.sha(Path(__file__)),
        validated_outputs=84,warm_cache_validation_outputs=12,same_thread_scopes=scopes,ordered_other_thread=other,
        cache_bypass_observation='warm same stream hits then stable TLS hits/allocations/growths during active scope; pool segments; cache resumes',
        graph=dict(passed=True,warm_before=graph_warm_before,before=graph_before,after=graph_after,pool_id=graph_pool,replays=8),
        limitation='ordered native scopes only; unordered cross-thread scope transitions and custom allocators not certified')


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(passed=False,performance_evidence=False)
    try:
        result=run(json.loads(args.config.read_text()))
    except BaseException as error:
        result['error']=dict(message=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,private_pool_scope_proven=True)),flush=True)


if __name__=='__main__':main()
