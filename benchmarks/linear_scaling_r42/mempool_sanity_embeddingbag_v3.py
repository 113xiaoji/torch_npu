"""Real NPU ordered private-pool/graph qualification; never performance evidence."""
import argparse
import concurrent.futures
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
import threading
import traceback
import time


def assert_warm_reached(before,after):
    assert after['enabled']==1 and after['hits']>before['hits'] and after['allocations']>before['allocations'], 'same original operator warm-cache path not reached'
    assert after['entries']>0 and after['retained_bytes']>0, 'warm operation retained no native workspace'


def assert_scope_bypass(before,inside):
    assert all(inside[k]==before[k] for k in ('hits','allocations','growths')), 'active pool unexpectedly used TLS cache'


def run(config, diagnostics, checkpoint):
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
    workload=W.load(config['workload'],'_r42_scope_original_workload')
    with torch.inference_mode():
        model,template,provenance=workload.load_original_model(config['upstream'])
        assert all(provenance['identity'].get(k)==v for k,v in W.C.IDENTITY.items())
        specs=W.R.build_request_bank(template[0].tolist(),[x.tolist() for x in template[1]],[x.tolist() for x in template[2]])
        bank=[(torch.tensor(row['dense'],dtype=torch.float32),[torch.tensor(v,dtype=torch.int64) for v in row['offsets']],
            [torch.tensor(v,dtype=torch.int64) for v in row['indices']]) for row in specs]
        input_hashes=[workload.helper.tensor_tree_sha256(row,torch) for row in bank]
        assert input_hashes[0]==W.C.IDENTITY['input_sha256'] and len(set(input_hashes))==8
        state_sha=workload.helper.tensor_tree_sha256(dict(model.state_dict()),torch)
        assert state_sha==W.C.IDENTITY['state_sha256']
        table=max(range(len(model.emb_l)),key=lambda i:bank[0][2][i].numel())
        assert len(model.emb_l)==26 and bank[0][2][table].numel()>0
        selected=copy.deepcopy(model.emb_l[table]).eval();selected.requires_grad_(False)
        selected_state=workload.helper.tensor_tree_sha256(dict(selected.state_dict()),torch)
        references=[selected(row[2][table],row[1][table]).clone() for row in bank]
        selected=selected.to('npu:0')
        inputs=[(row[2][table].to('npu',copy=True),row[1][table].to('npu',copy=True)) for row in bank]
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    diagnostics.update(operator='original_model.EmbeddingBag',table=table,
        original_model_identity=provenance['identity'],original_state_sha256=state_sha,
        selected_weight_state_sha256=selected_state,input_hashes=input_hashes,
        selected_indices_shape=list(bank[0][2][table].shape),selected_offsets_shape=list(bank[0][1][table].shape),
        validated_outputs=0,phases=[])
    checkpoint()
    def stamp(stage, **details):
        diagnostics['phases'].append(dict(stage=stage,monotonic_ns=time.monotonic_ns(),counters=proof.counters(),**details))
        checkpoint()
    mappings = W.native_mappings(config)

    def execute(repetitions=8):
        outputs=[]
        with torch.inference_mode(), torch.npu.stream(stream):
            for index in range(repetitions):
                outputs.append(selected(*inputs[index%8]))
            event = torch.npu.Event(enable_timing=False)
            event.record(stream)
            event.synchronize()
        return outputs

    def validate(outputs):
        for index,output in enumerate(outputs):
            actual=output.cpu()
            assert output.device.type == 'npu' and bool(torch.isfinite(actual).all())
            torch.testing.assert_close(actual,references[index%8],rtol=1e-4,atol=1e-5)
            diagnostics['validated_outputs']+=1;checkpoint()

    before=proof.counters();stamp('warm_before',before=before)
    warm=execute(12)
    after=proof.counters();stamp('warm_executed_before_numeric',before=before,after=after)
    validate(warm)
    after=proof.counters();stamp('warm_after_numeric',before=before,after=after)
    assert_warm_reached(before,after)
    del warm
    scopes=[]
    for options in ({},dict(no_split=True),dict(use_on_oom=True)):
        pool=torch.npu.MemPool(**options)
        first=proof.counters();stamp('scope_before',options=options,before=first)
        with torch.npu.use_mem_pool(pool):
            outputs=execute()
            inside=proof.counters()
            snapshot=pool.snapshot();stamp('scope_inside',options=options,before=first,inside=inside,private_segments=snapshot)
            assert snapshot, 'actual private pool has no segments'
            assert_scope_bypass(first,inside)
        validate(outputs)
        resumed=execute()
        final=proof.counters();stamp('scope_after',options=options,before=first,inside=inside,after=final)
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
            first=proof.counters();stamp('other_thread_active_before',before=first)
            outputs=execute()
            inside=proof.counters()
            stamp('other_thread_inside',before=first,inside=inside,private_segments=other['private_segments'])
            assert_scope_bypass(first,inside)
            validate(outputs)
        finally:
            released.set()
        assert future.result(timeout=60)
    resumed=execute()
    final=proof.counters();stamp('other_thread_after',before=first,inside=inside,after=final)
    assert final['hits'] > inside['hits']
    validate(resumed)
    other.update(before=first,inside=inside,after=final,validated_outputs=len(outputs)+len(resumed),handoff='threading.Event entered/released')
    del outputs,resumed

    graph=torch.npu.NPUGraph()
    graph_stream=torch.npu.Stream()
    graph_warm_before=proof.counters();stamp('graph_warm_before',before=graph_warm_before)
    with torch.npu.stream(graph_stream),torch.inference_mode():
        graph_warm_outputs=[]
        for _ in range(4):
            graph_warm_outputs.append(selected(*inputs[0]))
    graph_stream.synchronize()
    graph_before=proof.counters();stamp('graph_warm_after',before=graph_warm_before,after=graph_before)
    for graph_warm_output in graph_warm_outputs:
        validate([graph_warm_output])
    del graph_warm_outputs,graph_warm_output
    assert graph_before['hits'] > graph_warm_before['hits'], 'graph stream was not warm-cache qualified'
    stamp('graph_capture_before',before=graph_before)
    with torch.npu.graph(graph,stream=graph_stream),torch.inference_mode():
        graph_output=selected(*inputs[0])
    graph_after=proof.counters();stamp('graph_capture_after',before=graph_before,after=graph_after)
    assert all(graph_after[k] == graph_before[k] for k in ('hits','allocations','growths')), 'graph pool unexpectedly used TLS cache'
    for replay in range(8):
        graph.replay()
        torch.npu.synchronize()
        validate([graph_output]);stamp('graph_replay_validated',replay=replay)
    graph_pool=list(graph.pool())
    graph.reset()
    assert workload.helper.tensor_tree_sha256(dict(model.state_dict()),torch)==state_sha
    assert workload.helper.tensor_tree_sha256(dict(selected.state_dict()),torch)==selected_state
    assert diagnostics['validated_outputs']==88
    assert W.native_mappings(config)==mappings and proof.read()==treatment
    assert not sys._is_gil_enabled()
    W.verify_pins(config['pins'])
    W.source_identity(config)
    return dict(passed=True,performance_evidence=False,private_pool_scope_proven=True,natural_gil_off=True,
        native_sha256=config['native_sha256'],treatment=treatment,source_sha256=source_sha,
        provider_mappings=mappings,mempool_sanity_sha256=W.sha(Path(__file__)),
        validated_outputs=88,warm_cache_validation_outputs=12,same_thread_scopes=scopes,ordered_other_thread=other,
        cache_bypass_observation='warm same stream hits then stable TLS hits/allocations/growths during active scope; pool segments; cache resumes',
        operator='original_model.EmbeddingBag',selected_table=table,original_model_identity=provenance['identity'],input_hashes=input_hashes,
        diagnostics=diagnostics,graph=dict(passed=True,warm_validated_outputs=4,warm_before=graph_warm_before,before=graph_before,after=graph_after,pool_id=graph_pool,replays=8),
        limitation='single original EmbeddingBag and ordered native scopes only; full DLRM graph, unordered transitions and custom allocators not certified')


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    diagnostics={}
    def checkpoint():
        target=args.output.with_name('progress.json');temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n');temporary.replace(target)
    result=dict(passed=False,performance_evidence=False)
    try:
        result=run(json.loads(args.config.read_text()),diagnostics,checkpoint)
    except BaseException as error:
        result['error']=dict(message=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result['diagnostics']=diagnostics
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,private_pool_scope_proven=True)),flush=True)


if __name__=='__main__':main()
