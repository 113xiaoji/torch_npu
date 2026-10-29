"""Fail-closed contracts for R25 DLRM host simulation, never real-device gates."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics

ARMS = {'off': 0, 'queue_w1': 1, 'queue_w2': 2, 'queue_w4': 4}
R22_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
TOPOLOGY_SHA = '5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a'
SOURCE_SHA = '9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5'
STATE_SHA = '6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19'
INPUT_SHA = '239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6'
TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(value, label):
    require(type(value) is dict and set(value)=={'path','sha256'}, label+' exact path/SHA pin required')
    require(type(value['path']) is str and value['path'].startswith('/') and '..' not in PurePosixPath(value['path']).parts,
            label+' absolute lexical Linux path required')
    require(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),label+' SHA256 required')


def verify_pin(value, label):
    pin(value,label)
    require(sha(value['path'])==value['sha256'],label+' source/artifact SHA mismatch')
    return copy.deepcopy(value)


def candidate_pythonpath(runtime):
    candidate=str(PurePosixPath(runtime['torch_npu_package']).parent)
    require(runtime['pythonpath']==[candidate],
        'candidate-only pythonpath required; original model dependencies are fallback imports')
    return [candidate]


def validate_config(config):
    c=copy.deepcopy(config)
    require(set(c)=={'schema','cpus','numa_node','helpers','sdk','runtimes'},'unknown/missing config fields')
    require(type(c['schema']) is int and c['schema']==1,'schema1 required')
    cpus=c['cpus']
    require(type(cpus) is list and len(cpus)==len(set(cpus))==8
        and all(type(v) is int and v>=0 for v in cpus),'exactly eight unique CPUs required')
    require(type(c['numa_node']) is int and c['numa_node']>=0,'NUMA node required')
    require(set(c['helpers'])=={'r22','topology'},'two pinned pure/inference helper files required')
    for name,expected in [('r22',R22_SHA),('topology',TOPOLOGY_SHA)]:
        pin(c['helpers'][name],name)
        require(c['helpers'][name]['sha256']==expected,'unreviewed helper source '+name)
    sdk=c['sdk']
    require(set(sdk)=={'backend','real_sdk_qualified','source_pins','libraries'},'SDK exact fields required')
    require(sdk['backend']=='host_sim' and sdk['real_sdk_qualified'] is False,'SIM backend must never be real-qualified')
    require(type(sdk['source_pins']) is dict and bool(sdk['source_pins']),'new SDK source pins required')
    for name,digest in sdk['source_pins'].items():
        pin(dict(path=name,sha256=digest),'SDK source')
    require(set(sdk['libraries'])=={'libopapi.so','libascendcl.so'},'the same two new simulator libraries required')
    for name,value in sdk['libraries'].items():pin(value,name)
    require(type(c['runtimes']) is list and 1<=len(c['runtimes'])<=2,'one or two natural-GIL runtimes required')
    names=set()
    for rt in c['runtimes']:
        require(set(rt)=={'name','python','expected_gil','pythonpath','ld_library_path','artifact_pins',
                         'torch_npu_package','model_manifest'},'runtime exact fields required')
        require(rt['name'] in ('std','ft') and rt['name'] not in names,'unique std/ft runtime required');names.add(rt['name'])
        require(rt['expected_gil'] is (rt['name']=='std'),'natural std/FT GIL identity required')
        for key in ('python','torch_npu_package'):
            require(type(rt[key]) is str and rt[key].startswith('/') and '..' not in PurePosixPath(rt[key]).parts,'absolute lexical '+key+' required')
        for key in ('pythonpath','ld_library_path'):
            require(type(rt[key]) is list and rt[key] and all(type(v) is str and v.startswith('/') and ':' not in v for v in rt[key]),'explicit '+key+' required')
        candidate_pythonpath(rt)
        require(all(str(PurePosixPath(v['path']).parent) in rt['ld_library_path'] for v in sdk['libraries'].values()),'both new SDK library directories required')
        require(type(rt['artifact_pins']) is dict and rt['artifact_pins'],'new package/native artifact pins required')
        require(rt['torch_npu_package']+'/lib/libtorch_npu.so' in rt['artifact_pins'],'main native artifact pin required')
        for path,digest in rt['artifact_pins'].items():pin(dict(path=path,sha256=digest),'runtime artifact')
        pin(rt['model_manifest'],'original DLRM asset manifest')
    return c


def make_plan(config):
    c=validate_config(config)
    orders=[list(ARMS),list(reversed(ARMS)),['queue_w2','queue_w4','off','queue_w1']]
    def row(rt,arm,kind,index):
        return dict(runtime=rt['name'],arm=arm,workers=ARMS[arm],kind=kind,round=index,
                    warmup=0 if kind=='cold' else 20,samples=1 if kind=='cold' else 200,
                    timeout_seconds=600,fresh_process=True,diagnostic_trace=kind=='cold')
    return dict(schema=1,backend='host_sim',real_sdk_qualified=False,
        cold=[row(rt,a,'cold',0) for rt in c['runtimes'] for a in ARMS],
        perf=[row(rt,a,'perf',i) for rt in c['runtimes'] for i,order in enumerate(orders) for a in order],
        measurement='original unmodified DLRM forward entry through original torch.npu.synchronize return',
        pool_startup_included=False,synthetic_prepare_cpu_or_delay=False,
        order_policy='three matched four-arm rounds; forward, reverse, then declared two-arm rotation',
        numerical_claim=False,true_device_kernel_claim=False)


def worker_environment(config,runtime,row,inherited=None):
    incoming=os.environ if inherited is None else inherited
    env={k:v for k,v in incoming.items() if not (k.startswith(('PYTHON','FTLAB_','HDB_','CINDER','TORCH_NPU_','ASCEND_','TASK_QUEUE','PER_STREAM','QWEN_','LD_'))
        or k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))}
    env.update(PYTHONPATH=':'.join(candidate_pythonpath(runtime)),LD_LIBRARY_PATH=':'.join(runtime['ld_library_path']),
        PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',
        TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1' if runtime['name']=='ft' else '0',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        TASK_QUEUE_ENABLE='1',PER_STREAM_QUEUE='0',ASCEND_LAUNCH_BLOCKING='0',QWEN_ACL_HOST_SIM='1',
        FTLAB_HOST_REGIONS='0',FTLAB_HOST_NATIVE_RUN='0',HDB_REGION_PACKET='0',HDB_REGION_REAL_OPAPI='0',
        HDB_REGION_NATIVE_BUILDER='0',FTLAB_HOST_OWNER_RETIREMENT='0',HDB_REGION_TRACE='0',HDB_REGION_DIAGNOSTICS='0')
    return env


def validate_mapping(rows,pins):
    for name,expected in pins.items():
        found=[r for r in rows if r.get('name')==name]
        require(len(found)==1,'missing/duplicate loaded simulator mapping: '+name)
        require(Path(found[0]['path']).resolve()==Path(expected['path']).resolve()
            and found[0]['sha256']==expected['sha256'],'loaded simulator mapping SHA/path mismatch: '+name)
    return True


def integer(row,key):
    value=row.get(key)
    require(type(value) is int and value>=0,'missing/inexact counter: '+key)
    return value


def sim_identity(snapshot):
    require(snapshot.get('backend')=='host_sim' and snapshot.get('real_sdk_qualified') is False
        and snapshot.get('simulated_prepare') is True and snapshot.get('qualified_ops')==[],
        'native backend must explicitly remain simulated and never real-qualified')
    require(snapshot.get('first_error') in (None,''),'native first error: '+str(snapshot.get('first_error')))


def validate_mechanism(before,after,closed,*,forwards,workers):
    require(type(forwards) is int and forwards>0 and workers in (0,1,2,4),'exact forward/worker count required')
    for snap in (before,after,closed):sim_identity(snap)
    delta={}
    for name,values in after['operators'].items():
        old=before['operators'].get(name,{})
        delta[name]={}
        for key in ('seen','eligible','queued','prepared','executed','returned','fallback'):
            change=integer(values,key)-integer(old,key)
            require(change>=0,'native op counter regressed '+name+'/'+key);delta[name][key]=change
        queued=delta[name]['queued']
        require(all(delta[name][key]==queued for key in ('prepared','executed','returned')),
                'admitted/prepared/executed/returned mismatch '+name)
    admitted=integer(after['queue'],'admitted')-integer(before['queue'],'admitted')
    require(admitted==sum(v['queued'] for v in delta.values()),'queue and operator admission counts differ')
    require(integer(after['queue'],'retired')-integer(before['queue'],'retired')==admitted,'queue retirement count differs')
    embedding=delta.get('aclnnEmbeddingBag',{}).get('queued',0)
    if workers:
        require(embedding==26*forwards,'exactly 26 actual queued EmbeddingBag commands per forward required')
        require(integer(after['queue'],'workers_started')==workers,'requested worker count not actually started')
        order=closed.get('order_evidence',{})
        require(order.get('passed') is True and integer(order,'admitted_count')==integer(order,'executed_count')
            and order.get('admitted_sequence_hash')==order.get('executed_sequence_hash'),'closed native FIFO sequence/count mismatch')
    else:
        require(admitted==0,'disabled arm unexpectedly queued native work')
    for key in ('slots_in_use','retained_bytes'):
        require(integer(closed['queue'],key)==0,'native queue resource not zero at close: '+key)
    close_fields=('admission_closed','fifo_drained','device_synchronized','all_retired','workers_stopped','close_attempted','passed') if workers else ('admission_closed','all_retired')
    for key in close_fields:
        require(closed['closure'].get(key) is True,'native close did not pass '+key)
    if not workers:require(integer(closed['queue'],'workers_started')==0,'disabled arm started an owned core')
    seen=sum(v['seen'] for v in delta.values());eligible=sum(v['eligible'] for v in delta.values())
    return dict(passed=True,forwards=forwards,admitted=admitted,embedding_commands_per_forward=embedding//forwards,
        operators=delta,all_observed_ops=seen,eligible_observed_ops=eligible,
        coverage_scope='partial native operator coverage',all_operators_covered=False,
        real_sdk_qualified=False,all_resources_closed=True,optimization_active=workers>0,
        observed_count_scope='registered native hook calls only; other model operators are not counted')


def validate_sdk(snapshot,*,closed=False):
    require(snapshot.get('backend_kind')=='HOST_SIM_R25' and snapshot.get('numeric_supported') is False
        and snapshot.get('real_sdk_qualified') is False and snapshot.get('artificial_prepare_delay_ns')==0,
        'new simulator identity/scope mismatch')
    require(integer(snapshot,'errors')==integer(snapshot,'cache_stub_calls')==0,'SDK error or cache stub was called')
    if closed:
        for key in ('live_descriptors','live_executors','huge_pending','huge_active_threads'):
            require(integer(snapshot,key)==0,'SDK resource not zero '+key)
        require(integer(snapshot,'descriptors_created')==integer(snapshot,'descriptors_destroyed'),'SDK descriptors not balanced')
        require(integer(snapshot,'executors_created')==integer(snapshot,'executors_executed')+integer(snapshot,'executors_cancelled'),'SDK executors not balanced')
        require(integer(snapshot,'huge_init')==integer(snapshot,'huge_uninit'),'SDK huge thread scopes not balanced')
    return True


def validate_sdk_delta(before,after,*,forwards,queued_embeddings):
    old=before['operations']['aclnnEmbeddingBag'];new=after['operations']['aclnnEmbeddingBag']
    prepared=integer(new,'prepared')-integer(old,'prepared')
    executed=integer(new,'executed')-integer(old,'executed')
    cancelled=integer(new,'cancelled')-integer(old,'cancelled')
    require(prepared==executed==26*forwards and cancelled==0,
        'SIM SDK must Prepare and Execute exactly 26 original EmbeddingBag calls per forward')
    require(queued_embeddings in (0,prepared),'SIM SDK count differs from native queue admissions')
    return dict(passed=True,prepared=prepared,executed=executed,embedding_commands_per_forward=prepared//forwards,
                numerical_golden_claim=False,real_sdk_qualified=False)


def validate_cold(native,sdk,workers):
    require(sdk.get('tracing_enabled') is True and integer(sdk,'trace_dropped')==0,'SDK cold trace missing/truncated')
    if workers:
        order=native.get('order_evidence',{})
        require(order.get('passed') is True and integer(order,'admitted_count')==integer(order,'executed_count')==26
            and type(order.get('admitted_sequence_hash')) is int
            and order['admitted_sequence_hash']==order.get('executed_sequence_hash'),'native caller FIFO sequence proof missing')
        diagnostic=native.get('diagnostic_trace',{})
        require(diagnostic.get('enabled') is True and diagnostic.get('complete') is True
            and diagnostic.get('incomplete') is False and integer(diagnostic,'dropped')==0,'native cold trace missing/truncated')
        trace=diagnostic.get('events',[])
        require(type(trace) is list and len(trace)==26,'cold native prepare trace must contain exactly 26 EmbeddingBag records')
        callers={integer(row,'caller_tid') for row in trace};preparers={integer(row,'native_tid') for row in trace};consumers={integer(row,'consumer_tid') for row in trace}
        require(0 not in callers|preparers|consumers and len(callers)==len(consumers)==1 and len(preparers)==workers
            and callers.isdisjoint(preparers) and callers.isdisjoint(consumers) and preparers.isdisjoint(consumers),
            'cold caller/worker/ACL consumer TID handoff missing')
        require(preparers<=set(sdk.get('prepare_tids',[])) and consumers<=set(sdk.get('execute_tids',[])),
            'SDK prepare/execute native TIDs differ from queue handoff')
    return dict(passed=True,diagnostic_only=True,performance_eligible=False,real_sdk_qualified=False)


def distribution(values):
    require(type(values) is list and values and all(type(v) is int and v>0 for v in values),'positive exact host duration samples required')
    ordered=sorted(values)
    return dict(samples=len(values),median_ns=statistics.median(values),p95_ns=ordered[math.ceil(.95*len(ordered))-1])


def summarize(rows):
    require(bool(rows),'no complete first-screen rows')
    result={}
    for runtime in sorted({r['runtime'] for r in rows}):
        selected=[r for r in rows if r['runtime']==runtime];index={(r['round'],r['arm']):r for r in selected}
        require(len(index)==len(selected)==12 and set(index)=={(i,a) for i in range(3) for a in ARMS},'incomplete/duplicate three-round four-arm screen')
        result[runtime]={}
        for arm in list(ARMS)[1:]:
            pairs=[]
            for i in range(3):
                base=distribution(index[i,'off']['durations_ns']);candidate=distribution(index[i,arm]['durations_ns'])
                require(base['samples']==candidate['samples']==200,'first screen requires exactly 200 samples')
                pairs.append(dict(round=i,off=base,candidate=candidate,speedup=base['median_ns']/candidate['median_ns'],p95_ratio=base['p95_ns']/candidate['p95_ns']))
            ratios=[p['speedup'] for p in pairs]
            verdict='benefit observed in first screen' if min(ratios)>1 else ('regression observed in first screen' if max(ratios)<1 else 'mixed first screen; no stable direction')
            result[runtime][arm]=dict(paired_rounds=3,pairs=pairs,median_paired_speedup=statistics.median(ratios),
                speedup_range=[min(ratios),max(ratios)],classification=verdict,
                inference_scope='actual DLRM host forward through synchronize with simulated kernels',
                formal_significance_claim=False,numerical_golden_claim=False,true_device_kernel_claim=False,
                pool_startup_included=False)
    return result
