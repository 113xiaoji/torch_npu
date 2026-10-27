"""Plan/execute four fixed R22 arms in existing isolated runtimes; installs nothing."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback

BASELINE_WHEEL_SHA={
    'std':'7375e9891e587e3b3d507d11f42d56cc8df69b1ffc845980a79192e844894ff6',
    'ft':'575654b8d93f3a8be8860bb79bee8a11aad8b94e8c59c0a861418ac36bd31959'}
CONTRACTS_SHA256='5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a'


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def load_contracts():
    path=Path(__file__).with_name('measurement_contracts_r22.py')
    require(sha(path)==CONTRACTS_SHA256,'shared topology/stability contract source SHA mismatch')
    spec=importlib.util.spec_from_file_location('_r22_measurement_contracts',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--arms-manifest',type=Path,required=True)
    for name in ('runner','upstream','cann-prefix','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runner-sha256',required=True)
    p.add_argument('--cann-env',type=Path)
    p.add_argument('--runtime-receipt',type=Path)
    p.add_argument('--cpus',default='auto',help='eligible CPU pool, or auto=current allowed mask; topology chooses eight physical cores')
    p.add_argument('--execute',action='store_true')
    return p.parse_args(argv)


def load_arm_manifest(path):
    data=json.loads(Path(path).read_text())
    require(type(data.get('schema')) is int and data['schema']==1 and type(data.get('arms')) is list
            and len(data['arms'])==4,'four physical arms in manifest schema1 required')
    result={}
    for row in data['arms']:
        require(type(row) is dict,'arm binding must be an object')
        key=row.get('abi'),row.get('arm')
        require(key in {(a,b) for a in ('std','ft') for b in ('serial','worker4')} and key not in result,'invalid/duplicate arm')
        for name in ('python','wheel','build_receipt'):
            require(type(row.get(name)) is str and row[name].startswith('/') and '..' not in Path(row[name]).parts,
                    'absolute physical arm path required: '+name)
        for name in ('wheel_sha256','native_sha256','build_receipt_sha256'):
            require(type(row.get(name)) is str and re.fullmatch('[0-9a-f]{64}',row[name]),'physical artifact SHA required: '+name)
        if key[1]=='serial':
            require(row['wheel_sha256']==BASELINE_WHEEL_SHA[key[0]],'serial must use the original frozen R21 binary')
        else:
            require(row['wheel_sha256']!=BASELINE_WHEEL_SHA[key[0]],'candidate must be an independently pinned R22 artifact')
        adapter=row.get('dependency_adapter',{'kind':'none'})
        require(type(adapter) is dict and adapter.get('kind') in ('none','pth'),'explicit dependency adapter kind required')
        if adapter['kind']=='pth':
            require(key[1]=='worker4','original baseline environment may not acquire a dependency adapter')
            require(all(type(adapter.get(name)) is str and adapter[name].startswith('/') for name in ('path','target'))
                    and re.fullmatch('[0-9a-f]{64}',str(adapter.get('sha256',''))),'pinned explicit .pth path and target required')
        result[key]={**row,'dependency_adapter':adapter}
    for abi in ('std','ft'):
        require(result[abi,'serial']['python']!=result[abi,'worker4']['python'],'original and candidate runtimes must be separate')
    return result


def make_plan(args):
    cpus=[] if args.cpus=='auto' else [int(v) for v in args.cpus.split(',')]
    require(args.cpus=='auto' or (len(cpus)==len(set(cpus))>=8 and min(cpus)>=0),'at least eight unique eligible CPUs required')
    require(re.fullmatch('[0-9a-f]{64}',args.runner_sha256),'invalid runner SHA256')
    bindings=load_arm_manifest(args.arms_manifest)
    arms=[('std','serial'),('std','worker4'),('ft','serial'),('ft','worker4')]
    def row(abi,arm,kind,round_index):
        prefix='gate-'+abi+'-'+arm if kind=='gate' else ('stability-'+abi+'-'+arm if kind=='stability' else 'round'+str(round_index)+'-'+abi+'-'+arm)
        output=args.output/(prefix+'.json')
        iterations={'gate':1,'perf':2000,'stability':20000}[kind]
        binding=bindings[abi,arm]
        command=[binding['python'],'-I','-B',str(args.runner),'--abi',abi,'--arm',arm,'--kind',kind,
            '--upstream',str(args.upstream),'--cann-prefix',str(args.cann_prefix),'--cpus',args.cpus,
            '--wheel',binding['wheel'],'--expected-wheel-sha256',binding['wheel_sha256'],
            '--expected-native-sha256',binding['native_sha256'],'--threads','1',
            '--warmup','0' if kind=='gate' else '20','--iterations',str(iterations),'--output',str(output)]
        if args.runtime_receipt: command += ['--runtime-receipt',str(args.runtime_receipt)]
        if kind in ('perf','stability') and arm=='worker4': command += ['--gate-proof',str(args.output/('gate-'+abi+'-worker4.json'))]
        return dict(abi=abi,arm=arm,kind=kind,round=round_index,output=str(output),argv=command,binding=binding,
            warmup=0 if kind=='gate' else 20,iterations=iterations,cpus=cpus,fresh_process=True,
            timeout_seconds=3600 if kind=='stability' else 1800)
    gates=[row(*arm,'gate',0) for arm in arms]
    orders=[arms[i:]+arms[:i] for i in range(4)]+[list(reversed(arms))]
    repeat=[row(*arm,'perf',i+1) for i,order in enumerate(orders) for arm in order]
    stability=[row(*arm,'stability',0) for arm in arms]
    return dict(schema=1,plan_only=not args.execute,gates=gates,repeat=repeat,stability=stability,rounds=5,
        topology_pending=True,requested_eligible_cpus=args.cpus,contracts_sha256=CONTRACTS_SHA256,
        arms_manifest=dict(path=str(args.arms_manifest),sha256=sha(args.arms_manifest)),
        timing='forward through original device synchronize; diagnostics/trace disabled in perf',
        order_policy='four Latin-square rotations plus a declared reversed fifth round',
        cpu_evidence='cold threadCPU/monotonic window plus native TIDs; no perf/sched-trace assumption',
        installs_packages=False,rebuilds_runtime=False,changes_sdk_or_driver=False)


def bind_topology(plan,receipt,path,proof_sha):
    bound=copy.deepcopy(plan);cpus=receipt['selected_cpus']
    for row in bound['gates']+bound['repeat']+bound['stability']:
        row['cpus']=list(cpus);argv=row['argv'];argv[argv.index('--cpus')+1]=','.join(map(str,cpus))
        argv.extend(['--topology-proof',str(path),'--topology-proof-sha256',proof_sha])
    bound.update(topology_pending=False,topology_proof=dict(path=str(path),sha256=proof_sha),selected_cpus=list(cpus))
    return bound


def validate_result(row,abi,arm,kind):
    require(row.get('passed') is True and row.get('abi')==abi and row.get('arm')==arm and row.get('kind')==kind,
            'runner result failed or cohort identity differs')
    samples={'gate':1,'perf':2000,'stability':20000}[kind]
    timing=row.get('timing',{})
    require(timing.get('samples')==samples and len(timing.get('durations_ns',[]))==samples,
            'wrong sample count')
    require(row.get('forward_calls')=={'gate':2,'perf':2021,'stability':20021}[kind],'runner performed extra/missing forwards')
    require(all(type(v) is int and v>0 for v in timing['durations_ns']),'invalid synchronized sample')
    if arm=='worker4':
        require(row.get('actual_workers')==4 and row.get('optimization_active') is True,
                'worker4 fell back or did not execute on four actual workers')
        require(row.get('mechanism',{}).get('passed') is True,'real native lifecycle proof missing')
        if kind in ('perf','stability'): require(row['mechanism'].get('prepare_strategy')=='worker','final P1 perf/stability requires worker SDK preparation')
        if kind=='gate':
            require(row.get('fifo_proof',{}).get('passed') is True,'original FIFO proof missing')
            require(type(row.get('worker_probe',{}).get('cpu_parallelism_proven')) is bool,'CPU observation flag missing')
    else:
        require(row.get('actual_workers')==0 and row.get('optimization_active') is False,'serial baseline was changed')
    require(row.get('performance_conclusion_allowed') is (kind=='perf'),'instrumented gate is not performance data')
    if kind=='stability':
        require(row.get('stability',{}).get('passed') is True,'independent stability exposure did not pass')
        load_contracts().validate_memory_trend(row['stability'].get('checkpoints'))
    return row


def verify_dependency_adapter(binding,baseline):
    adapter=binding.get('dependency_adapter',{'kind':'none'})
    if adapter['kind']=='none': return dict(kind='none')
    path=Path(adapter['path']).resolve(strict=True); target=Path(adapter['target']).resolve(strict=True)
    prefix=Path(binding['python']).parent.parent.resolve(strict=True)
    old_prefix=Path(baseline['python']).parent.parent.resolve(strict=True)
    require(prefix!=old_prefix and path.is_relative_to(prefix) and path.suffix=='.pth',
            'dependency .pth must be inside the new candidate runtime')
    require(target.is_dir() and target.name=='site-packages' and target.is_relative_to(old_prefix),
            '.pth target must be the same-ABI original dependency site')
    require(sha(path)==adapter['sha256'],'dependency .pth source SHA mismatch')
    require(path.read_text(encoding='utf-8').splitlines()==[str(target)],
            '.pth must contain exactly its one absolute target, with no import code')
    return dict(kind='pth',path=str(path),sha256=sha(path),target=str(target),
                source_prefix=str(old_prefix),candidate_prefix=str(prefix),executes_pth_code=False)


PROFILE_PROBE=r'''
import importlib.metadata as m,json,re,sys,sysconfig
norm=lambda name:re.sub(r'[-_.]+','-',name).lower()
rows=[]
for d in m.distributions():
 name=d.metadata.get('Name')
 if name: rows.append(dict(name=norm(name),version=d.version,metadata_path=str(getattr(d,'_path',''))))
versions={}
for row in rows: versions.setdefault(row['name'],set()).add(row['version'])
selected={}
for name in ('torch','torch-npu','numpy'):
 d=m.distribution(name);selected[name]=dict(version=d.version,metadata_path=str(getattr(d,'_path','')))
print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,executable=sys.executable,
 python_version='.'.join(map(str,sys.version_info[:3])),soabi=sysconfig.get_config_var('SOABI'),
 gil_enabled=sys._is_gil_enabled(),versions={k:sorted(v) for k,v in versions.items()},
 selected=selected,all_distributions=rows)))
'''


def runtime_profile(binding,baseline,env,output):
    adapter=verify_dependency_adapter(binding,baseline)
    probe=subprocess.run([binding['python'],'-I','-B','-c',PROFILE_PROBE],env=env,
                         capture_output=True,text=True,timeout=120,check=False)
    require(probe.returncode==0,'runtime metadata probe failed: '+probe.stderr[-2000:])
    data=json.loads(probe.stdout)
    prefix=Path(binding['python']).parent.parent.resolve(strict=True)
    require(Path(data['prefix']).resolve()==prefix and data['python_version']=='3.14.6',
            'selected runtime prefix/version differs from manifest')
    expected='cpython-314t-aarch64-linux-gnu' if binding['abi']=='ft' else 'cpython-314-aarch64-linux-gnu'
    require(data['soabi']==expected and data['gil_enabled'] is (binding['abi']=='std'),'runtime ABI/natural GIL mismatch')
    require(all(len(v)==1 for v in data['versions'].values()),'multiple installed versions of one distribution')
    require(data['selected']['torch']['version']=='2.13.0+cpu'
            and data['selected']['numpy']['version']=='2.5.3'
            and data['selected']['torch-npu']['version']=='2.13.0','runtime dependency versions changed')
    require(Path(data['selected']['torch-npu']['metadata_path']).resolve().is_relative_to(prefix),
            'candidate resolved original torch_npu metadata instead of its own package')
    dependency_root=Path(adapter['target']) if adapter['kind']=='pth' else prefix
    for name in ('torch','numpy'):
        require(Path(data['selected'][name]['metadata_path']).resolve().is_relative_to(dependency_root),
                'dependency selected outside its explicitly declared source: '+name)
    check=subprocess.run([binding['python'],'-I','-B','-m','pip','check'],env=env,
                         capture_output=True,text=True,timeout=120,check=False)
    require(check.returncode==0,'runtime pip check failed: '+check.stdout[-2000:]+check.stderr[-2000:])
    data.update(dependency_adapter=adapter,pip_check=check.stdout.strip())
    write_new(output/('PROFILE-'+binding['abi']+'-'+binding['arm']+'.json'),data)
    return data


def sdk_environment(args,output):
    script=(args.cann_env or args.cann_prefix/'cann/set_env.sh').resolve(strict=True)
    base={key:os.environ[key] for key in ('HOME','USER','LOGNAME','TMPDIR') if key in os.environ}
    base.update(PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    result=subprocess.run(['/bin/bash','-c','set -ea; source "$1" >&2; env -0','r22-sdk',str(script)],
                          env=base,capture_output=True,check=False)
    (output/'SDK_SOURCE.log').write_bytes(result.stderr)
    require(result.returncode==0,'CANN environment script failed')
    env=dict(item.decode().split('=',1) for item in result.stdout.split(b'\0') if b'=' in item)
    for key in list(env):
        if key.startswith(('HDB_','FTLAB_','QWEN_')) or key in ('PYTHONPATH','PYTHONHOME','PYTHON_GIL',
                'LD_PRELOAD','LD_AUDIT','PYTHONSTARTUP','PYTHONINSPECT','BASH_ENV','ENV'):
            env.pop(key,None)
    # Match the frozen R21 serial environment before any numeric-library import.
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TASK_QUEUE_ENABLE='1',PER_STREAM_QUEUE='0',ASCEND_LAUNCH_BLOCKING='0',
        TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        FTLAB_HOST_REGIONS='0',FTLAB_HOST_NATIVE_RUN='0',HDB_REGION_PACKET='0',HDB_REGION_REAL_OPAPI='0',
        HDB_REGION_NATIVE_BUILDER='0',FTLAB_HOST_OWNER_RETIREMENT='0',HDB_NATIVE_PREPARE_STUB='0')
    return env,dict(script=str(script),sha256=sha(script),
        selected={key:env.get(key) for key in ('ASCEND_HOME_PATH','ASCEND_PATH','LD_LIBRARY_PATH','PATH',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def run_child(argv,env,stream,timeout=1800):
    process=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try: return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=10)


def execute(args,plan):
    require(not args.output.exists(),'new owned output directory required')
    require(sha(args.runner)==args.runner_sha256,'runner source SHA mismatch')
    require(sha(args.arms_manifest)==plan['arms_manifest']['sha256'],'arm manifest changed after planning')
    bindings=load_arm_manifest(args.arms_manifest)
    require(not args.output.resolve().is_relative_to(args.upstream.resolve())
            and not args.output.resolve().is_relative_to(args.cann_prefix.resolve()),'output cannot modify model/SDK trees')
    for (abi,arm),binding in bindings.items():
        require(sha(binding['wheel'])==binding['wheel_sha256'],'wheel SHA mismatch: '+abi+'-'+arm)
    args.output.mkdir(parents=True)
    result=dict(passed=False,plan=plan,controller_sha256=sha(__file__),runner_sha256=sha(args.runner),results=[])
    try:
        contracts=load_contracts()
        allowed=os.sched_getaffinity(0)
        eligible=sorted(allowed) if args.cpus=='auto' else [int(v) for v in args.cpus.split(',')]
        require(set(eligible)<=allowed,'declared topology CPU pool escapes current allowed affinity')
        topology=contracts.capture_topology(eligible)
        proof_path=args.output/'TOPOLOGY.json';write_new(proof_path,topology)
        plan=bind_topology(plan,topology,proof_path,sha(proof_path));result['plan']=plan
        env,sdk=sdk_environment(args,args.output); result['sdk_environment']=sdk
        receipts={}
        for (abi,arm),binding in bindings.items():
            path=Path(binding['build_receipt'])
            require(sha(path)==binding['build_receipt_sha256'],'build receipt source SHA mismatch')
            raw=json.loads(path.read_text()); wheel=raw.get('wheel',{})
            require(raw.get('status')=='built' and raw.get('abi')==abi and raw.get('canonical_source_unchanged') is True
                and wheel.get('sha256')==binding['wheel_sha256']
                and wheel.get('native',{}).get('torch_npu/lib/libtorch_npu.so',{}).get('sha256')==binding['native_sha256'],
                'build receipt does not match the physical artifact: '+abi+'-'+arm)
            receipts[abi+'-'+arm]=dict(path=str(path.resolve()),sha256=sha(path))
        result['build_receipts']=receipts
        profiles={}
        for (abi,arm),binding in bindings.items():
            print(json.dumps(dict(stage='runtime-profile',abi=abi,arm=arm)),flush=True)
            child_env={**env,'TORCH_NPU_EXPERIMENTAL_FREE_THREADED':'1' if abi=='ft' else '0',
                'HDB_REGION_DIAGNOSTICS':'0','HDB_REGION_TRACE':'0'}
            profiles[abi+'-'+arm]=runtime_profile(binding,bindings[abi,'serial'],child_env,args.output)
        reference=profiles['std-serial']['versions']
        require(all(profile['versions']==reference for profile in profiles.values()),
                'all-installed distribution version parity differs across the four arms')
        result['runtime_profiles']=profiles
        write_new(args.output/'PLAN.json',plan)
        for index,item in enumerate(plan['gates']+plan['repeat']+plan['stability']):
            child_env={**env,'TORCH_NPU_EXPERIMENTAL_FREE_THREADED':'1' if item['abi']=='ft' else '0',
                'HDB_REGION_DIAGNOSTICS':'1' if item['kind']=='gate' else '0',
                'HDB_REGION_TRACE':'1' if item['kind']=='gate' else '0'}
            print(json.dumps(dict(stage=index,abi=item['abi'],arm=item['arm'],kind=item['kind'],round=item['round'])),flush=True)
            log=args.output/(Path(item['output']).stem+'.log')
            with log.open('xb') as stream:
                code=run_child(item['argv'],child_env,stream,timeout=item['timeout_seconds'])
            raw=json.loads(Path(item['output']).read_text()) if Path(item['output']).is_file() else {}
            result['results'].append(dict(**{key:item[key] for key in ('abi','arm','kind','round','output')},
                exit_code=code,raw_sha256=sha(item['output']) if raw else None))
            require(code==0,'runner failed; preserve its raw/log: '+item['output'])
            validate_result(raw,item['abi'],item['arm'],item['kind'])
            require(raw.get('script_sha256')==args.runner_sha256
                    and raw.get('requested_wheel',{}).get('sha256')==item['binding']['wheel_sha256']
                    and raw.get('native_sha256')==item['binding']['native_sha256'],'runner used another physical artifact')
            require(raw.get('topology',{}).get('proof_sha256')==plan['topology_proof']['sha256']
                    and raw.get('cpus')==plan['selected_cpus'],'runner did not use the common physical-core topology receipt')
        rows=[(item,json.loads(Path(item['output']).read_text())) for item in plan['repeat']]
        summary={}
        for abi in ('std','ft'):
            for arm in ('serial','worker4'):
                cohort=[raw for item,raw in rows if item['abi']==abi and item['arm']==arm]
                throughput=[2000*1e9/raw['timing']['common_wall_ns'] for raw in cohort]
                p99=[sorted(raw['timing']['durations_ns'])[1979]/1e6 for raw in cohort]
                summary[abi+'-'+arm]=dict(batch_per_second=throughput,median_batch_per_second=statistics.median(throughput),
                    p99_ms=p99,median_p99_ms=statistics.median(p99),
                    maximum_ms=max(max(raw['timing']['durations_ns'])/1e6 for raw in cohort),
                    whole_process_peak_rss_kib=[raw['memory_after_run']['whole_process_peak_rss_kib'] for raw in cohort],
                    effective_cpu_parallelism=[raw['timing']['process_cpu_ns']/raw['timing']['common_wall_ns'] for raw in cohort])
        long_rows={item['abi']+'-'+item['arm']:json.loads(Path(item['output']).read_text())['stability'] for item in plan['stability']}
        result.update(passed=True,summary=summary,stability=long_rows,
            scope='four cold gates, twenty short perf and four separate stability processes; stability is excluded from perf summary; no automatic speedup claim')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally: write_new(args.output/'RESULT.json',result)
    return result


def main(argv=None):
    args=parse_args(argv);plan=make_plan(args)
    if not args.execute: print(json.dumps(plan,indent=2));return 0
    def stop(_signum,_frame): raise KeyboardInterrupt('controller received SIGTERM')
    previous=signal.signal(signal.SIGTERM,stop)
    try: result=execute(args,plan)
    finally: signal.signal(signal.SIGTERM,previous)
    print(json.dumps(dict(passed=result['passed'],output=str(args.output),error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
