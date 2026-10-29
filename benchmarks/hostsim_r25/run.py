"""Plan or execute a small matched R25 four-arm DLRM host-sim screen."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import contracts as c


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def fresh_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def group_exists(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False


def run_owned(argv,env,log,timeout):
    """Own the new session through normal exit, error, interruption or timeout."""
    c.require(os.name=='posix','native Linux process groups required')
    start=time.monotonic();process=None;timed_out=False;first=None
    with Path(log).open('wb') as stream:
        try:
            process=subprocess.Popen(argv,env=env,cwd=Path(log).parent,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:timed_out=True
        except BaseException as error:first=error
        finally:
            if process is not None and group_exists(process.pid):
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
                if group_exists(process.pid):
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired as error:
                    if first is None:first=error
    if first is not None:raise first
    return dict(pid=process.pid,pgid=process.pid,returncode=process.returncode,timed_out=timed_out,
                group_closed=not group_exists(process.pid),elapsed_seconds=time.monotonic()-start)


def validate_row(result,row,worker_sha,contracts_sha,gate=None):
    c.require(result.get('passed') is True,'worker rejected run: '+str(result.get('error')))
    for key in ('runtime','arm','kind','round','workers','warmup','samples'):
        c.require(result.get(key)==row[key],'worker cohort identity differs '+key)
    c.require(result.get('worker_sha256')==worker_sha and result.get('contracts_sha256')==contracts_sha,'worker contract source differs')
    c.require(result.get('forward_calls')==row['warmup']+row['samples'],'missing/extra original DLRM forward')
    c.require(c.distribution(result['durations_ns'])['samples']==row['samples'],'sample count differs')
    c.require(result.get('performance_eligible') is (row['kind']=='perf'),'diagnostic result cannot enter performance table')
    c.require(result.get('mechanism',{}).get('passed') is True,'dedicated SIM mechanism gate missing')
    if row['kind']=='cold':
        c.require(result.get('cold_gate',{}).get('passed') is True,'cold SIM gate failed')
    else:
        c.require(gate is not None and gate.get('cold_gate',{}).get('passed') is True,'matching cold gate required before performance')
        for key in ('cpu_identity','artifact_pins','mapping'):
            c.require(result['ready'][key]==gate['ready'][key],'performance identity differs from cold gate '+key)
        c.require(result['mechanism']['embedding_commands_per_forward']==gate['mechanism']['embedding_commands_per_forward'],
                  'performance native coverage differs from cold gate')
    return result


def load_helper(pin,name):
    c.verify_pin(pin,name)
    spec=importlib.util.spec_from_file_location(name,pin['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu_ticks():
    text=Path('/proc/stat').read_text();rows={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            values=list(map(int,fields[1:]));rows[int(fields[0][3:])]=(sum(values[:8]),values[3]+values[4])
    return rows


def preflight(config):
    c.require(sys.platform=='linux','execution requires Linux; plan-only works on other hosts')
    c.require(set(config['cpus'])<=os.sched_getaffinity(0),'requested CPUs outside controller affinity')
    topology=load_helper(config['helpers']['topology'],'_r25_pinned_topology').capture_topology(config['cpus'])
    c.require(topology['selected_cpus']==sorted(config['cpus']) and topology['choice']['numa_node']==config['numa_node']
        and topology['choice']['llc_count']==1,'eight physical CPUs must occupy the declared one NUMA/one LLC domain')
    for value in config['helpers'].values():c.verify_pin(value,'helper')
    for value in config['sdk']['libraries'].values():c.verify_pin(value,'new SDK artifact')
    for path,digest in config['sdk']['source_pins'].items():c.verify_pin(dict(path=path,sha256=digest),'SDK source')
    for rt in config['runtimes']:
        c.require(Path(rt['python']).is_file(),'missing lexical interpreter')
        c.verify_pin(rt['model_manifest'],'DLRM asset manifest')
        for path,digest in rt['artifact_pins'].items():c.verify_pin(dict(path=path,sha256=digest),'new package artifact')
    # Cold availability observation only. It is not injected into any model ROI.
    started=time.monotonic_ns();before=cpu_ticks();time.sleep(.25);after=cpu_ticks()
    observed=[]
    for cpu in config['cpus']:
        total=after[cpu][0]-before[cpu][0];idle=after[cpu][1]-before[cpu][1]
        observed.append(dict(cpu=cpu,total_ticks=total,idle_ticks=idle,busy_fraction=None if not total else 1-idle/total))
    return dict(topology=topology,background=dict(loadavg=Path('/proc/loadavg').read_text().strip(),
        observation_ns=time.monotonic_ns()-started,per_cpu=observed,exclusive_cpu_ownership_proven=False),
        natural_gil_required=True,PYTHON_GIL_override=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    config=c.validate_config(json.loads(args.config.read_text()))
    output=fresh_output(args.output.absolute());dump(output/'config.json',config)
    plan=c.make_plan(config);dump(output/'plan.json',plan)
    sources={name:c.sha(Path(__file__).with_name(name)) for name in ('run.py','worker.py','contracts.py')}
    dump(output/'runner-sources.json',sources)
    if not args.execute:
        print(json.dumps(dict(plan_only=True,output=str(output),cold_processes=len(plan['cold']),performance_processes=len(plan['perf']))))
        return
    rows=[];gates={}
    try:
        dump(output/'preflight.json',preflight(config))
        numactl=shutil.which('numactl');c.require(numactl,'numactl is required for explicit memory binding')
        for index,row in enumerate(plan['cold']+plan['perf']):
            rt=next(r for r in config['runtimes'] if r['name']==row['runtime'])
            label=f"{index:02d}-{row['runtime']}-{row['kind']}-r{row['round']}-{row['arm']}"
            row_path=output/(label+'-input.json');dump(row_path,row)
            result_dir=output/label
            command=[numactl,'--physcpubind='+','.join(map(str,config['cpus'])),'--membind='+str(config['numa_node']),
                rt['python'],'-B',str(Path(__file__).with_name('worker.py').absolute()),
                '--config',str(output/'config.json'),'--row',str(row_path),'--output',str(result_dir)]
            dump(output/(label+'-command.json'),dict(argv=command,source_pins=sources))
            lifecycle=run_owned(command,c.worker_environment(config,rt,row),output/(label+'-worker.log'),row['timeout_seconds'])
            dump(output/(label+'-lifecycle.json'),lifecycle)
            c.require(not lifecycle['timed_out'] and lifecycle['returncode']==0 and lifecycle['group_closed'],
                'owned worker failed/timed out or retained a process group: '+label)
            result=json.loads((result_dir/'result.json').read_text())
            validate_row(result,row,sources['worker.py'],sources['contracts.py'],gates.get((row['runtime'],row['arm'])))
            if row['kind']=='cold':gates[row['runtime'],row['arm']]=result
            else:rows.append(result)
            dump(output/'completed-rows.json',rows)
            print(json.dumps(dict(completed=label,median_ns=result['timing']['median_ns'],kind=row['kind'])),flush=True)
        summary=c.summarize(rows);dump(output/'summary.json',summary)
        lines=['# R25 actual DLRM host-sim first screen','',
            'Three matched rounds, 20 warmup + 200 samples each. Pool startup excluded. '
            'Original DLRM forward through synchronize; simulated kernels, no numerical or real-device claim.','',
            '| Runtime | Arm | Median paired speedup | Round range | Direction |',
            '|---|---|---:|---:|---|']
        for rt,arms in summary.items():
            for arm,value in arms.items():
                lo,hi=value['speedup_range']
                lines.append(f"| {rt} | {arm} | {value['median_paired_speedup']:.4f}x | {lo:.4f}–{hi:.4f}x | {value['classification']} |")
        (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except BaseException as error:
        dump(output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),completed_perf_rows=len(rows)))
        raise


if __name__=='__main__':main()
