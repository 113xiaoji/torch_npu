"""Owned fresh-process controller for native numerical and fault scenarios."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def group_exists(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--sdk',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('normal','faults','allocator'),required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(exist_ok=False)
    package=args.package.resolve(strict=True);sdk=args.sdk.resolve(strict=True)
    receipt=package.parents[1]/'build-evidence/final-result.json'
    build=json.loads(receipt.read_text())
    for name,value in build['outputs'].items():
        if sha(Path(name))!=value['sha256']:raise RuntimeError('final build pin mismatch '+name)
    here=Path(__file__).resolve().parent
    source_pins={str(path):sha(path) for path in [Path(__file__),here/'hostsim_immediate_scenarios.py',here/'hostsim_immediate_faults.py']}
    plan=dict(stage=args.stage,build_receipt=str(receipt),build_receipt_sha256=sha(receipt),build_outputs=build['outputs'],
              source_pins=source_pins,package=str(package),sdk=str(sdk),cpus=dict(e=[160,162],ei=[184,186]),
              environment_contract=dict(TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1',TASK_QUEUE_ENABLE='0',QWEN_ACL_HOST_SIM='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PER_STREAM_QUEUE='0',ASCEND_LAUNCH_BLOCKING='0',
                PYTHON_GIL=None,TORCH_DEVICE_BACKEND_AUTOLOAD='0'))
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    cases=['all'] if args.stage=='normal' else ['allocator'] if args.stage=='allocator' else [
        'first_record_before','first_record_after','post_body_record_before','post_body_record_after',
        'first_wait_before','first_wait_after','query_quarantine','prior_output_cleanup']
    def run(arm,case):
        name=arm+'-'+case;report=root/(name+'.json');log=root/(name+'.log')
        worker=here/('hostsim_immediate_faults.py' if args.stage=='faults' else 'hostsim_immediate_scenarios.py')
        env={key:value for key,value in os.environ.items() if not key.startswith(('PYTHON','LD_','TORCH','ASCEND','QWEN','TASK_QUEUE','PER_STREAM'))}
        env.update(PYTHONPATH=str(package.parent),PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',
            TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1',
            QWEN_ACL_HOST_SIM='1',TASK_QUEUE_ENABLE='0',PER_STREAM_QUEUE='0',ASCEND_LAUNCH_BLOCKING='0',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(sdk)+':/root/ft-pytorch-repro/envs/py314t/lib/python3.14t/site-packages/torch/lib')
        command=['taskset','-c',','.join(map(str,plan['cpus'][arm])),
            '/root/ft-pytorch-repro/envs/py314t/bin/python',str(worker),'--arm',arm,'--case',case,
            '--sdk',str(sdk),'--package',str(package),'--output',str(report)]
        start=time.monotonic();timed_out=False
        with log.open('x') as stream:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=150)
            except subprocess.TimeoutExpired:timed_out=True
            finally:
                if group_exists(process.pid):
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait(timeout=3)
        result=json.loads(report.read_text()) if report.exists() else {}
        row=dict(name=name,pid=process.pid,pgid=process.pid,returncode=process.returncode,
                 timed_out=timed_out,group_closed=not group_exists(process.pid),elapsed_seconds=time.monotonic()-start,
                 passed=result.get('passed') is True,report=str(report),report_sha256=sha(report) if report.exists() else None,
                 command=command,log=str(log),log_sha256=sha(log))
        row['environment']={key:env.get(key) for key in [*plan['environment_contract'],'PYTHONPATH','LD_LIBRARY_PATH']}
        (root/(name+'-lifecycle.json')).write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps({key:row[key] for key in ('name','returncode','passed','group_closed','elapsed_seconds')}),flush=True)
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for case in cases:
            futures=[pool.submit(run,arm,case) for arm in ('e','ei')]
            rows.extend(future.result() for future in futures)
    passed=all(row['passed'] and row['returncode']==0 and row['group_closed'] and not row['timed_out'] for row in rows)
    final=dict(passed=passed,all_owned_groups_closed=all(row['group_closed'] for row in rows),runs=rows,
               source_pins_unchanged=all(sha(Path(path))==value for path,value in source_pins.items()))
    (root/'completed.json').write_text(json.dumps(final,indent=2)+'\n')
    return 0 if passed and final['source_pins_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
