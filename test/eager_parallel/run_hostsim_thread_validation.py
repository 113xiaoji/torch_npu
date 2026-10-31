"""Isolated Release/ASan/TSan validation of the actual host-sim worker runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime-source',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    source=root/'worker_source'
    source.mkdir(exist_ok=True)
    logs=root/'logs'
    binaries=root/'bin'
    before={name:digest(args.runtime_source/name) for name in ('runtime.cpp','runtime.h')}
    for name in before: shutil.copy2(args.runtime_source/name,source/name)
    report={'schema':1,'runtime_snapshot':str(args.runtime_source),'runtime_before':before,
            'test_affinity':[184,186],'scope':'actual host-sim std::thread workers and API synchronization; not CANN qualification',
            'test_sha256':digest(root/'hostsim_runtime_thread_test.cpp'),'commands':[],'results':[]}
    runtime_dir='/usr/lib64/clang/17/lib/aarch64-openEuler-linux-gnu'
    modes={
        'release':['g++','-O2'],
        'asan':['clang++','-O1','-g','-fsanitize=address','-fno-omit-frame-pointer','-shared-libsan','-Wl,-rpath,'+runtime_dir],
        'tsan':['clang++','-O1','-g','-fsanitize=thread','-fno-omit-frame-pointer','-shared-libsan','-Wl,-rpath,'+runtime_dir],
    }
    environment=os.environ.copy()
    environment['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1'
    environment['TSAN_OPTIONS']='halt_on_error=1'
    def run(name,command,timeout=90):
        command=['taskset','-c','184,186']+[str(value) for value in command]
        report['commands'].append({'name':name,'argv':command})
        started=time.time()
        try:
            result=subprocess.run(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
            output,code=result.stdout,result.returncode
        except subprocess.TimeoutExpired as error:
            output,code=(error.stdout or b'')+b'\nTEST WATCHDOG EXPIRED\n',124
        log=logs/(name+'.log');log.write_bytes(output)
        row={'name':name,'returncode':code,'seconds':time.time()-started,'log':str(log),'sha256':digest(log)}
        report['results'].append(row);print(json.dumps(row),flush=True)
        return code
    for mode,compiler in modes.items():
        binary=binaries/(mode+'_hostsim_workers')
        command=compiler+['-std=c++20','-pthread','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',
            '-I'+str(source),str(source/'runtime.cpp'),str(root/'hostsim_runtime_thread_test.cpp'),'-o',str(binary)]
        if run(mode+'_hostsim_workers_compile',command)==0:
            run(mode+'_hostsim_workers_run',[binary],timeout=40)
            run(mode+'_hostsim_workers_ldd',['ldd',binary])
    report['runtime_after']={name:digest(args.runtime_source/name) for name in before}
    report['runtime_unchanged']=before==report['runtime_after']
    report['passed']=report['runtime_unchanged'] and all(row['returncode']==0 for row in report['results'])
    path=root/'worker-results.json';path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':report['passed'],'results':str(path)}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
