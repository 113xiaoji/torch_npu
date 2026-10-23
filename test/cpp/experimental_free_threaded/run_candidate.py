"""Same real-core tests as the recorded RED, now against candidate header."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,required=True)
args=parser.parse_args()
here=Path(__file__).resolve().parent
output=here/'candidate-results'/str(time.time_ns());output.mkdir(parents=True,exist_ok=False)
records=[]
def run(label,command,timeout=120):
    tick=time.time_ns()
    result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    row={'label':label,'command':command,'returncode':result.returncode,'stdout':result.stdout,
         'stderr':result.stderr,'started_unix_ns':tick,'ended_unix_ns':time.time_ns()}
    (output/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    records.append(row)
    if result.returncode:raise RuntimeError('native regression failed: '+label+' '+result.stderr[:1000])
    return row
flags=['g++','-std=c++17','-pthread','-Wall','-Wextra','-Wpedantic','-Werror','-O2','-g','-I'+str(args.source_root)]
for name,source in [('correctness','ordered_mpsc_queue_test.cpp'),('contracts','queue_contract_test.cpp')]:
    run('build-'+name,flags+[str(here/source),'-o',str(output/name)])
for rep in range(10):
    run('correctness-'+str(rep),[str(output/'correctness')])
    for mode in ['wake-contract','metadata-lifecycle','discard-lifecycle']:
        row=run(mode+'-'+str(rep),[str(output/'contracts'),mode])
        stats=json.loads(row['stdout'])
        if mode=='wake-contract':assert stats['contract_passed'] and not stats['early_drain_returns']
        else:assert stats['passed'] and not stats['rescue_request_stop']
header=args.source_root/'torch_npu/csrc/core/npu/OrderedMpscQueueCore.h'
summary={'status':'passed','source_root':str(args.source_root),
    'header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),
    'tests_per_repeat':['existing_correctness','running_nonempty_wake','metadata_idle_lifecycle','discard_idle_lifecycle'],
    'repeats':10,'records':records,'scope':'native real-core regression; not DLRM performance'}
(output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(here/'CANDIDATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':'passed','header_sha256':summary['header_sha256'],'repeats':10,'output':str(output)}))
