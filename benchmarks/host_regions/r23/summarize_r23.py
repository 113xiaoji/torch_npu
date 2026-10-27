"""Read-only compact replay/summary of the frozen main42 and optional caller8."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PINS = {
    'analyze_profile_r23.py': 'ac25791b995880ef114c530db6e099ec6eda00a9d29a558f0a587a4c6a51e0f8',
    'run_profile_campaign_r23.py': 'da4c32bb49c92d0c275db75777934bfd0b11a8576c2b48fcca5d949d1db8d84d',
    'run_caller_campaign_r23.py': 'b81f40da289c850052a56751d1180ad9d6753b80c3dd9971cfaa1b6243de8067',
    'instrument_caller_only_r23.py': 'f0a78ebc2296e71faab20cfd1cf5e03d4be14e624024163954bdd01caf7b32d7',
    'instrument_frozen_runner_r23.py': 'd7461dbe7f1d6f0481e865381d9a07aa9bb64bf9ceb02a335a6776e2b0b39865',
}
LABELS = ('original-serial', 'candidate-serial', 'worker4')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slim(distribution):
    return {key: distribution[key] for key in ('count', 'mean_ns', 'p50_ns', 'p99_ns')}


def aggregate(rows, analyzer):
    require(len(rows) == 18, 'exact eighteen stage pairs required')
    pooled = []; comparisons = []; pooled_comparisons = []; neighbor = []
    for abi in ('std', 'ft'):
        for label in LABELS:
            selected = sorted((r for r in rows if r['abi'] == abi and r['label'] == label), key=lambda r:r['round'])
            require([r['round'] for r in selected] == [1, 2, 3], 'missing/duplicate three-round label')
            require(all(len(r['_off_ns']) == len(r['_on_ns']) == 2000 for r in selected), 'full2000 pair required')
            off = [v for r in selected for v in r['_off_ns']]; on = [v for r in selected for v in r['_on_ns']]
            stage_values = {name:[v for r in selected for v in r['_stage_ns'][name]] for name in selected[0]['_stage_ns']}
            require(len(stage_values) == 7 and all(len(values) == 6000 for values in stage_values.values()), 'seven6000 stages required')
            require(sum(sum(values) for values in stage_values.values()) == sum(on), 'pooled stage totals do not conserve raw on durations')
            item = dict(abi=abi, label=label, off=slim(analyzer.distribution(off)), on=slim(analyzer.distribution(on)),
                stages={name:slim(analyzer.distribution(values)) for name,values in stage_values.items()}, stage_total_conserved=True)
            pooled.append(item)
            for r in selected:
                a, b = sum(r['_off_ns'])/2000, sum(r['_on_ns'])/2000
                neighbor.append(dict(abi=abi,label=label,round=r['round'],off_mean_ns=a,on_mean_ns=b,
                    on_over_off=b/a,on_minus_off_mean_ns=b-a,order=r.get('order'),stage_values_corrected=False))
        for round_index in (1, 2, 3):
            serial = next(r for r in rows if (r['abi'],r['label'],r['round']) == (abi,'candidate-serial',round_index))
            worker = next(r for r in rows if (r['abi'],r['label'],r['round']) == (abi,'worker4',round_index))
            a,b = sum(serial['_off_ns'])/2000,sum(worker['_off_ns'])/2000
            comparisons.append(dict(abi=abi,round=round_index,candidate_serial_off_mean_ns=a,worker4_off_mean_ns=b,
                                    worker4_over_candidate_serial_mean=b/a))
        serial = next(r for r in pooled if r['abi']==abi and r['label']=='candidate-serial')
        worker = next(r for r in pooled if r['abi']==abi and r['label']=='worker4')
        pooled_comparisons.append(dict(abi=abi,candidate_serial_off_mean_ns=serial['off']['mean_ns'],worker4_off_mean_ns=worker['off']['mean_ns'],
            worker4_over_candidate_serial_mean=worker['off']['mean_ns']/serial['off']['mean_ns']))
    return dict(pooled=pooled,neighbor_off_on=neighbor,same_candidate_rounds=comparisons,same_candidate_pooled=pooled_comparisons,
                rounds=[{k:v for k,v in row.items() if not k.startswith('_')} for row in rows])


class Reader:
    def __init__(self): self.inputs = {}; self.sources = {}
    def load(self, name, path, expected):
        require(sha(path)==expected, 'source SHA mismatch: '+str(path)); self.sources[str(path)]=expected
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module; spec.loader.exec_module(module); return module
    def read(self, path, expected=None):
        path=Path(path); data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        require(expected is None or digest==expected, 'input SHA mismatch: '+str(path))
        self.inputs[str(path.resolve())]=dict(sha256=digest,bytes=len(data)); return json.loads(data),digest
    def artifact(self, declared, root, expected):
        path=Path(declared)
        for candidate in (root/path.name,root/path.parent.name/path.name,path):
            if candidate.is_file() and sha(candidate)==expected: return self.read(candidate,expected)[0]
        raise ValueError('missing or wrong-SHA artifact: '+declared)


def replay(args, reader):
    analyzer=reader.load('_summary_analyzer',HERE/'analyze_profile_r23.py',PINS['analyze_profile_r23.py'])
    campaign=reader.load('_summary_main_campaign',HERE/'run_profile_campaign_r23.py',PINS['run_profile_campaign_r23.py'])
    driver=reader.load('_summary_caller_driver',HERE/'run_caller_campaign_r23.py',PINS['run_caller_campaign_r23.py'])
    for name,expected in campaign.PINS.items():
        require(sha(args.tools/name)==expected,'frozen tools source differs: '+name);reader.sources[str(args.tools/name)]=expected
    helper=reader.load('_summary_frozen_helper',args.tools/'run_real_measurement_r22.py',campaign.PINS['run_real_measurement_r22.py'])
    require(sha(HERE/'instrument_frozen_runner_r23.py')==PINS['instrument_frozen_runner_r23.py'],'base wrapper SHA differs')
    reader.sources[str(HERE/'instrument_frozen_runner_r23.py')]=PINS['instrument_frozen_runner_r23.py']
    cohort,cohort_sha=reader.read(args.cohort_result);driver.basic_cohort(cohort)
    items=cohort['plan']['gates']+cohort['plan']['repeat'];records=cohort['results']; raw={};profiles={}
    for item,record in zip(items,records):
        require(record.get('exit_code')==0 and record.get('output')==item['output'] and
            all(record.get(k)==item[k] for k in ('abi','label','kind','round','profile_mode')),'main42 record/plan identity differs')
        value=reader.artifact(item['output'],args.cohort_result.parent,record['raw_sha256'])
        campaign.validate_raw(helper,value,item,cohort['plan']);raw[item['output']]=value
        if item.get('profile_output'):
            profiles[item['output']]=(item['profile_output'],record['profile_sha256'])
    require(len(raw)==42 and len(profiles)==18,'main42 raw/eighteen sidecar replay incomplete')
    rows=[];repeat=cohort['plan']['repeat']
    for position,item in enumerate(repeat):
        if item['profile_mode']!='stages':continue
        matches=[(i,r) for i,r in enumerate(repeat) if (r['abi'],r['label'],r['round'],r['profile_mode'])==
                 (item['abi'],item['label'],item['round'],'off')]
        require(len(matches)==1 and abs(matches[0][0]-position)==1,'off/stages pair is not adjacent')
        off_index,off_item=matches[0];off=raw[off_item['output']]['timing']['durations_ns']
        declared,digest=profiles[item['output']]
        profile=reader.artifact(declared,args.cohort_result.parent,digest)
        campaign.validate_profile(profile,item,PINS['instrument_frozen_runner_r23.py'])
        analyzed=analyzer.analyze(raw[item['output']],profile);requests=analyzed['requests']
        rows.append(dict(abi=item['abi'],label=item['label'],round=item['round'],order=['off','stages'] if off_index<position else ['stages','off'],
            off=slim(analyzer.distribution(off)),on=slim(analyzed['raw_distribution']),
            stages={k:slim(v) for k,v in analyzed['stages'].items()},
            nested=analyzed['nested_summary'],worker_on=analyzed['worker_summary'] if item['label']=='worker4' else None,
            _off_ns=off,_on_ns=raw[item['output']]['timing']['durations_ns'],
            _stage_ns={name:[r['stages_ns'][name] for r in requests] for name in analyzed['stages']}))
    result=aggregate(rows,analyzer);result['main_result_sha256']=cohort_sha
    result['caller8']=None
    if args.caller_result:
        caller_result,_=reader.read(args.caller_result)
        require(caller_result.get('passed') is True and caller_result.get('completed_cases')==8 and
            len(caller_result.get('cases',[]))==len(caller_result.get('plan',{}).get('cases',[]))==8 and
            set((PINS['run_caller_campaign_r23.py'],PINS['instrument_caller_only_r23.py'],cohort_sha)) <=
            set(caller_result.get('source_pins',{}).values()),'caller8 source/main-result identity differs')
        caller=reader.load('_summary_caller_wrapper',HERE/'instrument_caller_only_r23.py',PINS['instrument_caller_only_r23.py'])
        cases=[]
        for item,record in zip(caller_result['plan']['cases'],caller_result['cases']):
            require(record.get('exit_code')==0 and all(record.get(k)==item[k] for k in ('abi','label','mode','output')),'caller8 record identity differs')
            value=reader.artifact(item['output'],args.caller_result.parent,record['raw_sha256'])
            validation_item=dict(item['cohort_item'],output=item['output']);campaign.validate_raw(helper,value,validation_item,cohort['plan'])
            row=dict(abi=item['abi'],label=item['label'],mode=item['mode'],raw_summary=analyzer.distribution(value['timing']['durations_ns']))
            require(row['raw_summary']==record['raw_summary'],'caller raw summary differs')
            if item['mode']=='caller_only':
                profile=reader.artifact(item['profile_output'],args.caller_result.parent,record['profile_sha256'])
                driver.validate_sidecar(profile,item,caller,analyzer,record['application_process']['pid'])
                row['caller_stages']=driver.caller_stages(value,profile,analyzer)
                require(row['caller_stages']==record['caller_stages'],'caller stages differ')
                groups=defaultdict(list)
                for event in profile['events']:groups[event['request']].append(event)
                totals=defaultdict(lambda:defaultdict(int))
                for events in groups.values():
                    for name,metrics in analyzer.nested_metrics(events).items():
                        for key,count in metrics.items():totals[name][key]+=count
                row['nested_per_request_mean']={name:{key:count/2000 for key,count in values.items()} for name,values in totals.items()}
                row['worker_observation']=None
            cases.append(row)
        pairs=driver.pair_summary(cases);require(pairs==caller_result['observer_pairs'],'caller observer pairs differ')
        result['caller8']=dict(cases=cases,observer_pairs=pairs,same_candidate_off=[dict(abi=abi,
            worker4_over_candidate_serial_mean=next(r for r in cases if (r['abi'],r['label'],r['mode'])==(abi,'worker4','off'))['raw_summary']['mean_ns']/
                next(r for r in cases if (r['abi'],r['label'],r['mode'])==(abi,'candidate-serial','off'))['raw_summary']['mean_ns']) for abi in ('std','ft')])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('cohort-result','tools','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--caller-result',type=Path);args=p.parse_args(argv)
    output=args.output.resolve();md=output.with_suffix('.md')
    require(not output.exists() and not md.exists() and output!=md,'new exclusive JSON/Markdown output required')
    for source in (args.cohort_result,args.caller_result):
        if source:require(not output.is_relative_to(source.resolve().parent),'summary output must be outside producer output trees')
    reader=Reader();report=dict(schema=1,passed=False,diagnostic_only=True,performance_conclusion_allowed=False,
        tool_sha256=sha(__file__),analysis_executes_device=False,
        quantiles='Nearest rank over actual concatenated6000 samples; per-round2000 P50/P99 remain separate; quantiles never add.',
        boundaries=['Seven stage pooled means conserve pooled on durations; stage quantiles do not add.',
            'Nested guard/policy/drain and worker wall/CPU overlap; do not add them to caller stages.',
            'Off/on is a neighboring fresh-process difference, not a subtraction/correction factor.',
            'Same-candidate comparison uses candidate-serial off versus worker4 off; individual duration means are not loop throughput.',
            'No new SDK acceptance, lock-wait proof or native query/execute/device attribution.'],
        native_measurements=dict(query_ns=None,execute_ns=None,queue_wait_ns=None,device_ns=None))
    try:report.update(replay(args,reader));report['passed']=True
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error))
    report.update(inputs=reader.inputs,sources=reader.sources)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    lines=['# R23 compact summary','','Diagnostic observer data; original acceptance is unchanged.','']
    if report['passed']:
        lines+=['Each pooled cell contains6000 actual samples. P50/P99 are pooled quantiles, not averages of round quantiles.','',
            '| ABI | label | off mean ms | on mean ms | off P50 ms | off P99 ms | on P50 ms | on P99 ms |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in report['pooled']:
            values=[row[mode][key]/1e6 for mode,key in (('off','mean_ns'),('on','mean_ns'),('off','p50_ns'),('off','p99_ns'),('on','p50_ns'),('on','p99_ns'))]
            lines.append('| '+row['abi']+' | '+row['label']+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        lines+=['','Seven phase pooled means, every round, neighboring observer ratios, same-candidate ratios and optional caller8 are in JSON.',
                'Nested/worker intervals overlap. Tail synchronization is not pure kernel time. Native durations remain unmeasured.']
    else:lines+=['Rejected: '+report['error']]
    with md.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=report['passed'],output=str(output),error=report.get('error'))),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
