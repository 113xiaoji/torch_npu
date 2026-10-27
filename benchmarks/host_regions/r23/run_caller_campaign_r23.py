"""One fixed eight-process caller-only supplement; default is read-only planning."""
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OBSERVER_SHA = '0a3c0aeab6cb8544b5494ee860ccfcb9ec81bb6e74aaace80352295e694b1fd9'
CALLER_SHA = 'f0a78ebc2296e71faab20cfd1cf5e03d4be14e624024163954bdd01caf7b32d7'
BASE_SHA = 'd7461dbe7f1d6f0481e865381d9a07aa9bb64bf9ceb02a335a6776e2b0b39865'
ANALYZER_SHA = 'ac25791b995880ef114c530db6e099ec6eda00a9d29a558f0a587a4c6a51e0f8'
RUNNER_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
REQUIRED_COHORT_SOURCES = (RUNNER_SHA, BASE_SHA,
    'da4c32bb49c92d0c275db75777934bfd0b11a8576c2b48fcca5d949d1db8d84d',
    '667ae9a01b60ab57aa50b90a7a0eda286003c8cacfd6738873233c79b9e416c8',
    '5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path, expected):
    require(sha(path) == expected, 'reviewed source SHA mismatch: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def read_json(path):
    data = Path(path).read_bytes(); return json.loads(data), hashlib.sha256(data).hexdigest()


def basic_cohort(cohort):
    require(cohort.get('passed') is True and cohort.get('diagnostic_only') is True
            and cohort.get('formal_performance_conclusion_allowed') is False and cohort.get('completed_cases') == 42,
            'passed main42 diagnostic RESULT required')
    plan = cohort.get('plan', {})
    require(plan.get('rounds') == 3 and len(plan.get('gates', [])) == 6 and len(plan.get('repeat', [])) == 36
            and len(cohort.get('results', [])) == 42, 'fixed main42 plan/records required')
    require(set(REQUIRED_COHORT_SOURCES) <= set(cohort.get('source_pins', {}).values()), 'main42 source identities missing')


def make_plan(args, cohort):
    basic_cohort(cohort); cases = []
    for position, (abi, label) in enumerate((('std', 'candidate-serial'), ('std', 'worker4'),
                                           ('ft', 'candidate-serial'), ('ft', 'worker4'))):
        matches = [r for r in cohort['plan']['repeat'] if r['abi'] == abi and r['label'] == label
                   and r['kind'] == 'perf' and r['round'] == 1 and r['profile_mode'] == 'off']
        require(len(matches) == 1, 'one original round1-off row required')
        item = matches[0]; original = item['argv']
        require(original.count('--output') == 1 and original[1:3] == ['-I', '-B'], 'unambiguous original isolated runner command required')
        modes = ('off', 'caller_only') if position % 2 == 0 else ('caller_only', 'off')
        for mode in modes:
            directory = args.output/(abi+'-'+label+'-'+mode); raw = directory/'RUNNER.json'
            runner = list(original); runner[runner.index('--output')+1] = str(raw)
            profile = directory/'CALLER.json' if mode == 'caller_only' else None
            argv = runner if profile is None else runner[:3]+[str(args.caller_wrapper), '--base-wrapper', str(args.base_wrapper),
                '--runner', runner[3], '--profile-mode', 'caller_only', '--profile-output', str(profile), '--']+runner[4:]
            cases.append(dict(abi=abi, label=label, mode=mode, directory=str(directory), output=str(raw),
                profile_output=str(profile) if profile else None, runner_argv=runner, argv=argv,
                cohort_item=copy.deepcopy(item), timeout_seconds=1800, fresh_process=True))
    return dict(schema=1, diagnostic_only=True, performance_conclusion_allowed=False, plan_only=not args.execute,
                fixed_attempts=8, no_retry=True, no_concurrency=True, rounds=1, cases=cases,
                source_argv_change='Only --output changes in original round1-off runner argv; caller_only adds its explicit outer wrapper.',
                order='std/FT candidate-serial/worker4; adjacent off/caller_only pair order alternates',
                exposure='unchanged perf20+2000; original gate/topology/artifact arguments retained')


def validate_sidecar(profile, case, caller, analyzer, application_pid):
    require(type(profile.get('schema')) is int and profile['schema'] == 1 and profile.get('profile_mode') == 'caller_only'
        and profile.get('tool_sha256') == CALLER_SHA and profile.get('base_wrapper', {}).get('sha256') == BASE_SHA
        and profile.get('runner', {}).get('sha256') == RUNNER_SHA and type(profile.get('original_exit_code')) is int and profile['original_exit_code'] == 0
        and profile.get('failure') is None and profile.get('diagnostic_only') is True
        and profile.get('performance_conclusion_allowed') is False and profile.get('instrumentation_complete') is True
        and profile.get('temporary_methods_restored') is True and type(profile.get('requests')) is int and profile['requests'] == 2000
        and profile.get('original_argv') == case['runner_argv'][4:]
        and type(profile.get('worker_hooks_installed')) is int and profile['worker_hooks_installed'] == 0
        and type(profile.get('worker_events')) is int and profile['worker_events'] == 0,
        'caller sidecar source/mode/CLI/restoration/zero-worker evidence differs')
    require(profile.get('restoration') and all(r.get('restored') is True for r in profile['restoration']), 'caller restoration records failed')
    events = profile.get('events'); analyzer.validate_events(events)
    require({r['native_tid'] for r in events} == {application_pid}, 'caller event TID differs from actual Python application PID')
    view = SimpleNamespace(events=lambda: events, request=profile['requests'], restoration=profile['restoration'],
                           worker_hook_installed=False, worker_target=None)
    complete, counts, snapshots = caller.coverage(view, 0, case['cohort_item']['runner_arm'], 'caller_only')
    require(complete and counts == profile.get('actual_event_counts') and snapshots == profile.get('snapshot_per_request'),
            'caller per-request fixed tags/snapshots differ')
    expected = {name: 2000 for name in caller.CORE+(caller.POLICY if case['cohort_item']['runner_arm'] == 'worker4' else ())}
    if snapshots: expected[caller.SNAPSHOT] = snapshots[0]['count']*2000
    require(profile.get('expected_event_counts') == expected, 'caller declared fixed counts differ')
    require(profile.get('worker_metrics_observed') is False and all(profile.get('native_measurements', {}).get(k, 'missing') is None
        for k in ('query_ns', 'execute_ns', 'queue_wait_ns', 'device_ns')), 'unmeasured native fields must be null')


def caller_stages(raw, profile, analyzer):
    durations = raw['timing']['durations_ns']; require(len(durations) == 2000, 'full2000 raw durations required')
    groups = defaultdict(list)
    for row in profile['events']: groups[row['request']].append(row)
    rows = []; previous = -1
    for index, value in enumerate(durations, 1):
        row = analyzer.partition_request(groups[index], value)
        require(row['start_ns'] >= previous, 'caller request timelines overlap or are reordered')
        previous = row['end_ns']; rows.append(row)
    return {name: {key: value for key, value in analyzer.distribution([r['stages_ns'][name] for r in rows]).items()
                   if key in ('mean_ns', 'p50_ns', 'p99_ns')} for name in rows[0]['stages_ns']}


def pair_summary(records):
    pairs = []
    for abi in ('std', 'ft'):
        for label in ('candidate-serial', 'worker4'):
            selected = [r for r in records if r['abi'] == abi and r['label'] == label]
            off = [r for r in selected if r['mode'] == 'off']; on = [r for r in selected if r['mode'] == 'caller_only']
            require(len(off) == len(on) == 1, 'one complete adjacent observer pair required')
            baseline, observed = off[0]['raw_summary']['mean_ns'], on[0]['raw_summary']['mean_ns']
            pairs.append(dict(abi=abi, label=label, off_mean_ns=baseline, caller_only_mean_ns=observed,
                caller_only_over_off=observed/baseline, observer_delta_mean_ns=observed-baseline,
                stage_values_corrected=False, boundary='Separate fresh-process observer comparison; no subtraction from stages or SDK attribution.'))
    return pairs


def execute(args, cohort, cohort_sha, plan, observer, caller, analyzer):
    require(sys.platform == 'linux' and os.getuid() == os.geteuid() != 0, 'ordinary nonroot Linux execution required')
    require(not args.output.exists(), 'new exclusive output directory required')
    for protected in (args.root/'r22', args.cohort_result.parent, args.tools, args.cann_prefix, args.root/'torch213/upstream'):
        require(not args.output.resolve().is_relative_to(protected.resolve()), 'output overlaps protected input tree')
    campaign = observer.load_campaign(); helper = campaign.load_helper(args); ownership = observer.load_ownership()
    require(sha(args.cohort_result) == cohort_sha, 'main RESULT changed after parsing')
    pins = observer.validate_cohort(args, cohort, campaign, helper)
    scope_path, launch_path = args.control/'SCOPE.json', args.control/'LAUNCH.json'
    scope, scope_sha = read_json(scope_path); launch, launch_sha = read_json(launch_path)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(boot == args.expected_boot_id == observer.EXPECTED_BOOT == scope.get('boot_id') == launch.get('boot_id'), 'fixed boot3bb identity differs')
    stopped = observer.require_campaign_stopped(launch, observer.proc_identity(launch['pid']))
    scan = observer.scan_campaign(args.cohort_result.parent)
    pins.update({str(Path(__file__).resolve()): sha(__file__), str(HERE/'run_msprof_r23.py'): OBSERVER_SHA,
        str(HERE/'run_profile_campaign_r23.py'): observer.CAMPAIGN_SHA, str(HERE/'owned_process_r23.py'): observer.OWNERSHIP_SHA,
        str(args.caller_wrapper): CALLER_SHA, str(args.base_wrapper): BASE_SHA, str(args.analysis_helper): ANALYZER_SHA,
        str(args.cohort_result): cohort_sha, str(scope_path): scope_sha, str(launch_path): launch_sha})
    for case in plan['cases']:
        require(sha(case['runner_argv'][3]) == RUNNER_SHA and case['runner_argv'][0] == case['cohort_item']['binding']['python'], 'original runner/Python identity differs')
        pins[case['runner_argv'][3]] = RUNNER_SHA
    def check():
        for path, expected in pins.items(): require(sha(path) == expected, 'caller input changed: '+path)
        require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'boot changed during caller cohort')
    check(); args.output.mkdir(parents=True)
    result = dict(schema=1, passed=False, diagnostic_only=True, performance_conclusion_allowed=False,
                  source_pins=pins, plan=plan, cases=[], start_ns=time.time_ns())
    try:
        env, sdk = helper.sdk_environment(args, args.output); result['sdk_environment'] = sdk
        require(sdk['sha256'] == campaign.CANN_ENV_SHA, 'SDK environment source differs from main cohort')
        idle = subprocess.run(['npu-smi', 'info'], env=env, capture_output=True, timeout=60, check=False)
        (args.output/'NPU_BEFORE.stdout').write_bytes(idle.stdout); (args.output/'NPU_BEFORE.stderr').write_bytes(idle.stderr)
        result['preflight'] = dict(boot_id=boot, original_campaign=stopped, process_scan=scan,
                                  npu=observer.validate_idle(idle.returncode, idle.stdout))
        observer.write_new(args.output/'PREFLIGHT.json', result['preflight']); observer.write_new(args.output/'PLAN.json', plan)
        for case in plan['cases']:
            check(); directory = Path(case['directory']); directory.mkdir()
            row = dict(abi=case['abi'], label=case['label'], mode=case['mode'], exit_code=None, output=case['output'])
            result['cases'].append(row)
            child_env = dict(env, TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1' if case['abi'] == 'ft' else '0',
                             HDB_REGION_DIAGNOSTICS='0', HDB_REGION_TRACE='0')
            with (directory/'child.log').open('xb') as log:
                observer.run_owned(case['argv'], child_env, log, directory, row, ownership, timeout=1800)
            require(row['exit_code'] == 0, 'caller child CLI failed; fail-fast without retry')
            row['application_process'] = row['profiler_process']  # Historical helper key; this argv is direct Python, no profiler.
            text = (directory/'child.log').read_text(encoding='utf-8', errors='replace')
            require('Traceback (most recent call last):' not in text and 'Exception ignored' not in text, 'unhandled child stderr failure')
            raw, row['raw_sha256'] = read_json(case['output'])
            item = copy.deepcopy(case['cohort_item']); item['output'] = case['output']
            campaign.validate_raw(helper, raw, item, cohort['plan'])
            durations = raw['timing']['durations_ns']; require(len(durations) == 2000, 'full2000 samples required')
            row['raw_summary'] = analyzer.distribution(durations)
            if case['mode'] == 'caller_only':
                profile, row['profile_sha256'] = read_json(case['profile_output'])
                validate_sidecar(profile, case, caller, analyzer, row['application_process']['pid'])
                row['profile_output'] = case['profile_output']; row['caller_stages'] = caller_stages(raw, profile, analyzer)
            row['files'] = observer.inventory(directory); check()
        result.update(passed=True, completed_cases=8, observer_pairs=pair_summary(result['cases']),
            scope='One bounded eight-process caller observer supplement; no native duration or formal speedup claim.')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result['end_ns'] = time.time_ns(); observer.write_new(args.output/'RESULT.json', result)
    return result


def main(argv=None):
    outer = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    outer.add_argument('--caller-wrapper', type=Path, default=HERE/'instrument_caller_only_r23.py')
    outer.add_argument('--base-wrapper', type=Path, default=HERE/'instrument_frozen_runner_r23.py')
    outer.add_argument('--analysis-helper', type=Path, default=HERE/'analyze_profile_r23.py')
    special, forwarded = outer.parse_known_args(argv)
    observer = load('_caller_reviewed_observer', HERE/'run_msprof_r23.py', OBSERVER_SHA)
    args = observer.parse_args(forwarded)
    for name in ('caller_wrapper', 'base_wrapper', 'analysis_helper'): setattr(args, name, getattr(special, name).resolve())
    cohort, cohort_sha = read_json(args.cohort_result); plan = make_plan(args, cohort)
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    caller = load('_caller_supplement_wrapper', args.caller_wrapper, CALLER_SHA)
    load('_caller_supplement_base', args.base_wrapper, BASE_SHA)
    analyzer = load('_caller_partition_helpers', args.analysis_helper, ANALYZER_SHA)  # Pure interval helpers, not the stages-only admission gate.
    def stop(_number, _frame): raise KeyboardInterrupt('caller campaign received SIGTERM')
    previous = signal.signal(signal.SIGTERM, stop)
    try: result = execute(args, cohort, cohort_sha, plan, observer, caller, analyzer)
    finally: signal.signal(signal.SIGTERM, previous)
    print(json.dumps(dict(passed=result['passed'], output=str(args.output), error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
