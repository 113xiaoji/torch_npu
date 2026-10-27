"""Bounded native observer; planning is read-only and never invokes SDK/NPU tools.

Option reference: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html
Installed private msprof compatibility must still be established on the target.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback

CAMPAIGN_SHA = 'da4c32bb49c92d0c275db75777934bfd0b11a8576c2b48fcca5d949d1db8d84d'
PROFILE_WRAPPER_SHA = 'd7461dbe7f1d6f0481e865381d9a07aa9bb64bf9ceb02a335a6776e2b0b39865'
WINDOW_SHA = '61d1593b203d46695e492590402ecc508de5350884a1d68e113773c6a16099c7'
OWNERSHIP_SHA = 'b9cd1c9de700e90991471ab05c30ee3e15cc437614d08a91febdd1516a71578c'
EXPECTED_BOOT = '3bb55319-5ec2-4dff-8cbc-23790efd6517'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('root', 'tools', 'control', 'cohort-result', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--window-wrapper', type=Path, default=Path(__file__).with_name('native_run_window_r23.py'))
    p.add_argument('--expected-boot-id', default=EXPECTED_BOOT)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    args.msprof = args.root / 'torch213/ascend920/cann-9.2.0/tools/profiler/bin/msprof'
    args.cann_prefix = args.root / 'torch213/ascend920'
    args.cann_env = args.cann_prefix / 'cann-9.2.0/set_env.sh'
    return args


def load_campaign():
    path = Path(__file__).with_name('run_profile_campaign_r23.py')
    require(sha(path) == CAMPAIGN_SHA, 'frozen R23 campaign controller SHA mismatch')
    spec = importlib.util.spec_from_file_location('_native_observer_campaign', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_ownership():
    path = Path(__file__).with_name('owned_process_r23.py')
    require(OWNERSHIP_SHA is not None and sha(path) == OWNERSHIP_SHA, 'reviewed ownership helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('_native_observer_ownership', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_plan(args, cohort):
    cases = []
    for abi in ('std', 'ft'):
        for label in ('candidate-serial', 'worker4'):
            matches = [r for r in cohort['plan']['repeat'] if r['abi'] == abi and r['label'] == label
                and r['round'] == 1 and r['kind'] == 'perf' and r['profile_mode'] == 'off']
            require(len(matches) == 1, 'exactly one passed round1-off item required')
            item = copy.deepcopy(matches[0]); original = item['argv']
            require(original.count('--output') == 1, 'runner output argument is ambiguous')
            argv = list(original); directory = args.output / (abi + '-' + label)
            raw = directory / 'RUNNER.json'; argv[argv.index('--output') + 1] = str(raw)
            window = directory / 'WINDOW.json'
            application = argv[:3] + [str(args.window_wrapper), '--runner', argv[3],
                '--window-output', str(window), '--'] + argv[4:]
            profiler = directory / 'profiler'
            command = [str(args.msprof), '--application=' + shlex.join(application), '--output=' + str(profiler),
                '--ascendcl=on', '--runtime-api=on', '--task-time=on']
            cases.append(dict(abi=abi, label=label, cohort_item=item, runner_argv=argv,
                application_argv=application, argv=command, output=str(raw), window_output=str(window),
                directory=str(directory), profiler_output=str(profiler), timeout_seconds=1800))
    return dict(schema=1, diagnostic_only=True, performance_conclusion_allowed=False, plan_only=not args.execute,
        cases=cases, fixed_attempts=4, no_retry=True, no_concurrency=True,
        source_argv_change='only original --output changes; explicit clock wrapper is an additional observer',
        profiler_is_observer=True, no_model_or_sdk_changes=True)


def validate_idle(code, raw):
    require(code == 0, 'npu-smi failed')
    text = raw.decode('utf-8', errors='strict')
    require('Health' in text and 'No running processes found' in text, 'NPU idle/health evidence missing')
    health = re.findall(r'^\|\s*\d+\s+Ascend\S*\s*\|\s*([^|]+)\|', text, flags=re.M)
    require(health and all(value.strip() == 'OK' for value in health), 'NPU Health is not OK')
    tail = text.split('Process id', 1)
    require(len(tail) == 2 and not re.search(r'^\|\s*\d+\s+\d+\s*\|\s*\d+', tail[1], re.M),
        'NPU process row is present or process table missing')
    return dict(passed=True, health_rows=len(health), running_processes_observed=False)


def validate_window(row, case, boot=EXPECTED_BOOT):
    require(row.get('schema') == 1 and row.get('window_complete') is True
        and row.get('temporary_module_bindings_restored') is True
        and row.get('diagnostic_only') is True and row.get('performance_conclusion_allowed') is False
        and row.get('original_exit_code') == 0 and row.get('original_report_passed') is True
        and row.get('failure') is None and row.get('samples') == 2000
        and row.get('common_wall_ns_matches') is True and row.get('tool_sha256') == WINDOW_SHA
        and row.get('runner', {}).get('sha256') == 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
        and row.get('original_argv') == case['runner_argv'][4:],
        'clock alignment window identity, coverage or restoration failed')
    process = row.get('application_process', {})
    require(process.get('proc_identity_observed') is True and process.get('host_boot_id') == boot
        and all(type(process.get(key)) is int and process[key] > 0 for key in ('pid', 'pgid', 'starttime_ticks')),
        'actual application PID/birth/boot provenance missing')


def run_owned(argv, env, stream, directory, row, ownership, timeout=1800,
        popen=subprocess.Popen, identity_reader=None):
    process = ownership.OwnedProcess(popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
    identity = None; first = None; observed = None
    try:
        read_identity = identity_reader or (lambda pid: ownership.parse_proc_stat(Path('/proc', str(pid), 'stat').read_text()))
        identity = read_identity(process.pid); row['profiler_process'] = identity
        deadline = time.monotonic() + timeout
        while True:
            observed = process.poll()  # WNOWAIT retains the leader PID/group anchor.
            if observed is not None:
                row['exit_code'] = observed; break
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(.05)
    except BaseException as error:
        first = error
    finally:
        try:
            row['termination'] = ownership.stop_owned_group(process, identity)
        except BaseException as error:
            row['termination_error'] = dict(type=type(error).__name__, error=str(error))
            if first is None:
                first = error
        try:
            write_new(directory / 'TERMINATION.json', dict(observed_profiler_cli=observed,
                profiler_process=identity, termination=row.get('termination'), error=row.get('termination_error')))
        except BaseException as error:
            if first is None:
                first = error
    if first is not None:
        raise first
    require(row['termination'].get('leader_reaped') is True
        and row['termination'].get('group_empty_observed_before_reap') is True
        and row['termination'].get('group_live_members') == [], 'profiler owned group did not close')
    return observed


def proc_identity(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return dict(pid=pid, exists=False)
    fields = text.rsplit(')', 1)[1].split()
    return dict(pid=pid, exists=True, pgid=int(fields[2]), starttime_ticks=int(fields[19]))


def require_campaign_stopped(launch, snapshot):
    require(type(launch.get('pid')) is int and type(launch.get('starttime_ticks')) is int
        and type(launch.get('pgid')) is int, 'original campaign birth identity missing')
    same = snapshot.get('exists') is True and snapshot.get('starttime_ticks') == launch['starttime_ticks']
    require(not same, 'original campaign PID/birth is still live')
    return dict(original_launch=launch, current_process=snapshot, original_instance_live=False,
        pid_reused=snapshot.get('exists') is True)


def scan_campaign(cohort_dir):
    live = []; vanished = []
    names = {'real_dlrm_runner_r22.py', 'instrument_frozen_runner_r23.py', 'run_profile_campaign_r23.py'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [os.fsdecode(x) for x in (entry / 'cmdline').read_bytes().split(b'\0') if x]
        except FileNotFoundError:
            vanished.append(int(entry.name)); continue
        # Any unreadable extant cmdline raises: do not certify an incomplete scan.
        if any(Path(token).name in names for token in tokens) and any(str(cohort_dir) in token for token in tokens):
            live.append(proc_identity(int(entry.name)))
    require(not live, 'live campaign processes remain')
    return dict(passed=True, live_campaign_processes=live, vanished_during_scan=vanished)


def validate_cohort(args, cohort, campaign, helper):
    require(cohort.get('passed') is True and cohort.get('diagnostic_only') is True
        and cohort.get('formal_performance_conclusion_allowed') is False, 'main diagnostic cohort did not pass')
    plan = cohort['plan']; items = plan['gates'] + plan['repeat']; records = cohort.get('results', [])
    require(plan.get('rounds') == 3 and len(plan['gates']) == 6 and len(plan['repeat']) == 36
        and len(items) == len(records) == 42 and cohort.get('completed_cases') == 42, 'main cohort is not fixed42')
    pins = cohort.get('source_pins', {})
    require(all(value in pins.values() for value in [CAMPAIGN_SHA, PROFILE_WRAPPER_SHA, *campaign.PINS.values()]),
        'main cohort source identity is incomplete')
    for path, expected in pins.items():
        require(sha(path) == expected, 'main cohort source/artifact changed: ' + path)
    profiles = cohort.get('runtime_profiles', {})
    require(set(profiles) == {'std-serial', 'std-worker4', 'ft-serial', 'ft-worker4'}, 'four runtime profiles missing')
    versions = profiles['std-serial']['versions']
    require(all(p.get('versions') == versions and p.get('pip_check') for p in profiles.values()), 'profile/pip parity missing')
    topology = plan['topology_proof']
    require(sha(topology['path']) == topology['sha256'], 'main topology proof changed')
    helper.load_contracts().verify_topology(json.loads(Path(topology['path']).read_text()), list(range(8)))
    verified_pins = dict(pins)
    verified_pins[topology['path']] = topology['sha256']
    for item, record in zip(items, records):
        require(record.get('exit_code') == 0 and record.get('output') == item['output']
            and all(record.get(k) == item[k] for k in ('abi', 'label', 'kind', 'round', 'profile_mode')),
            'main cohort case or observed CLI differs')
        require(sha(item['output']) == record.get('raw_sha256'), 'main raw SHA mismatch')
        verified_pins[item['output']] = record['raw_sha256']
        campaign.validate_raw(helper, json.loads(Path(item['output']).read_text()), item, plan)
        if item.get('profile_output'):
            require(sha(item['profile_output']) == record.get('profile_sha256'), 'main sidecar SHA mismatch')
            verified_pins[item['profile_output']] = record['profile_sha256']
            campaign.validate_profile(json.loads(Path(item['profile_output']).read_text()), item, PROFILE_WRAPPER_SHA)
    bindings = helper.load_arm_manifest(args.root / 'r22/candidate-envs-20261027-14/ARMS-r22.json')
    for (abi, arm), binding in bindings.items():
        require(sha(binding['wheel']) == binding['wheel_sha256'], 'physical wheel changed')
        require(sha(binding['build_receipt']) == binding['build_receipt_sha256'], 'build receipt changed')
        raw = json.loads(Path(binding['build_receipt']).read_text()); wheel = raw.get('wheel', {})
        require(raw.get('status') == 'built' and raw.get('abi') == abi and raw.get('canonical_source_unchanged') is True
            and wheel.get('sha256') == binding['wheel_sha256']
            and wheel.get('native', {}).get('torch_npu/lib/libtorch_npu.so', {}).get('sha256') == binding['native_sha256'],
            'build receipt artifact identity differs')
    return verified_pins


def inventory(directory):
    rows = []
    def failed(error):
        raise error
    for base, dirs, files in os.walk(directory, followlinks=False, onerror=failed):
        for name in dirs + files:
            require(not (Path(base) / name).is_symlink(), 'profiler output symlink is not inventoried')
        for name in files:
            path = Path(base) / name
            rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    return sorted(rows, key=lambda row: row['path'])


def execute(args, cohort, plan, campaign, helper):
    require(sys.platform == 'linux', 'execute requires real Linux target')
    require(not args.output.exists(), 'new exclusive output directory required')
    require(not args.output.resolve().is_relative_to((args.root / 'r22').resolve())
        and not args.output.resolve().is_relative_to(args.cohort_result.parent.resolve()), 'output overlaps frozen/cohort inputs')
    require(sha(args.window_wrapper) == WINDOW_SHA, 'window wrapper SHA mismatch')
    ownership = load_ownership()
    cohort_pins = validate_cohort(args, cohort, campaign, helper)
    scope_path = args.control / 'SCOPE.json'; launch_path = args.control / 'LAUNCH.json'
    scope = json.loads(scope_path.read_text()); launch = json.loads(launch_path.read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(boot == args.expected_boot_id == scope.get('boot_id') == launch.get('boot_id'), 'boot identity differs')
    stopped = require_campaign_stopped(launch, proc_identity(launch['pid']))
    scan = scan_campaign(args.cohort_result.parent)
    args.output.mkdir(parents=True)
    pins = {str(Path(__file__).resolve()): sha(__file__), str(args.window_wrapper): WINDOW_SHA,
        str(args.cohort_result): sha(args.cohort_result), str(scope_path): sha(scope_path),
        str(launch_path): sha(launch_path), str(args.msprof): sha(args.msprof)}
    pins[str(Path(__file__).with_name('owned_process_r23.py'))] = OWNERSHIP_SHA
    pins.update(cohort_pins)
    result = dict(schema=1, passed=False, diagnostic_only=True, performance_conclusion_allowed=False,
        plan=plan, source_pins=pins, cases=[], start_ns=time.time_ns())
    try:
        env, sdk = helper.sdk_environment(args, args.output); result['sdk_environment'] = sdk
        idle = subprocess.run(['npu-smi', 'info'], env=env, capture_output=True, timeout=60, check=False)
        (args.output / 'NPU_BEFORE.stdout').write_bytes(idle.stdout)
        (args.output / 'NPU_BEFORE.stderr').write_bytes(idle.stderr)
        result['preflight'] = dict(boot_id=boot, observed_ns=time.time_ns(), self=proc_identity(os.getpid()),
            original_campaign=stopped, process_scan=scan, npu=validate_idle(idle.returncode, idle.stdout))
        write_new(args.output / 'PREFLIGHT.json', result['preflight']); write_new(args.output / 'PLAN.json', plan)
        for case in plan['cases']:
            for path, expected in pins.items():
                require(sha(path) == expected, 'observer input changed')
            directory = Path(case['directory']); directory.mkdir()
            row = dict(abi=case['abi'], label=case['label'], exit_code=None, output=case['output'])
            result['cases'].append(row)
            child_env = dict(env, TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1' if case['abi'] == 'ft' else '0',
                HDB_REGION_DIAGNOSTICS='0', HDB_REGION_TRACE='0')
            try:
                with (directory / 'msprof.log').open('xb') as stream:
                    run_owned(case['argv'], child_env, stream, directory, row, ownership, timeout=1800)
                require(row['exit_code'] == 0, 'msprof/runner failed; guards and raw are preserved')
                raw_path = Path(case['output']); row['raw_sha256'] = sha(raw_path)
                item = copy.deepcopy(case['cohort_item']); item['output'] = case['output']
                campaign.validate_raw(helper, json.loads(raw_path.read_text()), item, cohort['plan'])
                window_path = Path(case['window_output']); window = json.loads(window_path.read_text())
                validate_window(window, case, boot)
                row['window_sha256'] = sha(window_path)
                row['application_process'] = window['application_process']
                row['profiler_files'] = inventory(Path(case['profiler_output']))
                require(any(f['bytes'] > 0 for f in row['profiler_files']), 'msprof produced no nonempty profiler artifact')
                row['native_content_validated'] = False
            finally:
                primary = sys.exc_info()[1]
                try:
                    row['files'] = inventory(directory)
                except BaseException as error:
                    row['inventory_error'] = dict(type=type(error).__name__, error=str(error))
                    if primary is None:
                        raise
            for path, expected in pins.items():
                require(sha(path) == expected, 'observer input changed during case')
        result.update(passed=True, native_content_validated=False,
            scope='four fixed profiler commands/application gates completed; native PID/window/content parsing remains required; no formal performance conclusion')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result['end_ns'] = time.time_ns(); write_new(args.output / 'RESULT.json', result)
    return result


def main(argv=None):
    args = parse_args(argv); cohort = json.loads(args.cohort_result.read_text()); plan = make_plan(args, cohort)
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    campaign = load_campaign(); helper = campaign.load_helper(args)
    def stop(_signal, _frame):
        raise KeyboardInterrupt('native observer received SIGTERM')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        result = execute(args, cohort, plan, campaign, helper)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(dict(passed=result['passed'], output=str(args.output), diagnostic_only=True, error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
