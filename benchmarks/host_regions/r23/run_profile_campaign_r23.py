"""Fresh-process R23 diagnostic cohort; default is host-only planning."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

PINS = {
    'run_real_measurement_r22.py': '667ae9a01b60ab57aa50b90a7a0eda286003c8cacfd6738873233c79b9e416c8',
    'real_dlrm_runner_r22.py': 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f',
    'measurement_contracts_r22.py': '5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a',
}
ARMS_SHA = '25d186a2d979ff6028d6d2b21dc6d442f258049af83a3644a623a4b938adedbb'
RUNTIME_SHA = '4445b2906137f7200318984097bd8354735e19c827fc9caa2a31e32d7ede670c'
CANN_ENV_SHA = '87f2563162c6e2179b3e1aeb71cf57f704de9e0e653ad527a7f5ccdea9ddcc2c'
CPUS = list(range(8))
LABELS = ('original-serial', 'candidate-serial', 'worker4')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('root', 'tools', 'output', 'wrapper'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--wrapper-sha256')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    require(1 <= args.rounds <= 6, 'rounds must be a predeclared bounded value from 1 to 6')
    args.runner = args.tools / 'real_dlrm_runner_r22.py'
    args.arms_manifest = args.root / 'r22/candidate-envs-20261027-14/ARMS-r22.json'
    args.upstream = args.root / 'torch213/upstream'
    args.cann_prefix = args.root / 'torch213/ascend920'
    args.cann_env = args.cann_prefix / 'cann-9.2.0/set_env.sh'
    args.runtime_receipt = args.root / 'r21/runtime-sqlite-r4/RESULT.json'
    args.cpus = '0,1,2,3,4,5,6,7'
    return args


def load_helper(args):
    for name, expected in PINS.items():
        require(sha(args.tools / name) == expected, 'frozen source SHA mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('_r23_frozen_controller', args.tools / 'run_real_measurement_r22.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def make_plan(args, helper):
    bindings = helper.load_arm_manifest(args.arms_manifest)
    arms = [(abi, label) for abi in ('std', 'ft') for label in LABELS]

    def row(abi, label, kind, round_index, mode):
        binding = copy.deepcopy(bindings[abi, 'serial' if label == 'original-serial' else 'worker4'])
        runner_arm = 'worker4' if label == 'worker4' else 'serial'
        prefix = ('gate-' if kind == 'gate' else 'round' + str(round_index) + '-') + abi + '-' + label
        if kind != 'gate':
            prefix += '-' + mode
        output = args.output / (prefix + '.json')
        forwarded = ['--abi', abi, '--arm', runner_arm, '--kind', kind,
            '--upstream', str(args.upstream), '--cann-prefix', str(args.cann_prefix), '--cpus', args.cpus,
            '--wheel', binding['wheel'], '--expected-wheel-sha256', binding['wheel_sha256'],
            '--expected-native-sha256', binding['native_sha256'], '--threads', '1',
            '--warmup', '0' if kind == 'gate' else '20', '--iterations', '1' if kind == 'gate' else '2000',
            '--output', str(output), '--runtime-receipt', str(args.runtime_receipt)]
        if runner_arm == 'worker4' and kind == 'perf':
            forwarded += ['--gate-proof', str(args.output / ('gate-' + abi + '-worker4.json'))]
        prefix_argv = [binding['python'], '-I', '-B']
        profile = args.output / (prefix + '.profile.json') if mode == 'stages' else None
        argv = (prefix_argv + [str(args.runner)] + forwarded if profile is None else
            prefix_argv + [str(args.wrapper), '--runner', str(args.runner), '--profile-output', str(profile),
                '--profile-mode', 'stages', '--'] + forwarded)
        return dict(abi=abi, label=label, runner_arm=runner_arm, kind=kind, round=round_index,
            profile_mode=mode, binding=binding, argv=argv, runner_argv=forwarded, output=str(output),
            profile_output=str(profile) if profile else None, warmup=0 if kind == 'gate' else 20,
            iterations=1 if kind == 'gate' else 2000, timeout_seconds=1800, cpus=list(CPUS), fresh_process=True)

    gates = [row(abi, label, 'gate', 0, 'off') for abi, label in arms]
    repeat = []
    for index in range(args.rounds):
        offset = (2 * index) % len(arms)
        order = arms[offset:] + arms[:offset]
        for position, (abi, label) in enumerate(order):
            modes = ('off', 'stages') if (index + position) % 2 == 0 else ('stages', 'off')
            repeat.extend(row(abi, label, 'perf', index + 1, mode) for mode in modes)
    return dict(schema=1, protocol='R23 six-arm stage diagnosis', diagnostic_only=True,
        formal_performance_conclusion_allowed=False, original_R22_unchanged=True, plan_only=not args.execute,
        gates=gates, repeat=repeat, rounds=args.rounds, selected_cpus=list(CPUS), topology_pending=True,
        arms_manifest=dict(path=str(args.arms_manifest)), frozen_source_pins=dict(PINS),
        order_policy='six-arm rotations by two positions; adjacent off/stages pair order alternates by slot and round',
        timing='20 warmup plus 2000 original forward-sync calls per diagnostic process',
        overlap_boundary='inclusive async stages are not additive; instrumented results are diagnostic only',
        installs_packages=False, rebuilds_runtime=False, stability_processes=0)


def check_sources(args, source_pins):
    for path, expected in source_pins.items():
        require(sha(path) == expected, 'source changed during campaign: ' + path)


def validate_raw(helper, raw, item, plan):
    helper.validate_result(raw, item['abi'], item['runner_arm'], item['kind'])
    binding = item['binding']
    require(raw.get('script_sha256') == PINS['real_dlrm_runner_r22.py']
        and raw.get('requested_wheel', {}).get('sha256') == binding['wheel_sha256']
        and raw.get('native_sha256') == binding['native_sha256'], 'runner physical artifact differs')
    require(raw.get('topology', {}).get('proof_sha256') == plan['topology_proof']['sha256']
        and raw.get('cpus') == CPUS, 'runner escaped the common CPU0-7 topology proof')


def validate_profile(profile, item, wrapper_sha):
    require(profile.get('diagnostic_only') is True and profile.get('performance_conclusion_allowed') is False
        and profile.get('profile_mode') == 'stages' and profile.get('original_exit_code') == 0
        and profile.get('failure') is None and profile.get('tool_sha256') == wrapper_sha
        and profile.get('runner', {}).get('sha256') == PINS['real_dlrm_runner_r22.py']
        and profile.get('original_argv') == item['runner_argv']
        and profile.get('temporary_methods_restored') is True
        and profile.get('instrumentation_complete') is True and profile.get('requests') == 2000,
        'profile identity, restoration or coverage failed')
    expected = profile.get('expected_event_counts', {})
    actual = profile.get('actual_event_counts', {})
    require(expected.get('forward') == 2000 and expected.get('tail_sync') == 2000
        and expected.get('model.apply_emb') == 2000
        and all(actual.get(key) == count for key, count in expected.items()), 'profile event counts differ')


def execute(args, plan, helper):
    require(sys.platform == 'linux', 'execution requires the real Linux target')
    require(not args.output.exists(), 'new exclusive output directory required')
    output = args.output.resolve()
    for protected in (args.tools, args.upstream, args.cann_prefix, args.root / 'r22'):
        require(not output.is_relative_to(protected.resolve()), 'output overlaps frozen input tree')
    require(sha(args.arms_manifest) == ARMS_SHA, 'original four-binding manifest SHA mismatch')
    wrapper_sha = sha(args.wrapper)
    require(args.wrapper_sha256 is None or args.wrapper_sha256 == wrapper_sha, 'wrapper SHA mismatch')
    sources = {str(args.tools / name): value for name, value in PINS.items()}
    sources.update({str(args.wrapper): wrapper_sha, str(Path(__file__).resolve()): sha(__file__),
        str(args.arms_manifest): ARMS_SHA, str(args.runtime_receipt): RUNTIME_SHA,
        str(args.cann_env): CANN_ENV_SHA})
    check_sources(args, sources)
    bindings = helper.load_arm_manifest(args.arms_manifest)
    receipts = {}
    for (abi, arm), binding in bindings.items():
        require(sha(binding['wheel']) == binding['wheel_sha256'], 'wheel SHA mismatch')
        path = Path(binding['build_receipt'])
        require(sha(path) == binding['build_receipt_sha256'], 'build receipt SHA mismatch')
        raw = json.loads(path.read_text()); wheel = raw.get('wheel', {})
        require(raw.get('status') == 'built' and raw.get('abi') == abi
            and raw.get('canonical_source_unchanged') is True and wheel.get('sha256') == binding['wheel_sha256']
            and wheel.get('native', {}).get('torch_npu/lib/libtorch_npu.so', {}).get('sha256') == binding['native_sha256'],
            'build receipt does not match physical artifact')
        receipts[abi + '-' + arm] = dict(path=str(path), sha256=sha(path))
        sources[str(path)] = binding['build_receipt_sha256']
    args.output.mkdir(parents=True)
    result = dict(schema=1, passed=False, diagnostic_only=True, formal_performance_conclusion_allowed=False,
        overall_R22_acceptance_changed=False, plan=plan, source_pins=sources, results=[], build_receipts=receipts,
        start_ns=time.time_ns())
    try:
        require(set(CPUS) <= os.sched_getaffinity(0), 'CPU0-7 not allowed')
        contracts = helper.load_contracts()
        topology = contracts.capture_topology(CPUS)
        require(topology['selected_cpus'] == CPUS, 'topology did not select exactly eight physical CPU0-7 cores')
        proof_path = args.output / 'TOPOLOGY.json'; write_new(proof_path, topology)
        proof_sha = sha(proof_path)
        plan['topology_proof'] = dict(path=str(proof_path), sha256=proof_sha)
        plan['topology_pending'] = False
        plan['arms_manifest']['sha256'] = ARMS_SHA
        plan['wrapper_sha256'] = wrapper_sha
        for item in plan['gates'] + plan['repeat']:
            suffix = ['--topology-proof', str(proof_path), '--topology-proof-sha256', proof_sha]
            item['argv'].extend(suffix); item['runner_argv'].extend(suffix)
        write_new(args.output / 'PLAN.json', plan)
        env, sdk = helper.sdk_environment(args, args.output); result['sdk_environment'] = sdk
        profiles = {}
        for (abi, arm), binding in bindings.items():
            child_env = dict(env, TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1' if abi == 'ft' else '0',
                HDB_REGION_DIAGNOSTICS='0', HDB_REGION_TRACE='0')
            profiles[abi + '-' + arm] = helper.runtime_profile(binding, bindings[abi, 'serial'], child_env, args.output)
        reference = profiles['std-serial']['versions']
        require(all(p['versions'] == reference for p in profiles.values()), 'four runtime distribution versions differ')
        result['runtime_profiles'] = profiles
        for index, item in enumerate(plan['gates'] + plan['repeat']):
            check_sources(args, sources)
            write_new(args.output / ('STATUS-' + str(index).zfill(3) + '.json'),
                dict(index=index, abi=item['abi'], label=item['label'], mode=item['profile_mode'], phase='launching'))
            child_env = dict(env, TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1' if item['abi'] == 'ft' else '0',
                HDB_REGION_DIAGNOSTICS='1' if item['kind'] == 'gate' else '0',
                HDB_REGION_TRACE='1' if item['kind'] == 'gate' else '0')
            path = Path(item['output'])
            record = dict(abi=item['abi'], label=item['label'], runner_arm=item['runner_arm'], kind=item['kind'],
                round=item['round'], profile_mode=item['profile_mode'], output=str(path), exit_code=None,
                raw_sha256=None, formal_performance_conclusion_allowed=False)
            result['results'].append(record)
            try:
                with path.with_suffix('.log').open('xb') as stream:
                    code = helper.run_child(item['argv'], child_env, stream, timeout=1800)
            except BaseException as error:
                record.update(error_type=type(error).__name__, error=str(error),
                    raw_sha256=sha(path) if path.is_file() else None)
                raise
            record.update(exit_code=code, raw_sha256=sha(path) if path.is_file() else None)
            check_sources(args, sources)
            require(code == 0, 'child failed; preserve raw/log: ' + str(path))
            require(path.is_file(), 'child raw is missing')
            validate_raw(helper, json.loads(path.read_text()), item, plan)
            if item['profile_output']:
                profile_path = Path(item['profile_output'])
                validate_profile(json.loads(profile_path.read_text()), item, wrapper_sha)
                record.update(profile_output=str(profile_path), profile_sha256=sha(profile_path))
        check_sources(args, sources)
        result.update(passed=True, completed_cases=len(result['results']),
            scope='six fresh gates and fixed six-label off/stages rounds; no stability, no optimization or formal speedup claim')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result['end_ns'] = time.time_ns()
        write_new(args.output / 'RESULT.json', result)
        write_new(args.output / 'STATUS.json', dict(passed=result['passed'], phase='complete' if result['passed'] else 'failed',
            completed_cases=len(result['results']), error=result.get('error'), diagnostic_only=True))
    return result


def main(argv=None):
    args = parse_args(argv); helper = load_helper(args); plan = make_plan(args, helper)
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    def stop(_signal, _frame):
        raise KeyboardInterrupt('R23 controller received SIGTERM')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        result = execute(args, plan, helper)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(dict(passed=result['passed'], diagnostic_only=True, output=str(args.output), error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
