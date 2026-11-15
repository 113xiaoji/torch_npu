"""Serial seven-arm A3 campaign; default three rounds are a pilot."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def load(name):
    spec = importlib.util.spec_from_file_location('r34_campaign_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load('contracts')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker_argv(launch, worker, config, output, row):
    policy = C.arm_policy(row['arm'])
    C.require(type(launch.get('argv_prefix')) is list
              and all(type(token) is str for token in launch['argv_prefix']), 'argv_prefix must be a JSON string list')
    prefix = [token.replace('{ingress}', policy['ingress']) for token in launch['argv_prefix']]
    return prefix + [worker, '--config', config, '--output', output, '--arm', row['arm'],
                     '--kind', row['kind'], '--round', str(row['round']),
                     '--samples', str(row['samples']), '--warmup', str(row['warmup'])]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan = C.campaign_plan(args.rounds)
    write(args.output / 'plan.json', dict(schema=1, plan=plan, rounds=args.rounds,
        pilot_only=args.rounds % 14 != 0, metric='original model forward through device synchronization',
        comparison='same candidate binary, off(TQ1), mutex/mpsc with workers1/2/4'))
    if args.plan_only:
        return
    C.require(args.launch and args.config and args.worker and args.container_output,
              'execution needs launch/config/worker/container-output')
    launch = json.loads(args.launch.read_text())
    bound = launch.get('worker_timeout_seconds')
    C.require(type(bound) is int and 30 <= bound <= 600
              and any(Path(token).name == 'timeout' for token in launch['argv_prefix']),
              'launch prefix must bound the actual container worker with timeout; declare worker_timeout_seconds')
    write(args.output / 'launch.json', launch)
    lifecycle, results = [], []
    audit = load('analyze')
    for i, row in enumerate(plan):
        label = f"{i:03d}-{row['kind']}-r{row['round']}-{row['arm']}"
        remote_output = args.container_output.rstrip('/') + '/' + label
        argv = worker_argv(launch, args.worker, args.config, remote_output, row)
        life = dict(label=label, plan_index=i, argv=argv, monotonic_start_ns=time.monotonic_ns())
        lifecycle.append(life)
        write(args.output / 'lifecycle.json', lifecycle)
        with (args.output / (label + '.log')).open('w') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            life['launcher_pid'] = process.pid
            try:
                life['returncode'] = process.wait(timeout=bound + 60)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                life.update(returncode=process.returncode, controller_timeout=True)
        life['monotonic_end_ns'] = time.monotonic_ns()
        if args.pull_results:
            copied = subprocess.run(['docker', 'cp', launch['container'] + ':' + remote_output,
                                     str(args.output / label)], capture_output=True, text=True, timeout=30)
            life['copy_returncode'] = copied.returncode
            if copied.returncode:
                life['copy_error'] = copied.stderr
        write(args.output / 'lifecycle.json', lifecycle)
        try:
            C.require(life['returncode'] == 0 and not life.get('controller_timeout'), 'worker failed or timed out: ' + label)
            result = json.loads((args.output / label / 'result.json').read_text())
            for key in ('kind', 'arm', 'round', 'warmup', 'samples'):
                C.require(result.get(key) == row[key], 'worker receipt does not match planned ' + key)
            result = audit.audit_result(result)
            results.append(result)
        except BaseException as error:
            write(args.output / 'status.json', dict(status='failed', completed=len(results),
                last=row, error=dict(type=type(error).__name__, message=str(error))))
            raise
        write(args.output / 'status.json', dict(status='running', completed=len(results), total=len(plan), last=row))
        print(json.dumps(dict(completed=len(results), **row, timing=result['timing'])), flush=True)
    summary = C.summarize([r for r in results if r['kind'] == 'perf'], args.rounds)
    write(args.output / 'summary.json', summary)
    write(args.output / 'status.json', dict(status='complete', completed=len(results), total=len(plan),
                                           pilot_only=args.rounds % 14 != 0))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--config')
    parser.add_argument('--worker')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--container-output')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--pull-results', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
