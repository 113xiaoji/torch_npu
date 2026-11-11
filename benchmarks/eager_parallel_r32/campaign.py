"""Serial fresh-process real-device campaign; launch prefix is supplied as JSON.

The five correctness workers must pass before any performance window starts.
Performance order balances position and immediate carryover over ten rounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ARMS = ('off', 'e', 'ei', 'ae', 'aei')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    launch = json.loads(args.launch.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    base = (0, 1, 4, 2, 3)
    orders = [tuple(ARMS[(i + shift) % 5] for i in base) for shift in range(5)]
    orders += [tuple(reversed(order)) for order in orders]
    plan = [dict(kind='correctness', round=0, arm=arm, samples=3, warmup=0) for arm in ARMS]
    plan += [dict(kind='perf', round=i, arm=arm, samples=3000, warmup=50)
             for i, order in enumerate(orders) for arm in order]
    write(args.output / 'plan.json', dict(plan=plan, worker=args.worker,
        metric='original forward + device synchronize; per-request latency',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    lifecycle = []
    results = []
    for i, row in enumerate(plan):
        label = f"{i:03d}-{row['kind']}-r{row['round']}-{row['arm']}"
        remote_output = args.container_output + '/' + label
        cmd = launch['argv_prefix'] + [args.worker, '--config', args.config,
            '--output', remote_output, '--arm', row['arm'], '--kind', row['kind'],
            '--round', str(row['round']), '--samples', str(row['samples']), '--warmup', str(row['warmup'])]
        start = time.monotonic_ns()
        with (args.output / (label + '.log')).open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        end = time.monotonic_ns()
        life = dict(label=label, host_launcher_pid=process.pid, returncode=code,
                    monotonic_start_ns=start, monotonic_end_ns=end)
        if args.pull_results:
            copied = subprocess.run(['docker', 'cp', launch['container'] + ':' + remote_output,
                                     str(args.output / label)], capture_output=True, text=True)
            life['result_copy_returncode'] = copied.returncode
            if copied.returncode:
                life['result_copy_error'] = copied.stderr
        lifecycle.append(life)
        write(args.output / 'lifecycle.json', lifecycle)
        result_path = args.output / label / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if code != 0 or result.get('passed') is not True:
            write(args.output / 'status.json', dict(status='failed', row=row, lifecycle=life,
                error=result.get('error'), completed=len(results)))
            raise RuntimeError('worker failed: ' + label)
        results.append(result)
        write(args.output / 'status.json', dict(status='running', completed=len(results),
            total=len(plan), last=row, timing=result['timing']))
        print(json.dumps(dict(completed=len(results), **row, timing=result['timing'])), flush=True)
    perf = {(r['round'], r['arm']): r for r in results if r['kind'] == 'perf'}
    summary = {}
    for arm in ARMS:
        medians = [perf[i, arm]['timing']['median_ns'] for i in range(10)]
        changes = [(perf[i, arm]['timing']['median_ns'] /
                    perf[i, 'off']['timing']['median_ns'] - 1) * 100 for i in range(10)]
        summary[arm] = dict(median_of_round_medians_ns=statistics.median(medians),
            round_medians_ns=medians, paired_latency_change_pct=statistics.median(changes),
            paired_changes_pct=changes, faster_rounds=sum(value < 0 for value in changes))
    write(args.output / 'summary.json', summary)
    write(args.output / 'status.json', dict(status='complete', completed=len(results), total=len(plan)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', required=True, type=Path)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--container-output', required=True)
    parser.add_argument('--pull-results', action='store_true',
                        help='copy each finished worker result from a container without a shared volume')
    run(parser.parse_args())
