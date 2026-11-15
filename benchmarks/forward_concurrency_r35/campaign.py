"""Serial fresh-process R35 correctness then performance campaign controller.

Launch JSON argv_prefix must end at the actual free-threaded interpreter, after
any timeout/container/loader prefix. Only GIL-on inserts ``-X gil=1`` there;
GIL-off uses that binary's natural default and must be verified by the worker.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def load(name):
    spec = importlib.util.spec_from_file_location('r35_campaign_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load('contracts')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_launch(launch):
    prefix = launch.get('argv_prefix')
    C.require(type(prefix) is list and bool(prefix)
              and all(type(token) is str and bool(token) for token in prefix),
              'argv_prefix must be a nonempty JSON string list')
    bound = launch.get('worker_timeout_seconds')
    C.require(type(bound) is int and 1 <= bound <= 600, 'worker timeout must be 1..600 seconds')
    timeouts = [index for index, token in enumerate(prefix)
                if Path(token).name in ('timeout', 'timeout.exe')]
    C.require(len(timeouts) == 1 and timeouts[0] + 2 < len(prefix)
              and prefix[timeouts[0] + 1] == str(bound),
              'actual worker must have timeout with declared bound in launch prefix')
    C.require(not any(token == '-X' or token.startswith('-Xgil')
                      or 'PYTHON_GIL' in token for token in prefix),
              'launch prefix must not override natural GIL mode')
    return bound


def worker_argv(launch, worker, config, output, row):
    validate_launch(launch)
    C.validate_case(row)
    prefix = list(launch['argv_prefix'])
    # Loader selection remains structured, without shell expansion.
    ingress = 'mpsc' if row['backend'] == 'mpsc4' else 'mutex'
    prefix = [token.replace('{ingress}', ingress) for token in prefix]
    gil_args = ['-X', 'gil=1'] if row['gil_mode'] == 'on' else []
    argv = prefix + gil_args + [str(worker), '--config', str(config), '--output', str(output)]
    for key in C.TASK_KEYS:
        argv += ['--' + key.replace('_', '-'), str(row[key])]
    return argv


def run(args):
    plan = C.campaign_plan(args.device, args.rounds, args.include_prepare)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'plan.json', dict(schema=1, device=args.device, rounds=args.rounds,
        include_prepare=args.include_prepare, plan=plan,
        metric='exact completed requests / completed wall window; nearest-rank request latency',
        order_interpretation='deterministic rotated alternating order; no formal balance claim'))
    if args.plan_only:
        return
    C.require(args.launch and args.config and args.worker, 'execution needs launch/config/worker')
    launch = json.loads(args.launch.read_text(encoding='utf-8'))
    bound = validate_launch(launch)
    C.require(not args.pull_results or (launch.get('container') and args.container_output),
              'pull-results needs container identity and container-output')
    write(args.output / 'launch.json', launch)
    audit = load('analyze')
    lifecycle, results = [], []
    for index, task in enumerate(plan):
        label = audit.receipt_label(index, task)
        local_output = args.output / label
        worker_output = (args.container_output.rstrip('/') + '/' + label
                         if args.container_output else str(local_output))
        argv = worker_argv(launch, args.worker, args.config, worker_output, task)
        entry = dict(label=label, plan_index=index, argv=argv, monotonic_start_ns=time.monotonic_ns())
        lifecycle.append(entry)
        write(args.output / 'lifecycle.json', lifecycle)
        try:
            with (args.output / (label + '.log')).open('w', encoding='utf-8') as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, shell=False)
                entry['launcher_pid'] = process.pid
                try:
                    entry['returncode'] = process.wait(timeout=bound + 30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    entry.update(returncode=process.returncode, controller_timeout=True)
            entry['monotonic_end_ns'] = time.monotonic_ns()
            if args.pull_results:
                copied = subprocess.run(['docker', 'cp', launch['container'] + ':' + worker_output,
                                         str(local_output)], capture_output=True, text=True,
                                        timeout=30, shell=False)
                entry['copy_returncode'] = copied.returncode
                C.require(copied.returncode == 0, 'worker receipt copy failed: ' + copied.stderr)
            write(args.output / 'lifecycle.json', lifecycle)
            C.require(entry.get('returncode') == 0 and not entry.get('controller_timeout'),
                      'worker failed or timed out: ' + label)
            result = audit.audit_result(json.loads((local_output / 'result.json').read_text(encoding='utf-8')))
            C.require(all(result.get(key) == task[key] for key in C.TASK_KEYS),
                      'worker receipt does not match planned task')
            if results:
                C.require(result['comparison_identity'] == results[0]['comparison_identity'],
                          'workload/interpreter/source identity changed between workers')
            results.append(result)
            write(args.output / 'status.json', dict(status='running', completed=len(results), total=len(plan), last=task))
            print(json.dumps(dict(completed=len(results), **task, metrics=result['metrics'])), flush=True)
        except BaseException as error:
            entry.setdefault('monotonic_end_ns', time.monotonic_ns())
            write(args.output / 'lifecycle.json', lifecycle)
            write(args.output / 'status.json', dict(status='failed', completed=len(results), total=len(plan),
                  last=task, error=dict(type=type(error).__name__, message=str(error))))
            raise
    summary = audit.audit_campaign(args.output)
    write(args.output / 'summary.json', summary)
    write(args.output / 'status.json', dict(status='complete', completed=len(results), total=len(plan)))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--config')
    parser.add_argument('--worker')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container-output')
    parser.add_argument('--device', choices=('cpu', 'npu'), default='npu')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--include-prepare', action='store_true')
    parser.add_argument('--pull-results', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
