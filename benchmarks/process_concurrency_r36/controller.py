"""Exec fresh single-caller processes and synchronize their measurement window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import contracts


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def reap(children):
    for process in children:
        if process.poll() is None:
            if os.name == 'posix':
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            else: process.terminate()
    deadline = time.monotonic() + 2
    for process in children:
        try: process.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            else: process.kill()
            process.wait(timeout=5)


def wait_for_markers(output, name, children, deadline):
    paths = [output / f'rank-{rank}' / name for rank in range(len(children))]
    while not all(path.exists() for path in paths):
        failed = [(rank, child.poll()) for rank, child in enumerate(children) if child.poll() is not None]
        if failed: raise RuntimeError('worker exited before ' + name + ': ' + str(failed))
        if time.monotonic() >= deadline: raise TimeoutError('waiting for ' + name)
        time.sleep(.002)


def memory_snapshot(output, world_size):
    """Sample all setup-complete callers before release; PSS sharing is comparable."""
    rows = []
    for rank in range(world_size):
        ready = json.loads((output / f'rank-{rank}/ready.json').read_text())
        row = dict(rank=rank, pid=ready.get('pid'), sampled_ns=time.monotonic_ns())
        try:
            for line in Path(f'/proc/{row["pid"]}/smaps_rollup').read_text().splitlines():
                if line.startswith(('Rss:', 'Pss:')):
                    key, value, unit = line.split()
                    contracts.require(unit == 'kB', 'unknown memory unit')
                    row[key[:-1].lower()+'_bytes'] = int(value)*1024
            contracts.require('rss_bytes' in row and 'pss_bytes' in row, 'memory fields unavailable')
            row['available'] = True
        except (OSError, ValueError) as error:
            row.update(available=False, error=str(error))
        rows.append(row)
    available = all(row['available'] for row in rows)
    result = dict(available=available, processes=rows, approximate_sequential_snapshot=True)
    if available:
        result.update(sum_rss_bytes=sum(row['rss_bytes'] for row in rows),
                      sum_pss_bytes=sum(row['pss_bytes'] for row in rows))
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    children, logs = [], []
    receipt = dict(schema=1, passed=False, world_size=args.world_size, total_requests=args.total_requests,
                   warmup_per_process=args.warmup, controller_pid=os.getpid(), children=[])
    try:
        launch = json.loads(args.launch.read_text())
        argv_prefix = launch['argv_prefix']
        contracts.require(type(argv_prefix) is list and bool(argv_prefix)
                          and all(type(x) is str and x for x in argv_prefix), 'invalid argv_prefix')
        contracts.require(args.world_size in (1, 2, 4) and args.total_requests >= args.world_size
                          and args.warmup >= 0 and args.timeout > 0, 'invalid campaign size')
        receipt['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        receipt['launch_sha256'] = hashlib.sha256(args.launch.read_bytes()).hexdigest()
        receipt['worker_sha256'] = hashlib.sha256(args.worker.read_bytes()).hexdigest()
        for rank in range(args.world_size):
            log = (args.output / f'worker-{rank}.log').open('w')
            logs.append(log)
            argv = argv_prefix + [str(args.worker), '--config', str(args.config), '--rank', str(rank),
                '--world-size', str(args.world_size), '--total-requests', str(args.total_requests),
                '--warmup', str(args.warmup), '--output', str(args.output)]
            children.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=os.name == 'posix'))
        deadline = time.monotonic() + args.timeout
        wait_for_markers(args.output, 'ready.json', children, deadline)
        receipt['memory_at_all_ready'] = memory_snapshot(args.output, args.world_size)
        start_ns = time.monotonic_ns() + 200_000_000
        write_json(args.output / 'go.json', dict(start_ns=start_ns))
        wait_for_markers(args.output, 'done.json', children, deadline)
        # Numeric copies/hashing only begin after every process completed all device work.
        write_json(args.output / 'validate.json', dict(released_ns=time.monotonic_ns()))
        for child in children:
            child.wait(timeout=max(.01, deadline-time.monotonic()))
        contracts.require(all(child.returncode == 0 for child in children), 'worker failure')
        reports = [json.loads((args.output / f'rank-{rank}/result.json').read_text())
                   for rank in range(args.world_size)]
        receipt['metrics'] = contracts.aggregate(reports, args.world_size, args.total_requests, start_ns)
        receipt['passed'] = True
    except BaseException as error:
        receipt['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        reap(children)
        for log in logs: log.close()
        receipt['children'] = [dict(rank=i, pid=p.pid, returncode=p.returncode) for i, p in enumerate(children)]
        write_json(args.output / 'controller.json', receipt)
    return 0 if receipt['passed'] else 1


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key in ('launch', 'config', 'worker', 'output'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--world-size', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--total-requests', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=24)
    parser.add_argument('--timeout', type=float, default=600)
    return run(parser.parse_args())


if __name__ == '__main__': sys.exit(main())
