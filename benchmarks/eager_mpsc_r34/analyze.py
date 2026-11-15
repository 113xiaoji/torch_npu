"""Offline stdlib audit; no worker, framework or device import is performed."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('r34_audit_contracts', Path(__file__).with_name('contracts.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def audit_result(row):
    C.require(row.get('schema') == 1 and row.get('passed') is True, 'unsuccessful worker receipt')
    kind = row.get('kind')
    C.require(kind in ('perf', 'correctness'), 'unknown receipt purpose')
    exposure = (50, 3000) if kind == 'perf' else (0, 3)
    C.require((row.get('warmup'), row.get('samples')) == exposure, 'receipt exposure differs')
    C.require(len(row.get('durations_ns', [])) == exposure[1], 'raw duration count differs')
    recomputed = C.distribution(row['durations_ns'])
    C.require(row.get('timing') == recomputed, 'reported timing differs from raw durations')
    for key in ('numeric_final', 'stage_validation'):
        C.require(row.get(key, {}).get('passed') is True, 'numerical gate failed: ' + key)
    gate = C.native_delta(row['before'], row['after'], row['arm'], exposure[1])
    close = C.validate_close(row['closed'], row['arm'])
    result = copy.deepcopy(row)
    result.update(native_timed_gate=gate, close_gate=close, timing=recomputed)
    return result


def audit_campaign(folder):
    plan = json.loads((folder / 'plan.json').read_text())
    expected = C.campaign_plan(plan['rounds'])
    C.require(plan['plan'] == expected, 'campaign plan changed')
    life = json.loads((folder / 'lifecycle.json').read_text())
    C.require(len(life) == len(expected), 'campaign is incomplete')
    rows = []
    previous_end = None
    identity = None
    for i, (entry, task) in enumerate(zip(life, expected)):
        C.require(entry.get('plan_index') == i and entry.get('returncode') == 0
                  and not entry.get('controller_timeout'), 'failed or reordered worker lifecycle')
        start, end = entry.get('monotonic_start_ns'), entry.get('monotonic_end_ns')
        C.require(type(start) is type(end) is int and start < end
                  and (previous_end is None or previous_end <= start), 'worker intervals overlap or are invalid')
        previous_end = end
        label = f"{i:03d}-{task['kind']}-r{task['round']}-{task['arm']}"
        C.require(entry.get('label') == label, 'lifecycle label differs')
        row = audit_result(json.loads((folder / label / 'result.json').read_text()))
        for key in ('kind', 'arm', 'round', 'warmup', 'samples'):
            C.require(row.get(key) == task[key], 'receipt/plan mismatch: ' + key)
        C.require(row.get('comparison_identity'), 'comparison identity is absent')
        if identity is None:
            identity = row['comparison_identity']
        C.require(row['comparison_identity'] == identity, 'candidate/workload identity changed')
        rows.append(row)
    return dict(schema=1, passed=True, worker_count=len(rows),
                timed_forward_count=sum(row['samples'] for row in rows if row['kind'] == 'perf'),
                pilot_only=plan['rounds'] % 14 != 0,
                summary=C.summarize([row for row in rows if row['kind'] == 'perf'], plan['rounds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit_campaign(args.campaign)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
