"""Offline receipt audit; importing this module never loads torch or an NPU SDK."""
import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('r35_audit_contracts', Path(__file__).with_name('contracts.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
audit_result = C.audit_result


def receipt_label(index, task):
    return f"{index:03d}-{task['kind']}-r{task['round']}-{task['case_id']}"


def audit_campaign(folder):
    folder = Path(folder)
    recorded = json.loads((folder / 'plan.json').read_text())
    require = C.require
    require(recorded.get('schema') == 1, 'unknown campaign schema')
    expected = C.campaign_plan(recorded['device'], recorded['rounds'], recorded['include_prepare'])
    require(recorded.get('plan') == expected, 'campaign plan differs')
    lifecycle = json.loads((folder / 'lifecycle.json').read_text())
    require(type(lifecycle) is list and len(lifecycle) == len(expected), 'campaign is incomplete')
    previous_end, pids, rows, identity = None, set(), [], None
    for index, (entry, task) in enumerate(zip(lifecycle, expected)):
        label = receipt_label(index, task)
        require(entry.get('plan_index') == index and entry.get('label') == label
                and entry.get('returncode') == 0 and not entry.get('controller_timeout'),
                'failed/reordered worker lifecycle')
        start, end = entry.get('monotonic_start_ns'), entry.get('monotonic_end_ns')
        require(type(start) is type(end) is int and start < end
                and (previous_end is None or previous_end <= start), 'workers overlap or clocks invalid')
        previous_end = end
        pid = entry.get('launcher_pid')
        require(type(pid) is int and pid > 0 and pid not in pids, 'worker requires a fresh process identity')
        pids.add(pid)
        row = audit_result(json.loads((folder / label / 'result.json').read_text()))
        require(all(row.get(key) == task[key] for key in C.TASK_KEYS), 'receipt does not match planned task')
        current = row['comparison_identity']
        if identity is None:
            identity = current
        require(current == identity, 'identity drift between correctness/performance workers')
        rows.append(row)
    summary = C.summarize([row for row in rows if row['kind'] == 'perf'],
                          recorded['device'], recorded['rounds'], recorded['include_prepare'])
    return dict(schema=1, passed=True, device=recorded['device'], worker_count=len(rows), summary=summary)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_campaign(args.campaign)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
