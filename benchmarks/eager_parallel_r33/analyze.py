"""Exclusive host-wall accounting and separate device busy-interval accounting.

Device activity overlaps host work. It is never added to the host partition.
The timestamp parser preserves nanoseconds in epoch-scale microsecond strings.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import statistics


def ns(microseconds):
    return int(Decimal(str(microseconds))*1000)


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def clip(intervals, start, end):
    return merge((max(a, start), min(b, end)) for a, b in intervals)


def duration(intervals):
    return sum(b-a for a, b in merge(intervals))


def intersection(left, right):
    left, right = merge(left), merge(right)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            out.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def device_window(intervals, start, middle, end):
    parts = [(max(a, start), min(b, end)) for a, b in intervals if a < end and b > start]
    union = merge(parts)
    total, busy = sum(b-a for a, b in parts), duration(union)
    return dict(kernel_sum_ns=total, kernel_union_ns=busy, kernel_overlap_ns=total-busy,
                kernel_forward_ns=duration(clip(union, start, middle)),
                kernel_final_sync_ns=duration(clip(union, middle, end)),
                wall_ns=end-start, compute_busy_share=busy/(end-start))


def host_window(window):
    n = window['samples']
    assert len(window['rows']) == n
    forward = sum(r['forward_ns'] for r in window['rows'])/n
    final = sum(r['final_sync_ns'] for r in window['rows'])/n
    total = sum(r['e2e_ns'] for r in window['rows'])/n
    assert all(r['forward_ns']+r['final_sync_ns'] == r['e2e_ns'] for r in window['rows'])
    groups = {phase: defaultdict(float) for phase in ('forward', 'final_sync')}
    for phase, rows in window['api_counters'].items():
        for row in rows:
            groups[phase][row['category']] += row['exclusive_ns']/n
    event_forward = groups['forward']['event']
    event_final = groups['final_sync']['event']
    internal = sum(groups['forward'][k] for k in ('sync', 'poll_sleep', 'copy_sync'))
    prepare = forward-event_forward-internal
    known = sum(groups['forward'][k] for k in ('workspace', 'descriptor', 'launch', 'stream', 'copy_async'))
    assert prepare >= 0 and final >= event_final
    return dict(host_prepare_submit_ns=prepare, event_ns=event_forward+event_final,
                internal_sync_ns=internal, final_sync_ns=final-event_final, e2e_ns=total,
                host_other_ns=prepare-known, sdk_launch_ns=groups['forward']['launch'],
                sdk_workspace_ns=groups['forward']['workspace'],
                sdk_descriptor_ns=groups['forward']['descriptor'],
                stream_api_ns=groups['forward']['stream'], poll_sleep_ns=groups['forward']['poll_sleep'],
                copy_sync_ns=groups['forward']['copy_sync'],
                partition_sum_ns=prepare+event_forward+event_final+internal+final-event_final,
                categories_ns={p: dict(g) for p, g in groups.items()})


COMPUTE = {'AI_CORE', 'AI_VECTOR_CORE', 'AI_CPU', 'MIX_AIC', 'MIX_AIV', 'MIX_AICPU', 'FFTS_PLUS',
           'KERNEL_AICORE', 'KERNEL_AIVEC', 'KERNEL_AICPU'}
NONCOMPUTE = {'PROFILING_ENABLE', 'PROFILING_DISABLE', 'EVENT_RECORD', 'EVENT_WAIT',
              'NOTIFY_RECORD', 'NOTIFY_WAIT', 'MEMCPY_ASYNC', 'MEMCPY', 'SDMA', 'WRITE_VALUE',
              'WAIT_VALUE', 'MAINTENANCE', 'STREAM_ACTIVE', 'STREAM_SWITCH'}


def kernels(events):
    hardware = {r['pid'] for r in events if r.get('ph') == 'M' and r.get('name') == 'process_name'
                and r.get('args', {}).get('name') == 'Ascend Hardware'}
    if not hardware:
        raise ValueError('actual Ascend Hardware process missing')
    result = []
    for r in events:
        if r.get('ph') != 'X' or r.get('pid') not in hardware:
            continue
        task = r.get('args', {}).get('Task Type', '')
        if task in COMPUTE:
            result.append(r)
        elif task not in NONCOMPUTE:
            raise ValueError('unknown hardware task type: '+repr(task))
    return result


def device_groups(device, count, expected_count=145):
    # A final caller-stream Sigmoid terminates each frozen DLRM forward after
    # both producer streams have joined. Consecutive frames are synchronized.
    # Group on device order, not an assumed zero host/device clock offset.
    anchors = sorted(ns(r['ts'])+ns(r['dur']) for r in device
                     if r['name'].startswith('aclnnSigmoid_'))
    if len(anchors) != count:
        raise ValueError('final Sigmoid count differs from synchronized forwards')
    groups = [[] for _ in anchors]
    for r in sorted(device, key=lambda x: ns(x['ts'])):
        end = ns(r['ts'])+ns(r['dur'])
        index = bisect_left(anchors, end)
        if index == count:
            raise ValueError('compute after last final Sigmoid')
        if index and ns(r['ts']) < anchors[index-1]:
            raise ValueError('compute crosses previous final Sigmoid')
        groups[index].append(r)
    if any(len(g) != expected_count for g in groups):
        raise ValueError('kernel completeness differs: '+str(Counter(map(len, groups))))
    return groups


def analyze_device(result, trace):
    data = json.loads(Path(trace).read_text(), parse_float=Decimal)
    events = data.get('traceEvents', []) if isinstance(data, dict) else data
    device = kernels(events)
    groups = device_groups(device, len(result['measurement']['rows']))
    bounds = []
    for sample, group in zip(result['measurement']['rows'], groups):
        begin = min(ns(r['ts']) for r in group)
        end = max(ns(r['ts'])+ns(r['dur']) for r in group)
        bounds.append((sample['realtime_start_ns']-begin, sample['realtime_end_ns']-end))
    lower, upper = max(x[0] for x in bounds), min(x[1] for x in bounds)
    if lower > upper:
        raise ValueError('no feasible constant host/device clock alignment')
    rows = []
    for sample, group in zip(result['measurement']['rows'], groups):
        start, middle, end = (sample[k] for k in ('realtime_start_ns', 'realtime_middle_ns', 'realtime_end_ns'))
        members = [(r, ns(r['ts']), ns(r['ts'])+ns(r['dur'])) for r in group]
        spans = [(a, b) for _, a, b in members]
        merged = merge(spans)
        total, busy = sum(b-a for a, b in spans), duration(merged)
        tail_lower = duration(intersection([(a+lower, b+lower) for a, b in merged], [(middle, end)]))
        tail_upper = duration(intersection([(a+upper, b+upper) for a, b in merged], [(middle, end)]))
        row = dict(kernel_sum_ns=total, kernel_union_ns=busy, kernel_overlap_ns=total-busy,
                   kernel_forward_ns_lower=busy-tail_upper, kernel_forward_ns_upper=busy-tail_lower,
                   kernel_final_sync_ns_lower=tail_lower, kernel_final_sync_ns_upper=tail_upper,
                   wall_ns=end-start, compute_busy_share=busy/(end-start))
        counts = Counter(r['name'] for r, _, _ in members)
        times = defaultdict(int)
        streams = defaultdict(set)
        for r, a, b in members:
            name = r['name']
            family = ('embedding_cast' if 'aclnnEmbeddingBag_Cast' in name else
                      'embedding' if name.startswith('aclnnEmbeddingBag_') else
                      'interaction' if name.startswith(('aclnnCat_', 'aclnnIndex_', 'aclnnBatchMatMul_')) else
                      'zero' if name.startswith('aclnnInplaceZero_') else 'mlp')
            times[family] += b-a
            streams[family].add(r['tid'])
        assert sum(v for k, v in counts.items() if 'aclnnEmbeddingBag_Cast' in k) == 100
        row.update(iteration=sample['iteration'], kernel_count=len(members), counts=dict(counts),
                   family_sum_ns=dict(times), streams={k: sorted(v) for k, v in streams.items()},
                   first_kernel_delay_ns_lower=min(a for _, a, _ in members)+lower-start,
                   last_kernel_to_forward_return_ns_lower=middle-max(b for _, _, b in members)-upper,
                   realtime_perf_difference_ns=(end-start)-sample['e2e_ns'])
        rows.append(row)
    fields = ['kernel_sum_ns', 'kernel_union_ns', 'kernel_overlap_ns', 'kernel_forward_ns_lower',
              'kernel_forward_ns_upper', 'kernel_final_sync_ns_lower', 'kernel_final_sync_ns_upper',
              'wall_ns', 'compute_busy_share', 'first_kernel_delay_ns_lower',
              'last_kernel_to_forward_return_ns_lower', 'realtime_perf_difference_ns']
    return dict(rows=rows, means={k: statistics.mean(r[k] for r in rows) for k in fields},
                grouping='device-order final Sigmoid fence; 145 kernels per synchronized forward',
                clock_offset_bounds_ns=dict(lower=lower, upper=upper,
                    definition='device timestamp + offset is host epoch; causal bounds, not a fitted offset'),
                family_means_ns={k: statistics.mean(r['family_sum_ns'].get(k, 0) for r in rows)
                                 for k in rows[0]['family_sum_ns']},
                profiled_e2e_ns=result['measurement']['timing']['e2e_ns']['mean_ns'],
                control_before_ns=result['control_before']['timing']['e2e_ns']['mean_ns'],
                control_after_ns=result['control_after']['timing']['e2e_ns']['mean_ns'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True, type=Path)
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = json.loads(args.result.read_text())
    assert result['passed'] and result['kind'] == 'trace'
    analysis = analyze_device(result, args.trace)
    args.output.write_text(json.dumps(analysis, indent=2)+'\n')
    print(json.dumps(analysis['means']), flush=True)


if __name__ == '__main__':
    main()
