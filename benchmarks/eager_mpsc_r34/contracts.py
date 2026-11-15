"""Standard-library gates shared by the real worker and offline audit."""
import math
import statistics

ARMS = ('off', 'mutex1', 'mutex2', 'mutex4', 'mpsc1', 'mpsc2', 'mpsc4')
IDENTITY = dict(model_sha256='9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5',
                state_sha256='6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19',
                input_sha256='c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1',
                dense_sha256='352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_policy(arm):
    require(arm in ARMS, 'unknown R34 arm')
    return dict(enabled=arm != 'off', ingress='mpsc' if arm.startswith('mpsc') else 'mutex',
                workers=int(arm[-1]) if arm != 'off' else 1)


def integer(row, key):
    value = row.get(key)
    require(type(value) is int and value >= 0, 'missing/invalid native counter: ' + key)
    return value


def delta(previous, current, key):
    value = integer(current, key) - integer(previous, key)
    require(value >= 0, 'native counter rolled back: ' + key)
    return value


def off_inactive(row):
    require(row.get('schema') == 1 and row.get('backend') == 'real_sdk'
            and row.get('mode') == 'off' and row.get('enabled') is False
            and row.get('first_error') is None and not row.get('qualified_ops')
            and row.get('operators') == {}, 'off must remain the inactive real PTA route')
    queue = row.get('queue', {})
    require(queue.get('quarantined') is False, 'off queue is quarantined')
    for key in ('admitted', 'retired', 'slots_in_use', 'retained_bytes', 'workers_started'):
        require(integer(queue, key) == 0, 'off unexpectedly started Prepare work: ' + key)


def native_delta(before, after, arm, forwards, expected_per_forward=25):
    policy = arm_policy(arm)
    require(type(forwards) is int and forwards > 0, 'positive exact forward count required')
    if not policy['enabled']:
        off_inactive(before)
        off_inactive(after)
        return dict(passed=True, queued=0, sdk_prepare_calls=0, scope='original PTA TQ1')
    for row in (before, after):
        require(row.get('schema') == 1 and row.get('mode') == 'eager_prepare'
                and row.get('enabled') is True and row.get('backend') == 'real_sdk'
                and row.get('real_sdk_qualified') is True and row.get('simulated_prepare') is False
                and row.get('first_error') is None, 'real SDK eager Prepare identity/qualification failed')
        require(row.get('ingress') == policy['ingress'] and row.get('workers') == policy['workers'],
                'native ingress/worker configuration differs')
        require(row.get('diagnostic_trace', {}).get('enabled') is False, 'formal trace must be off')
        require('aclnnEmbeddingBag' in row.get('qualified_ops', []), 'EmbeddingBag SDK is unqualified')
        require(row.get('queue', {}).get('quarantined') is False, 'native domain is quarantined')
        queue = row['queue']
        require(integer(queue, 'slots_in_use') == integer(queue, 'retained_bytes') == 0
                and integer(queue, 'admitted') == integer(queue, 'retired'),
                'synchronized measurement snapshot still retains native owners')
    require(type(expected_per_forward) is int and expected_per_forward > 0, 'positive admission target required')
    old_ops, new_ops = before.get('operators', {}), after.get('operators', {})
    keys = ('seen', 'eligible', 'queued', 'prepared', 'executed', 'returned', 'fallback')
    operators = {}
    for name in sorted(set(old_ops) | set(new_ops)):
        old = old_ops.get(name, {key: 0 for key in keys})
        new = new_ops.get(name, {})
        values = {key: delta(old, new, key) for key in keys}
        require(values['seen'] == values['eligible'] + values['fallback'], 'incomplete native route accounting')
        require(values['eligible'] == values['queued'] == values['prepared'] == values['executed'] == values['returned'],
                'missing Prepare/Execute/return acknowledgement')
        if values['queued']:
            values['sdk_prepare_calls'] = delta(
                dict(sdk_prepare_calls=old.get('sdk_prepare_calls', 0)), new, 'sdk_prepare_calls')
            require(values['sdk_prepare_calls'] == values['prepared'] > 0, 'real SDK preparation was not observed')
        else:
            values['sdk_prepare_calls'] = 0
        for optional in ('sdk_query_calls', 'sdk_cache_hits'):
            values[optional] = (delta({optional: old.get(optional, 0)}, new, optional)
                                if optional in new else None)
        values['fallback_reasons'] = {}
        for reason in set(old.get('fallback_reasons', {})) | set(new.get('fallback_reasons', {})):
            values['fallback_reasons'][reason] = delta(
                {reason: old.get('fallback_reasons', {}).get(reason, 0)},
                new.get('fallback_reasons', {}), reason)
        operators[name] = values
    embedding = operators.get('aclnnEmbeddingBag', {})
    require(embedding.get('queued', 0) == forwards * expected_per_forward,
            'actual EmbeddingBag admission differs from frozen per-forward target')
    queued = sum(item['queued'] for item in operators.values())
    require(queued > 0, 'zero actual SDK admission')
    require(delta(before['queue'], after['queue'], 'admitted') == queued, 'core admission count differs')
    require(delta(before['queue'], after['queue'], 'retired') == queued, 'device retirement ACK count differs')
    require(integer(after['queue'], 'workers_started') == policy['workers'], 'native workers did not start')
    return dict(passed=True, queued=queued,
                sdk_prepare_calls=sum(item['sdk_prepare_calls'] for item in operators.values()),
                sdk_query_calls=embedding.get('sdk_query_calls'), sdk_cache_hits=embedding.get('sdk_cache_hits'),
                operators=operators, model_core_coverage='partial',
                scope='real owned EmbeddingBag Prepare through original PTA FIFO; fallbacks included in E2E')


def validate_close(row, arm, capacity=8, byte_limit=536870912):
    if arm == 'off':
        off_inactive(row)
        return dict(passed=True, scope='inactive Prepare; original synchronization completed')
    policy = arm_policy(arm)
    require(row.get('first_error') is None and row.get('enabled') is False,
            'native close failed or admission remains enabled')
    require(row.get('ingress') == policy['ingress'], 'close ingress differs')
    queue = row.get('queue', {})
    require(queue.get('closed') is True and queue.get('quarantined') is False
            and integer(queue, 'slots_in_use') == integer(queue, 'retained_bytes') == 0
            and integer(queue, 'admitted') == integer(queue, 'retired'), 'native owners have not retired')
    for key in ('admission_closed', 'fifo_drained', 'device_synchronized', 'all_retired', 'workers_stopped', 'passed'):
        require(row.get('closure', {}).get(key) is True, 'native close ACK missing: ' + key)
    peaks = row.get('peaks', {})
    require(integer(peaks, 'slots_in_use') <= capacity and integer(peaks, 'retained_bytes') <= byte_limit,
            'reservation budget was exceeded')
    order = row.get('order_evidence', {})
    require(order.get('enabled') is True and order.get('passed') is True
            and integer(order, 'admitted_count') == integer(order, 'executed_count') == integer(queue, 'admitted')
            and integer(order, 'admitted_sequence_hash') == integer(order, 'executed_sequence_hash'),
            'real FIFO publication/execution order evidence is incomplete')
    return dict(passed=True, admitted=queue['admitted'], order_evidence=order)


def campaign_plan(rounds=3):
    require(type(rounds) is int and 1 <= rounds <= 140, 'round count must be 1..140')
    # Seven-arm Williams sequences: each position and directed predecessor pair
    # occurs twice in a complete 14-round cycle. A 3-round pilot is not balanced.
    base = (0, 1, 6, 2, 5, 3, 4)
    orders = [tuple(ARMS[(i + shift) % 7] for i in base) for shift in range(7)]
    orders += [tuple(reversed(order)) for order in orders]
    result = [dict(kind='correctness', round=0, arm=arm, warmup=0, samples=3) for arm in ARMS]
    result += [dict(kind='perf', round=i, arm=arm, warmup=50, samples=3000)
               for i in range(rounds) for arm in orders[i % 14]]
    return result


def distribution(values):
    require(type(values) is list and values and all(type(v) is int and v > 0 for v in values),
            'positive integer raw durations required')
    ordered = sorted(values)
    return dict(count=len(values), mean_ns=statistics.mean(values), median_ns=statistics.median(values),
                p95_ns=ordered[math.ceil(.95 * len(values)) - 1],
                p99_ns=ordered[math.ceil(.99 * len(values)) - 1])


def summarize(rows, rounds):
    index = {}
    identity = None
    for row in rows:
        require(row.get('passed') is True and row.get('kind') == 'perf'
                and row.get('arm') in ARMS and type(row.get('round')) is int
                and 0 <= row['round'] < rounds, 'invalid performance receipt')
        require(row.get('warmup') == 50 and row.get('samples') == 3000
                and len(row.get('durations_ns', [])) == 3000
                and integer(row, 'window_wall_ns') >= 2_000_000_000, 'fixed performance exposure differs')
        for key in ('native_timed_gate', 'close_gate', 'numeric_final', 'stage_validation'):
            require(row.get(key, {}).get('passed') is True, 'required acceptance gate failed: ' + key)
        require(row.get('comparison_identity'), 'comparison identity is absent')
        if identity is None:
            identity = row['comparison_identity']
        require(row['comparison_identity'] == identity, 'comparison identity changed across arms')
        key = row['round'], row['arm']
        require(key not in index, 'duplicate performance receipt')
        index[key] = dict(distribution(row['durations_ns']),
                          process_cpu_per_forward_ns=integer(row, 'window_process_cpu_ns') / 3000)
    require(len(index) == rounds * len(ARMS)
            and all((i, arm) in index for i in range(rounds) for arm in ARMS), 'missing performance arm/round')
    result = {}
    for arm in ARMS:
        changes = [(index[i, arm]['median_ns'] / index[i, 'off']['median_ns'] - 1) * 100 for i in range(rounds)]
        item = dict(rounds=rounds, paired_vs_off_pct=changes,
                    median_paired_vs_off_pct=statistics.median(changes),
                    faster_vs_off_rounds=sum(v < 0 for v in changes),
                    windows=[index[i, arm] for i in range(rounds)],
                    formal_balance=rounds % 14 == 0, pilot_only=rounds % 14 != 0)
        if arm.startswith('mpsc'):
            paired = [(index[i, arm]['median_ns'] / index[i, 'mutex' + arm[-1]]['median_ns'] - 1) * 100
                      for i in range(rounds)]
            item.update(paired_vs_mutex_pct=paired, median_paired_vs_mutex_pct=statistics.median(paired))
        result[arm] = item
    return result
