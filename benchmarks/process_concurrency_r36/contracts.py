"""Offline full-request process aggregation; no framework imports."""
import math


def require(value, message):
    if not value:
        raise ValueError(message)


def quantiles(values):
    rows = sorted(values)
    require(bool(rows), 'empty observations')
    return {name: rows[math.ceil(len(rows) * q) - 1] / 1e6
            for name, q in (('p50', .5), ('p95', .95), ('p99', .99))}


def aggregate(reports, world_size, total_requests, start_ns):
    require(world_size in (1, 2, 4) and total_requests >= world_size, 'invalid process workload')
    require(len(reports) == world_size and {r['rank'] for r in reports} == set(range(world_size)),
            'missing or duplicate process')
    requests = []
    for report in reports:
        for key in ('device', 'runtime', 'config_sha256', 'worker_sha256', 'comparison_identity', 'source_pins'):
            require(key in report and report[key] == reports[0].get(key), 'process identity differs: ' + key)
        require(report.get('gil_after') is True and report['runtime'].get('gil_before') is True,
                'GIL mode validation missing')
        require(report.get('original_full_model_executed') is True, 'full-model execution missing')
        require(report.get('passed') is True and report.get('shared_weights_mutated') is False,
                'failed process or mutated weights')
        require(report['world_size'] == world_size and report['total_requests'] == total_requests
                and report['start_ns'] == start_ns, 'inconsistent process campaign')
        expected = list(range(report['rank'], total_requests, world_size))
        rows = report['requests']
        require([r['request_id'] for r in rows] == expected, 'request partition mismatch')
        numeric = report['numeric_validation']
        require(numeric.get('passed') is True and numeric.get('validated_requests') == len(expected),
                'numeric validation incomplete')
        for row in rows:
            require(row['input_id'] == row['request_id'] % 8, 'request input mismatch')
            require(start_ns <= row['arrival_ns'] <= row['forward_return_ns'] <= row['response_ns'],
                    'invalid request interval')
        requests.extend(rows)
    require(sorted(r['request_id'] for r in requests) == list(range(total_requests)),
            'global requests incomplete')
    end_ns = max(r['response_ns'] for r in requests)
    require(end_ns > start_ns, 'empty completion window')
    return dict(completed_requests=total_requests, start_ns=start_ns, end_ns=end_ns,
                window_seconds=(end_ns-start_ns)/1e9,
                requests_per_second=total_requests * 1e9 / (end_ns-start_ns),
                response_ms=quantiles([r['response_ns']-r['arrival_ns'] for r in requests]),
                forward_ms=quantiles([r['forward_return_ns']-r['arrival_ns'] for r in requests]))
