"""Adversarial stdlib tests; fixtures are not hardware qualification evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[2]


def load(name='contracts'):
    source = ROOT / 'benchmarks/forward_concurrency_r35' / (name + '.py')
    if not source.exists():
        raise AssertionError('R35 ' + name + ' is not implemented')
    spec = importlib.util.spec_from_file_location('r35_test_' + name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def receipt(*, kind='correctness', callers=2, device='cpu', gil_mode='off'):
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)]
    total = 3000 if kind == 'perf' else 32
    records = []
    for request_id in range(total):
        worker = request_id % callers
        start = 1000 + (request_id // callers) * 100 + worker * 10
        cpu = (request_id // callers) * 80 + 1
        records.append(dict(request_id=request_id, worker_index=worker, input_id=request_id % 8,
            thread_ident=worker + 10, native_thread_id=worker + 20,
            arrival_ns=start, forward_return_ns=start + 40, response_ns=start + 60,
            thread_cpu_start_ns=cpu, forward_thread_cpu_end_ns=cpu + 20,
            thread_cpu_end_ns=cpu + 30))
    workers = [dict(worker_index=i, thread_ident=i + 10, native_thread_id=i + 20,
                    completed_requests=total // callers, thread_cpu_ns=total // callers * 50)
               for i in range(callers)]
    return dict(schema=1, passed=True, kind=kind, round=0,
        case_id=f'gil{gil_mode}-c{callers}-{"cpu" if device == "cpu" else "shared"}-off',
        device=device, gil_mode=gil_mode, callers=callers,
        stream_mode='cpu' if device == 'cpu' else 'shared', backend='off',
        total_requests=total, warmup_per_caller=24 if kind == 'perf' else 0,
        requests=records, workers=workers, start_ns=900, end_ns=records[-1]['response_ns'] + 100,
        gil=dict(expected=gil_mode == 'on', before=gil_mode == 'on', after=gil_mode == 'on',
                 build_disabled=1),
        execution_device=device, actual_device_type=device,
        comparison_identity=dict(model_sha256='9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5',
            state_sha256='6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19',
            input_sha256='c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1',
            dense_sha256='352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9',
            input_hashes=hashes,
            input_bank_sha256=hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest(),
            python_executable='/free-thread/bin/python3.13', python_sha256='a' * 64,
            source_sha256='b' * 64),
        numeric_validation=dict(passed=True, validated_requests=total, finite_count=total,
            shape_count=total, reference_compared_count=total, expected_shape=[2, 1],
            max_abs_error=0.0, atol=1e-5, rtol=1e-5),
        execution_policy=dict(closed_loop=True, shared_model=True, eval_mode=True,
            independent_input_slots=True, event_completion=device == 'npu',
            device_synchronize_in_request_loop=False, outputs_validated_outside_window=True,
            prepare_multi_caller=1))


def native_contracts():
    path = ROOT / 'benchmarks/eager_mpsc_r34/contracts.py'
    spec = importlib.util.spec_from_file_location('r35_native_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_receipt(backend='off', *, split_opapi=False):
    """Actual-style structure for offline adversarial tests, never device proof."""
    row = receipt(callers=4, device='npu')
    row.update(backend=backend, case_id='giloff-c4-shared-' + backend)
    count = row['total_requests'] * 25
    def snapshot(total):
        counters = {key: total for key in ('seen', 'eligible', 'queued', 'prepared', 'executed',
                                           'returned', 'sdk_prepare_calls')}
        counters.update(fallback=0, sdk_query_calls=0, sdk_cache_hits=total)
        return dict(schema=1, enabled=True, mode='eager_prepare', ingress='mpsc' if backend == 'mpsc4' else 'mutex',
            workers=4, backend='real_sdk', real_sdk_qualified=True, simulated_prepare=False,
            qualified_ops=['aclnnEmbeddingBag'], first_error=None, diagnostic_trace=dict(enabled=False),
            operators={'aclnnEmbeddingBag': counters},
            queue=dict(admitted=total, retired=total, slots_in_use=0, retained_bytes=0,
                       closed=False, quarantined=False, workers_started=4),
            peaks=dict(slots_in_use=2, retained_bytes=4096),
            closure={key: True for key in ('admission_closed', 'fifo_drained', 'device_synchronized',
                                         'all_retired', 'workers_stopped', 'passed')},
            order_evidence=dict(enabled=True, passed=True, admitted_count=total, executed_count=total,
                                admitted_sequence_hash=123, executed_sequence_hash=123))
    row['before'], row['after'] = snapshot(0), snapshot(count)
    if backend == 'off':
        for snap in (row['before'], row['after']):
            snap.update(enabled=False, mode='off', qualified_ops=[], operators={}, real_sdk_qualified=False)
            snap['queue'].update(admitted=0, retired=0, workers_started=0)
    row['closed'] = copy.deepcopy(row['after'])
    row['closed'].update(enabled=False, mode='off')
    row['closed']['queue']['closed'] = True
    native = native_contracts()
    row['native_gate'] = native.native_delta(row['before'], row['after'], backend, row['total_requests'])
    row['close_gate'] = native.validate_close(row['closed'], backend)
    row['provider_pins'] = {
        '/private/lib/libtorch_npu.so': 'd8943e56a5b234b2dc7cb0091861087a611437661b0ddd6479d5ca610a7ccaa4',
        '/usr/local/Ascend/lib/libascendcl.so': 'c' * 64,
    }
    if split_opapi:
        row['provider_pins'].update({'/usr/local/Ascend/lib/libopapi_nn.so': 'd' * 64,
                                     '/usr/local/Ascend/lib/libopapi_math.so': 'e' * 64})
    else:
        row['provider_pins']['/usr/local/Ascend/lib/libopapi.so'] = 'd' * 64
    row['mapping_before'] = [dict(path=path, sha256=digest) for path, digest in sorted(row['provider_pins'].items())]
    row['mapping_after'] = copy.deepcopy(row['mapping_before'])
    return row


class ContractsTests(unittest.TestCase):
    def test_nearest_rank_distribution_and_exact_completed_window_throughput(self):
        c = load()
        self.assertEqual(c.distribution([40, 10, 30, 20]),
            dict(count=4, mean_ns=25.0, median_ns=20, p95_ns=40, p99_ns=40,
                 min_ns=10, max_ns=40))
        metrics = c.request_metrics(receipt())
        self.assertEqual(metrics['throughput_requests_per_second'], 32 * 1e9 / (receipt()['end_ns'] - 900))
        self.assertEqual(metrics['host_forward']['median_ns'], 40)
        self.assertEqual(metrics['end_to_end']['median_ns'], 60)
        self.assertEqual(metrics['forward_thread_cpu']['median_ns'], 20)
        self.assertEqual(metrics['response_thread_cpu']['median_ns'], 30)
        self.assertEqual(metrics['host_forward_overlap']['max_concurrency'], 2)
        self.assertIn('does not prove', metrics['host_forward_overlap']['interpretation'])

    def test_exact_unique_ids_positive_clocks_and_closed_loop_are_enforced(self):
        c = load()
        for mutate in (
            lambda r: r['requests'].pop(),
            lambda r: r['requests'][1].update(request_id=0),
            lambda r: r['requests'][0].update(forward_return_ns=1000),
            lambda r: r['requests'][0].update(response_ns=1020),
            lambda r: r['requests'][0].update(forward_thread_cpu_end_ns=1),
            lambda r: r['requests'][2].update(arrival_ns=1020),
            lambda r: r['requests'][0].update(input_id=1),
            lambda r: r['workers'][0].update(completed_requests=99),
            lambda r: r.update(end_ns=1000),
        ):
            bad = receipt()
            mutate(bad)
            with self.subTest(records=bad['requests'][:3]), self.assertRaises(ValueError):
                c.request_metrics(bad)

    def test_fixed_exposure_and_device_gil_and_numeric_identity_fail_closed(self):
        c = load()
        self.assertTrue(c.audit_result(receipt())['passed'])
        for mutate in (
            lambda r: r.update(warmup_per_caller=24),
            lambda r: r['gil'].update(before=True),
            lambda r: r['gil'].update(after=True),
            lambda r: r['gil'].update(build_disabled=0),
            lambda r: r.update(actual_device_type='npu'),
            lambda r: r['numeric_validation'].update(finite_count=0),
            lambda r: r['numeric_validation'].update(max_abs_error=float('nan')),
            lambda r: r['numeric_validation'].update(expected_shape=[]),
            lambda r: r['execution_policy'].update(device_synchronize_in_request_loop=True),
            lambda r: r['execution_policy'].update(independent_input_slots=False),
            lambda r: r['comparison_identity'].update(model_sha256='f' * 64),
            lambda r: r['comparison_identity']['input_hashes'].__setitem__(0, '0' * 64),
        ):
            bad = receipt()
            mutate(bad)
            with self.subTest(row=bad), self.assertRaises(ValueError):
                c.audit_result(bad)

    def test_npu_prepare_is_only_four_giloff_callers_on_shared_default_stream(self):
        c = load()
        row = npu_receipt('mpsc4')
        self.assertTrue(c.audit_result(row)['passed'])
        for field, value in [('stream_mode', 'private'), ('callers', 2), ('gil_mode', 'on'), ('device', 'cpu')]:
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.audit_result(bad)

    def test_npu_requires_raw_evidence_and_recomputed_native_and_close_gates(self):
        c = load()
        for backend in ('off', 'mutex4', 'mpsc4'):
            original = npu_receipt(backend)
            self.assertTrue(c.audit_result(original)['passed'])
            for missing in ('before', 'after', 'closed', 'native_gate', 'close_gate',
                            'mapping_before', 'mapping_after', 'provider_pins'):
                bad = copy.deepcopy(original)
                bad.pop(missing)
                with self.subTest(backend=backend, missing=missing), self.assertRaises(ValueError):
                    c.audit_result(bad)
            for gate in ('native_gate', 'close_gate'):
                bad = copy.deepcopy(original)
                bad[gate]['invented'] = True
                with self.subTest(gate=gate), self.assertRaises(ValueError):
                    c.audit_result(bad)

    def test_npu_sdk_admission_qualification_nonshadow_and_retirement_fail_closed(self):
        c = load()
        for mutate in (
            lambda r: r['after']['operators']['aclnnEmbeddingBag'].update(sdk_prepare_calls=0),
            lambda r: r.update(after=copy.deepcopy(r['before'])),
            lambda r: r['after'].update(real_sdk_qualified=False),
            lambda r: r['after'].update(simulated_prepare=True),
            lambda r: r['after'].update(backend='native_immediate'),
            lambda r: r['closed']['closure'].update(all_retired=False),
            lambda r: r['closed']['order_evidence'].update(executed_sequence_hash=999),
        ):
            bad = npu_receipt('mpsc4')
            mutate(bad)
            with self.subTest(snapshot=bad['after']), self.assertRaises(ValueError):
                c.audit_result(bad)
        bad = npu_receipt('off')
        bad['after']['queue']['admitted'] = 1
        with self.assertRaises(ValueError):
            c.audit_result(bad)

    def test_off_and_closed_snapshots_cannot_hide_simulation_or_lose_admissions(self):
        c = load()
        for backend in ('off', 'mpsc4'):
            for snapshot in ('before', 'after', 'closed'):
                bad = npu_receipt(backend)
                bad[snapshot]['simulated_prepare'] = True
                with self.subTest(backend=backend, snapshot=snapshot), self.assertRaises(ValueError):
                    c.audit_result(bad)
        bad = npu_receipt('mpsc4')
        bad['closed']['queue'].update(admitted=0, retired=0)
        bad['closed']['order_evidence'].update(admitted_count=0, executed_count=0)
        bad['close_gate'] = native_contracts().validate_close(bad['closed'], 'mpsc4')
        with self.assertRaises(ValueError):
            c.audit_result(bad)

    def test_provider_mapping_requires_native_d894_ascendcl_opapi_and_matching_pins(self):
        c = load()
        for mutate in (
            lambda r: r['mapping_after'].pop(),
            lambda r: r['mapping_before'][0].update(sha256='e' * 64),
            lambda r: r['provider_pins'].update({'/private/lib/libtorch_npu.so': 'e' * 64}),
            lambda r: r['provider_pins'].pop('/usr/local/Ascend/lib/libopapi.so'),
            lambda r: r['mapping_before'].append(copy.deepcopy(r['mapping_before'][0])),
        ):
            bad = npu_receipt()
            mutate(bad)
            with self.assertRaises(ValueError):
                c.audit_result(bad)
        for name in ('libtorch_npu.so', 'libascendcl.so', 'libopapi.so'):
            bad = npu_receipt()
            for key in ('mapping_before', 'mapping_after'):
                bad[key] = [entry for entry in bad[key] if not entry['path'].endswith('/' + name)]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                c.audit_result(bad)
        for path in ('/private/lib/libtorch_npu.so', '/hostsim/lib/libopapi.so'):
            bad = npu_receipt()
            if 'hostsim' in path:
                bad['provider_pins'][path] = 'e' * 64
                for key in ('mapping_before', 'mapping_after'):
                    bad[key].append(dict(path=path, sha256='e' * 64))
            else:
                bad['provider_pins'][path] = 'e' * 64
                for key in ('mapping_before', 'mapping_after'):
                    next(entry for entry in bad[key] if entry['path'] == path)['sha256'] = 'e' * 64
            with self.assertRaises(ValueError):
                c.audit_result(bad)

    def test_cpu_cannot_borrow_native_snapshots_gates_or_provider_identity(self):
        c = load()
        npu = npu_receipt()
        for key in ('before', 'after', 'closed', 'native_gate', 'close_gate',
                    'mapping_before', 'mapping_after', 'provider_pins'):
            bad = receipt()
            bad[key] = npu[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.audit_result(bad)

    def test_real_cann_split_nn_and_math_mappings_need_both_pinned_libraries(self):
        c = load()
        original = npu_receipt(split_opapi=True)
        original.update(stream_mode='private', case_id='giloff-c4-private-off')
        try:
            audited = c.audit_result(original)
        except ValueError as error:
            self.fail('legitimate CANN split providers were rejected: ' + str(error))
        self.assertTrue(audited['passed'])
        self.assertTrue(c.audit_result(npu_receipt())['passed'])
        for name in ('libopapi_nn.so', 'libopapi_math.so'):
            bad = copy.deepcopy(original)
            for key in ('mapping_before', 'mapping_after'):
                bad[key] = [entry for entry in bad[key] if not entry['path'].endswith('/' + name)]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                c.audit_result(bad)
            bad = copy.deepcopy(original)
            bad['provider_pins']['/usr/local/Ascend/lib/' + name] = 'f' * 64
            with self.subTest(changed_pin=name), self.assertRaises(ValueError):
                c.audit_result(bad)

    def test_comparison_rejects_cpu_npu_identity_drift_and_correctness_as_perf(self):
        c = load()
        rows = [receipt(kind='perf', callers=i, gil_mode=g) for g in ('off', 'on') for i in (1, 2, 4)]
        result = c.summarize(rows, device='cpu', rounds=1)
        self.assertEqual(len(result['cases']), 6)
        self.assertEqual(result['performance_request_count'], 18000)
        for mutate in (
            lambda r: r[0].update(kind='correctness'),
            lambda r: r[0]['comparison_identity'].update(python_sha256='c' * 64),
            lambda r: r[0].update(device='npu'),
            lambda r: r.pop(),
        ):
            bad = copy.deepcopy(rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                c.summarize(bad, device='cpu', rounds=1)


if __name__ == '__main__':
    unittest.main()
