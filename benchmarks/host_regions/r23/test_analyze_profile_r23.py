"""Synthetic host interval semantics only; no measured SDK/device claims."""
import copy
import contextlib
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('analyze_profile_r23.py')


def module():
    if not PATH.is_file(): raise AssertionError('offline profile analyzer missing')
    spec = importlib.util.spec_from_file_location('r23_analyzer_test', PATH)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def event(name, start, end, tid=1, parent=None, identity=None):
    return dict(id=identity or name, parent=parent, request=1, name=name, native_tid=tid,
                start_ns=start, end_ns=end, start_thread_time_ns=start, end_thread_time_ns=end)


class AnalysisTests(unittest.TestCase):
    def headers(self, m):
        raw = dict(passed=True, phase='complete', kind='perf', abi='std', arm='serial',
                   script_sha256=m.RUNNER_SHA, forward_calls=2021,
                   timing=dict(samples=2000, requested_samples=2000, warmup=20, durations_ns=[130]*2000),
                   requested_wheel=dict(sha256='a'*64), wheel=dict(sha256='a'*64), native_sha256='b'*64)
        argv = ['--abi', 'std', '--arm', 'serial', '--kind', 'perf', '--warmup', '20', '--iterations', '2000',
                '--expected-wheel-sha256', 'a'*64, '--expected-native-sha256', 'b'*64]
        profile = dict(schema=1, runner=dict(sha256=m.RUNNER_SHA), tool_sha256=m.HARNESS_SHA,
                       profile_mode='stages', diagnostic_only=True, performance_conclusion_allowed=False,
                       instrumentation_complete=True, temporary_methods_restored=True, original_exit_code=0,
                       failure=None, requests=2000, restoration=[dict(restored=True)], original_argv=argv)
        return raw, profile

    def full_fixture(self, m):
        raw, profile = self.headers(m); events = []
        for request in range(1, 2001):
            for index, row in enumerate(self.rows(), 1):
                row.update(id=f'1:{request*10+index}', request=request)
                for key in ('start_ns', 'end_ns', 'start_thread_time_ns', 'end_thread_time_ns'):
                    row[key] += request*500
                events.append(row)
        counts = {name: 2000 for name in ('forward', 'tail_sync', *m.MODEL.values())}
        profile.update(events=events, actual_event_counts=counts, expected_event_counts=counts)
        return raw, profile

    def worker_fixture(self, m):
        raw, profile = self.full_fixture(m); raw['arm'] = 'worker4'
        raw['timing']['durations_ns'] = [13000]*2000
        profile['original_argv'][profile['original_argv'].index('--arm')+1] = 'worker4'
        events = profile['events']
        for row in events:
            for key in ('start_ns', 'end_ns', 'start_thread_time_ns', 'end_thread_time_ns'): row[key] *= 100
        caller = ('runtime.map', 'runtime.native_parallel', 'runtime.prepare_executor', 'provider.guard',
                  'policy.check', 'session.open', 'session.drain', 'session.close')
        for req in range(1, 2001):
            offset = req*50000
            def add(name, start, end, tid=1, parent=None, identity=None, **details):
                row = event(name, offset+start, offset+end, tid, parent, identity or f'{tid}:{req}:{name}')
                row.update(request=req, **details); events.append(row); return row['id']
            for i, name in enumerate(caller): add(name, 2500+i*10, 2505+i*10)
            for i in range(4): add('executor.submit', 2700+i*10, 2705+i*10, identity=f'1:{req}:submit{i}')
            for region, (start, stop) in enumerate(((0, 6), (6, 13), (13, 19), (19, 26))):
                tid = region+2
                parent = add('worker.region', 2900, 3990, tid, region=region, first=start, stop=stop)
                for name, left, right in (('session.begin', 2910, 2920), ('session.end', 3980, 3985), ('session.done', 3986, 3989)):
                    add(name, left, right, tid, parent)
                for item in range(start, stop):
                    add('region.body', 3000+(item-start)*100, 3010+(item-start)*100, tid, parent,
                        identity=f'{tid}:{req}:item{item}', region=region, item=item)
        profile['actual_event_counts'] = dict(Counter(row['name'] for row in events))
        profile['expected_event_counts'].update({name: 2000 for name in caller})
        profile['expected_event_counts'].update({'worker.region': 8000, 'executor.submit': 8000, 'region.body': 52000})
        return raw, profile

    def rows(self):
        return [event('forward', 10, 90), event('model.bottom_mlp', 12, 22),
                event('model.apply_emb', 24, 44), event('model.interact_features', 50, 60),
                event('model.top_mlp', 70, 80), event('tail_sync', 100, 120)]

    def test_partition_is_nonnegative_and_exactly_conserves_raw_duration(self):
        row = module().partition_request(self.rows(), 130)
        self.assertEqual(row['stages_ns'], dict(bottom=10, emb=20, interaction=10, top=10,
                                               **{'forward.misc': 30, 'tail_sync': 20, 'outer.misc': 30}))
        self.assertEqual(sum(row['stages_ns'].values()), 130)

    def test_overlapping_model_stages_are_rejected(self):
        rows = self.rows(); rows[2]['start_ns'] = 20
        with self.assertRaisesRegex(ValueError, 'overlap'):
            module().partition_request(rows, 130)

    def test_negative_outer_or_invalid_envelope_is_rejected(self):
        with self.assertRaises(ValueError): module().partition_request(self.rows(), 100)

    def test_cross_tid_model_stage_is_not_subtracted_from_caller(self):
        rows = self.rows(); rows[2]['native_tid'] = 2
        with self.assertRaisesRegex(ValueError, 'caller TID'):
            module().partition_request(rows, 130)

    def test_worker_overlap_is_separate_from_host_partition(self):
        m = module(); rows = self.rows() + [event('worker.region', 25, 40, tid=2),
                                           event('worker.region', 30, 50, tid=3)]
        row = m.partition_request(rows, 130)
        self.assertEqual(sum(row['stages_ns'].values()), 130)
        workers = m.worker_metrics(rows)
        self.assertEqual(workers['span_ns'], 25); self.assertEqual(workers['union_ns'], 25)
        self.assertEqual(workers['thread_cpu_ns'], 35)

    def test_cross_tid_parent_is_rejected_instead_of_deducted(self):
        rows = [event('forward', 10, 90, identity='1:1'),
                event('worker.region', 20, 30, tid=2, parent='1:1', identity='2:1')]
        with self.assertRaisesRegex(ValueError, 'cross-TID parent'):
            module().validate_events(rows)

    def test_nearest_rank_percentiles_are_in_nanoseconds(self):
        result = module().distribution([1, 2, 3, 4, 100])
        self.assertEqual(result['mean_ns'], 22); self.assertEqual(result['p50_ns'], 3)
        self.assertEqual(result['p95_ns'], 100); self.assertEqual(result['p99_ns'], 100)

    def test_header_gate_never_accepts_incomplete_or_unrestored_profile(self):
        m = module(); raw, profile = self.headers(m)
        self.assertEqual(len(m.validate_headers(raw, profile)), 2000)
        for key, value in (('instrumentation_complete', False), ('temporary_methods_restored', False),
                           ('requests', 1999), ('tool_sha256', 'c'*64)):
            altered = copy.deepcopy(profile); altered[key] = value
            with self.assertRaises(ValueError): m.validate_headers(raw, altered)

    def test_fixed_2000_request_fixture_is_fully_conserved(self):
        m = module(); raw, profile = self.full_fixture(m)
        result = m.analyze(raw, profile)
        self.assertTrue(result['passed']); self.assertEqual(len(result['requests']), 2000)
        self.assertTrue(all(sum(row['stages_ns'].values()) == row['raw_ns'] for row in result['requests']))
        self.assertIsNone(result['native_measurements']['device_ns'])

    def test_input_sha_binds_the_bytes_actually_parsed(self):
        m = module(); raw, profile = self.full_fixture(m)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); rp = folder/'raw.json'; pp = folder/'profile.json'; out = folder/'analysis.json'
            rp.write_text(json.dumps(raw)); pp.write_text(json.dumps(profile))
            different = copy.deepcopy(raw); different['timing']['durations_ns'] = [140]*2000
            original_read = Path.read_text
            def split_read(path, *args, **kwargs):
                return json.dumps(different) if path == rp else original_read(path, *args, **kwargs)
            with patch.object(Path, 'read_text', split_read), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(['--raw', str(rp), '--profile', str(pp), '--output', str(out)]), 0)
            answer = json.loads(out.read_text())
            self.assertEqual(answer['inputs']['raw']['sha256'], m.sha(rp))
            self.assertEqual(answer['requests'][0]['raw_ns'], 130)

    def test_a_missing_request_drain_cannot_be_hidden_by_global_count(self):
        m = module(); raw, profile = self.worker_fixture(m)
        drain = next(row for row in profile['events'] if row['request'] == 1 and row['name'] == 'session.drain')
        drain['request'] = 2  # Global count and all source/header fields remain unchanged.
        with self.assertRaisesRegex(ValueError, 'per-request caller'):
            m.analyze(raw, profile)

    def test_complete_fixed_worker_fixture_is_admitted_without_worker_wall_addition(self):
        m = module(); raw, profile = self.worker_fixture(m); answer = m.analyze(raw, profile)
        self.assertTrue(answer['passed']); self.assertEqual(len(answer['requests']), 2000)
        self.assertEqual(answer['requests'][0]['worker']['thread_cpu_ns'], 4360)
        self.assertEqual(sum(answer['requests'][0]['stages_ns'].values()), 13000)

    def test_worker_tree_after_its_request_embedding_is_rejected(self):
        m = module(); raw, profile = self.worker_fixture(m)
        for row in profile['events']:
            if row['request'] == 1 and row['native_tid'] == 2:
                for key in ('start_ns', 'end_ns', 'start_thread_time_ns', 'end_thread_time_ns'): row[key] += 10000
        with self.assertRaisesRegex(ValueError, 'worker.*embedding'):
            m.analyze(raw, profile)


if __name__ == '__main__': unittest.main()
