"""Correlate the unchanged frozen runner's common perf window with native clocks.

This diagnostic changes only the runner module's temporary time binding. The
original perf_counter_ns value is read first and returned without modification.
Correlation clocks are subsequent sequential reads, not simultaneous timestamps.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import uuid


RUNNER_SHA = 'a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f'
SAMPLES = 2000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_identity(proc_root=Path('/proc')):
    """Observe this application before its original main, outside timing."""
    pid = os.getpid()
    unsupported = dict(pid=pid, pgid=None, host_boot_id=None, starttime_ticks=None,
        proc_identity_observed=False)
    if sys.platform != 'linux':
        return unsupported
    try:
        stat = (Path(proc_root) / 'self/stat').read_text(encoding='ascii')
        head, tail = stat.rsplit(')', 1)
        fields = tail.split()
        observed_pid = int(head.split('(', 1)[0])
        pgid, starttime = int(fields[2]), int(fields[19])
        boot = (Path(proc_root) / 'sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
        require(observed_pid == pid and pgid > 0 and starttime > 0
            and str(uuid.UUID(boot)) == boot, 'application /proc birth identity is invalid')
        return dict(pid=pid, pgid=pgid, host_boot_id=boot, starttime_ticks=starttime,
            proc_identity_observed=True)
    except (OSError, UnicodeError, ValueError, IndexError):
        return unsupported


class TimeProxy:
    """Delegate all clocks and preserve every original perf-counter return."""
    def __init__(self, original, report):
        self.original = original
        self.report = report
        self.start = None
        self.end = None
        self.phase_calls = 0
        self.raw_clock_status = 'not_observed'
        self.raw_clock_error = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def boundary(self, value, samples):
        before = self.original.monotonic_ns()
        raw = None
        try:
            function = getattr(self.original, 'clock_gettime_ns', None)
            clock = getattr(self.original, 'CLOCK_MONOTONIC_RAW', None)
            if not callable(function) or clock is None:
                raise NotImplementedError('CLOCK_MONOTONIC_RAW is unavailable')
            raw = function(clock)
            if self.raw_clock_status != 'unsupported':
                self.raw_clock_status = 'supported'
        except (AttributeError, OSError, ValueError, NotImplementedError) as error:
            self.raw_clock_status = 'unsupported'
            self.raw_clock_error = type(error).__name__ + ': ' + str(error)
        epoch = self.original.time_ns()
        after = self.original.monotonic_ns()
        return dict(perf_counter_ns=value, monotonic_ns_before=before,
            clock_monotonic_raw_ns=raw, time_ns=epoch, monotonic_ns_after=after,
            samples_observed=samples, phase_perf_call=self.phase_calls,
            read_order=['perf_counter_ns', 'monotonic_ns_before',
                'clock_gettime_ns_MONOTONIC_RAW', 'time_ns', 'monotonic_ns_after'],
            boundary='perf was read first; correlation clocks were read afterwards; '
                'RAW and epoch reads lie within the monotonic before/after bracket')

    def perf_counter_ns(self):
        value = self.original.perf_counter_ns()
        if self.report.get('phase') == 'descriptive-timing':
            self.phase_calls += 1
            samples = len(self.report.get('timing', {}).get('durations_ns', []))
            if self.phase_calls == 1 and samples == 0:
                self.start = self.boundary(value, samples)
            elif samples == SAMPLES and self.end is None:
                self.end = self.boundary(value, samples)
        return value

    def proof(self):
        complete = (self.start is not None and self.end is not None
            and self.start['phase_perf_call'] == 1
            and self.end['phase_perf_call'] == 2 * SAMPLES + 2
            and self.raw_clock_status == 'supported'
            and self.start['clock_monotonic_raw_ns'] is not None
            and self.end['clock_monotonic_raw_ns'] is not None
            and self.end['perf_counter_ns'] > self.start['perf_counter_ns'])
        return dict(window_complete=complete, start=self.start, end=self.end,
            phase_perf_calls=self.phase_calls, expected_end_phase_perf_call=2 * SAMPLES + 2,
            raw_clock_status=self.raw_clock_status, raw_clock_error=self.raw_clock_error)


def run_with_window(runner, argv, output, runner_path):
    output = Path(output)
    require(not output.exists(), 'new exclusive window output required')
    application_process = process_identity()
    original_run, original_time = runner.run, runner.time
    captured = {}; code = None; failure = None

    def observed_run(args, report, cleanup):
        require(not captured, 'original run may be called exactly once')
        captured['report'] = report
        require(args.kind == 'perf' and args.warmup == 20 and args.iterations == SAMPLES,
            'window diagnosis requires original perf20+2000 exposure')
        proxy = TimeProxy(original_time, report)
        captured['proxy'] = proxy
        runner.time = proxy
        try:
            return original_run(args, report, cleanup)
        finally:
            runner.time = original_time

    runner.run = observed_run
    try:
        code = runner.main(argv)
        return code
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        runner.run = original_run
        runner.time = original_time
        report = captured.get('report', {})
        proxy = captured.get('proxy')
        proof = proxy.proof() if proxy is not None else dict(window_complete=False,
            start=None, end=None, raw_clock_status='not_observed', raw_clock_error=None)
        timing = report.get('timing', {})
        start, end = proof['start'], proof['end']
        common_matches = (start is not None and end is not None
            and timing.get('common_wall_ns') == end['perf_counter_ns'] - start['perf_counter_ns'])
        row = dict(schema=1, protocol='frozen-r22-common-window-clock-correlation-v1',
            diagnostic_only=True, performance_conclusion_allowed=False,
            application_process=application_process,
            original_exit_code=code, original_argv=list(argv), failure=failure,
            runner=dict(path=str(runner_path), sha256=sha(runner_path)), tool_sha256=sha(__file__),
            report_phase=report.get('phase'), original_report_passed=report.get('passed'),
            samples=len(timing.get('durations_ns', [])), original_common_wall_ns=timing.get('common_wall_ns'),
            common_wall_ns_matches=common_matches,
            temporary_module_bindings_restored=runner.time is original_time and runner.run is original_run,
            limits=['Extra clock reads occur only at the two common-window boundaries.',
                'Start correlation reads add observer cost inside original common wall.',
                'Clocks are sequential reads; the perf timestamp precedes the correlation bracket.',
                'Missing boundaries, unsupported RAW clock or incomplete original run cannot qualify a window.',
                'This wrapper supplies no native phase durations and makes no performance claim.'], **proof)
        row['window_complete'] = (row['window_complete'] and common_matches and code == 0
            and failure is None and row['samples'] == SAMPLES and report.get('passed') is True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(row, stream, indent=2, allow_nan=False); stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--window-output', type=Path, required=True)
    outer, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    path = outer.runner.resolve(strict=True)
    output = outer.window_output.resolve()
    require(sha(path) == RUNNER_SHA, 'original frozen runner SHA mismatch')
    require(not output.exists(), 'new exclusive window output required')
    spec = importlib.util.spec_from_file_location('_r23_native_window_frozen_runner', path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    args = runner.parse_args(forwarded)
    require(output not in (path, args.output.resolve(), args.wheel.resolve())
        and not output.is_relative_to(args.upstream.resolve()), 'window output overlaps original inputs/result')
    require(args.kind == 'perf' and args.warmup == 20 and args.iterations == SAMPLES,
        'window diagnosis requires original perf20+2000 exposure')
    return run_with_window(runner, forwarded, output, path)


if __name__ == '__main__':
    raise SystemExit(main())
