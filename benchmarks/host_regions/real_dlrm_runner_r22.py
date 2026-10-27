"""R22 real-device serial/four-worker DLRM validation on std/FT CPython 3.14.6.

The inference-only AST loader omits the sole training SummaryWriter import.
Original nine-file source bytes and all model/function AST nodes remain pinned.
This explicitly adapted import closure does not certify the training CLI.

worker4 requires the separately declared real native capability and its new
creation entry. An absent/unsupported capability fails before device/forward.
This runner never installs packages or changes JIT, formats or sparse settings.
"""
import argparse
import ast
import base64
import copy
from contextlib import contextmanager
from email.parser import Parser
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import statistics
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
import zipfile

BATCH, PARAMETERS = 128, 455153
STATE_SEED, INPUT_SEED = 20260917, 20260918
RTOL, ATOL = 1e-4, 1e-5
STATE_SHA = '6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19'
INPUT_SHA = '239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6'
UPSTREAM_COMMIT = '9bc1bc3602aea22afad3206910cec50f2dbe7364'
SOURCE_SHA = {
    'data_loader_terabyte.py': '0a348da7333c61dcacc073cf29e8a8d0db6985721c66af9c0158f69b33416537',
    'data_utils.py': '8672381cde6226cc5cf346eea7e9237e4cff8cf50a959174c27e128eb49c2c28',
    'dlrm_data_pytorch.py': 'ddfeba2d5f7a123f01bd8b176100cffb43fddd51470d978d8f2fcf6f3d537cfc',
    'dlrm_s_pytorch.py': '9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5',
    'extend_distributed.py': 'a8a880e6b3182dfb693d5618394885f4e8c8ad183e1b824175a40b1eb4d4779b',
    'mlperf_logger.py': '3794c3d5f5f8725e8751e71dd3ee3d9427a163cd0c476a025a4dedd5ecb87932',
    'optim/rwsadagrad.py': '9e0e96c266f00ee071ddf8ffecf79d56c429dacd0d8f41632262eb7db2a51bec',
    'tricks/md_embedding_bag.py': '422b654bfde0f2b9af367dc5e7ec2e48bedc0b87cb8d693e2bd87341a7325339',
    'tricks/qr_embedding_bag.py': '45263be6c43355ced5f48c0ed749e319b38f1acfb44aaa3603eafe04ab2779ce',
}
CAPTURE_FLAGS = ('FTLAB_HOST_REGIONS', 'FTLAB_HOST_NATIVE_RUN', 'HDB_REGION_PACKET', 'HDB_REGION_REAL_OPAPI',
                 'HDB_REGION_NATIVE_BUILDER', 'FTLAB_HOST_OWNER_RETIREMENT',
                 'HDB_REGION_TRACE', 'HDB_REGION_DIAGNOSTICS', 'HDB_NATIVE_PREPARE_STUB')
SNAPSHOT_FLAGS = (*CAPTURE_FLAGS, 'TASK_QUEUE_ENABLE', 'PER_STREAM_QUEUE', 'ASCEND_LAUNCH_BLOCKING',
                  'TORCH_DEVICE_BACKEND_AUTOLOAD', 'TORCH_NPU_EXPERIMENTAL_FREE_THREADED', 'PYTHON_GIL')
CANN_NAMES = ('libascendcl.so', 'libacl_op_compiler.so', 'libopapi.so', 'libhccl.so')
DSO_NAMES = (*CANN_NAMES, 'libtorch_npu.so', 'libtorch_python.so', 'libtorch_cpu.so',
             'libc10.so', 'libascend_hal.so', 'libgomp', 'libpython3.14')
FORBIDDEN_MAP_MARKERS = ('cinderx', 'acl_stub', 'acl-stub', 'hostsim', 'host-sim', 'host_sim',
                         'torch-npu-2130-sim', '/third_party/acl/libs/', '/stub/', '/stubs/')
CONTRACTS_SHA256='5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a'
SERIAL_ZERO_FIELDS=('captured','published','active_scopes','commands_live','packet_blocks_live',
    'packet_parameter_bytes_live','native_command_arenas_live','native_runs_live','native_active_drains',
    'native_queued_blocks','native_queued_commands','tokens_live','native_retirement_pending',
    'storage_owner_packets_live','storage_owner_cache_entries_live','python_callbacks')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stream_sha(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def load_contracts():
    path=Path(__file__).with_name('measurement_contracts_r22.py')
    require(sha256(path)==CONTRACTS_SHA256,'shared topology/stability contract source SHA mismatch')
    spec=importlib.util.spec_from_file_location('_r22_runner_contracts',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parse_cpus(value):
    require(re.fullmatch(r'\d+(?:,\d+)*', value), 'CPUs must be eight explicit nonnegative integers')
    cpus = list(map(int, value.split(',')))
    require(len(cpus) == len(set(cpus)) == 8, 'exactly eight unique CPUs are required')
    return sorted(cpus)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--abi', choices=('std', 'ft'), required=True)
    p.add_argument('--arm', choices=('serial', 'worker4'), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--cann-prefix', type=Path, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--expected-wheel-sha256', '--expected-wheel-sha', dest='expected_wheel_sha256', required=True)
    p.add_argument('--expected-native-sha256')
    p.add_argument('--runtime-receipt', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=('gate', 'perf','stability'), default='perf')
    p.add_argument('--topology-proof',type=Path,required=True)
    p.add_argument('--topology-proof-sha256',required=True)
    p.add_argument('--gate-proof', type=Path)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--threads', type=int, choices=(1,), default=1)
    return p.parse_args(argv)


def validate_real_capability(evidence):
    for key in ('real_capture_abi', 'opapi_owned_abi'):
        require(type(evidence.get(key)) is int and evidence[key] == 1, 'real capability ABI mismatch: ' + key)
    require(evidence.get('real_capture_supported') is True, 'native real OpAPI capability is unavailable')
    require(evidence.get('real_prepare_strategy') in ('consumer','worker'), 'unreviewed real preparation strategy')
    return {key: evidence[key] for key in ('real_capture_abi','opapi_owned_abi','real_capture_supported','real_prepare_strategy')}


def validate_serial_closed(row):
    for key in SERIAL_ZERO_FIELDS:
        require(type(row.get(key)) is int and row[key]==0,'serial capture/packet/owner/pending did not remain zero: '+key)
    exported='owned_opapi_payloads_live' in row
    if exported:
        for key in ('owned_opapi_payloads_live','owned_opapi_device_pending','owned_opapi_retained_bytes'):
            require(type(row.get(key)) is int and row[key]==0,'serial owned resource is not closed: '+key)
    return dict(passed=True,capture_disabled=True,owned_opapi_schema_exported=exported,
        owned_opapi_boundary='original R21 DSO does not export owned OpAPI schema; disabled capture and all original resource gauges verified' if not exported else 'exported owned gauges verified zero',
        checked_fields=list(SERIAL_ZERO_FIELDS))


def validate_real_delta(before, after, forwards, *, diagnostic=False):
    validate_real_capability(before); validate_real_capability(after)
    require(before['real_prepare_strategy'] == after['real_prepare_strategy'],'preparation strategy changed during run')
    require(type(forwards) is int and forwards > 0, 'positive exact forward count required')
    def value(row, key):
        result = row.get(key)
        require(type(result) is int and result >= 0, 'missing/inexact native counter: ' + key)
        return result
    def delta(key):
        result = value(after,key) - value(before,key)
        require(result >= 0, 'native counter regressed: ' + key)
        return result
    for key in ('real_runs_created','real_runs_destroyed'):
        require(delta(key) == forwards, 'real native run count does not match every forward: ' + key)
    commands = delta('owned_opapi_captured')
    require(commands > 0 and commands % forwards == 0, 'real OpAPI path missed a fixed-input forward')
    for suffix in ('leases_acquired','admission_acked','submit_attempts','submit_returns','device_retired'):
        require(delta('owned_opapi_' + suffix) == commands, 'owned command lifecycle did not close: ' + suffix)
    for suffix in ('cancelled','retirement_failures','eligible_fallback'):
        require(value(before,'owned_opapi_'+suffix) == value(after,'owned_opapi_'+suffix) == 0,
                'real path failed or fell back: ' + suffix)
    for key in ('real_runs_live','owned_opapi_payloads_live','owned_opapi_device_pending',
                'packet_blocks_live','packet_parameter_bytes_live','owned_opapi_retained_bytes','commands_live',
                'native_command_arenas_live',
                'python_callbacks','native_runs_live','native_active_drains','native_queued_blocks',
                'native_queued_commands','tokens_live','native_retirement_pending',
                'storage_owner_packets_live','storage_owner_cache_entries_live'):
        require(value(before,key) == value(after,key) == 0, 'native resource was not fully closed: ' + key)
    events = delta('owned_opapi_device_events_recorded')
    require(events > 0 and delta('owned_opapi_device_events_completed') == events,
            'block completion events did not close')
    workers = None
    if diagnostic:
        require(before.get('owned_opapi_diag_incomplete') is False and after.get('owned_opapi_diag_incomplete') is False,
                'native real diagnostic records are incomplete')
        threads = after.get('real_worker_threads')
        require(type(threads) is list and len(threads) == 4, 'four native worker records required')
        previous = {row['tid']:row for row in before.get('real_worker_threads', [])}
        tids = set(); captured = 0
        for row in threads:
            require(type(row) is dict and type(row.get('tid')) is int and row['tid'] > 0 and row['tid'] not in tids,
                    'invalid/duplicate native worker TID')
            tids.add(row['tid']); old = previous.get(row['tid'], {})
            for key in ('regions','owned_opapi_captured'):
                require(type(row.get(key)) is int and row[key] > old.get(key,0), 'native worker has no real work: ' + key)
            captured += row['owned_opapi_captured'] - old.get('owned_opapi_captured',0)
        require(captured == commands, 'per-worker command counts do not cover native captures')
        workers = len(tids)
    return dict(passed=True,forwards=forwards,commands=commands,commands_per_forward=commands//forwards,
        prepare_strategy=after['real_prepare_strategy'],
        actual_workers=workers,native_worker_observation_enabled=diagnostic,
        device_events=events,device_event_unit='block',device_pending_unit='block',
        all_native_resources_closed=True,eligible_fallback=0)


def match_real_gate(mechanism,gate):
    require(mechanism.get('passed') is True and gate.get('passed') is True,'real mechanism gate is not passed')
    rate=mechanism.get('commands_per_forward')
    require(type(rate) is int and rate>0 and type(gate.get('commands_per_forward')) is int
            and rate==gate['commands_per_forward'],'perf capture coverage differs from the real model gate')
    require(mechanism.get('prepare_strategy')==gate.get('prepare_strategy'),'perf preparation strategy differs from gate')


def validate_fifo(snapshot):
    require(snapshot.get('diagnostics_enabled') is True and snapshot.get('captured_consumption_tracked') is True,
            'FIFO proof requires real diagnostic tracking')
    for key in ('fifo_errors','queue_discarded','queue_abandoned','pending_consumption','queue_pending_total'):
        require(type(snapshot.get(key)) is int and snapshot[key] == 0, 'FIFO/queue did not close: ' + key)
    require(type(snapshot.get('published_hash')) is type(snapshot.get('consumed_hash')) is int
            and snapshot['published_hash'] >= 0 and snapshot['published_hash'] == snapshot['consumed_hash'],
            'captured FIFO hash missing or mismatched')
    published,consumed = snapshot.get('queue_published_trace'),snapshot.get('queue_consumed_trace')
    require(type(published) is type(consumed) is list and 0 < len(published) < 4096 and published == consumed,
            'full original-queue source FIFO trace missing/truncated/mismatched')
    trace = snapshot.get('trace')
    require(type(trace) is list and 0 < len(trace) < 4096 and all(type(row) is list and len(row)==4 for row in trace),
            'captured source-order trace missing/truncated')
    require([row[3] for row in trace] == list(range(1,len(trace)+1)), 'captured ordinal trace is not contiguous')
    require(type(snapshot.get('published')) is type(snapshot.get('consumed')) is int
            and snapshot['published'] == snapshot['consumed'] == len(trace), 'captured trace is incomplete')
    return dict(passed=True,queue_commands=len(published),captured_commands=len(trace),trace_truncated=False)


def wait_real_retirement(bridge, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        row = bridge.stats()
        fields=('owned_opapi_device_pending','owned_opapi_payloads_live','packet_blocks_live',
                'packet_parameter_bytes_live','owned_opapi_retained_bytes','commands_live','native_command_arenas_live')
        require(all(type(row.get(key)) is int and row[key]>=0 for key in fields),'retirement gauges unavailable or inexact')
        if all(row[key]==0 for key in fields):
            return row
        require(time.monotonic() < deadline, 'device resource retirement did not finish after synchronization')
        time.sleep(.001)


class ColdHandoffSpy:
    """Trap old callback/handoff paths only in cold diagnostic processes."""
    def __init__(self,provider,runtime,contracts,backends):
        self.targets=[(provider,name) for name in ('_EMIT','_EMIT_BATCH','CapturedCommand')]
        self.targets += [(provider.NativeBridge,'publish'),(runtime,'_Region'),(runtime,'RegionCommand'),
                         (contracts,'RegionCommand'),(backends,'CapturedCommand'),(backends,'RegionCommand')]
        self.saved=[]
        self.record=dict(enabled=True,forbidden_calls=0,restored=False,
                         targets=[name for _,name in self.targets],scope='legacy native-to-Python emitter/handoff paths only')
    def install(self):
        def forbidden(*args,**kwargs):
            self.record['forbidden_calls']+=1
            raise AssertionError('forbidden legacy Python handoff was invoked')
        try:
            for owner,name in self.targets:
                self.saved.append((owner,name,getattr(owner,name)));setattr(owner,name,forbidden)
        except BaseException:
            self.restore();raise
    def restore(self):
        for owner,name,value in reversed(self.saved):setattr(owner,name,value)
        self.saved.clear();self.record['restored']=True


class ColdWorkerProbe:
    """Ordinary Python worker entry observation, outside every timed window."""
    def __init__(self, runtime_module):
        self.module,self.saved,self.rows = runtime_module,runtime_module._native_work,[]
        self.lock = threading.Lock()
    def install(self):
        def observed(runtime,session,region,body,items):
            start,cpu = time.monotonic_ns(),time.thread_time_ns()
            try: return self.saved(runtime,session,region,body,items)
            finally:
                row = dict(tid=threading.get_native_id(),run=session.run,region=region.index,items=region.stop-region.start,
                    start_ns=start,end_ns=time.monotonic_ns(),cpu_ns=time.thread_time_ns()-cpu)
                with self.lock: self.rows.append(row)
        self.module._native_work = observed
    def restore(self): self.module._native_work = self.saved
    def proof(self):
        tids = sorted({row['tid'] for row in self.rows if row['items'] > 0})
        require(len(tids)==4 and all(row['cpu_ns'] > 0 for row in self.rows), 'four real Python worker entries required')
        groups={}
        for row in self.rows: groups.setdefault(row.get('run',0),[]).append(row)
        windows=[dict(run=run,wall_ns=max(row['end_ns'] for row in rows)-min(row['start_ns'] for row in rows),
                      cpu_ns=sum(row['cpu_ns'] for row in rows)) for run,rows in groups.items()]
        require(all(row['wall_ns'] > 0 for row in windows),'invalid worker observation window')
        wall = sum(row['wall_ns'] for row in windows)
        cpu = sum(row['cpu_ns'] for row in self.rows)
        return dict(rows=self.rows,tids=tids,actual_workers=4,worker_cpu_ns=cpu,window_wall_ns=wall,
            effective_worker_cpu_parallelism=cpu/wall,run_windows=windows,
            cpu_parallelism_proven=any(row['cpu_ns']>row['wall_ns']*1.02 for row in windows),
            proof_boundary='CPU sum/window >1.02 is sufficient, not necessary; overlapping wall spans alone are not CPU proof',
            timed_window_instrumented=False)


def install_real_worker_model(model, torch, torch_npu, bridge_module, report, cleanup, *, diagnostic=False):
    os.environ.update(HDB_REGION_REAL_OPAPI='1',FTLAB_HOST_NATIVE_RUN='1',HDB_REGION_PACKET='1',
                      HDB_REGION_NATIVE_BUILDER='0',FTLAB_HOST_OWNER_RETIREMENT='1')
    contracts = importlib.import_module('torch_npu._host_regions.contracts')
    runtime_module = importlib.import_module('torch_npu._host_regions.runtime')
    integration = importlib.import_module('torch_npu._host_regions.integration')
    provider = bridge_module.NativeCaptureProvider(torch,torch_npu,native_run=True,native_builder=False,real_opapi=True)
    runtime = runtime_module.RegionRuntime(contracts.RegionConfig(enabled=True,workers=4,regions=4,commands_per_region=32))
    cleanup.append(runtime.close)
    token = integration.install_instance(model,provider.bind_policy,runtime)
    cleanup.append(token.close)
    require(token.installed_count == 1, 'expected exactly the original independent embedding loop to be installed')
    report['region_installation'] = dict(installed_count=token.installed_count,reports=token.reports,
        path='explicit framework install_instance; original class/source unchanged; no global Module hook')
    probe = ColdWorkerProbe(runtime_module); probe.install(); cleanup.append(probe.restore)
    spy=None
    if diagnostic:
        spy=ColdHandoffSpy(bridge_module,runtime_module,contracts,importlib.import_module('torch_npu._host_regions.backends'))
        spy.install();cleanup.append(spy.restore);report['forbidden_handoff_spy']=spy.record
    return provider,probe,spy


def worker_capability():
    return dict(supported=False,optimization_active=False,actual_workers=0,
        reason='Real capability has not been verified from the actual installed R22 DSO',
        prerequisite='Exact native real ABI/support/strategy plus actual workers, FIFO and device-closure evidence')


def check_environment(env, *, diagnostic=False):
    require(env.get('QWEN_ACL_HOST_SIM') != '1', 'real NPU rejects QWEN_ACL_HOST_SIM=1')
    require(not env.get('LD_PRELOAD') and not env.get('LD_AUDIT'), 'LD_PRELOAD/LD_AUDIT must be empty')
    require('PYTHON_GIL' not in env and 'gil' not in sys._xoptions, 'PYTHON_GIL/-X gil overrides are forbidden')
    require(env.get('PER_STREAM_QUEUE', '0') == env.get('ASCEND_LAUNCH_BLOCKING', '0') == '0',
            'serial comparison preserves default asynchronous non-per-stream queue settings')
    require(all(env.get(k, '0') == ('1' if diagnostic and k in ('HDB_REGION_TRACE','HDB_REGION_DIAGNOSTICS') else '0')
                for k in CAPTURE_FLAGS),
            'serial real baseline requires capture/prepare/retirement/trace opt-ins disabled')


def verify_upstream(root, expected=SOURCE_SHA):
    root = Path(root).resolve(strict=True)
    found = {}
    for relative, pin in expected.items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), 'source escapes upstream snapshot')
        actual = sha256(path)
        require(actual == pin, 'official source SHA mismatch: ' + relative)
        found[relative] = actual
    return found


def compile_inference_source(source, filename):
    """Compile the pinned model after removing exactly one unused logging import."""
    tree = ast.parse(source, filename=filename)
    dump = lambda node: ast.dump(node, include_attributes=True)
    original_dump = dump(tree)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == 'torch.utils.tensorboard']
    require(len(imports) == 1, 'expected exactly one SummaryWriter import')
    removed = imports[0]
    require(removed in tree.body and removed.lineno == removed.end_lineno == 101
            and removed.col_offset == 0 and removed.level == 0
            and len(removed.names) == 1 and removed.names[0].name == 'SummaryWriter'
            and removed.names[0].asname is None,
            'only the exact top-level line101 SummaryWriter import may be omitted')
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    expected_names = ['time_wrap', 'dlrm_wrap', 'loss_fn_wrap', 'unpack_batch',
                      'LRPolicyScheduler', 'DLRM_Net', 'dash_separated_ints',
                      'dash_separated_floats', 'inference', 'run']
    require([n.name for n in definitions] == expected_names, 'pinned ten-definition structure changed')
    definition_dumps = [dump(n) for n in definitions]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == 'SummaryWriter']
    require(len(uses) == 1 and isinstance(uses[0].ctx, ast.Load), 'expected one training-only SummaryWriter reference')
    scope = parents[uses[0]]
    while scope not in definitions and scope in parents:
        scope = parents[scope]
    require(scope is definitions[-1], 'SummaryWriter reference escaped training run()')
    cli = [n for n in tree.body if isinstance(n, ast.If)
           and ast.dump(n.test) == ast.dump(ast.parse('__name__ == "__main__"', mode='eval').body)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run']
    require(len(cli) == len(calls) == 1 and not cli[0].orelse and len(cli[0].body) == 1
            and isinstance(cli[0].body[0], ast.Expr) and cli[0].body[0].value is calls[0]
            and not calls[0].args and not calls[0].keywords, 'run() may only be called by the original CLI guard')
    remaining = [dump(n) for n in tree.body if n is not removed]
    tree.body.remove(removed)
    require([dump(n) for n in tree.body] == remaining
            and [dump(n) for n in definitions] == definition_dumps, 'AST adapter changed another node')
    proof = dict(adapter='r21-v2-inference-only-single-import-omission', inference_only=True,
        executed_module_adapted=True,
        training_cli_certified=False, original_import_closure_certified=False,
        source_path=filename, source_sha256=hashlib.sha256(source).hexdigest(),
        removed_import=dict(line=101, text='from torch.utils.tensorboard import SummaryWriter', ast=dump(removed)),
        original_ast_sha256=hashlib.sha256(original_dump.encode()).hexdigest(),
        executed_ast_sha256=hashlib.sha256(dump(tree).encode()).hexdigest(),
        all_other_ast_nodes_unchanged=True, unchanged_definition_count=len(definitions),
        unchanged_definitions=expected_names,
        unchanged_definition_ast_sha256={n.name: hashlib.sha256(d.encode()).hexdigest()
                                         for n, d in zip(definitions, definition_dumps)},
        source_bytes_rewritten=False, fake_modules_or_writer_injected=False,
        boundary='Only DLRM_Net inference is used; run()/training/logging CLI is outside this validation')
    return compile(tree, filename, 'exec'), proof


def load_inference_module(upstream, report):
    upstream = Path(upstream).resolve(strict=True)
    verify_upstream(upstream)
    path = upstream / 'dlrm_s_pytorch.py'
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == SOURCE_SHA[path.name], 'model changed before AST loading')
    code, proof = compile_inference_source(source, str(path))
    name = 'dlrm_s_pytorch'
    require(name not in sys.modules and name != '__main__', 'official module already imported or CLI requested')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.origin == str(path), 'original source module spec is unavailable')
    module = importlib.util.module_from_spec(spec)
    proof.update(module_name=name, module_file=module.__file__, spec_origin=spec.origin,
                 execution='compile/exec of the explicitly adapted AST; original SourceFileLoader.exec_module is not called')
    report['model_loader_adapter'] = proof
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
        require('SummaryWriter' not in module.__dict__, 'unexpected SummaryWriter binding after adapted import')
        verify_upstream(upstream)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def verify_wheel_install(wheel, expected_sha, package, abi):
    wheel, package = Path(wheel).resolve(strict=True), Path(package).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{64}', expected_sha) and sha256(wheel) == expected_sha, 'wheel SHA mismatch')
    expected_abi = 'cp314t' if abi == 'ft' else 'cp314'
    filename = wheel.name.removesuffix('.whl').split('-')
    require(wheel.suffix == '.whl' and filename[0] == 'torch_npu' and filename[1].split('+')[0] == '2.13.0'
            and filename[-3:-1] == ['cp314', expected_abi]
            and filename[-1] in ('linux_aarch64', 'manylinux_2_28_aarch64'), 'wheel filename ABI/platform mismatch')
    payload = {}; tags = []
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate wheel members')
        metadata = [n for n in names if n.endswith('.dist-info/WHEEL')]
        require(len(metadata) == 1, 'wheel metadata missing or ambiguous')
        tags = Parser().parsestr(z.read(metadata[0]).decode()).get_all('Tag', [])
        require(tags and all(t.split('-')[:2] == ['cp314', expected_abi] and 'aarch64' in t.split('-')[-1] for t in tags), 'wheel metadata ABI/platform mismatch')
        for name in names:
            rel = PurePosixPath(name)
            require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name, 'unsafe wheel member path')
            if name.endswith('/') or not name.startswith('torch_npu/'):
                continue
            local = (package.parent / name).resolve(strict=True)
            require(local.is_relative_to(package) and local.is_file(), 'installed wheel payload escapes package')
            with z.open(name) as stream:
                pin = stream_sha(stream)
            require(sha256(local) == pin, 'installed package differs from supplied wheel: ' + name)
            payload[name] = pin
    require('torch_npu/lib/libtorch_npu.so' in payload, 'wheel main native library missing')
    return dict(path=str(wheel), sha256=expected_sha, tags=tags, payload_files_verified=len(payload),
                payload_sha256=payload, installed_package=str(package),
                scope='supplied wheel bytes and installed payload identity; not a device or parallel capability certificate')


def verify_extension_identity(torch_npu, package, abi):
    extension = torch_npu._C
    marker = getattr(extension, '_experimental_free_threaded_build', None)
    require(type(marker) is int and marker == (1 if abi == 'ft' else 0), 'native extension FT build marker mismatch')
    path, package = Path(extension.__file__).resolve(strict=True), Path(package).resolve(strict=True)
    expected = '_C.cpython-' + ('314t' if abi == 'ft' else '314') + '-aarch64-linux-gnu.so'
    require(path.parent == package and path.name == expected, 'loaded _C extension is outside the verified ABI package')
    return dict(path=str(path), sha256=sha256(path), build_marker=marker)


def mark_device_attempt(report, kind):
    require(kind in ('initialization', 'transfer', 'forward'), 'invalid device attempt phase')
    report['device_execution_attempted'] = True
    report['forward_attempted' if kind == 'forward' else 'device_' + kind + '_attempted'] = True


def parse_process_memory(text):
    values={line.split(':',1)[0]:line.split(':',1)[1].strip().split() for line in text.splitlines() if ':' in line}
    for key in ('VmRSS','VmHWM'):
        require(key in values and len(values[key])==2 and values[key][1]=='kB','Linux RSS memory units unavailable')
    return dict(rss_kib=int(values['VmRSS'][0]),whole_process_peak_rss_kib=int(values['VmHWM'][0]),
        timed_window_peak=False,scope='Linux process RSS/HWM, including setup and imports')


def memory_snapshot(torch):
    result=parse_process_memory(Path('/proc/self/status').read_text())
    try:
        result['npu_allocator']={name:int(getattr(torch.npu,name)(0)) for name in
            ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        result['npu_allocator']['scope']='framework allocator bytes/whole-process peaks; excludes untracked SDK pools'
    except (AttributeError,RuntimeError) as error:
        result['npu_allocator']=dict(available=False,reason=str(error))
    return result


def numeric_summary(actual, reference, record=None):
    require(len(actual) == len(reference) == BATCH, 'full 128-value output required')
    record = {} if record is None else record
    finite = all(math.isfinite(v) for v in [*actual, *reference])
    in_range = all(0 <= v <= 1 for v in actual)
    close = all(abs(a - b) <= ATOL + RTOL * abs(b) for a, b in zip(actual, reference))
    serializable = lambda values: [v if math.isfinite(v) else str(v) for v in values]
    record.update(cpu_reference_values=serializable(reference), npu_values=serializable(actual),
        npu_fp32_bytes_base64=base64.b64encode(struct.pack('<128f', *actual)).decode(),
        finite=finite, range_0_1=in_range, rtol=RTOL, atol=ATOL, allclose=close,
        max_abs_error=max(abs(a - b) for a, b in zip(actual, reference)) if finite else None)
    require(finite, 'nonfinite output/reference')
    require(in_range, 'sigmoid output outside [0,1]')
    require(close, 'full-output CPU/NPU allclose failed')
    return record


def tensor_tree_sha256(value, torch):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(payload).to_bytes(8, 'little')); digest.update(payload)
        elif isinstance(item, dict):
            digest.update(b'dict[')
            for key in sorted(item):
                require(isinstance(key, str), 'tensor-tree keys must be strings'); add(key); add(item[key])
            digest.update(b']')
        elif isinstance(item, (tuple, list)):
            digest.update(b'tuple[' if isinstance(item, tuple) else b'list[')
            for child in item: add(child)
            digest.update(b']')
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(json.dumps(item, allow_nan=False, separators=(',', ':')).encode() + b';')
        else:
            raise TypeError('unsupported tensor-tree type: ' + type(item).__name__)
    add(value)
    return digest.hexdigest()


def eager_proof(model, torch):
    bad = [name for name, module in model.named_modules()
           if getattr(module, '_compiled_call_impl', None) is not None or isinstance(module, torch.jit.ScriptModule)
           or any(cls.__name__ == 'OptimizedModule' for cls in type(module).__mro__)]
    require(not bad, 'Eager gate rejected modules: ' + repr(bad))
    return dict(mode='eager', module_count=sum(1 for _ in model.modules()), compiled_call_impl=False, script_or_optimized_modules=bad)


def sparse_proof(model, torch):
    values = {n: bool(m.sparse) for n, m in model.named_modules() if isinstance(m, torch.nn.EmbeddingBag)}
    require(len(values) == 26 and all(values.values()), 'original 26 EmbeddingBag modules must retain sparse=True')
    return values


def device_proof(model, inputs):
    parameters = {n: str(v.device) for n, v in model.named_parameters()}
    buffers = {n: str(v.device) for n, v in model.named_buffers()}
    dense, offsets, indices = inputs
    devices = [str(v.device) for v in (dense, *offsets, *indices)]
    require(parameters and set(parameters.values()) == {'npu:0'} and not set(buffers.values()) - {'npu:0'}, 'model is outside npu:0')
    require(len(devices) == 53 and set(devices) == {'npu:0'}, 'original inputs are outside npu:0')
    return dict(parameters=parameters, buffers=buffers, input_tensor_count=len(devices), input_devices=sorted(set(devices)))


def native_proof(cann_prefix, require_ascendcl=False):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith('/'):
            paths.add(parts[5].strip())
    forbidden = sorted(p for p in paths if any(x in p.lower() for x in FORBIDDEN_MAP_MARKERS))
    require(not forbidden, 'stub/HOSTSIM/CinderX library mapping: ' + repr(forbidden))
    cinderx = [n for n in sys.modules if n == 'cinderx' or n.startswith('cinderx.')]
    require(not cinderx, 'CinderX is loaded')
    libraries = []; checked = []; present = False
    for path in sorted(paths):
        if not any(n in Path(path).name for n in DSO_NAMES): continue
        actual = Path(path).resolve(strict=True)
        libraries.append(dict(mapped_path=path, real_path=str(actual), sha256=sha256(actual), bytes=actual.stat().st_size))
        if any(n in actual.name for n in CANN_NAMES):
            require(actual.is_relative_to(cann_prefix), 'CANN DSO outside requested prefix: ' + str(actual))
            checked.append(str(actual)); present |= 'libascendcl.so' in actual.name
    require(not require_ascendcl or present, 'initialized real NPU has no requested CANN libascendcl mapping')
    return dict(libraries=libraries, forbidden_mappings=forbidden, cinderx_modules=cinderx,
                cann_prefix_gate=dict(prefix=str(cann_prefix), checked_real_paths=checked,
                                      ascendcl_present=present, ascendcl_required=require_ascendcl))


def affinity_snapshot(cpus):
    rows = []
    for task in sorted(Path('/proc/self/task').iterdir(), key=lambda x: int(x.name)):
        try:
            values = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        rows.append(dict(tid=int(task.name), cpus=values))
    require(all(set(row['cpus']) <= set(cpus) for row in rows), 'a thread escaped the selected eight CPUs')
    return rows


def runtime_identity(torch, torch_npu, np, abi):
    require(sys.version_info[:3] == (3, 14, 6), 'requires project CPython3.14.6, not the CP311 historical control')
    require(str(torch.__version__) == '2.13.0+cpu' and np.__version__ == '2.5.3', 'requires Torch2.13.0+cpu and NumPy2.5.3')
    require(str(torch_npu.__version__).split('+')[0] == '2.13.0', 'PTA2.13 required')
    require(sys._is_gil_enabled() is (abi == 'std') and bool(sysconfig.get_config_var('Py_GIL_DISABLED')) is (abi == 'ft'), 'Python ABI/GIL mismatch after imports')
    require(sysconfig.get_config_var('SOABI') == 'cpython-314' + ('t' if abi == 'ft' else '') + '-aarch64-linux-gnu', 'SOABI mismatch')
    return dict(python=sys.version, executable=sys.executable, abi=sysconfig.get_config_var('SOABI'),
                gil_enabled=sys._is_gil_enabled(), py_gil_disabled=sysconfig.get_config_var('Py_GIL_DISABLED'),
                torch=str(torch.__version__), torch_git=torch.version.git_version, torch_path=torch.__file__,
                torch_npu=str(torch_npu.__version__), torch_npu_path=torch_npu.__file__, numpy=np.__version__,
                numpy_path=np.__file__, configure_args=sysconfig.get_config_var('CONFIG_ARGS'))


def compile_option(torch_npu):
    # Module.cpp:1220 reads jitCompile/jitCompileInit through the original native getter.
    value = torch_npu.npu.is_jit_compile_false()
    require(type(value) is bool, 'native default JIT option getter returned non-bool')
    return dict(api='torch_npu.npu.is_jit_compile_false -> _C._npu_is_jit_compile_false',
                raw_is_jit_compile_false=value, mutation_performed=False,
                boundary='original native compile-option getter, separate from Torch eager/graph state')


def validate_output(output, reference, torch, record):
    require(isinstance(output, torch.Tensor), 'original forward must return a Tensor')
    record.update(shape=list(output.shape), dtype=str(output.dtype), device=str(output.device))
    require(list(output.shape) == [BATCH, 1]
            and output.dtype == torch.float32 and str(output.device) == 'npu:0', 'output must be npu:0 FP32 [128,1]')
    actual = output.detach().cpu().contiguous()
    numeric_summary(actual.reshape(-1).tolist(), reference.reshape(-1).tolist(), record)
    require(bool(torch.allclose(actual, reference, rtol=RTOL, atol=ATOL)), 'Torch full-output allclose failed')
    record.update(torch_allclose=True,
        npu_fp32_bytes_base64=base64.b64encode(actual.numpy().tobytes()).decode())


def validate_checkpoint_identity(row,abi,cpus):
    require(type(row.get('completed')) is int and row['completed'] in range(1000,20001,1000),'unexpected stability checkpoint exposure')
    numeric=row.get('functional',{})
    require(numeric.get('shape')==[128,1] and numeric.get('device')=='npu:0' and numeric.get('dtype')=='torch.float32'
            and all(numeric.get(key) is True for key in ('finite','range_0_1','allclose','torch_allclose'))
            and len(numeric.get('cpu_reference_values',[]))==len(numeric.get('npu_values',[]))==128,
            'stability requires complete 128-value numeric evidence')
    require(row.get('source_files')==SOURCE_SHA,'stability source identity drift')
    identities=row.get('identities',{})
    require(identities.get('cpu_state_sha256')==identities.get('npu_state_sha256')==STATE_SHA
            and identities.get('cpu_input_sha256')==identities.get('npu_input_sha256')==INPUT_SHA,'stability weights/input identity drift')
    require(row.get('runtime',{}).get('gil_enabled') is (abi=='std'),'stability natural GIL drift')
    affinity=row.get('affinity')
    require(type(affinity) is list and affinity and all(type(task.get('cpus')) is list and task['cpus']
        and set(task['cpus'])<=set(cpus) for task in affinity),'stability thread escaped eight-physical-core mask')
    require(row.get('retirement',{}).get('passed') is True,'stability native resource/hit proof missing')
    memory=row.get('memory',{});allocator=memory.get('npu_allocator',{})
    require(all(type(memory.get(name)) is int and memory[name]>=0 for name in ('rss_kib','whole_process_peak_rss_kib')),
            'stability RSS/HWM unavailable')
    require(allocator.get('available') is not False and all(type(allocator.get(name)) is int and allocator[name]>=0
            for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')),'stability NPU allocator metrics unavailable')
    return row


def run(args, report, cleanup):
    report['phase'] = 'environment'
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'serial requires the native Linux AArch64 target')
    require(sys.version_info[:3] == (3, 14, 6), 'requires project CPython3.14.6')
    check_environment(os.environ,diagnostic=args.kind=='gate')
    require((args.warmup,args.iterations) == {'gate':(0,1),'perf':(20,2000),'stability':(20,20000)}[args.kind],
            'R22 requires fixed cold/perf/stability exposure')
    os.environ.setdefault('TASK_QUEUE_ENABLE', '1')
    report['flags_at_import'] = {k: os.environ.get(k) for k in SNAPSHOT_FLAGS}
    require(os.environ['TASK_QUEUE_ENABLE'] == '1', 'formal real control keeps requested TASK_QUEUE_ENABLE=1')
    cpus = parse_cpus(args.cpus)
    require(set(cpus) <= os.sched_getaffinity(0), 'selected CPUs are not available in this instance')
    require(re.fullmatch('[0-9a-f]{64}',args.topology_proof_sha256) and sha256(args.topology_proof)==args.topology_proof_sha256,
            'common topology proof source SHA mismatch')
    contracts=load_contracts();topology=json.loads(args.topology_proof.read_text())
    contracts.verify_topology(topology,cpus)
    report['topology']=dict(proof_path=str(args.topology_proof),proof_sha256=args.topology_proof_sha256,
        contracts_sha256=CONTRACTS_SHA256,selected_cpus=cpus,choice=topology['choice'],verified_before_import=True)
    os.sched_setaffinity(0, cpus)
    report['affinity_before_import'] = affinity_snapshot(cpus)
    cann = args.cann_prefix.resolve(strict=True)
    report['cann_prefix'] = dict(requested=str(args.cann_prefix), real_path=str(cann))
    report['source'] = dict(directory=str(args.upstream.resolve(strict=True)),
        upstream_commit_of_snapshot=UPSTREAM_COMMIT, files=verify_upstream(args.upstream), git_claim='exact nine-file bytes only')
    report['native_before_import'] = native_proof(cann)
    report['phase'] = 'dependency-import'
    # Preserve the original AArch64 static-TLS mitigation, without LD_PRELOAD.
    import sklearn.metrics
    report['dependency_import_order'] = 'sklearn.metrics before torch and torch_npu'
    import numpy as np
    import torch
    import torch_npu
    report['runtime'] = runtime_identity(torch, torch_npu, np, args.abi)
    report['wheel'] = verify_wheel_install(args.wheel, args.expected_wheel_sha256, Path(torch_npu.__file__).resolve().parent, args.abi)
    report['native_extension'] = verify_extension_identity(torch_npu, Path(torch_npu.__file__).resolve().parent, args.abi)
    native = Path(torch_npu.__file__).resolve().parent / 'lib/libtorch_npu.so'
    report['native_sha256'] = sha256(native)
    if args.expected_native_sha256:
        require(report['native_sha256'] == args.expected_native_sha256, 'loaded native library SHA mismatch')
    bridge_module = importlib.import_module('torch_npu._host_regions.torch_npu_provider')
    bridge = bridge_module.NativeBridge(native)
    report['native_capability_before'] = bridge.stats()
    loaded_python = {}
    for name in ('__init__.py','backends.py','contracts.py','frontend.py','integration.py','runtime.py','torch_npu_policy.py','torch_npu_provider.py'):
        qualified = 'torch_npu._host_regions' + ('' if name=='__init__.py' else '.'+name[:-3])
        module = importlib.import_module(qualified)
        path = Path(module.__file__).resolve(strict=True)
        relative = 'torch_npu/_host_regions/' + name
        require(path == Path(torch_npu.__file__).resolve().parent/'_host_regions'/name
                and sha256(path) == report['wheel']['payload_sha256'].get(relative), 'loaded host-region Python source mismatch: '+name)
        loaded_python[name] = dict(path=str(path),sha256=sha256(path))
    report['loaded_python_sources'] = loaded_python
    if args.arm == 'worker4':
        report['phase'] = 'worker-capability'
        report['worker_capability'] = worker_capability()
        certificate = validate_real_capability(report['native_capability_before'])
        bridge.enable_native_run(); bridge.enable_real_opapi()
        report['worker_capability'] = dict(supported=True,certificate=certificate,
            scope='native declaration plus later actual command/worker/device-closure gates; not environment evidence')
        if args.kind in ('perf','stability'):
            require(certificate['real_prepare_strategy']=='worker','final P1 perf requires actual worker SDK preparation')
            require(args.gate_proof is not None, 'worker4 perf requires its real diagnostic gate receipt')
            gate = json.loads(args.gate_proof.read_text())
            require(gate.get('passed') is True and gate.get('abi') == args.abi and gate.get('arm') == 'worker4'
                    and gate.get('kind') == 'gate' and gate.get('requested_wheel',{}).get('sha256') == args.expected_wheel_sha256
                    and gate.get('script_sha256') == sha256(__file__)
                    and gate.get('source',{}).get('files') == report['source']['files']
                    and gate.get('native_sha256') == report['native_sha256']
                    and gate.get('mechanism',{}).get('passed') is True and gate.get('fifo_proof',{}).get('passed') is True
                    and gate.get('mechanism',{}).get('prepare_strategy') == certificate['real_prepare_strategy']
                    and gate.get('worker_probe',{}).get('actual_workers') == 4,
                    'real worker gate is absent or belongs to another build/ABI')
            report['diagnostic_gate_input'] = dict(path=str(args.gate_proof.resolve()),sha256=sha256(args.gate_proof),
                                                  mechanism=gate['mechanism'])
    if args.runtime_receipt:
        receipt = json.loads(args.runtime_receipt.read_text(encoding='utf-8'))
        require(receipt.get('passed') is True, 'bootstrap runtime receipt is not passed')
        report['bootstrap_runtime_receipt'] = dict(path=str(args.runtime_receipt.resolve()), sha256=sha256(args.runtime_receipt))
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(args.threads)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Torch thread limits must remain 1')
    mark_device_attempt(report, 'initialization')
    require(torch.npu.is_available(), 'real NPU is unavailable')
    torch.npu.set_device(0)
    report['runtime'].update(device_count=torch.npu.device_count(), device_name=torch.npu.get_device_name(0),
        device_properties=str(torch.npu.get_device_properties(0)), torch_threads=1, torch_interop_threads=1)
    report['runtime']['precision_settings'] = {}
    for kind in ('matmul', 'conv'):
        try:
            report['runtime']['precision_settings'][kind + '_allow_hf32'] = bool(getattr(getattr(torch.npu, kind), 'allow_hf32'))
        except (AttributeError, RuntimeError):
            report['runtime']['precision_settings'][kind + '_allow_hf32'] = None
    report['real_compileoption_before'] = compile_option(torch_npu)
    report['effective_queue'] = dict(requested_environment='1', native_effective_value=None,
        native_interface_available=False, boundary='no reviewed native effective-queue getter in this wheel; environment is not runtime proof')
    report['native_after_import'] = native_proof(cann, True)
    report['affinity_after_import'] = affinity_snapshot(cpus)
    report['phase'] = 'official-model-and-cpu-reference'
    upstream = args.upstream.resolve(strict=True); sys.path.insert(0, str(upstream))
    require('dlrm_s_pytorch' not in sys.modules, 'official module was already imported')
    official = load_inference_module(upstream, report)
    for relative in SOURCE_SHA:
        module = sys.modules.get(relative[:-3].replace('/', '.'))
        require(module is not None and Path(module.__file__).resolve() == upstream / relative, 'official dependency imported outside pinned source: ' + relative)
    require(official.ext_dist.my_size <= 1, 'single-process original DLRM required')
    np.random.seed(STATE_SEED); torch.manual_seed(STATE_SEED)
    cpu = official.DLRM_Net(m_spa=16, ln_emb=np.array([1024] * 26), ln_bot=np.array([13,64,32,16]),
        ln_top=np.array([367,64,32,1]), arch_interaction_op='dot', arch_interaction_itself=False,
        sigmoid_top=2, ndevices=1).eval()
    cpu.requires_grad_(False)
    require(sum(p.numel() for p in cpu.parameters()) == PARAMETERS, 'parameter count mismatch')
    report['sparse_before'] = sparse_proof(cpu, torch)
    generator = torch.Generator(device='cpu').manual_seed(INPUT_SEED)
    dense = torch.rand((BATCH,13), generator=generator, dtype=torch.float32)
    offsets = [torch.arange(0, BATCH*2, 2, dtype=torch.int64) for _ in range(26)]
    indices = [torch.randint(1024, (BATCH*2,), generator=generator, dtype=torch.int64) for _ in range(26)]
    inputs = dense, offsets, indices
    state_sha = tensor_tree_sha256(dict(cpu.state_dict()), torch); input_sha = tensor_tree_sha256(inputs, torch)
    require(state_sha == STATE_SHA and input_sha == INPUT_SHA, 'historical original state/input identity changed')
    report['workload'] = dict(model_class='dlrm_s_pytorch.DLRM_Net', parameter_count=PARAMETERS,
        weights_kind='deterministic_random_structural_control', checkpoint=None, batch=BATCH, dtype='float32', sparse_dtype='int64',
        state_seed=STATE_SEED, input_seed=INPUT_SEED, tables=26, rows_per_table=1024, embedding_dim=16,
        bottom_mlp=[13,64,32,16], top_mlp=[367,64,32,1], interaction='dot', interaction_itself=False,
        pooling='sum', indices_per_bag=2, state_sha256=state_sha, input_tensor_tree_sha256=input_sha, sparse=True)
    report['eager_cpu'] = eager_proof(cpu, torch)
    with torch.inference_mode():
        reference = cpu(*inputs).detach().clone()
    require(list(reference.shape) == [BATCH,1] and reference.dtype == torch.float32
            and str(reference.device) == 'cpu' and bool(torch.isfinite(reference).all()), 'CPU reference shape/dtype/finite mismatch')
    report['cpu_reference_sha256'] = tensor_tree_sha256(reference, torch)
    mark_device_attempt(report, 'transfer')
    npu = copy.deepcopy(cpu).to('npu:0')
    npu_inputs = dense.to('npu:0'), [v.to('npu:0') for v in offsets], [v.to('npu:0') for v in indices]
    report['eager_before'] = eager_proof(npu, torch); report['devices_before'] = device_proof(npu, npu_inputs)
    report['sparse_npu_before'] = sparse_proof(npu, torch)
    require(tensor_tree_sha256(dict(npu.state_dict()), torch) == state_sha and tensor_tree_sha256(npu_inputs, torch) == input_sha,
            'NPU transfer changed weights or inputs')
    probe = spy = None
    if args.arm == 'worker4':
        real_provider,probe,spy = install_real_worker_model(npu,torch,torch_npu,bridge_module,report,cleanup,diagnostic=args.kind=='gate')
        bridge = real_provider.native
    report['native_before_forward'] = bridge.stats()
    caller = threading.get_native_id()
    def forward():
        require(threading.get_native_id() == caller, 'original forward caller changed')
        report['forward_calls'] += 1; mark_device_attempt(report, 'forward')
        return npu(*npu_inputs)
    with torch.inference_mode():
        report['phase'] = 'functional'
        output = forward(); torch.npu.synchronize()
        if probe is not None:
            report['functional_native_after'] = wait_real_retirement(bridge)
            report['functional_mechanism'] = validate_real_delta(report['native_before_forward'],report['functional_native_after'],1,
                                                                diagnostic=False)
        report['functional'] = {}
        validate_output(output, reference, torch, report['functional'])
        report['phase'] = 'warmup'
        for _ in range(args.warmup):
            output = forward(); torch.npu.synchronize()
        if probe is not None and args.kind in ('perf','stability'):
            report['worker_probe'] = probe.proof(); probe.restore()
            require(probe.module._native_work is probe.saved and spy is None,'cold observers must be absent from timing')
        report['eager_after_warmup'] = eager_proof(npu, torch)
        runtime_identity(torch, torch_npu, np, args.abi)
        report['affinity_before_timing'] = affinity_snapshot(cpus)
        report['native_before_timing'] = native_proof(cann, True)
        report['memory_before_timing'] = memory_snapshot(torch)
        report['real_compileoption_before_timing'] = compile_option(torch_npu)
        report['native_counters_before_timing'] = wait_real_retirement(bridge) if args.arm=='worker4' else bridge.stats()
        report['phase'] = 'descriptive-timing'
        timing = report['timing']; timing.update(warmup=args.warmup, requested_samples=args.iterations,
            scope='one fresh R22 process; cohort conclusion belongs to the four-arm controller',
            boundary='unchanged official forward entry through original device-wide synchronize return; setup/transfers excluded')
        wall_start = time.perf_counter_ns(); cpu_start = time.process_time_ns(); caller_start = time.thread_time_ns()
        if args.kind=='stability':report['stability']=dict(passed=False,checkpoint_interval=1000,checkpoints=[],excluded_from_short_performance=True)
        for iteration in range(args.iterations):
            started = time.perf_counter_ns(); output = forward(); torch.npu.synchronize()
            timing['durations_ns'].append(time.perf_counter_ns() - started)
            if args.kind=='stability' and (iteration+1)%1000==0:
                checkpoint=dict(completed=iteration+1,forward_calls=report['forward_calls'],functional={})
                report['stability']['checkpoints'].append(checkpoint)
                checkpoint_start=time.monotonic_ns()
                validate_output(output,reference,torch,checkpoint['functional'])
                checkpoint['source_files']=verify_upstream(upstream)
                checkpoint['identities']=dict(cpu_state_sha256=tensor_tree_sha256(dict(cpu.state_dict()),torch),
                    npu_state_sha256=tensor_tree_sha256(dict(npu.state_dict()),torch),
                    cpu_input_sha256=tensor_tree_sha256(inputs,torch),npu_input_sha256=tensor_tree_sha256(npu_inputs,torch))
                checkpoint['runtime']=runtime_identity(torch,torch_npu,np,args.abi)
                checkpoint['affinity']=affinity_snapshot(cpus)
                checkpoint['native']=wait_real_retirement(bridge) if args.arm=='worker4' else bridge.stats()
                checkpoint['retirement']=(validate_real_delta(report['native_before_forward'],checkpoint['native'],report['forward_calls'])
                    if args.arm=='worker4' else validate_serial_closed(checkpoint['native']))
                if args.arm=='worker4':match_real_gate(checkpoint['retirement'],report['diagnostic_gate_input']['mechanism'])
                checkpoint['memory']=memory_snapshot(torch)
                validate_checkpoint_identity(checkpoint,args.abi,cpus)
                checkpoint.update(passed=True,checkpoint_wall_ns=time.monotonic_ns()-checkpoint_start,
                    boundary='after exactly 1000 additional forward-sync calls; validation outside individual request latency')
        timing.update(common_wall_ns=time.perf_counter_ns()-wall_start, process_cpu_ns=time.process_time_ns()-cpu_start,
            caller_cpu_ns=time.thread_time_ns()-caller_start, samples=len(timing['durations_ns']),
            median_ms=statistics.median(timing['durations_ns'])/1e6, min_ms=min(timing['durations_ns'])/1e6,
            max_ms=max(timing['durations_ns'])/1e6)
        timing['p99_ms'] = sorted(timing['durations_ns'])[math.ceil(.99*len(timing['durations_ns']))-1]/1e6
        timing['batch_per_second'] = args.iterations*1e9/timing['common_wall_ns']
        timing['effective_cpu_parallelism'] = timing['process_cpu_ns']/timing['common_wall_ns']
        report['native_counters_after_timing'] = wait_real_retirement(bridge) if args.arm=='worker4' else bridge.stats()
        if args.kind=='stability':
            report['stability']['memory_trend']=contracts.validate_memory_trend(report['stability']['checkpoints'])
            report['stability']['passed']=True
            report['stability']['boundary']='one fresh 20000-forward exposure; checkpoint work included in loop wall; not short-process QPS/P99'
        if probe is not None and args.kind=='gate':
            report['worker_probe'] = probe.proof(); probe.restore()
            spy.restore()
            require(spy.record['forbidden_calls']==0,'legacy Python handoff occurred in the real native run')
        report['phase'] = 'post-run-validation'
        report['final_functional'] = {}
        validate_output(output, reference, torch, report['final_functional'])
    report['eager_after'] = eager_proof(npu, torch); report['devices_after'] = device_proof(npu, npu_inputs)
    report['sparse_npu_after'] = sparse_proof(npu, torch)
    final = dict(cpu_state_sha256=tensor_tree_sha256(dict(cpu.state_dict()), torch),
        npu_state_sha256=tensor_tree_sha256(dict(npu.state_dict()), torch),
        cpu_input_sha256=tensor_tree_sha256(inputs, torch), npu_input_sha256=tensor_tree_sha256(npu_inputs, torch))
    require(final['cpu_state_sha256'] == final['npu_state_sha256'] == state_sha
            and final['cpu_input_sha256'] == final['npu_input_sha256'] == input_sha, 'weights or inputs changed during inference')
    report['final_identities'] = final
    report['real_compileoption_after'] = compile_option(torch_npu)
    report['real_compileoption_changed'] = report['real_compileoption_before'] != report['real_compileoption_after']
    report['runtime_after'] = runtime_identity(torch, torch_npu, np, args.abi)
    report['native_after_run'] = native_proof(cann, True)
    report['affinity_after_run'] = affinity_snapshot(cpus)
    contracts.verify_topology(topology,cpus);report['topology']['verified_after_run']=True
    report['memory_after_run'] = memory_snapshot(torch)
    require(verify_upstream(upstream) == report['source']['files'], 'upstream changed during run')
    require(sha256(args.wheel) == args.expected_wheel_sha256, 'wheel artifact changed during run')
    report['final_native'] = wait_real_retirement(bridge) if args.arm=='worker4' else bridge.stats()
    if args.arm == 'worker4':
        report['mechanism'] = validate_real_delta(report['native_before_forward'],report['final_native'],report['forward_calls'],
                                                diagnostic=args.kind=='gate')
        if args.kind in ('perf','stability'): match_real_gate(report['mechanism'],report['diagnostic_gate_input']['mechanism'])
        require(report['worker_probe']['actual_workers'] == 4, 'worker4 did not execute on four actual Python worker threads')
        if args.kind=='gate':
            report['fifo_proof'] = validate_fifo(report['final_native'])
            require({row['tid'] for row in report['final_native']['real_worker_threads']} == set(report['worker_probe']['tids']),
                    'Python worker entry TIDs differ from native command-capture TIDs')
    report.update(passed=True, phase='complete', actual_workers=4 if args.arm=='worker4' else 0,
                  optimization_active=args.arm=='worker4',performance_conclusion_allowed=args.kind=='perf')


def main(argv=None):
    args = parse_args(argv)
    # Refuse existing/raw/input paths before owning any output file.
    require(not args.output.exists(), 'new output path required; previous raw is immutable')
    require(args.output.resolve() not in (Path(__file__).resolve(), args.wheel.resolve()) and
            not args.output.resolve().is_relative_to(args.upstream.resolve()), 'output cannot overwrite script/wheel/upstream')
    cpus = parse_cpus(args.cpus)
    report = dict(passed=False, phase='startup', start_epoch_ns=time.time_ns(), script_sha256=sha256(__file__),
        abi=args.abi, arm=args.arm, kind=args.kind, cpus=cpus, requested_workers=4 if args.arm=='worker4' else 0,
        actual_workers=0, optimization_active=False, device_execution_attempted=False, forward_calls=0,
        device_initialization_attempted=False, device_transfer_attempted=False, forward_attempted=False,
        performance_conclusion_allowed=False, timing={'durations_ns': []},
        requested_wheel=dict(path=str(args.wheel), sha256=args.expected_wheel_sha256),
        python=sys.version, executable=sys.executable, flags_before={k:os.environ.get(k) for k in SNAPSHOT_FLAGS},
        hostsim_environment=os.environ.get('QWEN_ACL_HOST_SIM'), ld_preload_present=bool(os.environ.get('LD_PRELOAD')),
        scope='real-device Eager random-DLRM serial/four-worker measurements; CPU overlap and speedup are separate evidence; no trained-model accuracy claim')
    cleanup = []
    try:
        require(args.warmup >= 0 and args.iterations > 0, 'warmup must be >=0 and iterations >0')
        require(re.fullmatch('[0-9a-f]{64}', args.expected_wheel_sha256), 'exact wheel SHA256 required')
        run(args, report, cleanup)
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        for close in reversed(cleanup):
            try: close()
            except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
        if cleanup_errors:
            report.update(passed=False,cleanup_errors=cleanup_errors,optimization_active=False,performance_conclusion_allowed=False)
        report.update(end_epoch_ns=time.time_ns(), flags_after={k:os.environ.get(k) for k in SNAPSHOT_FLAGS})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')
    print(json.dumps(dict(passed=report['passed'], phase=report['phase'], abi=args.abi, arm=args.arm,
        output=str(args.output), optimization_active=report['optimization_active'], forward_calls=report['forward_calls'],
        samples=len(report['timing']['durations_ns']), error=report.get('error')), ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
