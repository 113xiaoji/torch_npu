"""Disposable real-CANN lifecycle qualification; never a production bypass.

Four persistent Prepare threads own distinct plans. Another thread executes or
cancels them. Device completion precedes release of owning tensors and worker
TLS. Durations establish overlap only, never model performance.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def peak_overlap(intervals):
    events = [(start, 1) for start, _ in intervals]
    events += [(end, -1) for _, end in intervals]
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('parallel', 'cache', 'all'), default='all')
    parser.add_argument('--cache-api', choices=('v1', 'v2'), default='v1')
    parser.add_argument('--lengths', default='1,2,8,64,128,256')
    args = parser.parse_args()
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.init()
    root = Path('/usr/local/Ascend/ascend-toolkit/latest/lib64')
    nn = C.CDLL(str(root / 'libopapi_nn.so'), mode=C.RTLD_GLOBAL)
    base = C.CDLL(str(root / 'libnnopbase.so'), mode=C.RTLD_GLOBAL)
    rt = C.CDLL(str(root / 'libascendcl.so'), mode=C.RTLD_GLOBAL)
    ptr, i64, u64 = C.c_void_p, C.c_int64, C.c_uint64

    def signature(lib, name, parameters, result=C.c_int):
        function = getattr(lib, name)
        function.argtypes, function.restype = parameters, result
        return function

    get_context = signature(rt, 'aclrtGetCurrentContext', [C.POINTER(ptr)])
    set_context = signature(rt, 'aclrtSetCurrentContext', [ptr])
    create_tensor = signature(base, 'aclCreateTensor', [C.POINTER(i64), u64, C.c_int,
        C.POINTER(i64), i64, C.c_int, C.POINTER(i64), u64, ptr], ptr)
    destroy_tensor = signature(base, 'aclDestroyTensor', [ptr])
    destroy_executor = signature(base, 'aclDestroyAclOpExecutor', [ptr])
    init_huge = signature(base, 'InitHugeMemThreadLocal', [ptr, C.c_bool])
    uninit_huge = signature(base, 'UnInitHugeMemThreadLocal', [ptr, C.c_bool], None)
    release_huge = signature(base, 'ReleaseHugeMem', [ptr, C.c_bool], None)
    init_cache = signature(base, 'InitPTACacheThreadLocal', [], None)
    uninit_cache = signature(base, 'UnInitPTACacheThreadLocal', [], None)
    can_cache = signature(base, 'CanUsePTACache', [C.c_char_p], C.c_bool)
    set_key = signature(base, 'SetPTAHashKey', [u64], None)
    find_v1 = signature(base, 'PTAGetExecCache', [u64, C.POINTER(u64)], ptr)
    set_key_v2 = signature(base, 'SetPTACacheHashKey', [ptr, C.c_size_t], None)
    find_v2 = signature(base, 'PTAFindExecCache', [ptr, C.c_size_t, C.POINTER(u64)], ptr)
    add_address = signature(base, 'AddTensorAddrToCachedList', [ptr], None)
    query = signature(nn, 'aclnnEmbeddingBagGetWorkspaceSize', [ptr, ptr, ptr,
        C.c_bool, i64, C.c_bool, ptr, C.c_bool, i64, ptr, ptr, ptr, ptr,
        C.POINTER(u64), C.POINTER(ptr)])
    phase2 = signature(nn, 'aclnnEmbeddingBag', [ptr, u64, ptr, ptr])
    context = ptr()
    assert get_context(C.byref(context)) == 0 and context.value
    stream = int(torch.npu.current_stream().npu_stream)
    prepare_tids, execute_tids = set(), set()
    reports, waves = [], []
    run_namespace = 'R34-SDK-' + str(os.getpid()) + '-' + str(time.time_ns())

    class DlInfo(C.Structure):
        _fields_ = [('filename', C.c_char_p), ('base', ptr),
                    ('symbol', C.c_char_p), ('address', ptr)]

    dladdr = signature(C.CDLL(None), 'dladdr', [ptr, C.POINTER(DlInfo)])
    origins, files = {}, {}
    for function in (get_context, set_context, create_tensor, destroy_tensor,
            destroy_executor, init_huge, uninit_huge, release_huge, init_cache,
            uninit_cache, can_cache, set_key, find_v1, set_key_v2, find_v2,
            add_address, query, phase2):
        info = DlInfo()
        assert dladdr(C.cast(function, ptr), C.byref(info)) and info.filename
        path = Path(info.filename.decode()).resolve()
        origins[function.__name__] = str(path)
        if str(path) not in files:
            files[str(path)] = dict(size=path.stat().st_size,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(schema=2, passed=False, mode=args.mode, cache_api=args.cache_api,
        scope='EmbeddingBag SUM FP32 contiguous ND; sparse=True; no per-sample weights; padding_idx=-1',
        cache_allowed=bool(can_cache(b'aclnnEmbeddingBag')),
        cache_qualified=False, parallel_qualified=False,
        cache_environment={key: os.environ.get(key) for key in
            ('ACL_OP_EXEC_CACHE_SIZE', 'PTA_CACHE_ENABLE', 'TASK_QUEUE_ENABLE')},
        main_tid=threading.get_native_id(), symbol_origins=origins,
        mapped_libraries=files, reports=reports, waves=waves)

    def save():
        result['prepare_tids'] = sorted(prepare_tids)
        result['execute_tids'] = sorted(execute_tids)
        Path(args.output).write_text(json.dumps(result, indent=2))

    save()  # Preserve identity/ABI evidence even if device allocation is blocked.

    def bind():
        assert set_context(context) == 0

    def tensor(value):
        shape = (i64 * value.dim())(*value.shape)
        stride = (i64 * value.dim())(*value.stride())
        storage = (i64 * 1)(value.untyped_storage().nbytes() // value.element_size())
        handle = create_tensor(shape, value.dim(),
            {torch.float32: 0, torch.int64: 9}[value.dtype], stride,
            value.storage_offset(), 2, storage, 1,
            value.untyped_storage().data_ptr())
        assert handle, 'aclCreateTensor returned null'
        return handle

    def prepare(case, barrier, use_cache, expected_hit):
        bind()
        tid = threading.get_native_id()
        prepare_tids.add(tid)
        start = time.monotonic_ns()
        handles, size, executor = [], u64(), ptr()
        cache_active, huge_pending, hit = False, False, False
        try:
            barrier.wait(timeout=30)
            if use_cache:
                assert can_cache(b'aclnnEmbeddingBag'), 'SDK does not allow EmbeddingBag cache'
                init_cache()
                cache_active = True
                for value in case['values']:
                    add_address(value.untyped_storage().data_ptr())
                if args.cache_api == 'v1':
                    set_key(case['key'])
                    executor = ptr(find_v1(case['key'], C.byref(size)))
                else:
                    key = C.create_string_buffer(case['key_bytes'])
                    set_key_v2(key, len(case['key_bytes']))
                    executor = ptr(find_v2(key, len(case['key_bytes']), C.byref(size)))
                hit = bool(executor.value)
                assert hit == expected_hit, ('cache_hit', hit, expected_hit, case['key'])
            if not hit:
                assert init_huge(None, False) == 0
                huge_pending = True
                handles = [tensor(value) for value in case['values']]
            # Align entry to actual GetWorkspaceSize on all four miss plans.
            barrier.wait(timeout=30)
            native_start = time.monotonic_ns()
            if not hit:
                status = query(*handles[:3], False, 0, True, None, False, -1,
                    *handles[3:], C.byref(size), C.byref(executor))
                assert status == 0 and executor.value, ('query', status)
            native_end = time.monotonic_ns()
            if huge_pending:
                uninit_huge(None, False)
            if cache_active:
                uninit_cache()
                cache_active = False
            return dict(handles=handles, size=size.value, executor=executor,
                pool_pending=huge_pending, cache_hit=hit, prepare_tid=tid,
                interval=(start, time.monotonic_ns()),
                query_interval=None if hit else (native_start, native_end))
        except BaseException:
            barrier.abort()
            if executor.value:
                destroy_executor(executor)
            for handle in reversed(handles):
                destroy_tensor(handle)
            if huge_pending:
                uninit_huge(None, False)
                release_huge(None, False)
            if cache_active:
                uninit_cache()
            raise

    def consume(plan, workspace, cancel=False):
        bind()
        execute_tids.add(threading.get_native_id())
        if cancel:
            assert destroy_executor(plan['executor']) == 0
        else:
            status = phase2(workspace.data_ptr() if plan['size'] else None,
                            plan['size'], plan['executor'], stream)
            assert status == 0, ('execute', status)
        # CommonRun consumes executors; never destroy again after Execute.
        for handle in reversed(plan['handles']):
            assert destroy_tensor(handle) == 0
        if plan['pool_pending']:
            release_huge(None, False)

    torch.manual_seed(1701)
    batch, rows, dim = 128, 1024, 16
    held_cases = []  # Prevent address reuse from faking fresh-storage replay.

    def make_cases(length, wave_index, previous=None):
        cases = []
        for lane in range(4):
            weight = torch.randn(rows, dim) + wave_index / 7 + lane / 11
            ids = torch.randint(rows, (batch * length,), dtype=torch.int64)
            offsets = torch.arange(batch, dtype=torch.int64) * length
            reference = torch.nn.functional.embedding_bag(ids, weight, offsets,
                mode='sum', sparse=True)
            values = [value.npu() for value in (weight, ids, offsets)]
            values += [torch.full((batch, dim), float('nan'), device='npu'),
                torch.empty(0, dtype=torch.int64, device='npu'),
                torch.empty(batch, dtype=torch.int64, device='npu'),
                torch.empty(batch, dtype=torch.int64, device='npu')]
            addresses = [value.untyped_storage().data_ptr() for value in values]
            if previous:
                assert all(a != b for a, b in zip(addresses, previous[lane]['addresses'])
                           if a and b), 'address replay did not receive fresh storage'
            key_bytes = (run_namespace + '|length=' + str(length) + '|lane=' + str(lane)
                + '|rows=1024|dim=16|batch=128|sum|fp32|sparse|stream=' + str(stream)).encode()
            key = int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], 'little') or 1
            cases.append(dict(values=values, reference=reference, addresses=addresses,
                length=length, lane=lane, key=key, key_bytes=key_bytes))
        held_cases.append(cases)
        torch.npu.synchronize()
        return cases

    def run_wave(pools, consumer, cases, label, use_cache=False,
                 expected_hit=False, cancel=False, rotate=0):
        barrier = threading.Barrier(4)
        futures = [pools[(lane + rotate) % 4].submit(
            prepare, case, barrier, use_cache, expected_hit)
            for lane, case in enumerate(cases)]
        plans = [future.result(timeout=60) for future in futures]
        assert len({plan['prepare_tid'] for plan in plans}) == 4
        workspaces = [torch.empty(plan['size'], dtype=torch.uint8, device='npu') for plan in plans]
        torch.npu.synchronize()
        for plan, workspace in zip(plans, workspaces):
            consumer.submit(consume, plan, workspace, cancel).result(timeout=60)
        torch.npu.synchronize()
        for case, plan in zip(cases, plans):
            error = None
            if not cancel:
                actual = case['values'][3].cpu()
                torch.testing.assert_close(actual, case['reference'], atol=1e-4, rtol=1e-4)
                error = float((actual - case['reference']).abs().max())
            reports.append(dict(label=label, length=case['length'], lane=case['lane'],
                cache_key=case['key'], cache_hit=plan['cache_hit'], cancelled=cancel,
                prepare_tid=plan['prepare_tid'], addresses=case['addresses'],
                passed=True, max_abs_error=error))
        queries = [plan['query_interval'] for plan in plans if plan['query_interval']]
        waves.append(dict(label=label, length=cases[0]['length'],
            prepare_overlap=peak_overlap([plan['interval'] for plan in plans]),
            sdk_query_overlap=peak_overlap(queries), prepare_threads=4,
            cache_hits=sum(plan['cache_hit'] for plan in plans), cancelled=cancel))
        save()

    try:
        with torch.inference_mode(), ThreadPoolExecutor(1) as consumer:
            # Dedicated lanes distinguish same-worker and cross-worker replay.
            pools = [ThreadPoolExecutor(1) for _ in range(4)]
            try:
                for length in map(int, args.lengths.split(',')):
                    if args.mode in ('parallel', 'all'):
                        run_wave(pools, consumer, make_cases(length, 0), 'parallel_uncached')
                        run_wave(pools, consumer, make_cases(length, 1),
                                 'parallel_uncached_cancel', cancel=True)
                    if args.mode in ('cache', 'all'):
                        assert result['cache_allowed'], 'CANN reports cache disabled or unsupported'
                        previous = None
                        for wave_index, (label, expected, cancel, rotate) in enumerate([
                            ('cache_miss', False, False, 0),
                            ('cache_hit_new_addresses', True, False, 0),
                            ('cache_miss_other_worker', False, False, 1),
                            ('cache_hit_other_worker', True, False, 1),
                            ('cache_hit_cancel', True, True, 1),
                            ('cache_hit_after_cancel', True, False, 1)]):
                            cases = make_cases(length, wave_index + 2, previous)
                            run_wave(pools, consumer, cases, label, True, expected, cancel, rotate)
                            previous = cases
                torch.npu.synchronize()
            finally:
                for pool in pools:
                    pool.shutdown(wait=True)
        assert prepare_tids.isdisjoint(execute_tids)
        result['parallel_qualified'] = (len(prepare_tids) == 4 and bool(waves) and
            all(wave['prepare_overlap'] == 4 for wave in waves) and
            any(wave['sdk_query_overlap'] == 4 for wave in waves))
        result['cache_qualified'] = args.mode in ('cache', 'all') and all(
            any(report['label'] == label for report in reports) for label in
            ('cache_miss', 'cache_hit_new_addresses', 'cache_hit_other_worker',
             'cache_hit_cancel', 'cache_hit_after_cancel'))
        result['passed'] = result['parallel_qualified'] and (
            args.mode == 'parallel' or result['cache_qualified'])
        assert result['passed'], 'required overlap/cache evidence incomplete'
    except BaseException as error:
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        save()
        raise
    finally:
        save()
    print(json.dumps(dict(passed=result['passed'], mode=args.mode,
        cache_api=args.cache_api, parallel_qualified=result['parallel_qualified'],
        cache_qualified=result['cache_qualified'], reports=len(reports), waves=len(waves),
        prepare_tids=sorted(prepare_tids), execute_tids=sorted(execute_tids))), flush=True)


if __name__ == '__main__':
    main()
