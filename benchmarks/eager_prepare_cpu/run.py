"""Bounded real-aten CPU simulation campaign; no NPU, CANN, or 3.14t claim."""
import argparse
import collections
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("direct_eager", "replay_serial", "queue_w1", "queue_w2", "queue_w4")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def verify_cpu_source_pins(manifest_path=None):
    manifest_path = manifest_path or Path(__file__).with_name("SOURCE_PINS.json")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise RuntimeError("CPU source pin mismatch: " + relative)
    return {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "files": manifest["files"]}


def arm_order(round_index):
    return list(ARMS if round_index % 2 == 0 else reversed(ARMS))


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_ns(samples):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot summarize zero samples")
    return {"samples": len(samples), "mean_us": statistics.mean(samples) / 1000,
        "p50_us": _quantile(ordered, .5) / 1000,
        "p95_us": _quantile(ordered, .95) / 1000,
        "p99_us": _quantile(ordered, .99) / 1000,
        "min_us": ordered[0] / 1000, "max_us": ordered[-1] / 1000}


def overlap_evidence(prepare_records, consume_records):
    def intersects(left, right):
        return max(left["start_ns"], right["start_ns"]) < min(left["end_ns"], right["end_ns"])
    usable = [x for x in prepare_records if x["end_ns"] > x["start_ns"]
        and (x["thread_cpu_ns"] > 0 or (x.get("thread_cycles") or 0) > 0)]
    return {"native_worker_tids": sorted({x["tid"] for x in usable}),
        "native_prepare_overlap": any(a["tid"] != b["tid"] and intersects(a, b)
            for index, a in enumerate(usable) for b in usable[index + 1:]),
        "prepare_consume_overlap": any(intersects(a, b) for a in usable for b in consume_records),
        "prepare_cpu_ns_sum": sum(x["thread_cpu_ns"] for x in usable),
        "prepare_raw_thread_cycles_sum": (sum(x.get("thread_cycles") or 0 for x in usable)
            if any(x.get("thread_cycles") is not None for x in prepare_records) else None),
        "thread_cpu_zero_records": sum(x["thread_cpu_ns"] == 0 for x in prepare_records),
        "prepare_wall_span_ns": max((x["end_ns"] for x in usable), default=0)
            - min((x["start_ns"] for x in usable), default=0),
        "records": len(prepare_records)}


def configure_environment(cpu_budget=8):
    """One process-wide CPU budget shared by producer, consumer, native, Torch."""
    prior_env = {key: os.environ.get(key) for key in THREAD_ENV}
    for key in THREAD_ENV:
        os.environ[key] = "1"
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        original = [i for i in range(ctypes.sizeof(ctypes.c_size_t) * 8) if mask.value & (1 << i)]
        selected = original[:cpu_budget]
        if not selected or not kernel.SetProcessAffinityMask(handle, sum(1 << i for i in selected)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = [i for i in range(ctypes.sizeof(ctypes.c_size_t) * 8) if mask.value & (1 << i)]
        affinity_api = "Windows Get/SetProcessAffinityMask; active processor group"
    elif hasattr(os, "sched_getaffinity"):
        original = sorted(os.sched_getaffinity(0))
        selected = original[:cpu_budget]
        os.sched_setaffinity(0, selected)
        actual = sorted(os.sched_getaffinity(0))
        affinity_api = "sched_getaffinity/sched_setaffinity"
    else:
        raise RuntimeError("cannot enforce a shared CPU budget on this platform")
    if actual != selected or len(actual) > cpu_budget:
        raise RuntimeError("CPU affinity budget was not applied")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return {"python": sys.version, "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "gil_enabled": sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True,
        "platform": platform.platform(), "machine": platform.machine(), "torch": torch.__version__,
        "torch_intra_threads": torch.get_num_threads(), "torch_inter_threads": torch.get_num_interop_threads(),
        "thread_env": {key: os.environ[key] for key in THREAD_ENV}, "prior_thread_env": prior_env,
        "affinity_api": affinity_api, "original_allowed_cpus": original,
        "selected_cpus": selected, "actual_allowed_cpus": actual,
        "shared_cpu_budget": cpu_budget, "torch_parallel_info": torch.__config__.parallel_info()}


def calibrate(native, metadata, target_us):
    """One serial calibration establishes fixed iterations shared by every arm/model."""
    for _ in range(5):
        native.prepare_inline(metadata, 2000, trace=True)

    def measure(iterations):
        records = [native.prepare_inline(metadata, iterations, trace=True) for _ in range(9)]
        return statistics.median(x["wall_ns"] for x in records), records

    zero_cpu, zero_records = measure(0)
    probe_cpu, _ = measure(2000)
    slope = max(1, probe_cpu - zero_cpu) / 2000
    result = []
    for target in target_us:
        iterations = 0 if target == 0 else max(1, round((target * 1000 - zero_cpu) / slope))
        cpu, records = measure(iterations)
        # Tune serially, never separately for any queue or model.
        for _ in range(2 if target else 0):
            effective = max(1, cpu - zero_cpu)
            iterations = max(1, round(iterations * max(1, target * 1000 - zero_cpu) / effective))
            cpu, records = measure(iterations)
        result.append({"target_us": target, "iterations": iterations,
            "observed_body_wall": summarize_ns([x["wall_ns"] for x in records]),
            "observed_body_thread_cpu": summarize_ns([x["thread_cpu_ns"] for x in records]),
            "calibration_tid": records[0]["tid"]})
    return {"scope": "single serial native busy-work calibration, fixed iterations reused across every model and arm",
        "selection_clock": "native body wall_ns; Windows GetThreadTimes can return zero for short bodies",
        "estimated_wall_ns_per_iteration": slope,
        "cost_is_simulated": True, "not_cann_get_workspace_cost": True,
        "metadata_int64_count": len(metadata),
        "metadata_sha256": hashlib.sha256(json.dumps(metadata).encode()).hexdigest(),
        "zero_iteration_body": summarize_ns([x["wall_ns"] for x in zero_records]),
        "costs": result}


def validate_snapshot(snapshot, *, count, workers, capacity, byte_limit, checksum):
    for key in ("admitted", "retired", "prepare_calls", "execute_markers", "cpu_completed"):
        if snapshot[key] != count:
            raise AssertionError("native %s differs: %r != %r" % (key, snapshot[key], count))
    if snapshot["workers_started"] != workers:
        raise AssertionError("native workers_started differs")
    if snapshot["peak_slots_in_use"] > capacity or snapshot["peak_retained_bytes"] > byte_limit:
        raise AssertionError("native queue exceeded its configured bounds")
    if snapshot["checksum_xor"] != checksum:
        raise AssertionError("native prepare checksum differs")
    for key in ("slots_in_use", "retained_bytes", "queued", "preparing", "ready", "live_workers", "cancelled"):
        if snapshot[key] != 0:
            raise AssertionError("native resources/cancellations not zero after shutdown: " + key)
    if snapshot["real_npu"] or not snapshot["simulated_prepare"]:
        raise AssertionError("CPU receipt has incorrect scope")


def cold_cpu_proof(sim, torch, native, workload, tape, calibration, args):
    """One longer, separately reported trace supplies coarse-clock CPU evidence."""
    # 50ms bodies exceed coarse Windows thread-time accounting intervals. This
    # proof is never a performance cost, arm, warmup, sample, or speedup input.
    iterations = max(1, round(50_000_000 / calibration["estimated_wall_ns_per_iteration"]))
    actual_inline = native.prepare_inline(tape.ops[0].metadata, iterations, trace=True)
    before = sim.fingerprint(workload.model, workload.inputs)
    reference = sim.reference_run(workload.model, workload.inputs)
    started = time.perf_counter_ns()
    output, audit = sim.replay(tape, native=native, workers=4, capacity=args.capacity,
        byte_limit=args.byte_limit, iterations=iterations, trace=True, reference=reference)
    elapsed = time.perf_counter_ns() - started
    sim.assert_tree_close(torch, output, reference.output)
    after = sim.fingerprint(workload.model, workload.inputs)
    if before != after:
        raise AssertionError("cold CPU proof changed model/input data")
    overlap = overlap_evidence(audit["trace"], audit["consume_trace"])
    overlap["prepare_cpu_time_divided_by_wall_span"] = (
        overlap["prepare_cpu_ns_sum"] / overlap["prepare_wall_span_ns"]
        if overlap["prepare_wall_span_ns"] else 0)
    if not overlap["native_prepare_overlap"] or not overlap["prepare_cpu_ns_sum"]:
        raise AssertionError("cold trace did not establish native prepare CPU activity/overlap")
    first_trace = next(record for record in audit["trace"] if record["seq"] == 0)
    if first_trace["checksum"] != actual_inline["checksum"]:
        raise AssertionError("cold worker and inline preparation did not perform the same work")
    checksum = 0
    for record in audit["trace"]:
        checksum ^= record["checksum"]
    validate_snapshot(audit["native"], count=len(tape.ops), workers=4,
        capacity=args.capacity, byte_limit=args.byte_limit, checksum=checksum)
    return {"scope": "separate cold proof; excluded from all formal performance samples and ratios",
        "model": workload.name, "target_body_us": 50_000, "iterations": iterations,
        "actual_inline_body_calibration": actual_inline,
        "same_native_prepare_function_and_iterations": True,
        "first_metadata_inline_worker_checksum_match": True,
        "elapsed_wall_ns_excluded": elapsed, "before_hash": before, "after_hash": after,
        "validated_calls": audit["validated_calls"], "native": audit["native"],
        "prepare_trace": audit["trace"], "consume_trace": audit["consume_trace"], "overlap": overlap}


def correctness(sim, torch, native, workload, tape, costs, args):
    before = sim.fingerprint(workload.model, workload.inputs)
    reference = sim.reference_run(workload.model, workload.inputs)
    records, checksums = [], {}
    for cost in costs:
        iterations = cost["iterations"]
        checksum = sim.expected_checksum(native, tape, iterations)
        checksums[iterations] = checksum
        for workers in (0, 1, 2, 4):
            output, audit = sim.replay(tape, native=native, workers=workers,
                capacity=args.capacity, byte_limit=args.byte_limit, iterations=iterations,
                reference=reference, trace=workers == 4)
            sim.assert_tree_close(torch, output, reference.output)
            if audit["consume_order"] != list(range(len(tape.ops))):
                raise AssertionError("CPU consumption was not in original eager order")
            if audit["checksum_xor"] != checksum:
                raise AssertionError("serial/queue prepare work differs")
            if workers:
                validate_snapshot(audit["native"], count=len(tape.ops), workers=workers,
                    capacity=args.capacity, byte_limit=args.byte_limit, checksum=checksum)
            record = {"target_us": cost["target_us"], "iterations": iterations,
                "workers": workers, "validated_calls": audit["validated_calls"],
                "final_output_full_numeric_match": True, "native": audit["native"],
                "checksum_xor": checksum}
            if workers == 4:
                record.update({"cold_prepare_trace": audit["trace"],
                    "cold_consume_trace": audit["consume_trace"],
                    "cold_overlap": overlap_evidence(audit["trace"], audit["consume_trace"])})
            records.append(record)
    after = sim.fingerprint(workload.model, workload.inputs)
    if before != after:
        raise AssertionError("model/input data changed during correctness checks")
    return {"before_hash": before, "after_hash": after, "records": records}, checksums


def measure_arm(sim, torch, native, workload, tape, arm, cost, checksum, args):
    workers = int(arm[-1]) if arm.startswith("queue_w") else 0
    runner = None if arm == "direct_eager" else sim.ReplayRunner(tape, native=native,
        workers=workers, capacity=args.capacity, byte_limit=args.byte_limit,
        iterations=cost["iterations"], trace=False)
    durations, cpu_durations = [], []
    before = sim.fingerprint(workload.model, workload.inputs)
    startup_ns = runner.startup_ns if runner is not None else 0
    try:
        with torch.inference_mode():
            for index in range(args.warmup + args.samples):
                start_cpu, start_wall = time.process_time_ns(), time.perf_counter_ns()
                if runner is None:
                    output = workload.model(*workload.inputs)
                    audit = None
                else:
                    output, audit = runner.run()
                elapsed_wall = time.perf_counter_ns() - start_wall
                elapsed_cpu = time.process_time_ns() - start_cpu
                if index >= args.warmup:
                    durations.append(elapsed_wall)
                    cpu_durations.append(elapsed_cpu)
                if runner is not None and not workers and audit["checksum_xor"] != checksum:
                    raise AssertionError("serial native prepare checksum changed")
                if index == args.warmup + args.samples - 1:
                    last_output = output
                del output, audit
    finally:
        snapshot = runner.close() if runner is not None else None
    if workers:
        count = len(tape.ops) * (args.warmup + args.samples)
        validate_snapshot(snapshot, count=count, workers=workers, capacity=args.capacity,
            byte_limit=args.byte_limit, checksum=checksum if (args.warmup + args.samples) % 2 else 0)
        if runner.session.trace_records():
            raise AssertionError("formal hot path unexpectedly collected native trace")
    with torch.inference_mode():
        sim.assert_tree_close(torch, last_output, workload.model(*workload.inputs))
    after = sim.fingerprint(workload.model, workload.inputs)
    if before != after:
        raise AssertionError("model/input data changed during timed arm")
    return {"arm": arm, "pool_startup_ns_excluded": startup_ns,
        "durations_ns": durations, "process_cpu_ns": cpu_durations,
        "wall": summarize_ns(durations), "process_cpu": summarize_ns(cpu_durations),
        "native": snapshot, "input_weight_before_hash": before, "input_weight_after_hash": after,
        "trace_enabled": False, "last_timed_output_matches_direct_eager": True}


def campaign(args):
    sys.path.insert(0, str(ROOT))
    cpu_source_pins = verify_cpu_source_pins()
    environment = configure_environment(args.cpu_budget)
    import torch
    from tools.eager_prepare_cpu import load_native
    # Import our driver by explicit path, separate from the frozen model module.
    import importlib.util
    name = "_r24_cpu_simulation"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("simulation.py"))
    sim = importlib.util.module_from_spec(spec)
    sys.modules[name] = sim
    spec.loader.exec_module(sim)
    native_started = time.perf_counter_ns()
    native = load_native()
    native_load_ns = time.perf_counter_ns() - native_started
    models, pins = sim.load_pinned_models()
    requested_models = args.models or list(models.MODEL_NAMES)
    for name in requested_models:
        if name not in models.MODEL_NAMES:
            raise ValueError("unknown pinned model: " + name)
    workloads = [models.build_workload(torch, name) for name in requested_models]
    tapes, capture_ns, initial_hashes = {}, {}, {}
    for workload in workloads:
        initial_hashes[workload.name] = sim.fingerprint(workload.model, workload.inputs)
        started = time.perf_counter_ns()
        tapes[workload.name] = sim.capture(workload.model, workload.inputs)
        capture_ns[workload.name] = time.perf_counter_ns() - started
        if initial_hashes[workload.name] != sim.fingerprint(workload.model, workload.inputs):
            raise AssertionError("capture changed original model/input data")
    calibration = calibrate(native, tapes[workloads[0].name].ops[0].metadata, args.cost_us)
    report = {"schema": 1, "real_npu": False, "simulated_prepare": True,
        "scope": "trace-driven real-aten CPU simulation of parallel metadata prepare and ordered CPU consumption",
        "limitations": ["not a live transparent eager backend", "no real CANN SDK work or NPU execution",
            "controlled CPU busy-work costs do not estimate CANN workspace preparation",
            "direct eager is context only; queue/serial ratios measure simulated preparation and replay",
            "no inference about Python 3.14 free-threading or NPU performance"],
        "ratio_definitions": {
            "serial_over_queue_p50_speedup": "replay_serial P50 / queue P50; >1 means faster queue in this CPU simulation",
            "queue_over_serial_p50_latency_ratio": "queue P50 / replay_serial P50; <1 means lower queue latency",
            "replay_serial_vs_direct_p50_overhead_ratio": "replay_serial P50 / direct_eager P50; context, not an NPU comparison"},
        "environment": environment, "source_pins": pins, "cpu_source_pins": cpu_source_pins,
        "simulation_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("simulation.py"))},
        "native_core": native.core_info(),
        "native_load_or_build_ns_excluded": native_load_ns, "calibration": calibration,
        "configuration": dict(vars(args)), "capture_ns_excluded": capture_ns, "models": {}}
    # All model/arm/cost correctness gates precede any formal performance sample.
    checksum_cache = {}
    for workload in workloads:
        tape = tapes[workload.name]
        validation, checksums = correctness(sim, torch, native, workload, tape, calibration["costs"], args)
        checksum_cache[workload.name] = checksums
        report["models"][workload.name] = {"operators": len(tape.ops),
            "operator_histogram": dict(collections.Counter(str(op.func) for op in tape.ops)),
            "metadata_bytes": [len(op.metadata) * 8 for op in tape.ops],
            "correctness": validation, "performance": []}
        print("correctness passed: " + workload.name, flush=True)
    report["cold_cpu_proof"] = cold_cpu_proof(sim, torch, native, workloads[0],
        tapes[workloads[0].name], calibration, args)
    print("separate cold native CPU proof passed", flush=True)
    gc.collect()
    for workload in workloads:
        tape, model_report = tapes[workload.name], report["models"][workload.name]
        for cost in calibration["costs"]:
            rounds = []
            for round_index in range(args.rounds):
                arms = []
                for arm in arm_order(round_index):
                    arms.append(measure_arm(sim, torch, native, workload, tape, arm, cost,
                        checksum_cache[workload.name][cost["iterations"]], args))
                rounds.append({"round": round_index + 1, "arm_order": arm_order(round_index), "arms": arms})
            summaries = {arm: summarize_ns([duration for rnd in rounds for record in rnd["arms"]
                if record["arm"] == arm for duration in record["durations_ns"]]) for arm in ARMS}
            serial_p50 = summaries["replay_serial"]["p50_us"]
            model_report["performance"].append({"target_us": cost["target_us"],
                "iterations": cost["iterations"], "rounds": rounds, "pooled_wall": summaries,
                "serial_over_queue_p50_speedup": {arm: serial_p50 / summaries[arm]["p50_us"]
                    for arm in ARMS if arm.startswith("queue_")},
                "queue_over_serial_p50_latency_ratio": {arm: summaries[arm]["p50_us"] / serial_p50
                    for arm in ARMS if arm.startswith("queue_")},
                "replay_serial_vs_direct_p50_overhead_ratio": serial_p50 / summaries["direct_eager"]["p50_us"]})
            print("measured: %s simulated target %s us" % (workload.name, cost["target_us"]), flush=True)
        final_hash = sim.fingerprint(workload.model, workload.inputs)
        if final_hash != initial_hashes[workload.name]:
            raise AssertionError("campaign changed original model/input data")
        model_report["campaign_before_hash"] = initial_hashes[workload.name]
        model_report["campaign_after_hash"] = final_hash
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--cost-us", nargs="+", type=int, default=[0, 10, 50, 200])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--byte-limit", type=int, default=1048576)
    parser.add_argument("--cpu-budget", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.rounds < 1 or args.capacity < 1 or args.byte_limit < 1:
        parser.error("samples/rounds/capacity/byte-limit must be positive; warmup nonnegative")
    if not 1 <= args.cpu_budget <= 8 or any(cost < 0 for cost in args.cost_us):
        parser.error("CPU budget must be in [1,8]; simulated costs must be nonnegative")
    report = campaign(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON serialization keeps an explicit file result, including every slow sample.
    report["configuration"]["output"] = str(args.output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CPU SIMULATION report: " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
