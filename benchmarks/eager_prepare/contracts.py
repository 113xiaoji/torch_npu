"""Evidence/statistics contracts. No framework, queue or device is imported."""
import base64
import hashlib
import math
import sys

COUNTERS = ("seen", "eligible", "queued", "prepared", "executed", "returned", "fallback")
CLOSURE_FIELDS = ("admission_closed", "fifo_drained", "device_synchronized", "all_retired", "workers_stopped", "passed")


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(values):
    require(type(values) is list and values and all(type(x) is int and x > 0 for x in values),
        "positive integer duration samples required")
    ordered = sorted(values)
    rank = lambda q: ordered[math.ceil(q * len(ordered)) - 1]
    return dict(count=len(values), mean_ns=sum(values) / len(values), p50_ns=rank(.5),
        p95_ns=rank(.95), p99_ns=rank(.99), min_ns=ordered[0], max_ns=ordered[-1], quantiles="nearest rank")


def operator_delta(before, after):
    require(before.get("schema") == after.get("schema") == 1, "queue snapshot schema1 required")
    previous, current = before.get("operators", {}), after.get("operators", {})
    require(type(previous) is dict and type(current) is dict, "operator counters required")
    result = {}
    for name in sorted(set(previous) | set(current)):
        row = {}
        for key in COUNTERS:
            a, b = previous.get(name, {}).get(key, 0), current.get(name, {}).get(key, 0)
            require(type(a) is type(b) is int and 0 <= a <= b, "operator counter rollback/invalid value: " + name + "/" + key)
            row[key] = b - a
        reasons_before = previous.get(name, {}).get("fallback_reasons", {})
        reasons_after = current.get(name, {}).get("fallback_reasons", {})
        row["fallback_reasons"] = {}
        for reason in sorted(set(reasons_before) | set(reasons_after)):
            a, b = reasons_before.get(reason, 0), reasons_after.get(reason, 0)
            require(type(a) is type(b) is int and 0 <= a <= b, "fallback reason counter rollback")
            row["fallback_reasons"][reason] = b - a
        result[name] = row
    return result


def classify_coverage(operators, required_core, family_by_operator, calls, mode):
    require(type(calls) is int and calls > 0 and mode in ("off", "on"), "invalid coverage exposure")
    totals = {key: 0 for key in COUNTERS}
    families = {}
    for name, row in operators.items():
        require(all(type(row.get(key)) is int and row[key] >= 0 for key in COUNTERS), "invalid operator delta")
        require(row["returned"] <= row["executed"] <= row["prepared"] <= row["queued"] <= row["eligible"] <= row["seen"],
            "impossible admission/execution counters")
        for key in COUNTERS:
            totals[key] += row[key]
        family = family_by_operator.get(name, "unmapped:" + name)
        family_row = families.setdefault(family, {key: 0 for key in COUNTERS})
        for key in COUNTERS:
            family_row[key] += row[key]
    covered = []
    for family, minimum in required_core.items():
        row = families.get(family, {})
        if (mode == "on" and row.get("queued", 0) >= minimum * calls
                and row.get("returned", 0) >= minimum * calls and row.get("fallback", 0) == 0):
            covered.append(family)
    any_core = mode == "on" and any(families.get(name, {}).get("returned", 0) > 0 for name in required_core)
    status = "full" if len(covered) == len(required_core) else "partial" if any_core else "none"
    eligible_pass = (mode == "on" and totals["eligible"] > 0
        and totals["eligible"] == totals["queued"] == totals["prepared"] == totals["executed"] == totals["returned"])
    return dict(eligible_operator_coverage_pass=eligible_pass, model_core_coverage=status,
        missing_core_families=sorted(set(required_core) - set(covered)), core_targets_per_forward=required_core,
        forward_calls=calls, totals=totals, by_family=families, by_operator=operators,
        all_fallback=totals["queued"] == 0 and totals["fallback"] > 0,
        mixed_execution=totals["queued"] > 0 and totals["fallback"] > 0,
        scope="counts identify queue paths; they are not SDK phase timings or a speedup claim")


def validate_close(row, capacity=8, byte_limit=536870912):
    require(row.get("schema") == 1 and row.get("first_error") is None, "queue closed with first error")
    queue, closure = row.get("queue", {}), row.get("closure", {})
    require(queue.get("closed") is True and queue.get("slots_in_use") == 0 and queue.get("retained_bytes") == 0
        and queue.get("quarantined") is False, "owned queue resources not closed")
    peaks = row.get("peaks", {})
    require(type(peaks.get("slots_in_use")) is int and 0 <= peaks["slots_in_use"] <= capacity
        and type(peaks.get("retained_bytes")) is int and 0 <= peaks["retained_bytes"] <= byte_limit,
        "queue peak exceeded slot/retained-byte limits")
    require(all(closure.get(key) is True for key in CLOSURE_FIELDS), "explicit queue/device/worker closure missing")
    return True


def validate_off_inactive(row):
    require(row.get("schema") == 1 and row.get("mode") == "off" and row.get("enabled") is False
        and row.get("first_error") is None and row.get("qualified_ops") == [] and row.get("operators") == {},
        "fresh off must have no active qualification or operator domain")
    queue = row.get("queue", {})
    require(queue.get("quarantined") is False and queue.get("first_failed_sequence") is None,
        "fresh off has a failed/quarantined domain")
    require(all(type(queue.get(key)) is int and queue[key] == 0 for key in
        ("slots_in_use", "retained_bytes", "queued", "preparing", "ready", "admitted", "retired", "workers_started")),
        "fresh off has active workers or queue resources")
    require(row.get("peaks") == dict(slots_in_use=0, retained_bytes=0), "fresh off has nonzero resource peaks")
    return True


def tensor_record(tensor, include_bytes=False):
    require(tensor.device.type == "cpu", "identity reads CPU tensors only; no hidden device read")
    value = tensor.detach().contiguous()
    data = value.numpy().tobytes(order="C")
    row = dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(data),
        byteorder=sys.byteorder, sha256=hashlib.sha256(data).hexdigest())
    if include_bytes:
        row["data_base64"] = base64.b64encode(data).decode("ascii")
    return row


def model_input_identity(model, inputs):
    def walk(value):
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {str(key): walk(item) for key, item in sorted(value.items())}
        return tensor_record(value)
    state = {name: tensor_record(value) for name, value in sorted(model.state_dict().items())}
    return dict(state=state, inputs=walk(inputs))


def validate_output(torch, actual, reference, rtol=1e-4, atol=1e-5):
    a, b = actual.detach().cpu(), reference.detach().cpu()
    same_shape, same_dtype = tuple(a.shape) == tuple(b.shape), a.dtype == b.dtype
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    valid = same_shape and same_dtype and finite and bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    max_abs = float((a - b).abs().max().item()) if same_shape and finite and a.numel() else None
    return dict(passed=valid, shape=list(a.shape), dtype=str(a.dtype), values_checked=a.numel(),
        finite=finite, same_shape=same_shape, same_dtype=same_dtype, allclose=valid,
        max_abs_error=max_abs, rtol=rtol, atol=atol,
        actual=tensor_record(a, include_bytes=True), reference=tensor_record(b, include_bytes=True))


def compare_pair(off, on):
    for row, mode in ((off, "off"), (on, "on")):
        require(all(row.get(key) is True for key in ("passed", "numerical_pass", "performance_valid", "real_npu")),
            "paired performance requires actual successful NPU receipts")
        require(row.get("plan", {}).get("mode") == mode, "pair mode differs")
        require(row["plan"].get("warmup") == 20 and row["plan"].get("samples") == 2000
            and len(row.get("timing", {}).get("durations_ns", [])) == 2000, "pair exposure differs")
        require(row.get("identity_sha256") and row.get("artifact", {}).get("sha256")
            and row.get("harness_sources") and row.get("runtime") and row.get("device"), "pair identity evidence missing")
        for key in ("precision_before_setup", "precision_before_timing", "precision_after_run"):
            precision = row.get(key, {})
            require(type(precision) is dict and set(precision) == {"matmul_allow_hf32", "conv_allow_hf32"}
                and all(value is False for value in precision.values()), "paired FP32 precision differs: " + key)
        expectation = row["artifact"].get("expectation", {})
        manifest = row.get("harness_manifest", {})
        approval = expectation.get("harness_manifest", {})
        require(approval.get("path") and approval.get("sha256")
            and {key: manifest.get(key) for key in ("path", "sha256")} == approval,
            "pair lacks artifact-bound approved harness manifest")
        require(manifest.get("files") and row["harness_sources"] == {
            name: item["sha256"] for name, item in manifest["files"].items()}, "paired source closure differs from approved manifest")
        providers = {item["path"]: item["sha256"] for item in expectation.get("sdk_files", [])}
        for key in ("sdk_mappings", "sdk_mappings_after_run"):
            mappings = row.get(key)
            require(type(mappings) is list and mappings
                and len({item["path"] for item in mappings}) == len(mappings)
                and all(item["path"] in providers and item["sha256"] == providers[item["path"]] for item in mappings),
                "pair lacks approved actual SDK mapping observation: " + key)
    for key in ("abi", "model", "seed", "round", "warmup", "samples", "cpus", "torch_threads",
                "prepare_workers", "queue_capacity", "byte_limit", "numerical_tolerance"):
        require(off["plan"].get(key) == on["plan"].get(key), "paired plan differs: " + key)
    for key in ("identity_sha256", "harness_sources", "harness_manifest", "runtime", "device"):
        require(off[key] == on[key], "paired inputs/runtime/source/device differs: " + key)
    require(off["artifact"]["sha256"] == on["artifact"]["sha256"], "paired artifact differs")
    a, b = summarize(off["timing"]["durations_ns"]), summarize(on["timing"]["durations_ns"])
    require(off["timing"]["common_wall_ns"] > 0 and on["timing"]["common_wall_ns"] > 0, "invalid loop wall")
    return dict(abi=off["plan"]["abi"], model=off["plan"]["model"], round=off["plan"]["round"],
        performance_comparison_valid=True, off=a, on=b,
        mean_latency_ratio_on_over_off=b["mean_ns"] / a["mean_ns"],
        p99_ratio_on_over_off=b["p99_ns"] / a["p99_ns"],
        throughput_ratio_on_over_off=off["timing"]["common_wall_ns"] / on["timing"]["common_wall_ns"],
        eligible_operator_coverage_pass=on.get("eligible_operator_coverage_pass", False),
        model_core_coverage=on.get("model_core_coverage", "none"), speedup_claim_proven=False,
        scope="one matched fresh-process pair; ratios are descriptive, core coverage and parallelism are separate")


def prepare_trace_proof(before, after):
    """Fresh-process post-calibration sample evidence, never a first-call claim."""
    a, b = before.get("diagnostic_trace", {}), after.get("diagnostic_trace", {})
    result = dict(prepare_parallelism_proven=False, trace_valid=False, actual_prepare_tids=[],
        scope="native Prepare inclusive CPU/wall spans; not pure SDK query or device durations")
    if a.get("enabled") is not True or b.get("enabled") is not True:
        return dict(result, reason="diagnostic trace was not enabled")
    if a.get("dropped") != 0 or b.get("dropped") != 0:
        return dict(result, reason="trace overflow cannot prove complete sampled exposure")
    try:
        previous = {row["sequence"] for row in a.get("events", [])}
        all_events = b["events"]
        require(len({row["sequence"] for row in all_events}) == len(all_events), "duplicate native sequence")
        events = [row for row in all_events if row["sequence"] not in previous]
        if not events:
            return dict(result, reason="no newly observed queued Prepare events")
        for row in events:
            require(all(type(row.get(key)) is int and row[key] > 0 for key in
                ("native_tid", "caller_tid", "consumer_tid", "start_ns", "end_ns")), "native event identity/time missing")
            require(row["end_ns"] > row["start_ns"] and type(row.get("start_thread_cpu_ns")) is int
                and type(row.get("end_thread_cpu_ns")) is int
                and row["end_thread_cpu_ns"] >= row["start_thread_cpu_ns"], "invalid native CPU/wall interval")
        tids = sorted({row["native_tid"] for row in events})
        span = max(row["end_ns"] for row in events) - min(row["start_ns"] for row in events)
        cpu = sum(row["end_thread_cpu_ns"] - row["start_thread_cpu_ns"] for row in events)
        return dict(result, trace_valid=True, prepare_parallelism_proven=len(tids) >= 2 and cpu / span > 1.02,
            actual_prepare_tids=tids, caller_tids=sorted({row["caller_tid"] for row in events}),
            consumer_tids=sorted({row["consumer_tid"] for row in events}), events=events,
            event_count=len(events), prepare_cpu_ns=cpu, prepare_span_ns=span, cpu_over_wall=cpu / span,
            four_worker_activity_observed=len(tids) == 4,
            boundary="two distinct native Prepare TIDs plus summed thread CPU/span>1.02; calibration1 then sample1, no retry")
    except (KeyError, TypeError, ValueError) as error:
        return dict(result, reason=str(error))
