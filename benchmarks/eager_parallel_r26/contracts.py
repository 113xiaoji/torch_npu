"""Standard-library-only validation of R26 evidence, not evidence generation.

Artificial schema fixtures can pass these validators. Only receipts produced by
an independently qualified runtime and pinned harness can establish NPU claims.
"""
import math
import re
import statistics

ARMS = ("O", "B", "E", "EI")
COUNTERS = ("seen", "admitted", "executed", "fallback", "original_executed", "returned")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def summarize(values):
    require(bool(values) and all(integer(value, 1) for value in values), "positive integer durations required")
    ordered = sorted(values)
    return dict(count=len(values), median=statistics.median(ordered),
                p95=ordered[math.ceil(len(ordered) * .95) - 1], quantiles="nearest rank")


def validate_coverage(operators, required_per_forward, forwards):
    """Every required call executes once, returns once, and hits the new route.

    'executed' counts the candidate route, 'original_executed' the fallback
    route. They are disjoint; returned counts both. Caller supplies only the
    required optimized families (Embedding for E, the qualified chain for EI).
    """
    validate_execution_counts(operators, required_per_forward, forwards)
    for name, minimum in required_per_forward.items():
        row = operators[name]
        expected = minimum * forwards
        require(row["admitted"] == row["executed"] == expected and row["fallback"] == row["original_executed"] == 0,
                "required route fell back or did not execute exactly once: " + name)
    return dict(passed=True, forwards=forwards, required_per_forward=required_per_forward,
                all_fallback=False, scope="coverage only; no concurrency or speedup claim")


def validate_execution_counts(operators, required_per_forward, forwards):
    require(integer(forwards, 1) and isinstance(required_per_forward, dict) and bool(required_per_forward),
            "nonempty coverage exposure required")
    for name, minimum in required_per_forward.items():
        require(integer(minimum, 1), "positive per-forward target required")
        row = operators.get(name, {})
        require(all(integer(row.get(key)) for key in COUNTERS), "missing/invalid counters: " + name)
        expected = minimum * forwards
        require(row["seen"] == row["returned"] == expected, "missing/duplicate calls: " + name)
        require(row["admitted"] == row["executed"] and row["fallback"] == row["original_executed"]
                and row["executed"] + row["original_executed"] == expected,
                "candidate/original calls do not partition actual execution: " + name)


def device_overlap_proof(trace, *, provider_kind):
    result = dict(device_overlap_proven=False, dependency_join_verified=False)
    try:
        require(provider_kind == "real", "real provider required")
        require(trace.get("clock_domain") == "device" and trace.get("complete") is True,
                "complete intervals in one calibrated device clock domain required")
        intervals = trace.get("intervals", [])
        require(len(intervals) >= 3, "producer and consumer intervals required")
        nodes = {}
        for row in intervals:
            require(isinstance(row.get("id"), str) and row["id"] and row["id"] not in nodes,
                    "missing/duplicate interval identity")
            require(isinstance(row.get("stream"), str) and row["stream"], "stream identity missing")
            require(integer(row.get("start_ns")) and integer(row.get("end_ns"), 1)
                    and row["end_ns"] > row["start_ns"], "invalid device interval")
            nodes[row["id"]] = row
        edges = trace.get("dependencies", [])
        require(edges and all(isinstance(edge, (list, tuple)) and len(edge) == 2 for edge in edges),
                "explicit dependency edges required")
        require(len(set(map(tuple, edges))) == len(edges), "duplicate dependency edges")
        joins = {(row["producer"], row["consumer"]): row for row in trace.get("joins", [])}
        require(len(joins) == len(trace.get("joins", [])), "duplicate join evidence")
        for producer, consumer in edges:
            require(producer in nodes and consumer in nodes and producer != consumer, "invalid dependency identity")
            a, b = nodes[producer], nodes[consumer]
            require(a["end_ns"] <= b["start_ns"], "consumer started before producer completion")
            if a["stream"] != b["stream"]:
                join = joins.get((producer, consumer), {})
                require(integer(join.get("record_ns")) and integer(join.get("wait_ns"))
                        and a["end_ns"] <= join["record_ns"] <= join["wait_ns"] <= b["start_ns"],
                        "cross-stream event publication/join missing or out of order")
        # A shared fan-in consumer makes the overlap relevant to this pipeline.
        overlaps = []
        for index, a in enumerate(intervals):
            for b in intervals[index + 1:]:
                common = {dst for src, dst in edges if src == a["id"]} & {dst for src, dst in edges if src == b["id"]}
                if common and a["stream"] != b["stream"] and max(a["start_ns"], b["start_ns"]) < min(a["end_ns"], b["end_ns"]):
                    overlaps.append([a["id"], b["id"]])
        require(bool(overlaps), "no independent device producer overlap observed")
        return dict(result, device_overlap_proven=True, dependency_join_verified=True, overlapping_producers=overlaps)
    except (KeyError, TypeError, ValueError) as error:
        return dict(result, reason=str(error))


def validate_cohort(rows, *, rounds=10):
    """Pair by process/round; summarize process medians, never pooled samples."""
    require(integer(rounds, 1) and len(rows) == rounds * len(ARMS), "missing/extra cohort arms")
    seen, processes, identity, coverage_targets = set(), set(), None, None
    per_arm = {arm: [] for arm in ARMS}
    required_identity = {"candidate_sha256", "source_sha256", "model_sha256", "state_sha256", "input_sha256",
                         "runtime", "device", "precision"}
    precision_keys = {"autocast", "matmul_precision", "tf32", "deterministic"}
    for row in rows:
        require(row.get("schema") == 1 and integer(row.get("round")) and row["round"] < rounds,
                "invalid schema/round")
        key = row["round"], row.get("arm")
        require(key[1] in ARMS and key not in seen, "duplicate/unknown arm")
        seen.add(key)
        process = row.get("process_id")
        require(isinstance(process, str) and process and process not in processes, "fresh process identity required")
        processes.add(process)
        require(row.get("evidence_kind") == "real_npu" and row.get("provider_kind") == "real",
                "CPU/NoOp evidence cannot qualify device performance")
        require(row.get("numerical_pass") is True and row.get("diagnostic") is False,
                "numeric gate and separate non-diagnostic performance process required")
        current = row.get("identity", {})
        require(set(current) == required_identity, "complete artifact/source/state/input/runtime/device/precision identity required")
        require(all(re.fullmatch("[0-9a-f]{64}", str(current[name])) for name in required_identity if name.endswith("sha256")),
                "invalid identity SHA256")
        require(isinstance(current["precision"], dict) and set(current["precision"]) == precision_keys,
                "complete numerical precision settings required")
        require(all(type(current["precision"][name]) is bool for name in ("autocast", "tf32", "deterministic"))
                and current["precision"]["matmul_precision"] in ("highest", "high", "medium"), "invalid precision settings")
        identity = current if identity is None else identity
        require(current == identity, "paired source/artifact/model/input/runtime/device/precision differs")
        require(row.get("coverage_pass") is True, "required operator coverage gate did not pass")
        samples = row.get("samples", [])
        require(len(samples) >= 200 and integer(row.get("forward_calls"), 200)
                and row["forward_calls"] == len(samples), "sample/forward exposure differs")
        targets = row.get("required_per_forward", {})
        require(isinstance(targets, dict) and set(targets) >= {"embedding", "cat", "bmm", "index"},
                "joint pipeline counter targets required")
        coverage_targets = targets if coverage_targets is None else coverage_targets
        require(targets == coverage_targets, "paired operator coverage targets differ")
        operators = row.get("operators", {})
        validate_execution_counts(operators, targets, row["forward_calls"])
        if key[1] == "O":
            require(all(operators[name]["executed"] == 0 for name in targets), "original arm used candidate route")
        else:
            optimized = {"embedding": targets["embedding"]} if key[1] == "E" else targets
            validate_coverage(operators, optimized, row["forward_calls"])
        require(integer(row.get("sample_window_ns"), 2000000000), "sample window below two seconds")
        for name in ("process_cpu_ns", "context_switches", "peak_memory_bytes", "host_sync_count"):
            require(integer(row.get(name)), "process resource/synchronization counter missing: " + name)
        streams = row.get("expected_streams")
        require(isinstance(streams, list) and streams and len(set(streams)) == len(streams), "expected stream set required")
        require(key[1] not in ("E", "EI") or len(streams) == 3, "parallel arms require consumer plus two side streams")
        for entry in samples:
            require(type(entry.get("forward_calls")) is int and entry["forward_calls"] == 1, "sample repeats/misses forward")
            require(entry.get("cpu_window") == "forward_through_full_join"
                    and entry.get("device_window") == "start_through_all_stream_join", "incorrect timing window")
            require(integer(entry.get("cpu_wall_ns"), 1) and integer(entry.get("device_elapsed_ns"), 1), "separate CPU/device durations required")
            require(entry.get("joined_streams") == streams, "default-stream-only or incomplete device join")
        require(row["sample_window_ns"] >= sum(entry["cpu_wall_ns"] for entry in samples), "CPU samples exceed their window")
        per_arm[key[1]].append(dict(cpu_wall_ns=summarize([entry["cpu_wall_ns"] for entry in samples]),
                                    device_elapsed_ns=summarize([entry["device_elapsed_ns"] for entry in samples])))
    arms = {}
    for arm, processes_for_arm in per_arm.items():
        arms[arm] = {clock: dict(median=statistics.median(p[clock]["median"] for p in processes_for_arm),
                                p95=statistics.median(p[clock]["p95"] for p in processes_for_arm))
                     for clock in ("cpu_wall_ns", "device_elapsed_ns")}
    return dict(schema=1, rounds=rounds, arms=arms, cohort_contract_pass=True, speedup_claim_proven=False,
                aggregation="median across independent process medians/p95s; CPU and device clocks kept separate",
                boundary="receipt validation alone does not prove actual execution, overlap, or a promotion threshold")
