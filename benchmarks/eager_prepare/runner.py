"""R24 inference harness. Default plans only; CPU smoke never certifies NPU work."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import sys
import sysconfig
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
TOPOLOGY_SHA = "5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a"
FAMILIES = {"aclnnEmbeddingBag": "embedding_bag", "aclnnMm": "mm",
            "aclnnAdd": "pointwise", "aclnnRelu": "pointwise"}
HARNESS_FILES = frozenset("benchmarks/eager_prepare/" + name for name in
    (".gitignore", "analyze.py", "campaign_plan.py", "contracts.py", "models.py", "README.md", "runner.py")) | frozenset(
    "test/eager_prepare/" + name for name in
    ("test_campaign_plan.py", "test_contracts.py", "test_models.py", "test_runner.py"))


def family_mapping(extra):
    if type(extra) is not dict or any(type(k) is not str or type(v) is not str for k, v in extra.items()):
        raise ValueError("explicit operator-family mapping must be a dictionary of strings")
    if any(name in FAMILIES and FAMILIES[name] != family for name, family in extra.items()):
        raise ValueError("known pointwise/core schema cannot be relabeled")
    return dict(FAMILIES, **extra)


def drain_once(adapter, state):
    if state["attempted"]:
        raise RuntimeError("queue close was already attempted; uncertain cleanup is not retried")
    state["attempted"] = True
    return adapter.drain_close()


def load_file(name, path=None):
    source = HERE / (name + ".py") if path is None else Path(path)
    spec = importlib.util.spec_from_file_location("_r24_benchmark_" + name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def approved_artifact(args):
    if args.artifact is None or not args.artifact_sha256:
        raise ValueError("explicit pinned R24 artifact receipt required")
    raw = args.artifact.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.artifact_sha256:
        raise ValueError("artifact receipt digest differs")
    return json.loads(raw)


def verify_harness(expected):
    """Pure-stdlib bootstrap, before loading any harness helper or framework."""
    approval = expected.get("harness_manifest", {})
    if (type(approval) is not dict or set(approval) != {"path", "sha256"}
            or not Path(approval["path"]).is_absolute()
            or Path(approval["path"]).resolve() != HERE.resolve() / "SOURCE_PINS.json"):
        raise ValueError("artifact-bound local harness manifest path required")
    path = Path(approval["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != approval["sha256"]:
        raise ValueError("approved harness manifest digest differs")
    manifest = json.loads(raw)
    files = manifest.get("files", {})
    if manifest.get("schema") != 1 or type(files) is not dict or set(files) != HARNESS_FILES:
        raise ValueError("complete eleven-file harness source closure required")
    repo = HERE.resolve().parents[1]
    for name, item in files.items():
        source = repo / name
        if (type(item) is not dict or set(item) != {"bytes", "sha256"}
                or type(item["bytes"]) is not int or item["bytes"] < 0
                or not source.resolve().is_relative_to(repo)
                or source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]):
            raise ValueError("approved harness source differs: " + name)
    return dict(path=approval["path"], sha256=approval["sha256"], files=files)


def write_new(path, row):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2, allow_nan=False); stream.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=("recommendation", "branch_mlp", "branch_cnn", "transformer"), required=True)
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--abi", choices=("std", "ft"), default="std")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--purpose", choices=("benchmark", "qualification"), default="benchmark")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-sha256")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--host-smoke", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    defaults = (0, 1) if args.purpose == "qualification" else (1, 3) if args.host_smoke else (20, 2000)
    if args.warmup is None:
        args.warmup = defaults[0]
    if args.samples is None:
        args.samples = defaults[1]
    return args


def make_plan(args):
    return dict(schema=1, plan_only=not (args.execute or args.host_smoke), model=args.model,
        mode=args.mode, abi=args.abi, seed=args.seed, round=args.round, warmup=args.warmup,
        samples=args.samples, purpose=args.purpose, cpus=list(range(8)), torch_threads=1, prepare_workers=4,
        queue_capacity=8, byte_limit=536870912, fresh_process_required=True,
        precision=dict(matmul_allow_hf32=False, conv_allow_hf32=False),
        numerical_tolerance=dict(rtol=1e-4, atol=1e-5), performance_instrumentation=args.purpose == "qualification",
        timing="model entry through the unchanged torch.npu.synchronize return; setup/validation excluded",
        adapter="explicit schema1 torch_npu._eager_prepare API; real qualification comes from qualified_ops",
        scope="new R24 synthetic workloads; no reuse of R22/R23 measurements")


def move_tree(value, device):
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    return value.to(device)


def precision_snapshot(npu, c):
    result = dict(matmul_allow_hf32=npu.matmul.allow_hf32, conv_allow_hf32=npu.conv.allow_hf32)
    c.require(all(value is False for value in result.values()), "fixed FP32 protocol requires matmul/conv HF32=False")
    return result


def freeze_precision(npu, c):
    npu.matmul.allow_hf32 = False
    npu.conv.allow_hf32 = False
    return precision_snapshot(npu, c)


def validate_configuration(configuration, args, expected, c):
    enabled = args.mode == "on"
    c.require(configuration.get("schema") == 1 and configuration.get("enabled") is enabled
        and configuration.get("mode") == ("eager_prepare" if enabled else "off"), "queue mode differs")
    c.require(configuration.get("diagnostic_trace", {}).get("enabled") is
        (args.purpose == "qualification" and enabled), "diagnostic trace mode differs")
    if enabled:
        c.require(set(configuration["qualified_ops"]) == set(expected["qualified_ops"]), "operator qualification changed")
    else:
        c.validate_off_inactive(configuration)
    return dict(status="artifact_qualification_matched" if enabled else "no_active_qualification",
        active_qualification=enabled, observed_qualified_ops=configuration["qualified_ops"],
        scope="on native attestation only; off does not activate or borrow the artifact's qualified set")


def prepare_inference(torch, work, device, synchronize, adapter, args, expected, c):
    """Finish unqueued transfers/reference before configuring the cold domain."""
    model = copy.deepcopy(work.model).to(device)
    inputs = move_tree(work.inputs, device)
    synchronize()
    with torch.inference_mode():
        reference = work.model(*work.inputs)
        configuration = qualification = None
        if adapter is not None:
            c.require(not torch.is_grad_enabled() and torch.is_inference_mode_enabled(),
                "queue configuration requires the inference/no-grad caller")
            configuration = adapter.configure(enabled=args.mode == "on", workers=4, capacity=8,
                byte_limit=536870912, allowed_ops=expected["allowed_ops"],
                diagnostic_trace=args.purpose == "qualification" and args.mode == "on", trace_limit=256)
            qualification = validate_configuration(configuration, args, expected, c)
    return model, inputs, reference, configuration, qualification


def affinities():
    return [dict(tid=int(p.name), cpus=sorted(os.sched_getaffinity(int(p.name))))
        for p in Path("/proc/self/task").iterdir() if p.name.isdigit()]


def validate_native_before_import(args, c):
    c.require(sys.platform == "linux" and platform.machine() == "aarch64", "real benchmark requires Linux AArch64")
    c.require(sys.version_info[:3] == (3, 14, 6), "real benchmark requires project CPython3.14.6")
    c.require(args.artifact is not None and args.artifact_sha256 and sha(args.artifact) == args.artifact_sha256,
        "explicit pinned R24 artifact receipt required")
    exposure = (0, 1) if args.purpose == "qualification" else (20, 2000)
    c.require((args.warmup, args.samples) == exposure, "fixed native benchmark/qualification exposure required")
    c.require(not os.environ.get("LD_PRELOAD") and not os.environ.get("LD_AUDIT")
        and os.environ.get("QWEN_ACL_HOST_SIM") != "1", "preload/audit/HOSTSIM is forbidden")
    c.require("PYTHON_GIL" not in os.environ and "gil" not in sys._xoptions, "natural ABI/GIL required")
    c.require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
        and os.environ.get("TORCH_NPU_EXPERIMENTAL_FREE_THREADED", "0") == ("1" if args.abi == "ft" else "0"),
        "explicit backend import and same-ABI opt-in required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        c.require(os.environ.get(key) == "1", key + " must be 1 before imports")
    c.require(os.environ.get("TASK_QUEUE_ENABLE") == "1"
        and os.environ.get("PER_STREAM_QUEUE", "0") == os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "0",
        "preserve asynchronous original task queue1")
    for key in ("FTLAB_HOST_REGIONS", "FTLAB_HOST_NATIVE_RUN", "HDB_REGION_PACKET", "HDB_REGION_REAL_OPAPI"):
        c.require(os.environ.get(key, "0") == "0", "legacy region path must be disabled: " + key)
    expected = approved_artifact(args)
    c.require(expected.get("schema") == 1 and expected.get("abi") == args.abi, "artifact ABI/schema mismatch")
    # Keep lexical venv identity; resolving the executable would hide R23's launch-context failure.
    c.require(Path(sys.executable).absolute() == Path(expected["python"]).absolute()
        and Path(sys.prefix).resolve() == Path(expected["prefix"]).resolve(), "actual venv executable/prefix differs")
    c.require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == expected["boot_id"], "host boot differs")
    c.require(set(range(8)) <= os.sched_getaffinity(0), "CPU0-7 unavailable")
    os.sched_setaffinity(0, set(range(8)))
    path = HERE.parent / "host_regions/measurement_contracts_r22.py"
    c.require(sha(path) == TOPOLOGY_SHA, "portable topology contract changed")
    topology = load_file("topology", path).capture_topology(list(range(8)))
    c.require(topology["selected_cpus"] == list(range(8)), "eight physical CPU0-7 cores required")
    for key in ("wheel", "native"):
        c.require(sha(expected[key]["path"]) == expected[key]["sha256"], key + " artifact changed")
    for row in expected["sdk_files"]:
        c.require(Path(row["path"]).is_absolute() and str(Path(row["path"]).resolve()) == row["path"],
            "approved SDK paths must be resolved canonical absolute paths")
        c.require(sha(row["path"]) == row["sha256"], "SDK provider file changed")
    return expected, topology


def verify_package(torch, torch_npu, expected, args, c):
    c.require(torch.__version__ == expected["torch_version"]
        and torch_npu.__version__ == expected["torch_npu_version"], "framework versions differ")
    soabi = "cpython-314t-aarch64-linux-gnu" if args.abi == "ft" else "cpython-314-aarch64-linux-gnu"
    c.require(sysconfig.get_config_var("SOABI") == soabi and sys._is_gil_enabled() is (args.abi == "std"),
        "actual ABI/natural GIL differs")
    package = Path(torch_npu.__file__).resolve().parent
    c.require(package.is_relative_to(Path(expected["prefix"]).resolve()), "torch_npu resolved outside candidate prefix")
    with zipfile.ZipFile(expected["wheel"]["path"]) as wheel:
        members = [entry for entry in wheel.infolist() if not entry.is_dir() and entry.filename.startswith("torch_npu/")]
        c.require(members and len({entry.filename for entry in members}) == len(members), "invalid wheel package inventory")
        for entry in members:
            c.require(".." not in Path(entry.filename).parts and "\\" not in entry.filename, "unsafe wheel path")
            path = package.parent / entry.filename
            c.require(hashlib.sha256(wheel.read(entry.filename)).hexdigest() == sha(path), "installed wheel payload differs: " + entry.filename)
    native = package / "lib/libtorch_npu.so"
    c.require(native.resolve() == Path(expected["native"]["path"]).resolve()
        and sha(native) == expected["native"]["sha256"], "loaded package native artifact differs")
    return dict(executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
        soabi=soabi, gil_enabled=sys._is_gil_enabled(), torch=torch.__version__, torch_npu=torch_npu.__version__,
        torch_path=str(Path(torch.__file__).resolve()), torch_npu_path=str(package), native_sha256=sha(native))


def sdk_maps(expected, c):
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            paths.add(str(Path(parts[5]).resolve()))
    native = str(Path(expected["native"]["path"]).resolve())
    c.require({p for p in paths if Path(p).name == "libtorch_npu.so"} == {native}, "another native DSO was interposed")
    providers = {str(Path(row["path"]).resolve()): row["sha256"] for row in expected["sdk_files"]}
    c.require(len(providers) == len(expected["sdk_files"]), "duplicate approved SDK paths")
    ascendcl = [p for p in paths if Path(p).name.startswith("libascendcl.so")]
    cann = Path(expected["cann_prefix"]).resolve()
    names = {Path(p).name for p in providers}
    provider_prefixes = ("libascendcl", "libopapi", "libcust_opapi", "libnnopbase", "libaclnn")
    # Include all private-CANN mappings and any alternate copy of an approved
    # SDK basename or known ACLNN provider, wherever the loader found it.
    actual = {p for p in paths if Path(p).is_relative_to(cann) or Path(p).name in names
        or Path(p).name.startswith(provider_prefixes)}
    c.require(ascendcl and all(Path(p).is_relative_to(cann) for p in ascendcl),
        "real private AscendCL mapping missing or differs")
    c.require(actual <= set(providers), "unapproved actual SDK/provider mapping: " + repr(sorted(actual - set(providers))))
    observed = [dict(path=p, sha256=sha(p)) for p in sorted(actual)]
    c.require(all(item["sha256"] == providers[item["path"]] for item in observed), "mapped SDK provider hash differs")
    return observed


def execute(args):
    # The outer launcher pins this bootstrap runner and the artifact digest.
    # Never execute local helpers before that artifact's approved source closure.
    harness = verify_harness(approved_artifact(args)) if args.execute else None
    c, models = load_file("contracts"), load_file("models")
    c.require(args.output is not None and not args.output.exists(), "new exclusive result file required")
    c.require(args.samples > 0 and args.warmup >= 0 and args.round > 0, "invalid exposure")
    expected = topology = None
    if args.execute:
        expected, topology = validate_native_before_import(args, c)
        for protected in (expected["prefix"], expected["cann_prefix"]):
            c.require(not args.output.resolve().is_relative_to(Path(protected).resolve()), "output overlaps runtime/SDK")
    else:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row = dict(schema=1, passed=False, numerical_pass=False, eligible_operator_coverage_pass=False,
        model_core_coverage="none", performance_valid=False, real_npu=False,
        prepare_parallelism_proven=False,
        purpose=("real_npu_" + args.purpose) if args.execute else "cpu_host_smoke", plan=make_plan(args),
        start_ns=time.time_ns(), phase="imports", timing=dict(durations_ns=[]),
        harness_sources={name: sha(HERE / name) for name in ("runner.py", "models.py", "contracts.py")})
    if expected is not None:
        row["artifact"] = dict(path=str(args.artifact), sha256=args.artifact_sha256, expectation=expected)
        row["harness_manifest"] = harness
        row["harness_sources"] = {name: item["sha256"] for name, item in harness["files"].items()}
    adapter = None
    close_state = dict(attempted=False)
    try:
        if args.execute:
            # Preserve the already-qualified AArch64 static-TLS import order.
            import sklearn.metrics
            row["dependency_import_order"] = "sklearn.metrics before torch and torch_npu"
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        row["torch_threads"] = dict(intra=torch.get_num_threads(), inter=torch.get_num_interop_threads())
        row["host_runtime"] = dict(python=sys.version, executable=sys.executable, torch=torch.__version__)
        if args.execute:
            import torch_npu
            row["runtime"] = verify_package(torch, torch_npu, expected, args, c)
            row["phase"] = "device-initialization"
            c.require(torch.npu.is_available(), "real NPU unavailable")
            torch.npu.set_device(0)
            row["device"] = dict(name=torch.npu.get_device_name(0), properties=str(torch.npu.get_device_properties(0)),
                count=torch.npu.device_count(), boot_id=expected["boot_id"], topology=topology)
            c.require(row["device"]["name"] == expected["device_name"], "actual NPU device differs")
            row["precision_before_setup"] = freeze_precision(torch.npu, c)
            adapter = importlib.import_module("torch_npu._eager_prepare")
            row["native_before_setup"] = adapter.snapshot()
            c.validate_off_inactive(row["native_before_setup"])
            synchronize = torch.npu.synchronize
            row["real_npu"] = True
        else:
            synchronize = lambda: None
        row["phase"] = "model-setup"
        work = models.build_workload(torch, args.model, args.seed)
        row["identity"] = c.model_input_identity(work.model, work.inputs)
        row["identity_sha256"] = hashlib.sha256(json.dumps(row["identity"], sort_keys=True).encode()).hexdigest()
        row["required_core_per_forward"] = work.required_core
        model, inputs, reference, configuration, qualification = prepare_inference(
            torch, work, "npu:0" if args.execute else "cpu", synchronize, adapter, args, expected, c)
        if args.execute:
            row["configuration"], row["qualification"] = configuration, qualification
            c.require(all(p.device.type == "npu" and p.device.index == 0 for p in model.parameters()), "model parameters not on actual NPU0")
        with torch.inference_mode():
            row["phase"] = "calibration"
            output = model(*inputs); synchronize()
            row["calibration"] = c.validate_output(torch, output, reference)
            c.require(row["calibration"]["passed"], "full calibration values differ")
            if args.execute:
                row["sdk_mappings"] = sdk_maps(expected, c)
            for _ in range(args.warmup):
                output = model(*inputs); synchronize()
            before = adapter.snapshot() if adapter else None
            if args.execute:
                row["precision_before_timing"] = precision_snapshot(torch.npu, c)
                row["affinity_before_timing"] = affinities()
                c.require(all(set(x["cpus"]) <= set(range(8)) for x in row["affinity_before_timing"]), "thread escaped total CPU0-7 budget")
                c.require(sys._is_gil_enabled() is (args.abi == "std"), "natural GIL changed before timing")
            row["phase"] = "timing"
            wall_start, cpu_start, caller_start = time.perf_counter_ns(), time.process_time_ns(), time.thread_time_ns()
            for _ in range(args.samples):
                start = time.perf_counter_ns(); output = model(*inputs); synchronize()
                row["timing"]["durations_ns"].append(time.perf_counter_ns() - start)
            row["timing"].update(common_wall_ns=time.perf_counter_ns() - wall_start,
                process_cpu_ns=time.process_time_ns() - cpu_start, caller_cpu_ns=time.thread_time_ns() - caller_start)
            row["timing"].update(c.summarize(row["timing"]["durations_ns"]))
            row["timing"]["batch_per_second"] = args.samples * 1e9 / row["timing"]["common_wall_ns"]
            row["forward_calls"] = 1 + args.warmup + args.samples
            row["phase"] = "retirement"
            if adapter:
                after = adapter.snapshot(); row["native_before_timing"], row["native_after_timing"] = before, after
                row["closure"] = drain_once(adapter, close_state)
                if args.mode == "on":
                    c.validate_close(row["closure"])
                    row["closure_kind"] = "active_domain_retired"
                else:
                    for state in (before, after, row["closure"]):
                        c.validate_off_inactive(state)
                    row["closure_kind"] = "fresh_off_no_active_domain"
                c.require(before.get("first_error") is None and after.get("first_error") is None, "queue failed during timing")
                delta = c.operator_delta(before, after)
                for name, counters in delta.items():
                    c.require(not counters["queued"] or (name in row["configuration"]["qualified_ops"]
                        and name in row["configuration"]["allowed_ops"]), "unqualified operator entered queue")
                families = family_mapping(expected.get("operator_families", {}))
                row["coverage"] = c.classify_coverage(delta, work.required_core,
                    families, args.samples, args.mode)
                if args.purpose == "qualification":
                    row["prepare_parallelism"] = c.prepare_trace_proof(before, after)
                    row["prepare_parallelism_proven"] = row["prepare_parallelism"]["prepare_parallelism_proven"]
                row.update({key: row["coverage"][key] for key in ("eligible_operator_coverage_pass", "model_core_coverage")})
                row["affinity_after_run"] = affinities()
                c.require(all(set(x["cpus"]) <= set(range(8)) for x in row["affinity_after_run"]), "thread escaped CPU budget")
                c.require(sys._is_gil_enabled() is (args.abi == "std"), "natural GIL changed during run")
                row["gil_after_run"] = sys._is_gil_enabled()
                row["precision_after_run"] = precision_snapshot(torch.npu, c)
            row["final_numerical"] = c.validate_output(torch, output, reference)
            observed = c.model_input_identity(type("State", (), {"state_dict": lambda self: {
                k: v.detach().cpu() for k, v in model.state_dict().items()}})(), move_tree(inputs, "cpu"))
            c.require(observed == row["identity"], "weights or inputs changed")
            row["numerical_pass"] = row["final_numerical"]["passed"]
            c.require(row["numerical_pass"], "full final output values differ")
            if args.execute:
                row["sdk_mappings_after_run"] = sdk_maps(expected, c)
                c.require(sha(args.artifact) == args.artifact_sha256, "artifact receipt changed")
                for key in ("wheel", "native"):
                    c.require(sha(expected[key]["path"]) == expected[key]["sha256"], "artifact changed during run")
                for item in expected["sdk_files"]:
                    c.require(sha(item["path"]) == item["sha256"], "SDK provider changed during run")
                c.require(verify_harness(expected) == harness, "approved harness identity changed during run")
            else:
                c.require(all(sha(HERE / name) == value for name, value in row["harness_sources"].items()), "harness source changed")
            row["performance_valid"] = bool(args.execute and args.purpose == "benchmark")
            row["passed"] = True
            row["phase"] = "complete"
            row["scope"] = "numeric/measurement validity is independent of full/partial/none core queue coverage; no speedup assertion"
    except BaseException as error:
        row.update(passed=False, performance_valid=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if adapter and not close_state["attempted"]:
            try:
                row["failure_cleanup"] = drain_once(adapter, close_state)
            except BaseException as error:
                row["cleanup_error"] = dict(type=type(error).__name__, error=str(error))
        row["close_attempted"] = close_state["attempted"]
        row["end_ns"] = time.time_ns()
        write_new(args.output, row)
    return row


def main(argv=None):
    args = parse_args(argv)
    if not args.execute and not args.host_smoke:
        print(json.dumps(make_plan(args), indent=2)); return 0
    previous = None
    if hasattr(signal, "SIGTERM"):
        def stop(_signum, _frame):
            raise KeyboardInterrupt("R24 runner received SIGTERM")
        previous = signal.signal(signal.SIGTERM, stop)
    try:
        result = execute(args)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    print(json.dumps({key: result[key] for key in ("passed", "numerical_pass", "eligible_operator_coverage_pass",
        "model_core_coverage", "performance_valid", "real_npu")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
