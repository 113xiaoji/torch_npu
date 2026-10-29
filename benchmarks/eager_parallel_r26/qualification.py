"""Pin and runtime qualification schema. Configuration is never runtime proof.

This command checks configured files/revision using the standard library. It
does not initialize NPU, infer vendor contracts, or certify a scheduling mode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SDK_CONTRACTS = ("target_stream", "event_record_submission_ack", "event_wait", "event_query",
                 "executor_lifetime", "descriptor_lifetime", "workspace_lifetime", "error_submission_state")
UNKNOWN = ("", "unknown", "none", "unqualified")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_pins(rows):
    require(isinstance(rows, list) and rows, "nonempty physical file pins required")
    found = []
    for row in rows:
        path = Path(row.get("path", "")).resolve(strict=True)
        require(path.is_file() and re.fullmatch("[0-9a-f]{64}", str(row.get("sha256", ""))), "invalid physical file pin")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(digest == row["sha256"], "physical provider/source SHA mismatch: " + str(path))
        found.append(dict(row, path=str(path), bytes=path.stat().st_size, sha256=digest))
    require(len({row["path"] for row in found}) == len(found), "duplicate physical file pins")
    return found


def qualify(config, observation=None):
    require(config.get("schema") == 1, "qualification schema 1 required")
    device, provider = config.get("device_kind"), config.get("provider_kind")
    require((device, provider) in (("cpu", "none"), ("host_sim", "noop"), ("npu", "real")),
            "device/provider mismatch: NoOp is never a real NPU provider")
    root = Path(config.get("source_root", "")).resolve(strict=True)
    revision = config.get("revision", "")
    require(re.fullmatch("[0-9a-f]{40}", revision), "exact source revision required")
    actual_revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(actual_revision == revision, "actual source revision differs")
    sources, libraries = file_pins(config.get("source_files")), file_pins(config.get("loaded_libraries"))
    require(all(Path(row["path"]).is_relative_to(root) for row in sources), "source pin escapes declared source root")
    runtime = config.get("runtime", {})
    require(isinstance(runtime, dict) and set(runtime) == {"python", "torch", "torch_npu", "driver", "cann"}
            and all(isinstance(value, str) and value for value in runtime.values()), "explicit runtime version pins required")
    if device == "npu":
        require(isinstance(config.get("soc"), str) and config["soc"].lower() not in UNKNOWN, "known pinned SoC required")
        require(all(config.get("sdk_contracts", {}).get(name) is True for name in SDK_CONTRACTS),
                "all SDK contracts need explicit qualification; booleans alone are not runtime evidence")
    report = dict(schema=1, pins_verified=True, device_kind=device, provider_kind=provider,
        compute_backend={"cpu": "cpu", "host_sim": "host_sim_noop", "npu": "npu"}[device],
        schedule_backend="unobserved", source_root=str(root), revision=revision, source_files=sources,
        configured_libraries=libraries, runtime=runtime, soc=config.get("soc"),
        real_device_qualified=False, real_device_numerical_claim=False, device_overlap_claim=False,
        performance_claim=False, runtime_observed=False, reason="configuration pins are not actual runtime qualification")
    if observation is None:
        return report
    require(isinstance(observation, dict) and observation.get("schema") == 1, "runtime observation schema 1 required")
    require(observation.get("provider_kind") == provider and observation.get("compute_backend") == report["compute_backend"],
            "actual runtime provider/compute backend differs")
    require(observation.get("runtime") == runtime and observation.get("soc") == config.get("soc"), "actual runtime versions/SoC differ")
    mapped = file_pins(observation.get("loaded_libraries"))
    expected = {(row["path"], row["sha256"]) for row in libraries}
    require({(row["path"], row["sha256"]) for row in mapped} == expected, "actual mapped provider closure differs")
    schedule = observation.get("schedule_backend")
    require(schedule in ("original", "shadow", "device_parallel"), "unknown actual scheduling backend")
    report.update(runtime_observed=True, schedule_backend=schedule, actual_loaded_libraries=mapped)
    if schedule == "device_parallel":
        require(device == "npu" and observation.get("real_sdk_qualified") is True
                and observation.get("device_parallel_qualified") is True
                and observation.get("source") == "native_runtime",
                "device parallel scheduling lacks actual native runtime qualification")
        require(observation.get("sdk_contracts") == config.get("sdk_contracts"), "actual SDK qualification differs")
        report.update(real_device_qualified=True, reason="native runtime and configured physical pins match")
    else:
        report["reason"] = "original/shadow executes its actual compute backend; parallel scheduling remains unqualified"
    return report


def require_execution_qualification(report):
    require(report.get("pins_verified") is True and report.get("runtime_observed") is True
            and report.get("real_device_qualified") is True and report.get("schedule_backend") == "device_parallel"
            and report.get("provider_kind") == "real" and report.get("compute_backend") == "npu",
            "actual runtime qualification required before R26 parallel execution")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = qualify(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
