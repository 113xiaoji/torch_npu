"""Plan O/B/E/EI fresh-process cohorts; execution stays fail-closed.

The production binding currently supports serial shadow only. It does not yet
provide real SDK qualification, all-stream timing or a qualified device runner.
No config switch in this tool fabricates those production capabilities.
"""
import argparse
import json
from pathlib import Path

ARMS = ("O", "B", "E", "EI")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_plan(config):
    require(config.get("schema") == 1, "experiment schema 1 required")
    require(config.get("modes") == list(ARMS), "joint cohort requires O/B/E/EI exactly")
    require(type(config.get("rounds")) is int and config["rounds"] >= 10, "at least ten independent paired rounds required")
    require(type(config.get("samples")) is int and config["samples"] >= 200, "at least 200 samples/process required")
    require(type(config.get("warmup")) is int and config["warmup"] >= 1, "separate warmup required")
    require(type(config.get("minimum_window_seconds")) in (int, float) and config["minimum_window_seconds"] >= 2,
            "sample window must last at least two seconds")
    require(config.get("host_prepare") is False and type(config.get("side_streams")) is int and config["side_streams"] == 2,
            "first PTA-only experiment requires host Prepare off and two side streams")
    jobs = []
    for round_id in range(config["rounds"]):
        order = ARMS[round_id % 4:] + ARMS[:round_id % 4]
        jobs.extend(dict(round=round_id, arm=arm, process="fresh", samples=config["samples"],
                         warmup=config["warmup"], minimum_window_seconds=config["minimum_window_seconds"])
                    for arm in order)
    return dict(schema=1, config=config, jobs=jobs, execution_performed=False, real_device_qualified=False,
        numerical_claim=False, device_overlap_claim=False, performance_claim=False,
        model_forward_modified=False, cann_kernel_modified=False,
        timing=dict(cpu="original forward through full join", device="device event end after all stream joins"),
        boundary="plan only; E/EI unavailable until the actual native runtime passes SDK/device qualification")


def execute(config):
    # Do not treat an input qualification JSON, NPU availability, or a shadow
    # snapshot as authority to create a multi-stream runtime. The binding's
    # current API has no qualified E/EI implementation or measurement adapter.
    raise ValueError("R26 execution unavailable: actual runtime qualification and device measurement adapter are not implemented; serial shadow is not parallel qualification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_plan(config) if args.plan_only else execute(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
