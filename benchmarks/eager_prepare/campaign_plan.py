"""Print a fixed R24 paired campaign. This planner never launches a process."""
import argparse
import hashlib
import json
from pathlib import Path

MODELS = ("recommendation", "branch_mlp", "branch_cnn", "transformer")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--std-artifact", type=Path, required=True)
    p.add_argument("--ft-artifact", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args(argv)
    if args.rounds != 3:
        raise ValueError("initial R24 campaign is fixed at three rounds; do not choose repeats from results")
    return args


def make_plan(args):
    cases = []
    runner = Path(__file__).with_name("runner.py").resolve()
    for abi in ("std", "ft"):
        path = getattr(args, abi + "_artifact")
        raw = path.read_bytes(); receipt = json.loads(raw)
        if receipt.get("schema") != 1 or receipt.get("abi") != abi:
            raise ValueError("artifact ABI/schema differs")
        artifact_sha = hashlib.sha256(raw).hexdigest()
        for index in range(3):
            order = MODELS[index:] + MODELS[:index]
            for name in order:
                # Counterbalance starting mode across models/ABIs, reverse in the next round.
                start_on = (index + MODELS.index(name) + (abi == "ft")) % 2
                for mode in (("on", "off") if start_on else ("off", "on")):
                    output = args.output_root / (abi + "-" + name + "-round" + str(index + 1) + "-" + mode + ".json")
                    argv = [receipt["python"], "-I", "-B", str(runner), "--execute", "--model", name,
                        "--abi", abi, "--mode", mode, "--round", str(index + 1), "--artifact", str(path),
                        "--artifact-sha256", artifact_sha, "--output", str(output)]
                    cases.append(dict(abi=abi, model=name, mode=mode, round=index + 1, argv=argv,
                        output=str(output), warmup=20, samples=2000, fresh_process=True, timeout_seconds=1800))
    return dict(schema=1, plan_only=True, cases=cases, rounds=3, benchmark_processes=48,
        concurrency=1, cpu_budget=list(range(8)), launches_processes=False,
        note="run ordered qualification first; an outer reviewed controller must own timeout/group cleanup; no auto retry")


def main(argv=None):
    print(json.dumps(make_plan(parse_args(argv)), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
