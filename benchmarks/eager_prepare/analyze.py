"""Validate one off/on pair using exact supplied input hashes; imports no Torch."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for mode in ("off", "on"):
        p.add_argument("--" + mode, type=Path, required=True)
        p.add_argument("--" + mode + "-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("new exclusive comparison output required")
    path = Path(__file__).with_name("contracts.py")
    spec = importlib.util.spec_from_file_location("_r24_analysis_contracts", path)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    rows, inputs = [], []
    for mode in ("off", "on"):
        source = getattr(args, mode); raw = source.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        c.require(digest == getattr(args, mode + "_sha256"), "comparison input SHA mismatch")
        rows.append(json.loads(raw)); inputs.append(dict(path=str(source), sha256=digest, bytes=len(raw)))
    result = c.compare_pair(*rows); result["inputs"] = inputs
    result["contracts_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
