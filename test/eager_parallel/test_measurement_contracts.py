"""Receipt schema tests use artificial records, never real-device evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "benchmarks/eager_parallel_r26" / (name + ".py")
    if not path.is_file():
        raise AssertionError("R26 " + name + " is not implemented")
    spec = importlib.util.spec_from_file_location("r26_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace():
    return dict(clock_domain="device", complete=True,
        intervals=[dict(id="a", stream="s1", start_ns=10, end_ns=30),
                   dict(id="b", stream="s2", start_ns=12, end_ns=40),
                   dict(id="cat", stream="s0", start_ns=45, end_ns=60)],
        dependencies=[["a", "cat"], ["b", "cat"]],
        joins=[dict(producer="a", consumer="cat", record_ns=31, wait_ns=42),
               dict(producer="b", consumer="cat", record_ns=41, wait_ns=43)])


def sample():
    return dict(forward_calls=1, cpu_wall_ns=20000000, device_elapsed_ns=18000000,
                cpu_window="forward_through_full_join", device_window="start_through_all_stream_join",
                joined_streams=["s0", "s1", "s2"])


def rows():
    result = []
    for round_id in range(2):
        for arm in ("O", "B", "E", "EI"):
            targets = dict(embedding=26, cat=2, bmm=1, index=1)
            operators = {}
            for family, count in targets.items():
                total = 200 * count
                candidate = arm in ("B", "EI") or (arm == "E" and family == "embedding")
                operators[family] = dict(seen=total, returned=total, admitted=total if candidate else 0,
                    executed=total if candidate else 0, fallback=0 if candidate else total,
                    original_executed=0 if candidate else total)
            result.append(dict(schema=1, round=round_id, arm=arm, process_id="fixture-" + str(round_id) + arm,
                evidence_kind="real_npu", provider_kind="real", numerical_pass=True, diagnostic=False,
                identity=dict(candidate_sha256="a" * 64, source_sha256="b" * 64, model_sha256="c" * 64,
                    state_sha256="d" * 64, input_sha256="e" * 64, runtime="fixture", device="fixture",
                    precision=dict(autocast=False, matmul_precision="highest", tf32=False, deterministic=False)),
                expected_streams=["s0", "s1", "s2"], samples=[sample() for _ in range(200)],
                forward_calls=200, sample_window_ns=4000000000, process_cpu_ns=500000000,
                context_switches=3, peak_memory_bytes=1024, host_sync_count=200,
                coverage_pass=True, required_per_forward=targets, operators=operators))
    return result


class MeasurementContractTests(unittest.TestCase):
    def setUp(self):
        self.c = load("contracts")

    def test_device_overlap_requires_real_intervals_and_dependency_join(self):
        self.assertTrue(self.c.device_overlap_proof(trace(), provider_kind="real")["device_overlap_proven"])
        for mutation in ("host_clock", "missing_join", "early_consumer", "same_stream", "noop", "incomplete"):
            bad, provider = trace(), "real"
            if mutation == "host_clock": bad["clock_domain"] = "cpu"
            if mutation == "missing_join": bad["joins"].pop()
            if mutation == "early_consumer": bad["intervals"][2]["start_ns"] = 20
            if mutation == "same_stream": bad["intervals"][1]["stream"] = "s1"
            if mutation == "noop": provider = "noop"
            if mutation == "incomplete": bad["complete"] = False
            with self.subTest(mutation=mutation):
                self.assertFalse(self.c.device_overlap_proof(bad, provider_kind=provider)["device_overlap_proven"])

    def test_all_fallback_and_duplicate_execution_cannot_pass_coverage(self):
        good = dict(seen=52, admitted=52, executed=52, fallback=0, original_executed=0, returned=52)
        self.assertTrue(self.c.validate_coverage({"embedding": good}, {"embedding": 26}, 2)["passed"])
        for bad in (dict(good, admitted=0, executed=0, fallback=52, original_executed=52),
                    dict(good, executed=104), dict(good, returned=51), dict(good, seen=True)):
            with self.assertRaises(ValueError):
                self.c.validate_coverage({"embedding": bad}, {"embedding": 26}, 2)

    def test_pairing_keeps_cpu_and_device_statistics_separate(self):
        report = self.c.validate_cohort(rows(), rounds=2)
        self.assertEqual(report["rounds"], 2)
        self.assertEqual(report["arms"]["EI"]["cpu_wall_ns"]["median"], 20000000)
        self.assertEqual(report["arms"]["EI"]["device_elapsed_ns"]["median"], 18000000)
        self.assertFalse(report["speedup_claim_proven"])

    def test_cohort_rejects_missing_duplicate_mismatched_or_simulated_arms(self):
        mutations = ("missing", "duplicate", "identity", "noop", "cpu", "trace", "repeat_forward", "all_fallback",
                     "default_only", "cold_window", "short", "missing_precision", "reused_process", "forged_coverage")
        for mutation in mutations:
            bad = rows()
            if mutation == "missing": bad.pop()
            if mutation == "duplicate": bad.append(copy.deepcopy(bad[0]))
            if mutation == "identity": bad[-1]["identity"]["candidate_sha256"] = "f" * 64
            if mutation == "noop": bad[-1]["provider_kind"] = "noop"
            if mutation == "cpu": bad[-1]["evidence_kind"] = "host_sim"
            if mutation == "trace": bad[-1]["diagnostic"] = True
            if mutation == "repeat_forward": bad[-1]["samples"][0]["forward_calls"] = 2
            if mutation == "all_fallback": bad[-1]["coverage_pass"] = False
            if mutation == "default_only": bad[-1]["samples"][0]["joined_streams"] = ["s0"]
            if mutation == "cold_window": bad[-1]["samples"][0]["cpu_window"] = "cold_trace"
            if mutation == "short": bad[-1]["sample_window_ns"] = 100
            if mutation == "missing_precision": del bad[-1]["identity"]["precision"]["autocast"]
            if mutation == "reused_process": bad[-1]["process_id"] = bad[0]["process_id"]
            if mutation == "forged_coverage": bad[-1]["operators"] = copy.deepcopy(bad[0]["operators"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.c.validate_cohort(bad, rounds=2)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.q = load("qualification")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.provider = Path(self.tmp.name) / "provider.bin"
        self.provider.write_bytes(b"fixture-noop-not-a-vendor-library")
        self.pin = dict(path=str(self.provider), sha256=hashlib.sha256(self.provider.read_bytes()).hexdigest())
        source = ROOT / "benchmarks/host_regions/real_dlrm_runner_r22.py"
        self.source_pin = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        self.config = dict(schema=1, device_kind="host_sim", provider_kind="noop", soc=None,
            source_root=str(ROOT), revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_files=[dict(self.source_pin)], loaded_libraries=[dict(self.pin, role="opapi")], sdk_contracts={},
            runtime=dict(python="fixture", torch="fixture", torch_npu="fixture", driver="none", cann="noop"))

    def test_hostsim_pins_verified_but_no_device_or_numeric_claim(self):
        report = self.q.qualify(self.config)
        self.assertTrue(report["pins_verified"])
        for key in ("real_device_qualified", "real_device_numerical_claim", "device_overlap_claim", "performance_claim"):
            self.assertFalse(report[key])

    def test_wrong_provider_hash_source_revision_and_noop_npu_rejected(self):
        for mutation in ("hash", "revision", "provider", "soc", "source_escape"):
            bad = copy.deepcopy(self.config)
            if mutation == "hash": bad["loaded_libraries"][0]["sha256"] = "0" * 64
            if mutation == "revision": bad["revision"] = "0" * 40
            if mutation == "provider": bad["device_kind"] = "npu"
            if mutation == "soc": bad.update(device_kind="npu", provider_kind="real", soc="unknown")
            if mutation == "source_escape": bad["source_files"] = [self.pin]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.q.qualify(bad)

    def test_real_config_alone_cannot_qualify_actual_runtime(self):
        bad = copy.deepcopy(self.config)
        bad.update(device_kind="npu", provider_kind="real", soc="Ascend910B")
        bad["sdk_contracts"] = {name: True for name in self.q.SDK_CONTRACTS}
        report = self.q.qualify(bad)
        self.assertFalse(report["real_device_qualified"])
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.q.require_execution_qualification(report)

    def test_actual_shadow_observation_stays_unqualified_for_parallel(self):
        observation = dict(schema=1, provider_kind="noop", compute_backend="host_sim_noop",
            schedule_backend="shadow", runtime=copy.deepcopy(self.config["runtime"]), soc=None,
            loaded_libraries=[dict(self.pin, role="opapi")], real_sdk_qualified=False,
            device_parallel_qualified=False)
        report = self.q.qualify(self.config, observation)
        self.assertTrue(report["runtime_observed"])
        self.assertEqual(report["schedule_backend"], "shadow")
        self.assertFalse(report["real_device_qualified"])
        with self.assertRaises(ValueError):
            self.q.require_execution_qualification(report)
        observation["loaded_libraries"][0]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.q.qualify(self.config, observation)

    def test_plan_only_does_not_import_torch_or_require_npu(self):
        run = load("run")
        plan = run.build_plan(dict(schema=1, modes=["O", "B", "E", "EI"], rounds=10,
            samples=200, warmup=20, minimum_window_seconds=2, host_prepare=False, side_streams=2))
        self.assertFalse(plan["execution_performed"])
        self.assertEqual(len(plan["jobs"]), 40)
        self.assertFalse(plan["real_device_qualified"])
        self.assertEqual(plan["jobs"][4]["arm"], "B")
        path = Path(self.tmp.name) / "config.json"
        path.write_text(json.dumps(plan["config"]), encoding="utf-8")
        output = Path(self.tmp.name) / "plan.json"
        result = subprocess.run([sys.executable, "-S", str(ROOT / "benchmarks/eager_parallel_r26/run.py"),
            "--config", str(path), "--plan-only", "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(output.read_text())["execution_performed"])

    def test_execute_without_real_qualification_fails_before_forward(self):
        run = load("run")
        with self.assertRaisesRegex(ValueError, "qualification"):
            run.execute(dict(schema=1))


if __name__ == "__main__":
    unittest.main()
