import importlib.util
import copy
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[2] / "benchmarks/eager_prepare/contracts.py"


def load():
    if not SOURCE.is_file():
        raise AssertionError("R24 benchmark contracts are not implemented")
    spec = importlib.util.spec_from_file_location("r24_contract_tests", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_off_snapshot():
    """Actual Configure(false)/Snapshot shape: EagerPrepare.cpp State defaults,
    empty operators and Core::SnapshotData{}; no invented activated closure.
    This is host schema evidence, not a native SDK execution receipt.
    """
    return dict(schema=1, mode="off", enabled=False, qualified_ops=[], operators={},
        queue=dict(slots_in_use=0, retained_bytes=0, queued=0, preparing=0, ready=0,
            admitted=0, retired=0, closed=False, workers_started=0, quarantined=False,
            first_failed_sequence=None), peaks=dict(slots_in_use=0, retained_bytes=0),
        closure=dict(admission_closed=True, fifo_drained=False, device_synchronized=False,
            all_retired=True, workers_stopped=True, close_attempted=False, passed=True),
        first_error=None, diagnostic_trace=dict(enabled=False, limit=256, events=[], dropped=0))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.m = load()

    def test_statistics_keep_mean_and_nearest_rank_quantiles_distinct(self):
        row = self.m.summarize([1, 2, 3, 4, 100])
        self.assertEqual(row["mean_ns"], 22)
        self.assertEqual(row["p50_ns"], 3)
        self.assertEqual(row["p95_ns"], 100)
        self.assertEqual(row["p99_ns"], 100)
        for values in ([], [True], [0], [-1], [1.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.m.summarize(values)

    def classify(self, operators, required=None, calls=2, mode="on"):
        return self.m.classify_coverage(
            operators,
            required or {"embedding_bag": 2, "mm": 1},
            {"emb": "embedding_bag", "mm": "mm", "relu": "pointwise"},
            calls,
            mode,
        )

    @staticmethod
    def hit(count, **changes):
        row = dict(seen=count, eligible=count, queued=count, prepared=count,
                   executed=count, returned=count, fallback=0, fallback_reasons={})
        row.update(changes)
        return row

    def test_eligible_success_can_still_be_partial_model_coverage(self):
        row = self.classify({"emb": self.hit(4)})
        self.assertTrue(row["eligible_operator_coverage_pass"])
        self.assertEqual(row["model_core_coverage"], "partial")
        self.assertEqual(row["missing_core_families"], ["mm"])

    def test_only_pointwise_and_all_fallback_do_not_certify_model_core(self):
        row = self.classify({"relu": self.hit(8)})
        self.assertTrue(row["eligible_operator_coverage_pass"])
        self.assertEqual(row["model_core_coverage"], "none")
        row = self.classify({"emb": self.hit(4, eligible=0, queued=0,
            prepared=0, executed=0, returned=0, fallback=4,
            fallback_reasons={"not_qualified": 4})})
        self.assertFalse(row["eligible_operator_coverage_pass"])
        self.assertEqual(row["model_core_coverage"], "none")

    def test_core_full_and_eligible_fallback_are_independent(self):
        row = self.classify({"emb": self.hit(4), "mm": self.hit(2)})
        self.assertTrue(row["eligible_operator_coverage_pass"])
        self.assertEqual(row["model_core_coverage"], "full")
        row = self.classify({"emb": self.hit(4, returned=3), "mm": self.hit(2)})
        self.assertFalse(row["eligible_operator_coverage_pass"])

    def test_counter_rollback_is_rejected(self):
        before = {"schema": 1, "operators": {"mm": self.hit(3)}}
        after = {"schema": 1, "operators": {"mm": self.hit(2)}}
        with self.assertRaises(ValueError):
            self.m.operator_delta(before, after)

    def test_close_requires_explicit_full_closure_and_no_quarantine(self):
        raw = dict(schema=1, first_error=None,
            queue=dict(slots_in_use=0, retained_bytes=0, quarantined=False, closed=True),
            peaks=dict(slots_in_use=2,retained_bytes=1024),
            closure={key: True for key in ("admission_closed", "fifo_drained",
                "device_synchronized", "all_retired", "workers_stopped", "passed")})
        self.m.validate_close(raw)
        raw["queue"]["retained_bytes"] = 1
        with self.assertRaises(ValueError):
            self.m.validate_close(raw)

    def test_backpressure_peak_must_not_exceed_the_declared_resource_limits(self):
        raw = dict(schema=1, first_error=None,
            queue=dict(slots_in_use=0, retained_bytes=0, quarantined=False, closed=True),
            peaks=dict(slots_in_use=9,retained_bytes=1024),
            closure={key: True for key in ("admission_closed", "fifo_drained",
                "device_synchronized", "all_retired", "workers_stopped", "passed")})
        with self.assertRaises(ValueError):self.m.validate_close(raw)

    def test_actual_fresh_off_state_proves_inactivity_not_fabricated_active_closure(self):
        self.assertTrue(hasattr(self.m, "validate_off_inactive"), "fresh off inactivity contract is missing")
        raw = fresh_off_snapshot()
        self.assertTrue(self.m.validate_off_inactive(raw))
        self.assertFalse(raw["queue"]["closed"])
        self.assertFalse(raw["closure"]["device_synchronized"])
        for key in ("workers_started", "retained_bytes", "admitted"):
            bad=copy.deepcopy(raw); bad["queue"][key]=1
            with self.subTest(key=key), self.assertRaises(ValueError):self.m.validate_off_inactive(bad)

    def test_paired_comparison_requires_same_artifacts_inputs_and_real_npu(self):
        off = dict(passed=True, numerical_pass=True, performance_valid=True, real_npu=True,
            plan=dict(mode="off", abi="std", model="branch_mlp", seed=1, round=1, samples=2000, warmup=20),
            identity_sha256="weights-and-inputs", artifact=dict(sha256="same-artifact"),
            harness_sources={"models.py": "same-source"}, runtime=dict(soabi="same", gil_enabled=True),
            device=dict(name="same", boot_id="same", properties="same"),
            timing=dict(durations_ns=[100]*2000, common_wall_ns=200000))
        approval=dict(path="/repo/benchmarks/eager_prepare/SOURCE_PINS.json",sha256="a"*64)
        off["harness_manifest"]=dict(approval,files={"models.py":dict(bytes=1,sha256="same-source")})
        off["artifact"]["expectation"]=dict(harness_manifest=approval,
            sdk_files=[dict(path="/approved/libascendcl.so",sha256="b"*64),
                dict(path="/approved/libopapi.so",sha256="c"*64)])
        off["sdk_mappings"]=off["sdk_mappings_after_run"]=[dict(path="/approved/libascendcl.so",sha256="b"*64)]
        for key in ("precision_before_setup", "precision_before_timing", "precision_after_run"):
            off[key]=dict(matmul_allow_hf32=False, conv_allow_hf32=False)
        on = copy.deepcopy(off);on["plan"]["mode"]="on"
        on["timing"]=dict(durations_ns=[80]*2000,common_wall_ns=160000)
        on["model_core_coverage"]="partial";on["eligible_operator_coverage_pass"]=True
        pair=self.m.compare_pair(off,on)
        self.assertEqual(pair["mean_latency_ratio_on_over_off"],.8)
        self.assertEqual(pair["throughput_ratio_on_over_off"],1.25)
        self.assertEqual(pair["model_core_coverage"],"partial")
        self.assertFalse(pair["speedup_claim_proven"])
        for key,value in [("identity_sha256","different"),("artifact",{"sha256":"different"}),("real_npu",False),
                ("harness_manifest",dict(approval,sha256="d"*64)),
                ("precision_after_run",dict(matmul_allow_hf32=False,conv_allow_hf32=True)),
                ("sdk_mappings_after_run",[dict(path="/unapproved/libopapi.so",sha256="c"*64)])]:
            bad=copy.deepcopy(on);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.m.compare_pair(off,bad)
        bad=copy.deepcopy(on);del bad["harness_manifest"]
        with self.assertRaises(ValueError):self.m.compare_pair(off,bad)
        # Same approved receipt permits a legitimate additional provider to load only in on mode.
        on["sdk_mappings_after_run"].append(dict(path="/approved/libopapi.so",sha256="c"*64))
        self.assertTrue(self.m.compare_pair(off,on)["performance_comparison_valid"])

    def test_cold_trace_requires_actual_distinct_workers_and_cpu_overlap(self):
        def event(seq,tid,start,end):
            return dict(sequence=seq,operator="emb",native_tid=tid,caller_tid=10,consumer_tid=30,
                start_ns=start,end_ns=end,start_thread_cpu_ns=100,end_thread_cpu_ns=109)
        before=dict(diagnostic_trace=dict(enabled=True,dropped=0,events=[]))
        after=dict(diagnostic_trace=dict(enabled=True,dropped=0,events=[event(1,20,1,11),event(2,21,2,12)]))
        proof=self.m.prepare_trace_proof(before,after)
        self.assertTrue(proof["prepare_parallelism_proven"])
        self.assertEqual(proof["actual_prepare_tids"],[20,21])
        bad=copy.deepcopy(after);bad["diagnostic_trace"]["events"][1]["native_tid"]=20
        self.assertFalse(self.m.prepare_trace_proof(before,bad)["prepare_parallelism_proven"])
        bad=copy.deepcopy(after);bad["diagnostic_trace"]["dropped"]=1
        self.assertFalse(self.m.prepare_trace_proof(before,bad)["prepare_parallelism_proven"])


if __name__ == "__main__":
    unittest.main()
