import contextlib
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).parents[2] / "benchmarks/eager_prepare/runner.py"


class RunnerTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SOURCE.is_file(), "R24 runner is not implemented")
        spec = importlib.util.spec_from_file_location("r24_runner_tests", SOURCE)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_default_plan_neither_imports_framework_nor_creates_output(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new.json"
            with mock.patch.object(module, "execute", side_effect=AssertionError("planning executed")), contextlib.redirect_stdout(io.StringIO()) as stream:
                code = module.main(["--model", "branch_mlp", "--output", str(output)])
            self.assertEqual(code, 0)
            row = json.loads(stream.getvalue())
            self.assertTrue(row["plan_only"])
            self.assertEqual(row["warmup"], 20)
            self.assertEqual(row["samples"], 2000)
            self.assertFalse(output.exists())

    def test_cpu_smoke_validates_math_without_claiming_native_or_queue_coverage(self):
        self.load()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "HOST.json"
            env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
            p = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--model", "branch_mlp",
                "--host-smoke", "--samples", "3", "--warmup", "1", "--output", str(output)],
                capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            row = json.loads(output.read_text())
            self.assertTrue(row["numerical_pass"])
            self.assertFalse(row["real_npu"])
            self.assertFalse(row["performance_valid"])
            self.assertFalse(row["eligible_operator_coverage_pass"])
            self.assertEqual(row["model_core_coverage"], "none")
            self.assertEqual(len(row["timing"]["durations_ns"]), 3)

    def test_pointwise_catalog_entry_cannot_be_relabeled_as_matrix_core(self):
        module=self.load()
        with self.assertRaises(ValueError):
            module.family_mapping({"aclnnAdd":"mm"})
        self.assertEqual(module.family_mapping({"aclnnBmm":"bmm"})["aclnnBmm"],"bmm")

    def test_cold_qualification_has_fixed_bounded_trace_exposure_not_perf2000(self):
        module=self.load()
        args=module.parse_args(["--model","recommendation","--mode","on","--purpose","qualification"])
        plan=module.make_plan(args)
        self.assertEqual((plan["warmup"],plan["samples"]),(0,1))
        self.assertTrue(plan["performance_instrumentation"])

    def test_failed_close_is_not_retried_or_replaced_by_cleanup(self):
        module=self.load();state={"attempted":False};original=RuntimeError("SDK cleanup uncertain")
        adapter=mock.Mock();adapter.drain_close.side_effect=original
        with self.assertRaises(RuntimeError) as caught:module.drain_once(adapter,state)
        self.assertIs(caught.exception,original)
        self.assertTrue(state["attempted"])
        with self.assertRaises(RuntimeError):module.drain_once(adapter,state)
        adapter.drain_close.assert_called_once_with()

    def test_actual_public_off_api_is_configured_only_after_setup_sync_reference_in_inference(self):
        module = self.load(); c = module.load_file("contracts")
        self.assertTrue(hasattr(module, "prepare_inference"), "ordered setup/configure entry is missing")
        import torch
        events = []
        class Model(torch.nn.Module):
            def to(self, device):
                events.append("model_transfer"); return super().to(device)
            def forward(self, value):
                events.append("cpu_reference"); return value + 1
        # Exercise the actual public bridge, not a pretend adapter that returns
        # the artifact's Emb qualification for fresh off. Only native I/O is stubbed.
        from test_contracts import fresh_off_snapshot
        def native_configure(*args):
            events.append("native_configure")
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(args[0]); self.assertFalse(args[-2])
            return json.dumps(fresh_off_snapshot())
        public_source = SOURCE.parents[2] / "torch_npu/_eager_prepare.py"
        spec = importlib.util.spec_from_file_location("_r24_actual_public_off", public_source)
        public = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"torch_npu": SimpleNamespace(_C=SimpleNamespace(
                _npu_eager_prepare_configure=native_configure))}):
            spec.loader.exec_module(public)
        work = SimpleNamespace(model=Model(), inputs=(torch.ones(2),))
        args = module.parse_args(["--model", "branch_mlp", "--mode", "off"])
        original_move = module.move_tree
        def transfer(value, device):
            events.append("input_transfer"); return original_move(value, device)
        with mock.patch.object(module, "move_tree", side_effect=transfer):
            _, _, reference, configuration, qualification = module.prepare_inference(
                torch, work, "cpu", lambda: events.append("setup_sync"), public, args,
                dict(allowed_ops=["aclnnEmbeddingBag"], qualified_ops=["aclnnEmbeddingBag"]), c)
        self.assertLess(events.index("model_transfer"), events.index("native_configure"))
        self.assertLess(events.index("input_transfer"), events.index("setup_sync"))
        self.assertLess(events.index("setup_sync"), events.index("native_configure"))
        self.assertLess(events.index("cpu_reference"), events.index("native_configure"))
        self.assertEqual(reference.tolist(), [2, 2])
        self.assertEqual(configuration["qualified_ops"], [])
        self.assertEqual(qualification["status"], "no_active_qualification")

    def test_fresh_off_snapshot_cannot_be_used_as_on_qualification(self):
        module = self.load(); c = module.load_file("contracts")
        self.assertTrue(hasattr(module, "validate_configuration"), "mode-specific qualification check missing")
        from test_contracts import fresh_off_snapshot
        expected = dict(qualified_ops=["aclnnEmbeddingBag"])
        raw = fresh_off_snapshot()
        args = module.parse_args(["--model", "recommendation", "--mode", "on"])
        with self.assertRaises(ValueError): module.validate_configuration(raw, args, expected, c)
        args.mode = "off"
        proof = module.validate_configuration(raw, args, expected, c)
        self.assertEqual(proof["observed_qualified_ops"], [])
        self.assertFalse(proof["active_qualification"])
        raw["qualified_ops"] = ["aclnnEmbeddingBag"]
        with self.assertRaises(ValueError): module.validate_configuration(raw, args, expected, c)

    def test_hf32_is_frozen_via_actual_public_option_properties_and_drift_rejected(self):
        module = self.load(); c = module.load_file("contracts")
        self.assertTrue(hasattr(module, "freeze_precision"), "HF32 freeze is missing")
        # Execute the actual Python option properties; mock only native option I/O.
        source = SOURCE.parents[2] / "torch_npu/npu/npu_config.py"
        tree = ast.parse(source.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
            and node.name in ("_allowHF32Matmul", "_allowHF32Conv")]
        self.assertEqual(len(classes), 2)
        options = {}
        backend = SimpleNamespace(_npu_setOption=options.update,
            _npu_getOption=lambda name: options.get(name, "").encode())
        namespace = {"torch_npu": SimpleNamespace(_C=backend)}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
        npu = SimpleNamespace(matmul=namespace["_allowHF32Matmul"](), conv=namespace["_allowHF32Conv"]())
        self.assertTrue(npu.conv.allow_hf32)  # Actual default is not our FP32 protocol.
        precision = module.freeze_precision(npu, c)
        self.assertEqual(precision, dict(matmul_allow_hf32=False, conv_allow_hf32=False))
        self.assertEqual(options, dict(ALLOW_MATMUL_HF32="disable", ALLOW_CONV_HF32="disable"))
        npu.conv.allow_hf32 = True
        with self.assertRaises(ValueError): module.precision_snapshot(npu, c)

    def sdk_fixture(self, module, mappings):
        expected = dict(native=dict(path="/venv/lib/libtorch_npu.so"), cann_prefix="/approved/cann",
            sdk_files=[dict(path="/approved/cann/lib64/" + name, sha256="approved-hash")
                for name in ("libascendcl.so", "libopapi.so")])
        actual_path = module.Path
        maps_file = mock.Mock(read_text=mock.Mock(side_effect=mappings))
        def path(value):
            if str(value) == "/proc/self/maps":
                return maps_file
            return actual_path(value)
        return expected, path

    def test_actual_unapproved_provider_is_not_hidden_by_unused_approved_provider(self):
        module = self.load(); contracts = module.load_file("contracts")
        maps = "\n".join("0-1 r-xp 0 0:0 1 " + name for name in (
            "/venv/lib/libtorch_npu.so", "/approved/cann/lib64/libascendcl.so",
            "/unreviewed/cann/lib64/libopapi.so"))
        expected, path = self.sdk_fixture(module, [maps])
        with mock.patch.object(module, "Path", side_effect=path), mock.patch.object(module, "sha", return_value="approved-hash"):
            with self.assertRaisesRegex(ValueError, "unapproved.*SDK|SDK.*unapproved"):
                module.sdk_maps(expected, contracts)

    def test_lazy_unapproved_provider_is_rejected_by_end_observation(self):
        module = self.load(); contracts = module.load_file("contracts")
        initial = "\n".join("0-1 r-xp 0 0:0 1 " + name for name in (
            "/venv/lib/libtorch_npu.so", "/approved/cann/lib64/libascendcl.so"))
        expected, path = self.sdk_fixture(module, [initial, initial + "\n0-1 r-xp 0 0:0 1 /other/libopapi_nn.so"])
        with mock.patch.object(module, "Path", side_effect=path), mock.patch.object(module, "sha", return_value="approved-hash"):
            first = module.sdk_maps(expected, contracts)
            self.assertEqual(len(first), 1)  # An unused approved provider is allowed to remain absent.
            with self.assertRaisesRegex(ValueError, "unapproved.*SDK|SDK.*unapproved"):
                module.sdk_maps(expected, contracts)

    def harness_fixture(self, module, root):
        source_root = root / "benchmarks/eager_prepare"
        source_root.mkdir(parents=True)
        inventory = json.loads(SOURCE.with_name("SOURCE_PINS.json").read_text())["files"]
        for name in inventory:
            destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(("fixture " + name).encode())
        manifest = source_root / "SOURCE_PINS.json"
        row = dict(schema=1, files={name: dict(bytes=(root / name).stat().st_size,
            sha256=module.sha(root / name)) for name in inventory})
        manifest.write_text(json.dumps(row))
        expected = dict(harness_manifest=dict(path=str(manifest), sha256=module.sha(manifest)))
        return manifest, row, expected

    def test_approved_harness_rejects_prelaunch_source_drift_before_loading_helpers(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, row, expected = self.harness_fixture(module, root)
            artifact = root / "ARTIFACT.json"; artifact.write_text(json.dumps(expected))
            (root / "benchmarks/eager_prepare/models.py").write_text("unapproved prelaunch source")
            args = module.parse_args(["--model", "branch_mlp", "--execute", "--artifact", str(artifact),
                "--artifact-sha256", module.sha(artifact), "--output", str(root / "RAW.json")])
            with mock.patch.object(module, "HERE", manifest.parent), mock.patch.object(module, "load_file",
                    side_effect=AssertionError("unapproved helpers loaded")) as loader:
                with self.assertRaisesRegex(ValueError, "harness source"):
                    module.execute(args)
                loader.assert_not_called()

    def test_manifest_self_resigning_does_not_replace_artifact_approval(self):
        module = self.load()
        self.assertTrue(hasattr(module, "verify_harness"), "artifact-bound harness verification is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, row, expected = self.harness_fixture(module, root)
            changed = root / "test/eager_prepare/test_models.py"
            changed.write_text("changed and locally re-signed")
            row["files"]["test/eager_prepare/test_models.py"] = dict(bytes=changed.stat().st_size, sha256=module.sha(changed))
            manifest.write_text(json.dumps(row))
            with mock.patch.object(module, "HERE", manifest.parent), self.assertRaisesRegex(ValueError, "manifest.*digest"):
                module.verify_harness(expected)

    def test_full_source_closure_detects_change_during_execution(self):
        module = self.load()
        self.assertTrue(hasattr(module, "verify_harness"), "full source-closure verification is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, row, expected = self.harness_fixture(module, root)
            with mock.patch.object(module, "HERE", manifest.parent):
                before = module.verify_harness(expected)
                self.assertEqual(len(before["files"]), 11)
                (root / "benchmarks/eager_prepare/README.md").write_text("changed during execution")
                with self.assertRaisesRegex(ValueError, "harness source"):
                    module.verify_harness(expected)


if __name__ == "__main__":
    unittest.main()
