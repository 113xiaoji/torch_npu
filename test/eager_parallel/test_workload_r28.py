"""The L sweep changes bag cardinality while retaining the frozen workload."""
import importlib.util
import os
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
INPUT_HASHES = {
    2: '239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6',
    8: '354bb7c9746244dc8a22386447e00250df317fcca7f789dbde2279d9ee1e2373',
    16: 'db62db6ca45db2105c8e5e6b6ef87ca5199ffa9b0487646cf1b2c9407cc087de',
    32: '21978f4f7e17dbab0dc1e241649536b32d44b73d29197f1678c8191d0a7e312b',
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_inputs():
    generator = torch.Generator().manual_seed(20260918)
    dense = torch.rand((128, 13), generator=generator, dtype=torch.float32)
    offsets = [torch.arange(0, 256, 2, dtype=torch.int64) for _ in range(26)]
    indices = [torch.randint(1024, (256,), generator=generator, dtype=torch.int64) for _ in range(26)]
    return dense, offsets, indices


class Workload(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'benchmarks/eager_parallel_r28/workload.py'
        self.assertTrue(path.is_file(), 'R28 bag-length workload adapter is missing')
        self.module = load(path, 'r28_workload_test')
        self.reference = load(ROOT / 'benchmarks/eager_parallel_r26/reference.py', 'r28_base_test')
        self.helper = self.reference.load_frozen_helper()

    def test_l2_retains_exact_original_input_bytes(self):
        inputs = original_inputs()
        result = self.module.build_inputs(inputs, bag_length=2)
        self.assertIs(result, inputs)
        self.assertEqual(self.helper.tensor_tree_sha256(result, torch), self.reference.ORIGINAL_INPUT_SHA256)

    def test_all_lengths_keep_dense_and_each_bags_original_and_nested_ids(self):
        original = original_inputs()
        original_sha = self.helper.tensor_tree_sha256(original, torch)
        hashes, previous = [], None
        for length in (2, 8, 16, 32):
            inputs = self.module.build_inputs(original, bag_length=length)
            self.assertIs(inputs[0], original[0])
            for table in range(26):
                ids, offsets = inputs[2][table], inputs[1][table]
                self.assertEqual(ids.shape, (128 * length,))
                self.assertEqual(offsets.shape, (128,))
                self.assertEqual(ids.dtype, torch.int64)
                self.assertEqual(offsets.dtype, torch.int64)
                self.assertTrue(torch.equal(offsets, torch.arange(0, 128 * length, length)))
                self.assertTrue(torch.equal(ids.reshape(128, length)[:, :2], original[2][table].reshape(128, 2)))
                self.assertTrue(bool(((ids >= 0) & (ids < 1024)).all()))
                if previous is not None:
                    self.assertTrue(torch.equal(ids.reshape(128, length)[:, :previous[0]],
                                                previous[1][2][table].reshape(128, previous[0])))
            digest = self.helper.tensor_tree_sha256(inputs, torch)
            self.assertEqual(digest, INPUT_HASHES[length])
            repeated = self.module.build_inputs(original, bag_length=length)
            self.assertEqual(self.helper.tensor_tree_sha256(repeated, torch), digest)
            hashes.append(digest)
            previous = length, inputs
        self.assertEqual(len(set(hashes)), 4)
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), original_sha)

    def test_extra_ids_use_one_fixed_maximum_length_draw_for_every_table(self):
        inputs = self.module.build_inputs(original_inputs(), bag_length=8)
        generator = torch.Generator().manual_seed(20260919)
        for table in range(26):
            extras = torch.randint(1024, (128, 30), generator=generator, dtype=torch.int64)
            self.assertTrue(torch.equal(inputs[2][table].reshape(128, 8)[:, 2:], extras[:, :6]))

    def test_only_predeclared_lengths_and_frozen_inputs_are_admitted(self):
        for length in (True, 2.0, 0, 1, 4, 64, '8', None):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'bag_length'):
                self.module.build_inputs(original_inputs(), bag_length=length)
        inputs = original_inputs()
        inputs[2][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'original input'):
            self.module.build_inputs(inputs, bag_length=8)

    def test_changed_frozen_reference_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'eager_parallel_r28/workload.py'
            adapter.parent.mkdir()
            adapter.write_bytes(Path(self.module.__file__).read_bytes())
            reference = root / 'eager_parallel_r26/reference.py'
            reference.parent.mkdir()
            reference.write_text("raise RuntimeError('unverified reference executed')\n")
            with self.assertRaisesRegex(ValueError, 'frozen R26 reference SHA mismatch'):
                load(adapter, 'r28_untrusted_reference')

    @unittest.skipUnless(os.environ.get('R28_ORIGINAL_SOURCE'), 'complete pinned original source is not configured')
    def test_original_loader_preserves_model_state_and_dense_across_all_lengths(self):
        # The frozen original loader intentionally admits only one load per
        # process. Exercise every L in its own interpreter, like the campaign.
        selected = os.environ.get('R28_SINGLE_LENGTH')
        if selected is None:
            code = (
                'import importlib.util,json,sys,unittest;'
                'sys.path[:]=json.loads(sys.argv.pop());'
                'path=sys.argv.pop();sys.argv=[path];'
                's=importlib.util.spec_from_file_location("r28_original_child",path);'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
                'suite=unittest.TestSuite([m.Workload("test_original_loader_preserves_model_state_and_dense_across_all_lengths")]);'
                'result=unittest.TextTestRunner(verbosity=2).run(suite);'
                'raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)'
            )
            for length in (2, 8, 16, 32):
                environment = dict(os.environ, R28_SINGLE_LENGTH=str(length))
                result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve()),
                    json.dumps(sys.path)], env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, f'L={length}\n' + result.stdout + result.stderr)
            return
        upstream = os.environ['R28_ORIGINAL_SOURCE']
        previous = None
        for length in (int(selected),):
            model, inputs, provenance = self.module.load_original_model(upstream, bag_length=length)
            identity, workload = provenance['identity'], provenance['workload']
            self.assertEqual(identity['model_sha256'], self.reference.ORIGINAL_MODEL_SHA256)
            self.assertEqual(identity['state_sha256'], self.reference.ORIGINAL_STATE_SHA256)
            self.assertEqual(identity['dense_sha256'], self.module.DENSE_SHA256)
            self.assertEqual(identity['input_sha256'], INPUT_HASHES[length])
            self.assertEqual(self.helper.tensor_tree_sha256(dict(model.state_dict()), torch), identity['state_sha256'])
            self.assertEqual(workload['bag_length'], length)
            self.assertEqual((workload['batch'], workload['tables'], workload['embedding_dim'], workload['vocabulary']),
                             (128, 26, 16, 1024))
            self.assertEqual(len(workload['tables_detail']), 26)
            for table in workload['tables_detail']:
                self.assertEqual(table['indices_shape'], [128 * length])
                self.assertEqual(table['logical_indices_shape'], [128, length])
                self.assertEqual(table['offsets_shape'], [128])
                self.assertEqual(table['offset_first'], 0)
                self.assertEqual(table['offset_last'], 127 * length)
                self.assertEqual(table['bag_length_min'], length)
                self.assertEqual(table['bag_length_max'], length)
            if previous is not None:
                self.assertTrue(torch.equal(inputs[0], previous[1][0]))
                for current, prior in zip(inputs[2], previous[1][2]):
                    self.assertTrue(torch.equal(current.reshape(128, length)[:, :previous[0]],
                                                prior.reshape(128, previous[0])))
            with torch.inference_mode():
                expected = model(*inputs)
                stages = self.module.capture_stages(model, inputs)
                self.assertTrue(self.module.error_record(stages['output'], expected, rtol=1e-4, atol=1e-5)['passed'])
                self.assertTrue(self.module.validate_stages(model, inputs, stages, lambda: None)['passed'])
            previous = length, inputs

    def test_stage_checks_detect_bag_fault_hidden_by_saturated_sigmoid(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bot_l = torch.nn.Identity()
                self.emb_l = torch.nn.ModuleList([torch.nn.EmbeddingBag(1024, 13, mode='sum') for _ in range(26)])
                for embedding in self.emb_l:
                    torch.nn.init.constant_(embedding.weight, 1)
                self.top_l = torch.nn.Sequential(torch.nn.Linear(364, 1), torch.nn.Sigmoid())
                torch.nn.init.constant_(self.top_l[0].weight, 1)
                torch.nn.init.constant_(self.top_l[0].bias, 100)

            def interact_features(self, dense, bags):
                packed = torch.stack([dense] + bags, dim=1)
                gram = torch.bmm(packed, packed.transpose(1, 2))
                li = [i for i in range(27) for j in range(i)]
                lj = [j for i in range(27) for j in range(i)]
                return torch.cat([dense, gram[:, li, lj]], dim=1)

        model = Model().eval()
        inputs = self.module.build_inputs(original_inputs(), bag_length=32)
        with torch.inference_mode():
            expected = self.module.capture_stages(model, inputs)
            self.assertEqual(len(expected['embeddings']), 26)
            good = self.module.validate_stages(model, inputs, expected, lambda: None)
            self.assertTrue(good['passed'])
            self.assertEqual(len(good['embedding_errors']), 26)
            model.emb_l[0].weight.add_(1)
            bad = self.module.validate_stages(model, inputs, expected, lambda: None)
        self.assertTrue(bad['output_error']['passed'], 'the test needs sigmoid saturation to hide the bag fault')
        self.assertFalse(bad['passed'])
        self.assertFalse(bad['embedding_errors'][0]['passed'])
        self.assertFalse(bad['interaction_error']['passed'])
        self.assertFalse(bad['checks_in_timing'])


if __name__ == '__main__':
    unittest.main()
