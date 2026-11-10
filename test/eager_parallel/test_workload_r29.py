"""The high-L sweep retains the complete frozen R28 L=32 input prefix."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
R28_SHA256 = '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960'
INPUT_HASHES = {
    32: '21978f4f7e17dbab0dc1e241649536b32d44b73d29197f1678c8191d0a7e312b',
    64: '8378240dfb662f5b55f2246725edcf7faa48f411f68cc5aeeb391c49186f89ee',
    128: '3aea09989f9fbabd7eb9f2adc75b300a7a02dfae7e5fc23acaceeb395db3643c',
    256: '9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782',
}
INPUT_POLICY = 'original R28 L=32 IDs per bag plus fixed per-table Bx224 prefix'


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
        path = ROOT / 'benchmarks/eager_parallel_r29/workload.py'
        self.assertTrue(path.is_file(), 'R29 high-L workload adapter is missing')
        self.module = load(path, 'r29_workload_test')
        self.base = self.module.base
        self.helper = self.base.helper

    def inputs32(self):
        return self.base.build_inputs(original_inputs(), bag_length=32)

    def test_l32_retains_exact_r28_input_object_and_bytes(self):
        inputs = self.inputs32()
        self.assertEqual(self.module.BAG_LENGTHS, (32, 64, 128, 256))
        result = self.module.build_inputs(inputs, bag_length=32)
        self.assertIs(result, inputs)
        self.assertEqual(self.helper.tensor_tree_sha256(result, torch), INPUT_HASHES[32])

    def test_all_lengths_preserve_original_ids_and_nested_high_l_prefixes(self):
        original = self.inputs32()
        original_sha = self.helper.tensor_tree_sha256(original, torch)
        previous = None
        for length in (32, 64, 128, 256):
            with self.subTest(length=length):
                inputs = self.module.build_inputs(original, bag_length=length)
                self.assertIs(inputs[0], original[0])
                self.assertEqual(inputs[0].dtype, torch.float32)
                self.assertEqual(len(inputs[1]), 26)
                self.assertEqual(len(inputs[2]), 26)
                for table in range(26):
                    ids, offsets = inputs[2][table], inputs[1][table]
                    self.assertEqual(ids.shape, (128 * length,))
                    self.assertEqual(offsets.shape, (128,))
                    self.assertEqual(ids.dtype, torch.int64)
                    self.assertEqual(offsets.dtype, torch.int64)
                    self.assertTrue(torch.equal(offsets, torch.arange(0, 128 * length, length)))
                    self.assertTrue(torch.equal(ids.reshape(128, length)[:, :32],
                                                original[2][table].reshape(128, 32)))
                    self.assertTrue(bool(((ids >= 0) & (ids < 1024)).all()))
                    if previous is not None:
                        self.assertTrue(torch.equal(ids.reshape(128, length)[:, :previous[0]],
                                                    previous[1][2][table].reshape(128, previous[0])))
                digest = self.helper.tensor_tree_sha256(inputs, torch)
                self.assertEqual(digest, INPUT_HASHES[length])
                repeated = self.module.build_inputs(original, bag_length=length)
                self.assertEqual(self.helper.tensor_tree_sha256(repeated, torch), digest)
                previous = length, inputs
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), original_sha)

    def test_extras_use_fixed_maximum_draw_and_keep_repeated_ids(self):
        inputs = self.module.build_inputs(self.inputs32(), bag_length=64)
        generator = torch.Generator().manual_seed(20260920)
        duplicates = 0
        for table in range(26):
            extras = torch.randint(1024, (128, 224), generator=generator, dtype=torch.int64)
            actual = inputs[2][table].reshape(128, 64)
            self.assertTrue(torch.equal(actual[:, 32:], extras[:, :32]))
            duplicates += sum(row.numel() - row.unique().numel() for row in actual)
        self.assertGreater(duplicates, 0, 'random duplicates must survive rather than being deduplicated')

    def test_invalid_lengths_and_nonfrozen_l32_inputs_are_rejected(self):
        inputs = self.inputs32()
        for length in (True, 32.0, 0, 2, 8, 16, 48, 512, '64', None):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'bag_length'):
                self.module.build_inputs(inputs, bag_length=length)
        inputs[2][0][0] = (int(inputs[2][0][0]) + 1) % 1024
        for length in (32, 64):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'base input'):
                self.module.build_inputs(inputs, bag_length=length)
        with self.assertRaisesRegex(ValueError, 'base input'):
            self.module.build_inputs(original_inputs(), bag_length=64)

    def test_frozen_r28_and_stage_validation_are_reused_exactly(self):
        path = ROOT / 'benchmarks/eager_parallel_r28/workload.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), R28_SHA256)
        self.assertEqual(self.module.R28_WORKLOAD_SHA256, R28_SHA256)
        self.assertEqual(self.module.BASE_INPUT_SHA256, INPUT_HASHES[32])
        self.assertEqual(self.module.DENSE_SHA256, self.base.DENSE_SHA256)
        self.assertIs(self.module.reference, self.base.reference)
        self.assertIs(self.module.helper, self.base.helper)
        self.assertIs(self.module.error_record, self.base.error_record)
        self.assertIs(self.module.capture_stages, self.base.capture_stages)
        self.assertIs(self.module.validate_stages, self.base.validate_stages)

    def test_changed_r28_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'eager_parallel_r29/workload.py'
            adapter.parent.mkdir()
            adapter.write_bytes(Path(self.module.__file__).read_bytes())
            base = root / 'eager_parallel_r28/workload.py'
            base.parent.mkdir()
            base.write_text("raise RuntimeError('unverified R28 executed')\n")
            with self.assertRaisesRegex(ValueError, 'frozen R28 workload SHA mismatch'):
                load(adapter, 'r29_untrusted_base')

    @unittest.skipUnless(os.environ.get('R29_ORIGINAL_SOURCE'), 'complete pinned original source is not configured')
    def test_original_loader_preserves_model_state_and_dense_for_each_length(self):
        # The R22 original loader admits one source import per process.
        # Each L therefore runs in a fresh interpreter, matching the campaign.
        selected = os.environ.get('R29_SINGLE_LENGTH')
        if selected is None:
            code = (
                'import importlib.util,json,sys,unittest;'
                'sys.path[:]=json.loads(sys.argv.pop());'
                'path=sys.argv.pop();sys.argv=[path];'
                's=importlib.util.spec_from_file_location("r29_original_child",path);'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
                'suite=unittest.TestSuite([m.Workload("test_original_loader_preserves_model_state_and_dense_for_each_length")]);'
                'result=unittest.TextTestRunner(verbosity=2).run(suite);'
                'raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)'
            )
            for length in (32, 64, 128, 256):
                environment = dict(os.environ, R29_SINGLE_LENGTH=str(length))
                result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve()),
                    json.dumps(sys.path)], env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, f'L={length}\n' + result.stdout + result.stderr)
            return
        length = int(selected)
        model, inputs, provenance = self.module.load_original_model(
            os.environ['R29_ORIGINAL_SOURCE'], bag_length=length)
        identity, workload = provenance['identity'], provenance['workload']
        self.assertEqual(identity['model_sha256'], self.base.reference.ORIGINAL_MODEL_SHA256)
        self.assertEqual(identity['state_sha256'], self.base.reference.ORIGINAL_STATE_SHA256)
        self.assertEqual(identity['dense_sha256'], self.module.DENSE_SHA256)
        self.assertEqual(identity['input_sha256'], INPUT_HASHES[length])
        self.assertEqual(self.helper.tensor_tree_sha256(dict(model.state_dict()), torch), identity['state_sha256'])
        self.assertEqual(workload['bag_length'], length)
        self.assertEqual((workload['batch'], workload['tables'], workload['embedding_dim'], workload['vocabulary']),
                         (128, 26, 16, 1024))
        self.assertEqual(workload['dense_shape'], [128, 13])
        self.assertEqual(workload['base_input_sha256'], INPUT_HASHES[32])
        self.assertEqual(workload['original_input_sha256'], self.base.reference.ORIGINAL_INPUT_SHA256)
        self.assertEqual(workload['extra_input_seed'], 20260920)
        self.assertEqual(workload['input_policy'], INPUT_POLICY)
        self.assertEqual(len(workload['tables_detail']), 26)
        for table, detail in enumerate(workload['tables_detail']):
            self.assertEqual(detail['table'], table)
            self.assertEqual(detail['indices_shape'], [128 * length])
            self.assertEqual(detail['logical_indices_shape'], [128, length])
            self.assertEqual(detail['offsets_shape'], [128])
            self.assertEqual(detail['offset_first'], 0)
            self.assertEqual(detail['offset_last'], 127 * length)
            self.assertEqual(detail['bag_length_min'], length)
            self.assertEqual(detail['bag_length_max'], length)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        self.assertEqual(len(model.emb_l), 26)
        self.assertTrue(all(embedding.mode == 'sum' for embedding in model.emb_l))
        self.assertTrue(all(tuple(embedding.weight.shape) == (1024, 16) for embedding in model.emb_l))
        with torch.inference_mode():
            expected = model(*inputs)
            stages = self.module.capture_stages(model, inputs)
            self.assertTrue(self.module.error_record(stages['output'], expected, rtol=1e-4, atol=1e-5)['passed'])
            self.assertTrue(self.module.validate_stages(model, inputs, stages, lambda: None)['passed'])


if __name__ == '__main__':
    unittest.main()
