"""The fixed mixed profile takes per-table prefixes of frozen R29 L=256 IDs."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
R29_SHA256 = '361851afe6fe1e31f5ab78b88c7f191592403ffef96c90372b8028e78ac4a3f9'
BASE_INPUT_SHA256 = '9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782'
INPUT_SHA256 = '40d6a16803f328885b66357bbcfdb680eb5578b15a641a61421069e4755be511'
BAG_LENGTHS = (3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1)
INPUT_POLICY = 'per-table prefixes of frozen R29 L=256 IDs; fixed MLPerf multi-hot length vector'


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
        path = ROOT / 'benchmarks/eager_parallel_r30/workload.py'
        self.assertTrue(path.is_file(), 'R30 mixed workload adapter is missing')
        self.module = load(path, 'r30_workload_test')
        self.base = self.module.base
        self.helper = self.module.helper

    def inputs256(self):
        inputs32 = self.base.base.build_inputs(original_inputs(), bag_length=32)
        return self.base.build_inputs(inputs32, bag_length=256)

    def test_fixed_profile_and_complete_input_hash(self):
        self.assertEqual(self.module.PROFILE_NAME, 'mixed')
        self.assertEqual(self.module.BAG_LENGTHS, BAG_LENGTHS)
        self.assertEqual(self.module.INPUT_SHA256, INPUT_SHA256)
        original = self.inputs256()
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), BASE_INPUT_SHA256)
        inputs = self.module.build_inputs(original)
        self.assertEqual(self.helper.tensor_tree_sha256(inputs, torch), INPUT_SHA256)
        repeated = self.module.build_inputs(original, bag_length='mixed')
        self.assertEqual(self.helper.tensor_tree_sha256(repeated, torch), INPUT_SHA256)

    def test_all_tables_use_exact_contiguous_sample_prefixes_and_offsets(self):
        original = self.inputs256()
        original_sha = self.helper.tensor_tree_sha256(original, torch)
        inputs = self.module.build_inputs(original, bag_length='mixed')
        self.assertIs(inputs[0], original[0])
        self.assertEqual(inputs[0].shape, (128, 13))
        self.assertEqual(inputs[0].dtype, torch.float32)
        self.assertEqual(len(inputs[1]), 26)
        self.assertEqual(len(inputs[2]), 26)
        for table, length in enumerate(BAG_LENGTHS):
            with self.subTest(table=table, length=length):
                ids, offsets = inputs[2][table], inputs[1][table]
                self.assertEqual(ids.shape, (128 * length,))
                self.assertEqual(offsets.shape, (128,))
                self.assertEqual(ids.dtype, torch.int64)
                self.assertEqual(offsets.dtype, torch.int64)
                self.assertTrue(ids.is_contiguous())
                self.assertTrue(torch.equal(offsets, torch.arange(0, 128 * length, length)))
                self.assertTrue(torch.equal(ids.reshape(128, length),
                                            original[2][table].reshape(128, 256)[:, :length]))
                self.assertTrue(bool(((ids >= 0) & (ids < 1024)).all()))
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), original_sha)

    def test_l1_keeps_first_original_id_and_longer_bags_keep_both(self):
        original = original_inputs()
        inputs = self.module.build_inputs(self.inputs256())
        for table, length in enumerate(BAG_LENGTHS):
            with self.subTest(table=table, length=length):
                retained = min(length, 2)
                self.assertTrue(torch.equal(inputs[2][table].reshape(128, length)[:, :retained],
                                            original[2][table].reshape(128, 2)[:, :retained]))

    def test_construction_draws_no_random_ids_and_keeps_duplicates(self):
        original = self.inputs256()
        random_state = torch.get_rng_state().clone()
        with patch.object(torch, 'randint', side_effect=AssertionError('new random IDs drawn')):
            inputs = self.module.build_inputs(original)
        self.assertTrue(torch.equal(torch.get_rng_state(), random_state))
        duplicates = sum(row.numel() - row.unique().numel()
                         for length, ids in zip(BAG_LENGTHS, inputs[2])
                         for row in ids.reshape(128, length))
        self.assertGreater(duplicates, 0, 'repeated IDs must survive the prefix operation')

    def test_invalid_profile_tags_are_rejected(self):
        original = self.inputs256()
        for tag in (True, 1, 32, 256, 1.0, None, '', 'MIXED', 'mixed ', '256', ('mixed',)):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'bag_length'):
                self.module.build_inputs(original, bag_length=tag)

    def test_nonfrozen_base_inputs_are_rejected(self):
        original = self.inputs256()
        original[2][0][0] = (int(original[2][0][0]) + 1) % 1024
        with self.assertRaisesRegex(ValueError, 'base input'):
            self.module.build_inputs(original)
        with self.assertRaisesRegex(ValueError, 'base input'):
            self.module.build_inputs(original_inputs())
        inputs32 = self.base.base.build_inputs(original_inputs(), bag_length=32)
        with self.assertRaisesRegex(ValueError, 'base input'):
            self.module.build_inputs(inputs32)

    def test_frozen_r29_and_stage_validation_are_reused_exactly(self):
        path = ROOT / 'benchmarks/eager_parallel_r29/workload.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), R29_SHA256)
        self.assertEqual(self.module.R29_WORKLOAD_SHA256, R29_SHA256)
        self.assertEqual(self.module.BASE_INPUT_SHA256, BASE_INPUT_SHA256)
        self.assertEqual(self.module.DENSE_SHA256, self.base.DENSE_SHA256)
        self.assertIs(self.module.reference, self.base.reference)
        self.assertIs(self.module.helper, self.base.helper)
        self.assertIs(self.module.error_record, self.base.error_record)
        self.assertIs(self.module.capture_stages, self.base.capture_stages)
        self.assertIs(self.module.validate_stages, self.base.validate_stages)

    def test_changed_r29_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'eager_parallel_r30/workload.py'
            adapter.parent.mkdir()
            adapter.write_bytes(Path(self.module.__file__).read_bytes())
            base = root / 'eager_parallel_r29/workload.py'
            base.parent.mkdir()
            base.write_text("raise RuntimeError('unverified R29 executed')\n")
            with self.assertRaisesRegex(ValueError, 'frozen R29 workload SHA mismatch'):
                load(adapter, 'r30_untrusted_base')

    def test_invalid_loader_profile_is_rejected_before_loading_original(self):
        with patch.object(self.base, 'load_original_model', side_effect=AssertionError('original loaded')):
            for tag in (True, 1, 32, 256, 1.0, None, '', 'MIXED', 'mixed ', '256', ('mixed',)):
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'bag_length'):
                    self.module.load_original_model('unused source', bag_length=tag)

    @unittest.skipUnless(os.environ.get('R30_ORIGINAL_SOURCE'), 'complete pinned original source is not configured')
    def test_original_loader_preserves_model_state_dense_and_sources_for_mixed_profile(self):
        upstream = Path(os.environ['R30_ORIGINAL_SOURCE'])
        sources_before = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                          for name in self.helper.SOURCE_SHA}
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        model, inputs, provenance = self.module.load_original_model(upstream)
        identity, workload = provenance['identity'], provenance['workload']
        self.assertEqual(identity['model_sha256'], self.module.reference.ORIGINAL_MODEL_SHA256)
        self.assertEqual(identity['state_sha256'], self.module.reference.ORIGINAL_STATE_SHA256)
        self.assertEqual(identity['dense_sha256'], self.module.DENSE_SHA256)
        self.assertEqual(identity['input_sha256'], INPUT_SHA256)
        self.assertEqual(self.helper.tensor_tree_sha256(dict(model.state_dict()), torch), identity['state_sha256'])
        self.assertEqual(self.helper.tensor_tree_sha256(inputs[0], torch), self.module.DENSE_SHA256)
        self.assertEqual(self.helper.tensor_tree_sha256(inputs, torch), INPUT_SHA256)
        expected_details = [dict(table=table, indices_shape=[128 * length],
            logical_indices_shape=[128, length], offsets_shape=[128], offset_first=0,
            offset_last=127 * length, bag_length_min=length, bag_length_max=length)
            for table, length in enumerate(BAG_LENGTHS)]
        self.assertEqual(workload, dict(bag_length='mixed', bag_lengths=list(BAG_LENGTHS),
            batch=128, tables=26, embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
            base_input_sha256=BASE_INPUT_SHA256,
            original_input_sha256=self.module.reference.ORIGINAL_INPUT_SHA256,
            extra_input_seed=20260920, input_policy=INPUT_POLICY, tables_detail=expected_details))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        self.assertEqual(len(model.emb_l), 26)
        self.assertTrue(all(embedding.mode == 'sum' for embedding in model.emb_l))
        self.assertTrue(all(tuple(embedding.weight.shape) == (1024, 16) for embedding in model.emb_l))
        with torch.inference_mode():
            expected = model(*inputs)
            stages = self.module.capture_stages(model, inputs)
            self.assertTrue(self.module.error_record(stages['output'], expected, rtol=1e-4, atol=1e-5)['passed'])
            self.assertTrue(self.module.validate_stages(model, inputs, stages, lambda: None)['passed'])
        sources_after = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                         for name in self.helper.SOURCE_SHA}
        self.assertEqual(sources_after, sources_before)
        self.assertEqual(sources_after, self.helper.SOURCE_SHA)


if __name__ == '__main__':
    unittest.main()
