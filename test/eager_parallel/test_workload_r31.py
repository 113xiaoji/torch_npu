"""The MetaSyn projection replaces only sparse inputs of the original DLRM."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
R28_SHA256 = '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960'
PROJECTION_SHA256 = '5eb2663c2d00d64db83c9d5b858c78f80849841216b242c93cfcd72d2dfbade5'
INPUT_SHA256 = 'c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1'
SOURCE_SHA256 = '1d4d3ca295fade795cbf973d9b29e721b7c76a2130684582d9f4ab579c95e772'
SOURCE_GIT = '6f46ba7906e47a2f6bab6499056480d07d784eb8'
ROUTING_PROFILE = dict(large_embeddings=2, small_embeddings=24, min_embedding_bag_length=64)
INPUT_POLICY = 'MetaSyn first128 bags; lossless int64 IDs modulo1024'
SELECTED_TABLES = [238, 644, 12, 445, 373, 515, 14, 348, 843, 91, 649, 734, 381,
                   151, 640, 682, 146, 6, 258, 54, 730, 309, 623, 576, 111, 465]


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
        path = ROOT / 'benchmarks/eager_parallel_r31/workload.py'
        self.assertTrue(path.is_file(), 'R31 MetaSyn workload adapter is missing')
        self.module = load(path, 'r31_workload_test')
        self.base = self.module.base
        self.helper = self.module.helper
        self.projection_path = path.with_name('meta26_inputs.json')
        self.payload = json.loads(self.projection_path.read_bytes())

    def expected_workload(self):
        details = [dict(table=i, source_table=row['source_table'],
            indices_shape=[row['indices_total']], offsets_shape=[128], offset_first=0,
            offset_last=row['offsets'][-1], bag_lengths=row['lengths'],
            bag_length_min=min(row['lengths']), bag_length_max=max(row['lengths']),
            mean_length=row['mean_length']) for i, row in enumerate(self.payload['tables'])]
        return dict(bag_length='meta26', batch=128, tables=26, embedding_dim=16,
            vocabulary=1024, dense_shape=[128, 13],
            original_input_sha256=self.module.reference.ORIGINAL_INPUT_SHA256,
            source_git=SOURCE_GIT, source_sha256=SOURCE_SHA256,
            projection_sha256=PROJECTION_SHA256,
            selected_source_tables=SELECTED_TABLES, selection=self.payload['selection'],
            source_shape=[856, 65536], input_policy=INPUT_POLICY,
            routing_profile=ROUTING_PROFILE, tables_detail=details)

    def test_pinned_projection_and_complete_tensor_input_hash(self):
        self.assertEqual(self.module.PROFILE_NAME, 'meta26')
        self.assertEqual(self.module.INPUT_SHA256, INPUT_SHA256)
        self.assertEqual(self.module.PROJECTION_SHA256, PROJECTION_SHA256)
        self.assertEqual(self.module.ROUTING_PROFILE, ROUTING_PROFILE)
        raw = self.projection_path.read_bytes()
        self.assertEqual(len(raw), 218691)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PROJECTION_SHA256)
        self.assertEqual(self.payload['selected_source_tables'], SELECTED_TABLES)
        self.assertEqual(self.payload['source_git'], SOURCE_GIT)
        self.assertEqual(self.payload['source_sha256'], SOURCE_SHA256)
        self.assertEqual(self.payload['source_shape'], [856, 65536])
        inputs = self.module.build_inputs(original_inputs())
        self.assertEqual(self.helper.tensor_tree_sha256(inputs, torch), INPUT_SHA256)
        repeated = self.module.build_inputs(original_inputs(), bag_length='meta26')
        self.assertEqual(self.helper.tensor_tree_sha256(repeated, torch), INPUT_SHA256)

    def test_each_bag_keeps_exact_projected_source_prefix_and_ragged_offsets(self):
        original = original_inputs()
        inputs = self.module.build_inputs(original)
        self.assertIs(inputs[0], original[0])
        self.assertEqual(inputs[0].shape, (128, 13))
        self.assertEqual(inputs[0].dtype, torch.float32)
        self.assertEqual(len(inputs[1]), 26)
        self.assertEqual(len(inputs[2]), 26)
        for table, row in enumerate(self.payload['tables']):
            with self.subTest(table=table):
                offsets, indices = inputs[1][table], inputs[2][table]
                self.assertEqual(offsets.dtype, torch.int64)
                self.assertEqual(indices.dtype, torch.int64)
                self.assertEqual(offsets.shape, (128,))
                self.assertEqual(indices.shape, (row['indices_total'],))
                self.assertEqual(offsets.tolist(), row['offsets'])
                self.assertEqual(indices.tolist(), row['indices'])
                self.assertTrue(offsets.is_contiguous() and indices.is_contiguous())
                self.assertTrue(bool(((indices >= 0) & (indices < 1024)).all()))
                cursor = 0
                for bag, length in enumerate(row['lengths']):
                    self.assertEqual(int(offsets[bag]), cursor)
                    self.assertEqual(indices[cursor:cursor + length].tolist(),
                                     row['indices'][cursor:cursor + length])
                    cursor += length
                self.assertEqual(cursor, indices.numel())
                lengths = torch.diff(torch.cat((offsets, offsets.new_tensor([indices.numel()]))))
                self.assertEqual(lengths.tolist(), row['lengths'])
        self.assertEqual(sum(ids.numel() for ids in inputs[2]), 50850)

    def test_empty_and_ragged_bags_obey_embedding_bag_sum_semantics(self):
        inputs = self.module.build_inputs(original_inputs())
        empty = inputs[2][1]
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(inputs[1][1].tolist(), [0] * 128)
        weight = torch.arange(1024 * 16, dtype=torch.float32).reshape(1024, 16) / 16
        for table in (0, 1, 2, 9):
            with self.subTest(table=table):
                offsets, ids = inputs[1][table], inputs[2][table]
                actual = torch.nn.functional.embedding_bag(ids, weight, offsets, mode='sum',
                                                          include_last_offset=False)
                expected = torch.stack([weight[ids[start:start + length]].sum(dim=0)
                    for start, length in zip(offsets.tolist(), self.payload['tables'][table]['lengths'])])
                self.assertEqual(actual.shape, (128, 16))
                self.assertTrue(torch.equal(actual, expected))
        lengths = [row['lengths'] for row in self.payload['tables']]
        self.assertGreater(sum(length == 0 for row in lengths for length in row), 128)
        self.assertGreater(len(set(lengths[2])), 1)
        self.assertEqual([i for i, row in enumerate(lengths) if sum(row) / 128 >= 64], [2, 9])
        self.assertEqual([sum(lengths[i]) / 128 for i in (2, 9)], [102.2265625, 76.8125])

    def test_construction_preserves_original_tensors_rng_and_repeated_ids(self):
        original = original_inputs()
        before = self.helper.tensor_tree_sha256(original, torch)
        random_state = torch.get_rng_state().clone()
        with patch.object(torch, 'randint', side_effect=AssertionError('new random IDs drawn')):
            inputs = self.module.build_inputs(original)
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), random_state))
        self.assertGreater(sum(ids.numel() - ids.unique().numel() for ids in inputs[2]), 0)
        inputs[2][0][0] = (int(inputs[2][0][0]) + 1) % 1024
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), before)
        self.assertEqual(self.helper.tensor_tree_sha256(self.module.build_inputs(original), torch), INPUT_SHA256)

    def test_invalid_profile_tags_are_rejected(self):
        original = original_inputs()
        for tag in (True, 1, 32, 256, 1.0, None, '', 'META26', 'meta26 ', 'mixed', ('meta26',)):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'bag_length'):
                self.module.build_inputs(original, bag_length=tag)

    def test_nonfrozen_original_input_is_rejected(self):
        for leaf in ('dense', 'offsets', 'indices'):
            original = original_inputs()
            if leaf == 'dense':
                original[0][0, 0] += 1
            elif leaf == 'offsets':
                original[1][0][1] += 1
            else:
                original[2][0][0] = (int(original[2][0][0]) + 1) % 1024
            with self.subTest(leaf=leaf), self.assertRaisesRegex(ValueError, 'original input SHA'):
                self.module.build_inputs(original)

    def test_frozen_r28_and_stage_callbacks_are_reused_exactly(self):
        path = ROOT / 'benchmarks/eager_parallel_r28/workload.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), R28_SHA256)
        self.assertEqual(self.module.R28_WORKLOAD_SHA256, R28_SHA256)
        self.assertEqual(self.module.DENSE_SHA256, self.base.DENSE_SHA256)
        for name in ('reference', 'helper', 'error_record', 'capture_stages', 'validate_stages'):
            self.assertIs(getattr(self.module, name), getattr(self.base, name))

    def test_changed_r28_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'eager_parallel_r31/workload.py'
            adapter.parent.mkdir()
            adapter.write_bytes(Path(self.module.__file__).read_bytes())
            base = root / 'eager_parallel_r28/workload.py'
            base.parent.mkdir()
            base.write_text("raise RuntimeError('unverified R28 executed')\n")
            with self.assertRaisesRegex(ValueError, 'frozen R28 workload SHA mismatch'):
                load(adapter, 'r31_untrusted_base')

    def test_changed_projection_is_rejected_before_json_parse(self):
        original = original_inputs()
        read_bytes = Path.read_bytes
        for altered in (b'not JSON', self.projection_path.read_bytes() + b' '):
            def read(path):
                return altered if path == self.projection_path else read_bytes(path)
            with self.subTest(altered=altered[:10]), patch.object(Path, 'read_bytes', read), \
                    patch.object(self.module.json, 'loads', side_effect=AssertionError('unverified JSON parsed')):
                with self.assertRaisesRegex(ValueError, 'projection SHA mismatch'):
                    self.module.build_inputs(original)

    def test_constructed_lengths_ids_and_offsets_are_revalidated(self):
        # The byte pin is the trust boundary, but tensor construction also checks
        # the actual bag contract rather than trusting metadata after parsing.
        for field in ('lengths', 'indices', 'offsets'):
            payload = json.loads(self.projection_path.read_bytes())
            if field == 'lengths':
                payload['tables'][0]['lengths'][0] = 1
            elif field == 'indices':
                payload['tables'][0]['indices'][0] = 1024
            else:
                payload['tables'][0]['offsets'][0] = 1
            with self.subTest(field=field), patch.object(self.module.json, 'loads', return_value=payload):
                with self.assertRaisesRegex(ValueError, 'constructed bag'):
                    self.module.build_inputs(original_inputs())

    def test_invalid_loader_tag_is_rejected_before_loading_original(self):
        with patch.object(self.module.reference, 'load_original_model',
                          side_effect=AssertionError('original loaded')):
            with self.assertRaisesRegex(ValueError, 'bag_length'):
                self.module.load_original_model('unused source', bag_length='mixed')

    def loader_fixture(self):
        # Full pinned sources are covered below when configured. This narrow
        # stub lets the local suite exercise wrapper provenance and fail-closed
        # identity checks without importing the original dependency closure.
        state = {'sentinel': torch.tensor([123.0])}
        class Model:
            def state_dict(self):
                return state
        model, original = Model(), original_inputs()
        provenance = dict(identity=dict(model_sha256=self.module.reference.ORIGINAL_MODEL_SHA256,
            state_sha256=self.module.reference.ORIGINAL_STATE_SHA256,
            input_sha256=self.module.reference.ORIGINAL_INPUT_SHA256), source_files={'sentinel': 'unchanged'})
        real_hash = self.helper.tensor_tree_sha256
        def digest(value, torch_module):
            if isinstance(value, dict) and value.get('sentinel') is state['sentinel']:
                return self.module.reference.ORIGINAL_STATE_SHA256
            return real_hash(value, torch_module)
        return model, original, provenance, digest

    def test_loader_calls_original_reference_and_reports_exact_provenance(self):
        model, original, provenance, digest = self.loader_fixture()
        original_before = self.helper.tensor_tree_sha256(original, torch)
        with patch.object(self.module.reference, 'load_original_model',
                          return_value=(model, original, provenance)) as original_loader, \
                patch.object(self.base, 'load_original_model', side_effect=AssertionError('uniform L loader called')), \
                patch.object(self.helper, 'tensor_tree_sha256', side_effect=digest):
            actual_model, inputs, result = self.module.load_original_model('pinned source')
        original_loader.assert_called_once_with('pinned source')
        self.assertIs(actual_model, model)
        self.assertIs(inputs[0], original[0])
        self.assertEqual(self.helper.tensor_tree_sha256(original, torch), original_before)
        self.assertEqual(result['identity'], dict(model_sha256=self.module.reference.ORIGINAL_MODEL_SHA256,
            state_sha256=self.module.reference.ORIGINAL_STATE_SHA256,
            input_sha256=INPUT_SHA256, dense_sha256=self.module.DENSE_SHA256))
        self.assertEqual(result['source_files'], {'sentinel': 'unchanged'})
        self.assertEqual(result['workload'], self.expected_workload())

    def test_loader_rejects_changed_state_dense_or_full_input_identity(self):
        for fault, message in (('state', 'state SHA'), ('dense', 'dense input SHA'), ('input', 'input SHA')):
            model, original, provenance, digest = self.loader_fixture()
            def changed(value, torch_module):
                if (fault == 'state' and isinstance(value, dict)
                        or fault == 'dense' and value is original[0]
                        or fault == 'input' and isinstance(value, tuple) and value is not original):
                    return '0' * 64
                return digest(value, torch_module)
            with self.subTest(fault=fault), \
                    patch.object(self.module.reference, 'load_original_model', return_value=(model, original, provenance)), \
                    patch.object(self.helper, 'tensor_tree_sha256', side_effect=changed):
                with self.assertRaisesRegex(ValueError, message):
                    self.module.load_original_model('pinned source')

    @unittest.skipUnless(os.environ.get('R31_ORIGINAL_SOURCE'), 'complete pinned original source is not configured')
    def test_original_loader_preserves_model_sources_and_passes_all_stages(self):
        upstream = Path(os.environ['R31_ORIGINAL_SOURCE'])
        sources_before = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                          for name in self.helper.SOURCE_SHA}
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        model, inputs, provenance = self.module.load_original_model(upstream)
        self.assertEqual(provenance['identity'], dict(model_sha256=self.module.reference.ORIGINAL_MODEL_SHA256,
            state_sha256=self.module.reference.ORIGINAL_STATE_SHA256,
            input_sha256=INPUT_SHA256, dense_sha256=self.module.DENSE_SHA256))
        self.assertEqual(self.helper.tensor_tree_sha256(dict(model.state_dict()), torch),
                         self.module.reference.ORIGINAL_STATE_SHA256)
        self.assertEqual(self.helper.tensor_tree_sha256(inputs[0], torch), self.module.DENSE_SHA256)
        self.assertEqual(self.helper.tensor_tree_sha256(inputs, torch), INPUT_SHA256)
        self.assertEqual(provenance['workload'], self.expected_workload())
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        self.assertEqual(len(model.emb_l), 26)
        self.assertTrue(all(embedding.mode == 'sum' and not embedding.include_last_offset for embedding in model.emb_l))
        self.assertTrue(all(tuple(embedding.weight.shape) == (1024, 16) for embedding in model.emb_l))
        with torch.inference_mode():
            output = model(*inputs)
            stages = self.module.capture_stages(model, inputs)
            self.assertEqual(output.shape, (128, 1))
            self.assertEqual(stages['interaction'].shape, (128, 367))
            self.assertTrue(all(bag.shape == (128, 16) for bag in stages['embeddings']))
            self.assertTrue(self.module.error_record(stages['output'], output, rtol=1e-4, atol=1e-5)['passed'])
            synchronized = []
            self.assertTrue(self.module.validate_stages(model, inputs, stages,
                            lambda: synchronized.append(True))['passed'])
            self.assertEqual(len(synchronized), 2)
            self.assertTrue(torch.equal(stages['embeddings'][1], torch.zeros((128, 16))))
        sources_after = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
                         for name in self.helper.SOURCE_SHA}
        self.assertEqual(sources_after, sources_before)
        self.assertEqual(sources_after, self.helper.SOURCE_SHA)


if __name__ == '__main__':
    unittest.main()
