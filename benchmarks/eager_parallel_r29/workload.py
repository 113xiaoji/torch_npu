"""High-L extension of the byte-pinned original R28 L=32 workload.

The original model, weights, dense input and all 32 R28 IDs per bag stay fixed.
Each larger length appends a prefix of one maximum-size per-table extra draw.
This module is benchmark scaffolding, never a replacement model definition.
"""
import hashlib
import importlib.util
from pathlib import Path

R28_WORKLOAD_SHA256 = '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960'
BASE_INPUT_SHA256 = '21978f4f7e17dbab0dc1e241649536b32d44b73d29197f1678c8191d0a7e312b'
BAG_LENGTHS = (32, 64, 128, 256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


path = Path(__file__).resolve().parents[1] / 'eager_parallel_r28/workload.py'
require(hashlib.sha256(path.read_bytes()).hexdigest() == R28_WORKLOAD_SHA256,
        'frozen R28 workload SHA mismatch')
spec = importlib.util.spec_from_file_location('r29_frozen_r28_workload', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
reference = base.reference
helper = base.helper
DENSE_SHA256 = base.DENSE_SHA256
error_record = base.error_record
capture_stages = base.capture_stages
validate_stages = base.validate_stages


def build_inputs(inputs32, *, bag_length):
    """Retain every R28 L=32 ID and append a deterministic nested prefix."""
    import torch
    require(type(bag_length) is int and bag_length in BAG_LENGTHS,
            'bag_length must be one of 32, 64, 128, 256')
    require(helper.tensor_tree_sha256(inputs32, torch) == BASE_INPUT_SHA256,
            'base input SHA differs from the frozen R28 L=32 workload')
    if bag_length == 32:
        return inputs32
    generator = torch.Generator(device='cpu').manual_seed(helper.INPUT_SEED + 2)
    dense, _, base_indices = inputs32
    offsets, indices = [], []
    for ids in base_indices:
        extras = torch.randint(1024, (128, 224), generator=generator, dtype=torch.int64)
        indices.append(torch.cat((ids.reshape(128, 32), extras[:, :bag_length - 32]), dim=1).reshape(-1))
        offsets.append(torch.arange(0, 128 * bag_length, bag_length, dtype=torch.int64))
    return dense, offsets, indices


def load_original_model(upstream, *, bag_length=32):
    """Load the frozen original model through R28; vary only bag cardinality."""
    import torch
    model, inputs32, provenance = base.load_original_model(upstream, bag_length=32)
    inputs = build_inputs(inputs32, bag_length=bag_length)
    state_sha = helper.tensor_tree_sha256(dict(model.state_dict()), torch)
    dense_sha = helper.tensor_tree_sha256(inputs[0], torch)
    require(state_sha == reference.ORIGINAL_STATE_SHA256, 'original state SHA changed during L construction')
    require(dense_sha == DENSE_SHA256, 'original dense input SHA changed during L construction')
    details = []
    for table, (offsets, indices) in enumerate(zip(inputs[1], inputs[2])):
        lengths = torch.diff(torch.cat((offsets, offsets.new_tensor([indices.numel()]))))
        require(tuple(offsets.shape) == (128,) and tuple(indices.shape) == (128 * bag_length,)
                and int(offsets[0]) == 0 and bool((lengths == bag_length).all()),
                'constructed bag shape/offset contract differs')
        details.append(dict(table=table, indices_shape=list(indices.shape),
            logical_indices_shape=[128, bag_length], offsets_shape=list(offsets.shape),
            offset_first=int(offsets[0]), offset_last=int(offsets[-1]),
            bag_length_min=int(lengths.min()), bag_length_max=int(lengths.max())))
    provenance['identity'].update(input_sha256=helper.tensor_tree_sha256(inputs, torch), dense_sha256=dense_sha)
    provenance['workload'] = dict(bag_length=bag_length, batch=128, tables=26, embedding_dim=16,
        vocabulary=1024, dense_shape=[128, 13], base_input_sha256=BASE_INPUT_SHA256,
        original_input_sha256=reference.ORIGINAL_INPUT_SHA256, extra_input_seed=helper.INPUT_SEED + 2,
        input_policy='original R28 L=32 IDs per bag plus fixed per-table Bx224 prefix', tables_detail=details)
    return model, inputs, provenance
