"""Fixed mixed bag lengths over the byte-pinned original R29 L=256 workload.

The original model, weights and dense input remain unchanged. Each table takes
its requested per-sample ID prefix, retaining repeated IDs without a new draw.
L=1 keeps the first of the original two IDs. This is benchmark scaffolding.
"""
import hashlib
import importlib.util
from pathlib import Path

R29_WORKLOAD_SHA256 = '361851afe6fe1e31f5ab78b88c7f191592403ffef96c90372b8028e78ac4a3f9'
BASE_INPUT_SHA256 = '9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782'
INPUT_SHA256 = '40d6a16803f328885b66357bbcfdb680eb5578b15a641a61421069e4755be511'
PROFILE_NAME = 'mixed'
BAG_LENGTHS = (3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


path = Path(__file__).resolve().parents[1] / 'eager_parallel_r29/workload.py'
require(hashlib.sha256(path.read_bytes()).hexdigest() == R29_WORKLOAD_SHA256,
        'frozen R29 workload SHA mismatch')
spec = importlib.util.spec_from_file_location('r30_frozen_r29_workload', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
reference = base.reference
helper = base.helper
DENSE_SHA256 = base.DENSE_SHA256
error_record = base.error_record
capture_stages = base.capture_stages
validate_stages = base.validate_stages


def build_inputs(inputs256, *, bag_length='mixed'):
    """Take each table's fixed prefix of every frozen R29 L=256 sample."""
    import torch
    require(type(bag_length) is str and bag_length == PROFILE_NAME,
            "bag_length must be 'mixed'")
    require(helper.tensor_tree_sha256(inputs256, torch) == BASE_INPUT_SHA256,
            'base input SHA differs from the frozen R29 L=256 workload')
    dense, _, base_indices = inputs256
    indices = [ids.reshape(128, 256)[:, :length].contiguous().reshape(-1)
               for ids, length in zip(base_indices, BAG_LENGTHS)]
    offsets = [torch.arange(0, 128 * length, length, dtype=torch.int64) for length in BAG_LENGTHS]
    return dense, offsets, indices


def load_original_model(upstream, *, bag_length='mixed'):
    """Load the frozen original model through R29 and select its mixed IDs."""
    import torch
    require(type(bag_length) is str and bag_length == PROFILE_NAME,
            "bag_length must be 'mixed'")
    model, inputs256, provenance = base.load_original_model(upstream, bag_length=256)
    inputs = build_inputs(inputs256, bag_length=bag_length)
    state_sha = helper.tensor_tree_sha256(dict(model.state_dict()), torch)
    dense_sha = helper.tensor_tree_sha256(inputs[0], torch)
    input_sha = helper.tensor_tree_sha256(inputs, torch)
    require(state_sha == reference.ORIGINAL_STATE_SHA256, 'original state SHA changed during mixed construction')
    require(dense_sha == DENSE_SHA256, 'original dense input SHA changed during mixed construction')
    require(input_sha == INPUT_SHA256, 'mixed input SHA differs from the frozen profile')
    details = []
    for table, (offsets, indices, length) in enumerate(zip(inputs[1], inputs[2], BAG_LENGTHS)):
        lengths = torch.diff(torch.cat((offsets, offsets.new_tensor([indices.numel()]))))
        require(tuple(offsets.shape) == (128,) and tuple(indices.shape) == (128 * length,)
                and int(offsets[0]) == 0 and bool((lengths == length).all()),
                'constructed bag shape/offset contract differs')
        details.append(dict(table=table, indices_shape=list(indices.shape),
            logical_indices_shape=[128, length], offsets_shape=list(offsets.shape),
            offset_first=int(offsets[0]), offset_last=int(offsets[-1]),
            bag_length_min=int(lengths.min()), bag_length_max=int(lengths.max())))
    provenance['identity'].update(input_sha256=input_sha, dense_sha256=dense_sha)
    provenance['workload'] = dict(bag_length=PROFILE_NAME, bag_lengths=list(BAG_LENGTHS),
        batch=128, tables=26, embedding_dim=16, vocabulary=1024, dense_shape=[128, 13],
        base_input_sha256=BASE_INPUT_SHA256, original_input_sha256=reference.ORIGINAL_INPUT_SHA256,
        extra_input_seed=helper.INPUT_SEED + 2,
        input_policy='per-table prefixes of frozen R29 L=256 IDs; fixed MLPerf multi-hot length vector',
        tables_detail=details)
    return model, inputs, provenance
