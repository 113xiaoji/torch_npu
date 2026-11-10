"""Bag-length-only sweep over the byte-pinned original R26 DLRM workload.

The model, weights, dense input and original two IDs in each bag are retained.
Every length draws the same maximum-size extra-ID tensor and takes a prefix.
This module is benchmark scaffolding, never a replacement model definition.
"""
import hashlib
import importlib.util
from pathlib import Path

REFERENCE_SHA256 = '162cd92ddc2cb58819ee6e9a921b86c29933ab0651ccf5ed66c1fb4a29c17c4f'
DENSE_SHA256 = '352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9'
BAG_LENGTHS = (2, 8, 16, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


path = Path(__file__).resolve().parents[1] / 'eager_parallel_r26/reference.py'
require(hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256, 'frozen R26 reference SHA mismatch')
spec = importlib.util.spec_from_file_location('r28_frozen_r26_reference', path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
helper = reference.load_frozen_helper()
error_record = reference.error_record


def build_inputs(original, *, bag_length):
    """Keep original IDs per bag; append a nested, deterministic ID prefix."""
    import torch
    require(type(bag_length) is int and bag_length in BAG_LENGTHS,
            'bag_length must be one of 2, 8, 16, 32')
    require(helper.tensor_tree_sha256(original, torch) == reference.ORIGINAL_INPUT_SHA256,
            'original input SHA differs from the frozen L=2 workload')
    if bag_length == 2:
        return original
    generator = torch.Generator(device='cpu').manual_seed(helper.INPUT_SEED + 1)
    dense, _, base_indices = original
    offsets, indices = [], []
    for base in base_indices:
        extras = torch.randint(1024, (128, 30), generator=generator, dtype=torch.int64)
        indices.append(torch.cat((base.reshape(128, 2), extras[:, :bag_length - 2]), dim=1).reshape(-1))
        offsets.append(torch.arange(0, 128 * bag_length, bag_length, dtype=torch.int64))
    return dense, offsets, indices


def load_original_model(upstream, *, bag_length=2):
    """Load the frozen original model and vary only sparse input cardinality."""
    import torch
    model, original, provenance = reference.load_original_model(upstream)
    inputs = build_inputs(original, bag_length=bag_length)
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
        vocabulary=1024, dense_shape=[128, 13], original_input_sha256=reference.ORIGINAL_INPUT_SHA256,
        extra_input_seed=helper.INPUT_SEED + 1,
        input_policy='original two IDs per bag plus fixed per-table Bx30 prefix', tables_detail=details)
    return model, inputs, provenance


def capture_stages(model, inputs):
    """Observe original modules outside timing, without changing model.forward."""
    dense, offsets, indices = inputs
    bottom = model.bot_l(dense)
    embeddings = [embedding(indices[table], offsets[table]) for table, embedding in enumerate(model.emb_l)]
    interaction = model.interact_features(bottom, embeddings)
    return dict(embeddings=embeddings, interaction=interaction, output=model.top_l(interaction))


def validate_stages(model, inputs, expected, synchronize):
    """Catch intermediate errors even when the original final sigmoid saturates."""
    actual = capture_stages(model, inputs)
    synchronize()
    require(len(actual['embeddings']) == len(expected['embeddings']) == 26,
            'stage validation requires 26 original EmbeddingBag outputs')
    embeddings = [error_record(a.cpu(), e, rtol=1e-4, atol=1e-5)
                  for a, e in zip(actual['embeddings'], expected['embeddings'])]
    interaction = error_record(actual['interaction'].cpu(), expected['interaction'], rtol=1e-4, atol=1e-4)
    output = error_record(actual['output'].cpu(), expected['output'], rtol=1e-4, atol=1e-5)
    result = dict(passed=all(row['passed'] for row in embeddings) and interaction['passed'] and output['passed'],
                  embedding_errors=embeddings, interaction_error=interaction, output_error=output,
                  checks_in_timing=False)
    # Dropping side-stream tensors can enqueue allocator completion markers.
    # Include their cleanup in this untimed validation, before the SDK snapshot.
    del actual
    synchronize()
    return result
