"""Pinned MetaSyn sparse-input projection for the original R26 DLRM model.

The public synthetic MetaSyn input's first 128 bags from 26 selected tables
are projected to the original 1024-entry vocabulary. Ragged lengths, empty
bags and repeated IDs are retained. This is a small projection of synthetic
data, not a full-scale Meta production workload. Model, weights and dense
inputs remain the frozen original FP32 sum-mode DLRM.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

R28_WORKLOAD_SHA256 = '300e42507175d38ad35f6c22a8af1a0a663d7ad22e1fa8b6ae1683ccbbfc6960'
PROJECTION_SHA256 = '5eb2663c2d00d64db83c9d5b858c78f80849841216b242c93cfcd72d2dfbade5'
INPUT_SHA256 = 'c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1'
PROFILE_NAME = 'meta26'
ROUTING_PROFILE = dict(large_embeddings=2, small_embeddings=24, min_embedding_bag_length=64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


path = Path(__file__).resolve().parents[1] / 'eager_parallel_r28/workload.py'
require(hashlib.sha256(path.read_bytes()).hexdigest() == R28_WORKLOAD_SHA256,
        'frozen R28 workload SHA mismatch')
spec = importlib.util.spec_from_file_location('r31_frozen_r28_workload', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
reference = base.reference
helper = base.helper
DENSE_SHA256 = base.DENSE_SHA256
error_record = base.error_record
capture_stages = base.capture_stages
validate_stages = base.validate_stages


def _load_projection():
    raw = Path(__file__).with_name('meta26_inputs.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PROJECTION_SHA256, 'frozen MetaSyn projection SHA mismatch')
    return json.loads(raw)


def build_inputs(original, *, bag_length='meta26'):
    """Keep the original dense tensor and use only the byte-pinned sparse IDs."""
    import torch
    require(type(bag_length) is str and bag_length == PROFILE_NAME,
            "bag_length must be 'meta26'")
    require(helper.tensor_tree_sha256(original, torch) == reference.ORIGINAL_INPUT_SHA256,
            'original input SHA differs from the frozen L=2 workload')
    payload = _load_projection()
    require(len(payload['tables']) == 26, 'constructed bag table count differs')
    offsets, indices = [], []
    for table, row in enumerate(payload['tables']):
        table_offsets = torch.tensor(row['offsets'], dtype=torch.int64)
        table_indices = torch.tensor(row['indices'], dtype=torch.int64)
        lengths = torch.diff(torch.cat((table_offsets, table_offsets.new_tensor([table_indices.numel()]))))
        require(row['model_table'] == table and tuple(table_offsets.shape) == (128,)
                and tuple(table_indices.shape) == (row['indices_total'],)
                and table_offsets.dtype == table_indices.dtype == torch.int64
                and int(table_offsets[0]) == 0 and lengths.tolist() == row['lengths']
                and bool((lengths >= 0).all())
                and bool(((table_indices >= 0) & (table_indices < 1024)).all()),
                'constructed bag shape/offset/ID contract differs')
        offsets.append(table_offsets)
        indices.append(table_indices)
    return original[0], offsets, indices


def load_original_model(upstream, *, bag_length='meta26'):
    """Load original definitions directly, then replace only sparse inputs."""
    import torch
    require(type(bag_length) is str and bag_length == PROFILE_NAME,
            "bag_length must be 'meta26'")
    model, original, provenance = reference.load_original_model(upstream)
    inputs = build_inputs(original, bag_length=bag_length)
    state_sha = helper.tensor_tree_sha256(dict(model.state_dict()), torch)
    dense_sha = helper.tensor_tree_sha256(inputs[0], torch)
    input_sha = helper.tensor_tree_sha256(inputs, torch)
    require(state_sha == reference.ORIGINAL_STATE_SHA256, 'original state SHA changed during MetaSyn construction')
    require(dense_sha == DENSE_SHA256, 'original dense input SHA changed during MetaSyn construction')
    require(input_sha == INPUT_SHA256, 'MetaSyn input SHA differs from the frozen projection')
    payload = _load_projection()
    details = []
    for table, row in enumerate(payload['tables']):
        details.append(dict(table=table, source_table=row['source_table'],
            indices_shape=[row['indices_total']], offsets_shape=[128], offset_first=0,
            offset_last=row['offsets'][-1], bag_lengths=row['lengths'],
            bag_length_min=min(row['lengths']), bag_length_max=max(row['lengths']),
            mean_length=row['mean_length']))
    provenance['identity'].update(input_sha256=input_sha, dense_sha256=dense_sha)
    provenance['workload'] = dict(bag_length=PROFILE_NAME, batch=128, tables=26, embedding_dim=16,
        vocabulary=1024, dense_shape=[128, 13], original_input_sha256=reference.ORIGINAL_INPUT_SHA256,
        source_git=payload['source_git'], source_sha256=payload['source_sha256'],
        projection_sha256=PROJECTION_SHA256, selected_source_tables=payload['selected_source_tables'],
        selection=payload['selection'], source_shape=payload['source_shape'],
        input_policy='MetaSyn first128 bags; lossless int64 IDs modulo1024',
        routing_profile=dict(ROUTING_PROFILE), tables_detail=details)
    return model, inputs, provenance
