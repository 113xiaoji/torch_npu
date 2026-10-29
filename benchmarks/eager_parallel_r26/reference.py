"""Finite-input FP64 arithmetic oracle, independent of ATen bag-sum/BMM.

Python float operations are binary64. CPU ATen is an independent comparator;
neither CPU arithmetic nor a NoOp provider qualifies NPU numerical behavior.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ORIGINAL_MODEL_SHA256 = "9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5"
ORIGINAL_STATE_SHA256 = "6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19"
ORIGINAL_INPUT_SHA256 = "239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6"
FROZEN_HELPER_SHA256 = "a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f"


def require(value, message):
    if not value:
        raise ValueError(message)


def matrix(value, name):
    rows = [[float(x) for x in row] for row in value]
    require(rows and rows[0] and all(len(row) == len(rows[0]) for row in rows),
            name + " must be a nonempty rectangular matrix")
    require(all(math.isfinite(x) for row in rows for x in row), name + " outside finite reference domain")
    return rows


def embedding_bag_sum_fp64(weight, indices, offsets, *, include_last_offset=False,
                          padding_idx=None, per_sample_weights=None):
    """F.embedding_bag sum semantics; None and negative padding stay distinct.

    This is a numerical reference, not the optimized route's eligibility rule.
    In particular sample weights can be compared here while remaining fallback.
    """
    weight = matrix(weight, "weight")
    indices, offsets = list(indices), list(offsets)
    require(all(type(x) is int and 0 <= x < len(weight) for x in indices), "invalid indices")
    require(all(type(x) is int and 0 <= x <= len(indices) for x in offsets)
            and offsets == sorted(offsets), "invalid offsets")
    require(not offsets or offsets[0] == 0, "offsets must start at zero")
    require(bool(offsets) or not indices, "indices require offsets")
    if include_last_offset:
        require(bool(offsets) and offsets[-1] == len(indices), "last offsets must delimit all indices")
    if padding_idx is not None:
        require(type(padding_idx) is int and -len(weight) <= padding_idx < len(weight), "invalid padding")
        padding_idx %= len(weight)
    if per_sample_weights is not None:
        require(len(per_sample_weights) == len(indices), "sample weights length differs from indices")
        require(all(math.isfinite(float(x)) for x in per_sample_weights), "sample weights outside finite domain")
    ends = offsets[1:] if include_last_offset else offsets[1:] + [len(indices)]
    result = []
    for begin, end in zip(offsets, ends):
        row = [0.0] * len(weight[0])
        for position in range(begin, end):
            index = indices[position]
            if index == padding_idx:
                continue
            scale = 1.0 if per_sample_weights is None else float(per_sample_weights[position])
            for dimension, value in enumerate(weight[index]):
                row[dimension] += value * scale
        result.append(row)
    return result


def interaction_fp64(dense, embeddings, *, itself=False, mode="dot"):
    """Original DLRM li/lj order: outer i, inner j in range(i + offset)."""
    dense = matrix(dense, "dense")
    embeddings = [matrix(value, "embedding") for value in embeddings]
    batch, dimension = len(dense), len(dense[0])
    require(all(len(value) == batch and len(value[0]) == dimension for value in embeddings),
            "embedding and dense shapes differ")
    require(mode in ("dot", "cat"), "unsupported interaction mode")
    packed = [[dense[b]] + [value[b] for value in embeddings] for b in range(batch)]
    if mode == "cat":
        return dict(T=packed, Z=None, Zflat=None, li=[], lj=[],
                    R=[[x for feature in row for x in feature] for row in packed])
    count = 1 + len(embeddings)
    pairs = [(i, j) for i in range(count) for j in range(i + int(itself))]
    gram = []
    for row in packed:
        values = []
        for left in row:
            products = []
            for right in row:
                total = 0.0
                for k in range(dimension):
                    total += left[k] * right[k]
                products.append(total)
            values.append(products)
        gram.append(values)
    flat = [[row[i][j] for i, j in pairs] for row in gram]
    return dict(T=packed, Z=gram, Zflat=flat, li=[i for i, _ in pairs], lj=[j for _, j in pairs],
                R=[dense[b] + flat[b] for b in range(batch)])


def error_record(actual, expected, *, rtol, atol):
    import torch
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    same_shape = actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    passed = same_shape and finite and bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
    error = (actual - expected).abs() if same_shape and finite else None
    return dict(passed=passed, shape=list(actual.shape), finite=finite, rtol=rtol, atol=atol,
                max_abs_error=float(error.max()) if error is not None and error.numel() else None)


def compare_four_outputs(actual, expected, auxiliary_value_indices):
    """Check all shapes/dtypes, but read only explicitly defined auxiliary cells.

    The mapping must explicitly describe all three auxiliary outputs. Empty
    coordinate lists mean no defined values; never copy those tensors to CPU.
    The same-SoC/SDK baseline qualification must supply this semantic mapping.
    """
    import torch
    require(len(actual) == len(expected) == 4, "EmbeddingBag requires four outputs")
    require(set(auxiliary_value_indices) == {1, 2, 3}, "explicit auxiliary semantics required")
    metadata = all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(actual, expected))
    if not metadata:
        return dict(passed=False, metadata_pass=False, main=None, auxiliary=[],
                    reason="shape/dtype mismatch; no tensor values read")
    main = error_record(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    auxiliary = []
    for index in (1, 2, 3):
        checked = 0
        equal = True
        for coordinate in auxiliary_value_indices[index]:
            require(isinstance(coordinate, (list, tuple)) and len(coordinate) == actual[index].ndim,
                    "invalid auxiliary coordinate rank")
            require(all(type(c) is int and 0 <= c < size for c, size in zip(coordinate, actual[index].shape)),
                    "invalid auxiliary coordinate")
            a, b = actual[index][tuple(coordinate)], expected[index][tuple(coordinate)]
            equal &= bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
            checked += 1
        auxiliary.append(dict(output=index, defined_values_checked=checked, passed=equal))
    return dict(passed=metadata and main["passed"] and all(row["passed"] for row in auxiliary),
                metadata_pass=metadata, main=main, auxiliary=auxiliary)


def load_frozen_helper():
    path = Path(__file__).resolve().parents[1] / "host_regions/real_dlrm_runner_r22.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_HELPER_SHA256, "frozen loader helper SHA mismatch")
    spec = importlib.util.spec_from_file_location("r26_frozen_original_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_original_model(upstream):
    """Load frozen inference definitions without editing original model bytes.

    Reuses the audited helper's single unused SummaryWriter import omission.
    All nine source files and ten model/function AST definitions remain pinned.
    The full dependencies must already exist; no fake import modules are added.
    """
    helper = load_frozen_helper()
    upstream = Path(upstream).resolve(strict=True)
    source_pins = helper.verify_upstream(upstream)
    import numpy as np
    import torch
    report = dict(source_files=source_pins, frozen_helper_sha256=FROZEN_HELPER_SHA256)
    sys.path.insert(0, str(upstream))
    try:
        official = helper.load_inference_module(upstream, report)
        for relative in helper.SOURCE_SHA:
            loaded = sys.modules.get(relative[:-3].replace("/", "."))
            require(loaded is not None and Path(loaded.__file__).resolve() == upstream / relative,
                    "original dependency imported outside pinned closure: " + relative)
        require(official.ext_dist.my_size <= 1, "single process original model required")
        np.random.seed(helper.STATE_SEED)
        torch.manual_seed(helper.STATE_SEED)
        model = official.DLRM_Net(m_spa=16, ln_emb=np.array([1024] * 26),
            ln_bot=np.array([13, 64, 32, 16]), ln_top=np.array([367, 64, 32, 1]),
            arch_interaction_op="dot", arch_interaction_itself=False, sigmoid_top=2, ndevices=1).eval()
        model.requires_grad_(False)
        require(sum(p.numel() for p in model.parameters()) == helper.PARAMETERS, "original parameter count changed")
        generator = torch.Generator(device="cpu").manual_seed(helper.INPUT_SEED)
        dense = torch.rand((128, 13), generator=generator, dtype=torch.float32)
        offsets = [torch.arange(0, 256, 2, dtype=torch.int64) for _ in range(26)]
        indices = [torch.randint(1024, (256,), generator=generator, dtype=torch.int64) for _ in range(26)]
        inputs = dense, offsets, indices
        state_sha = helper.tensor_tree_sha256(dict(model.state_dict()), torch)
        input_sha = helper.tensor_tree_sha256(inputs, torch)
        require(state_sha == ORIGINAL_STATE_SHA256 and input_sha == ORIGINAL_INPUT_SHA256,
                "original state/input SHA differs from frozen workload")
        report.update(identity=dict(model_sha256=ORIGINAL_MODEL_SHA256, state_sha256=state_sha,
                                    input_sha256=input_sha), eager=helper.eager_proof(model, torch),
                      sparse=helper.sparse_proof(model, torch))
        helper.verify_upstream(upstream)
        return model, inputs, report
    finally:
        sys.path.remove(str(upstream))


def cpu_original_report(upstream):
    import torch
    model, inputs, provenance = load_original_model(upstream)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            actual = model(*inputs)
            x = model.bot_l(inputs[0])
            bags, errors = [], []
            for table, embedding in enumerate(model.emb_l):
                expected = embedding_bag_sum_fp64(embedding.weight.detach().tolist(), inputs[2][table].tolist(),
                                                  inputs[1][table].tolist(), padding_idx=embedding.padding_idx)
                bags.append(expected)
                errors.append(error_record(embedding(inputs[2][table], inputs[1][table]),
                    torch.tensor(expected, dtype=torch.float64), rtol=1e-4, atol=1e-5))
            oracle = interaction_fp64(x.tolist(), bags)
            expected_interaction = torch.tensor(oracle["R"], dtype=torch.float64)
            cpu_bags = [embedding(inputs[2][table], inputs[1][table]) for table, embedding in enumerate(model.emb_l)]
            interaction = model.interact_features(x, cpu_bags)
            interaction_error = error_record(interaction, expected_interaction, rtol=1e-4, atol=1e-4)
            # Bottom/top MLPs remain the original CPU modules. Only bag-sum and
            # Interaction are replaced by independent binary64 arithmetic here.
            output_error = error_record(actual, model.top_l(expected_interaction.float()), rtol=1e-4, atol=1e-5)
    finally:
        torch.set_num_threads(previous_threads)
    return dict(provenance, schema=1, evidence_kind="cpu_original_model_reference", torch=str(torch.__version__),
        original_model_executed=True, cpu_reference_pass=all(row["passed"] for row in errors)
        and interaction_error["passed"] and output_error["passed"], embedding_errors=errors,
        interaction_error=interaction_error, model_output_error=output_error,
        real_device_numerical_claim=False, device_overlap_claim=False, performance_claim=False,
        boundary="original CPU forward versus explicit FP64 bags/Interaction with original CPU bottom/top MLPs")


def cpu_reference_report():
    """Predeclared stratified shapes, with empty/repeated/padding bags in each."""
    import torch
    import torch.nn.functional as functional
    cases = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for tables, batch, dimension in ((1, 1, 8), (2, 3, 16), (7, 4, 31), (26, 128, 16), (39, 2, 128)):
            generator = torch.Generator().manual_seed(26000 + tables)
            dense = torch.rand((batch, dimension), generator=generator) - 0.5
            values, explicit, errors = [], [], []
            for table in range(tables):
                # A strided input validates the CPU comparator's layout handling.
                weight = (torch.rand((17, dimension * 2), generator=generator) - 0.5)[:, ::2]
                bag_lists = [[] if (b + table) % 4 == 0 else [0, (b + table) % 17, (b + table) % 17]
                             for b in range(batch)]
                indices, offsets = [], []
                for bag in bag_lists:
                    offsets.append(len(indices))
                    indices.extend(bag)
                padding = None if table % 2 == 0 else 0
                explicit.append(embedding_bag_sum_fp64(weight.tolist(), indices, offsets, padding_idx=padding))
                values.append(functional.embedding_bag(torch.tensor(indices, dtype=torch.int64), weight,
                    torch.tensor(offsets, dtype=torch.int64), mode="sum", padding_idx=padding))
                errors.append(error_record(values[-1], torch.tensor(explicit[-1], dtype=torch.float64), rtol=1e-4, atol=1e-5))
            for mode, itself in (("dot", False), ("dot", True), ("cat", False)):
                expected = interaction_fp64(dense.tolist(), explicit, itself=itself, mode=mode)
                packed = torch.cat([dense] + values, dim=1).view(batch, tables + 1, dimension)
                if mode == "dot":
                    gram = torch.bmm(packed, packed.transpose(1, 2))
                    # Construct indices separately, in the original model order.
                    li = torch.tensor([i for i in range(tables + 1) for j in range(i + int(itself))])
                    lj = torch.tensor([j for i in range(tables + 1) for j in range(i + int(itself))])
                    result = torch.cat([dense, gram[:, li, lj]], dim=1)
                else:
                    result = torch.cat([dense] + values, dim=1)
                error = error_record(result, torch.tensor(expected["R"], dtype=torch.float64), rtol=1e-4, atol=1e-4)
                cases.append(dict(tables=tables, batch=batch, dimension=dimension, mode=mode, itself=itself,
                    embedding_pass=all(row["passed"] for row in errors), embedding_errors=errors,
                    interaction_pass=error["passed"], interaction_error=error))
    finally:
        torch.set_num_threads(previous_threads)
    return dict(schema=1, evidence_kind="cpu_reference", torch=str(torch.__version__),
        precision=dict(matmul_precision=torch.get_float32_matmul_precision(), autocast=False,
                       deterministic=torch.are_deterministic_algorithms_enabled()),
        cpu_reference_pass=all(row["embedding_pass"] and row["interaction_pass"] for row in cases),
        real_device_numerical_claim=False, device_overlap_claim=False, performance_claim=False,
        original_model_executed=False, cases=cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-source", type=Path,
                        help="optional complete pinned nine-file original DLRM source; dependencies must already be installed")
    args = parser.parse_args(argv)
    report = cpu_reference_report()
    if args.original_source is not None:
        report["original_model"] = cpu_original_report(args.original_source)
        report["original_model_executed"] = True
        report["cpu_reference_pass"] &= report["original_model"]["cpu_reference_pass"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0 if report["cpu_reference_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
