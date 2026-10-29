"""Trace-driven CPU simulation, using real aten calls and the production queue core.

This is deliberately not a live CPU dispatch backend or an NPU/SDK measurement.
The tape contains operator identities, metadata, value references, and initial
inputs/parameters/buffers. It never contains captured intermediate Tensor values.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PIN_MANIFEST_SHA256 = "ada6b4d11bae79ad93d69478349aa56e6ee37c600e21bc85229f9945f20e624a"


class UnsupportedCapture(RuntimeError):
    pass


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorRef:
    value_id: int


@dataclass(frozen=True)
class EncodedTree:
    leaves: tuple
    spec: object


@dataclass(frozen=True)
class Call:
    func: object
    args: EncodedTree
    kwargs: EncodedTree
    outputs: EncodedTree
    output_ids: tuple
    metadata: tuple


@dataclass(frozen=True)
class Tape:
    ops: tuple
    initial_values: dict
    input_ids: tuple
    output: EncodedTree
    initial_metadata: dict


@dataclass(frozen=True)
class Reference:
    calls: tuple
    output: object


def load_pinned_models():
    """Import exactly the separately frozen workload source, checking both pins."""
    manifest_path = ROOT / "benchmarks/eager_prepare/SOURCE_PINS.json"
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest != PIN_MANIFEST_SHA256:
        raise RuntimeError("frozen source manifest SHA256 changed: " + actual_manifest)
    manifest = json.loads(manifest_bytes)
    model_path = ROOT / "benchmarks/eager_prepare/models.py"
    actual_models = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_models != manifest["files"]["benchmarks/eager_prepare/models.py"]["sha256"]:
        raise RuntimeError("frozen models.py SHA256 changed: " + actual_models)
    name = "_r24_pinned_cpu_workloads_" + actual_models
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name], {"manifest_sha256": actual_manifest,
        "models_sha256": actual_models, "models_path": str(model_path)}


def _flatten(value):
    from torch.utils._pytree import tree_flatten
    return tree_flatten(value)


def _unflatten(leaves, spec):
    from torch.utils._pytree import tree_unflatten
    return tree_unflatten(leaves, spec)


def _tensor_meta(tensor):
    if tensor.device.type != "cpu" or tensor.layout.__str__() != "torch.strided":
        raise UnsupportedCapture("only dense strided CPU tensors are supported")
    return (str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())


def _check_func(torch, func, args=()):
    if not isinstance(func, torch._ops.OpOverload) or func.namespace != "aten":
        raise UnsupportedCapture("custom operator/side effects are unsupported: " + str(func))
    if func._schema.is_mutable:
        raise UnsupportedCapture("mutation is unsupported: " + str(func))
    if torch.Tag.nondeterministic_seeded in func.tags:
        raise UnsupportedCapture("RNG is unsupported: " + str(func))
    # aten.index.Tensor's generic tag covers boolean masks too. Integer advanced
    # indexing has fixed output shape; the frozen recommendation model needs it.
    fixed_integer_index = str(func) == "aten.index.Tensor" and len(args) > 1 and all(
        index is None or isinstance(index, torch.Tensor) and index.dtype in (torch.int32, torch.int64)
        for index in args[1])
    if torch.Tag.data_dependent_output in func.tags or (
            torch.Tag.dynamic_output_shape in func.tags and not fixed_integer_index):
        raise UnsupportedCapture("data-dependent operator is unsupported: " + str(func))
    if str(func).split(".")[1] in {"_local_scalar_dense", "item", "nonzero", "masked_select", "unique_dim", "_unique2"}:
        raise UnsupportedCapture("data-dependent operator is unsupported: " + str(func))


def _constant(torch, leaf):
    if leaf is None or isinstance(leaf, (bool, int, float, complex, str,
            torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return leaf
    raise UnsupportedCapture("unsupported non-tensor attribute: " + type(leaf).__name__)


def _word(value):
    return int.from_bytes(hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def _metadata(torch, func, args, kwargs, outputs):
    """Only operator attributes and tensor metadata cross the native boundary."""
    words = [_word(str(func))]
    for tree in (args, kwargs, outputs):
        leaves, spec = _flatten(tree)
        words.extend((_word(str(spec)), len(leaves)))
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                dtype, shape, stride, offset = _tensor_meta(leaf)
                words.extend((1, _word(dtype), len(shape), *shape, *stride, offset))
            elif isinstance(leaf, bool):
                words.extend((2, int(leaf)))
            elif isinstance(leaf, int):
                if not -(1 << 63) <= leaf < (1 << 63):
                    raise UnsupportedCapture("integer attribute exceeds int64")
                words.extend((3, leaf))
            elif isinstance(leaf, float):
                words.extend((4, struct.unpack("q", struct.pack("d", leaf))[0]))
            else:
                _constant(torch, leaf)
                words.extend((5, _word(repr(leaf))))
    return tuple(words)


def _initial_tensors(torch, model, inputs):
    input_leaves, _ = _flatten(inputs)
    input_tensors = [x for x in input_leaves if isinstance(x, torch.Tensor)]
    return input_tensors, input_tensors + list(model.parameters()) + list(model.buffers())


def fingerprint(model, inputs):
    """Full data hashes are validation only; they are never prepare inputs."""
    import torch
    _, values = _initial_tensors(torch, model, inputs)
    digest = hashlib.sha256()
    with torch.inference_mode():
        for tensor in values:
            digest.update(repr(_tensor_meta(tensor)).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture(model, inputs):
    """Run original eager inference once without FX, compile, or model edits."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    input_tensors, initial_tensors = _initial_tensors(torch, model, inputs)
    held, ids, initial, initial_meta = [], {}, {}, {}

    def register(tensor, initial_value=False):
        identity = id(tensor)
        if identity not in ids:
            value_id = len(ids)
            ids[identity] = value_id
            held.append(tensor)  # Prevent Python id reuse for the entire capture.
            if initial_value:
                initial[value_id], initial_meta[value_id] = tensor, _tensor_meta(tensor)
        return TensorRef(ids[identity])

    for tensor in initial_tensors:
        register(tensor, True)
    input_ids = tuple(ids[id(tensor)] for tensor in input_tensors)

    def encode(value, output=False):
        leaves, spec = _flatten(value)
        encoded = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                if output:
                    encoded.append(register(leaf))
                elif id(leaf) in ids:
                    encoded.append(TensorRef(ids[id(leaf)]))
                else:
                    raise UnsupportedCapture("tensor constant is not an initial input/parameter/buffer")
            else:
                encoded.append(_constant(torch, leaf))
        return EncodedTree(tuple(encoded), spec)

    calls = []

    class CaptureMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            _check_func(torch, func, args)
            encoded_args, encoded_kwargs = encode(args), encode(kwargs)
            result = func(*args, **kwargs)
            encoded_outputs = encode(result, True)
            calls.append(Call(func, encoded_args, encoded_kwargs, encoded_outputs,
                tuple(x.value_id for x in encoded_outputs.leaves if isinstance(x, TensorRef)),
                _metadata(torch, func, args, kwargs, result)))
            return result

    with torch.inference_mode(), CaptureMode():
        result = model(*inputs)
    encoded_result = encode(result)
    # Only initial tensors escape. All captured intermediate tensors are released.
    return Tape(tuple(calls), initial, input_ids, encoded_result, initial_meta)


def _clone_tree(torch, value):
    leaves, spec = _flatten(value)
    return _unflatten([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in leaves], spec)


def reference_run(model, inputs):
    """A separate original eager execution, retained only for correctness checks."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    calls = []

    class ReferenceMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            _check_func(torch, func, args)
            result = func(*args, **(kwargs or {}))
            calls.append((str(func), _clone_tree(torch, result)))
            return result

    with torch.inference_mode(), ReferenceMode():
        output = model(*inputs)
    return Reference(tuple(calls), _clone_tree(torch, output))


def assert_tree_close(torch, actual, expected):
    actual_leaves, actual_spec = _flatten(actual)
    expected_leaves, expected_spec = _flatten(expected)
    if actual_spec != expected_spec:
        raise AssertionError("output pytree structure differs")
    for index, (left, right) in enumerate(zip(actual_leaves, expected_leaves)):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6, equal_nan=True,
                msg=lambda msg: "output leaf %d: %s" % (index, msg))
        elif type(left) is not type(right) or left != right:
            raise AssertionError("non-tensor output differs at leaf %d" % index)


def _resolve(encoded, values):
    try:
        return _unflatten([values[x.value_id] if isinstance(x, TensorRef) else x
            for x in encoded.leaves], encoded.spec)
    except KeyError as exc:
        raise ReplayError("operator dependency was not produced by this replay: " + str(exc)) from exc


def expected_checksum(native, tape, iterations):
    result = 0
    for op in tape.ops:
        result ^= native.prepare_inline(op.metadata, iterations)["checksum"]
    return result


def _last_uses(tape):
    uses = {}
    keep = {x.value_id for x in tape.output.leaves if isinstance(x, TensorRef)}
    for seq, op in enumerate(tape.ops):
        for value_id in op.output_ids:
            uses[value_id] = max(seq, uses.get(value_id, seq))
        for tree in (op.args, op.kwargs):
            for leaf in tree.leaves:
                if isinstance(leaf, TensorRef):
                    uses[leaf.value_id] = seq
    release = [[] for _ in tape.ops]
    for value_id, seq in uses.items():
        if value_id not in keep:
            release[seq].append(value_id)
    return tuple(tuple(ids) for ids in release)


class _Execution:
    """Per-forward value environment, never shared with any other replay."""
    def __init__(self, runner, reference, initial_overrides, fail_consumer_at):
        self.runner, self.reference = runner, reference
        self.fail_consumer_at = fail_consumer_at
        self.values = dict(runner.tape.initial_values)
        for value_id, value in (initial_overrides or {}).items():
            if value_id not in self.values or _tensor_meta(value) != runner.tape.initial_metadata[value_id]:
                raise ReplayError("initial override must preserve static tensor metadata")
            self.values[value_id] = value
        if reference is not None and len(reference.calls) != len(runner.tape.ops):
            raise ReplayError("independent eager call count differs")
        self.audit = {"consume_order": [], "validated_calls": 0, "trace": [],
            "consume_trace": [], "native": None, "checksum_xor": 0, "base_seq": runner.next_seq}

    def execute(self, local_seq):
        import torch
        runner, audit = self.runner, self.audit
        if local_seq != len(audit["consume_order"]):
            raise ReplayError("consume sequence order differs: " + str(local_seq))
        if local_seq == self.fail_consumer_at:
            raise ReplayError("injected consumer failure")
        op = runner.tape.ops[local_seq]
        begin = runner.native.now_ns() if runner.trace and runner.native is not None else None
        result = op.func(*_resolve(op.args, self.values), **_resolve(op.kwargs, self.values))
        end = runner.native.now_ns() if runner.trace and runner.native is not None else None
        leaves, spec = _flatten(result)
        if spec != op.outputs.spec or len(leaves) != len(op.outputs.leaves):
            raise ReplayError("static output structure changed")
        for expected, actual in zip(op.outputs.leaves, leaves):
            if isinstance(expected, TensorRef):
                self.values[expected.value_id] = actual
            elif type(expected) is not type(actual) or expected != actual:
                raise ReplayError("static non-tensor output changed")
        if self.reference is not None:
            expected_name, expected_output = self.reference.calls[local_seq]
            if expected_name != str(op.func):
                raise ReplayError("independent eager callsite order differs")
            assert_tree_close(torch, result, expected_output)
            audit["validated_calls"] += 1
        for value_id in runner.release_after[local_seq]:
            self.values.pop(value_id, None)
        if runner.trace:
            audit["consume_trace"].append({"seq": audit["base_seq"] + local_seq,
                "tid": threading.get_native_id(), "start_ns": begin, "end_ns": end})
        audit["consume_order"].append(local_seq)

    def output(self):
        result = _resolve(self.runner.tape.output, self.values)
        if self.reference is not None:
            import torch
            assert_tree_close(torch, result, self.reference.output)
        return result


class ReplayRunner:
    """Persistent native pool and Python consumer, shared by warmups and samples.

    Only one forward is active at once. Absolute native sequences increase across
    forwards; per-forward aten dependencies resolve from a fresh value environment.
    Pool construction and destruction are outside run()'s forward measurement.
    """
    def __init__(self, tape, *, native=None, workers=0, capacity=16,
            byte_limit=1048576, iterations=0, trace=False):
        self.tape, self.native, self.workers = tape, native, workers
        self.iterations, self.trace, self.next_seq = iterations, trace, 0
        self.release_after = _last_uses(tape)
        self.session, self.consumer = None, None
        self._request, self._done = threading.Event(), threading.Event()
        self._stop, self._closed, self._execution = False, False, None
        self._consumer_error = self._producer_error = None
        self.final_snapshot = None
        started = time.perf_counter_ns()
        if workers:
            if native is None:
                raise ValueError("native bridge is required for queue simulation")
            self.session = native.Session(workers, capacity, byte_limit, iterations, trace=trace)
            self.consumer = threading.Thread(target=self._consume,
                name="CPU-aten-ordered-consumer", daemon=True)
            self.consumer.start()
        self.startup_ns = time.perf_counter_ns() - started

    def _consume(self):
        import torch
        with torch.inference_mode():
            while True:
                self._request.wait()
                self._request.clear()
                if self._stop:
                    return
                execution = self._execution
                try:
                    for local_seq in range(len(self.tape.ops)):
                        seq = self.session.next()
                        expected = execution.audit["base_seq"] + local_seq
                        if seq != expected:
                            raise ReplayError("native consume sequence order differs: " + str(seq))
                        execution.execute(local_seq)
                        self.session.retire(seq)  # Real CPU operator has returned.
                except BaseException as exc:
                    self._consumer_error = exc
                    self.session.abort(str(exc))  # Wake a capacity-blocked producer.
                finally:
                    self._done.set()
                if self._consumer_error is not None:
                    return

    def run(self, *, reference=None, initial_overrides=None,
            sequence_order=None, fail_consumer_at=None):
        import torch
        if self._closed or self._consumer_error is not None or self._producer_error is not None:
            raise ReplayError("replay runner is closed or aborted")
        started = time.perf_counter_ns()
        execution = _Execution(self, reference, initial_overrides, fail_consumer_at)
        if not self.workers:
            with torch.inference_mode():
                order = sequence_order if sequence_order is not None else range(len(self.tape.ops))
                for seq in order:
                    if self.native is not None:
                        execution.audit["checksum_xor"] ^= self.native.prepare_inline(
                            self.tape.ops[seq].metadata, self.iterations, trace=self.trace)["checksum"]
                    execution.execute(seq)
        else:
            if sequence_order is not None:
                raise ValueError("sequence fault injection is serial-only")
            self._execution = execution
            self._done.clear()
            self._request.set()
            try:
                for local_seq, op in enumerate(self.tape.ops):
                    self.session.submit(self.next_seq + local_seq, op.metadata)
            except BaseException as exc:
                self._producer_error = exc
                self.session.abort(str(exc))
            if not self._done.wait(timeout=30):
                self.session.abort("consumer timeout")
                raise ReplayError("CPU consumer did not finish after abort")
            error = self._consumer_error or self._producer_error
            if error is not None:
                raise error
        result = execution.output()
        self.next_seq += len(self.tape.ops)
        execution.audit["wall_ns"] = time.perf_counter_ns() - started
        self._execution = None  # Do not retain previous-forward tensors.
        return result, execution.audit

    def snapshot(self):
        return self.session.snapshot() if self.session is not None else None

    def close(self):
        if self._closed:
            return self.final_snapshot
        self._closed = True
        if self.session is not None:
            self._stop = True
            self._request.set()
            self.session.close()
            self.consumer.join(timeout=5)
            if self.consumer.is_alive():
                self.session.abort("consumer shutdown timeout")
                raise ReplayError("CPU consumer did not stop")
            self.final_snapshot = self.session.shutdown()
        return self.final_snapshot


def replay(tape, *, native=None, workers=0, capacity=16, byte_limit=1048576,
        iterations=0, trace=False, reference=None, initial_overrides=None,
        sequence_order=None, fail_consumer_at=None):
    """Single-forward convenience wrapper; campaigns use persistent ReplayRunner."""
    runner = ReplayRunner(tape, native=native, workers=workers, capacity=capacity,
        byte_limit=byte_limit, iterations=iterations, trace=trace)
    try:
        output, audit = runner.run(reference=reference, initial_overrides=initial_overrides,
            sequence_order=sequence_order, fail_consumer_at=fail_consumer_at)
    finally:
        snapshot = runner.close()
    audit["native"] = snapshot
    if runner.session is not None:
        audit["trace"] = runner.session.trace_records()
        audit["checksum_xor"] = snapshot["checksum_xor"]
    return output, audit
