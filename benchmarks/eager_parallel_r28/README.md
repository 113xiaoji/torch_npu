# Bag-length-only DLRM host-sim comparison

The original model source, forward, FP32 weights, dense input, batch128,
embedding dimension16,26tables and1024rows/table are retained. Only the actual
number of indices in each bag changes among L=2,8,16,32. L2 is byte-identical
to the earlier frozen input. Larger bags retain its two original IDs, then
append a prefix of one fixed per-table128x30 integer tensor seeded20260919.
Repeated IDs remain repeated; indices are not deduplicated.

`workload.py` wraps the frozen R26 loader and records model/state/dense/input
hashes, every table's input shape and actual offset differences. It does not
define a replacement model. `controller.py` runs the existing R27 native
candidate and numerical SDK, with no native rebuild or SDK change.

An exact qualified Python3.14.6t/Torch2.13/NumPy2.4.4 runtime, candidate,
helpers and SDK must be supplied through a base configuration. Paths and hashes
are explicit; arbitrary runtime version substitution is rejected. The runner
tree retains the repository-relative `benchmarks/` hierarchy because the frozen
loader checks its neighboring helper's bytes.

```sh
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_COLD --kind cold
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_PERF --kind perf --qualification NEW_COLD/complete.json
```

The default/fixed campaign has12cold processes and144performance processes:
four L values,12rounds, three arms off/E/EI. Every performance process has
50warmup and3000timed original forwards. All processes are serialized and
bound to the same eight physical CPUs on NUMA1/oneLLC; monotonic start/end
receipts prove that the process intervals do not overlap. L order rotates
across rounds, and each arm permutation is used twice for each L.

Each arm receives identical model/state/dense values and identical inputs
within a given L. The predeclared per-L input hashes are checked independently.
Cold times the first NPU forward and one sample. Performance first checks eight
full forwards. In both modes, a separate untimed staged pass checks all26bags,
Interaction and the output; this prevents a saturated sigmoid from hiding an
intermediate error. Stage tensors are released and allocator markers drained
before final resource snapshots. No stage checks enter measured durations.

The measured SDK call counts remain EmbeddingBag26/Cat2/BMM1/Index1/Addmm6/
Relu5/Sigmoid1 per forward for every L. E/EI must actually execute13Embedding
calls per producer per forward; EI must use its independent Interaction stream.
The controller recomputes these gates from raw before/after counts, separately
for warmup, timing and the additional staged pass.

Cold qualification must complete before performance, with the same source,
artifact/runtime identities, environment, boot and workload profile. A result
below a two-second performance window is rejected; no delays or per-L sample
padding are added. Statistics are paired independently for each L; cold data
and different lengths are never pooled. CPU numerical simulation timings do
not establish real NPU performance.
