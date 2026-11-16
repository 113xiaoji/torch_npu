# R42 controlled experiment

R42 adds an opt-in atomic capture-pool activity query to the R41 native
workspace-cache hit path. `TORCH_NPU_WORKSPACE_POOL_QUERY=locked` remains the
default. The main DLRM model, original weights, eight MetaSyn input banks,
request engine, CPU8..15 budget, natural GIL-off runtime and private streams
remain fixed. No R42 throughput, p99, CPU/PSS improvement or near-linear
scaling result has been measured.

## What has passed

- Native build `0dc493a1cd8b36a98d30482df55a3c8dce0baf1ca9edc8fc5dd8e2a58bf35753`,
  an additive two-translation-unit rebuild with1120 R41 objects reused.
- Independent controlled-A/B review and CPU tests of actual pool mutation
  code, option parsing and workspace admission.
- Actual NPU qualification on the original model's largest nonempty table2
  EmbeddingBag: locked88 + atomic88 outputs. The same original operator and
  weights are used for warm, three native MemPool options, ordered cross-thread
  scope observation and capture/replay. All eight original banks are used in
  ordinary scope tests; graph warm/capture/eight replays use fixed original
  bank0. Private segment pool IDs, cache bypass/restoration, numerical outputs,
  original state/input hashes, source/native/provider and effective flags passed.
- The scope stage was independently sealed after fresh verification of175
  runtime/source/SDK/actual-provider pins and21 raw evidence pins using the
  same exact image in an owned devices-empty CPU verifier.

These checks cover this single original operator and ordered native scopes.
They do not certify full DLRM graphs, unordered cross-thread pool transitions,
custom allocators or uninterrupted exclusive use of the accelerator.

## What remains

V4 consumes only complete sealed stages with matching native/source/runtime/
SDK/environment/input bindings:

`scope176 → fresh model384 → seal → fresh smoke288 → seal → STOP ready`

The cross-model stage is a four-task numerical check for each MLP/CNN/attention
case. Its executor submits four task indices but records no native thread
roster or entry barrier; it does not prove four distinct callers entered
simultaneously. The formal DLRM engine has separate explicit caller, thread
roster and barrier gates.

A separate explicit decision is required before27 fresh interleaved performance
cells (three arms ×1/2/4 callers ×three rounds,3000 requests/warm24). The three
arms are frozen R41 condition/cache1 without a pool-query flag, R42 locked and
R42 atomic. No historical performance is reused. Incomplete old smoke sets do
not count as a complete stage.

The actual standalone-mm diagnostic found zero workspace-cache activity;
`mempool_sanity.py` is retained only for old source-closure/evidence fidelity.
Use `mempool_sanity_embeddingbag_v3.py` for the positive-workspace qualifier.
The original DLRM diagnostic did reach the cache and passed its CPU comparisons.

Foreign SGL processes repeatedly interrupted the selected two-chip card after
fresh idle checks. All owned containers/holders/namespaces were cleaned; no
foreign task was changed. Resource checks combine per-chip processes and usage,
UDA and fuser. Observations are sampled and do not establish a reserved card.

See [R42_STATUS.md](../../patches/linear_scaling_r42/R42_STATUS.md) for current
stage status and raw archive hashes. Host orchestration is specific to the pinned A3 runtime
under `/tmp/zyz-r42-capture-query-20261117/campaign-runner`; it is prepared and
reviewed separately from the measured worker source closure. Importing the
worker modules or running host `--plan-only` does not execute NPU work.
