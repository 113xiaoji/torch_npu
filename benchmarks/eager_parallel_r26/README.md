# R26 reference and evidence contracts

These tools implement CPU mathematical references, file/runtime qualification
schemas, fresh-process experiment planning, and receipt validation. They do not
implement or certify production E/EI scheduling. The current native serial
shadow runtime executes the original backend operators and records entries;
this is distinct from both NoOp simulation and qualified device parallelism.

`run.py --execute` deliberately fails before any forward until a production
device qualification and measurement adapter exists. A configuration file,
`torch.npu.is_available()`, successful shadow execution, or fabricated receipt
cannot enable E/EI. The schema unit tests contain artificial records, not real
NPU evidence. No speedup is established by these files.

`host_e2e.py` is a separate executable host-sim timing worker for the available
off/shadow modes. It measures diagnostic overhead, not E/EI scheduling. The
completed paired screen is recorded in [the host E2E report](../../docs/R26_HOST_E2E.md).

```bash
python benchmarks/eager_parallel_r26/host_e2e.py --config shadow-config.json --output fresh-off --arm off --kind perf --round 0 --warmup 50 --samples 3000
python benchmarks/eager_parallel_r26/host_e2e.py --config shadow-config.json --output fresh-shadow --arm shadow --kind perf --round 0 --warmup 50 --samples 3000
```

Use the shadow config described below, additionally pinning `host_e2e.py` in
`source_pins`; its legacy `forwards` field is not used. Launch each row with the
same CPU/NUMA binding and explicit environment in a fresh owned process group.
Run cold off/shadow gates first (`--kind cold --warmup 0 --samples 1`), then ten
paired rounds with alternating arm order. A perf window shorter than two seconds
fails: choose a larger fixed sample count and rerun the complete cohort. Samples
are original forward through synchronize; setup, warmup, snapshots and close are
excluded. `host_latency_measured` is separate from the false real-device,
device-overlap and optimization-speedup flags.

## Commands

Run from the repository root. Each output file must be new; commands do not
overwrite a prior result or R25 artifact.

```bash
python -m unittest discover -s test/eager_parallel -p test_reference.py -v
python -m unittest discover -s test/eager_parallel -p test_measurement_contracts.py -v
python benchmarks/eager_parallel_r26/reference.py --output cpu-reference.json
python benchmarks/eager_parallel_r26/run.py --config benchmarks/eager_parallel_r26/experiment.example.json --plan-only --output plan.json
python benchmarks/eager_parallel_r26/qualification.py --config target.json --output qualification.json
python -m unittest discover -s test/npu -p test_eager_parallel_feature_pipeline.py -v
```

Planning and qualification use only the standard library. Planning works with
`python -S` and needs neither PyTorch nor NPU. Qualification hashes actual files
and checks the declared source root's Git HEAD. The CLI reports configured pins
only: `runtime_observed=false` and `real_device_qualified=false`.

For the complete original CPU model, supply the already installed nine-file
upstream source and its normal dependencies:

```bash
python benchmarks/eager_parallel_r26/reference.py --original-source /path/to/pinned/upstream --output original-cpu-reference.json
```

The optional unittest uses `R26_DLRM_SOURCE` for that same directory. Without it,
the original model test is explicitly skipped; synthetic arithmetic tests still
run. When a path is explicitly supplied, missing files/dependencies or mismatched
identities fail. The NPU entry also requires this variable and actual native SDK
qualification; unsupported devices/bindings return SKIP, never a numerical PASS.

## Numerical scope

`reference.py` computes sum bags and Gram/dot interactions using explicit
binary64 scalar loops. CPU `functional.embedding_bag`, `bmm` and indexing are a
separate implementation. Original lower-triangle order is outer `i`, inner
`j in range(i + self_interaction)`. Dense features stay first; cat mode and self
interactions have independent expected layouts. Cases cover table counts
1/2/7/26/39, variable B/D, empty/repeated bags, padding None/0/negative,
non-contiguous CPU weights, and optional sample weights. This finite-input
reference does not qualify low precision, NaN/Inf, long-bag numerical domains,
training, or NPU layouts.

`compare_four_outputs` checks all output shapes/dtypes. The caller must provide
the same-SoC/SDK qualification's exact list of defined coordinates for auxiliary
outputs 1/2/3. An empty list authorizes zero value reads; no entire undefined
auxiliary tensor is copied to CPU or compared. Shape/dtype failure stops before
any value read. A mathematical sum oracle does not invent vendor auxiliary
semantics.

Original-model reuse is bound to the frozen helper
`benchmarks/host_regions/real_dlrm_runner_r22.py`, SHA256
`a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f`.
Its audited loader removes only the unused SummaryWriter import and leaves all
ten model/function AST definitions unchanged. It verifies all nine sources,
including original `dlrm_s_pytorch.py` SHA256
`9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5`.
The original 455153-parameter, B128/N26/D16/L2 workload retains its frozen state
and input SHA256. No fake dependency modules or rewritten forward are used.
Full CPU output is compared with explicit FP64 bags/Interaction and the original
CPU bottom/top MLP modules; this is not a full FP64 MLP oracle or NPU result.

## Qualification schema 1

`target.json` requires these fields:

| Field | Meaning |
| --- | --- |
| `device_kind`, `provider_kind` | Exactly `(cpu,none)`, `(host_sim,noop)`, or `(npu,real)` |
| `source_root`, `revision` | Existing source checkout and exact current 40-digit Git revision |
| `source_files` | Nonempty list of `{path, sha256}`; resolved files must remain under source_root |
| `loaded_libraries` | Configured physical provider/library `{path, sha256, role}` pins; not a claim those files are mapped |
| `runtime` | Explicit string pins for python, torch, torch_npu, driver and cann |
| `soc` | Null for CPU/host-sim; explicit known target for NPU |
| `sdk_contracts` | NPU requires target_stream, event_record_submission_ack, event_wait, event_query, executor_lifetime, descriptor_lifetime, workspace_lifetime, error_submission_state |

The `qualify(config, observation)` API separately validates a runtime observation:
schema 1; actual provider kind, compute backend, SoC/runtime identity, mapped
library closure, and scheduling backend. Parallel qualification additionally
requires native runtime provenance, actual SDK/parallel qualification, and SDK
contract agreement. This API validates a supplied observation; it does not
collect or authenticate one. No CLI accepts an observation JSON as permission to
execute. Shadow stays unqualified even when actual compute is real NPU. Host-sim
always has false real-device numerical/overlap/performance claim fields.

## Measurement contract

`contracts.py` verifies O/B/E/EI paired cohorts, requiring ten fresh-process
rounds by default, at least 200 samples/process, and a sample window of at least
two seconds. Each timed sample contains one original forward through complete
join. CPU wall durations and device event durations are stored and summarized
separately. The device end point must follow joins of the actual consumer and
both side streams. Diagnostic traces belong in separate processes.

Each receipt binds candidate/source/model/state/input/runtime/device/precision
identity. Process IDs represent process births, not reusable numeric PIDs. Each
required operator family has a per-forward target and exact disjoint candidate
versus original execution counters. Original executes only its original route;
E must hit Embedding; EI must hit the entire declared pipeline. Boolean coverage
claims cannot bypass counter checks. Counter definitions are in
`validate_execution_counts`; they are this receipt schema, not a claim that the
current native shadow counters already expose it.

## Actual framework shadow verification on host-sim

`shadow_hostsim.py` loads a freshly built, explicitly pinned torch_npu package,
the fixed original DLRM and the typed NoOp provider. It runs five original
forwards with shadow disabled, then five with shadow enabled. This checks real
framework entry coverage, including exactly 26 Embedding calls with four
outputs each, two Cats with 29 input tensors in total, one BMM and one Index per
forward. SDK Embedding query/execute deltas and resource closure are checked
independently. No NoOp output values are copied to CPU or treated as numerical
evidence. All numerical, device-overlap and performance claim flags stay false.

```bash
python benchmarks/eager_parallel_r26/shadow_hostsim.py --config shadow-config.json --output fresh-shadow-result
```

The external config contains `schema:1`, `compute_backend:"host_sim_noop"`, the
eight externally bound `cpus`, absolute `torch_npu_package`, `original_source`
and `legacy_deps` directories, and `forwards` (1..64). `artifact_pins` and
`source_pins` map absolute paths to SHA256; the latter includes this runner,
`reference.py` and the frozen R22 helper. `libraries` maps `libtorch_npu.so`,
`libopapi.so` and `libascendcl.so` to `{path,sha256}`. An optional
`native_header_sha256` records the compiled protocol source identity. The process
must use natural FT CPython 3.14.6, candidate-only PYTHONPATH, explicit matching
LD_LIBRARY_PATH, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, `QWEN_ACL_HOST_SIM=1`,
Torch/BLAS threads one, task queue one and per-stream queue zero. Bind CPU/NUMA
externally; run in an owned process group with a timeout and saved lifecycle.

The runner uses the actual public `scope()` cleanup path and only writes
`passed:true` after every final GIL/artifact check. Its coverage proof is limited
to the declared pipeline entries; it does not fill Composite alias coverage
gaps or qualify the still-unimplemented production E/EI adapter.

Aggregation takes the median across process medians/p95s and never pools
individual samples across processes. It records process CPU time, context
switches, peak memory and host synchronization counts. It does not implement the
promotion confidence interval/noise gate and always leaves
`speedup_claim_proven=false`.

`device_overlap_proof` requires complete device-clock intervals for independent
producers on different streams, an actual shared consumer, and publication/join
evidence for every cross-stream dependency. A CPU interval, same-stream pair,
early consumer, missing join, incomplete trace or NoOp provider cannot prove
device overlap. Passing this validator on a synthetic fixture remains a schema
test, not a measured timeline.
