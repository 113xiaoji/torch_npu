# R25 actual DLRM host-sim first screen

This runner answers a narrow measured question: with the same new native
candidate and the same new simulator provider loaded in every arm, does the
native eager Prepare queue improve original DLRM host forward latency through
the original `torch.npu.synchronize()` return? It never qualifies a real SDK,
device kernel, numerical output, training path, or model quality.

The old R24 CPU queue tests and these local host fixtures are not DLRM
performance evidence. Actual results exist only after the controller runs the
original model on a native, explicitly pinned host-sim runtime.

## Fixed protocol

- One ABI is enough for the first screen. FT CPython 3.14.6 uses its natural GIL
  state (`false`); std 3.14.6 uses `true`. `PYTHON_GIL` is removed before launch
  and forbidden in the worker. The lexical virtual-environment interpreter path
  is preserved. Adding std later is an independent same-protocol screen.
- All four arms use the **same candidate package**, built with
  `TORCH_NPU_EAGER_PREPARE_HOST_SIM=1`, and the same explicitly pinned new
  `libascendcl.so` and `libopapi.so`: `off`, `queue_w1`, `queue_w2`, `queue_w4`.
  It is a global native operator opt-in. There is no model rewrite, region AST,
  legacy worker adapter, or artificial preparation CPU/delay in the benchmark.
- Eight explicit distinct physical CPUs, one NUMA domain and one LLC. The
  controller verifies sysfs with the pinned R22 topology helper, binds CPU and
  memory using `numactl`, and records a fresh cold `/proc/stat` observation and
  `/proc/loadavg`. Every observed process thread must stay within the budget.
  This records background load and topology, not exclusive CPU ownership.
  The historical 144,146,148,150,152,154,156,158 mask may be used only after
  current availability/topology verification by the operator.
- Torch intra/inter-op, OMP, MKL, OpenBLAS and NumExpr threads are one. The
  original asynchronous `TASK_QUEUE_ENABLE=1`, `PER_STREAM_QUEUE=0`, and
  `ASCEND_LAUNCH_BLOCKING=0` settings are identical in every arm.
- Four independent cold diagnostic processes run first, one per arm. Native
  Prepare trace and SDK trace are enabled only here. Each queued arm must show
  exactly 26 actual EmbeddingBag commands in its single original forward, requested actual
  Prepare worker TIDs, the distinct original caller and ACL consumer TIDs,
  matching source-sequence counts/hashes, and full resource closure.
- Three fresh-process matched rounds follow. Each arm loads and transfers the
  same model, configures its pool, warms up 20 forwards, then measures exactly
  200 original forwards. Pool creation/import/load/transfer/warmup are outside
  the sample window and pool startup time is separately recorded. Every sample
  begins immediately before the original model call and ends immediately after
  original synchronize returns. There are no sleeps or synthetic prep loops in
  the model measurement window. The cold availability observation is outside
  all model processes and timing windows.
- Declared arm orders are `(off,w1,w2,w4)`, `(w4,w2,w1,off)`, and
  `(w2,w4,off,w1)`. This is a small counterbalanced first screen, not a complete
  four-position Latin square. Each candidate is paired with that round's off
  median. The three pair ratios, range, median and p95 ratios are reported.
  Samples within a process are not pooled as 600 independent A/B experiments;
  the same round's off process is shared across candidate comparisons.
- A direction consistent in three rounds is labeled only “observed in first
  screen”; mixed rounds produce no stable-direction label. There is no formal
  statistical-significance claim. Extend exposure/rounds only after inspecting
  this screen and fresh machine noise, in a fresh output directory.

## Original workload identity

The inference loader is reused by **exact path plus SHA**, not edited/copied:

| Item | SHA256 / value |
|---|---|
| `real_dlrm_runner_r22.py` inference/source/identity helpers | `a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f` |
| `measurement_contracts_r22.py` topology helper | `5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a` |
| original upstream `dlrm_s_pytorch.py` | `9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5` |
| original upstream commit | `9bc1bc3602aea22afad3206910cec50f2dbe7364` |
| model state tree | `6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19` |
| input tensor tree | `239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6` |
| Torch git | `cf30153c4c131c8164ee7798e5022d810682e2cb` |
| NumPy | `2.4.4` |

The original manifest's complete source/dependency file sets and runtime
dependency pins are verified before numerical libraries/model code import.
Its dependency list verifies the complete historical snapshot; it does not
claim to describe every file actually imported by this run. The candidate is
first in the search path, followed by the interpreter's native packages, with
the original model source and dependency snapshot appended as fallbacks.
This keeps the FT interpreter's NumPy 2.4.4 and its `cpython-314t` extension
ahead of the old dependency snapshot's incompatible standard-ABI NumPy.
The loaded NumPy package and `_core._multiarray_umath` extension must both
match explicit runtime artifact path/SHA pins. Ready receipts record those
actual imports and the effective search path; the Torch git, natural GIL,
source, state and input checks remain required.
R22 verifies all nine original source files again and imports original helpers
from the pinned source tree. Its only loader adaptation omits the exact unused
top-level line 101 `SummaryWriter` import; all other AST nodes and every model
function are unchanged, and the adaptation proof is stored. This does not
certify the training CLI or the original logging import closure. None of R22's
real-device gates are invoked or weakened by this separate SIM runner.

The model remains FP32, 455153 parameters, batch 128, 13 dense features, 26
1024-row tables of width 16, two indices per bag, sum pooling, sparse=True,
bottom MLP `[13,64,32,16]`, top MLP `[367,64,32,1]`, **original MLP bias=True**,
dot interaction without self-interaction, state seed 20260917 and input seed
20260918. No forward method is assigned or modified. CPU state/input bytes are
hashed before transfer. Output is checked for shape `[128,1]`, FP32 and NPU
device metadata only. NoOp provider output bytes are never a numerical golden.

## Runtime configuration

Configuration is external and deliberately contains no arbitrary environment
override. It records actual source/build artifact pins **before import**. All
paths below are explanatory placeholders to replace with verified physical
paths; no file is installed or overwritten by this runner.

```json
{
  "schema": 1,
  "cpus": [144,146,148,150,152,154,156,158],
  "numa_node": 1,
  "helpers": {
    "r22": {"path": "/isolated/tools/real_dlrm_runner_r22.py", "sha256": "a70a29413c63fc2fcd820950d105dd78eb2742d8277cdd855ce5f2d515c7061f"},
    "topology": {"path": "/isolated/tools/measurement_contracts_r22.py", "sha256": "5c62b59f811eab2b120bead3d843ee1d9979365851f5967e7c3caca8b850441a"}
  },
  "sdk": {
    "backend": "host_sim",
    "real_sdk_qualified": false,
    "source_pins": {"/isolated/source/r25_hostsim_opapi.cpp": "REPLACE_WITH_ACTUAL_SOURCE_SHA256"},
    "libraries": {
      "libascendcl.so": {"path": "/isolated/new-sdk/libascendcl.so", "sha256": "REPLACE_WITH_ACTUAL_ARTIFACT_SHA256"},
      "libopapi.so": {"path": "/isolated/new-sdk/libopapi.so", "sha256": "REPLACE_WITH_ACTUAL_ARTIFACT_SHA256"}
    }
  },
  "runtimes": [{
    "name": "ft",
    "python": "/existing/ft-venv/bin/python3",
    "expected_gil": false,
    "pythonpath": ["/isolated/new-candidate"],
    "ld_library_path": ["/isolated/new-sdk", "/existing/other-required-libs"],
    "torch_npu_package": "/isolated/new-candidate/torch_npu",
    "artifact_pins": {
      "/isolated/new-candidate/torch_npu/lib/libtorch_npu.so": "REPLACE_WITH_ACTUAL_ARTIFACT_SHA256",
      "/isolated/new-candidate/torch_npu/_eager_prepare.py": "REPLACE_WITH_ACTUAL_SOURCE_SHA256"
    },
    "model_manifest": {"path": "/existing/original/ft-dlrm-assets.json", "sha256": "REPLACE_WITH_ACTUAL_MANIFEST_SHA256"}
  }]
}
```

The manifest remains the original runtime-specific asset manifest, including
its original `source_dir`, `import_deps`, source/dependency hashes and seeds.
`pythonpath` must contain only the candidate package's parent directory.
Legacy dependency directories are rejected there and remain fallback paths.
Add the native NumPy `__init__.py` and `_core/_multiarray_umath` extension to
`artifact_pins` using their actual interpreter-specific absolute paths/hashes.
`PYTHONPATH` selects only the isolated new candidate and explicit existing
dependencies. `LD_LIBRARY_PATH` is replaced with the explicit list, never
extended from an inherited host value. The worker checks the imported package
directory and native artifact hashes, and actual mapped simulator paths/SHA.
Duplicate or deleted mappings fail. The identical map/artifact/input/state
identity and native EmbeddingBag coverage must match each arm's cold gate
before its performance result is accepted.

## Commands and evidence

```bash
# Pure standard-library host fixtures; no torch/device/model execution.
python -m unittest discover -s test/hostsim_r25 -v

# Plan only, no child model process. Output directory must not exist.
python benchmarks/hostsim_r25/run.py --config /absolute/r25.json --output /absolute/fresh-plan

# Only the controller owner launches this after isolated artifacts are ready.
python benchmarks/hostsim_r25/run.py --config /absolute/r25.json --output /absolute/fresh-screen --execute
```

The controller creates a separate POSIX session/process group for each owned
worker, uses a bounded 600-second timeout, and terminates the group on failure
or interruption. A still-live descendant makes closure fail. It preserves
configuration, declared plan, runner source hashes, topology/background
observation, exact argv, worker logs, lifecycle, every completed result and the
first controller error. The worker saves an original error before cleanup and
preserves that error if cleanup also fails. Failed runs never produce a timing
summary. Existing output directories are never reused.

Every enabled arm must balance native admitted/prepared/phase2-entry/return and
retired command counts; simulator Prepare/Execute counts independently match
exactly 26 EmbeddingBag calls per original forward. Cold native Prepare traces
must contain exactly 26 records; SDK traces can also contain other legitimate
model operators. All queue slots/retained
bytes, SDK descriptors/executors and huge-memory thread scopes close to zero.
The disabled arm starts no owned core and must retain zero queue admissions,
workers and resources; it has no enabled-core close to claim. SIM identity
remains `host_sim` / `HOST_SIM_R25`, `real_sdk_qualified=false`,
`qualified_ops=[]`; supported native SIM schemas are separately reported.

Current qualification covers EmbeddingBag only. Add/Relu/Mm and other original
model operations retain their original eager path. Reports expose registered
hook seen/eligible/queued/fallback counters and explicitly describe **partial
native operator coverage**. Those counters are not an all-ATen census. An
all-fallback enabled arm fails rather than becoming a pipeline result.

`summary.json` and `report.md` are created only after all cold gates and all
12 performance processes for each selected ABI pass. A pool-startup-free host
latency change is the allowed conclusion; true kernels, device throughput,
numerical equivalence, CPU availability from wall-minus-CPU, and training
behavior are outside the measured scope.
