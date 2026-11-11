# R32 real A3 validation

The model and input are the frozen R31 MetaSyn26 projection. The full input
SHA256 remains `c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1`.
The original model's entirely empty table is retained.

Apply the native source replacement in `patches/eager_parallel_r32/op-plugin/`
to the pinned op-plugin checkout, after verifying the canonical LF base and
replacement hashes in `patches/eager_parallel_r32/overlay.json`. It changes both
EmbeddingBag entries: a completely empty SUM output is initialized with native
zeros before the original CANN call. Argument validation, auxiliary tensors and
the original nonempty operator body remain in place. This is a source overlay,
not a Python dispatcher override.

Build R31 with `TORCH_NPU_EAGER_PREPARE_HOST_SIM=OFF` against real CANN and build
a fresh `_C` binding for the matching Python/Torch ABI. Do not reuse host-sim
binaries or its SDK007. All five arms use this same corrected native package:

| Arm | Native scheduling | Minimum metadata mean L |
| --- | --- | --- |
| off | disabled | 0 |
| e | Embedding | 0 |
| ei | Embedding and Interaction | 0 |
| ae | selective Embedding | 64 |
| aei | selective Embedding and Interaction | 64 |

`empty_regression.py --output FILE` performs allocator-poisoned real-operator
regression checks against CPU for both entries, index dtypes, offset conventions,
and complete/partial emptiness. This is a separate test workload; it does not
alter the DLRM inputs used in the comparison.

`real_device.py` accepts a JSON configuration with `package`, `upstream`,
`workload`, `routing_gates`, `measurement`, `cpus` and a `pins` path-to-SHA256 map.
The workload is R31 `workload.py`, routing gates are R31 `host_e2e.py`, and the
measurement helper is R26 `host_e2e.py`. Only their pure measurement/validation
functions are reused; host-sim runtime APIs are never loaded or called.
The worker requires Python 3.14.6t with natural GIL disabled, Torch git
`cf30153c4c131c8164ee7798e5022d810682e2cb`, NumPy 2.4.4, original upstream DLRM
source and its real import dependencies. It compares all 26 embedding outputs,
Interaction and the final output outside the timed window, and checks actual
native scheduling/event counters, closure, loaded CANN mappings and source pins.
The mapping gate accepts the monolithic opapi library or CANN 9.1's installed
`libopapi_nn.so` and `libopapi_math.so`; the latter are resolved first by the
pinned op-plugin loader. Loaded feature libraries must be included in the pins.

`campaign.py` runs from the host. Its launch JSON contains `argv_prefix`, the
argument array that launches the worker interpreter in the task's environment.
`--output` is the host result directory and `--container-output` is the same
result directory seen inside the container. When no volume is shared, use
`--pull-results` to copy each finished result back to the host outside timing.
The launch JSON then also needs `container`, its Docker container name.
It runs five correctness processes first,
then ten balanced Williams rounds, with 50 warmups and 3000 samples per arm per
round. Each sample contains the original forward and device synchronization.
Setup, H2D, CPU reference, stage checks and snapshots are excluded. Raw durations,
worker results, process intervals and per-round comparisons are retained.

The native runtime's global qualification flags remain unchanged; this benchmark
provides evidence for this fixed workload on the tested device, not universal
SDK/model certification. Native routing counts do not prove overlapping kernel
execution; a device timeline is needed for that stronger claim.
