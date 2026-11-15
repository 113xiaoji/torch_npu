# R34 real A3 eager Prepare ingress comparison

Current actual results and unfinished gates: [A3_VALIDATION.md](A3_VALIDATION.md).
The candidate is compiled and CPU-tested; exact CANN9.1 v1 SDK qualification
passed for the reviewed uniform and original ragged inputs. Actual queue/model
E2E is pending an uncontended device. There is no R34 speedup claim.

The primary experiment calls the byte-pinned original DLRM with the R31 MetaSyn26
input (`c4507e…`) directly. It compares seven arms in the **same newly built
candidate package**: off(TQ1), mutex workers1/2/4 and MPSC workers1/2/4. R32's
empty-SUM output fix must remain in both on/off routes. Model definitions,
weights, sparse=True, FP32 precision, IDs and iteration synchronization stay fixed.

The worker uses `_eager_prepare`; it never selects `_eager_parallel` shadow or
`native_immediate`. `sdk_probe.py` is a separate SDK lifecycle qualification
tool. A successful SDK probe alone is not a model or performance result.

`real_device.py` imports the original R31 workload and R26 measurement helper.
Each fresh process completes CPU reference/setup/H2D before configuration,
validates eight original NPU forwards and all 26 embedding/Interaction stages,
then measures the original `model(*inputs)` through `torch.npu.synchronize()`.
Formal windows have tracing off, 50 warmups and 3000 timed forwards. Setup,
numeric checks, snapshots and close are excluded from individual durations.
The wider process CPU/wall window also includes timing-loop overhead.

All enabled arms require real qualified SDK preparation, positive admissions,
25 nonempty EmbeddingBag queued calls per forward, and matching queued,
prepared, SDK-Prepare, executed, returned and retired counts. Empty-table and
unsupported-operator fallback fences are recorded and included in E2E latency.
Coverage is explicitly partial. Workspace queries may be zero in the warm
executor-cache state; workloads and cache policy are not changed to force them.

The final close must prove admission closed, original FIFO drained, device
synchronized, all owners retired, workers stopped and real FIFO order hashes
matched. Quarantine, uncertain publication, missing counters and zero admission
fail the worker. Failed close is not retried. The outer launcher must bound and
reap a failed process; forced termination is never a successful close.

## Integration contract

Native schema1 snapshots must supply:

- `ingress: "mutex" | "mpsc"`, `workers`, `backend: "real_sdk"`,
  `real_sdk_qualified: true`, `simulated_prepare: false` on enabled arms.
- Existing per-operator `seen`, `eligible`, `queued`, `prepared`, `executed`,
  `returned`, `fallback`, `fallback_reasons`; **new per-operator
  `sdk_prepare_calls`** counts successful real payload Prepare, including legal
  cache hits. Optional `sdk_query_calls`/`sdk_cache_hits` are reported when present.
- Existing queue admissions/retirements, workers started, resource budgets and
  closure flags. Timed snapshots follow synchronization and must be drained.
- Real `order_evidence` after close: `enabled`, `passed`, `admitted_count`,
  `executed_count`, `admitted_sequence_hash`, `executed_sequence_hash`.

The R24 CANN9.2 binary profile does not qualify the A3 CANN9.1 providers. Native
qualification must be backed by the separately verified actual provider ABI,
cache, executor ownership, cross-thread execution and cleanup contract. This
harness refuses an unqualified or fallback-only package.

Worker config is a JSON object with `package`, `upstream`, `workload`,
`measurement`, `cpus: [8,9,10,11,12,13,14,15]`, and `pins` (absolute path to SHA256).
`workload` points to frozen `benchmarks/eager_parallel_r31/workload.py`;
`measurement` points to frozen `benchmarks/eager_parallel_r26/host_e2e.py`.
Pin the complete helper closure, R34 `real_device.py`/`contracts.py`, candidate
`__init__.py`, `_eager_prepare.py`, `_C`, `libtorch_npu.so`, and **every actually
mapped Ascend toolkit/provider library**. The source loader also verifies its
existing frozen upstream model/input closure. Generate these pins only after
the final native/package/SDK qualification freeze.

Stage private copies of `__init__.py`, `_eager_prepare.py` and `_C` under the
candidate package: their resolved import paths must identify that candidate,
rather than resolve back into the frozen R32 package. Other unchanged large
libraries may be symlinks; pin their actual resolved mapped paths as well as
any lexical package paths required by the configuration. The new native
`libtorch_npu.so` must be the actual candidate, not the frozen control.

Launch with the established private ELF loader and Python3.14.6t, real device14
mapped to logical0, natural GIL=false, Torch2.13.0+cpu git `cf30153…`, NumPy2.4.4,
CPU8..15, Torch intra/inter-op=1. Set `TASK_QUEUE_ENABLE=1`, `PER_STREAM_QUEUE=0`,
`ASCEND_LAUNCH_BLOCKING=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, and
`TORCH_NPU_EXPERIMENTAL_FREE_THREADED=1`. No forced GIL, host simulator, preload,
API tap or profiler is allowed in this cohort.

`launch.json` follows the R32 argument-list pattern. `argv_prefix` must include
an actual **container-side** `timeout -k 10s 180s` before the loader/interpreter,
and `worker_timeout_seconds: 180`. Add
`TORCH_NPU_EAGER_PREPARE_INGRESS={ingress}` to the container's `env` arguments.
The controller substitutes the selector as one argument, without shell parsing.
If omitted, the worker sets the arm's selector before imports. An inherited
contradictory selector fails. For `--pull-results`, include `container` in launch
JSON. Without it, each worker's output must already be visible under the local
campaign directory at the same label (shared volume).

## Commands

Local offline tests and plan generation perform no device work:

```text
python -m unittest discover -s test/eager_mpsc_r34 -v
python benchmarks/eager_mpsc_r34/campaign.py --rounds 3 --plan-only --output NEW_PLAN_DIRECTORY
```

After actual SDK/candidate qualification, use the matched host launcher:

```text
python benchmarks/eager_mpsc_r34/campaign.py --launch launch.json --config CONTAINER_CONFIG_JSON --worker CONTAINER_R34_REAL_DEVICE_PY --container-output NEW_CONTAINER_OUTPUT --output NEW_HOST_OUTPUT --rounds 3 --pull-results
python benchmarks/eager_mpsc_r34/analyze.py --campaign NEW_HOST_OUTPUT --output NEW_AUDIT_JSON
```

Every correctness arm runs before any performance window. Default three rounds
produce 7 correctness workers plus 21 performance workers (63,000 timed
forwards), labeled **pilot only**. A full 14-round Williams cycle balances each
position and each directed adjacent pair twice: 98 performance workers and
294,000 timed forwards. Use `--rounds 14` in a new directory after the pilot;
do not merge partial/changed cohorts into the formal result.

The offline analyzer imports only standard-library contracts and recomputes
native deltas, final FIFO order/close gates, raw duration statistics, exposure,
serial lifecycle and identical candidate/workload identities. It reports paired
changes versus off(TQ1), MPSC versus mutex at each worker count, p95/p99 and
process CPU per forward. Recorded numerical gates come from the worker's full
CPU/NPU comparisons; offline auditing does not reexecute tensor arithmetic.

One model caller with several Prepare workers measures Prepare-pool scaling.
It does not establish multiple concurrent caller support or SDK-internal
lock-freedom. Separate native producer/publication stress is required for those
claims; allocator and the rest of PTA are outside this ingress comparison.
