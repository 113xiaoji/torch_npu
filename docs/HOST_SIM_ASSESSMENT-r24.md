# Existing host-sim applicability to R24

2026-10-29. User asked whether their existing host-sim can validate the Eager
parallel-Prepare/original-consumer FIFO scenario. This is an audit, not a claim
that the R24 candidate has already been built or run on host-sim.

## Finding

The environment is reusable and provides a closer framework integration test
than a standalone operation replay. It exercises real PyTorch/TorchNPU Eager,
the native operator bridge, interpreter ABI/GIL state and the original TaskQueue.
The current simulator does not, as-is, implement the ACLNN query/executor API
needed by the new owned Prepare path. Add a separate simulation provider; do not
modify or bypass the exact real-SDK qualification guard.

## Fresh read-only remote evidence

- Existing CPU host was reachable; hostname node04, architecture aarch64.
- Ordinary and free-threaded interpreter files exist at the previous runtime
  locations. No installed runtime or source tree was changed by this audit.
- Fresh isolated standard-library checks report CPython3.14.6 in both runtimes,
  ordinary SOABIcpython-314/GILtrue and free-threaded SOABIcpython-314t/GILfalse.
  This does not claim the post-TorchNPU-import GIL state was freshly tested.
- Default existing ACL simulation library observations:

| Library | bytes | SHA256 |
|---|---:|---|
| libascendcl.so | 76224 | 2c454e9e7c39136da3e862567310ca0b02f54a0aba87da16f80e89774988dba4 |
| libacl_op_compiler.so | 77000 | 60e1fb105d9a3a781f11d4753e6c00921a96ffcfbfd0ec34d17f265bf19f80a2 |

Both nm -D calls exited0. The default library directory contains no libopapi.so.
Across those two libraries the inspected symbol set lacks aclCreateTensor,
aclnnAddGetWorkspaceSize/aclnnAdd, aclnnMmGetWorkspaceSize/aclnnMm and
InitHugeMemThreadLocal/UnInitHugeMemThreadLocal/ReleaseHugeMem. Legacy creator
symbols are present but the corresponding original source returns NULL.

A fresh isolated8-byte positive control on that exact libascendcl succeeded:
host-sim malloc returned a real non-null pointer, host-to-sim and sim-to-host
memcpy returned0 and roundtrip bytes matched, then free returned0. No model,
kernel, installed file or global simulator configuration was changed. This
confirms real host memory copy rather than inferring it only from source text.

Historical native-prepare evidence-r3 std/FT receipts and verify_backend_gate.py
independently recorded the same two library hashes and blocked before adapter
build. Thus new symbol absence is not merely assumed from a source filename.

## Simulator versions must remain distinct

- Original ACL hostsim performs real host malloc/free and synchronous memcpy.
  Streams/events use opaque small allocations, completion operations are no-ops,
  and legacy operator execution does not compute outputs.
- The later R20 allocating ACL stub adds descriptor/data-buffer allocation and
  release counters. Its handles are opaque rather than complete tensor metadata;
  it still lacks the new ACLNN executor profile and CPU numerical kernels.
- HostDispatchBench explicitly labels its existing Qwen Prefill/Decode and DLRM
  paths eager/acl_host_sim/immediate_stub. Compiled/replay adapters are separate
  unimplemented modes, not evidence of native R24 integration.

## Recommended combined validation

1. Existing host-sim for real Eager/frontend/native/originalFIFO integration,
   sequence/copy/consume counters, thread identity, GIL and dispatch CPU cost.
2. An isolated HOST_SIM_R24 provider with exact typed metadata, unique executor
   ownership, configurable CPU Prepare work, out-of-order readiness, direct
   consume-once Execute and cancel/error counters. Distinct backend identity and
   library hashes; real CANN9.2 qualification remains unchanged.
3. Stateful simulated completion to test delayed retirement. Immediate no-op
   events prove fence invocation only, not safety while device work is pending.
4. Actual CPU model execution/reference for full numerical comparisons. Existing
   no-op simulator output must not serve as a numerical golden; implementing CPU
   Execute kernels would be a separate additional simulator capability.

Prepared CPU operation-trace simulation remains useful for value-dependency and
scheduling-cost tests. It must be called simulation rather than a transparent
live CPU Eager backend. Neither route supplies real NPU performance or SDK cache/
HugeMem/device-retirement qualification.

## Existing source/evidence references

- Original and allocating ACL source under
  experiments/coarse-npu-fastpath-20261024/target-native-builder-r20/.
- experiments/native-prepare-20261024/verify_backend_gate.py and evidence-r3/.
- experiments/coarse-npu-pipeline-20261024/REPORT.md and native-contract-r1/.
- Existing HostDispatchBench README and src/host_dispatch_bench/execution.py.

R24 has also exposed an actual first-publication INIT drain deadlock using the
real OrderedMpscQueueCore. Early commit ACK is required; this illustrates the
value of exercising the actual host framework FIFO rather than only a model
of a queue. The production ACK fix and independent recheck are recorded separately.
