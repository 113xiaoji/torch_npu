# R24 Eager Prepare queue: CPU simulation results

Date:2026-10-29. User requested CPU simulation while the real environment is
unavailable. All numbers below are CPU simulation results, not NPU throughput,
CANN GetWorkspaceSize timings or Python3.14t gains.

## Implemented and tested path

Unchanged synthetic PyTorch inference models produce a real Eager aten sequence.
The simulation records tensor identities, metadata and dependencies, discards
captured intermediate values, and invokes actual CPU operators against values
produced by each new replay. Initial input/parameter/buffer tensors are retained
by reference and checked for full-data immutability; they are not claimed cloned.

Native Prepare workers receive copied integer metadata and perform controlled
native CPU work. The extension directly includes the production
EagerPrepareQueueCore. One consumer waits for the native ticket, executes the
actual CPU operator and returns its credit. This is trace-driven simulation,
not a transparent live CPU TorchNPU backend.

Real CANN/NPU operator preparation, original live TorchNPU FIFO integration and
device-completion fences are outside this campaign. They have separate source
and host-contract tests, not a real device qualification result.

## Environment and controls

- Windows AMD64, CPython3.13.12 with GIL enabled, PyTorch2.7.1+cpu.
- Shared process affinity:logical CPUs0–7; Torch intra/inter and BLAS threads1.
  This is eight logical CPUs, not an eight-physical-core claim.
- Queue capacity8, native simulated metadata byte budget1MiB; workers1/2/4.
  This metadata budget does not account for all live PyTorch tensor memory.
- Pools persist across warmup and samples. Startup, model capture, calibration,
  correctness gates and separate cold proof are excluded and recorded.
- Four target costs0/10/50/200us are implemented by one globally calibrated
  fixed iteration count per cost, shared by every model/arm. Median calibration
  observations were0.5/10.1/50.2/199.8us on the calibration metadata.
- The synthetic Prepare work is applied to every captured aten operation,
  including views/transpose/unsqueeze. It does not mean each such operation has
  a real CANN preparation phase or maps one-to-one to an NPU kernel.
- Three rounds, five warmups and30timed forwards per arm/round. Five arms are
  direct Eager context, same-trace serial preparation, and queues W1/W2/W4.
  Round2 reverses arm order. Each pooled arm has90actual samples.

## Results

The table is pooled serial-replay P50 divided by pooled W4 P50. Values below1
mean the queue is slower. Direct Eager has no injected preparation work and is
reported separately in raw results; these ratios must not be called speedup over
ordinary CPU Eager or over a real NPU.

| Model family | Captured aten calls/forward | target0us | target10us | target50us | target200us |
|---|---:|---:|---:|---:|---:|
|26-table recommendation |70|0.406x|0.726x|2.559x|3.998x|
|Branched MLP |10|0.506x|0.769x|2.029x|3.129x|
|Branched CNN |11|0.721x|0.890x|1.330x|3.138x|
|Unfused Transformer |32|0.361x|0.712x|2.561x|3.699x|

At target200us, the corresponding forward P50 values are:

| Model | Serial replay ms | W4 replay ms |
|---|---:|---:|
|Recommendation |16.12655|4.03380|
|Branched MLP |2.21690|0.70850|
|Branched CNN |2.84250|0.90575|
|Transformer |7.19670|1.94540|

The tested grid demonstrates a granularity tradeoff: queue overhead dominates
the0/10us cases, while50/200us controlled preparation work benefits from the
parallel queue in these workloads. No precise crossover threshold or real SDK
cost is inferred between grid points.

## Correctness, lifecycle and actual parallelism

-64correctness records cover4models x4costs x4simulation arms. Per-operator and
  full final CPU outputs match independent direct Eager execution. Input/weight
  hashes are unchanged. Last timed output of every arm block is also compared;
  this is not a claim that every timed sample ran full numerical validation.
-193native snapshots, checksums, counts and capacity/byte bounds were audited;
  native resources and live workers return to zero after every closed arm.
-33native core contract tests include early-commit synchronous-publisher
  deadlock regression, order, capacity, cancellation and ownership release.
  CPU bridge13tests and model/driver17tests passed independently.
- A separate long-body proof avoids Windows short-body CPU clock granularity:
  recommendation/W4, target50ms per operation,70operations and four actual native
  worker TIDs. Recorded worker CPU sum3.625s / Prepare span0.9416643s =3.849567x.
  The proof validates real multi-core native work under the mechanism; it does
  not establish precise CPU overlap in each short-cost performance cell.
- Short Windows GetThreadTimes observations of0ns are preserved. Raw cycles
  are not converted into invented CPU nanoseconds.

## Evidence and review

- Formal raw report: `benchmarks/eager_prepare_cpu/results/full-20261029-01.json`,
  1262908bytes, SHA256
  `c32a46fb5b61911740386097f235f8ce2895bc4e6e369b9d2601e273c875c2d9`.
- CPU source manifest14files SHA256
  `2b89dd773477e6480beaf2b46238e4ab4759f8a1b7ba0be1b75b6772efcfca7e`.
- Compiled/loaded production core SHA256
  `704f99c5cfd8faeb8640dd52965f8e5b3d85a1a9069a26ab68d957d0b6fdb23d`.
- Independent review recomputed all7200wall samples,7200processCPU samples,
  quantiles,48ratio/inverse pairs,64correctness records,193snapshots and14pins.
  No confirmed blocking finding within the supported CPU simulation domain.
- See `benchmarks/eager_prepare_cpu/INDEPENDENT_REVIEW.md` and the native/model
  test evidence. Raw results and build binaries remain ignored local artifacts.

## Existing host-sim role

The user's existing host-sim can exercise actual PyTorch/TorchNPU Eager and the
original framework queue, closer to the production connection than this replay.
Its current legacy stub lacks ACLNN query/owned-executor/HugeMem interfaces and
its kernels/events do not provide real arithmetic or delayed device completion.
An isolated explicitly simulated provider is required for the new Prepare path;
real CANN qualification must remain separate. See HOST_SIM_ASSESSMENT-r24.md.
