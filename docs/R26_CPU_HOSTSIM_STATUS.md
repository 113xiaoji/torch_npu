# R26 CPU/host-sim implementation checkpoint

This checkpoint implements and tests the SDK-independent dependency protocol,
joint feature-pipeline simulator, CPU references and native entry diagnostics.
It does **not** enable or claim production NPU side-stream scheduling. The user
confirmed that real NPU hardware is still unavailable and requested CPU/host-sim
validation first. No speedup is claimed by this checkpoint.

The subsequent [host E2E screen](R26_HOST_E2E.md) measured same-package off versus
shadow over ten paired rounds. Shadow increased paired median latency by 2.10%
and p95 by 1.99%. This quantifies diagnostic cost, not the unfinished E/EI optimizer.

## Implemented components

- `DependencyScheduler.h`: caller-owned logical sequence, whole-allocation
  RAW/WAR/WAW effects, explicit entry frontier, two producer stream identities,
  deduplicated consumer fan-in, publication/completion separation, bounded node/
  retained-byte/history state, allocation/event generations and retirement.
  Backend and launch callbacks cannot reenter mutation APIs. Failed or uncertain
  device submission cannot be replayed through the original path.
- Allocation failure tests drove transactional admission and nonallocating
  retirement bookkeeping. The graph transaction currently copies bounded state;
  it is a correctness-oriented implementation without a hot-path speed claim.
- The deterministic fake device executes actual CPU arithmetic through the
  protocol: 26 Embedding producers with out-of-order completion, two stream-tail
  waits, Cat, Gram, ordered triangular Index and final Cat. Four completion seeds
  match an independent long-double scalar oracle. This is an SDK-independent
  simulated device, not the installed CANN runtime.
- Native `EntryGuard` and the `_eager_parallel` bridge expose default-off
  **shadow** mode. Original operators and original stream execute unchanged;
  entries/boundaries are counted. Shadow owns no additional device work, so its
  close acknowledgment is not a device-synchronization acknowledgment.
- Code generators put entry guards before backend wrapper metadata reads;
  central storage resize, set-storage and host scalar observation are recorded.
  The R26 op-plugin overlay is independent of frozen R24/R25 payloads.
- CPU numerical references, original-model source/state/input validation,
  qualification and measurement schemas, plan-only tooling and real-NPU test
  entry are provided. Parallel `--execute` refuses to run without the missing
  qualified execution adapter; it never silently substitutes shadow.

## Numerical evidence

On the existing AArch64 CPU test host, natural free-threaded CPython 3.14.6 and
Torch 2.13.0+cpu executed the original DLRM and the new CPU reference report.
All 15 stratified reference cases and the original 26-table model checks passed.

| Original CPU model stage | Maximum absolute error against the declared reference |
| --- | ---: |
| Each of 26 Embedding outputs | 1.862645149230957e-9 |
| Interaction output `[128,367]` | 7.892543729681734e-9 |
| Final model output `[128,1]` | 5.960464477539063e-8 |

The final comparison combines explicit FP64 bags/Interaction with the original
CPU bottom/top MLPs; it is not a full FP64 MLP oracle. Model definitions and
forward remain unchanged. The frozen inference loader omits only the previously
audited unused SummaryWriter import. Auxiliary values are compared only where
the per-backend contract defines them.

| Evidence identity | SHA256 |
| --- | --- |
| Original model source | `9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5` |
| Original state tree | `6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19` |
| Original input tree | `239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6` |
| Original CPU report | `30bed1a275f8fa5e6b84a8cdd7e5b6ebd6a0701ac77f410efcea9460c8796ead` |

## Native integration evidence and limits

Real target Torch 2.13/CPython 3.14t headers compiled the entry collector type
surface. A separately built probe linked the production `EntryGuard.cpp` and
`EagerParallelPython.cpp` with a small test module initializer. Actual CPU mm/cat,
nested entries and original exception preservation passed, with natural GIL
false before and after import. Disabled first-use Invoke/Boundary allocated no
memory in the instrumented probe. This is a production-source binding smoke;
it does not substitute for a complete torch_npu library build.

All five real target code-generation steps completed. The generated-source audit
found 1,955 distinct schemas, including the Embedding/Cat/Bmm/Index chain.
Composite `reshape`, `transpose.int` and `slice.Tensor` do not yet have complete
entry coverage. The runtime consequently reports
`full_entry_coverage_qualified=false`, `effects_registered=false`,
`real_sdk_qualified=false` and `device_parallel_qualified=false`.

SSH access temporarily failed before the full build could start. After actual
connectivity recovered, frozen source and generation hashes were checked again.
A fresh complete AArch64 FT native build and new `_C` both passed, with no old
object reuse and ccache disabled. All 5,738 captured code-source hashes matched
after compilation. Actual objects include the new EntryGuard/Python binding and
generated RegisterNPU/OpInterface units. All owned build processes exited before
model validation. Prior R25 sources and artifacts are untouched.

The new package then ran the original DLRM on host-sim: five forwards with
diagnostics disabled and five with native shadow enabled. The enabled group
recorded 130 Embedding calls and 520 output tensors, ten Cats and 145 input
tensors, five BMMs and five Index calls. Each group's simulator SDK independently
prepared/executed 130 Embedding calls. Output metadata was `[128,1]`, FP32, NPU0;
NoOp output values were never a numerical oracle. All SDK temporary resources
closed, native active calls were zero, natural GIL stayed false and the worker
process group exited. The 250 outer entries and 260 nested entries are diagnostic
coverage, not 250 new scheduled device nodes.

| Actual integration artifact/evidence | SHA256 |
| --- | --- |
| Full rebuilt main library | `84a0ac2b98a05c041ed7b8537caf849f3c1bf4d25a53eb916ffdb76606c6def3` |
| Fresh FT `_C` | `7b498e65b46fd508132786c639dc121b65997e3fabfbc9675c83e29e33a4ddcb` |
| Actual shadow verification runner | `9c291af200063522ff858403c8d0ff69bf33c0c08dc646e94d4fd8e2c844dc45` |
| Private validation evidence archive | `efb88cf139bf8104b736a204b656406d659b7ee09cd9fd3eb8c58d04b284a0b1` |

The archive has 56 pinned evidence files plus its manifest. Independent review
verified all member hashes, source/model/input identity, coverage counts, build
receipts and sanitizer results. Raw logs and binaries are not included in Git.
Focused local tests cover 48 existing R26 tests (47 pass and one optional
original-source test skipped locally) plus four shadow-runner gate tests, all
four passing. The original model that was skipped locally was actually run on
the AArch64 host as documented above. The real-NPU test is explicitly skipped
without actual device qualification.

## Sanitizer evidence

The current protocol revision
`073d87a6140b8ecc48a3caaedddc0d81806f6ef7b81d9ef684b7d9f4094aad50`
passed Linux Release CTest 2/2, AddressSanitizer CTest 2/2 and ThreadSanitizer
CTest 2/2. The GCC TSan runtime was missing; existing Clang 17 compiler-rt was
used. Its shared runtime preserved the test replacement allocator without a
duplicate-definition workaround. A clean probe passed and a deliberately racy
probe was detected (exit 66) before the full protocol tests. No system runtime
installation or sanitizer suppression was added. These checks cover the tested
SDK-independent protocol and fake-device arithmetic, not vendor CANN internals
or the unimplemented production device adapter.

## Remaining implementation and validation

1. Complete Composite/manual alias/observation coverage before execution admission.
2. Connect real allocator DataPtr leases/generations and mutation guards; holding
   StorageImpl alone does not protect a replaced allocation.
3. Implement/qualify the actual event-publication, stream-wait and completion
   adapter, then connect the protocol to production E/EI execution.
4. Extend the passing standalone Linux Release/ASan/TSan checks to the completed
   adapter, then run full numerical/device-overlap gates and paired whole-model
   performance on real hardware.

These are functional and validation prerequisites, not merely a missing NPU
benchmark. The current CPU/fake-device proofs and shadow diagnostics cannot be
described as a completed production multi-stream optimizer or a guarantee that
the unimplemented adapter has no synchronization problems.
