# R24 independent harness review — 2026-10-29

Reviewed only `benchmarks/eager_prepare`, `test/eager_prepare`, the public
`torch_npu._eager_prepare` adapter, and the referenced immutable topology
contract. Native integration was still being edited and is not certified by
this review. No source, original R22/R23 file, staging area, or commit was
changed. This report is the only review-authored file.

Review baseline: `SOURCE_PINS.json` SHA256
`7cc9fbd807c4dc313b0e84e801a6f1cb9e74e8fa7ebd6fb7c2b6aff0c12736a4`.
All 11 listed file hashes and byte sizes matched when reviewed. Runner SHA256
was `38e68d325513e8183ee69fa5edb059ffd5d2376d5c8eb2e22a7cffa8c21a7ebe`.

## Actionable findings

### [P1] Bind the actual ACLNN/provider mappings, including after timing

Location: `runner.py:168–181`, with the sole mapping check at `runner.py:255`
and the end-of-run file-only checks at `runner.py:300–305`.

`sdk_maps()` requires the mapped `libtorch_npu.so` and mapped AscendCL to
match approved paths, but handles the other SDK providers by returning only
the intersection of mapped paths and receipt paths. An approved `libopapi.so`
may never be loaded while an unapproved copy supplies actual ACLNN operations;
the unknown mapping is silently omitted. Hashing the unused approved file
before/after execution does not establish the identity of the used provider.
There is also no second mapping check after timing, so later provider loading
is outside the checked observation.

Host-only reproduction used the real `sdk_maps()` and `contracts.require()`
with a fake `/proc/self/maps` reader. The fixture reported these mapped paths:

```text
/venv/lib/libtorch_npu.so
/approved/cann/lib64/libascendcl.so
/unreviewed/cann/lib64/libopapi.so
```

The receipt approved `/venv/lib/libtorch_npu.so`, a `cann_prefix` of
`/approved/cann`, and `sdk_files` containing the approved AscendCL plus
`/approved/cann/lib64/libopapi.so`. The mocked file-hash function returned the
receipt digest for approved files. The actual observed result was:

```text
UNAPPROVED_OPAPI_ACCEPTED [{'path': '/approved/cann/lib64/libascendcl.so', 'sha256': 'approved-hash'}]
```

No exception was raised. This test did not load any SDK or device library.

Required correction: identify the full set of actual providers relevant to
the qualified SDK operation contracts, require their resolved paths/hashes to
be in the approved provider set, and reject unexpected alternatives. Repeat
that observation after timing and bind it into the result/pair identity. A
meaningful regression test must present an approved but unused provider and
an unapproved mapped alternative and require rejection; also test a provider
first appearing after calibration.

### [P2] Bind execution to the approved harness manifest before loading code

Location: `runner.py:185`, `runner.py:201`, `runner.py:299–305`,
`contracts.py:133–140`; the manifest is `SOURCE_PINS.json`.

The reviewed manifest is never read or validated by the runner. The artifact
receipt also carries no required harness-manifest digest. At startup the
runner loads `contracts.py` and `models.py`, then records the current hashes
of only those files and `runner.py`; the final check verifies that those
three hashes stayed unchanged during that process. Consequently, a stale or
modified harness copied before both paired launches can produce matching
off/on source hashes and `performance_valid=true` without matching this
approved baseline. The analyzer checks equality between the two observations,
not equality with an independently pinned manifest. This is a source identity
gap, not a report that the currently reviewed files have drifted.

Evidence: all 11 manifest entries were verified independently in this review,
but the executable validation path has no `SOURCE_PINS` reference; its only
harness-source inventory is the three-name tuple at line 201. The successful
pair fixture in `test_contracts.py` likewise accepts an arbitrary
`harness_sources={"models.py": "same-source"}` dictionary without any approved
manifest identity.

Required correction: require an explicit approved harness-manifest SHA in
the artifact receipt, verify that manifest and all listed source bytes before
loading harness helpers, and verify all listed sources again at completion.
Carry that manifest identity into the result and require it to match in pair
analysis. The reviewed outer launcher can provide the trust anchor for the
runner file itself. Test a source change before launch, a changed manifest,
and a changed non-three-file manifest entry; all must be rejected rather than
accepted because both modes happen to use the same drifted copy.

## Scope clarification, not an additional blocking finding

Qualification runs one calibration forward, then snapshots, then one traced
sample (`runner.py:251–258`, `runner.py:265–287`). The trace delta therefore
describes the second forward after calibration. This is an allowed bounded
diagnostic separate from the uninstrumented performance run; it need not
capture the first invocation. The wording “cold” should explicitly mean a
fresh-process diagnostic with calibration1 + sample1, not first-call SDK
workspace-query cost or cold-cache timing. The current code already labels
Prepare spans inclusive rather than pure SDK timings, and never marks this
qualification as performance-valid.

## Verification and positive observations

Independent command, run from the worktree root with the installed host
CPython 3.13 interpreter:

```text
python -I -B -m unittest discover -s test/eager_prepare -p 'test_*.py' -v
Ran 20 tests in 3.113s
OK
```

These are host tests, including a real Torch CPU subprocess smoke. The
existing four workload CPU-smoke receipts were inspected as CPU-only evidence;
their `real_npu` and `performance_valid` fields are false. This reviewer did
not repeat or claim target SDK/NPU execution.

- Recommendation, branched MLP, CNN and Transformer execute the declared
  core model operations; tests hook actual modules/functions rather than
  merely counting declared members. The harness does not patch model forward
  methods or compile model graphs.
- Calibration and final output validation examine every tensor value, require
  finite values, shape/dtype equality and fixed-tolerance allclose, and retain
  complete output/reference bytes. Input/model identity is checked before and
  after execution. Numerical observers are outside the timed loop.
- Coverage is explicitly separated into eligible-operation coverage and
  model-core full/partial/none, independently of numerical success and
  performance validity. Configured workers do not constitute parallelism proof.
- The campaign planner generates 48 ordered cases with three predeclared
  rounds, fresh-process commands and 20 warmups/2000 samples. Statistics use
  mean and nearest-rank quantiles correctly. The external process owner is
  still responsible for launching, timeout, exit/group reaping and actual
  process evidence.
- The runner checks lexical venv executable/prefix, ABI/natural GIL,
  wheel-installed payload equality, native DSO hash, explicit thread settings
  and CPU0–7 affinity/topology. The two findings above are the remaining
  reviewed identity gaps.
- Queue first-error/closure/resource failures invalidate the run, and a failed
  close is not retried. The native owner confirmed that successful close
  disables Prepare admission while keeping the original FIFO usable for the
  subsequent CPU validation transfers; that confirmation is compatibility
  context, not independent native build verification.
- `git diff --name-only -- benchmarks/host_regions test/host_regions` returned
  no tracked changes. New R24 integration-test files in that area were not
  mistaken for edits to original R22/R23 evidence.

The two identity findings should be resolved before accepting target
performance receipts. The 20 passing host tests do not establish CPython
3.14.6 std/FT startup, NPU correctness, qualified SDK lifetime behavior, actual
Prepare overlap, or cross-model performance. Target startup is externally
blocked according to the parent task; no speedup or completion claim is
supported by this review.

## Focused closure review — 2026-10-29

**P1 and P2 are closed for the revised, pinned harness source.** This is a
source/host-regression closure only; it is not target NPU/SDK qualification or
permission to treat CPU evidence as performance results. The original findings
above remain intact as the record of the previous baseline.

The independently checked new `SOURCE_PINS.json` SHA256 is
`d5128ee6dd39fb65627b714eea831b75284210ddc41a8621f544a5d721a3943e`.
All 11 file sizes/hashes match. In particular:

```text
runner.py    3838854a51171417436fad1f980b9133de9046b57b907462dd73f022fe3caeba
contracts.py 310847637259054b26976642abef9266f4ab3f70ca860c089766a2b231a34540
```

**P1 closure:** `runner.py:209–232` now checks all mapped private-CANN files,
alternate copies of approved provider basenames, and the declared AscendCL/
ACLNN provider families against the approved resolved path/hash inventory.
Unapproved actual mappings cannot disappear through a set intersection.
The same validation is called after calibration (`runner.py:311`) and at
completion (`runner.py:356`). Both observations are retained, and pair analysis
requires them to be nonempty, unique, and approved by the artifact inventory
(`contracts.py:143–149`). Additional providers may legitimately be loaded only
in on mode if the same approved artifact already allows them; exact observed
mapping-set equality is therefore not incorrectly required between modes.

I reran the original review's PurePosixPath-based fake-maps reproduction,
without loading an SDK or device. It now produces:

```text
ORIGINAL_UNAPPROVED_PROVIDER_REPRO_REJECTED unapproved actual SDK/provider mapping: ['/unreviewed/cann/lib64/libopapi.so']
```

The regression for an unapproved provider appearing only in the second mapping
observation also passes. Code inspection confirms the production execution
path makes both calls, rather than merely testing the validator twice.

**P2 closure:** `approved_artifact()` verifies the externally supplied artifact
digest, and `verify_harness()` checks its approved local manifest digest plus
the exact 11-file inventory, every file size/hash, and path confinement
(`runner.py:56–88`). `execute()` performs this before loading either helper
(`runner.py:235–239`). The result carries the full approved manifest and source
inventory (`runner.py:256–259`), and completion repeats the full verification
(`runner.py:357–362`). Pair analysis requires artifact-to-manifest binding,
manifest-to-source binding, and matching full manifest/source identities
between off/on (`contracts.py:135–142`, `contracts.py:153–155`). The externally
reviewed launcher remains the trust anchor for the initially running bootstrap
and artifact SHA; this responsibility is explicit in the source and README.

Independent execution of `verify_harness()` with the externally supplied new
manifest SHA returned `APPROVED_SOURCE_CLOSURE 11`. Regression tests verify
that prelaunch source drift is rejected before the helper loader is called,
local manifest re-signing cannot replace the artifact approval, and a change
to README outside the former three-file inventory fails the final verifier.

Independent full host-suite rerun:

```text
python -I -B -m unittest discover -s test/eager_prepare -p 'test_*.py' -v
Ran 25 tests in 3.183s
OK
```

There were 25 passing tests and no skips. This includes all five added
regressions and the updated pair-binding cases. The author's
`host-evidence-20261029-review-fixes/VERIFICATION.json` was inspected; both
referenced stdout/stderr file sizes and SHA256 digests independently match.
The independent rerun establishes the current green result; the claimed
earlier red phase was not replayed against the historical source here.

The README now explicitly defines the fresh-process diagnostic as one full
calibration followed by one traced sample, excluding claims of first-call
query or cold-cache performance (`README.md:209–212`). The earlier scope
clarification is therefore resolved as well.

Only this closure section was appended. No harness source, test, source pin,
native implementation, stage, or commit was changed by the reviewer. No
CPython 3.14.6 std/FT target startup, actual SDK mapping, NPU numerical run,
native Prepare overlap, or three-round performance campaign was executed by
this closure review; those remain separate target gates.
