# Generic class-methods CPU result

This broader CinderX treatment has no throughput gain in the fixed original
DLRM experiment. Model, weights, eight-input bank, batch128, natural FT3.14.6t,
Torch2.13, CPU64..71 and Torch1/1 remain fixed. Six fresh32-request correctness
cells precede three interleaved rounds of3000requests and24warmups/caller.
Formal timing has no perf or Python profiler.

| Callers | CPython median TPS | Class-methods median TPS | Matched-round median change |
|---:|---:|---:|---:|
| 1 | 534.740 | 522.657 | -3.32% |
| 2 | 1044.932 | 986.393 | -5.51% |
| 4 | 1914.816 | 1812.359 | -4.74% |

Throughput columns are medians of three windows. Changes are medians of the
three matched-round ratios and do not equal ratios of those throughput medians.
Paired four/one scaling medians are3.58084 for CPython and3.54459 for CinderX.
PSS was not collected.

Selection generically inspects user classes/submodules through `type.__dict__`,
source-root and module ownership, excluding constructors. It contains no DLRM
method-name selector. Ten methods compile; an independent plain-CPython call
inventory identifies the five actually called by this input bank. Separate
same-configuration diagnostics for1/2/4callers observe positive exact
`__CINDER_JIT` native PCs/active frames for each hot method and each caller.
The other five methods remain compiled but unexercised. Inclusive frames are
not entry counts, full native coverage or additive CPU-time percentages.

All24receipts and54,192outputs passed raw independent audits: original inputs,
state, provider/source/SO identities, complete selected-method sets through
four stages, caller GIL/TIDs, empty deoptimization snapshots and recomputed
metrics. Maximum absolute output error is zero. Raw perf data was independently
decoded and compared with archived text before per-method coverage recomputation.

The original controller's final summary used an obsolete display key and raised
KeyError after all24workers succeeded. The original controller/log/status are
retained. A separate finalizer and auditor recomputed from unchanged receipts;
no timing was rerun or replaced to fix the summary.

Evidence archive:7,170,993bytes,164payload files. SHA256:
`51eb511e94fe7ce4a89c9943df27caf47d3b9335b09a866330e1c16bdc0d8fc1`.
The native CinderX SO is the same pinned `b0809aac...` build in the adjacent
qualification manifest. This result applies to this treatment/workload only;
it does not establish performance for all CinderX models or for an NPU.
