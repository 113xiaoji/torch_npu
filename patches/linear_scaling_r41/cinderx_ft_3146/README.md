# Experimental CinderX FT 3.14.6 qualification

Base CinderX commit: `6b0e282c37a8725253e068295e90eb2e321d6876`.
Apply `ft-source-selection.patch` followed by `ft-borrows-3146.patch` to a clean
copy. These patches select the actual free-threaded borrowed source and adapt
the two changed 3.14.6 instrumentation functions through the repository's
UpstreamBorrow generator. Ref-debug accounting stays under `Py_REF_DEBUG`.
No GIL, assertion or atomic-reference-count safety is disabled.

`adapt_ft_borrows.py` records the exact qualification directory layout. Its
`FileParser` adapter uses libclang with the selected FT headers; declaration
extraction/output still use the repository `TemplateFileProcessor`. It rejects
error/fatal diagnostics. Fresh regeneration was byte-identical. The manifest
pins CPython source, target headers, generator, patches, compiler and DSO.

The tested runtime is CPython 3.14.6t, natural GIL-off, AArch64, Torch2.13.0+cpu.
The build uses Clang18.1.8 after GCC13.4 reproduced a compiler ICE. Pass the exact
FT `Python_EXECUTABLE` and `Python_INCLUDE_DIR`; `_cinderx.so` has no ABI suffix,
so an import alone does not prove a correct FT build. Only this exact runtime
is qualified here. Future micro versions and other interpreter ABIs require
their own verification.

The process-local `clone3_compat` executable is inherited from the existing
runtime: it returns ENOSYS for clone3 so the libc fallback works under the
existing container policy. No container/host security policy is relaxed.

Correctness checks include GIL-off before/after imports, CPU Tensor arithmetic,
parallel original DLRM outputs/state, and empty JIT deoptimization snapshots.
Separate diagnostics observe PCs/active frames in exact per-process
`__CINDER_JIT` text ranges. Those samples prove native activity, not per-call
entry counts, full native coverage or speedup. Timed CPU workers have no
perf observer.

The first CPU A/B compiles only `DLRM_Net.forward`, a short dispatcher. Its
three-round throughput medians are:

| Runtime | 1 caller | 2 callers | 4 callers |
|---|---:|---:|---:|
| CPython FT | 532.556 | 1020.072 | 1903.863 |
| CinderX FT, forward only | 526.534 | 984.247 | 1867.708 |

These are requests/second on CPU64..71 with the original input bank/model,
batch128, Torch1/1, 3000 requests and 24 warmups/caller. All 24 receipts and
54,192 outputs passed independent raw-record audit. This narrow treatment
does not improve CPU throughput. It does not settle the value of compiling
actual user-model helper bodies. No NPU conclusion follows from this table.

CPU evidence archive SHA256:
`0b63efefa14873484e98b30ec8f087353a134fb9af97171d4dc291d0f6afce0f`.
PSS was not collected in this CPU experiment.

The separate [generic class-methods experiment](CLASS_METHODS_CPU_RESULTS.md)
also has no CPU throughput gain, despite verified native activity for every
actually called user-model method. Further optimization requires hotspot
evidence rather than a compilation-state claim.
