# R34 exact CANN 9.1.0 eager profile overlay

This overlay adds an independently qualified Eager Prepare profile. It does not
change `VerifiedSdkProfile()` for CANN 9.2.0 or
`OwnedOpApiSdkContractAvailable()` for R22. `CheckOwnedSdkProfile()` accepts the
new profile only while `NativeJobActive()` is true. A caller-side eager capture
checks its own `RealSdkContractAvailable()` capability before native Prepare
and Execute recheck the owned SDK gate. `BeginOwnedCache()` also rechecks it.

The legacy public `SdkContractAvailable()` inline, `EagerSdkQualified()`,
`EagerArgumentDomain()`, `EagerOwnedCommand`, and `EnqueueEagerArguments()` retain
their old definitions to avoid conflicting weak inline definitions in the
unmodified objects. New `EagerSdkQualifiedR34()` and
`EagerArgumentDomainR34()` names are used by the R34 dispatch macro. The native
Configure caller must use `RealSdkContractAvailable()` in a real build.

The reviewed mapped NN provider is 17,562,248 bytes, SHA256
`6644fd0f5f1e7a5cdb7e549acc1057703bc81eb269bb95890b878a7d7f7aa131`.
The reviewed mapped base provider is 4,328,144 bytes, SHA256
`308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b`.
Every required symbol must resolve into its corresponding attested provider;
an unavailable symbol or wrong mapped file size/hash closes the profile.

Evidence supplied by the A3 qualification owner:

- `sdk-qualification-v1-complete-003.json`: 192 cases, 144 numerical executions
  and 48 cancellations, four overlapping query workers, uniform lengths
  1/2/8/64/128/256. Probe source SHA256
  `d28b760602bc03da013ee7a1af082c5c14349845e746971ef08ee9280163e80d`.
- `sdk-probe-v1-ragged-meta26-004.json`: the original frozen MetaSyn26 inputs,
  25 nonempty tables, 800 cases/200 four-worker waves; fresh addresses, local
  cache misses/hits, cross-worker misses, cancellation and later cache hits.
  Probe source SHA256
  `6b37b8502bb72af1b400f70893130b903f5108eacf0cf8b2126078e5b1833e5a`.

The admitted 9.1.0 typed domain is forward-only SUM, FP32 weight, int64
indices/offsets, sparse=true, no per-sample weights, no last offset, padding=-1,
weight [1024,16], offsets [128], and exactly the reviewed positive indices
counts in the profile header. All seven tensors must be contiguous ND with
zero storage offset. Output metadata is [128,16] FP32 and [0]/[128]/[128]
int64. The original empty table stays on the original route. Other legal
shapes and v2 cache callers return false from the argument gate and use the
existing fenced fallback. No model or input is rewritten.

The overlay's common/base headers originate from the actual R32 A3 tree,
preserving its R31 immediate macro bodies and the existing original-route
behavior. Only the four final eager macro references and synchronous fallback
qualification reference change in common.h. Copying common_base.h into the
shadow directory is necessary for its quoted include to find the new eager
header. Remaining headers resolve from the original op-plugin utils directory.
Recompile EagerPrepare.cpp, this common.cpp, and the actual R32 EmbeddingBag
source; replace all three objects when relinking. Preserve its empty-SUM fix.

Host-only verification:

```text
cmd /d /c patches\eager_mpsc_r34\test\run_profile_test.cmd
python patches/eager_mpsc_r34/test/verify_sources.py
```

The cold profile test was observed failing before implementation, then passing
with `/W4 /WX`. Ragged counts were separately added as a failing positive test
while their qualification flag was false; after the A3 owner reported the
complete receipt, enabling that flag made it pass. Wrong hash, size, mapping,
missing symbols, wrong-provider symbols, R22 isolation and fixed-shape rejects
are covered. These cold checks establish policy behavior, not native NPU
integration correctness or any speedup. The runtime still retains its core
state mutex and downstream PTA/allocator/SDK synchronization.
