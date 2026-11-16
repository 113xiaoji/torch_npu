# CodeExtra::calls ordering audit, before implementation/testing

The proposed change preserves every atomic increment and exact modulo-uint64 count. It changes only the FT RMW ordering from seq_cst to relaxed; non-FT code, counter frequency, cold-policy gates, object ownership, frames and published state remain unchanged.

Source audited: isolated `/home/zyz/r41-cinderx-ft-20261116/src`, extracted HEAD6b0e282c plus existing FT port. Baseline native SHA b0809aac744059cfcf7069fae65ec2f1c6041231a81f1d4094b94384017d9315 remains immutable.

## All production increment/read consumers

- `Common/code_extra.h:60-73`: FT increment uses `_Py_atomic_add_uint64` (seq_cst); **all FT count readers already use `_Py_atomic_load_uint64_relaxed`**. No acquire/SC reader exists for `calls`. Target CPython3.14.6t atomic header exports `_Py_atomic_add_uint64`, but no relaxed uint64-add API. The same CodeExtra header already uses GNU `__atomic_*` builtins for other fields.
- `Interpreter/3.14/Includes/ceval_macros.h:446-463`: `CI_UPDATE_CALL_COUNT` obtains CodeExtra, reads separate `skey_word` with acquire, skips the preexisting decided-cold case, increments once, then calls `is_adaptive_enabled`. Generated entry sites and cinder-bytecodes invoke this macro. The proposed change does not alter those sites or the cold gate.
- `Interpreter/3.14/interpreter.c:70-72`: count is compared to the adaptive threshold only when delay-adaptive is enabled; used to choose interpreter specialization timing.
- `Jit/pyjit.cpp:338-364`: `countCalls`/`readAutoJitGateState` read the relaxed count; `recordAutoJitInterpretedCall` preserves its existing auto-classify/cold gates and increments through the same helper.
- `Jit/pyjit.cpp:433-456`: ROI recompile gate compares count to separately published `roi_recompile_floor`. The count supplies a quantity; it does not publish code or frame state.
- `Jit/pyjit.cpp:572-630`: AutoJIT compares the count snapshot to global/classified thresholds; compile/publication use the existing guarded paths after eligibility checks. No other data is read on the strength of acquiring the count.
- `Jit/pyjit.cpp:4139-4147`: ROI backoff snapshots the count, calculates the next recompile floor, then release-publishes `roi_recompile_floor` and `roi_ctl`. Those separate release stores remain unchanged.
- `Jit/pyjit.cpp:2418-2426`: public `count_interpreted_calls` returns the numerical count. Exact updates and joined final totals are preserved.
- `RuntimeTests/backend_test.cpp:306-327`: `CallCountTest` expects an exact30, supporting retention of exact counting.
- Other supported-source branches: 3.15 uses the same helper; 3.12 has legacy direct GIL-protected accesses. This candidate is qualified only for the existing3.14.6t artifact, and the non-FT branch is untouched.

## Lifetime and independent state publication

- `Common/code.cpp:164-181`: CodeExtra slot requests CPython-managed `PyMem_Free`; finalization invalidates the module slot index. `codeExtra`/`codeExtraIfExists` return null when module/index is unavailable.
- `Common/code.cpp:184-227,229-252`: a code-object critical section serializes get/create and slot installation; `PyMem_Calloc` initializes all fields to zero before `PyUnstable_Code_SetExtra`. The live function/frame owns its code object. Counter ordering is not used to establish this lifetime. No active free-list operation using the union's `next` was found in this source.
- `Jit/context.cpp:509-542`: compiled pointer publication uses separate `_Py_atomic_store_ptr_release`, matching acquire under the JIT entrypoint guard; cache is cleared before `CompiledFunction` free. Counter is not consulted.
- `Jit/pyjit.cpp:4458 onward`: cached compiled entry reads `jit_compiled` with acquire, checks exact globals/builtins and instrumentation. Counter is not a readiness marker.
- `Jit/behavior_classifier.cpp:1293-1314`: classification uses `skey_word` acquire/release helpers. ROI state has its own release/acquire/relaxed fields and existing entrypoint synchronization. No calls-based state publication found.

Conclusion: no count-ordering dependency was found beyond atomicity, per-location modification order and arithmetic. Relaxed RMW preserves those properties. This is a consumer audit, not a proof of performance benefit or all CinderX thread safety.

## Actual sampled PC chain

FP native CPU-clock997Hz, original bank8, c1/c4, own process and ROI only:

- Ci_EvalFrame ELF PC `0x37e7d4`: c1=15/c4=31 tip samples. Disassembly: `0x37e7d0 bl __aarch64_ldadd8_acq_rel`, then `0x37e7d4 bl Ci_GetDelayAdaptiveCode`. Neighbor `0x37e7c8` has7/15 samples. Readelf decoded-line maps the latter to atomic-add header and the former to `Interpreter/3.14/interpreter.c:71`. This is the call-count RMW/adaptive-policy block.
- Another hot cluster `0x37c524/52c/5c0` is object local/shared refcount acquisition and CAS (`__aarch64_cas8_acq_rel`); it is **not** this counter and will not be changed.
- This block's limited samples cannot explain the entire observed3%-5% regression. Unknown tip symbols are8%-9% in these profiles, and ~47%-50% tips are in resolved Torch/ATen native code. The candidate tests one small contributor.

The pre-measurement checklist above was subsequently completed: independent review; actual-header concurrency, exact-total, monotonic and threshold tests; an isolated dependent-TU build; existing AutoJIT/count regressions; fresh native qualification; and an uninstrumented paired A/B. See [RESULTS.md](RESULTS.md) for outcomes and limitations.
