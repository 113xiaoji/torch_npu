# Native CPU bridge verification — 2026-10-29

Host: Windows x64, CPython 3.13.12, MSVC 19.44.35225.0. The extension is compiled
as C++17 with `/O2 /W4 /WX`; it has no Torch, CANN, or SDK linkage. The independent
model driver uses installed Torch 2.7.1 CPU. No dependencies were installed.

## Test-first evidence

Before the package/extension existed, the actual command
`python -m unittest discover -s test/eager_prepare_cpu_native -v` exited 1:
`test_bridge_available ... FAIL`, `AssertionError: unexpectedly None : CPU native
bridge is missing`; the other nine checks explicitly skipped for the absent
bridge. This observed missing-bridge RED is preserved here as a transcript
excerpt; the original tool output also contains it.

After the initial native implementation, all ten checks passed. Subsequent
actual-counter coverage first failed because `thread_cycles` was absent
(`tools/eager_prepare_cpu/.cache/cycles_red.log`), then passed with the genuine
QueryThreadCycleTime counter. The stale-DLL regression first failed because a
cached module was returned despite a changed source identity
(`.cache/identity_red.log`), then passed after adding the compile-time/current
core SHA comparison. The identity test changes only a temporary fake header
path; it never edits the production header.

## Current result

```text
python -m unittest discover -s test/eager_prepare_cpu_native -v
Ran 13 tests ... OK
```

Fresh output: `tools/eager_prepare_cpu/.cache/bridge_green.log`. The final command
rebuilt the extension because the source fingerprint changed; its cache folder
contains `compile.log`, `build_manifest.json`, and generated `build_info.h`.

The preceding twelve-test version was also executed in **30 independent Python
processes**, each with a 20-second subprocess deadline: 30/30 passed. These runs
include real Python producer/consumer threads, native workers, and no arbitrary
sleeps. The later thirteenth check covers only loader identity rejection.

Coverage: compiled production-core identity, serial/parallel checksum equality
at 0/100/1000 iterations, gated reversed Prepare completion with ordered consume,
K=1 backpressure until retirement, no next crossing an unfinished CPU operation,
close waking an empty consumer, original CPU error waking a blocked producer,
injected Prepare failure, abort releasing a gated worker and consumer, persistent
sequence validation, bounded traces, hot-mode trace absence, actual OS worker
TID/raw cycles, and stale-binary rejection. All shutdown/abort checks finish with
zero live workers and zero successfully acknowledged CPU ops for failure cases.

## Separate heavy cold CPU evidence

`tools/eager_prepare_cpu/.cache/heavy_thread_cpu.json` contains a separate
four-job, two-worker run at 10,000,000 common iterations. It is **excluded from
the model performance campaign**. The actual worker TIDs were 32360 and 43816;
the four GetThreadTimes deltas were 15,625,000, 15,625,000, 31,250,000, and
31,250,000 ns. Raw cycles were 44,023,372, 44,182,619, 45,646,867, and 45,896,289.
Their native wall intervals overlap. These are observed CPU counters, not
fabricated thread times from sleep or cycle/frequency conversions. Short normal
samples can report zero thread CPU ns and retain that zero honestly.

## Frozen source identities

- Production core: `704f99c5cfd8faeb8640dd52965f8e5b3d85a1a9069a26ab68d957d0b6fdb23d`
- Native bridge: `2a99c55763847ec406e21faa6f780ef43dd6010d13b43c95eca183d729a5c744`
- Builder/loader: `1d0d02cf0e3d834076bd00f7bc8fe4794a290179599c74e8cc26b40e43e092f9`

The core is included directly, not copied. The binary carries its compilation
hash; the loader compares it to the current source. Model correctness, campaign
statistics, and capture/replay validity are verified separately by the model
driver owner. None of this evidence proves NPU/SDK behavior, NPU performance,
real device retirement, or Python 3.14 free-threaded speedup. Linux-specific
code is present but untested here.
