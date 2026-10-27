# R23 stage profiling tools

This directory preserves sixteen reviewed R23 tools/tests as exact source
bytes, with two supplemental offline-summary files distributed separately.
They observe the pinned R22 runner without modifying the production
implementation, EmbeddingBag source, SDK, or frozen model inputs.

At publication, the main 42-case diagnostic cohort has been measured. The new
eight-process caller-only supplement and four native-profiler cases have not
run. No new device success, native phase timing, or performance improvement is
claimed here. R22 four-worker throughput regressed to 0.479x (std) and 0.570x
(free-threaded) of the respective serial baselines.

## Host tests

The sixteen-file bundle was verified with Python 3.13.12 on Windows: 63 tests,
61 passed, zero failures/errors, and two explicit Linux process-test skips.
The target Linux host must pass all 63 with zero skips before device execution.
The two additional offline-summary checks bring full repository discovery to
65 tests (63 passed, two Linux skips). Tests do not execute an SDK or NPU.

From the repository root:

```sh
R23_FROZEN_RUNNER="$PWD/benchmarks/host_regions/real_dlrm_runner_r22.py" \
R23_PROVIDER_SOURCE="$PWD/torch_npu/_host_regions/torch_npu_provider.py" \
python -I -S -B -m unittest discover -s benchmarks/host_regions/r23 -p 'test_*.py' -q
```

The external source identities remain a70a2941 for the runner and d4b9f2aa for
the provider. Test environment variables select their physical paths, without
relaxing the full SHA256 checks.

## Controlled execution and distribution

The three campaign controllers (`run_profile_campaign_r23.py`,
`run_msprof_r23.py`, and `run_caller_campaign_r23.py`) default to plan-only;
execution requires explicit `--execute` and the existing artifact, boot,
process, topology, SDK, and qualification inputs. Use new output directories.
The caller-only mode has its own sidecar gate; it is not a stages sidecar.
Native collection success does not establish validated native contents.
Profiler guards and SDK attestation must remain enabled.

The [source ZIP](R23_COMPLETE_TOOLS-20261027-02.zip) contains exactly the sixteen
flat Python files. Its SHA256 is
`c8e1121176cb39b1f20a1cf1ec7511ad804d77b6b0415a75f49528069a3d30b7`.
The separate [deployment pins](R23_COMPLETE_TOOLS-20261027-02.DEPLOY_PINS.json)
record every member's length and SHA256, plus the required target test result.
No private raw receipts, wheels, SDK sources/binaries, or credentials are
distributed. Diagnostic on/off observer cost and native process/window
correlation remain separate evidence requirements; stage values are not
corrected by subtracting an average observer overhead.

## Optional offline summary

`summarize_r23.py` replays the recorded 42 raw results and 18 stage sidecars
through pinned validators, plus optional caller-only results. It pools the
actual samples before computing quantiles and writes new JSON/Markdown outside
producer output trees. Its source and two math checks are separate from the
unchanged sixteen-file ZIP and deployment pins. A remote summary does not
establish that the full private raw archive has been downloaded locally.
