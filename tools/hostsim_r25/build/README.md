# Isolated R25 FT native build

`build_native_ft.py` consumes a prepared full source tree at
`<attempt>/src/torch-npu`, a stable new HostSim provider, and the pinned existing
FT package as a Python-file base. It creates fresh CMake/Ninja outputs and
recompiles every TorchNPU/profiler translation unit plus the separate `_C`
extension. Unchanged vendor static archives such as DVM remain pinned inputs.
All writes must remain inside `/home/r25-eager-prepare-hostsim-20261029`.

Use a new attempt for every invocation. Save the source-copy and overlay receipts
before running. The source must include the complete frozen R24 feature and the
separate verified R25 op-plugin overlay. This script does not initialize or alter
old repositories, install packages, run models, or measure performance.

```sh
python3 build_native_ft.py \
  --attempt /home/r25-eager-prepare-hostsim-20261029/attempt-004-semantic-none-ft \
  --provider /home/r25-eager-prepare-hostsim-20261029/sdk/provider-002/libopapi.so \
  --package-base /home/coarse-npu-pipeline-20261024-r1/variants/coarse-r20/native-command-packet-r17/packages-ft/torch_npu
```

`build-evidence/` records source and SDK hashes, actual commands, full logs,
output SHA256/ELF IDs, and runtime search paths. The copied `_C` and main library
are removed before compilation. No parent object cache is used. Build CPUs
default to 160,162,...,182, excluding reserved measurement CPUs 144..158.

`rebind_provider_ft.py` is restricted to a provider identity change after a
successful full parent build. It verifies complete Ninja dependencies, effective
compiler arguments, source/library bytes and independently copied object hashes.
Only the generated profile's exact translation-unit closure may change. It
rebuilds that closure, links the main library and rebuilds `_C`; the effective
linker RPATH is preserved without general shell evaluation. Keep both original
and corrected output receipts when repairing an audit failure.

A changed public template header requires a new full build. The semantic-None
correction uses this path; it does not mix old and new template definitions.
Fixed-ELF-offset diagnostic modules and their raw results remain under ignored
`results/` and are not production or benchmark components.
