# Experimental four-worker host regions (R22)

This branch publishes the implementation used for the R22 DLRM experiment.
It is **default off** and **did not pass overall acceptance**. Four workers
performed real work in parallel, but synchronized end-to-end inference was
slower than the original serial baseline. Two submit-failure lifetime windows
remain unproven. This is an experimental implementation, not a speedup claim.

## Where the code is

- [`torch_npu/_host_regions`](../../torch_npu/_host_regions): restricted AST
  recognition, effect policy, persistent Python workers, admission, cold close,
  quarantine and interpreter-exit ownership.
- [`torch_npu/csrc/core/npu`](../../torch_npu/csrc/core/npu): capture packets,
  native bindings, tensor/storage owners, worker TLS, source-order queue
  publication and device retirement. Framework integration also changes
  `OpCommand`, `OpParamMaker`, allocator and H2D/recovery paths.
- [`patches/host_regions/op-plugin`](../../patches/host_regions/op-plugin):
  **all seven actual op-plugin source files**, including typed owned arguments,
  SDK preparation and the exact NN-provider binary profile. These files are
  deliberately visible in this repository despite op-plugin being a submodule.
- [`test/host_regions`](../../test/host_regions): portable Python lifetime and
  provider tests, loading this checkout's implementation without importing the
  real NPU extension.
- [`benchmarks/host_regions`](../../benchmarks/host_regions): the frozen
  synchronized four-arm runner, measurement controller and topology contracts.

The base is the previously published `7d45a86cd1fdf70d8177445f0c79268cc75c7516`
branch, based on TorchNPU 2.13.0 at
`46936d18a18ba219fc5ed310fe2cc6409f94114b`. This is not a rebase onto the
repository's current master. [Source pins](r22_source_pins.json) describe the
46 implementation inputs and five inherited integration inputs. Their bytes
match the tested R21 prerequisites plus the final R22 source13 overlay. The
historical assembled build tree did not carry a Git-clean provenance claim;
these pins identify its selected source bytes, not a historical Git commit.

## Apply the op-plugin implementation before building

The parent Git repository records op-plugin as a gitlink. Changing files inside
an uncommitted submodule is insufficient to distribute their contents. This
branch therefore includes their full source and a checked application tool.
The upstream submodule URL and revision remain explicit and unchanged.

```sh
git -c core.autocrlf=false submodule update --init third_party/op-plugin
python tools/host_regions/apply_op_plugin_overlay.py --apply
python tools/host_regions/apply_op_plugin_overlay.py
```

The tool requires the exact op-plugin commit
`312921458e7d396c54a1e92e90385d7d5f02dede`, checks the bundled file hashes, and
refuses to overwrite conflicting edits. Its default operation only checks the
state: JSON `state="ready"` means clean and eligible to apply, while
`state="applied"` means all seven published hashes match. Exit code zero alone
does not distinguish these states. Before building, run `--apply` and confirm
`state="applied"`. The tool does not fetch code, install packages or change SDKs.
After application, `git -C third_party/op-plugin diff` shows tracked-file
changes, and `git -C third_party/op-plugin status --short --untracked-files=all`
also shows new headers. The full seven files remain visible in this parent
repository's overlay. The
parent's `ignore=dirty` setting must not be treated as evidence of application;
the explicit hash check is required.

Initialize the other normal upstream dependencies following the project's
build instructions. No SDK sources, SDK binaries, vendor device binaries or
wheel artifacts have been newly vendored by this change.

## Build and enable deliberately

Use the appropriate same-ABI PyTorch installation and the normal TorchNPU build
instructions. Historical validation used Linux AArch64, CPython 3.14.6,
PyTorch 2.13.0+cpu, NumPy 2.5.3 and CANN 9.2.0 on a 910C A3.
Build std and free-threaded artifacts in separate source/build/runtime trees.
For the free-threaded build the existing opt-in must be explicit:

```sh
TORCH_NPU_EXPERIMENTAL_FREE_THREADED=1 python3.14t setup.py bdist_wheel
```

The build switch defaults off. `FTLAB_HOST_REGIONS` also defaults to `0`.
The region backend is restricted to the certified read-only EmbeddingBag sum
body; it does not parallelize arbitrary Python or training code. Runtime
capture requires explicit real-packet/real-OpAPI/native-builder/owner-retirement
options and a matching native capability. The included device runner sets and
validates these options for its declared arm; enabling a flag alone is not a
device safety certificate. The experimental lock-free queue must remain off.

The SDK profile is intentionally narrow: the EmbeddingBag provider is
`libopapi_nn.so`, with its matching nnopbase library, exact symbols, lengths and
SHA256 identities. `kOwnedSdkLifecycleValidated=true` does not bypass the runtime
binary attestation. Another SDK must be separately qualified; do not edit the
profile or remove guards merely to make an unsupported library pass.

The retained-byte budget is 512 MiB, with independent count/overflow checks.
Normal close retires native work before worker TLS can disappear. If cleanup
cannot be acknowledged, the runtime refuses new work and close, retains strong
owners and workers, and fail-stops at the pre-thread interpreter-exit hook.
Such a process requires an externally owned termination protocol; it must not
be interpreted as a successful graceful close.

## Validation and observed results

Portable host tests, without an SDK or device:

```sh
python -I -S -B -m unittest discover -s test/host_regions -p 'test_*r22.py' -q
```

The overlay safety tests exercise temporary Git repositories and require Git:

```sh
python -I -S -B -m unittest discover -s test/host_regions -p 'test_apply_op_plugin_overlay.py' -q
```

The three published runtime/provider suites passed 42/42 tests after relocation
to the actual checkout (Python 3.13, Windows). This does not replace the
historical native compilation and CPython 3.14 std/FT device checks. Production
source bytes are unchanged by publication; the test import paths were adapted.

The historical device protocol used CPU0-7 as a shared total budget: four cold
gates, five fresh 2000-sample runs per arm after 20 warmups, and four separate
20000-sample stability processes. Every timed forward includes the original
device synchronization. P99 below is the median of each run's nearest-rank P99.

| Arm | Median batches/s | Median P99 ms |
|---|---:|---:|
| std serial | 627.033267 | 1.659447 |
| std four-worker | 300.633211 | 3.665345 |
| free-threaded serial | 565.629367 | 1.843593 |
| free-threaded four-worker | 322.449409 | 3.402667 |

Four-worker throughput was 0.479x and 0.570x the respective arm medians. Every
same-round comparison also regressed. Cold and warmup probes established real
four-worker CPU overlap; timed whole-process CPU usage cannot attribute costs
to individual workers. No bottleneck attribution experiment was performed.

Full model numeric checks, the four region-return close/ordinary-exit cases,
and the defined stability contract passed. Region-return observations had
10/14/19/16 accepted commands not yet entering SDK Execute, followed by native
zero closure, four stopped workers and all 53248 region output values checked.
At the fixed second-submit fault, both corrected-prefix cases had already
entered Execute for all 26 accepted commands. The unique Python submit error,
quarantine, real worker preservation, pre-thread fail-stop and external group
reaping were observed, but **both accepted-before-Execute fault windows stayed
unproven**. Prepared-but-unpublished commands do not satisfy that requirement.

Stability is a bounded exposure result. Framework allocated-memory median
growth was zero; FT four-worker reserved-memory median grew 18 MiB, which the
frozen contract did not reject. Four-worker reserved ranges were 400-472 MiB
(std) and 364-472 MiB (FT), versus 22 MiB serial. This excludes untracked SDK
private pools and is not a whole-device peak or universal leak claim.

The full private raw receipts remain outside Git. Public derived results and
their original archive hashes are in [r22_results.json](r22_results.json).
The included measurement controller is a historical artifact-bound protocol.
It requires the original R21 serial wheels with SHA256
`7375e9891e587e3b3d507d11f42d56cc8df69b1ffc845980a79192e844894ff6`
(std) and
`575654b8d93f3a8be8860bb79bee8a11aad8b94e8c59c0a861418ac36bd31959`
(FT), plus their matching build receipts. Those external artifacts are not
distributed here. Rebuilding a serial baseline produces a new artifact and
does not satisfy that gate automatically. A general comparison against a new
baseline needs a separately declared protocol and new pins; do not remove the
historical checks and label it the same experiment. Candidate artifacts also
need their own explicit paths, native hashes and same-ABI build receipts.
The standalone runner accepts explicit artifact arguments, but a single run
does not recreate the original four-arm cohort. The runner's DLRM commit and
nine-file hashes are declared in its source; the inference loader does not
certify the upstream training CLI.
