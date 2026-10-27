# Host region lifetime tests

Install these three test files in `test/host_regions/` in the TorchNPU
repository. They load the repository's eight `torch_npu/_host_regions` modules
under private package names, without importing the real `torch_npu` package.
Only the Python standard library is needed; these tests do not run an SDK,
NPU workload, or performance measurement.

From the repository root, run:

```sh
python3.14 -I -S -B -m unittest discover -s test/host_regions -p 'test_*r22.py' -q
```

To test a different checkout while keeping these test files elsewhere, set
`HOST_REGIONS_REPO_ROOT` to that checkout's root:

```sh
HOST_REGIONS_REPO_ROOT=/path/to/checkout python3.14 -I -S -B -m unittest discover -s /path/to/tests/host_regions -p 'test_*r22.py' -q
```

PowerShell accepts the same override:

```powershell
$env:HOST_REGIONS_REPO_ROOT = (Resolve-Path '/path/to/checkout').Path
python -I -S -B -m unittest discover -s /path/to/tests/host_regions -p 'test_*r22.py' -q
```

The three runtime/provider files contain 42 tests for submit failures, ownership quarantine,
provider contracts, cold close, and interpreter shutdown ordering. Exit tests
create temporary child processes; intentionally unacknowledged fake retirement
must stop and be terminated by the test parent. These fixtures exercise real
Python workers with fake native boundaries, and provide no device coverage.
Set `R22_CLOSE_TEST_ARTIFACTS` to an existing directory to retain their child logs
in a chosen location.

The R22 device measurements found four-worker throughput below the serial
baseline: 0.479 times for std and 0.570 times for free-threaded Python. The two
submit-failure accepted-before-Execute windows remained unproven. Passing this
host suite does not change those results or establish overall acceptance.

The separate `test_apply_op_plugin_overlay.py` suite adds nine temporary-Git
tests for the complete source overlay, dirty/staged input protection, exact
base matching and idempotency. Run it with:

```sh
python -I -S -B -m unittest discover -s test/host_regions -p 'test_apply_op_plugin_overlay.py' -q
```
