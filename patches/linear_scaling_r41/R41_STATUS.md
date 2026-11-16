# R41 verification snapshot

This is an opt-in experiment. Linear NPU scaling has not been demonstrated.

The native A3 library was rebuilt from 1122 objects. Its SHA256 is
`43491001a25d7637e42ed74393a765f984907bfc53fd1812372e8807dfa16b95`.
Two explicit compilation repairs redirected stale dependency output paths and
included `HostRegionCapture.h`. All final compile/link return codes are zero;
the frozen D894 baseline hash is unchanged.

Real-NPU correctness passed for five arms (rebuilt off, Event notification,
workspace cache, both, frozen D894) at 1/2/4 callers: 15 cells, 480 outputs.
MLP, CNN and Attention also passed in all four new-library arms: 768 outputs.
The original DLRM architecture/weights/input bank and natural FT mode remain
pinned. Every output is compared, model state is checked, and native close is
acknowledged. Enabled workspace arms require actual hits from each caller.

The first complete worker observed maximum absolute error
`8.940696716308594e-08` against the original CPU reference. CPU mechanism tests
cover Event wakeups/counting, cache ownership/bounds, private allocator pools,
unsafe/recovery admission and effective native option receipts.

The complete three-round NPU performance comparison and independent final audit
passed: 45 fresh windows, 135000 fresh outputs plus 1248 sealed correctness
outputs. The best four-caller treatment improves matched-round throughput by
8.41% (median) over D894 and 6.19% over rebuilt off. Its paired four/one scaling
mean is 2.75, below the 3.6 milestone. See [R41_RESULTS.md](R41_RESULTS.md) for
phase/latency values and the archive SHA. Earlier partial timings are excluded
from final performance claims. Host observation
tooling needed fixes for Docker18 CLI support, nested mounts under read-only
parents, and /proc identities during process teardown. An owned CPU child
reproduced cgroup reparenting during exit. Samplers preserve incomplete samples;
they reject unknown/foreign ownership and any uncertainty overlapping a timed
ROI. A final audit must use the stopped/joined observer's entire log, including
late-published samples. Sampled monitoring does not prove uninterrupted device
exclusivity.

Correctness can be reused only through a sealed proof binding unchanged
native/worker/model/runtime/SDK/image/input files. This reuses 1,248 validated
outputs, and zero prior performance cells. New performance workers still run
all provider/GIL/getter/numerical/state/close gates. The phase/scaling report
must distinguish paired-round ratios from ratios of throughput medians.

The CinderX FT build patches and CPU forward-only result are in
[cinderx_ft_3146](cinderx_ft_3146/README.md). Both that narrow treatment and the
separate [generic user-class method experiment](cinderx_ft_3146/CLASS_METHODS_CPU_RESULTS.md)
have no positive CPU throughput gain. Native hotspot diagnosis is needed before
further runtime changes; these CPU results are not NPU performance evidence.
