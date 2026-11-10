# High-L continuation with a fresh L32 control

This campaign uses the same original DLRM, FP32 weights, dense input, B128,
D16,26tables,V1024 and sum pooling as R28. It adds actual bag lengths64,128,
256 plus a contemporaneous32 control. Native candidate, numerical SDK and
Python runtime are reused without rebuilding or substituting their versions.

`workload.py` imports byte-pinned R28 scaffolding. It retains all32 original
R28 IDs per bag. A fixed generator(seed20260920) draws128x224 extra IDs for
each table; larger lengths append the first L-32 columns. Repeated IDs are
not removed. Thus all lengths share nested prefixes, and L32 remains byte
identical to the earlier input. Model source/forward/state/dense hashes stay
fixed; every length has its own predeclared complete input hash.

The controller retains R28's validated process/timing/count logic. Each length
runs12rounds, each arm50warmup plus3000timed forwards, in serial fresh processes.
Four L values rotate through Latin positions; all six arm orders occur twice
per L. The12cold processes must first pass model/input/stage/count/stream/runtime
gates. The144performance processes then provide432000timed samples.

```sh
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_COLD --kind cold
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_PERF --kind perf --qualification NEW_COLD/complete.json
```

The measured boundary remains original model forward through synchronization.
The26bag outputs,Interaction and final output are also compared outside timing,
so a saturated sigmoid cannot hide incorrect intermediate results. Native/SDK
call counts and actual13/13 Embedding stream routing are recomputed separately
for warmup,timed samples,and the additional untimed staged pass.

Report each L independently against its same-round off result. The fresh32
control and first profitable sampled length can bracket an observed crossing;
four sampled points do not locate an exact numerical threshold or guarantee
monotonic behavior. All timings are CPU numerical host-simulation, not NPU
device performance qualification. Runtime and results belong under the user's
204 `/home/zyz/` task namespace.
