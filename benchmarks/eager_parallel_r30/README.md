# Mixed bag lengths on the original DLRM

Only the sparse input lengths change. The original model/forward, FP32 weights,
dense input, B128, D16, V1024, 26 tables, native package, numerical SDK and
qualified Python/Torch runtime are retained from R29.

The per-table lengths are taken from the public MLPerf DLRMv2 configuration:

```text
3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
```

[Configuration source](https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/b200_preview_merlin_hugectr/train.py#L55-L82).
This reuses its length vector on the existing original DLRM. It does not run
the full DLRMv2 model or its production traffic/dataset. Lengths are fixed per
table across samples. IDs are deterministic per-sample prefixes of the frozen
R29 L256 input; repeated IDs remain, with no new random draws. A length-1 bag
retains the first of the original two IDs. The complete input SHA256 is
`40d6a16803f328885b66357bbcfdb680eb5578b15a641a61421069e4755be511`.

Each sample has 214 IDs across all tables. Existing round-robin stream routing
is unchanged: 13 tables per producer, with 138 and 76 IDs per sample. Equal
operator counts do not imply equal work. No load-balancing change is included.

Three cold processes must pass the original-model, runtime, stage, call-count,
stream and closure gates. Then 36 serialized fresh processes run 12 rounds of
off/E/EI, with all six arm permutations twice. Each process performs 50 warmup
and 3000 timed forwards, for 108000 timed samples. Full forward-through-sync
timing, numerical checks outside timing, minimum wall windows and all existing
R29 qualification requirements remain in place.

```sh
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_COLD --kind cold
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_PERF --kind perf --qualification NEW_COLD/complete.json
```

Artifacts belong under `/home/zyz/r30-mixed-l-20261110` on 192.168.41.204.
Report paired process median/p95 changes, CPU cost and correctness, including
negative results. This is CPU numerical host-simulation, not NPU qualification.
