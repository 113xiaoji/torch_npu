# Selective Embedding scheduling on a MetaSyn projection

This experiment retains the original DLRM, FP32 weights, dense input, B128,
26 tables, V1024, D16, sum pooling and original forward. It compares one fresh
PTA candidate under five treatments:

| Arm | Native mode | Minimum mean bag length |
| --- | --- | ---: |
| off | disabled | 0 |
| e | Embedding streams | 0, all qualified calls |
| ei | Embedding and Interaction streams | 0, all qualified calls |
| ae | Embedding streams | 64 |
| aei | Embedding and Interaction streams | 64 |

The selective policy is opt-in. Zero keeps the existing E/EI routing. For a
ragged EmbeddingBag invocation, the predicate uses `indices.numel()/bags`, with
`bags=offsets.numel()-include_last_offset`. It reads tensor metadata and does
not copy device offsets or synchronize to inspect their values. A table can
contain individual long bags while its mean stays below the threshold.

Independent small calls run the original caller-stream body without per-small
output-owner/event registration. Aliases or uncertain allocation metadata
conservatively fence before the original body. Fresh caller-frontier events
protect later side-stream reads; event handles are not rerecorded while live
waits can refer to them. Explicit close also fences small-only caller work.
The current admitted profile remains one inference caller, device 0/default
caller stream, native caching allocation, FP32 unweighted SUM and queue mode 0.

## Data

`meta26_inputs.json` is a fixed projection of the public MetaSyn 2021 base trace
from [facebookresearch/dlrm_datasets](https://github.com/facebookresearch/dlrm_datasets)
at revision `6f46ba7906e47a2f6bab6499056480d07d784eb8`. Source gzip identity:

```text
bytes  2398183998
SHA256 1d4d3ca295fade795cbf973d9b29e721b7c76a2130684582d9f4ab579c95e772
```

`project_meta.py` sorts all 856 tables by full-batch mean length and source
index, selects 26 midpoint quantiles, then permutes them with seed 20261110.
Each selected table contributes its original first 128 bags. Source IDs stored
as float32 must round-trip losslessly through int64 before modulo 1024 mapping.
Original ragged lengths, empty bags and the wholly empty table are preserved.
Selection and threshold are fixed before performance measurement.

The projection has 50850 IDs per batch, mean L15.2794471154. Model tables 2 and 9
have mean L102.2265625 and L76.8125; threshold64 routes those two to producers
and the remaining24 to the caller. This projects length/access inputs into the
small original model; it does not reproduce Meta's full table/model scale or
private production traffic. The derived fixture retains the upstream Meta
copyright and permission notice in [META_LICENSE](META_LICENSE).

```text
projection SHA256 5eb2663c2d00d64db83c9d5b858c78f80849841216b242c93cfcd72d2dfbade5
input tree SHA256 c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1
```

Regenerate from a verified source gzip with the same Torch CPU runtime:

```sh
python project_meta.py --source-gzip fbgemm_t856_bs65536.pt.gz \
  --output NEW_DIRECTORY/meta26_inputs.json
```

## Measurement

Five cold processes must pass first. Then50 serialized fresh processes run10
rounds, each with50 warmup and3000 timed forwards:150000 timed samples. Five-arm
Williams sequences and their reverses balance every position and directed
neighbor pair twice. Compare each candidate to its same-round off result.

```sh
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_COLD --kind cold
python controller.py --base-config BASE.json --runner-root RUNNER \
  --output NEW_PERF --kind perf --qualification NEW_COLD/complete.json
```

The worker verifies actual caller/producer/consumer SDK counts, native policy
and lifecycle counters, all26 bag outputs, Interaction, final output, original
model identities, loaded libraries, affinity and natural free threading. Extra
numerical checks and startup/input preparation remain outside forward-to-sync
timing. This is CPU numerical host-simulation; real NPU qualification is pending.

**Current checkpoint:** implementation and native protocol/lifetime validation
passed, but selective host-sim qualification and performance are not complete.
The first cold off/E receipts passed before SSH to204 disconnected. Performance
has not been launched. Inspect remote state before restarting any controller.
All remote artifacts belong to `/home/zyz/r31-selective-meta-20261110`.
