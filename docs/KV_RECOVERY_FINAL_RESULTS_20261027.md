# Final KV-sharing recovery results — 27 October 2026

Stop this conversion recipe at the tested budget. None of the sharing variants
passes the predefined 5% quality screen. Restricting updates to affected Q/K/V
projections made recovery worse than updating all parameters.

## Matched comparison

All rows start from the same original dense A-512 checkpoint, use the same
134,217,728 recovery tokens in the same actual per-rank order, and train 2,048
steps on eight B200s. Teacher, 0.5 CE + 0.5 KL loss, LR 5e-5, 102-step warmup,
cosine schedule, seed and native SDPA runtime are identical. Only fixed sharing
placement and, in the last row, the trainable parameter set differ.

| Model | KV cache retained | Final BF16 PPL | Penalty vs dense |
|---|---:|---:|---:|
| Dense A, all parameters | 100% | 19.2416 | — |
| Uniform25, all parameters | 75% | 20.4610 | +6.34% |
| Upper25, all parameters | 75% | 20.6189 | +7.16% |
| Full50, all parameters | 50% | 22.6557 | +17.74% |
| Uniform25, affected Q/K/V only | 75% | 23.0779 | +19.94% |

The historical B continuation (PPL 22.3924) started from an already-shared B-512
checkpoint, so it is excluded from this matched-initialization table.

Uniform25 shares K/V in seven preceding-layer pairs: (2,3), (6,7), (10,11),
(14,15), (18,19), (22,23), (26,27), zero-based. The Q/K/V-only variant updates
producers' Q/K/V and consumers' Q; it freezes every other parameter. This trains
44,040,192 parameters (7.58% of this model), without altering its forward pass.

Q/K/V-only minus dense NLL: **+0.181800 nats/token**, paired-document 95% CI
**[0.171678, 0.195069]**. Versus all-parameter uniform25: **+0.120355**,
CI **[0.111458, 0.131936]**, or **12.79% higher PPL**. It is worse on all 256
validation documents in both comparisons. The quality gate is upper CI <=
log(1.05), not a training-success marker; it fails decisively.

Training-precision Q/K/V-only PPL at steps 0 / 512 / 1024 / 1536 / 2048:
102.1039 / 23.9668 / 23.3071 / 23.1072 / 23.0754. It learns, but the gain slows
as the fixed cosine LR reaches zero. This does not prove a universal ceiling.
Fit time including validation/saves was 398.62 seconds, versus 506.47 for its
all-parameter control. This is training time, not measured inference speed.

## Verification and conclusion

Eight-step/eight-GPU smoke and the full fit passed. Exact frozen-parameter SHA
is unchanged, selected parameters changed, dense-parent validation and checkpoint
reload differences are zero. Both controls match all-rank sample-order hashes,
131,072 unique documents, initial checkpoint hashes and training recipe. The
summary was independently recomputed locally and matches exactly. All 13
exported artifacts pass their node-provided SHA256 receipts.

The queue completed **09:36:38 SGT**, restored/verified communicating burns at
**09:37:49 SGT**, and independent health at **09:55:47 SGT** confirmed all eight
burn workers, released guard and advancing collectives: 1,250 cycles and
1,387.94 GiB logical payload per rank. No model/data/env/driver changes.

All four authorized follow-up training jobs are complete. No more training,
pruning sweep or speed-only evaluation is justified under this screen. The
best tested sharing conversion remains uniform25 with all parameters trained,
but it still misses the quality threshold. This rejects this short recovery
recipe on this model; it does not reject KV sharing generally. One seed and
256 reused 1,024-token development prefixes do not establish downstream or
long-context quality. Restricted tuning/partial sharing are known techniques,
not a novel contribution; this is not a reproduction of SwiftKV's recipe.

## Evidence

QKV run: `6y3o-efficient-kv-qkv-20261027-a01`, launch `3932ad4`, health `d1e92f3`.
[Summary](results/kv_qkv_20261027/supervised/run/summary.json),
[full result](results/kv_qkv_20261027/supervised/run/qkv/result.json),
[completion](results/kv_qkv_20261027/supervised/run/complete.json),
[burn verification](results/kv_qkv_20261027/supervised/burn-verified.json),
[SHA receipts](results/kv_qkv_20261027/artifact-receipts.json),
[archive verification](results/kv_qkv_20261027/archive-verification.json).

Exact local archive: `temp/kv_qkv_results_20261027/`; raw result log:
`temp/kv_qkv_health01.log`, SHA256
`3ffd72cea3ebd77556ac10ea603a412dd10732fe3eaa9e98abe75e7f8db51e32`.
Durable copy: `docs/results/kv_qkv_20261027/`. Large checkpoints remain on B200.
Earlier [placement](KV_PLACEMENT_RESULTS_20261027.md) and
[recovery](KV_RECOVERY_RESULTS_20261027.md) reports contain their own evidence.
