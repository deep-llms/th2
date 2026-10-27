# KV-sharing placement results — 27 October 2026

Less sharing reduces the quality loss, but none passes the predefined 5% PPL
screen. Protecting early layers did not outperform evenly spaced sharing here.

| Model | KV cache retained | Final BF16 PPL | Penalty vs dense | Paired NLL difference, 95% CI |
|---|---:|---:|---:|---|
| Dense A | 100% | 19.2416 | — | — |
| uniform25: share layers 3,7,...,27 | 75% | 20.4610 | +6.34% | +0.061445 [0.058500, 0.064551] |
| upper25: share layers 15,17,...,27 | 75% | 20.6189 | +7.16% | +0.069134 [0.066089, 0.072258] |
| full50: share every odd layer | 50% | 22.6557 | +17.74% | +0.163337 [0.156541, 0.171005] |

Indices are zero-based. Each consumer uses the preceding producer's K/V; all
other operations and the SDPA runtime are unchanged. No MLP pruning.
All three start from the same dense A-512 weights and train 2,048 steps on the
same 134.22M tokens, seed/order, teacher, loss and optimizer schedule as the
completed dense A recovery, which is reused as the matched control.
The earlier B continuation (PPL 22.3924) has a different initialization and is
not a matched comparison for this table.

Every placement is worse than dense on all 256 reused validation documents.
Upper-minus-uniform NLL is +0.007689 [0.005495, 0.009953]. The gate is upper
paired-document CI <= log(1.05), not merely a mean PPL within 5%.
These are one-seed development results, not downstream or independent-test claims.

Training-precision PPL curves (final inference precision differs slightly):

| Steps | upper25 | uniform25 | full50 |
|---:|---:|---:|---:|
| 0 | 230.7010 | 102.1039 | 227587.2700 |
| 512 | 22.0773 | 21.9118 | 25.3555 |
| 1024 | 21.2232 | 21.0222 | 23.5655 |
| 1536 | 20.7325 | 20.5684 | 22.8172 |
| 2048 | 20.6155 | 20.4587 | 22.6544 |

Learning slows as LR approaches zero. This does not prove an optimization ceiling.
A fourth broad placement/pruning sweep is not justified. The one remaining
training job tests restricted Q/K/V adaptation on uniform25; see
[the controlled follow-up](KV_QKV_RECOVERY_20261027.md).

All checks pass: unchanged dense parent losses, exact per-rank sample-order
match and one-pass coverage, frozen teacher, zero reload difference, numerical
GPU gates and eight-GPU smoke. Fit times (including validation/saves) were
506.48 / 506.47 / 491.20 seconds for upper25 / uniform25 / full50.
Cache fractions are structural, not measured serving speedups.

Queue completed **08:29:36 SGT**, burns verified **08:30:47 SGT**. Independent
health at **09:11:42 SGT** confirms eight known workers, guard released and
advancing collectives (2,840 cycles, 3,153.40 GiB payload/rank). Recheck identities
before stopping: workers 618321–618328 under launcher 618252.

Run `6y3o-efficient-kv-placement-20261027-a01`; launch bca23aa, final health 60dcf16.
[Summary](results/kv_placement_20261027/supervised/run/summary.json),
[completion](results/kv_placement_20261027/supervised/run/complete.json),
[SHA receipts](results/kv_placement_20261027/artifact-receipts.json).
Raw log: `temp/kv_placement_final01.log` (SHA256
1906ef653d0a038811026afbaeea5e9539b4e2dcdc9563a51d8723507283fad3).
All exported bytes are verified; large checkpoints remain on B200.
