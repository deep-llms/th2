# Remaining proxy arms: interim results

Latest status: **7 October, 03:33:24 Singapore**, monitor `a2b6e36`:
P6 production running at step 620/2500, checkpoint-500 saved. Started at
03:07:51 Singapore. All eight GPUs active (94–100% utilization), finite latest
loss/gradient logs, approximately 2.35 seconds/update. First three completed
results below are unchanged. Six arms follow P6 in the original order.
Status log SHA256: `487db823c7942cdb6346d424648cba822b08678254b5b90ab555d9627e60eab2`.

Verified snapshot: **7 October 2026, 03:07:16 Singapore** (6 October 19:07:16 UTC).
Read-only monitor `c642279`, job `th2-tjx3-proxy-remaining-results-20261007-a01`;
host `thiennh-p6-tjx3-worker-0`. Original production launch remains `258bde6`.

Three of ten production arms completed their requested 2,500 updates. All three
production checkpoint/backend validators passed. No stage failure so far.
`stopped` in result.json means the planned cutoff on the unchanged 28,600-step
schedule, not a crash.

| Arm | Held-out LM loss | Difference from A | Trainer runtime |
|---|---:|---:|---:|
| A, reused FA4 baseline | 3.4771733830 | — | 88.30 min |
| P7-simple | 3.4745054830 | -0.0026679000 | 102.02 min |
| P7 | 3.4769314640 | -0.0002419190 | 110.91 min |
| P4-iso, all heads | 3.4896725928 | +0.0124992098 | 95.55 min |

Lower loss is better. All rows have seed/data seed 42, FA4, 2,500 updates,
1,048,576 input tokens per update, and the same validation rows/targets.
`comparable_config` equality against A passed locally for each completed arm.
This is a token-matched, single-seed comparison, not an equal-compute comparison.

P7-simple has approximately 0.266% lower perplexity than A. The gain is small
and needs downstream/seed confirmation. P7 is nearly tied; all-head P4-iso is
worse. Proxy-disabled losses are 3.6128287513, 3.5993818208 and 3.6276750057
respectively. These show reliance within each model, not superiority over A.

P6 had passed its numerical gate and was running its separate 25-step smoke.
All eight GPUs were occupied at 98–99% utilization, about 120,766 MiB each.
Remaining production order: P6 → P5 → P7-mlp → P7-kq → P7-ems → P4 → P6-iso.
Supervisor reports running; no final completion/burn receipt yet. Automatic
burn restoration remains configured in the original supervisor.

Downloaded 15 small JSON artifacts and verified each source SHA256. Local
evidence: `artifacts/proxy-remaining-results-20261007-a01/` (Git-ignored),
including completed results/recipes, baseline, validators and checksum index.
Monitor log SHA256:
`11489017c3f7d9094b12e98d9911aff78a373d86ac574f00ffb8c22ac124c904`.
No weights downloaded; no training, environment, data or queue changes.
