# Remaining proxy arms: results

## Final results — 7 October 2026, 15:55:24 Singapore

Read-only monitor `a9289bf`, job `th2-tjx3-proxy-remaining-results-20261007-a03`.
All ten arms finished at 2500 updates. All 63 stages exited successfully,
including all production checkpoint/backend validators and final comparisons.
Shared recipes match A under `comparable_config`. Queue completed at
15:37:34 Singapore; automatic burn restoration verified at 15:38:45 after
all GPUs were checked free. At 15:55:24 the same eight burn workers
308656–308663 occupied all eight GPUs at 100% utilization (~155212 MiB each).
Communicating collective progress was verified by the supervisor.

| Arm | Held-out LM loss | Difference from A | Trainer runtime |
|---|---:|---:|---:|
| A, reused FA4 baseline | 3.4771733830 | — | 88.30 min |
| P6-iso | 3.4662995700 | -0.0108738130 | 96.48 min |
| P7-simple | 3.4745054830 | -0.0026679000 | 102.02 min |
| P7 | 3.4769314640 | -0.0002419190 | 110.91 min |
| P7-mlp | 3.4773446524 | +0.0001712694 | 111.76 min |
| P7-ems | 3.4777271990 | +0.0005538160 | 111.72 min |
| P7-kq | 3.4777975959 | +0.0006242128 | 110.88 min |
| P6 | 3.4782647578 | +0.0010913748 | 99.93 min |
| P4 | 3.4822870036 | +0.0051136206 | 97.30 min |
| P5 | 3.4838803344 | +0.0067069514 | 95.85 min |
| P4-iso | 3.4896725928 | +0.0124992098 | 95.55 min |

P6-iso is the strongest single-seed validation result: loss lower by
0.0108738130 (~1.0815% lower perplexity), versus 0.0026679000 for P7-simple.
P6-iso is also better than LM-trained P6, which cautions against assuming that
LM gradients into the predictor help. These results do not establish a causal
mechanism or downstream/seed robustness; those evaluations remain pending.
All comparisons are equal-token, not equal-time. Trainer runtime includes
validation/checkpointing, excluding startup and separate acceptance gates.

Downloaded and source-SHA256-verified 47 small JSON artifacts under
`artifacts/proxy-remaining-results-20261007-a03/`, including final comparison,
complete.json, supervisor and burn receipt. queue.json and gpus.json also saved.
Source log: `temp/remaining-results-20261007-a03.log`, SHA256
`a89bda91ee38b0598779a767ee8f893922fff93033232388ec144a265076f95a`.
No new training, evaluation, process termination or queue changes performed.

## Updated snapshot — 7 October 2026, 11:19:26 Singapore

Read-only monitor `5b3e750`, job `th2-tjx3-proxy-remaining-results-20261007-a02`.
Seven of ten production arms completed 2500 steps and passed checkpoint/backend
validation. Shared `comparable_config` equals the reused A recipe for all seven.
No failed stages. P7-ems is running on eight GPUs at step 1250/2500 in the
captured log, checkpoint-1000 present, approximately 2.64 seconds/update.
P4 (all heads) and P6-iso remain queued afterward. Supervisor remains running;
final completion and burn receipts are not yet due. Queue unchanged.

| Arm | Held-out LM loss | Difference from A | Trainer runtime |
|---|---:|---:|---:|
| A, reused FA4 baseline | 3.4771733830 | — | 88.30 min |
| P7-simple | 3.4745054830 | -0.0026679000 | 102.02 min |
| P7 | 3.4769314640 | -0.0002419190 | 110.91 min |
| P4-iso | 3.4896725928 | +0.0124992098 | 95.55 min |
| P6 | 3.4782647578 | +0.0010913748 | 99.93 min |
| P5 | 3.4838803344 | +0.0067069514 | 95.85 min |
| P7-mlp | 3.4773446524 | +0.0001712694 | 111.76 min |
| P7-kq | 3.4777975959 | +0.0006242128 | 110.88 min |

P7-simple remains best by a small margin. P6/P7-mlp/P7-kq are numerically close
to A but slightly worse; P5 and all-head P4-iso are further behind. These are
single-seed, equal-token results, not statistical or downstream conclusions.
Runtime includes Trainer validation/checkpoints, excluding startup/smoke gates.

Pulled and SHA256-verified 33 JSON artifacts under
`artifacts/proxy-remaining-results-20261007-a02/`; queue.json is also retained.
Log: `temp/remaining-results-20261007-a02.log`, SHA256
`0ca9725339eed1d2d64fcd5a9f1901aa3cf6f18de995506d5cfde61df55431ef`.
No model weights pulled and no training/environment/data/queue mutations.

Earlier status: **7 October, 03:33:24 Singapore**, monitor `a2b6e36`:
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
