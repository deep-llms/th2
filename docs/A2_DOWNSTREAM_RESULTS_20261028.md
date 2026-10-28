# A2 downstream results — 28 October 2026

Completed at **22:53:07 SGT** in **49 min 48 sec**. All 14 checkpoints completed all nine tasks. Burns restored at 22:54:13 and independently verified at 22:56:41; no GPU job remains queued.

Fixed zero-shot screen, seed 42, context limit 4096, identical 78,259 likelihood requests per checkpoint. Original parents have 24 layers; S1 removes six layers and heals using pure KD for 0.2B or 0.8B tokens. P300/P1T denote original 300B/1T-token parents. FP32 parameters, BF16 autocast, native SDPA, batch 2. No benchmark tuning.

Accuracy below is percent. The MC mean is the unweighted average of eight multiple-choice tasks; LAMBADA is separate. HellaSwag, ARC, PIQA and OpenBookQA use acc_norm; BoolQ, WinoGrande and SciQ use acc. These are fixed screen metrics, not a universal leaderboard average.

## Summary

| Dataset / parent | Original MC mean | S1 0.2B | Change (pp) | S1 0.8B |
|---|---:|---:|---:|---:|
| dclm / P300 | 62.406 | 57.049 | -5.357 | — |
| dclm / P1T | 64.447 | 58.812 | -5.635 | — |
| fwe / P300 | 63.767 | 57.519 | -6.248 | — |
| fwe / P1T | 64.620 | 58.797 | -5.823 | — |
| nem / P300 | 65.877 | 60.179 | -5.698 | 61.388 |
| nem / P1T | 68.042 | 61.795 | -6.247 | 63.208 |
| **Mean, six parents** | **64.860** | **59.025** | **-5.835** | — |

P1T-minus-P300 accuracy damage at 0.2B is +0.277 pp (DCLM), −0.425 pp (FWE), and +0.549 pp (NEM); mean +0.134 pp. Thus this downstream screen does **not** consistently replicate the positive NLL damage contrast across all three datasets. P1T retains better absolute average accuracy in all three. No significance claim: one checkpoint pair per dataset, shared evaluation items, and no downstream confidence intervals calculated.

NEM benefits from 0.8B healing: MC mean rises 60.179→61.388% (P300) and 61.795→63.208% (P1T), still below their original 65.877%/68.042%. This evaluates S1 only; no downstream conclusion about S2 or inference speed.

## DCLM: all tasks

| Task | P300-original | P300-S1-0.2B | P1T-original | P1T-S1-0.2B |
|---|---:|---:|---:|---:|
| hellaswag | 65.057 | 54.202 | 68.592 | 56.941 |
| arc_easy | 67.003 | 56.734 | 68.434 | 56.566 |
| arc_challenge | 37.116 | 30.717 | 40.358 | 32.338 |
| piqa | 75.680 | 71.654 | 76.551 | 72.688 |
| winogrande | 62.036 | 58.642 | 63.536 | 63.141 |
| boolq | 62.355 | 63.639 | 65.902 | 64.924 |
| openbookqa | 39.200 | 33.400 | 39.200 | 34.000 |
| sciq | 90.800 | 87.400 | 93.000 | 89.900 |
| lambada_openai | 65.476 | 53.542 | 67.650 | 56.647 |
| **MC mean (8 tasks)** | **62.406** | **57.049** | **64.447** | **58.812** |
| LAMBADA PPL | 4.847 | 8.807 | 4.378 | 7.730 |

## FWE: all tasks

| Task | P300-original | P300-S1-0.2B | P1T-original | P1T-S1-0.2B |
|---|---:|---:|---:|---:|
| hellaswag | 61.422 | 52.051 | 64.469 | 54.182 |
| arc_easy | 72.475 | 61.195 | 70.960 | 58.754 |
| arc_challenge | 42.662 | 34.044 | 42.918 | 34.727 |
| piqa | 75.136 | 71.055 | 75.245 | 70.947 |
| winogrande | 61.089 | 56.590 | 63.852 | 60.458 |
| boolq | 62.355 | 62.416 | 63.914 | 64.006 |
| openbookqa | 41.800 | 35.200 | 42.400 | 37.200 |
| sciq | 93.200 | 87.600 | 93.200 | 90.100 |
| lambada_openai | 51.776 | 43.528 | 54.318 | 46.361 |
| **MC mean (8 tasks)** | **63.767** | **57.519** | **64.620** | **58.797** |
| LAMBADA PPL | 10.253 | 17.771 | 9.026 | 15.082 |

## NEM: all tasks

| Task | P300-original | P300-S1-0.2B | P300-S1-0.8B | P1T-original | P1T-S1-0.2B | P1T-S1-0.8B |
|---|---:|---:|---:|---:|---:|---:|
| hellaswag | 67.019 | 57.777 | 60.038 | 70.135 | 59.819 | 62.647 |
| arc_easy | 72.811 | 63.300 | 65.783 | 75.463 | 65.446 | 68.392 |
| arc_challenge | 45.051 | 36.092 | 37.799 | 47.867 | 37.372 | 39.676 |
| piqa | 76.333 | 72.198 | 73.232 | 77.312 | 74.048 | 74.701 |
| winogrande | 62.747 | 61.721 | 61.405 | 64.878 | 62.668 | 63.457 |
| boolq | 67.156 | 63.945 | 64.251 | 72.080 | 67.706 | 68.287 |
| openbookqa | 43.000 | 36.000 | 37.800 | 42.800 | 37.200 | 36.800 |
| sciq | 92.900 | 90.400 | 90.800 | 93.800 | 90.100 | 91.700 |
| lambada_openai | 55.599 | 49.544 | 51.174 | 59.558 | 52.009 | 54.201 |
| **MC mean (8 tasks)** | **65.877** | **60.179** | **61.388** | **68.042** | **61.795** | **63.208** |
| LAMBADA PPL | 8.123 | 13.115 | 11.173 | 6.608 | 10.500 | 9.198 |

## Verification and local evidence

- Both representative GPU gates passed at batch 2: maximum manual CE error 1.327e-7 nats/token; batch error 0 for original and 0.002837 for pruned; zero greedy mismatches. Fixed thresholds unchanged. The failed batch-8 attempt is preserved; it produced no benchmark scores.
- All 14 checkpoint identities, score counts, precision, batch sizes, shared prompt hashes, source hashes and terminal artifacts verified. Native supervisor exited zero, released guard, and restored eight communicating burn workers; collective cycles advanced 180→190 at independent inspection.
- [Full metric JSON](results/a2_downstream_20261028/final/supervised/run/downstream/results.json), [terminal verification](results/a2_downstream_20261028/final/terminal-verification.json), [artifact checksums](results/a2_downstream_20261028/final/SHA256SUMS.json).
- Local original log: `temp/a2_downstream_health03.log`, SHA256 `8c02811fa5068df41e751d09d993cce36abeed3a67326b02bbcf9b73dad332ca`. Extracted receipts: `temp/a2_downstream_health03/`.
- Full per-request scores, canonical replay files and checkpoints remain on B200; score/replay hashes and exact paths are recorded in the archived downstream `complete.json`. They have not been downloaded as full files.
- Run `6y3o-a2-downstream-20261028-a02`, source `a3da3fc`, launch `b08fc10`, health `f0e768e`. Existing runtimes/data unchanged. CPU replay printed a harmless missing-Git-metadata diagnostic in the source-only snapshot; all replay processes and metric validation succeeded.
- Phase0A remains complete (58 fits); no Phase0B launched. Conservative ledger including this eval and cleanup is approximately 13.02 node-hours, below 15h report / 18h hard caps.
