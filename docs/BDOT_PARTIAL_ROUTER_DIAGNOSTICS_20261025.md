# Partial-router full-validation diagnostics — 25 October 2026

Evaluation only, completed **25 October 2026, 14:35:39 SGT**.
All four requested checkpoints were evaluated unchanged; no training or optimizer
steps were run. Hashes of checkpoint weights and metadata match before/after.

Each policy scores the same **9,704 validation sequences / 9,915,698 next-token
targets**. Routing maps cover **all 9,936,896 input positions and 79,495,168 routing
decisions**, including EOS/document starts even where their LM labels are masked.
The old usage maps covered only 4,096 sequences; the maps here replace that subset
for this comparison. FA4 BF16, sequence 1,024, eight evaluation ranks, batch 16/rank;
the original tokenizer, EOS, document isolation, reset positions and validation
order are reused from each checkpoint's data recipe.

`Own` uses the trained router. `Forced default` chooses the home module at each
step with the same learned weights. `Random` samples uniformly within the permitted
radius-1 window, allowing reuse. These are inference interventions, not retrained
dense/random controls. Only the routed MLP slice changes; fixed attention and the
remaining MLP channels keep their trained placement. Random routing uses seed 0
for seed-0 checkpoints and seed 1 for the seed-1 checkpoint, reproducing the earlier
evaluation convention; its replay also depends on batch size and rank count.

All maps and module-load tables below describe **own routing**. Off-default share
is the fraction of routing decisions choosing another module, not the fraction
of whole-model computation changed. Loss is NLL in nats/token; lower is better.

## Losses and off-default routing

| Checkpoint | Off-default | Own loss | Forced-default loss | Random loss | Default − own | Random − own |
| --- | --- | --- | --- | --- | --- | --- |
| Partial-25-cosine, seed 0, step 4,768 | 14.4480% | 3.495407 | 3.494408 | 3.557807 | -0.000999 | +0.062400 |
| Partial-25-cosine, seed 1, step 4,768 | 12.3656% | 3.496507 | 3.495690 | 3.560467 | -0.000816 | +0.063960 |
| Partial-25-cosine, seed 0, step 10,000 | 20.2274% | 3.297071 | 3.296087 | 3.366806 | -0.000984 | +0.069735 |
| Partial-50-cosine-detach, seed 0, step 4,768 | 8.8867% | 3.497266 | 3.495615 | 3.635753 | -0.001650 | +0.138487 |

The own-route losses reproduce the saved full-validation results within the
1e-5-nat check. All routing-map row sums, column loads, off-default totals and
radius-1 constraints were checked exactly (ratios within floating-point tolerance).
Forced-default loss differences measure dependence on routing **at these weights**;
they do not replace a matched training comparison.

Forced default is slightly better in all four checkpoints (0.000816–0.001650
nats/token); random routing is worse by 0.062400–0.138487 nats/token. Thus the
routers are active, but their learned choices do not improve validation NLL over
the home path at these weights. These small default-path differences are not a
statistical-significance claim.

There is no severe module-load collapse: load/uniform ranges are 0.9912–1.0065
for Partial-25 seed 0, 0.9851–1.0193 for seed 1, 0.9370–1.0714 at 10k, and
0.9895–1.0112 for Partial-50-detach. At 10k, steps 6/7 choose each other's modules
for about 43% of positions, while those modules' aggregate loads remain nearly
uniform. Balanced loads alone therefore do not establish useful routing.

## Partial-25-cosine, seed 0, step 4,768

Rows are middle execution steps 0–7; columns are routed module IDs 0–7.
Entries are **percent of that step's input positions** (each row sums to 100%,
up to rounding). The diagonal is the default route. These steps correspond to
decoder layers 2–9 in zero-based indexing, after the two stem layers.

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off-default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 88.404 | 11.596 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 11.596% |
| 1 | 11.913 | 84.968 | 3.119 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 15.032% |
| 2 | 0.000 | 4.086 | 94.059 | 1.854 | 0.000 | 0.000 | 0.000 | 0.000 | 5.941% |
| 3 | 0.000 | 0.000 | 1.943 | 95.068 | 2.989 | 0.000 | 0.000 | 0.000 | 4.932% |
| 4 | 0.000 | 0.000 | 0.000 | 2.974 | 83.639 | 13.386 | 0.000 | 0.000 | 16.361% |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 13.376 | 84.441 | 2.183 | 0.000 | 15.559% |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.131 | 75.908 | 21.961 | 24.092% |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 22.071 | 77.929 | 22.071% |

| Module | Total routed positions | Share of all decisions | Load / uniform target |
| --- | --- | --- | --- |
| M0 | 9,968,359 | 12.5396% | 1.0032× |
| M1 | 10,001,555 | 12.5813% | 1.0065× |
| M2 | 9,849,587 | 12.3902% | 0.9912× |
| M3 | 9,926,558 | 12.4870% | 0.9990× |
| M4 | 9,937,385 | 12.5006% | 1.0000× |
| M5 | 9,932,719 | 12.4947% | 0.9996× |
| M6 | 9,953,009 | 12.5203% | 1.0016× |
| M7 | 9,925,996 | 12.4863% | 0.9989× |

Uniform load is 12.5% of decisions, or 9,936,896 selections per module. These counts exclude the fixed MLP core and attention work.

## Partial-25-cosine, seed 1, step 4,768

Rows are middle execution steps 0–7; columns are routed module IDs 0–7.
Entries are **percent of that step's input positions** (each row sums to 100%,
up to rounding). The diagonal is the default route. These steps correspond to
decoder layers 2–9 in zero-based indexing, after the two stem layers.

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off-default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 91.145 | 8.855 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8.855% |
| 1 | 8.320 | 89.206 | 2.474 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10.794% |
| 2 | 0.000 | 3.865 | 95.586 | 0.549 | 0.000 | 0.000 | 0.000 | 0.000 | 4.414% |
| 3 | 0.000 | 0.000 | 2.016 | 96.449 | 1.535 | 0.000 | 0.000 | 0.000 | 3.551% |
| 4 | 0.000 | 0.000 | 0.000 | 1.517 | 83.479 | 15.005 | 0.000 | 0.000 | 16.521% |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 15.052 | 83.592 | 1.356 | 0.000 | 16.408% |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.401 | 80.151 | 18.448 | 19.849% |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18.532 | 81.468 | 18.532% |

| Module | Total routed positions | Share of all decisions | Load / uniform target |
| --- | --- | --- | --- |
| M0 | 9,883,697 | 12.4331% | 0.9946× |
| M1 | 10,128,276 | 12.7407% | 1.0193× |
| M2 | 9,944,503 | 12.5096% | 1.0008× |
| M3 | 9,789,307 | 12.3143% | 0.9851× |
| M4 | 9,943,426 | 12.5082% | 1.0007× |
| M5 | 9,936,575 | 12.4996% | 1.0000× |
| M6 | 9,940,765 | 12.5049% | 1.0004× |
| M7 | 9,928,619 | 12.4896% | 0.9992× |

Uniform load is 12.5% of decisions, or 9,936,896 selections per module. These counts exclude the fixed MLP core and attention work.

## Partial-25-cosine, seed 0, step 10,000

Rows are middle execution steps 0–7; columns are routed module IDs 0–7.
Entries are **percent of that step's input positions** (each row sums to 100%,
up to rounding). The diagonal is the default route. These steps correspond to
decoder layers 2–9 in zero-based indexing, after the two stem layers.

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off-default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 91.610 | 8.390 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8.390% |
| 1 | 8.213 | 80.054 | 11.734 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 19.946% |
| 2 | 0.000 | 11.975 | 84.821 | 3.204 | 0.000 | 0.000 | 0.000 | 0.000 | 15.179% |
| 3 | 0.000 | 0.000 | 10.586 | 89.414 | 0.000 | 0.000 | 0.000 | 0.000 | 10.586% |
| 4 | 0.000 | 0.000 | 0.000 | 1.081 | 89.586 | 9.333 | 0.000 | 0.000 | 10.414% |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 9.133 | 90.114 | 0.753 | 0.000 | 9.886% |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.840 | 55.841 | 43.319 | 44.159% |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 43.259 | 56.741 | 43.259% |

| Module | Total routed positions | Share of all decisions | Load / uniform target |
| --- | --- | --- | --- |
| M0 | 9,919,327 | 12.4779% | 0.9982× |
| M1 | 9,978,442 | 12.5523% | 1.0042× |
| M2 | 10,646,455 | 13.3926% | 1.0714× |
| M3 | 9,310,779 | 11.7124% | 0.9370× |
| M4 | 9,809,614 | 12.3399% | 0.9872× |
| M5 | 9,965,351 | 12.5358% | 1.0029× |
| M6 | 9,922,361 | 12.4817% | 0.9985× |
| M7 | 9,942,839 | 12.5075% | 1.0006× |

Uniform load is 12.5% of decisions, or 9,936,896 selections per module. These counts exclude the fixed MLP core and attention work.

## Partial-50-cosine-detach, seed 0, step 4,768

Rows are middle execution steps 0–7; columns are routed module IDs 0–7.
Entries are **percent of that step's input positions** (each row sums to 100%,
up to rounding). The diagonal is the default route. These steps correspond to
decoder layers 2–9 in zero-based indexing, after the two stem layers.

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off-default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 97.347 | 2.653 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.653% |
| 1 | 3.773 | 95.169 | 1.059 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.831% |
| 2 | 0.000 | 1.125 | 96.653 | 2.222 | 0.000 | 0.000 | 0.000 | 0.000 | 3.347% |
| 3 | 0.000 | 0.000 | 2.216 | 86.882 | 10.902 | 0.000 | 0.000 | 0.000 | 13.118% |
| 4 | 0.000 | 0.000 | 0.000 | 10.904 | 86.300 | 2.796 | 0.000 | 0.000 | 13.700% |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 2.951 | 96.521 | 0.528 | 0.000 | 3.479% |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.533 | 84.617 | 14.850 | 15.383% |
| 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 14.581 | 85.419 | 14.581% |

| Module | Total routed positions | Share of all decisions | Load / uniform target |
| --- | --- | --- | --- |
| M0 | 10,048,149 | 12.6399% | 1.0112× |
| M1 | 9,832,262 | 12.3684% | 0.9895× |
| M2 | 9,929,710 | 12.4910% | 0.9993× |
| M3 | 9,937,648 | 12.5009% | 1.0001× |
| M4 | 9,952,107 | 12.5191% | 1.0015× |
| M5 | 9,922,046 | 12.4813% | 0.9985× |
| M6 | 9,909,683 | 12.4658% | 0.9973× |
| M7 | 9,963,563 | 12.5335% | 1.0027× |

Uniform load is 12.5% of decisions, or 9,936,896 selections per module. These counts exclude the fixed MLP core and attention work.

## Evidence

[Verified results and checkpoint hashes](results/bdot_partial_router_full_validation_20261025.json),
[exact per-step counts/shares](results/bdot_partial_router_usage_full_20261025.csv),
[exact module loads](results/bdot_partial_router_loads_full_20261025.csv).
The JSON includes all three policies' PPL, off-default share, counts and provenance.
Raw downloaded status: `temp/partial_router_diagnostics_20261025/final-status.json`.
The original frozen evaluator was used without model/loss changes. Accelerate
configuration was copied and verified; only the verified idle burns were reclaimed,
GPUs were checked free, and communicating burns were restored after evaluation.
Burns were restored at **14:36:50 SGT**, verified live and communicating at
**14:37:54 SGT**. Launch commit: `3f8b269`; final read-only inspection: `718c5c1`.

[Earlier consolidated results](BDOT_RESULTS_20261025.md).
