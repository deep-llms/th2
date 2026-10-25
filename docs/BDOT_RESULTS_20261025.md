# B-DOT consolidated follow-up results — 25 October 2026

This file collects the completed results that were scattered across JSON receipts,
CURRENT_TASK and PROJECT_NOTES: all 12 partial-MLP/router runs, second-seed checks,
learning curves, the fresh 10k pair, downstream evaluation, both short STOP studies,
the optimized 5k/10k STOP continuation and throughput/numerical checks. Interrupted
runs are recorded separately. All dates/times in this report use Singapore time
(SGT, UTC+8). The latest evidence was verified **25 October 2026, 13:19 SGT**.

**Main findings:** Partial-25-cosine nearly matches dense D but has not beaten it
on full validation or aggregate downstream accuracy. DualCE makes an early exit
more accurate, and its training implementation is much faster after optimization.
However, none of the tested learned STOP policies achieves the required combined
quality and decoding-speed improvement.

## Scope and comparison rules

All research runs here use Qwen3-0.6B dimensions with **12 layers: 2 stem + 8 middle
+ 2 coda**, on English `nht10/cx_sampled`, subset `qwen3_0.6b_base_en_30B`.
The initial models are trained **from scratch**; STOP studies adapt their trained
checkpoints. The matched dense control **D** uses fixed module order and step FiLM
in the B-DOT implementation. It is distinct from the earlier native Qwen baseline.

Shared training: sequence 1,024; microbatch 16 × accumulation 4 × 8 B200 GPUs =
524,288 input tokens/update; HF Trainer/Accelerate; FA4 BF16 with FP32 parameters;
decoder activation checkpointing disabled. Packing adds EOS at document boundaries,
isolates documents and resets positions. Data order is fixed at seed 0. Model/router
seed is 0 unless a second-seed row explicitly says 1.

Final validation is always **9,704 sequences / 9,915,698 scored next-token targets**.
Loss is LM NLL in nats/token, excluding auxiliary/router penalties; PPL = exp(loss).
Lower loss/PPL is better. A positive loss gap means worse than the stated control.
Input-token budgets differ from scored-token counts because of label masking.
Periodic validation uses only 1,024 sequences; calibration uses separate training
rows. Neither should be substituted for the full held-out result.

| Stage | Training schedule | Input tokens per fit | Comparison limit |
| --- | --- | --- | --- |
| Partial/router screen and seed confirmation | Fresh 4,768 updates; 96 warmup; LR 3e-4, cosine floor 3e-5 | 2.4998B | Matched short pilot |
| Fresh D / Partial-25-cosine pair | Fresh 10,000 updates; 200 warmup; same LR/floor | 5.24288B | Equal steps/tokens, not equal wall time |
| STOP-v1 / STOP-v2 | Original 10k parent + 1,000 new updates; fresh AdamW; 20 warmup; LR 3e-5, floor 3e-6 | 0.524288B additional | 11k lifetime; dual objectives cost extra compute |
| Optimized STOP pair | Original D-10k + 10,000 new updates; fresh AdamW; 200 warmup; LR 3e-5, floor 3e-6 | 5.24288B additional | Assess at +5k/+10k; unchanged 10k schedule across native midpoint resume |

The original pilot and v3 results are already documented in full:
[pilot, N-A, R and diagnostics](BDOT_PILOT_RESULTS_20261020.md),
[v3 arms, interventions and C1](BDOT_V3_RESULTS_20261022.md).
Their historical statements about pending studies are superseded by this report.
The earlier native Qwen 5k run had loss **3.467754**, PPL **32.0646**, and Trainer
runtime **50m40s**, but used a **57,220-step schedule with 1,145 warmup steps**;
it is not a matched control for the short pilots. See the
[native baseline report](ATTENTION_BASELINE_LAUNCH_20261020.md) and
[final receipt](results/native_baseline_final_20261020.json).

## 1. Twelve partial-MLP and router runs

These variants keep attention and normalization at their original depth. They
route 25%, 50% or 100% of the middle-layer SwiGLU channels; the rest form a fixed
core. The split preserves the original backbone parameters and initialization.
The unit-forward straight-through gate acts only on the routed slice. Both learned
and random controls use curriculum A: first 10% home-only, then radius 1 with reuse.

`random` selects uniformly among permitted modules. `detach` detaches the input
before the router's own normalization, preventing router gradients from entering
the backbone through that input; router parameters still receive LM/z-loss gradients.
`cosine` normalizes token and router-weight directions with fixed score scale 3.2.
`cosine-detach` combines both. Other routing constraints/regularizers remain unchanged.
`MLP-only` means 100% routed MLP width, not an attention-free model.

| Arm, seed 0 | Final loss | PPL | Δ loss vs D | Benchmark s/update | Benchmark throughput / D |
| --- | --- | --- | --- | --- | --- |
| D (reused seed-0 control) | 3.494464 | 32.9326 | 0.000000 | — | — |
| Partial-25 | 3.502879 | 33.2109 | +0.008415 | 0.7173 | 94.49% |
| Partial-25-random | 3.510149 | 33.4533 | +0.015686 | 0.6802 | 99.64% |
| Partial-50 | 3.530076 | 34.1266 | +0.035612 | 0.7117 | 95.24% |
| Partial-50-random | 3.525737 | 33.9788 | +0.031274 | 0.6769 | 100.14% |
| MLP-only | 3.556062 | 35.0250 | +0.061598 | 0.6898 | 98.26% |
| MLP-only-random | 3.562083 | 35.2365 | +0.067620 | 0.6616 | 102.45% |
| Partial-25-detach | 3.498403 | 33.0626 | +0.003940 | 0.6998 | 97.98% |
| Partial-50-detach | 3.505151 | 33.2865 | +0.010687 | 0.6973 | 98.34% |
| Partial-25-cosine | 3.495407 | 32.9637 | +0.000944 | 0.7208 | 95.12% |
| Partial-50-cosine | 3.499308 | 33.0925 | +0.004844 | 0.7241 | 94.69% |
| Partial-25-cosine-detach | 3.496596 | 33.0029 | +0.002133 | 0.7084 | 96.80% |
| Partial-50-cosine-detach | 3.497266 | 33.0250 | +0.002802 | 0.7072 | 96.96% |

All 12 fits completed 4,768 steps. Every router follow-up improved its original
same-width learned router, but none beat D. Partial-25-cosine came closest.
Cosine plus detachment helps at 50% width relative to cosine alone, but not at 25%.

The timing columns are **whole-update synthetic benchmarks**, not total fit times.
The two batches used their own dense timing controls (0.6778 and 0.6857 s/update).
Learned-router timing removed the home prior to exercise multiple experts;
research training retained it. Do not read small timing differences as speed wins.

[All 12 final results, checks and timings](results/bdot_mlp_router_final_20261023.json).

## 2. Second-seed confirmation and learning curves

| Arm | Seed-0 loss | Seed-1 loss | Mean paired loss gap vs D | Geometric-mean PPL gap |
| --- | --- | --- | --- | --- |
| D | 3.494464 | 3.494008 | 0.000000 | 0.000% |
| Partial-25-cosine | 3.495407 | 3.496507 | +0.001721 | +0.172% |
| Partial-50-cosine-detach | 3.497266 | 3.496573 | +0.002684 | +0.269% |

Both routers stay within D + 0.01 nats in both seeds; neither wins either seed.
These are two initialization/router seeds with **fixed data order**, not a broad
multi-seed or statistical-equivalence claim. D controls were reused after recipe,
data and checkpoint-reload verification.

[Source](results/bdot_router_confirmation_final_20261023.json).

| Arm / seed | Mean periodic gap, steps 3,000–4,000 | Mean periodic gap, steps 4,000–4,700 |
| --- | --- | --- |
| Partial-25-cosine-s0 | -0.000143 | +0.000103 |
| Partial-50-cosine-detach-s0 | +0.000937 | +0.001259 |
| Partial-25-cosine-s1 | +0.001562 | +0.001440 |
| Partial-50-cosine-detach-s1 | +0.001814 | +0.001996 |

The late periodic curves remain close to D rather than showing a growing lead.
These means use the smaller periodic validation set, not full-validation scores.

[Complete recorded learning curves](results/bdot_router_curves_20261023.json).

## 3. Fresh 10,000-step pair

Both arms restarted from scratch with the matched full 10k schedule. This is not
a continuation of the 4,768-step pilot. All 40 checkpoints per arm were retained
at 250-step intervals, including optimizer, scheduler and all eight RNG states.

| Arm | Final loss | PPL | Trainer runtime (min) |
| --- | --- | --- | --- |
| D | 3.294948 | 26.9760 | 112.57 |
| Partial-25-cosine | 3.297071 | 27.0333 | 119.19 |

Partial-25-cosine is **+0.002123 nats** worse
(+0.213% PPL), with **5.88% more Trainer time**.
Longer training preserves the near-match but provides no quality win. Runtime
includes periodic evaluation/saving. The queue completed 23 October, 10:03:46 SGT.

[Source](results/bdot_long_pair_final_20261023.json).

## 4. Downstream evaluation of the fresh 10k checkpoints

Nine English **zero-shot** tasks, evaluation seed 42, context 1,024, identical
prompts/example sets across arms, using the lm-eval-harness task definitions.
These are neither fine-tuning nor few-shot results. All 26,618 examples/model
and paired prompt hashes were verified. Two requests/model required identical
left truncation; the others fit. Eight concurrent workers completed the full
evaluation in approximately 177.5 seconds.

Accuracy is in percent; the gap is Partial minus D in percentage points.
`acc_norm` uses the harness's normalized choice score; `acc` is its ordinary
accuracy metric. The average gives each of the eight multiple-choice tasks
equal weight and excludes LAMBADA.

| Task | Metric | Examples | D | Partial-25-cosine | Gap (pp) |
| --- | --- | --- | --- | --- | --- |
| hellaswag | acc_norm | 10,042 | 32.3043 | 32.1450 | -0.1593 |
| arc_easy | acc_norm | 2,376 | 39.5623 | 38.8889 | -0.6734 |
| arc_challenge | acc_norm | 1,172 | 23.3788 | 23.2082 | -0.1706 |
| piqa | acc_norm | 1,838 | 63.2753 | 63.7106 | +0.4353 |
| winogrande | acc | 1,267 | 51.1444 | 48.7766 | -2.3678 |
| boolq | acc | 3,270 | 61.3761 | 62.0183 | +0.6422 |
| openbookqa | acc_norm | 500 | 27.0000 | 27.4000 | +0.4000 |
| sciq | acc | 1,000 | 71.9000 | 71.8000 | -0.1000 |
| **MC average** | Macro mean | 8 tasks | 46.2427 | 45.9935 | -0.2492 |

| LAMBADA OpenAI (5,153 examples) | D | Partial-25-cosine |
| --- | --- | --- |
| Accuracy (%) | 23.3262 | 22.8799 |
| Target-word PPL | 102.0429 | 105.9512 |

Partial wins BoolQ, PIQA and OpenBookQA; D wins the other five multiple-choice
tasks and LAMBADA. There is no aggregate downstream advantage in this checkpoint
pair; no significance test or equal-compute claim follows from these numbers.
Alternate harness metrics and request metadata remain in the source report.

[Source](results/bdot_downstream_final_20261023.json).

## 5. STOP evaluation conventions

STOP skips remaining middle computations for selected tokens, but held tokens
still write K/V at later steps and execute the two coda layers. There is **no
KV-cache memory saving claim**. A mean middle depth near 8 means very little
work is being skipped; it is not a whole-model speedup measure.

Tables abbreviate `Partial-25-cosine` as `Partial-25-cos`.

Training and packed PPL use FA4. Cached decoding uses SDPA for **all compared
policies**, with query compaction where supported. Reported decode times are
whole-model milliseconds per cached, teacher-forced decode step on the same
fixed token streams: four warmup steps, then 32 measured steps, median of three
trials. Batch 32 processes 32 tokens per step. These are not free-generation or
serving benchmarks and exclude prefill. Prefill timings
and all raw trials are in the linked JSONs. Backend/cache tests passed their
tolerances; this does not mean bitwise-equal logits or identical threshold
decisions across packed and cached execution.

A separate frozen-backbone linear STOP head learns BCE from heuristic labels:
fixed-depth next-token NLL minus full-depth NLL ≤ 0.01 nats. It is fit on 256 unused
training sequences, then thresholds are calibrated on the next 256. The original
validation split is used only to report the frozen policy. Mixed token-wise exits
are evaluated directly; their quality is not inferred from the label heuristic.
`Random` controls match the selected mean depth on calibration, so their realized
held-out depths may differ slightly.

The reported practical gate requires cache checks to pass, held-out loss ≤ ordinary
full-depth D + 0.01 nats, and at least **10% lower batch-1/context-128 decode time**
than that D control. All tested selected policies fail this joint gate.

### STOP-v1: random-STOP adaptation, 1,000 new steps

Four branches: D / Partial-25-cosine × ordinary continuation / random-STOP
adaptation. Random-STOP training uses minimum middle depth 2 and per-step hazard
0.05. Each starts from its original 10k parent with fresh AdamW; all use the same
next 524,288,000 input tokens. They finish at 11k lifetime steps.

In this first study, threshold calibration constrained loss relative to the
branch's own full-depth path. The practical gate still compares against ordinary
D. All selected learned thresholds were 0.7.

| Checkpoint / branch | Inference policy | Loss | PPL | Δ loss vs full D control | Mean middle steps / 8 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 3.284418 | 26.6934 | +0.000000 | 8.00000 |
| D-continue | Learned, threshold 0.7 | 3.287716 | 26.7816 | +0.003298 | 7.96973 |
| D-continue | Random, p=0.001421 | 3.293132 | 26.9271 | +0.008714 | 7.97034 |
| D-STOP | Full depth | 3.297605 | 27.0478 | +0.013187 | 8.00000 |
| D-STOP | Learned, threshold 0.7 | 3.303101 | 27.1969 | +0.018683 | 7.88053 |
| D-STOP | Random, p=0.005721 | 3.310330 | 27.3942 | +0.025912 | 7.88092 |
| Partial-25-cos-continue | Full depth | 3.286587 | 26.7514 | +0.002169 | 8.00000 |
| Partial-25-cos-continue | Learned, threshold 0.7 | 3.289743 | 26.8360 | +0.005325 | 7.97071 |
| Partial-25-cos-continue | Random, p=0.001229 | 3.293908 | 26.9480 | +0.009490 | 7.97439 |
| Partial-25-cos-STOP | Full depth | 3.299210 | 27.0912 | +0.014792 | 8.00000 |
| Partial-25-cos-STOP | Learned, threshold 0.7 | 3.304685 | 27.2400 | +0.020267 | 7.87126 |
| Partial-25-cos-STOP | Random, p=0.005903 | 3.312032 | 27.4408 | +0.027614 | 7.87713 |

| Branch | Trainer runtime (min) | Full decode ms, B1/C128 | Learned decode ms, B1/C128 |
| --- | --- | --- | --- |
| D-continue | 11.41 | 15.097 | 16.846 |
| D-STOP | 10.91 | 15.159 | 16.822 |
| Partial-25-cos-continue | 12.21 | 19.127 | 20.950 |
| Partial-25-cos-STOP | 12.24 | 18.669 | 19.928 |

Learned heads outperform matched-random policies in held-out loss, but skip too
little depth to recover their overhead. Random-STOP adaptation improves shallow
exits while worsening full-depth quality. Every selected learned policy is slower
than its own full-depth path at all four measured batch/context settings.

The following fixed-exit results are **calibration PPL on 256 sequences**, not the
full held-out results above; they demonstrate the shallow-quality tradeoff.

| Branch | Fixed exit 2 PPL | Fixed exit 4 PPL | Fixed exit 6 PPL |
| --- | --- | --- | --- |
| D-continue | 284.3478 | 138.2673 | 44.5263 |
| D-STOP | 76.1516 | 52.3429 | 33.9000 |
| Partial-25-cos-continue | 265.9263 | 136.7096 | 43.3585 |
| Partial-25-cos-STOP | 74.9907 | 51.7980 | 33.7117 |

Completed 23 October, 12:41:45 SGT. All four checkpoints per branch were retained.

[Source](results/bdot_stop_final_20261024.json).

### STOP-v2: later exits and auxiliary exit training, 1,000 new steps

Six **new** fits: D / Partial-25-cosine × three methods below. They start from
the original 10k parents, not STOP-v1 adapted weights. The two ordinary continuation
checkpoints from STOP-v1 are reused as controls and evaluated with the new policy.
Thus this is six new training fits and eight checkpoint evaluations, not eight fits.

| Method | Training change |
| --- | --- |
| LateDrop6 | One traversal; randomly choose full depth or exit at middle step 6 with probability 0.5. |
| DualCE6 | Full-depth LM loss + 0.5 × actual exit-6 LM cross-entropy. |
| DualKD6 | DualCE6 + 0.5 × KL(detached full-depth distribution ∥ exit-6 distribution), temperature 1. |

Only one STOP opportunity, after middle step 6. Both traversals are enclosed in
one DDP forward; full-path routing regularization/balancing is counted once.
Input-token progress counts each training example once despite dual computation.
Calibration now requires loss within +0.01 nats of **both** the branch's full path
and its matching ordinary continuation control. If even full depth fails, the
policy falls back to no STOP and is marked infeasible.

| Checkpoint / branch | Inference policy | Loss | PPL | Δ loss vs full D control | Mean middle steps / 8 |
| --- | --- | --- | --- | --- | --- |
| D-DualKD6 | Full depth | 3.300008 | 27.1129 | +0.015590 | 8.00000 |
| D-DualKD6 | Fixed exit 6 | 3.416842 | 30.4730 | +0.132424 | 6.00000 |
| D-DualKD6 | Random, p=0.000000 | 3.300008 | 27.1129 | +0.015590 | 8.00000 |
| Partial-25-cos-DualKD6 | Full depth | 3.301325 | 27.1486 | +0.016907 | 8.00000 |
| Partial-25-cos-DualKD6 | Fixed exit 6 | 3.414484 | 30.4013 | +0.130066 | 6.00000 |
| Partial-25-cos-DualKD6 | Random, p=0.000000 | 3.301325 | 27.1486 | +0.016907 | 8.00000 |
| D-DualCE6 | Full depth | 3.292428 | 26.9081 | +0.008010 | 8.00000 |
| D-DualCE6 | Fixed exit 6 | 3.447184 | 31.4118 | +0.162765 | 6.00000 |
| D-DualCE6 | Learned, threshold 0.8 | 3.292598 | 26.9127 | +0.008180 | 7.98756 |
| D-DualCE6 | Random, p=0.006672 | 3.293445 | 26.9355 | +0.009027 | 7.98667 |
| Partial-25-cos-DualCE6 | Full depth | 3.293928 | 26.9485 | +0.009510 | 8.00000 |
| Partial-25-cos-DualCE6 | Fixed exit 6 | 3.443526 | 31.2971 | +0.159108 | 6.00000 |
| Partial-25-cos-DualCE6 | Learned, threshold 0.8 | 3.294115 | 26.9536 | +0.009697 | 7.98356 |
| Partial-25-cos-DualCE6 | Random, p=0.007935 | 3.295095 | 26.9800 | +0.010677 | 7.98415 |
| D-LateDrop6 | Full depth | 3.299804 | 27.1073 | +0.015386 | 8.00000 |
| D-LateDrop6 | Fixed exit 6 | 3.442216 | 31.2561 | +0.157798 | 6.00000 |
| D-LateDrop6 | Random, p=0.000000 | 3.299804 | 27.1073 | +0.015386 | 8.00000 |
| Partial-25-cos-LateDrop6 | Full depth | 3.300990 | 27.1395 | +0.016572 | 8.00000 |
| Partial-25-cos-LateDrop6 | Fixed exit 6 | 3.438274 | 31.1332 | +0.153856 | 6.00000 |
| Partial-25-cos-LateDrop6 | Random, p=0.000000 | 3.300990 | 27.1395 | +0.016572 | 8.00000 |
| D-continue | Full depth | 3.284418 | 26.6934 | +0.000000 | 8.00000 |
| D-continue | Fixed exit 6 | 3.798434 | 44.6312 | +0.514016 | 6.00000 |
| D-continue | Learned, threshold 0.6 | 3.287190 | 26.7675 | +0.002772 | 7.97255 |
| D-continue | Random, p=0.014420 | 3.290731 | 26.8625 | +0.006313 | 7.97113 |
| Partial-25-cos-continue | Full depth | 3.286587 | 26.7514 | +0.002169 | 8.00000 |
| Partial-25-cos-continue | Fixed exit 6 | 3.769277 | 43.3487 | +0.484859 | 6.00000 |
| Partial-25-cos-continue | Learned, threshold 0.6 | 3.289241 | 26.8225 | +0.004823 | 7.97428 |
| Partial-25-cos-continue | Random, p=0.012424 | 3.291633 | 26.8867 | +0.007215 | 7.97511 |

| Branch | Trainer runtime (min) | Selected policy | Calibration feasible | Selected decode ms, B1/C128 | Practical gate |
| --- | --- | --- | --- | --- | --- |
| D-DualKD6 | 40.67 | Full depth | No | 14.646 | Fail |
| Partial-25-cos-DualKD6 | 42.04 | Full depth | No | 18.859 | Fail |
| D-DualCE6 | 35.82 | Learned, threshold 0.8 | Yes | 16.472 | Fail |
| Partial-25-cos-DualCE6 | 37.21 | Learned, threshold 0.8 | Yes | 19.308 | Fail |
| D-LateDrop6 | 11.11 | Full depth | No | 15.308 | Fail |
| Partial-25-cos-LateDrop6 | 12.04 | Full depth | No | 19.119 | Fail |
| D-continue | Reused | Learned, threshold 0.6 | Yes | 16.603 | Fail |
| Partial-25-cos-continue | Reused | Learned, threshold 0.6 | Yes | 19.627 | Fail |

DualCE6 preserves enough full-depth quality to select a learned exit, but only
about 0.62% of D tokens and 0.82% of Partial tokens exit at step 6 (derived from
held-out mean depth). Decoding remains slower. DualKD6 and LateDrop6 fail the
calibration quality allowance even at full depth. KD improves fixed-exit quality
more than CE alone, at greater full-depth cost. An occasional lower no-STOP timing
does not demonstrate a STOP speedup. Completed 24 October, 19:59:28 SGT.

[Source](results/bdot_stop_v2_final_20261024.json).

## 6. DualCE execution optimization and verification

The old chunked CE repeatedly padded/accumulated full-logit slice gradients, and
the dual objective recomputed an identical prefix. The new opt-in execution uses
native CE and shares that prefix. It also skips auxiliary K/V writes with no
consumers after all auxiliary tokens exit. The architecture, objective and
normalization are unchanged. Native CE is used here with KL weight zero.

Local A100/SDPA full-model tests used microbatch 2, accumulation 4, sequence 1,024:
D ≈0.451 s/update, old DualCE ≈0.860, native CE alone ≈0.812, and native CE plus
prefix sharing ≈0.570. This is a **33.8% whole-update reduction**, not a B200
production measurement. A much larger loss-only speedup is not a whole-step claim.

[Local numerical, DDP, resume and timing evidence](results/bdot_stop_optimization_local_20261024.json).

The final B200 gate used all eight GPUs, the real full-width model, FA4 BF16,
microbatch 16 × accumulation 4, sequence 1,024, grouped execution, and two timing
repeats in reversed order. Peak memory below is PyTorch allocated memory per GPU,
not total device usage.

| Implementation | Seconds / update | Two repeats (s) | Peak allocated (GiB) |
| --- | --- | --- | --- |
| D control | 0.669636 | 0.669286, 0.669985 | 52.65 |
| Old DualCE6 | 2.125681 | 2.122777, 2.128584 | 76.50 |
| Optimized DualCE6 | 0.861304 | 0.861788, 0.860820 | 64.52 |

The optimization reduces update time by **59.48%** (2.47× old DualCE throughput).
Optimized DualCE still costs **1.286× D** per update. These are measured whole
training updates, including accumulation/backward/optimizer, not attention-only timings.

Full-width, original-parent B200 tests: native CE alone matched loss, logits,
gradients and one AdamW update exactly in the fixture. Shared-prefix execution
matched loss/logits; gradient relative L2 difference was **0.2917%**, cosine
**0.9999957**, and maximum parameter difference after one AdamW update **5.97e-5**.
BF16 accumulation order changes, so this is not bitwise trajectory equivalence.
Microbatch-16 capacity, DDP, exact eight-rank native resume on a reduced-width
12-layer fixture, and real evaluator smoke checks passed before training.

Raw B200 numerical checks and timings:
[downloaded final status](../temp/stop_optimized_20261025/health-a10.json).
The reduced-width resume test is a bounded correctness check, not a convergence guarantee.

[Final B200 optimization and training report](results/bdot_stop_optimized_final_20261025.json).

## 7. Optimized continuation: +5k and +10k steps from D-10k

Both branches started **fresh from the original D-10k weights**, with fresh AdamW
and a full 10k adaptation schedule. They did not resume either the short 1k fits
or the interrupted unoptimized run. Ordinary D uses full LM loss; DualCE6 uses
full LM + 0.5 exit-6 CE, with native CE/shared-prefix execution.

Training uses packed rows [5,120,000, 10,240,000). Head fitting uses
[10,240,000, 10,240,256), calibration [10,240,256, 10,240,512), beyond the entire
planned adaptation segment. Both +5k and +10k assessments use these disjoint
head/calibration ranges. At +5k, native checkpoint resume preserves optimizer,
scheduler, RNG and data progress under the same 10k schedule. Evaluation does
not modify the saved training weights.

The +5k checkpoint is 15k lifetime steps; +10k is 20k lifetime. These are
continuation schedules after the original 10k cosine decay, not a fresh 20k
schedule or a direct continuation of the earlier 57,220-step native baseline.

### +5,000 adaptation steps / 15,000 lifetime steps

| Checkpoint / branch | Inference policy | Loss | PPL | Δ loss vs full D control | Mean middle steps / 8 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 3.266547 | 26.2206 | +0.000000 | 8.00000 |
| D-continue | Fixed exit 6 | 3.815045 | 45.3788 | +0.548498 | 6.00000 |
| D-continue | Learned, threshold 0.6 | 3.268615 | 26.2749 | +0.002068 | 7.97281 |
| D-continue | Random, p=0.011913 | 3.272039 | 26.3650 | +0.005492 | 7.97612 |
| D-DualCE6 | Full depth | 3.274890 | 26.4403 | +0.008343 | 8.00000 |
| D-DualCE6 | Fixed exit 6 | 3.367921 | 29.0181 | +0.101374 | 6.00000 |
| D-DualCE6 | Learned, threshold 0.8 | 3.275513 | 26.4568 | +0.008966 | 7.94547 |
| D-DualCE6 | Random, p=0.023918 | 3.277349 | 26.5054 | +0.010802 | 7.95213 |

| Branch | Full decode ms, B1/C128 | Learned decode ms, B1/C128 | Practical gate |
| --- | --- | --- | --- |
| D-continue | 14.990 | 15.817 | Fail |
| D-DualCE6 | 14.797 | 15.676 | Fail |

### +10,000 adaptation steps / 20,000 lifetime steps

| Checkpoint / branch | Inference policy | Loss | PPL | Δ loss vs full D control | Mean middle steps / 8 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 3.253432 | 25.8790 | +0.000000 | 8.00000 |
| D-continue | Fixed exit 6 | 3.823307 | 45.7553 | +0.569876 | 6.00000 |
| D-continue | Learned, threshold 0.6 | 3.255445 | 25.9312 | +0.002014 | 7.97266 |
| D-continue | Random, p=0.011890 | 3.259113 | 26.0265 | +0.005682 | 7.97616 |
| D-DualCE6 | Full depth | 3.262074 | 26.1036 | +0.008643 | 8.00000 |
| D-DualCE6 | Fixed exit 6 | 3.348610 | 28.4631 | +0.095179 | 6.00000 |
| D-DualCE6 | Learned, threshold 0.8 | 3.262729 | 26.1207 | +0.009297 | 7.94561 |
| D-DualCE6 | Random, p=0.023819 | 3.264411 | 26.1647 | +0.010979 | 7.95234 |

| Branch | Full decode ms, B1/C128 | Learned decode ms, B1/C128 | Practical gate |
| --- | --- | --- | --- |
| D-continue | 14.628 | 15.440 | Fail |
| D-DualCE6 | 14.707 | 15.565 | Fail |

### Total training time and interpretation

Times below sum both training stages; intervening standalone STOP evaluation is
excluded. Job elapsed includes launch/exit overhead. Trainer time includes its
periodic evaluation/saving. Resumed Trainer aggregate `train_steps_per_second`
and aggregate objective-loss summaries are not used to compare the stages.

| Branch | New / lifetime updates | Summed Trainer time (min) | Summed training-job time (min) | Saved checkpoints |
| --- | --- | --- | --- | --- |
| D-continue | 10,000 / 20,000 | 112.69 | 114.34 | 40 |
| D-DualCE6 | 10,000 / 20,000 | 145.85 | 147.48 | 40 |

At 20k lifetime steps, full-depth DualCE trails full D by **0.008643 nats**
(about **0.87% PPL**). Learned DualCE exits after step 6 for about **2.72%** of
tokens, saving only **0.68% of middle steps** on average; these percentages are
derived from its held-out mean depth. This is too little to recover policy overhead.
Fixed exit 6 is faster (see the latency appendix) and substantially more accurate
after DualCE than after ordinary training, but its **PPL 28.4631** remains worse
than full D's **25.8790**. Longer adaptation improves losses without producing a
useful learned STOP speed–quality point.

All 18 verification/training/evaluation jobs passed. Each arm retained all 40
250-step checkpoints with weights, optimizer, scheduler and eight RNG states.
Training and final evaluation completed **25 October, 12:44:52 SGT**; eight
communicating GPU burns were restored **12:46:03**, independently verified live
**13:19:46**. These are timestamped observations, not a new live status check.

[Source](results/bdot_stop_optimized_final_20261025.json).

## 8. Interrupted runs and superseded verification attempts

The first unoptimized long queue (`6y3o-dot-stop-long-20261024-a01`) was stopped
with authorization before the matched pair finished. Its D control completed
+5k adaptation steps: **loss 3.266553, PPL 26.220797**, Trainer runtime
**56.01 minutes**. DualCE's last reported training step was **2,180**, with last
complete checkpoint **2,000**. There is no matched final +5k/+10k result for that
DualCE run. Its outputs were preserved; it must not be counted as a completed pair.
Source: [downloaded pre-stop status](../temp/stop_long_20261024/health-a04.json)
and [post-stop status](../temp/stop_long_20261024/health-a05.json).

Optimized verification attempt `a01` was canceled before training because timing
probes selected the reference rather than production grouped backend. Attempt
`a02` passed numerical/capacity/DDP/resume/timing checks, then failed before training
because a frozen CLI snapshot lacked an eagerly imported launch helper. The
import packaging was fixed and tested; `a03` reran the gates and produced the
completed results above. A02 timing (D 0.6666, old DualCE 2.1191, optimized 0.8621
s/update) agrees with the final A03 measurement. These attempts are verification
history, not extra scientific training replicates.
See [preserved progress/evidence](results/bdot_stop_optimized_progress_20261025.json).

## 9. Cached decoding latency appendix

All values are median milliseconds per cached, teacher-forced decode step. Columns are
**batch size / initial context length**. Comparisons should be made within a
study/checkpoint; small cross-study timing changes can reflect measurement noise.
`—` means that setting was not measured for that policy. PPL and mean depths for
these policies appear above. Random p is a per-opportunity probability, not a
percentage speed reduction.

### STOP-v1, +1k

| Branch | Policy | 1 / 128 | 1 / 512 | 32 / 128 | 32 / 512 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 15.097 | 15.142 | 15.236 | 15.245 |
| D-continue | Learned, threshold 0.7 | 16.846 | 16.967 | 17.030 | 17.074 |
| D-continue | Random, p=0.001421 | 16.108 | — | 16.089 | — |
| D-STOP | Full depth | 15.159 | 15.091 | 15.240 | 15.147 |
| D-STOP | Learned, threshold 0.7 | 16.822 | 16.743 | 16.846 | 16.840 |
| D-STOP | Random, p=0.005721 | 16.099 | — | 15.851 | — |
| Partial-25-cos-continue | Full depth | 19.127 | 19.302 | 20.903 | 21.062 |
| Partial-25-cos-continue | Learned, threshold 0.7 | 20.950 | 20.624 | 22.989 | 22.910 |
| Partial-25-cos-continue | Random, p=0.001229 | 20.214 | — | 22.229 | — |
| Partial-25-cos-STOP | Full depth | 18.669 | 18.621 | 20.327 | 20.337 |
| Partial-25-cos-STOP | Learned, threshold 0.7 | 19.928 | 20.085 | 22.255 | 22.692 |
| Partial-25-cos-STOP | Random, p=0.005903 | 19.406 | — | 22.007 | — |

### STOP-v2, +1k (including reused controls)

| Branch | Policy | 1 / 128 | 1 / 512 | 32 / 128 | 32 / 512 |
| --- | --- | --- | --- | --- | --- |
| D-DualKD6 | Full depth | 14.646 | 14.646 | 14.717 | 14.696 |
| D-DualKD6 | Fixed exit 6 | 13.285 | 13.234 | 13.518 | 13.311 |
| D-DualKD6 | Random, p=0.000000 | 15.449 | 15.268 | 15.381 | 15.458 |
| Partial-25-cos-DualKD6 | Full depth | 18.859 | 18.880 | 20.110 | 20.536 |
| Partial-25-cos-DualKD6 | Fixed exit 6 | 16.356 | 17.771 | 17.577 | 17.640 |
| Partial-25-cos-DualKD6 | Random, p=0.000000 | 19.428 | 19.461 | 20.925 | 23.578 |
| D-DualCE6 | Full depth | 15.498 | 18.291 | 15.689 | 15.563 |
| D-DualCE6 | Fixed exit 6 | 14.150 | 14.016 | 14.093 | 14.037 |
| D-DualCE6 | Learned, threshold 0.8 | 16.472 | 16.445 | 16.573 | 16.517 |
| D-DualCE6 | Random, p=0.006672 | 16.146 | 16.195 | 16.271 | 16.243 |
| Partial-25-cos-DualCE6 | Full depth | 18.453 | 18.489 | 19.966 | 20.045 |
| Partial-25-cos-DualCE6 | Fixed exit 6 | 16.109 | 16.046 | 17.310 | 17.385 |
| Partial-25-cos-DualCE6 | Learned, threshold 0.8 | 19.308 | 21.126 | 22.004 | 20.894 |
| Partial-25-cos-DualCE6 | Random, p=0.007935 | 18.998 | 20.852 | 22.598 | 20.888 |
| D-LateDrop6 | Full depth | 15.308 | 15.235 | 15.345 | 15.329 |
| D-LateDrop6 | Fixed exit 6 | 13.852 | 13.878 | 13.910 | 13.898 |
| D-LateDrop6 | Random, p=0.000000 | 15.982 | 15.840 | 15.937 | 15.986 |
| Partial-25-cos-LateDrop6 | Full depth | 19.119 | 19.046 | 20.535 | 21.071 |
| Partial-25-cos-LateDrop6 | Fixed exit 6 | 16.596 | 16.535 | 17.559 | 17.711 |
| Partial-25-cos-LateDrop6 | Random, p=0.000000 | 19.628 | 19.564 | 21.143 | 21.180 |
| D-continue | Full depth | 15.542 | 15.396 | 15.464 | 15.570 |
| D-continue | Fixed exit 6 | 14.081 | 14.024 | 14.779 | 14.155 |
| D-continue | Learned, threshold 0.6 | 16.603 | 16.332 | 16.195 | 16.110 |
| D-continue | Random, p=0.014420 | 16.033 | 16.189 | 16.165 | 16.133 |
| Partial-25-cos-continue | Full depth | 19.014 | 18.756 | 20.464 | 20.590 |
| Partial-25-cos-continue | Fixed exit 6 | 16.365 | 16.310 | 17.583 | 17.737 |
| Partial-25-cos-continue | Learned, threshold 0.6 | 19.627 | 21.649 | 21.488 | 21.893 |
| Partial-25-cos-continue | Random, p=0.012424 | 19.273 | 21.632 | 23.253 | 21.263 |

### Optimized pair, +5k

| Branch | Policy | 1 / 128 | 1 / 512 | 32 / 128 | 32 / 512 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 14.990 | 15.048 | 15.092 | 15.105 |
| D-continue | Fixed exit 6 | 13.618 | 13.605 | 13.722 | 13.722 |
| D-continue | Learned, threshold 0.6 | 15.817 | 15.826 | 15.856 | 18.699 |
| D-continue | Random, p=0.011913 | 15.804 | 18.350 | 15.647 | 15.642 |
| D-DualCE6 | Full depth | 14.797 | 14.718 | 14.842 | 14.782 |
| D-DualCE6 | Fixed exit 6 | 13.418 | 13.412 | 13.497 | 13.494 |
| D-DualCE6 | Learned, threshold 0.8 | 15.676 | 15.635 | 15.569 | 15.516 |
| D-DualCE6 | Random, p=0.023918 | 17.666 | 15.655 | 17.750 | 15.561 |

### Optimized pair, +10k

| Branch | Policy | 1 / 128 | 1 / 512 | 32 / 128 | 32 / 512 |
| --- | --- | --- | --- | --- | --- |
| D-continue | Full depth | 14.628 | 14.642 | 14.685 | 14.691 |
| D-continue | Fixed exit 6 | 13.307 | 13.376 | 13.369 | 13.349 |
| D-continue | Learned, threshold 0.6 | 15.440 | 15.432 | 15.406 | 15.428 |
| D-continue | Random, p=0.011890 | 15.282 | 15.229 | 15.264 | 15.245 |
| D-DualCE6 | Full depth | 14.707 | 14.752 | 14.770 | 14.781 |
| D-DualCE6 | Fixed exit 6 | 13.378 | 13.349 | 13.406 | 13.425 |
| D-DualCE6 | Learned, threshold 0.8 | 15.565 | 15.439 | 15.458 | 15.375 |
| D-DualCE6 | Random, p=0.023819 | 15.307 | 15.302 | 15.284 | 15.258 |

## Evidence and coverage

The linked final JSON reports retain exact metrics, raw trial values, checkpoint
paths, recipe checks and source receipts. The tables above round only for display.
All substantive completed follow-up studies through the stated observation time
are included; smoke/acceptance checks are supporting evidence, not additional
research arms. No downstream evaluation of the STOP-adapted 11k/15k/20k checkpoints
is present in these receipts; the downstream table covers the original 10k pair.

Earlier standalone technical reports cover
[GPU verification](BDOT_GPU_VERIFICATION_20261018.md),
[B200 verification](BDOT_B200_VERIFICATION_20261018.md),
[FA4 resume](BDOT_FA4_RESUME_20261018.md),
[grouped GEMM](BDOT_GROUPED_GEMM_20261020.md),
[dispatch optimization](BDOT_DISPATCH_OPTIMIZATION_20261020.md), and
[FA2/FA4 environment selection](ATTENTION_BASELINE_LAUNCH_20261020.md).
FA4 remained selected after the three-runtime comparison; the numerical and
timing records are in [attention comparison](results/attention_comparison_20261020.json).
The later native training/resume completion is in the native baseline report,
superseding the comparison receipt's historical `training_resume_verification_pending` flag.

STOP correctness fixtures are retained in
[initial STOP verification](results/bdot_stop_verification_20261023.json),
[STOP-v2 local checks](results/bdot_stop_v2_local_20261024.json), and
[long-continuation checks](results/bdot_stop_long_local_20261024.json).

These small-model results support a near-match for the selected partial router
and a real training-execution improvement. They do not establish an equal-compute
quality advantage, statistical equivalence, downstream gains for STOP, or a
general impossibility of early exits. The tested learned STOP configurations
have not produced a useful decoding-speed gain at the required quality.
