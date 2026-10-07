# Proxy arms: perplexity and downstream results

Snapshot: 8 October 2026. This summarizes the completed **document-isolated P1–P7 screen**, including the chat-designed variants. B200 checked at 04:48 Singapore: all STS-B/BoolQ numerical and smoke/reload checks passed; 27 of 32 production fine-tuning fits are complete, with P6/BoolQ seed 44 running. Final test-set evaluations and scores remain pending.

## Setup and interpretation

Qwen3-0.6B architecture trained **from scratch**, English `cx_sampled_old`, seed 42. Each run stops at 2,500 updates: 2.62144B input tokens, 1,048,576 tokens/update, sequence length 2,048. EOS separates documents; attention is document-isolated and RoPE positions reset per document. Full schedule: 28,600 updates, 1,430 warmup updates. FA4 unless explicitly labeled SDPA.

LM loss is token-weighted validation NLL, excluding auxiliary loss; **PPL = exp(LM loss)**. Validation contains 4,882 packed sequences and 9,981,660 eligible next-token targets. Lower loss/PPL is better; higher downstream scores are better. Comparisons match training tokens, not training time. One pretraining seed does not establish robustness.

## What the arms change

Block numbers are 1-based. Unless stated otherwise, proxies occur at blocks 2, 4, …, 24. A four-block MLP target means the sum of MLP outputs from the current block through the next three, standardized per channel and detached. Auxiliary weight ramps from 0 to 0.1 over 250 updates.

**Names:** `-iso` means isolating the predictor from **LM gradients**, not document isolation (all these runs already isolate documents). `-block` in P1/P3 means blocking **auxiliary gradients into the backbone**; LM gradients can still train their predictors. `-4h` means four **query heads**, not four KV heads: query heads 12–15 use KV groups 6–7 (zero-based). P7 and all its variants use this same four-head allocation; the other twelve heads retain native attention.

The gradient descriptions below refer to pretraining. Supervised fine-tuning deliberately allows task gradients into the predictors, as described in its results section.

| Arm | Design / difference from its parent |
|---|---|
| A | Vanilla Qwen3-0.6B architecture, no proxy or auxiliary loss. |
| P1-block | A tokenwise 1024→256→1024 SiLU predictor estimates the four-block MLP target. Add its gated, RMS-normalized output to the input of K/V projections for four query heads (two KV groups). Gate starts at 0. Auxiliary gradients train only the predictor; LM gradients also train the predictor. |
| P1-block, α-init=1 | Exactly P1-block with the gate initialized to 1 instead of 0; this is a fresh training run, not an evaluation-time multiplier. |
| P3-block | P1-style K/V injection, but uses three document-reset exponential moving summaries (decays 0.5/0.9/0.99) to predict summaries of deeper residual increments. Gate starts at 0; auxiliary gradients are blocked from the backbone. |
| P4 | Independent tokenwise predictors of the four-block MLP target. Add normalized predictions to the **value-projection input only**, across all heads; queries/keys remain native. Gate starts at 1. LM and auxiliary losses both train the predictors. |
| P4-iso | P4 with LM gradients stopped at the predictors; only auxiliary loss trains them. Gates and native projections still learn from LM loss. |
| P4-4h / P4-iso-4h | Corresponding P4 parent, but modify values for only the last **four query heads** (two KV groups), instead of all sixteen. Predictor size and auxiliary objective are unchanged. |
| P5 | P4-style all-head value injection, but replace independent predictors with a width-256 residual stream carried between proxy blocks. The stream learns only from auxiliary loss. |
| P6 | Same tokenwise predictor and target as P4, but add the RMS-normalized prediction **directly to the residual stream before the block**. Scale it by detached input RMS and a trainable channel gate initialized to 0.1. LM and auxiliary losses both train the predictors. |
| P6-iso | Exactly P6 with predictor LM gradients stopped. Auxiliary loss trains the predictors; LM loss still trains the gates and backbone. |
| P7 | Preserve native K/V entries and add separate proxy K/V entries. Four query heads attend jointly to both through **one softmax**, with no gate. Predictor uses a width-256 bottleneck and a causal four-tap convolution. Target is the four-block residual increment (attention + MLP contributions). Auxiliary objective is cosine + 0.5 × relational KL. Predictor learns only from auxiliary loss; LM trains proxy projections and backbone. |
| P7-mlp | P7 with the target changed to the four-block **MLP sum**. It retains the convolution and relational KL; the name does not mean those were removed. |
| P7-ems | P7 plus document-reset exponential moving summaries of the width-256 input features, at decays 0.9 and 0.99. Add both summaries to the convolution output before SiLU. Keep P7's target, attention and auxiliary losses. |
| P7-kq | P7 with proxy **keys**, but reuse native values for the proxy entries. No separate proxy-value projection. Queries remain native. |
| P7-simple | P7's same four-head joint native/proxy attention, but a **tokenwise 1024→256→1024 SiLU predictor, no convolution/EMS, four-block MLP-sum target, cosine-only auxiliary loss**. No relational KL or gate. Predictor is auxiliary-only; LM trains the separate proxy K/V projections and backbone. |

### P7-simple compared directly with original P7

Let `u_l` be the RMS-normalized input to block `l`, `h_l` its residual-stream output, and `m_l` its MLP output. `sg` means stop-gradient; `N` is the existing running per-channel target standardization and clipping.

| Component | Original P7 | P7-simple |
|---|---|---|
| Predictor | Project `sg(u_l)` to width 256, causal depthwise convolution over the current and previous three tokens, SiLU, project to width 1024. Convolution respects document boundaries. | `p_l = W2 SiLU(W1 sg(u_l))`, bias-free 1024→256→1024. Each predictor sees only the current token's block input; that input already contains earlier-layer context. |
| Target | `sg(N(h_(l+3) - h_(l-1)))`: the four-block increment, including attention and MLP contributions. | `sg(N(m_l + m_(l+1) + m_(l+2) + m_(l+3)))`: only the four MLP contributions. |
| Training objective | `LM + λ × (cosine_loss + 0.5 × relational_KL)` | `LM + λ × cosine_loss`; remove relational KL and its query sampling. |
| Auxiliary schedule | `λ = 0.1 × min(step/250, 1)` | Unchanged. |
| Attention | Independent proxy K/V projections from the RMS-normalized prediction; four heads jointly attend to native and proxy entries in one softmax. No additive gate. | Unchanged. Native entries remain available; proxy entries do not replace them. |
| Gradient routing | Auxiliary loss trains the predictor only. LM loss trains the backbone and proxy K/V projections, with the predictor output detached at that boundary. | Unchanged. |
| Placement / visibility | Blocks 2, 4, …, 24; a query at token `t` sees native and proxy entries at `j ≤ t` in the same document. | Unchanged, including document-reset RoPE positions. |

The deeper targets are computed during the ordinary training forward pass and used only for auxiliary supervision. At evaluation, attention consumes the **prediction**, not the actual deeper target. No separate teacher model is added.

Relative to **P4-iso-4h**, P7-simple keeps the predictor, target and cosine loss, but consumes the estimate as additional attention entries instead of adding it to the value-projection input. It adds independent K/V projections and removes P4's additive gate, so this is not a parameter-matched comparison. Relative to **P7-mlp**, P7-simple additionally removes the convolution and relational KL.

Definitions: [P1/P3 spec](proxy_heads_P1_P3_spec_v3.md), [P4–P6 implementation](P4_P5_P6_IMPLEMENTATION.md), [P7 implementation, including P7-simple](P7_IMPLEMENTATION.md).

## Validation loss and perplexity

Main completed screen. Runtime is Trainer time including evaluation/checkpointing, excluding startup and separate acceptance tests.

| Arm | LM loss ↓ | PPL ↓ | Time (min) |
| --- | ---: | ---: | ---: |
| A | 3.477173 | 32.3681 | 88.30 |
| P6-iso | 3.466300 | 32.0180 | 96.48 |
| P7-simple | 3.474505 | 32.2819 | 102.02 |
| P7 | 3.476931 | 32.3603 | 110.91 |
| P7-mlp | 3.477345 | 32.3736 | 111.76 |
| P7-ems | 3.477727 | 32.3860 | 111.72 |
| P7-kq | 3.477798 | 32.3883 | 110.88 |
| P6 | 3.478265 | 32.4034 | 99.93 |
| P4 | 3.482287 | 32.5340 | 97.30 |
| P5 | 3.483880 | 32.5859 | 95.85 |
| P4-iso | 3.489673 | 32.7752 | 95.55 |

Earlier completed 2,500-step runs on the same isolated-document validation recipe. Downstream evaluations below did not include these variants.

| Arm | LM loss ↓ | PPL ↓ |
| --- | ---: | ---: |
| A (dense SDPA) | 3.477941 | 32.3930 |
| P1-block | 3.477280 | 32.3716 |
| P3-block | 3.478909 | 32.4243 |
| P1-block, α-init=1 | 3.482319 | 32.5351 |
| P4-4h | 3.479081 | 32.4299 |
| P4-iso-4h | 3.479564 | 32.4456 |

Sources: [main comparison](../artifacts/proxy-remaining-results-20261007-a03/training/seed-42/comparison.json), [main report](PROXY_REMAINING_RESULTS_20261007.md), [earlier run records](PROJECT_NOTES.md). PPL above is calculated from the stored full-precision LM loss, not the rounded table value.

## Zero-shot evaluation

All eleven main-screen checkpoints, nine English tasks, no fine-tuning or demonstrations. Scores are percentages. Primary score uses `acc_norm` where available (ARC-C/E, Belebele, HellaSwag, PIQA), otherwise `acc`. Mean is an equally weighted descriptive average, not an official benchmark composite.

| Arm | ARC-C | ARC-E | Belebele | HellaSwag | PAWS-X | PIQA | WinoGrande | XNLI | XStoryCloze | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 21.50 | 37.29 | 22.89 | 29.18 | 50.25 | 61.70 | 51.93 | 39.36 | 55.92 | 41.113 |
| P6-iso | 21.42 | 36.95 | 23.11 | 29.13 | 49.30 | 61.37 | 51.14 | 39.84 | 55.00 | 40.807 |
| P7-simple | 21.93 | 37.12 | 23.11 | 28.99 | 47.40 | 61.32 | 50.99 | 42.33 | 55.72 | 40.990 |
| P7 | 21.50 | 36.95 | 23.00 | 29.50 | 49.35 | 61.59 | 51.14 | 37.55 | 55.13 | 40.635 |
| P7-mlp | 21.93 | 36.62 | 23.00 | 28.98 | 46.85 | 61.70 | 51.54 | 39.60 | 55.53 | 40.637 |
| P7-ems | 21.16 | 36.78 | 23.11 | 29.09 | 50.55 | 61.26 | 51.85 | 39.96 | 55.46 | 41.026 |
| P7-kq | 21.67 | 36.99 | 23.00 | 29.40 | 46.30 | 61.86 | 51.78 | 41.69 | 54.86 | 40.839 |
| P6 | 21.93 | 36.41 | 22.67 | 29.08 | 52.00 | 61.81 | 50.28 | 40.72 | 54.47 | 41.039 |
| P4 | 21.33 | 36.20 | 22.89 | 29.21 | 47.55 | 61.32 | 50.99 | 41.61 | 54.93 | 40.668 |
| P5 | 21.16 | 37.04 | 23.00 | 29.27 | 48.65 | 60.88 | 50.43 | 41.65 | 54.53 | 40.734 |
| P4-iso | 21.59 | 36.78 | 23.11 | 28.75 | 47.70 | 60.72 | 50.83 | 42.61 | 54.80 | 40.765 |

Raw accuracy for the tasks that also expose normalized accuracy (other task scores are unchanged). Raw mean uses `acc` across all nine tasks.

| Arm | ARC-C | ARC-E | Belebele | HellaSwag | PIQA | Raw mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 17.66 | 39.90 | 22.89 | 28.26 | 62.35 | 40.947 |
| P6-iso | 18.60 | 39.81 | 23.11 | 28.07 | 62.08 | 40.773 |
| P7-simple | 17.41 | 39.98 | 23.11 | 27.94 | 61.53 | 40.713 |
| P7 | 17.83 | 40.07 | 23.00 | 28.16 | 62.35 | 40.510 |
| P7-mlp | 18.52 | 39.77 | 23.00 | 28.18 | 62.30 | 40.587 |
| P7-ems | 17.06 | 38.72 | 23.11 | 28.02 | 62.35 | 40.788 |
| P7-kq | 16.81 | 40.99 | 23.00 | 28.04 | 61.92 | 40.598 |
| P6 | 18.09 | 38.93 | 22.67 | 28.07 | 62.30 | 40.836 |
| P4 | 18.09 | 38.72 | 22.89 | 28.17 | 61.59 | 40.504 |
| P5 | 17.06 | 39.81 | 23.00 | 27.88 | 63.00 | 40.670 |
| P4-iso | 17.41 | 40.11 | 23.11 | 28.11 | 61.53 | 40.690 |

Source: [validated full zero-shot metrics](../artifacts/downstream-monitor-20261007-a08/full-validation.json). XNLI here uses 2,490 English development examples; the supervised table uses the separate 5,010-example test set.

## Few-shot evaluation

Original step-2,500 checkpoints, no weight updates. Demonstrations are identical across arms. HellaSwag uses 10 shots, ARC-C 25, the other six tasks 5. Belebele is excluded because its available demonstration pool was the test split. Primary metrics follow the zero-shot table.

| Task | Shots | A | P6 | P6-iso | P7-simple |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC-C | 25 | 22.782 | 22.355 | 21.416 | 22.611 |
| ARC-E | 5 | 36.532 | 35.985 | 36.827 | 35.522 |
| HellaSwag | 10 | 29.297 | 29.138 | 29.257 | 29.227 |
| PAWS-X | 5 | 49.450 | 51.050 | 49.250 | 50.550 |
| PIQA | 5 | 61.153 | 61.045 | 61.045 | 61.317 |
| WinoGrande | 5 | 50.987 | 49.645 | 51.855 | 49.882 |
| XNLI | 5 | 38.394 | 36.426 | 37.068 | 37.912 |
| XStoryCloze | 5 | 54.136 | 54.666 | 54.600 | 54.600 |
| Mean | — | 42.841 | 42.539 | 42.665 | 42.702 |

Raw accuracy for tasks with a different normalized score; remaining task scores are unchanged.

| Task | A | P6 | P6-iso | P7-simple |
| --- | ---: | ---: | ---: | ---: |
| ARC-C | 17.065 | 17.747 | 18.089 | 18.345 |
| ARC-E | 38.721 | 38.047 | 40.530 | 38.089 |
| HellaSwag | 28.062 | 27.863 | 28.092 | 27.644 |
| PIQA | 61.806 | 61.643 | 61.752 | 62.459 |
| Raw eight-task mean | 42.328 | 42.136 | 42.654 | 42.435 |

This eight-task mean is not directly comparable with the nine-task zero-shot mean. Sources: [5-shot](../artifacts/fewshot-monitor-20261008-a01/full-5shot-validation.json), [10-shot](../artifacts/fewshot-monitor-20261008-a01/full-10shot-validation.json), [25-shot](../artifacts/fewshot-monitor-20261008-a01/full-25shot-validation.json), [protocol](DOWNSTREAM_EVAL_20261007.md).

## Supervised fine-tuning

Each fit starts from its original step-2,500 pretraining checkpoint. Full model/task head fine-tuning, auxiliary loss disabled; predictors also receive task gradients, including P6-iso and P7-simple. PAWS-X: 3 epochs. NLI: 1 epoch on English MultiNLI, evaluated on English XNLI. Learning rate selected on development data from 1e-5/3e-5; every arm/task selected 3e-5.

Test accuracy (%), mean ± sample standard deviation over fine-tuning seeds 42/43/44. These are three fine-tuning seeds sharing one pretraining seed.

| Arm | PAWS-X | English XNLI |
| --- | ---: | ---: |
| A | 91.850 ± 0.278 | 78.922 ± 0.111 |
| P6 | 90.300 ± 0.218 | 79.188 ± 0.579 |
| P6-iso | 90.717 ± 0.751 | 78.955 ± 0.445 |
| P7-simple | 90.917 ± 0.029 | 78.696 ± 0.306 |

Per-seed test accuracy (%), in seed order 42 / 43 / 44.

| Arm | PAWS-X | English XNLI |
| --- | ---: | ---: |
| A | 92.100 / 91.550 / 91.900 | 78.802 / 78.942 / 79.022 |
| P6 | 90.550 / 90.150 / 90.200 | 78.523 / 79.581 / 79.461 |
| P6-iso | 91.150 / 89.850 / 91.150 | 78.443 / 79.242 / 79.182 |
| P7-simple | 90.900 / 90.900 / 90.950 | 78.583 / 79.042 / 78.463 |

Sources: [A/P6/P7-simple results](../artifacts/finetune-monitor-20261007-a07/summary.json), [P6-iso results](../artifacts/p6iso-finetune-monitor-20261007-a03/summary.json), [fine-tuning protocol](SUPERVISED_FINETUNING_20261007.md).

**STS-B and BoolQ: no final results.** Two acceptance attempts stopped before fitting. The first used BF16 SDPA as the reference; the second exposed sensitivity of relative raw logits from a random two-class head. The corrected gate uses FP32 math SDPA, bounds classification probability/loss differences, and retains hidden-state/gradient checks and the regression output check. The fresh retry is launched with the training model and recipe unchanged. All eight numerical gates and all eight smoke/save/reload validations passed. At 04:48 Singapore, 27 of 32 production fits were complete; P6/BoolQ seed 44 was running. All eight development selectors chose 3e-5. The 24 final evaluations have not started. [Diagnosis and acceptance criteria](SUPERVISED_FINETUNING_20261007.md#task-output-gate-correction--8-october-2026).

## Evaluation-only proxy ablations

Disabling proxies in an already-trained model measures reliance on that branch; it does not compare against a separately trained A. The following use the same checkpoints as the main PPL table.

| Arm | Normal LM loss | Proxy-disabled LM loss | Proxy-disabled PPL |
| --- | ---: | ---: | ---: |
| P6-iso | 3.466300 | 3.604318 | 36.7566 |
| P7-simple | 3.474505 | 3.612829 | 37.0708 |
| P7 | 3.476931 | 3.599382 | 36.5756 |
| P7-mlp | 3.477345 | 3.622096 | 37.4159 |
| P7-ems | 3.477727 | 3.627387 | 37.6144 |
| P7-kq | 3.477798 | 3.602408 | 36.6865 |
| P6 | 3.478265 | 3.643162 | 38.2125 |
| P4 | 3.482287 | 3.804142 | 44.8867 |
| P5 | 3.483880 | 3.657854 | 38.7780 |
| P4-iso | 3.489673 | 3.627675 | 37.6252 |

Earlier P1/P3 evaluation-time gate multipliers (no retraining). Multiplier 0 disables the proxy.

| Gate multiplier | P1-block loss | P1-block PPL | P3-block loss | P3-block PPL |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.477273 | 32.3713 | 3.479060 | 32.4292 |
| 1 | 3.477280 | 32.3716 | 3.478909 | 32.4243 |
| 2 | 3.477290 | 32.3719 | 3.478983 | 32.4267 |
| 5 | 3.477309 | 32.3725 | 3.480716 | 32.4830 |

Sources: main comparison above; [P1 gate sweep](../artifacts/proxy-gate-sweep-20261005/supervised/run/P1-block/summary.json), [P3 gate sweep](../artifacts/proxy-gate-sweep-20261005/supervised/run/P3-block/summary.json).

## What the results currently support

- **P6-iso has the lowest validation PPL** (about 1.08% below A), with about 9.3% more Trainer time. P7-simple is second, with a smaller gain and greater runtime overhead.
- **A has the highest primary zero-shot and few-shot averages** (normalized accuracy where available). P6-iso has the highest raw-accuracy few-shot average, so that ranking depends on the metric. Fine-tuning has not shown a consistent proxy advantage: A wins PAWS-X; P6's small XNLI gain is within the observed fine-tuning-seed spread.
- Proxy-disabled degradation shows that models use their branches. It does not establish benefit over A, significance, or superiority at equal training time.
- Two corresponding P7 variants are **implemented, not trained**: `P7-simple-sparse` uses the same six locations/four-block target as P6-iso-sparse; `P7-simple-short` uses the same twelve locations/two-block target as P6-iso-short. Both retain P7-simple's four-head joint native/proxy attention, isolated tokenwise predictor and cosine-only loss. [Definitions](P7_IMPLEMENTATION.md#p7-simple-variants-based-on-p6-iso-8-october-2026).
- Two P6-iso variants are **implemented, not trained**, with no results: `P6-iso-sparse` uses six injection locations `{2, 6, 10, 14, 18, 22}` with the original four-block target; `P6-iso-short` retains all twelve locations and uses `m_l + m_(l+1)`. These are separate experiments; all other P6-iso settings remain the same. [Implementation and usage](P4_P5_P6_IMPLEMENTATION.md#p6-iso-follow-ups-sparse-placement-and-shorter-targets). V1/V3 and uncompleted P1/P3 ablations are not assigned scores.

Earlier B/F/G and task-/consumer-aware experiments used different training/attention protocols and are not pooled into these tables. Their records remain in [PROJECT_NOTES.md](PROJECT_NOTES.md); pretrained-weight probes are separate from this from-scratch screen.
