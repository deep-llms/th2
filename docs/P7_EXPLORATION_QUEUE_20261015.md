# P7 longer runs and five controlled variants

**Prepared only. The user explicitly requested no launch.** `commands.sh` stays
at `#0`. No B200 work, burn reclamation, environment installation or remote push
is part of this preparation.

## Schedule correction and queue

The previous 12-layer checkpoints completed a **2,500-step** cosine schedule,
with 125 warmup steps and a 10% minimum LR. They were not early-stopped from a
10k/full-data schedule. Extending that horizon would require a deliberate LR
transition; it is not an ordinary identical-schedule resume.

The prepared alternative is three **fresh 7,500-step** runs with 375 warmup
steps, followed by five fresh 2,500-step runs with 125 warmup steps. All retain
the cosine schedule's 10% minimum LR. This gives the two existing P7 methods
more training while staying near the requested overnight budget.

| Order | Label / arm | Steps | Auxiliary objective |
|---|---|---:|---|
| 1 | `a-long` / A | 7,500 | None |
| 2 | `targetgrad-long` / P7-simple-targetgrad | 7,500 | Cosine, live target, λ=0.1 |
| 3 | `lm-long` / P7-simple-lm | 7,500 | Cosine, detached target, λ=0.1; LM trains predictor |
| 4 | `lm-sparse` / P7-simple-lm-sparse | 2,500 | Parent cosine, λ=0.1 |
| 5 | `lm-keyonly` / P7-simple-lm-keyonly | 2,500 | Parent cosine, λ=0.1 |
| 6 | `lm-valueonly` / P7-simple-lm-valueonly | 2,500 | Parent cosine, λ=0.1 |
| 7 | `lm-noaux` / P7-simple-lm-noaux | 2,500 | None; λ is fixed at zero |
| 8 | `lm-nexttoken` / P7-simple-lm-nexttoken | 2,500 | Last-proxy next-token CE, λ=0.1; no cosine objective |

Shared: 12 layers initialized from scratch, seed/data seed 42, proxy seed 43,
eight GPUs per training job, microbatch 16, accumulation 4, sequence 2,048,
1,048,576 input tokens/update, English data, EOS boundaries, document isolation,
reset RoPE and FA4. `train.py`, HF Trainer and Accelerate remain the training
entry point. Keep every checkpoint at 250-step intervals. The auxiliary ramp
remains the existing 250 updates, independently of LR warmup.

Only targetgrad enables its four validated optimization flags. Those flags
remain rejected for other P7 arms; their CUDA acceptance has not been extended.

Using the measured ~1.05 s/update for A, ~1.087 for optimized targetgrad and
~1.17 for the previous LM variant, the three longer runs take roughly 6.9 hours
of training updates. Five 2,500-step variants at ~1.1–1.25 s/update add roughly
3.8–4.3 hours. Validation, compilation, checkpoints and downstream evaluation
are additional. **Budget roughly 11–13 hours provisionally**, rather than
promise a hard 12-hour finish: the new variants have not been timed on B200.

## What each new arm tests

The parent P7-simple-lm predicts a hidden-size vector from a detached shallow
input through its existing two-layer SiLU MLP. The main LM loss trains the
predictor through its attention memory entries. Cosine matching trains it
against the detached, standardized sum of four MLP outputs. Neither auxiliary
route trains the shallow input or deep target. There is no separate LM.

All five variants retain this architecture and routing except for the explicit
change below. They retain four selected query heads in two complete KV groups.
Native and memory entries share one causal, document-isolated softmax. P7
allows the current token's causal prediction; no future hidden state is an input.

- **Sparse:** proxy sites 2 and 6 instead of 2, 4, 6 and 8. Four-block target
  windows become non-overlapping. Retained parameters initialize exactly like
  the parent at the same sites. Tests whether fewer interventions are better.
- **Key-only:** memory entries use learned proxy keys and the corresponding
  native values. No proxy value projection is retained. Tests learned routing
  without new value content.
- **Value-only:** memory entries use the corresponding already-rotated native
  keys and learned proxy values. Proxy key projection/norm are removed. With
  duplicate native keys, joint attention is mathematically equivalent to
  attending to `(native_value + proxy_value)/2`. This is a controlled content
  ablation, not independently addressable additional memory.
- **No auxiliary loss:** exactly the parent's forward architecture, trained
  solely by the main LM loss. Cosine/target statistics remain detached
  diagnostics; the returned auxiliary objective and logged auxiliary loss are
  zero. Tests whether matching the deep target helps beyond LM supervision.
- **Next-token:** same forward architecture; replaces matching with a direct
  next-token objective on the **last proxy site only** (block 8 at this depth).
  Read out the prediction through the backbone's current final RMSNorm and
  vocabulary matrix, detached for this loss. No auxiliary gradient reaches
  either readout parameter or the backbone. Those parameters still train
  normally through the main LM loss. Earlier predictors receive main-LM
  supervision; only the final predictor receives this additional CE.

For next-token CE, choose packed offsets `t=0,8,16,... < T-1`, and predict
`labels[t+1]`. Exclude cross-document transitions, padding and ignored labels
using the main LM's eligibility rule. CE is normalized by the **global selected
target count across ranks and accumulation microbatches**, not by all input
tokens. Fixed stride avoids extra sampling RNG and limits vocabulary work to
about one eighth of another final readout. Compute it in the existing LM chunk
size; honor auxiliary checkpointing. No next-token head or loss runs during
ordinary logits-only inference.

The next-token arm is a hypothesis, not a guarantee of useful deep information.
Its comparison against noaux asks whether direct predictor supervision helps;
comparison against the parent asks whether it is better than hidden matching.
Cosine remains available as a diagnostic, explicitly separate from CE metrics.

## Comparisons and automatic checks

Compare the three 7,500-step endpoints with one another. Compare the short
variants with **existing seed-42, full-2,500-step A and P7-simple-lm**. The queue
checks matching recipe/data fingerprints before publishing those comparisons.
Do not compare a long-schedule step-2,500 checkpoint with a short-schedule
endpoint as if their LR histories matched.

The queue has 38 stages: one seven-arm CUDA numerical gate; eight three-update
eight-GPU smokes and their validators; a recipe/data check against the original
control before production; eight full training jobs and their
validators; one matched-summary stage; downstream numerical, smoke and full
stages. Any failed stage stops subsequent work. Smoke directories are separate
and never supply production weights. No existing outputs or data caches are
deleted. New output roots are mandatory.

Downstream evaluation covers all eight new endpoints using the existing nine
benchmarks and frozen shots: HellaSwag 10; ARC-C 25; PIQA/ARC-E/WinoGrande/BoolQ
5; SciQ/LAMBADA/BLiMP 0. Independent evaluation jobs share the eight GPUs via
the existing concurrent harness runner. Numerical checks now use each
checkpoint's declared step, allowing 7,500 and 2,500 in the same scope.
Existing short-control downstream results can be reused under that same
protocol. No new fine-tuning or paired-loss sweep is included.

Future authorized launch uses the existing supervisor: require the previous
job's successful completion **and** verified burn handoff; verify environment
and local inputs; copy/check Accelerate and run `accelerate env`; inspect GPU
ownership; reclaim only identified approved burns; verify free GPUs before
training; restore and verify communicating burns on success or failure.
Require 2,000 GiB free for checkpoint retention. Nothing in local queue
generation stops GPU processes or starts training.

## Code and local preparation

- Architecture/routing: `deep_kv/__init__.py`, `deep_kv/proxy.py`,
  `deep_kv/proxy_memory.py`.
- CE normalization and logs: `deep_kv/proxy_training.py`.
- Saved metadata and comparisons: `deep_kv/report.py`.
- Queue: `scripts/p7_exploration_queue.py`; future authorized launcher:
  `scripts/launch_p7_exploration.sh`.
- Tests: `tests/test_p7_exploration.py`, `tests/test_p7_exploration_queue.py`.

Offline preparation only:

```bash
python -m scripts.p7_exploration_queue make \
  --recipe proxy_heads.12layer.b200.json \
  --output temp/p7-exploration-preview/jobs.json \
  --rows /mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01/rows.json \
  --reference /mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01
python run_experiments.py --config temp/p7-exploration-preview/jobs.json --list
```

The local manifest is a preview. Generate the executable manifest on B200
through the prepared launcher after authorization, so paths point to that
machine's checkout. The launcher preserves the previous NCCL/offline exports
and does not install anything or download from the GPU node.

## Validation record

Seven new model test groups passed locally: independent joint-attention and CE
oracles, gradient routing, initialization and parameter budgets, document
isolation, FP32/BF16 and activation-checkpoint equivalence, exact Trainer
save/resume, standalone reload/reporting, correct accumulation denominators,
and real two-rank Gloo Trainer jobs for **all five** new arms. Six groups passed
in 138.347 s; the added BF16 group passed in 1.687 s.

A broader 32-test run passed 28 tests, including queue validation, existing
confirmation workflow, optimized targetgrad checks, shared computation and
checkpoint/logits evaluation. Four harness-dependent tests were blocked by
missing local `lm_eval`; no dependency was installed. Logs:
`temp/p7-exploration-tests-20261015.log`,
`temp/p7-exploration-bf16-20261015.log`, and
`temp/p7-exploration-regression-20261015.log`.

The local 38-stage preview passes `run_experiments --list`; shell syntax and
`git diff --check` pass. **New-arm compiled CUDA numerics, eight-GPU smokes,
actual memory/speed and B200 downstream checks are still pending.** They are
mandatory queued gates before full training, not claimed local successes.

Existing consumer and combined-target/LM parent regressions also passed: **10
tests in 121.486 s**, including their real Trainer and distributed checks. Log:
`temp/p7-exploration-parent-regression-20261015.log`.

Final queue checks passed **4 tests in 16.041 s**, including early rejection
of changed training fingerprints or LR settings. Log:
`temp/p7-exploration-queue-final-20261015.log`. Final preview:
`temp/p7-exploration-prepared-final-20261015/jobs.json` (38 stages).

The final checkpoint-validator/queue run passed **5 tests in 13.347 s**. It
exercises both backend receipt formats for every new arm, missing-rank-state
rejection, absent key/value projections, noaux coefficient enforcement, and
next-token metric validation. Log:
`temp/p7-exploration-validator-final-20261015.log`.
