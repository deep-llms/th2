# P7 longer runs and five controlled variants

**Authorized and running as of 15 October, 13:28 Singapore time.** The user
subsequently requested launch. All numerical and distributed smoke gates passed;
A was at step 238/7,500 on all eight B200 GPUs. Active run:
`q359-p7-exploration8-20261015-a03`. `commands.sh` is inactive after submission;
the independent supervisor continues the queue and restores/verifies burns on
exit. Earlier preparation-only statements below describe the pre-launch review.

Two startup issues were corrected before any production training: the runner's
source copy has no `.git`, and the frozen diagnostic rows use `{index,data}`
envelopes. No training/data cache was reused from either failed attempt. The
second attempt restored verified communicating burns before the fresh retry.

CUDA numerical evidence is in
[the saved gate results](results/p7_exploration_numerics_20261015.json). Value-only
merged vs original attention passed on both backends: full-gradient relative L2
0.498% (SDPA), 0.484% (FA4); predictor-gradient relative L2 0.395%/0.385%.
All eight real Trainer smokes and the original training-data/recipe comparison
passed. A's early steady updates are about 1.058 s; full new-arm throughput is
still pending. Keep the provisional 11–13-hour whole-queue estimate.

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
  ablation, not independently addressable additional memory. The implementation
  now averages the selected values and uses one native-size attention call for
  all heads. See the numerical qualification below.
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

## Review: which experiments are worth their cost?

These are controlled tests of different explanations, not eight equally
promising candidates. My priority among the five short variants is:

| Priority | Variant | Why run it / limitation |
|---|---|---|
| 1 | LM-only, no auxiliary loss | Most direct test of whether hidden matching helps or conflicts with LM learning. A win alone would support extra LM-trained capacity, not useful deep-state prediction. |
| 2 | Sparse | Tests whether dense intervention is counterproductive and halves predictor sites. Any win also changes capacity and cost; it does not isolate overlap alone. |
| 3 | Key-only | Tests learned routing over existing native content. Distinct from predicting additional value content. |
| 4 | Value-only | Useful content control, now with cheaper attention. Duplicate keys reduce it to a fixed half-native/half-proxy value mixture, so it is not an independently addressable memory method. |
| 5 | Next-token | Tests direct predictive supervision instead of matching a hidden vector. Lowest confidence: the detached readout is itself learning from scratch, its feature space may poorly suit shallow memory, and CE weight 0.1 is not scale-matched to cosine weight 0.1. |

The longer matched A/targetgrad/LM comparison is defensible as a single bounded
test of training horizon. Targetgrad has small same-direction downstream gains
across two seeds, not an established reliable win; LM is a useful utility-trained
reference, not the best observed downstream method. A longer run might help or
erase the difference. Do not commit to another sweep solely because one seed
or one benchmark improves. Inspect PPL, the existing downstream aggregate, and
actual time together. These eight runs do not cover every cross-product of
routing, loss and density, and should not expand automatically.

The prepared manifest's order and eight-run scope are unchanged. If the budget
must shrink, retain noaux and sparse before the more speculative short variants.
No new run has been authorized or launched by this review.

## Value-only throughput change and numerical limits

With identical keys and identical masks for the two copies, zero-dropout
attention satisfies, in real arithmetic:

`softmax([QK, QK]) [V; Vp] = softmax(QK) ((V + Vp) / 2)`.

Only the selected two KV groups use the mean; every other group retains its
native values. The keys remain live: the gradient to the single native key
equals the sum of gradients through its two copies. No detach, parameter,
initialization, optimizer group, objective, data order or schedule changes.
The disabled-proxy baseline path is unchanged.

Previously each proxy site used native-head attention plus a doubled-entry
selected-head attention. The new value-only implementation uses one ordinary
attention over all heads, on both SDPA and FA4. At 12 layers/four sites this
reduces attention calls from 16 to 12 and eliminates doubled FA4 queries at
those sites. MAC estimates and runtime receipt validation reflect that change.
Saved `memory_entries` metadata is version 2 with
`implementation=average_values_single_attention`, so the earlier prototype is
not silently treated as identical metadata during resume/report checks.

The average is formed in FP32 and cast back to the attention dtype. Algebraic
equivalence does **not** promise bitwise BF16 equivalence: rounding and backend
reduction order change. The original duplicate-entry path remains available
as a numerical reference. CPU tests compare independent attention equations,
full-model outputs, backbone/predictor gradients, BF16, checkpoint replay,
document isolation, save/reload, actual Trainer resume and two-rank training.
The prepared CUDA gate additionally compares the merged implementation against
the original calculation separately for **both** backends, including predictor
gradients. It uses the existing limits (loss difference <0.01; gradient relative
L2 <3%; hidden/logit relative L2 <2%) and fails the queue before production.

This is a reduction in redundant work, **not a measured B200 throughput gain**.
CUDA acceptance and whole-update timing remain pending. No custom kernel or
environment change was needed. Sparse sites and removed key/value projections
already avoid unused work. Noaux/next-token already omit cosine backward;
next-token computes one sampled readout at only the last site. Keep their
existing detached diagnostics for interpretability. Do not extend targetgrad's
compile/custom-kernel flags to these LM-gradient routes without separate checks.

Review validation (local CPU, 15 October): 31 relevant tests passed: eight
exploration tests (140.985 s), four queue tests plus one artifact validator and
six consumer regressions, four combined-route regressions (31.698 s), and eight
existing speed-path regressions (23.615 s). The initial regression command
misspelled `tests.test_p7_combined` as `tests.test_proxy_combined`; its 11 actual
tests passed, and the corrected four-test module passed separately. This was
a test-collection error, not a model failure. Logs:
`temp/p7-exploration-review-{full,regression,combined,speed-regression}-20261015.log`.
Python compilation and `git diff --check` passed. These are targeted tests, not
a claim that every repository test or real CUDA execution has passed.

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
