# B-DOT pilot results — 20–21 October 2026

**Both routing curricula completed training but performed worse than the matched
dense control.** P1-A was slightly better than P1-B. Neither met the quality
requirement for expanding the research to the next phase.

**21 October update:** N-A and random-routing R both completed. N-A reduces
the deficit but still trails D. R slightly outperforms both learned routing
curricula; learned routing has not shown an advantage over chance in this pilot.
Checkpoint diagnostics now show severe module-load imbalance and insufficient
balancing strength in both learned routers. This supports testing the specified
balancing correction, but does not establish that it will improve quality.

The experiments follow [BDOT_v2_spec.md](BDOT_v2_spec.md), Stage 0a. Each model
uses Qwen3-0.6B dimensions with 12 layers: two fixed stem layers, eight middle
steps, and two fixed coda layers. All were trained from scratch, seed 0.

In routed models, each token chooses a middle module within one position of
that step's normal module. With reuse enabled, modules can be repeated, leaving
others unused; N-A forbids this.
The selected module supplies Q, Q normalization, output projection and MLP;
K/V and the attention-input normalization remain specific to the execution
step. Every token still executes eight middle steps. All arms use
step FiLM, a learned per-step scale and shift on the MLP input.

| Experiment | What it tests |
|---|---|
| **D: dense control** | Fixed module order throughout training, with FiLM and the same B-DOT implementation. This is the matched reference; the earlier native Qwen run used a different schedule. |
| **P1-A: narrow → wide** | Learns routing after an initial fixed-order phase. After 477 completed updates, the ±1 window opens. Random exploration starts at 5% and decays to zero by update 954. Uses hard top-1 routing. |
| **P1-B: soft → hard** | Opens the ±1 window immediately. Mixes the updates from two selected modules for the first 716 updates, then uses hard top-1. Exploration starts at 5% at that transition and ends at update 2,384. The soft phase adds compute. |
| **N-A: no reuse** | Same recipe and curriculum as P1-A, but every token must use each middle module exactly once. Tests ordering without repetition or omission. |
| **R: random routing** | Uniform random selection within the ±1 window from the start; reuse allowed, no curriculum. Chance-routing training control; unlike P1-A, it has no initial fixed-order phase. |

Shared recipe: **4,768 updates, approximately 2.5B input tokens**, sequence
length 1,024, global batch 524,288 tokens (microbatch 16 × accumulation 4 ×
8 B200 GPUs). HF Trainer/Accelerate, BF16 and FA4; peak LR 3e-4, 96 warmup
updates, cosine decay to 10% of peak. This is a complete pilot schedule.
The English pool is `nht10/cx_sampled`, subset `qwen3_0.6b_base_en_30B`, with
identical data order, EOS boundaries, document isolation and reset positions.

Final validation covers the same **9,704 sequences / 9,915,698 scored tokens**.
Loss is language-model NLL in nats/token, excluding router penalties; lower is
better. Runtime is Trainer elapsed time, including periodic evaluation/saving.

| Arm | Final loss | PPL | Loss difference vs D | PPL increase vs D | Runtime |
|---|---:|---:|---:|---:|---:|
| **D** | **3.49446** | **32.93** | — | — | 53m08s |
| P1-A | 3.61594 | 37.19 | +0.12148 | +12.92% | 58m20s |
| P1-B | 3.63270 | 37.81 | +0.13824 | +14.83% | 60m40s |
| N-A | 3.52600 | 33.99 | +0.03153 | +3.20% | 59m24s |
| R | 3.61016 | 36.97 | +0.11570 | +12.27% | 55m59s |

Both routed models continue learning. On the fixed 1,024-sequence periodic
validation set, losses decrease as follows (this smaller set differs from the
final validation above):

| Update | D | P1-A | P1-B |
|---|---:|---:|---:|
| 1,000 | 4.03478 | 4.11024 | 4.15954 |
| 2,500 | 3.65296 | 3.77093 | 3.84401 |
| 4,768 | 3.51469 | 3.63447 | 3.64903 |

Routing is active: final off-default choices are **45.21% for P1-A** and
**37.50% for P1-B**. Both concentrate work on a few modules, as quantified
below. Soft-start curriculum B did not resolve the quality deficit.

Before the pilots, **T12** passed eight-GPU checkpoint/resume comparisons on a
reduced-width 12-layer fixture, with identical routes, losses and final states.
**T14** compared native Qwen and fixed B-DOT without FiLM for 500 updates:
training/validation loss gaps were 0.00307/0.00411 nats, both below 0.01.
These checks support pipeline correctness without excluding every routing bug.

The pilot requires routed loss ≤ D + 0.01 nats; P1-A and P1-B fail that condition
by a substantial margin. N-A improves on P1-A by 0.08994 nats; its remaining
gap to D is 0.03153 nats, so it also fails this quality condition. This supports
investigating constrained routing if work continues, without isolating module
load balance as the cause. These are single-seed results at this scale.
R beats P1-A by 0.00578 nats and P1-B by 0.02254 nats, while trailing D by
0.11570 nats. Thus neither learned-routing curriculum beats the random-training
control here. R has no routing curriculum, so these comparisons do not isolate
learned selection from curriculum effects. Route interventions are now complete
(below). Downstream evaluation has not been run. Phase B remains unsupported.

## Checkpoint diagnostics — 21 October 2026

At the final checkpoint, own-route losses reproduce the saved results **exactly**.
The following interventions use the same full validation set and change routing
only, without retraining. These random-route evaluations are **not** the trained
random-routing control R in the first table.

| Arm | Own-route loss | Forced default loss | Random-route loss | Default − own |
|---|---:|---:|---:|---:|
| P1-A | 3.615941 | 3.745657 | 4.849492 | +0.129716 |
| P1-B | 3.632704 | 4.450526 | 4.743842 | +0.817822 |
| N-A | 3.525998 | 3.530833 | 4.054829 | +0.004835 |

Default routing hurts P1-A/B, consistent with modules adapting to their learned
paths. It does not show an advantage over training with fixed depth. N-A's
learned ordering improves its own default-path loss by only **0.004835 nats**.

Usage and router statistics below cover the first **4,096 frozen validation
sequences**, rather than the full loss population. There are eight middle
modules; balanced load would be 12.5% per module. Reuse rate is the percentage
of tokens repeating at least one module across their eight middle steps.

| Arm | Mean distinct modules/token | Reuse rate | Smallest / largest module load | Median raw-logit gap | Balance-bias range |
|---|---:|---:|---:|---:|---:|
| P1-A | 4.3910 | 99.8617% | 0.5491% / 32.5924% | 15.1621 | 8.5817 |
| P1-B | 5.0001 | 100.0000% | 0.0000% / 37.5000% | 44.6060 | 9.0157 |
| N-A | 8.0000 | 0.0000% | 12.5000% / 12.5000% | — | — |

Both P1 arms meet the response document's **weak-balancing condition**, at
updates 2,500 and 4,768: loads fall below half or above twice the uniform target,
while the median raw-logit gap exceeds the balance-bias range. P1-B's tiny final
z-loss still coexists with large gaps; controlling logsumexp does not prevent
losing candidates from becoming very negative.

Correctness checks found no illegal routes or nonunit gates in seven observed
checkpoint prefixes. Replaying the actual balance update on 476 logged count
vectors per arm gives the expected sign throughout. This is not an exact
historical per-update audit, because beta was not logged at every update.

Gradient checks require a qualification: direct full-LM finite differences at
the original saturated logits cannot resolve many tiny gradients. Stable local
gate checks pass at those logits. Conditioned full-LM checks, centering one
router at a time while retaining weights, routes and forward loss, pass **8/8
steps per arm** (maximum relative error **0.2147% / 0.0769%** for P1-A/B).
The initial failed probes and their limitations remain in the detailed report;
this is evidence for correct gradient machinery, not a conclusive full-LM check
at every original saturated logit.

There are no flagged within-phase validation spikes. P1-B's soft-to-hard
transition has a separately reported **+0.170650-nat jump**. Existing full-update
timing gives hard-routed P1-A **84.2% of native Qwen throughput**, passing the
60% requirement; this timing excludes evaluation and checkpoint saves.

**Decision:** Gate 0a still fails on quality. N-A remains outside D + 0.01, so
R_N is not triggered. The predeclared next candidate is **P1-A′**, selecting
with `argmax(softmax(z) + beta)` over allowed candidates instead of
`argmax(z + beta)`, otherwise retaining P1-A settings. It is **not implemented
or launched**. These single-seed results do not yet justify Phase B or longer
training.

Details and numerical limitations: [Step 1 report](BDOT_PILOT_DIAGNOSTICS_20261021.md).
Evidence: [diagnostic JSON](results/bdot_pilot_diagnostics_20261021.json),
[balancing snapshots](results/bdot_pilot_balancing_20261021.csv),
[per-step module usage](results/bdot_pilot_usage_20261021.csv).
[Raw diagnostic files](../temp/pilot_diagnostics_20261021/).

## Completion and evidence

D, P1-A, P1-B, N-A and R completed successfully, with all 20 checkpoints per
arm retained. R finished **21 October at 9:50 a.m. SGT**. After the diagnostic
follow-ups, the eight communicating GPU burns were restored at **12:51 p.m.**
and independently verified at **12:53 p.m. SGT**. No further training is queued.
Result files were downloaded and hash-checked. Sources: [final results](results/bdot_pilot_final_20261020.json),
[T12/T14 checks](results/bdot_pilot_launch_20261020.json),
[P1-A diagnostics](results/bdot_pilot_A_analysis_20261020.json).
Raw local files: [D](../temp/pilot_final_20261020/D/),
[P1-A](../temp/pilot_final_20261020/P1-A/),
[P1-B](../temp/pilot_final_20261020/P1-B/) (`final_eval.json`,
`train_results.json`, `metrics.jsonl`).
N-A: [final receipt](results/bdot_N_final_20261021.json), including final metrics,
checkpoint checks, burn verification and verified download hashes.
[Raw N-A files](../temp/pilot_N_final_20261021/prepared/runs/N-A/) contain
`final_eval.json`, `train_results.json` and `metrics.jsonl`.

R: [launch checks](results/bdot_R_launch_20261021.json),
[final receipt](results/bdot_R_final_20261021.json),
[raw files](../temp/pilot_R_final_20261021/prepared/runs/R/).
