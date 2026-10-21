# B-DOT pilot results — 20 October 2026

**Both routing curricula completed training but performed worse than the matched
dense control.** P1-A was slightly better than P1-B. Neither met the quality
requirement for expanding the research to the next phase.

**21 October update:** N-A and random-routing R both completed. N-A reduces
the deficit but still trails D. R slightly outperforms both learned routing
curricula; learned routing has not shown an advantage over chance in this pilot.

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
**37.50% for P1-B**. P1-A later concentrates work on a few modules. This
suggests an optimization/module-utilization problem, but does not establish
its cause. Soft-start curriculum B did not resolve the quality deficit.

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
learned selection from curriculum effects. Route interventions and downstream
evaluation were not run. Phase B remains unsupported.

D, P1-A, P1-B, N-A and R completed successfully, with all 20 checkpoints per
arm retained. R finished **21 October at 9:50 a.m. SGT**. The eight GPU burns
were restored at **9:52 a.m.** and independently verified at **11:54 a.m. SGT**.
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
