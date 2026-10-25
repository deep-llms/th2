# B-DOT v3 pilot results — 22 October 2026

See [the consolidated follow-up results](BDOT_RESULTS_20261025.md) for the complete
partial-MLP/router tables, second seeds, 10k pair, downstream evaluation and STOP
studies through 25 October. Pending/queued statements below are historical.

All **13 training runs and 13 evaluations completed successfully**, without retries. The queue finished at **05:32:12 SGT** after **13h14m** from the first research run. Automatic communicating burns were verified at 05:33:23; the independent 05:50:33 check matched all eight workers, advancing collective payload, 100% GPU utilization and a released ownership guard.

Every run completed 4,768 updates with the same 2.5B-token pilot recipe (sequence 1,024, 524,288 input tokens/update, 96 warmup updates). Model/routing seeds vary as labeled; data order stays fixed. Each final checkpoint contains optimizer/scheduler and eight RNG files; checkpoints 1,000/2,500/4,768 are retained. Architecture definitions and checks are in [the implementation report](BDOT_V3_IMPLEMENTATION_20261021.md).

C1, the conditional random/no-PF control, subsequently completed at **09:04:48 SGT**
on 22 October. Its result and verified burn handoff are included below; the
original thirteen-job batch above and this one-run follow-up are separate queues.

## Final validation

Loss is LM NLL in nats/token; lower is better. All scores use the same 9,704 sequences / 9,915,698 scored tokens. Gap is to D at the same seed. D seed 0 is the verified carry-over from v2.

| Arm | Seed | Loss | PPL | Gap to D |
|---|---:|---:|---:|---:|
| D (v2 reference) | 0 | 3.494464 | 32.933 | — |
| P1-A-prime | 0 | 5.859132 | 350.420 | +2.364669 |
| V2 | 0 | 3.556648 | 35.046 | +0.062184 |
| C2 | 0 | 3.556570 | 35.043 | +0.062106 |
| V1 | 0 | 3.543922 | 34.602 | +0.049459 |
| V3 | 0 | 3.583273 | 35.991 | +0.088809 |
| C3 | 0 | 3.569939 | 35.514 | +0.075476 |
| V4 | 0 | 3.612010 | 37.040 | +0.117546 |
| D | 1 | 3.494008 | 32.918 | +0.000000 |
| V2 | 1 | 3.558835 | 35.122 | +0.064826 |
| C2 | 1 | 3.554696 | 34.977 | +0.060687 |
| V5 | 0 | 3.654560 | 38.650 | +0.160096 |
| V6 | 0 | 3.543685 | 34.594 | +0.049221 |
| V7 | 0 | 3.555634 | 35.010 | +0.061171 |
| C1 (conditional follow-up) | 0 | 3.558461 | 35.109 | +0.063998 |

- **None of the new arms beats D.** V6 (query-only placement FiLM) and V1 (no placement FiLM) are effectively tied: their losses differ by only 0.000237. Both remain about 0.049 nats worse than D.
- **V2 fails the quality criteria across two seeds.** Its mean loss is 3.557741: 0.063505 above same-seed D and 0.002109 worse than random C2. Own routes help its adapted checkpoint by 0.029907 on average, but this does not translate to beating chance routing or D.
- Both placement FiLM maps hurt versus no PF (V2−V1=+0.012726); widening the window also hurts (V3−V2=+0.026625). Reuse remains costly (V4−V1=+0.068087).
- P1-A′ completed numerically but showed loss spikes and ended at 5.859132. Probability-plus-beta selection alone did not fix v2. All capped learned arms have zero eligible-decision saturation above 0.99, so fixing that statistic was not sufficient for better LM quality.

## Route interventions

All three losses use full frozen validation. Positive default−own means learned routes help this already-trained model; the separately trained chance controls are still necessary for judging whether learned routing helps training.

| Run | Own loss | Forced-default loss | Random-route loss | Default−own |
|---|---:|---:|---:|---:|
| P1-A-prime-s0 | 5.859132 | 5.426644 | 7.436788 | -0.432489 |
| V2-s0 | 3.556648 | 3.578659 | 3.709964 | +0.022011 |
| C2-s0 | 3.556570 | 3.553922 | 3.556570 | -0.002647 |
| V1-s0 | 3.543922 | 3.549957 | 3.768431 | +0.006035 |
| V3-s0 | 3.583273 | 3.630570 | 3.717696 | +0.047298 |
| C3-s0 | 3.569939 | 3.567837 | 3.569939 | -0.002102 |
| V4-s0 | 3.612010 | 3.686298 | 4.104290 | +0.074288 |
| D-s1 | 3.494008 | 3.494008 | 3.494008 | +0.000000 |
| V2-s1 | 3.558835 | 3.596637 | 3.625167 | +0.037802 |
| C2-s1 | 3.554696 | 3.551637 | 3.554696 | -0.003058 |
| V5-s0 | 3.654560 | 3.683126 | 3.988671 | +0.028567 |
| V6-s0 | 3.543685 | 3.549202 | 3.766307 | +0.005517 |
| V7-s0 | 3.555634 | 3.576783 | 3.704648 | +0.021149 |
| C1-s0 | 3.558461 | 3.555269 | 3.558461 | -0.003193 |

## C1 conditional control and decision

C1 copies C2 with placement FiLM disabled: random routing, no reuse, radius 1,
curriculum none, seed 0. Training/model code and all other recipe settings are
unchanged. The run completes 4,768 updates without retries; training/evaluation/
comparison take **58m04s** together. The training data recipe matches C2 and V1,
and evaluation matches their frozen 9,704 sequences / 9,915,698 scored tokens.

**V1 beats C1 by 0.014539 nats**, meeting the 0.01-nat chance-control criterion
at seed 0. This is a positive single-seed result for V1's complete learned-routing
recipe, which includes curriculum A; C1 uses no curriculum. It does not isolate
the effect of learned routing from that curriculum and is not a two-seed finding.

The predeclared PF contingency selects V1 because it beats V2 by 0.012726.
V1 still misses two seed-0 criteria: default-path gain 0.006035 (<0.01) and
gap to D 0.049459 (>0.01). It does not qualify for second-seed confirmation under
section 6.2, so the pilot cannot advance. No two-seed V1 gate result is claimed.
V5/V6/V7 remain exploratory and outside the predefined candidate pool.

C1 retains checkpoints 1,000/2,500/4,768, with final optimizer, scheduler and
all eight RNG-state files verified. Communicating burns were restored at
**09:05:54 SGT**; the **09:51:00 SGT** live check matches workers 298035–298042,
all eight GPUs at 100%, advancing collective progress and a released guard.
No additional training is queued.

## Evidence

- [Final results and operational receipts](results/bdot_v3_final_20261022.json).
- [Downloaded result files](../temp/v3_status_20261022/results/) (per-arm evaluations, metrics, route samples, checkpoint metadata and queue/burn receipts).
- [Launch/numerical verification](results/bdot_v3_launch_20261021.json).
- Remote root: `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01`.

All **124 exported files** and the result archive match the remote SHA256 manifest.

C1 follow-up evidence: [completion and live burn receipt](results/bdot_C1_final_20261022.json),
[downloaded results](../temp/c1_final_20261022/results/).
Remote root: `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-C1-20261022-a01`.

All **18 C1 result files**, their queue-artifact hashes and the archive SHA256
are verified locally.
