# B-DOT pilot: checkpoint diagnostics

Date: 21 October 2026. All times below are Singapore time (SGT).

**Conclusion: both learned routers meet the predeclared weak-balancing condition.** No wrong-sign balancing, illegal routes or nonunit gates were found. The gradient follow-up supports correct ST machinery; direct full-LM finite differences at the original saturated logits have precision limits, described below. The next experiment indicated by the response document is **P1-A′: select with `argmax(softmax(z) + beta)`**, retaining the other P1-A settings. It has **not** been implemented or launched by this diagnostic task.

Protocol: [pilot response](BDOT_pilot_results_response.md), Step 1. R and N-A were already completed; no training was performed here. This diagnoses the failed pilot, and does not establish that the correction will improve loss.

## Evaluation and interventions

Final checkpoints: step 4,768. Each intervention scores the same **9,704 sequences / 9,915,698 target tokens**, FA4/grouped experts, BF16, eight GPUs, batch 16 per rank, routing seed 0. As-trained losses reproduce the saved final results **exactly**. Random intervention replay also requires the same batch size and world size. This intervention is different from training the random-routing control R from scratch.

| Arm | Own routes loss / PPL | Default path loss / PPL | Random route loss / PPL | Default − own |
|---|---:|---:|---:|---:|
| P1-A | 3.615941 / 37.186 | 3.745657 / 42.337 | 4.849492 / 127.675 | +0.129716 |
| P1-B | 3.632704 / 37.815 | 4.450526 / 85.672 | 4.743842 / 114.875 | +0.817822 |
| N-A | 3.525998 / 33.988 | 3.530833 / 34.152 | 4.054829 / 57.675 | +0.004835 |

Default routing hurts P1-A/B, consistent with module adaptation to their learned paths. It does not show that learned routing is better than fixed-depth training. N-A benefits from learned order by only **0.004835 nats**, below the 0.01 default-path criterion.

Previously trained controls: D **3.494464**, R **3.610161**, N-A **3.525998**. P1-A is **0.005780 worse than R**. N-A is **0.031535 worse than D**, so the rule for running R_N (within 0.01 of D) is not met. Reuse removal closes much of P1-A’s gap, but does not eliminate it.

## Module use

Usage is measured on the **first 4,096 frozen validation sequences**, not the full loss population. Module and routing-step IDs are zero-based. Uniform total load is **12.5% per module**.

| Arm | Mean distinct modules/token | Reuse rate | Off-default choices |
|---|---:|---:|---:|
| P1-A | 4.3910 | 99.8617% | 45.2100% |
| P1-B | 5.0001 | 100.0000% | 37.4985% |
| N-A | 8.0000 | 0.0000% | 24.9637% |

| Arm | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1-A | 12.2094% | 24.7373% | 0.5491% | 1.2384% | 25.2611% | 2.2199% | 32.5924% | 1.1924% |
| P1-B | 0.0016% | 24.9984% | 0.0000% | 37.5000% | 0.0000% | 12.5000% | 12.5000% | 12.5000% |
| N-A | 12.5000% | 12.5000% | 12.5000% | 12.5000% | 12.5000% | 12.5000% | 12.5000% | 12.5000% |

Distinct-module histogram (% of tokens):

| Arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1-A | 0.0000 | 0.0000 | 0.2832 | 68.8510 | 23.3505 | 6.6544 | 0.7226 | 0.1383 |
| P1-B | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 99.9866 | 0.0131 | 0.0000 | 0.0000 |
| N-A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 |

### P1-A: module choice at each routing step (%)

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97.67 | 2.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.33 |
| 1 | 0.00 | 99.97 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 |
| 2 | 0.00 | 95.60 | 4.05 | 0.35 | 0.00 | 0.00 | 0.00 | 0.00 | 95.95 |
| 3 | 0.00 | 0.00 | 0.31 | 9.54 | 90.15 | 0.00 | 0.00 | 0.00 | 90.46 |
| 4 | 0.00 | 0.00 | 0.00 | 0.02 | 99.96 | 0.02 | 0.00 | 0.00 | 0.04 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 11.98 | 17.74 | 70.28 | 0.00 | 82.26 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 99.93 | 0.07 | 0.07 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 90.53 | 9.47 | 90.53 |

### P1-B: module choice at each routing step (%)

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01 | 99.99 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 99.99 |
| 1 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 |
| 3 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 |

### N-A: module choice at each routing step (%)

| Step | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | Off default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60.93 | 39.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 39.07 |
| 1 | 39.07 | 44.56 | 16.37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 55.44 |
| 2 | 0.00 | 16.37 | 62.47 | 21.17 | 0.00 | 0.00 | 0.00 | 0.00 | 37.53 |
| 3 | 0.00 | 0.00 | 21.17 | 78.62 | 0.22 | 0.00 | 0.00 | 0.00 | 21.38 |
| 4 | 0.00 | 0.00 | 0.00 | 0.22 | 99.26 | 0.53 | 0.00 | 0.00 | 0.74 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.53 | 79.61 | 19.86 | 0.00 | 20.39 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 19.86 | 77.49 | 2.65 | 22.51 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.65 | 97.35 | 2.65 |

## Balancing and router scale

Validation router statistics below use the same first 4,096 sequences.

| Arm | Checkpoint | Beta range | Raw gap median / p90 | Mean logsumexp on validation | Recent training mean(logsumexp²) |
|---|---:|---:|---:|---:|---:|
| P1-A | 1000 | 1.0460 | 50.1199 / 65.1255 | 0.0126534 | 0.0323752 |
| P1-A | 2500 | 4.0461 | 11.0988 / 23.1967 | 0.0536381 | 1.74803 |
| P1-A | 4768 | 8.5817 | 15.1621 / 22.9547 | 0.147036 | 1.3701 |
| P1-B | 1000 | 1.5300 | 3.3500 / 57.3535 | 1.38668 | 3.82716 |
| P1-B | 2500 | 4.4800 | 56.1437 / 199.2262 | 9.78032e-06 | 0.120095 |
| P1-B | 4768 | 9.0157 | 44.6060 / 145.9966 | 0.00451966 | 0.000915677 |

P1-A’s final median gap **15.16** exceeds beta range **8.58**; P1-B’s **44.61** exceeds **9.02**. Both have module loads below half or above twice the 12.5% target. Both already meet this weak-balancing rule at checkpoint 2,500.

Beta/sign checks: replaying the **actual `finish_update` implementation** on all **476 logged count vectors per arm** gives the expected direction in every case. All three saved-checkpoint beta intervals are compatible with the logged directions and the maximum contribution of unlogged updates. This is not an exact historical per-update audit: beta was not logged every update. The last load log for checkpoint 4,768 is at 4,760.

Z-loss histories are available in the JSON evidence. **Mean logsumexp was not logged during training**; the table supplies checkpoint-time validation measurements instead. Mean(logsumexp²) cannot recover mean(logsumexp). P1-B’s final small z-loss coexists with huge candidate gaps: controlling logsumexp does not prevent losing candidates from becoming very negative.

Full beta/load vectors at all checkpoints: [balancing CSV](results/bdot_pilot_balancing_20261021.csv). Exact per-step counts/shares: [usage CSV](results/bdot_pilot_usage_20261021.csv).

## Correctness and numerical limits

- Across all seven observed prefixes: **zero illegal choices; zero gates differing from one**.
- Original gradient probe: full 12-layer checkpoint, one 128-token prefix (127 scored targets), fixed routes and fixed base probability denominators; FP32 math SDPA/reference projections, TF32 disabled. Four well-resolved P1-A steps pass full-LM finite differences. Others, and all P1-B steps, do not pass the direct full-LM test: derivatives as small as ~2e-43 cannot resolve a ~3.5-valued FP32 loss. The original failure flags are preserved.
- Follow-up: stable FP64 local differences of the gate use `softplus`/`expm1` and the actual LM gate adjoint, avoiding subtraction of nearly equal probabilities. These pass at all original logits. For the full-LM test, freeze logit offsets from the original forward and center one router’s allowed logits at a time, retaining the same weights/routes and exactly the same forward loss. All eight steps of each arm pass at this well-conditioned point, with unchanged tolerances. This verifies the machinery; it does not make the original saturated full-LM finite differences conclusive.
- A preliminary probe centered all eight routers simultaneously. It created an artificial high-gain chain (P1-B directional derivative up to ~1.23e8), so its finite differences were not local at the chosen step sizes. Its failures are preserved in the evidence; the corrected probe centers just the router being tested, leaves the others unchanged, and uses smaller perturbations with unchanged error tolerances.
- CPU tests verify state restoration, observer neutrality, no-reuse masks, two-rank CLI behavior and deliberately flipped-gradient rejection. No trained weights, optimizer, training source or environment was changed; temporary evaluator hooks are removed afterward.

P1-A: centered full-LM check **8/8 passed**, maximum relative error **0.2147%** (two perturbation sizes, 0.002 and 0.001); forward loss unchanged.
P1-B: centered full-LM check **8/8 passed**, maximum relative error **0.0769%** (two perturbation sizes, 0.002 and 0.001); forward loss unchanged.

## Stability, throughput and operations

Neither P1-A nor P1-B has a flagged within-phase spike (49 validation observations each). Under the spec, transitions reset the running minimum: **P1-B has a +0.170650-nat soft-to-hard transition jump**, which is reported separately; P1-A’s widening jump is zero.

T8 evidence is reused from the accepted [dispatch measurement](results/bdot_dispatch_optimization_20261020.json): full training updates, all eight GPUs, sequence 1,024, microbatch 16 × accumulation 4. No throughput training was rerun.

| Native Qwen | D | Hard-routed P1-A configuration |
|---:|---:|---:|
| 0.6176 s/update | 0.6646 s/update | 0.7332 s/update |
| 100% reference | 92.9% throughput | **84.2% throughput** |

The 60% throughput criterion passes. This is a synthetic full-update measurement, separate from complete training wall time including validation/checkpointing.

Main evaluation queue: **12:21:17–12:27:06 SGT**, followed by verified communicating burns at **12:28:11 SGT**. The main queue and both numerical follow-ups use the existing supervisor: copy and verify Accelerate, identify/stop only known burn workers, require all eight GPUs free, run evaluations, automatically restore/verify burns. Corrected follow-up completed at **12:50:02 SGT**, burns were restored at **12:51:08 SGT**, and a fresh check at **12:53:47 SGT** verified all eight communicating workers and the released guard.

All result hashes and route-sample hashes were checked; the final checkpoints’ weight hashes match across the main and follow-up evaluations. Raw files are under `/disk/thuat/DOT/temp/pilot_diagnostics_20261021/`; durable numerical evidence is [the JSON receipt](results/bdot_pilot_diagnostics_20261021.json). The initial directory export returned “no valid paths”; exact-file exports plus a checksum-verified result-only archive were used. A transient Dropbox download rate limit was resolved by sequential retrieval. The first all-router-centered numerical follow-up failed its validation and restored burns; its evidence is preserved, and the corrected probe uses a fresh output directory.

**Decision:** Step 1 supports the document’s bounded-score balancing correction P1-A′ as the next single experiment. It does not justify Phase B, extra curricula, longer training, or a second seed yet. No further training was launched.
