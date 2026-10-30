# A2 Q: quantization on the same parents

Version 1.0, 2026-10-31. Written for the agent that runs the A2 experiments, and fixed before any quantization run.

- **This file is the pre-registration.** Record its SHA-256 in every output.
- **Unless stated here, everything follows** `A2_pilot_spec.md` v2.1 (parents, loading, precision, data sets) and the Stage 1b downstream protocol (`A2_stage1b_fwe_spec.md` v1.1, batch 2).
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** run only what this file lists.

## 0. Why

Stage 1 (DCLM) and Stage 1b (FineWeb-Edu) gave two results for depth pruning with healing:
- training length leaves a lasting penalty: E ≈ +0.02 nats at 3.2B on both datasets, flat over budgets;
- the LR-state penalty fades: A_init falls from about +0.07 to +0.01.

For post-training quantization (PTQ), Catalan-Tatjer, Ajroldi and Geiping (ICLR 2026) report the opposite driver: the error rises when the LR decays, not with the token budget. This experiment applies PTQ and one-shot depth pruning to the same twelve parents, so that both kinds of compression are measured on the same 2×2 of training length × LR state.

## 1. Two Stage 1b follow-ups (CPU and text only)

Do these first, alongside section 4.

**1a. Pooled downstream analysis (post hoc).** This analysis was defined after the per-dataset results had been seen; label it so.
- **Inputs:** the saved per-item scores of the six Stage 1 models (DCLM) and the six Stage 1b models (FineWeb-Edu), on the same 78,259 requests.
- **Estimate:** for each contrast of the Stage 1 and Stage 1b interval tables, on the MC mean and on LAMBADA, the pooled estimate is the mean of the DCLM and FineWeb-Edu estimates.
- **Interval:** 10,000 resamples, seed 0. Draw items as in the per-dataset bootstrap, and apply each resample's draws to both datasets.
- **Report:**
  - the pooled estimate and its 95% interval;
  - the per-dataset estimates, which must equal the published ones;
  - the correlation between the two datasets' resampled values.
- **Sensitivity,** reported beside the main result and never instead of it: the same analysis with a 7-task MC mean without BoolQ. The reason is that these parents score at BoolQ's majority-class rate (62.2%).
- Add the result to the Stage 1b report as a post-hoc section.

**1b. T17 note (text only).** Add one paragraph to the Stage 1b report:
- the Stage 1 and Stage 1b reruns of T17 differ by about 0.00015 nats;
- the software versions recorded for each run (torch, CUDA, transformers, accelerate and any others), side by side, and whether they match;
- the GPUs (the same 8 UUIDs);
- the cause if known, otherwise "not identified";
- no label depends on it.

## 2. Parents and sets

- **Parents:** all twelve of v2.1 section 4.1, at the pinned commits, loaded as in v2.1 section 3.2. These are S72, P300, S188 and P1T of `dclm`, `fwe` and `nem`.
  - Work on `dclm` and `fwe` first, then `nem`.

  | Parent | Tokens | LR state |
  |---|---|---|
  | S72 | 297B | Stable |
  | P300 | 300B | Annealed |
  | S188 | 776B | Stable |
  | P1T | 1,000B | Annealed |

- **Sets:**
  - the **final set** (CulturaX, v2.1 section 3.1) is the decision set;
  - the **own-corpus set** is a diagnostic only: the DCLM and FineWeb-Edu held-out sets of Stage 1 and 1b, and for `nem` the v2.1 secondary set.
- **Precision:** use the v2.1 section 3.2 procedure for every val. A quantized model's scored representation is its dequantized weights rounded to bf16 values, held in fp32 tensors.
- **Per-sequence storage:** for every model and set, save the mean NLL of each sequence. Every val and every interval below comes from these files.

## 3. Compressions

All compressions are one-shot, with no healing.

| ID | What |
|---|---|
| `rtn4`, `rtn3` | Round-to-nearest, weight-only, 4 and 3 bits |
| `gptq4`, `gptq3` | GPTQ, weight-only, 4 and 3 bits, calibration set cal0 |
| `gptq3-cal1` | As `gptq3`, with calibration set cal1. For the `dclm` and `fwe` parents only; it measures calibration noise |
| `pre-S1` | Layer set S1 removed, no healing: the Δ_pre of v2.1 |

**Common to all quantizers:**
- **What is quantized:** every linear weight in the 24 decoder layers, 7 per layer. Assert 168 matrices and 1,610,612,736 weights.
- **What is not quantized:** the embedding (tied with the LM head), the norms and the biases.
- **Grid:** asymmetric min-max, per group of 128 consecutive input columns in each output row. Compute it in fp32 from the parent's weights:
  - scale = (max − min) / (2^b − 1);
  - zero = round(−min / scale);
  - q = clamp(round(w / scale) + zero, 0, 2^b − 1);
  - ŵ = (q − zero) · scale.

**GPTQ** (Frantar et al., 2023), following the reference implementation (IST-DASLab `gptq`):
- **Calibration:** cal0 is the first 128 sequences of the Phase 0A CulturaX heal stream (4,096 tokens each); cal1 is the next 128.
- **Per linear layer:**
  - H = 2XXᵀ/n over all calibration tokens. Run the forward passes with the v2.1 procedure, and accumulate H in fp32.
  - Dampening: 0.01 × mean(diag H).
  - Block size: 128.
  - Groups: 128, with the grid above, computed when each group starts.
  - No activation reordering.
- **Order:** decoder layers in sequence, with the calibration activations passed through the layers already quantized. Inside a layer, use true-sequential order: attention input projections, then attention output, then MLP input projections, then MLP output. Re-run the layer after each step.
- **Libraries:** a library may replace this implementation only if it supports the model and these exact settings. Record which was used.

**Run IDs:** `Q-<ds>-<parent>-<compression>`, for example `Q-fwe-P1T-gptq3` or `Q-dclm-S72-pre-S1`.

## 4. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects as unverified. Fix what can be fixed, rerun those results, and continue.

| Test | What | Expected |
|---|---|---|
| QT1 | Parent vals on the final set, against any values already recorded (Phase 0A, Stage 1, Stage 1b) | Equal within 1e-6 |
| QT2 | Δ_pre(S1) on the final set, against the Stage 1 and 1b values (P300, P1T and S72 of `dclm` and `fwe`) | Equal within 1e-6 |
| QT3 | Quantizer: each group holds at most 2^b distinct values, and quantizing ŵ again changes nothing | Holds |
| QT4 | GPTQ with H set to the identity, on one layer, against RTN | Bitwise equal |
| QT5 | Determinism: `gptq3` of one parent, run twice | Identical weights |
| QT6 | cal0 and cal1 are disjoint from the final set, the tuning set and each other, by document id and by content hash | No overlap |
| QT7 | The weight count and the list of quantized matrices | As in section 3 |

## 5. Quantities and labels

**Damage.** For dataset d, compression c and parent X, the damage is D_c(X) = val(c(X)) − val(X) on the final set. For `pre-S1`, D is Δ_pre. Also report the relative error D / val(X), the measure used by Catalan-Tatjer et al.

**Contrasts.** Each contrast has a reference parent R:

| Contrast | Formula | R | Compares |
|---|---|---|---|
| L_ann | D(P1T) − D(P300) | P300 | Length at the annealed state (300B vs 1T) |
| L_stab | D(S188) − D(S72) | S72 | Length at the stable state (297B vs 776B) |
| R_short | D(P300) − D(S72) | S72 | LR state at 300B (annealed vs stable) |
| R_long | D(P1T) − D(S188) | S188 | LR state at the long end. The lengths differ (1T vs 776B), so it is described only |
| I | L_ann − L_stab | — | The interaction; described only |

**Intervals:** paired bootstrap over final-set sequences, 10,000 resamples, seed 0, with the same draws for all models of a dataset.

**Labels** for L_ann, L_stab and R_short, for each dataset and compression. Let T = max(0.003, 0.10 · D(R)) and t = max(0.003, 0.03 · D(R)). Evaluate in this order, and report the truth value of every clause:
1. "more damage" if C ≥ T and the 95% interval excludes 0;
2. "less damage" if C ≤ −T and the interval excludes 0;
3. "no effect" if |C| < t;
4. "grey" otherwise.

**Across datasets:** for each compression and contrast, list the three datasets' labels. Call the result "consistent" if they agree.

**Calibration flag:** for `gptq3`, flag any contrast smaller than twice its cal0-vs-cal1 difference.

## 6. Comparison table (reported, no decision)

**One table per dataset:**
- rows: the compressions `pre-S1`, `rtn4`, `rtn3`, `gptq4` and `gptq3`;
- columns: L_ann, L_stab, R_short and R_long, each in nats and relative to D(R), with the labels.

**Under the `dclm` and `fwe` tables,** add the healed-pruning results of Stage 1 and 1b at 3.2B: E and A_init. Note that A_init compares two students, not two damages.

## 7. Downstream (reported only; run last)

- **Bit-width:** choose the GPTQ bit-width whose mean D over the eight `dclm` and `fwe` parents is closest to 0.10 nats, which is about the damage of healed pruning at 3.2B.
- **Models:** score its eight quantized models and the two S188 parents of `dclm` and `fwe` with the Stage 1b protocol. The other parents already have scores.
- **Intervals:** paired item bootstrap, 10,000 resamples, seed 0, per dataset. Report each parent's drop, and the contrasts L_ann, L_stab and R_short of the drops, on the MC mean and on LAMBADA.

## 8. Order, budget and outputs

**Order:**
1. Section 1, on CPU, alongside the rest.
2. The tests, then the parent vals and `pre-S1` for all twelve parents.
3. `rtn4` and `rtn3` for all twelve parents.
4. `gptq4` and `gptq3` for `dclm` and `fwe`, then `gptq3-cal1`.
5. **Post a short note:** the section 5 labels for `dclm` and `fwe`. Do not wait for a reply.
6. `gptq4` and `gptq3` for `nem`.
7. Section 7 (downstream).
8. The final report.

**Budget:** about 1 node-hour for steps 2–6, and about 1 for the downstream. Report the ledger; there is no cap.

**Outputs** go in `results/a2_quant_20261031/`:
- `config/`, with this file's SHA-256;
- `manifests/`: the cal0 and cal1 ids, and the GPTQ implementation with its versions;
- the per-sequence NLL files;
- `runs.jsonl`, with one line per model and set;
- `tests.json`;
- `decision_quant.json`, with all contrasts, intervals and labels;
- `REPORT.md`, with the section 6 tables, the downstream, the section 1 follow-ups, the deviations and the ledger.
