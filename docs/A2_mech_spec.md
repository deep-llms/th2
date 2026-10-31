# A2 M: why each compression fails

Version 1.0, 2026-10-31. Written for the agent that runs the A2 experiments, and fixed before any run.

- **This file is the pre-registration.** Record its SHA-256 in every output.
- **Unless stated here, everything follows** `A2_quant_spec.md` v1.0 (parents, sets, precision, per-sequence storage, contrasts and intervals) and `A2_pilot_spec.md` v2.1 (layer removal, hooks, small set).
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** run only what this file lists. There is no training, and no new compression.

## 0. What is to be explained

A2 Q and Stages 1 and 1b found the following. Relative contrasts are given in the order DCLM / FineWeb-Edu / Nemotron.

- **One-shot pruning** damage grows with training length, at both LR states:
  - L_stab: +19 / +22 / +35%;
  - L_ann: +28 / +37 / +28%.

  Annealing adds less: R_short is +8 / 0 / +21%.
- **PTQ** damage grows with annealing:
  - R_short: GPTQ-3 +11 / +24 / +29%; RTN-3 +20 / +39 / +41%.

  Length at the stable state adds little: GPTQ-3 L_stab is +1 / +20 / +2%.
- **After healing,** P1T keeps only about 55% of its advantage over P300, and the lost part does not come back between 0.2B and 3.2B tokens.

These probes test candidate mechanisms on the same public checkpoints. They measure associations across a 2×2 of length and LR state, not causes.

## 1. Hypotheses

These are fixed now, before any probe runs.

| ID | Hypothesis | Probe |
|---|---|---|
| H-P1 | Longer training makes layers less redundant, so removing them costs more | Single-layer bypass profile; BI of the S1 block |
| H-P2 | Longer training makes layers more interdependent: removing a block costs more, relative to the sum of its layers | Super-additivity ratio ρ of S1 |
| H-Q1 | Annealing makes weights harder to quantize: a larger quantization error relative to the weights | Weight statistics; RTN error in weight space |
| H-Q2 | Annealing makes the model more sensitive to the same relative weight noise | Gaussian noise at a fixed relative size |
| H-E | The part of P1T's advantage that healing does not restore lies in rare or hard tokens | Per-token decomposition |

## 2. Probes

### 2.1 Weights (CPU only; all twelve parents)

For each of the 168 matrices that A2 Q quantizes (A2 Q section 3), record:
- the excess kurtosis of its weights;
- the mean, over groups of 128, of (max − min) / rms(group);
- the relative RTN error in weight space, ‖Ŵ − W‖² / ‖W‖², for `rtn4` and `rtn3`, with the A2 Q grid.

Report the values per matrix, and the parameter-weighted mean per parent.

### 2.2 Noise sensitivity (all twelve parents; final set)

- **N-iso(σ):** add σ · rms(W) · N(0, 1) to every quantized matrix W, for σ ∈ {0.1, 0.2, 0.3}. Here rms is taken over the whole matrix.
- **N-rtn3:** in each group, add uniform noise on [−s/2, s/2], where s is that group's `rtn3` step. This noise has the error size of `rtn3`, but its pattern is random.
- **Seeds:** use seeds 0 and 1 for each. Seed the noise per matrix from the pair (seed, matrix index).
- **Scored representation:** round the noisy weights to bf16 values, as in A2 Q. The damage is D = val(noisy) − val(parent), from per-sequence files.

### 2.3 Layers (all twelve parents; small set of v2.1 section 3.1)

- **Single-layer bypass:** for each layer l = 0 … 23, remove only layer l, and record val − val(parent).
- **Hooks,** in one forward pass:
  - per layer: BI = 1 − mean cos(h_l, h_{l+1}), and the relative update norm, as in v2.1 section 3.4;
  - for the S1 block: BI_S1 = 1 − mean cos(h_16, h_22).
- **Super-additivity:** ρ = Δ_pre(S1) / Σ_{l ∈ S1} (single-layer damage of l). Compute both terms on the small set.

### 2.4 Per-token decomposition (`dclm` and `fwe`; final set)

- **Models:**
  - the parents P300, P1T, S72 and S188;
  - the one-shot `pre-S1` of P300 and of P1T;
  - the saved 3.2B students of P300, P1T and X72, from Stage 1 and Stage 1b.
- **Storage:** save the NLL of every token.
- **Buckets:** two kinds, each made of 10 buckets with equal numbers of token occurrences.
  - **Frequency:** the count of the token's type in the final set.
  - **Difficulty:** the S188 parent's NLL at that position. S188 is outside every contrast below, which avoids selection bias.
- **Report, per bucket:**
  - the parent gap g_par = NLL(P300) − NLL(P1T), and the student gap g_stu, the same for the 3.2B students;
  - the retention r = g_stu / g_par. Flag any bucket where |g_par| < 0.005;
  - the one-shot extra damage, Δ(P1T) − Δ(P300), for `pre-S1`;
  - A_init per bucket: NLL(X72 student) − NLL(P300 student).

## 3. Readout

This readout is fixed now.

**Relative contrasts.** For each metric m (one number per parent from section 2), each dataset and each contrast (L_ann, L_stab and R_short, with the reference parents of A2 Q section 5), the relative contrast is C / m(R). For the noise probes, use the mean of the two seeds, and flag any ordering below that flips between seeds.

**Two patterns:**
- **Pruning pattern:** on every dataset, L_stab and L_ann are both positive and both larger than R_short. One-shot pruning damage meets this.
- **PTQ pattern:** on every dataset, R_short is positive and larger than L_stab. The `rtn3` and `gptq3` damages meet this.

**Readings:**
- H-P1 is supported if the S1 sum of single-layer damages, or BI_S1, shows the pruning pattern.
- H-P2 is supported if ρ shows the pruning pattern.
- H-Q1 is supported if the `rtn3` weight error shows the PTQ pattern.
- H-Q2 is supported if the damage under N-iso(0.2) shows the PTQ pattern.
- H-Q1 and H-Q2 can both hold.
- H-E is supported if, on both `dclm` and `fwe`:
  - mean retention in the 3 rarest frequency buckets is below that in the 3 most frequent;
  - mean retention in the 3 hardest difficulty buckets is below that in the 3 easiest.

**Also report, without a decision:**
- every clause's truth value;
- the ratio of N-rtn3 damage to `rtn3` damage for each parent, which shows whether the size of the `rtn3` error explains its damage;
- across the twelve parents, after subtracting each dataset's mean: the Pearson correlation of each metric with the damage of `pre-S1`, `rtn3` and `gptq3`.

## 4. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects.

| Test | What | Expected |
|---|---|---|
| MT1 | Parent vals, and Δ_pre(S1), on the final set, against A2 Q | Equal within 1e-6 |
| MT2 | N-iso with σ = 0 | Reproduces the parent val within 1e-6 |
| MT3 | N-rtn3, per matrix: the largest noise against s/2, and the variance against s²/12 | No value above s/2; variance within 5% |
| MT4 | Hooks: a layer replaced by the identity | Its BI and update norm are 0 |
| MT5 | Single-layer bypass of layers 0, 11 and 23, against identity hooks (as v2.1 T4(b)) | Logits within 1e-4 in fp32 |
| MT6 | The per-token NLLs, averaged per sequence | Reproduce the per-sequence vals within 1e-6 |

## 5. Order, budget and outputs

**Order:**
1. Section 2.1, on CPU, alongside the rest.
2. Section 2.3.
3. Section 2.2.
4. Section 2.4.
5. The readout and the report.

**Budget:** about 1 node-hour. Report the ledger; there is no cap.

**Outputs** go in `results/a2_mech_20261031/`:
- `config/`, with this file's SHA-256;
- `weights.json`;
- the per-sequence NLL files of the noise probes;
- `layers.json`, with the profiles, BI, update norms and ρ;
- the per-token arrays, kept locally and not in Git, plus `buckets.json`;
- `tests.json`;
- `decision_mech.json`, with every relative contrast, every clause and the correlations;
- `REPORT.md`.
