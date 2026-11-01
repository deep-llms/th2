# A2 D: depth dependence, and healing to 12.8B

Version 1.1, 2026-10-31. Written for the agent that runs the A2 experiments, and fixed before any run. Version 1.1 adds Part C (healing to 12.8B); Parts A and B are unchanged from version 1.0.

- **This file is the pre-registration.** Record its SHA-256 in every output.
- **Unless stated here, everything follows:**
  - `A2_quant_spec.md` v1.0 for the parents, sets, precision, per-sequence storage, contrasts and intervals;
  - `A2_stage1_spec.md` v3.3 and `A2_stage1b_fwe_spec.md` v1.1 for healing;
  - `A2_pilot_spec.md` v2.1 for layer removal and the layer sets S2–S4.
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** pruning only. Run only what this file lists.

## 0. Why

**The current block, S1.** For S1 (layers 16–21), training length drives pruning damage, and the LR state matters less and fades with healing.

**The A2 M single-layer profile.** It suggests that this depends on depth. The pattern is the same on all three datasets:
- in layers 1–4, longer training raises the damage of removing a layer by +31 to +290%, while annealing *lowers* it (−7 to −39%);
- in layers 22–23, annealing raises it (+13 to +33%), and length matters much less.

**The question.** If this holds for removing whole blocks, then the answer to "what makes a model hard to prune" depends on which layers are pruned. Part A maps this with one-shot pruning. Part B checks whether it survives healing, at 0.2B tokens, where E for S1 has already reached its plateau.

**Part C** extends the main S1 result on DCLM to 12.8B healing tokens, the next step registered in Stage 1. It asks two things:
- does the length penalty E stay flat?
- does the LR-state penalty A_init keep shrinking? Extrapolating 0.2B → 3.2B gives anywhere from about −0.015 to +0.004 at 12.8B, so the 3.2B label "anneal first" would likely not hold.

## 1. Layer sets

The 24 layers form 12 groups of two: g0 = layers 0–1, …, g11 = layers 22–23.

**Windows:** W1 … W9, each three adjacent groups (6 layers, the size of S1). Wk = g_k, g_{k+1}, g_{k+2}:

| Window | Layers | Window | Layers | Window | Layers |
|---|---|---|---|---|---|
| W1 | 2–7 | W4 | 8–13 | W7 | 14–19 |
| W2 | 4–9 | W5 | 10–15 | W8 | 16–21 (= S1) |
| W3 | 6–11 | W6 | 12–17 | W9 | 18–23 |

g0 is never removed, as in the pilot.

**Also:** the pilot sets S2 (chosen by Block Influence, per dataset), S3 (interleaved) and S4 (random), exactly as recorded in Phase 0A.

## 2. Part A: one-shot pruning (all twelve parents, final set)

- **What to run:** for each parent and each set (W1 … W9, S2, S3, S4), remove the layers with no healing. Record D = val(pruned) − val(parent), from per-sequence files.
- **Contrasts and intervals:** compute L_ann, L_stab, R_short and R_long, in nats and relative to D(R), with the A2 Q reference parents and intervals (10,000 resamples, seed 0).

**Predictions** (fixed now):
- **A1, annealing matters more deeper:** on every dataset, R_short(W9) > R_short(W8) > R_short(W1).
- **A2, length matters more early:** on every dataset, L_ann(W1) > L_ann(W8) and L_stab(W1) > L_stab(W8).
- **A3, a sign flip early:** R_short(W1) < 0 on at least two of the three datasets.
- **A4, the pattern switches at the end:** on every dataset, R_short(W9) ≥ L_stab(W9), so W9 does not show the pruning pattern of A2 M.

**Also report:**
- the full table, with every relative contrast per set and dataset;
- for each dataset, the Spearman correlation between the window index (1 … 9) and each of R_short, L_ann and L_stab;
- the truth value of every prediction clause.

**Post a short note** when Part A ends: the window table, and A1–A4. Then continue to Part B without waiting.

## 3. Part B: healing at 0.2B (`dclm` and `fwe`)

- **Blocks:** W1 (early, layers 2–7) and W9 (terminal, layers 18–23).
- **Arms:** P300, P1T and X72, exactly as in Stage 1 and 1b. X72 is S72 with the block removed, healed from the P300 teacher.
- **Recipe:**
  - the Stage 1 recipe, with only the layer set changed;
  - each dataset's own heal stream (the first 762 steps);
  - LR 1e-3, the LR chosen for S1 on both datasets. Do not run a new grid; both arms of every contrast share the LR.
- **Runs:** 2 blocks × 3 arms × 2 datasets = 12. Run IDs are `D-<ds>-<arm>-<block>-0.2B-lr1e-3`.

**Quantities.** Per block and dataset, on the final set:
- E = Δ_post(P1T) − Δ_post(P300), and its relative value E / Δ_post(P300);
- A_init = val(X72) − val(P300), and A_init / Δ_post(P300).

Put them next to S1's values at 0.2B from Stage 1 and 1b:

| | E | E / Δ_post(P300) | A_init | A_init / Δ_post(P300) |
|---|---|---|---|---|
| DCLM | +0.021 | 10% | +0.073 | 36% |
| FineWeb-Edu | +0.026 | 13% | +0.078 | 40% |

**Predictions** (fixed now):
- **B1:** on both datasets, E / Δ_post(P300) is larger for W1 than for S1.
- **B2:** on both datasets, A_init / Δ_post(P300) is larger for W9 than for S1.

**Also report:**
- the Stage 1 labels for E (persists, grey, erased, reversed) applied to each block at 0.2B, for description only;
- every clause's truth value.

## 4. Part C: healing to 12.8B (`dclm`, S1)

- **Arms:** P300, then P1T, then X72, exactly as in Stage 1.
- **Recipe:** the Stage 1 recipe with S1 and LR 1e-3, for a budget of 12.8B tokens:
  - 48,828 updates;
  - a warm-up of 977 updates, then linear decay to 0;
  - one full schedule per run. Never take an intermediate checkpoint as a result.
- **Data:** the DCLM heal stream of Stage 1, extended in the same shuffled order (`default_rng(2030)`, nested prefixes) to its first 3,124,992 sequences. Its first 781,248 sequences must be the Stage 1 3.2B stream (DT6).
- **Run IDs:** `ST1-dclm-<arm>-S1-12.8B-lr1e-3`.
- **Interruptions:** if a run is interrupted, resume it from its last saved state with the same data order and LR schedule, and record the resume.
- **Saving:** save the three students as bf16, as in Stage 1.

**Quantities, on the final set:**
- E(12.8B) and A_init(12.8B), next to the 0.2B, 0.8B and 3.2B values of Stage 1;
- the ratios E(12.8B) / E(3.2B) and A_init(12.8B) / A_init(3.2B);
- the Stage 1 labels (v3.3 section 4), applied at 12.8B.

**Predictions** (fixed now):
- **C1:** E(12.8B) ≥ +0.008, so the label is "persists".
- **C2:** A_init(12.8B) < +0.01, so the label is not "anneal first". Report which label it gets.

**Downstream** (reported only): the three 12.8B students, with the Stage 1b protocol (batch 2). Report the paired intervals for the P1T − P300 drop and for X72 − P300, on the MC mean and on LAMBADA.

**Post a short note** after each Part C run: its val and Δ_post, and E or A_init as soon as it is defined. Continue without waiting.

## 5. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects.

| Test | What | Expected |
|---|---|---|
| DT1 | W8 one-shot D against A2 Q `pre-S1`, all twelve parents | Equal within 1e-6 |
| DT2 | Layer removal: the kept original indices, for every set (as v2.1 section 3.2) | As listed in section 1 |
| DT3 | S2, S3 and S4 against the sets recorded in Phase 0A | Identical |
| DT4 | Before Part B: rerun `ST1-dclm-P300-S1-0.2B-lr1e-3` once with the current code | Final-set val within 0.002 of Stage 1's 2.942554079 |
| DT5 | Each Part B and Part C run: the full LR schedule, the data order, and the save and reload | As in Stage 1 (T6, T16) |
| DT6 | The Part C stream: the hash of its first 781,248 sequences against the Stage 1 3.2B stream, and its sequence count | Identical; at least 3,124,992 sequences |

## 6. Order, budget and outputs

**Order:**
1. Part A, then the short note.
2. DT4.
3. Part B: `dclm` W9, `fwe` W9, `dclm` W1, `fwe` W1, with the arms in the order P300, P1T, X72.
4. **Post the Parts A and B report.** It is final for those parts.
5. DT6, then Part C, with a note after each run.
6. The Part C downstream, then the final report.

**Budget:**
- Part A: about 0.6 node-hours.
- DT4 and Part B: about 2.7.
- Part C: about 39 (3 runs × about 12.9 hours, plus evaluation and downstream).
- In total, about 42 node-hours, or roughly 1.8 days.

Report the ledger; there is no cap.

**Outputs** go in `results/a2_depth_20261031/`:
- `config/`, with this file's SHA-256;
- the per-sequence NLL files;
- `runs.jsonl`;
- `tests.json`;
- `decision_depth.json`, with every contrast, interval, ratio and clause;
- the three 12.8B students, and the Part C downstream scores;
- `REPORT.md`, with the Part A depth table, the Part B table next to S1, the Part C table next to the Stage 1 budgets, the predictions, the deviations and the ledger.

## Changes in version 1.1

- **Part C added:** healing to 12.8B for P300, P1T and X72 on DCLM with S1, with predictions C1 and C2, test DT6 and the downstream of the three students.
- Parts A and B, their predictions and tests DT1–DT5 are unchanged. A run already started under version 1.0 continues as it is.
