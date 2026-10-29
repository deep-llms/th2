# A2 Stage 1: healing on the parents' own corpus (DCLM)

Version 3.3, 2026-10-29. Written for the agent that runs the A2 experiments, and fixed before any Stage 1 run. The changes from versions 3.0–3.2 are listed at the end.

- **This file is the pre-registration.** Record its SHA-256 in every output. Do not change the arms, budgets, LR grid, metrics or decision rules after any Stage 1 result is seen. If something cannot be done as written, stop and report; do not improvise.
- **Everything not stated here follows `A2_pilot_spec.md` version 2.1:** software, loading, layer removal and precision (section 3.2), the KD recipe except its data (section 3.3), the divergence definition (rule 1.3), the records (rule 1.4) and the "do not" list (rule 1.5).
- **Relation to the pilot:**
  - Phase 0A stands as reported. It is the CulturaX-healing condition.
  - Phase 0B is not run. Its LR-state question moves here as arm X72, healed on DCLM. Its stable-state length question (S72 vs S188) is deferred.

## 0. Summary

**Why this stage.** Phase 0A healed on CulturaX text, for at most 0.8B tokens. In practice a pruned model is healed on its parent's own pretraining data, and for longer. Stage 1 repeats the key comparisons that way, up to 3.2B tokens.

**The questions** (layer set S1, healing on DCLM):
1. **Training length:** after healing, is P1T (1T tokens) damaged more or less than P300 (300B tokens)?
2. **LR state:** at almost the same length, is it better to prune the stable checkpoint S72 and heal it from P300, or to prune the annealed P300?

**Arms.** All come from the DCLM runs, with the parents and pinned commits of v2.1 section 4.1:

| Arm | Student pruned from | Teacher |
|---|---|---|
| P300 | P300 | P300 |
| P1T | P1T | P1T |
| X72 | S72 (1T run, `iter_0072000`) | P300 |

**Layer set:** S1 only: layers 16–21 (groups g8–g10), removed from every arm.

**Budgets:** 3.2B tokens for the decisions; 0.8B and 0.2B to show the trend.

**What it can and cannot show.**
- One dataset, one layer set and one recipe. The labels are pre-registered screening rules, not significance tests.
- Healing on DCLM means P1T has already read more of the healing text than P300: about 25% of the corpus against 7.5%. That is part of the practical question. Decisions are scored on a neutral set (section 1), so that this advantage is not counted a second time in evaluation.
- X72 vs P300 compares two checkpoint choices under this recipe. As in v2.1 section 6.4(i), it does not isolate the LR state alone.

## 1. Data

**Heal stream.** The DCLM-baseline documents prepared by the project owner (about 3.2B GPT-NeoX tokens).
- **Record** the source repository, revision and files, how the documents were selected, and their ids.
- **Order:** if the documents are not a uniform random sample in random order (for example, whole files in stored order), shuffle them once at the document level with `numpy.random.default_rng(2030)` before packing, and record it.
- **Tokenise** as in v2.1 section 3.1: the parents' GPT-NeoX tokenizer, `add_special_tokens=False`, no BOS, and the end-of-text id 0 after each document.
- **Pack:** concatenate in order and cut into 4,096-token sequences; drop the final partial one.
- **Size:** at least 781,248 sequences (12,207 steps × 64). If there are fewer, stop and report.
- **Every run reads the stream from its first sequence, in order.** The budgets are therefore nested: the 0.2B and 0.8B runs read the beginning of the 3.2B runs' data.

**Evaluation sets.**

| Set | Content | Use |
|---|---|---|
| Final set | v2.1's final set (CulturaX validation), unchanged | All decision quantities |
| Tuning set | v2.1's tuning set (CulturaX), unchanged | The LR choice and the divergence check |
| DCLM held-out | 1,024 × 4,096 tokens of DCLM-baseline documents that are not in the heal stream (for example, the documents that follow it, or another file), built the same way | Reported only, labelled "own corpus: the parents may have read part of it in pretraining, P1T more than P300" |

No choice in this spec uses the final set or the DCLM held-out set.

## 2. Runs

| Runs | Arms | Heal tokens (steps) | LR |
|---|---|---|---|
| LR grid | P300, P1T | 0.2B (762) | each of {1e-4, 3e-4, 1e-3} |
| Main heal | P300, P1T, X72 | 3.2B (12,207) | the chosen LR |
| Mid heal | P300, P1T, X72 | 0.8B (3,051) | the chosen LR |
| Short heal | X72 | 0.2B (762) | the chosen LR |

P300 and P1T at 0.2B are the grid runs at the chosen LR.

**One LR for every arm and budget.**
- For each grid LR, take the mean of the two arms' tuning-set val after healing. The chosen LR is the smallest grid LR whose mean is within 0.001 nats of the lowest mean.
- A grid LR at which either arm diverges is not a candidate.
- If the lowest mean is at 1e-4 or at 1e-3, run one more value beyond that edge (3e-5 or 3e-3) for both arms, and apply the rule again over all four values. Do this at most once.
- **Why both arms:** choosing on one arm would give that arm its best LR and the other a worse one, which would bias E.
- **Known limit:** the LR is chosen at 0.2B and used up to 3.2B, and longer runs often prefer a somewhat lower LR. This applies to all arms alike.

**Recipe:** v2.1 section 3.3, with the DCLM heal stream. The warm-up W = max(10, ⌈0.02 n⌉) is 16, 62 and 245 updates at 0.2B, 0.8B and 3.2B.

**Order:** the grid; then the 3.2B runs (P300, P1T, then X72); then the 0.8B runs (in the same order); then X72 at 0.2B. If the batch is cut short, the decision quantities are complete first.

**Divergence** (as defined in v2.1 rule 1.3, on the tuning set):
- a diverged grid run is not a candidate;
- if any other run diverges, stop and report. Do not rerun it at another LR, because all arms must share one LR.

**Run IDs:** `ST1-dclm-<arm>-S1-<budget>-lr<value>`, for example `ST1-dclm-P1T-S1-3.2B-lr3e-4`. Record `arm`, `parent` and `teacher` in `runs.jsonl` as in v2.1 section 6.1.

**Totals:** 13 heal runs, 51,108 steps, about 13.4B KD tokens. The three 3.2B runs carry 72% of them; the six grid runs are short (0.2B each).
- At Phase 0A's throughput (about 145K tokens/s): about 26 node-hours of KD, plus about 1–2 for evaluations.
- If the optimised code is 2.5× faster: about 11–12 node-hours.
- **Budget:** re-estimate after T8. Report before launching if the re-estimate exceeds 30 node-hours; the hard cap is 36. Stop any run that exceeds 1.5× its own estimate, and report it.

## 3. Before the heals

**Parent measurements** (P300, P1T and S72 of DCLM):
- val on the final, tuning and DCLM held-out sets;
- Δ_pre(S1) on the same three sets;
- the annealing gain val(S72) − val(P300) on the final set.

The profiles of v2.1 section 3.4 are not needed.

**Tests.** Send the results before any heal starts.

| Test | What | Pass |
|---|---|---|
| T1, T3 | As in v2.1: T1 for DCLM P300, P1T and S72; T3 for DCLM S72, the first branch parent to be used | As in v2.1. A failing S72 stops arm X72 only |
| T5, T6, T12, T16 | As in v2.1, run with the Stage 1 code | As in v2.1 |
| T7 | The heal stream and the DCLM held-out set are disjoint, by document id and by content hash | No overlap |
| T8 | Throughput of the Stage 1 code; re-estimate the whole workload | Section 2, budget |
| T17 | **Same code, same results.** The optimised code must keep 64 × 4,096 tokens per step, the LR schedule and the precision procedure. With it, rerun `P0-nem-P300-S1-0.05B-lr<chosen>` exactly as in Phase 0A (CulturaX stream) | Its final-set val is within 0.002 nats of the original run's. Otherwise stop and report: comparisons with Phase 0A would not hold |
| T18 | **DCLM stream diagnostics:** the sequence count; the share of end-of-text tokens; a decode of the first and last sequences; val(P300) and val(P1T) on the DCLM held-out set next to their final-set val | Recorded. Flag a parent whose DCLM loss is not below its CulturaX loss, which may signal a tokenisation problem |

## 4. Decision quantities

All on the final set, using the scored representation (v2.1 section 3.2). For each budget H ∈ {0.2B, 0.8B, 3.2B}, at the chosen LR:
- **E(H)** = Δ_post(P1T, S1, H) − Δ_post(P300, S1, H). Positive means P1T is damaged more.
- **A_init(H)** = v(X72, S1, H) − v(P300, S1, H). Negative means pruning the stable checkpoint is better.

**Labels at 3.2B.** Evaluate in this order, and report the truth value of every clause.

Training length:
1. INCOMPLETE if E(3.2B) is undefined;
2. "persists" if E ≥ +0.008 nats;
3. "reversed" if E ≤ −0.008;
4. "erased" if |E| < 0.003;
5. "grey" otherwise.

LR state:
1. INCOMPLETE if A_init(3.2B) is undefined;
2. "anneal first" if A_init ≥ +0.01 nats;
3. "prune before you anneal" if A_init ≤ −0.01;
4. "no effect" if |A_init| < 0.003;
5. "grey" otherwise.

**Report without a decision:**
- E and A_init at 0.2B, 0.8B and 3.2B, with the ratios E(3.2B)/E(0.8B) and A_init(3.2B)/A_init(0.8B) when the denominator is at least 0.003 in size. A ratio of 0.5 or more suggests a stable effect; a smaller one, an effect that is still shrinking.
- Next to them, Phase 0A's CulturaX-healed contrast for DCLM S1 at 0.05B and 0.2B, labelled as a different healing condition. At 0.2B the two differ in the healing data (and in how the LR was chosen), not in the budget.
- E_pre = Δ_pre(P1T, S1) − Δ_pre(P300, S1), and val(P1T) − val(P300).
- All of the above on the DCLM held-out set, with its label.
- The annealing gain val(S72) − val(P300).
- Phase 0A's two T14 noise differences, next to E and A_init.
- The grid: every tuning-set val, the chosen LR, and each arm's own best grid LR (by the same rule, applied to that arm alone).
- E(0.2B) with P300 and P1T each at its own best grid LR, next to E(0.2B) at the common LR. Springer et al. (2025, arXiv 2503.19206) compare checkpoints of different pretraining lengths both at fixed LRs and at each checkpoint's tuned LR, and find that the conclusion can depend on how the best LR shifts with pretraining length. If the two arms' best LRs differ, say so next to the 3.2B labels.

**Downstream (reported only).** Use the nine tasks and settings of the 28 October evaluation (`reviews/A2_DOWNSTREAM_RESULTS_20261028.md`: zero-shot, seed 42, context 4,096, the same metric for each task).
- **Models:** the parents P300, P1T and S72, and the 3.2B students of the three arms. Save per-item scores.
- **Report:** the 8-task MC mean and LAMBADA, with paired bootstrap 95% intervals (10,000 resamples over items, seed 0) for:
  - each student's drop from its parent;
  - the difference of drops, P1T minus P300;
  - the difference X72 minus P300.

## 5. Outputs

Put everything under `results/a2_stage1/`, as in v2.1 section 7:
- `config/` (with this file's SHA-256), `manifests/` (the heal stream's document ids and order, its hash, and the DCLM held-out ids), `tests.json` and `parents.jsonl`;
- `runs.jsonl`, with the fields of v2.1 section 7 plus the DCLM held-out values;
- `downstream/`, with per-item scores and the intervals;
- `decision_stage1.json` and `REPORT.md`: tests and flags, deviations, the decision tables, the exposure caveat of section 0, and compute against budget.

Save every healed student as bf16 safetensors (about 2.6 GB each, about 34 GB in all).

## 6. What follows (the project owner decides; do not start any of it)

- **An effect persists or is reversed:** heal the arms with that effect for 12.8B tokens, and replicate at 3.2B on FineWeb-Edu.
- **Both are erased:** write the result up as "standard healing erases it", or fold it into A1.
- **Grey:** report, and the owner decides.

## Appendix: arithmetic

- **Steps** at 262,144 tokens per step: 0.2B → 762; 0.8B → 3,051; 3.2B → 12,207 (3,199,991,808 tokens, 781,248 sequences).
- **All runs:** grid 6 × 762 = 4,572; X72 at 0.2B 762; 0.8B 3 × 3,051 = 9,153; 3.2B 3 × 12,207 = 36,621. Total 51,108 steps = 13,397,655,552 tokens. At 145K tokens/s that is 92,398 s, about 25.7 node-hours.
- **Without X72** (10 runs): 4,572 + 6,102 + 24,414 = 35,088 steps, about 9.2B tokens, about 17.6 node-hours, or about 7 at 2.5×.

## Changes in version 3.3

- **Restored the three 0.8B heals of version 3.0** (13 runs again), at the project owner's choice. They are not needed for the 3.2B labels. They give a three-point trend, and running them in the same batch keeps one code version for every budget.

## Changes in version 3.2

- **Added one report line, at no cost:** E at 0.2B with each arm at its own best grid LR, read from the grid runs (section 4).

## Changes in version 3.1

- **Dropped the three 0.8B heals.** The grid's P300 and P1T runs at the chosen LR, and X72's 0.2B run, already give a lower point under the same healing condition, so the trend is read from 0.2B to 3.2B. A 0.8B run can be added later with the same data and recipe if a figure needs it. This saves 2.4B tokens, about 4.6 node-hours at Phase 0A's throughput.
