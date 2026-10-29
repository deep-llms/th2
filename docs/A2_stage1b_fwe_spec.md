# A2 Stage 1b: replication on FineWeb-Edu

Version 1.1, 2026-10-30. Written for the agent that runs the A2 experiments, and fixed before any FineWeb-Edu run. Version 1.1 sets a run order with an interim report (section 4).

- **This file is the pre-registration.** Record its SHA-256 in every output, next to that of the Stage 1 spec.
- **Everything follows `A2_stage1_spec.md` version 3.3** (SHA-256 `9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e`), with DCLM replaced by FineWeb-Edu as listed in section 1. If something cannot be done as written, stop and report; do not improvise.

## 1. What changes

| Item | Stage 1 (DCLM) | Stage 1b (FineWeb-Edu) |
|---|---|---|
| Parents (v2.1 section 4.1) | `dclm` P300, P1T and S72 | `fwe` P300 (300B `main`, `de8d9259fcac769157070eb63cd67fa89d4c2db1`), P1T (1T `main`, `ca1795337a73d08235c904609c4b6daa89ecd6ca`) and S72 (1T `iter_0072000`, `2acd776f32a524e413b02f9c2c5f1f3a49c8d4e5`) |
| Heal stream | The DCLM pool | The FineWeb-Edu pool already prepared and verified for Stage 1. It must be the FineWeb-Edu release the parents were trained on (their run name says FineWeb-Edu-1.4T); record the repository and revision. Shuffle and pack exactly as in v3.3 section 1: once at the document level with `numpy.random.default_rng(2030)`, nested prefixes, at least 781,248 sequences |
| Own-corpus held-out set | DCLM held-out | FineWeb-Edu held-out: 1,024 × 4,096 tokens of FineWeb-Edu documents that are not in the heal stream, built the same way. Reported only |
| Share of the corpus the parents read | P300 about 7.5%, P1T about 25% | P300 about 21%, P1T about 71% |
| Run IDs | `ST1-dclm-…` | `ST1-fwe-…`, for example `ST1-fwe-P1T-S1-3.2B-lr1e-3` |
| Output folder | `results/a2_stage1/` | `results/a2_stage1_fwe/` |

**Unchanged:**
- the arms (P300, P1T, X72), the layer set S1 and the budgets (0.2B, 0.8B, 3.2B);
- the LR grid {1e-4, 3e-4, 1e-3} with its edge rule. **Run the grid again on FineWeb-Edu; do not reuse DCLM's chosen LR;**
- the recipe, the CulturaX final and tuning sets, the decision quantities and labels, and the downstream evaluation (parents and 3.2B students).

**The run order differs** from v3.3: see section 4.

## 2. Code and tests

- **Code:** use the Stage 1 code (training commit `0c1c148`, downstream `13686ed`). If anything on the numerical path has changed, run T17 again first.
- **Tests:** T1 for FineWeb-Edu P300, P1T and S72; T3 for FineWeb-Edu S72; T7 (heal stream vs FineWeb-Edu held-out); T18 for the FineWeb-Edu stream; T16 during the heals. Run T5, T6 and T12 again only if the code has changed. A failing S72 stops arm X72 only.
- **Parent measurements:** as in v3.3 section 3, with the FineWeb-Edu held-out set in place of the DCLM one.

## 3. Reading across the two datasets

These rules are fixed now, before any FineWeb-Edu result. First report FineWeb-Edu's own labels exactly as in v3.3 section 4. Then add this summary.

**Training length** (FineWeb-Edu's label at 3.2B):
- "persists": **replicated**;
- "grey": **weaker on FineWeb-Edu**;
- "erased" or "reversed": **not replicated**.

**LR state:** report FineWeb-Edu's label, and whether A_init shrinks at each budget step (0.2B → 0.8B → 3.2B), as it did on DCLM.

**Also report, without a decision:**
- E and A_init at every budget on both datasets, side by side;
- the difference E(FineWeb-Edu) − E(DCLM) at each budget. P1T has read about 71% of FineWeb-Edu, so if healing on familiar text helps, this difference should be negative;
- each arm's own best grid LR, and whether it matches DCLM's (1e-3).

**If training length is "not replicated":** the planned next step is an exposure test that heals the FineWeb-Edu pair on the DCLM stream. Do not start it; the project owner decides.

## 4. Run order and interim report

This replaces the order of v3.3 section 2. All 13 runs are still made.

**Part 1 (first, after the tests and parent measurements of section 2):**
1. the LR grid: P300 and P1T at 0.2B, at each of {1e-4, 3e-4, 1e-3}, plus the edge pair if the rule calls for it. When the grid ends, post a short note (the grid table, the chosen LR and E at 0.2B) and carry on;
2. P300 at 3.2B, then P1T at 3.2B, at the chosen LR;
3. downstream for the P300 and P1T parents and their two 3.2B students.

**Interim report,** sent as soon as part 1 ends (`REPORT_interim.md` and `decision_stage1_fwe_interim.json`):
- the tests and parent measurements;
- the grid, the chosen LR and each arm's own best LR;
- E at 0.2B and 3.2B on the CulturaX final set, with each arm's Δ_post, and the same on the FineWeb-Edu held-out set as a diagnostic;
- the training-length label at 3.2B and the cross-dataset reading of section 3, with DCLM's values beside them;
- the downstream scores of the four models, and the paired interval for the difference of drops, P1T minus P300.

**Part 2 (straight after part 1; do not wait for a reply):**
4. X72 at 3.2B;
5. P300, P1T and X72 at 0.8B;
6. X72 at 0.2B;
7. downstream for the S72 parent and the X72 student.

**Final report:** it adds the LR-state label, the 0.8B points and ratios, and every remaining item of v3.3 section 4. The interim training-length label stands unless an error is found; if any interim number changes, say which and why.

## 5. Budget

- At Stage 1's measured speed (0.949 s per update), the 13 heal runs, plus at most one edge pair, take about 13.5 node-hours of training. With parent measurements, tests and downstream, the total is about 15 node-hours.
- The grid note comes after about 2 node-hours, and the interim report after about 8.5–9.
- Re-estimate after the tests. Report before launching if the re-estimate exceeds 20 node-hours; the hard cap is 24. Stop any run that exceeds 1.5× its own estimate, and report it.

## 6. Outputs

As in v3.3 section 5, under `results/a2_stage1_fwe/`. Add `decision_stage1_fwe.json` and the interim files of section 4. In `REPORT.md`, include the cross-dataset summary of section 3 and the exposure caveat (P1T has read about 71% of FineWeb-Edu, P300 about 21%).

## Changes in version 1.1

- **Run order:** the LR grid and the 3.2B runs of P300 and P1T come first, with a short note after the grid and an interim report on the training-length question after the 3.2B runs. The X72 runs and the 0.8B runs follow without waiting. No run is added or removed.
