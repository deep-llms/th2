# A2 Y: why does the length penalty E persist?

Version 1.0, 2026-11-04. Written for the agent that runs the A2 experiments, and fixed before any run.

- **This file is the pre-registration.** Record its SHA-256 in every output.
- **This file holds all pending work, in run order.**
  - **Before it:** when this file arrives, pause the running A2 D Part C rerun (P300, 12.8B) at its next saved state, and keep that state.
  - **After it:** resume the A2 D Part C reruns as `A2_depth_spec.md` v1.1 (SHA-256 `2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe`, already in `docs/`) prescribes. That means P300 from its saved state (restart it if the resume fails), then P1T, then the Part C downstream and the final A2 D report.
- **Unless stated here, everything follows** `A2_stage1_spec.md` v3.3 and `A2_stage1b_fwe_spec.md` v1.1. That covers the parents, the heal recipe (S1 aside), the CulturaX final set, precision, per-sequence storage, and the definitions of Δ_pre, Δ_post and E.
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** pruning only. Run only what this file lists.

## 0. Why

**What is known about E.** E = Δ_post(P1T) − Δ_post(P300), the extra damage that long training adds, stays at about +0.02 nats:
- from 0.2B to 12.8B healing tokens (DCLM);
- on DCLM and on FineWeb-Edu;
- for early, middle and terminal blocks (A2 D Part B);
- at every heal LR from 1e-4 to 1e-3.

**Two explanations remain:**
- **A, capacity:** a smaller student cannot hold the extra knowledge of long training, whichever layers go. It predicts that E grows with the number of layers removed (Part 3).
- **B, information lost at the cut:** removing layers discards what they held, and healing cannot rebuild it from logits alone. It predicts that keeping part of that information, by merging the removed layers into their neighbours instead of dropping them, lowers E (Part 2).

**Part 1** measures how much E varies between runs. This gives both readings a noise scale.

## 1. Part 1: run-to-run spread of E (S1, 0.2B)

- **Streams:** two new heal streams per dataset. Shuffle the same prepared pool at the document level with `numpy.random.default_rng(2031)` and `default_rng(2032)`, in place of 2030, and pack them as before. Each run reads its stream's first 762 steps.
- **Runs:** P300 and P1T arms, on `dclm` and `fwe`, S1, LR 1e-3, otherwise the Stage 1 recipe. That is 2 streams × 2 arms × 2 datasets = 8 runs. Run IDs are `Y1-<ds>-<arm>-S1-0.2B-seed<2031|2032>`.
- **Quantities:**
  - E for each of the three streams: 2030 (the existing Stage 1 and 1b values, +0.021096 and +0.026224), 2031 and 2032;
  - per dataset, the mean, the standard deviation σ, and the range.
- **Noise scale** used below: τ = max(2σ, 0.003), with σ the mean of the two datasets' σ.

## 2. Part 2: merge instead of drop (S1, 0.2B)

**Variants.** The same 18-layer student as S1: layers 0–15, 22 and 23 are kept, and 16–21 removed. Layers 15 and 22 receive merged weights:

| Variant | Layer 15 becomes | Layer 22 becomes |
|---|---|---|
| `rdsc` (LaCo's merge) | θ_15 + Σ_{k=16..18} (θ_k − θ_15) | θ_22 + Σ_{k=19..21} (θ_k − θ_22) |
| `avg` | mean(θ_15, θ_16, θ_17, θ_18) | mean(θ_19, θ_20, θ_21, θ_22) |

- **Which parameters:** every weight matrix and bias of attention and MLP. The receiving layers keep their own norm weights.
- **One-shot:** for each variant and arm, record the val and Δ_pre on the final set, next to the `pre-S1` drop values of A2 Q.
- **Heals:** P300 and P1T arms, each from its own parent as teacher, on `dclm` and `fwe`, 0.2B, LR 1e-3, with the 2030 stream. That is 2 variants × 2 arms × 2 datasets = 8 runs. Run IDs are `Y2-<ds>-<arm>-<variant>-0.2B`.

**Quantities:** E_variant, against E_drop = the mean E of Part 1 for that dataset.

**Readings** (fixed now; for each variant):
- **"B supported":** E_variant ≤ E_drop − τ on both datasets.
- **"no effect":** |E_variant − E_drop| < τ on both datasets.
- **"mixed":** otherwise.

**Automatic extension:** for each variant read as "B supported", also run it at 0.8B for P300 and P1T on both datasets (4 runs per variant). Compare with E_drop at 0.8B from Stage 1 and 1b: +0.0208 and +0.0228.

## 3. Part 3: number of layers removed (0.2B and 0.8B)

**Sets.** Contiguous blocks that end at layer 21, like S1, so that layers 22 and 23 are always kept:

| Layers removed | Set | Layers kept |
|---|---|---|
| 3 | 19–21 | 21 |
| 6 | 16–21 (S1; existing values) | 18 |
| 9 | 13–21 | 15 |
| 12 | 10–21 | 12 |

- **Heals:** P300 and P1T arms, each from its own parent as teacher, on `dclm` and `fwe`, at 0.2B (762 steps) and at 0.8B (3,051 steps), LR 1e-3, with the 2030 stream. That is 3 new sets × 2 arms × 2 datasets × 2 budgets = 24 runs. Run IDs are `Y3-<ds>-<arm>-L<3|9|12>-<budget>`.
- **One-shot:** also record Δ_pre of every set and arm.
- **Quantities:** E and E / Δ_post(P300) for every set and budget, with S1's existing values (0.2B: +0.0211 and +0.0262; 0.8B: +0.0208 and +0.0228).

**Readings** (fixed now; at 0.8B):
- **"A supported":** E(3) < E(6) < E(9) < E(12) on both datasets.
- **"A not supported":** E(12) − E(3) < τ on both datasets.
- **"mixed":** otherwise.

Also report the same ordering at 0.2B, and the Spearman correlation between the number of layers removed and E.

**Automatic extension:** if, for the 12-layer set on a dataset, E changes by more than 20% between 0.2B and 0.8B, also run that set at 3.2B for P300 and P1T on that dataset.

## 4. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects.

| Test | What | Expected |
|---|---|---|
| YT1 | Before Part 1, on the current node: rerun `ST1-dclm-P300-S1-0.2B-lr1e-3` once | Final-set val within 0.002 of Stage 1's 2.942554079 |
| YT2 | The 2031 and 2032 streams: disjoint from the tuning and final sets (by document id and content hash), not identical to the 2030 stream, and sequence counts recorded | Holds |
| YT3 | The merged layers: recompute each from the formula, and check the 18 kept original indices | Exact; as in section 2 |
| YT4 | Layer removal for the 3-, 9- and 12-layer sets: the kept original indices | As in section 3 |
| YT5 | Every run: the full LR schedule, the data order, and the save and reload | As in Stage 1 (T6, T16) |

## 5. Order, budget and outputs

**Order:**
1. Pause the A2 D P300 rerun.
2. YT1, then Part 1.
3. **Post a short note:** σ and τ.
4. Part 2, then its automatic extension if triggered.
5. **Post a short note:** the Part 2 readings.
6. Part 3, then its automatic extension if triggered.
7. The report.
8. Resume the A2 D Part C reruns.

**Budget:**
- Part 1 with YT1: about 2 node-hours.
- Part 2: about 1.7, plus about 3.4 per extended variant.
- Part 3: about 12, plus about 5 for each dataset that triggers 3.2B.
- In total, about 16 node-hours without extensions and about 30 with all of them.

Report the ledger; there is no cap.

**Outputs** go in `results/a2_why_20261104/`:
- `config/`, with this file's SHA-256;
- `manifests/`, with the new streams, merge formulas and kept indices;
- the per-sequence NLL files;
- `runs.jsonl`;
- `tests.json`;
- `decision_why.json`, with every E, σ, τ, ratio, reading and clause;
- `REPORT.md`.
