# A2 S: a second model family (SmolLM3), and heals at the stable state

Version 1.0, 2026-10-31. Written for the agent that runs the A2 experiments, and fixed before any run.

- **This file is the pre-registration.** Record its SHA-256 in every output.
- **When to run:** after the Parts A–B report of `A2_depth_spec.md` v1.1, and before its Part C. Then continue with Part C (DT6 first).
- **Unless stated here, everything follows:**
  - `A2_quant_spec.md` v1.0 for the sets, precision, per-sequence storage, contrasts and intervals;
  - `A2_stage1_spec.md` v3.3 and `A2_stage1b_fwe_spec.md` v1.1 for healing.
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** pruning only. Run only what this file lists.

## 0. Why

**What open-sci-ref shows.** Longer training makes depth pruning more damaging. At the stable state, the one-shot damage grows by +19 / +22 / +35% from S72 (297B tokens) to S188 (776B). This rests on one model family. Two pieces are missing:
- **Part 1, breadth:** does the same hold in another family? SmolLM3-3B is one run whose LR stays constant through its stage 1 (0–8T tokens), on one data mix, with checkpoints every 94.4B tokens. It gives a clean curve of length at the stable state.
- **Part 2, healing at the stable state:** the persistent penalty E has so far been measured between two annealed models. Healing S72 and S188 shows whether it also holds when neither model is annealed.

SmolLM3 cannot replicate E itself: it has no short annealed model, and its LR decay (the last 10% of steps) coincides with a change of data mix.

## 1. Part 1: SmolLM3 one-shot length curve

**Checkpoints:** from `HuggingFaceTB/SmolLM3-3B-checkpoints`, take the stage-1 checkpoints at steps 40,000 × k, for k ∈ {1, 2, 3, 4, 8, 16, 32, 64, 86}. That is about 94B, 189B, 283B, 377B, 755B, 1.51T, 3.02T, 6.04T and 8.12T tokens.
- **Descriptive only:** also take the last stage-2 checkpoint (step 4,200,000, stable, about 9.9T) and the final stage-3 checkpoint (step 4,720,000, annealed, about 11.1T). This contrast is confounded by the stage-3 data mix.
- **Record:** the revision name and commit of each checkpoint, its step and its tokens (step × 2,359,296).
- **Format:** convert to the Hugging Face format if needed, and record how.

**Blocks:** read L, the number of layers, from the config. Remove a quarter of the layers, at the relative depths of the open-sci-ref blocks:

| Block | Layers removed | Open-sci-ref analogue (L = 24) |
|---|---|---|
| B_early | ⌊L/12⌋ … ⌊L/3⌋ − 1 | W1, layers 2–7 |
| B_S1 | ⌊2L/3⌋ … ⌊11L/12⌋ − 1 | S1, layers 16–21 |
| B_end | ⌊3L/4⌋ … L − 1 | W9, layers 18–23 |

**Evaluation:**
- **Text:** the documents of the CulturaX final set, in their stored order, tokenized with the SmolLM3 tokenizer.
- **One document per sequence:** truncate each document at 4,096 tokens, with no packing. SmolLM3 was trained with intra-document masking. Record the number of documents and of tokens.
- **Precision:** the v2.1 procedure. Save per-sequence NLLs.
- **What to record:** for each checkpoint, the val of the parent and of each block removed with no healing, and D = val(pruned) − val(parent).

**Quantities, for each block:**
- D against tokens, and D relative to the first checkpoint;
- the **matched pair** k = 3 (283B) → k = 8 (755B), which spans 2.67× tokens, like S72 → S188 (297B → 776B): L_stab = (D(755B) − D(283B)) / D(283B), with a paired bootstrap interval over documents (10,000 resamples, seed 0);
- the Spearman correlation between log tokens and D, over the nine stage-1 checkpoints;
- the descriptive end pair: D(11.1T) − D(9.9T), relative to D(9.9T).

**Predictions** (fixed now):
- **SM1:** for B_S1, the matched-pair L_stab is positive, and its interval excludes 0.
- **SM2:** for B_S1, the Spearman correlation is at least 0.7.
- **SM3:** B_early rises more than B_S1 from the first to the last stage-1 checkpoint, in relative terms.

Report B_end and the end pair without a prediction.

## 2. Part 2: heals at the stable state (open-sci-ref)

- **Arms:** S72 and S188 of `dclm` and `fwe`. Each is healed from its own parent as teacher, like the P300 and P1T arms.
- **Recipe:** S1, 0.2B (762 updates), LR 1e-3, each dataset's own heal stream; otherwise the Stage 1 recipe.
- **Runs:** 4. Run IDs are `ST1-<ds>-<arm>-S1-0.2B-lr1e-3`.

**Quantities, per dataset, on the final set:**
- E_stab = Δ_post(S188) − Δ_post(S72), and E_stab / Δ_post(S72). Put them next to E_ann at 0.2B from Stage 1 and 1b (+0.021 and +0.026) and its ratio (10% and 13%).
- Also, without a decision: Δ_post(P300) − Δ_post(S72), the damage after healing of an annealed against a stable parent at 300B, each with its own teacher.

**Prediction** (fixed now):
- **SP1:** on both datasets, E_stab ≥ +0.008.

## 3. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects.

| Test | What | Expected |
|---|---|---|
| ST1 | SmolLM3: config (layers, heads, which layers lack RoPE), tokenizer, and the parent vals across checkpoints | Recorded. Flag if the final checkpoint's val is not below the first's |
| ST2 | SmolLM3 layer removal: removing no layer reproduces the logits; each kept layer keeps its original RoPE or NoPE setting | Bitwise in fp32; holds |
| ST3 | Per-document evaluation: no attention across documents, and the token count | Holds; recorded |
| ST4 | S72 and S188 parent vals against A2 Q | Equal within 1e-6 |
| ST5 | Each Part 2 run: full LR schedule, data order, save and reload | As in Stage 1 (T6, T16) |

## 4. Order, budget and outputs

**Order:**
1. Start the SmolLM3 downloads in the background.
2. Part 2: `dclm` S72, `dclm` S188, `fwe` S72, `fwe` S188.
3. Part 1, once the downloads are complete.
4. The report.
5. Then A2 D Part C.

**Budget:** about 2 node-hours (Part 2 about 1, Part 1 about 1), plus the download time. Report the ledger; there is no cap.

**Outputs** go in `results/a2_breadth_20261031/`:
- `config/`, with this file's SHA-256;
- `manifests/`, with the SmolLM3 revisions, commits, steps, conversion, and document and token counts;
- the per-sequence NLL files;
- `runs.jsonl`;
- `tests.json`;
- `decision_breadth.json`, with every quantity, interval and clause;
- `REPORT.md`.
