# A2 F: the healed length penalty in a second model family (SmolLM3), and a 1.3B from-scratch reference

Version 1.1, 2026-11-07. Written for the agent that runs the A2 experiments, and fixed before any run.

- **What changed from v1.0 (not sent):**
  - the LR grid is capped at η_pk/2, and E at η_pk/4 is always reported;
  - the optimizer settings are stated;
  - the automatic extensions have a noise floor;
  - the readings have a convergence guard;
  - a second data order is added.
- **This file is the pre-registration.** Record its SHA-256 in every output.
- **It holds every pending part, in run order** (section 1). It ends by resuming the A2 D reruns of `A2_depth_spec.md` v1.1.
- **Unless stated here, everything follows:**
  - `A2_quant_spec.md` v1.0 for the sets, precision and per-sequence storage;
  - `A2_stage1_spec.md` v3.3 and `A2_stage1b_fwe_spec.md` v1.1 for healing: forward-KL KD from the unpruned parent, 2% warm-up then linear decay to 0, batch 64 × 4,096 tokens;
  - `A2_breadth_spec.md` v1.0 for SmolLM3: checkpoints, layer removal with RoPE/NoPE kept, one document per evaluation sequence.
- **Standing rule (project owner, 30 Oct 2026).** Never stop, and never wait for a reply. When a check fails, a limit is passed, or something cannot be done as written:
  1. post a short note at once: what happened, what you do instead, and that you are continuing;
  2. take the option that keeps the most of this plan, and mark every result it affects;
  3. keep going.

  This applies to code as well: no check may exit a job. The owner stops the work if needed.
- **Scope:** pruning only. Run only what this file lists.

## 0. Why

- **The penalty.** In open-sci-ref, the healed length penalty E = Δ_post(long) − Δ_post(short) is about 0.02 nats between the annealed parents P300 and P1T.
  - It persists from 0.2B to 12.8B heal tokens.
  - It barely depends on how many layers are removed (A2 Y).
- **What A2 S added:**
  - **The one-shot length effect replicates in SmolLM3.** B_S1 damage rises by +7.6% from 283B to 755B, and its Spearman correlation with log tokens over nine stage-1 checkpoints is 1.0.
  - **The penalty also holds between two stable checkpoints.** For S72 and S188, E_stab is +0.011 (DCLM) and +0.015 (FWE). That is about half of E in nats. The parents' own gap is smaller too, so relative to that gap it is as large or larger:

    | Pair | Parent gap G = val(short) − val(long), DCLM / FWE | E, DCLM / FWE | E / G, DCLM / FWE |
    |---|---|---|---|
    | Stable (S72 → S188), heal LR η_pk/4 | 0.022 / 0.018 | +0.011 / +0.015 | 0.51 / 0.81 |
    | Annealed (P300 → P1T), 0.2B, heal LR η_pk/4 | 0.051 / 0.052 | +0.021 / +0.026 | 0.41 / 0.50 |
- **Why SmolLM3:** a second family can therefore be tested at the stable state. SmolLM3 has stage-1 checkpoints at a constant LR on one data mix, but no short annealed model.
- **Caveat, shared with the S72/S188 pair:** the heal schedule decays to 0, so healing also anneals a stable student. If the gain from annealing differs with length, it shifts E. The comparison with S72/S188 is like for like.
- **Why the 1.3B model:** the open-sci-ref 1.3B model is the 1.7B with FFN 5,440 instead of 8,192. It has the same non-embedding size as the S1 student, about 1.21B parameters. It is the from-scratch reference that answers "why not use the smaller model?"
- **LR scales differ 20×:**

  | | Pretraining peak LR η_pk | Pretraining batch | Heal batch / pretraining batch |
  |---|---|---|---|
  | open-sci-ref (1T runs) | 4e-3 | 4.12M tokens | 1/16 |
  | SmolLM3 (published) | 2e-4 | 2.36M tokens, AdamW β = (0.9, 0.95), weight decay 0.1 (none on embeddings), clipping 1.0 | 1/9 |

  - So heal LRs must be set relative to η_pk; absolute values do not transfer.
  - The open-sci-ref heals used 1e-3 = η_pk/4, the NLL-best value of its grid. At 3e-3 (0.75 η_pk), both students were worse and E doubled.

## 1. Order

1. **In the background, while the A2 D reruns continue:**
   - download the SmolLM3 checkpoints (section 3);
   - re-tokenize the FWE heal stream and the CulturaX tuning documents with the SmolLM3 tokenizer.
2. **Pause the A2 D reruns** at their next saved checkpoint once step 1 is done. Record the step.
3. **Part F0:** the 1.3B reference, evaluation only.
4. **Part F1:** SmolLM3 checks FT1–FT4.
5. **Part F2:** LR grid and second data order, then **Part F3:** budgets, then **Part F4:** the third checkpoint.
6. **The report.**
7. **Resume the A2 D reruns** per `A2_depth_spec.md` v1.1. They are filler now: no claim depends on them, and the owner may stop them at any time.

## 2. Part F0: the 1.3B from-scratch reference (evaluation only)

**Models:**
- `open-sci/open-sci-ref-v0.01-1.3b-dclm-300B-4096`.
- The FineWeb-Edu twin, expected at `open-sci/open-sci-ref-v0.01-1.3b-fineweb-edu-1.4t-300B-4096`. Confirm the name on Hugging Face.
- If several variants exist (for example `-lr…-2`), use the one without a suffix and list the others.

**Record:**
- revision and commit;
- the config (layers, hidden size, FFN size, heads);
- the non-embedding parameter count. Expected about 1.205B, against 1.208B for the S1 student.

**Evaluate:** the CulturaX tuning and final sets, with the same code, procedure and per-sequence storage as the parents and students. Record the code hash.

**Quantities, per dataset, on the final set (descriptive, no prediction):**
- **The 1.3B against the P300 parent:** NLL(1.3B), and NLL(1.3B) − NLL(P300 parent).
- **The S1 students against the 1.3B:**
  - For the P300 S1 students: NLL(student) − NLL(1.3B) at 0.2B, 0.8B and 3.2B, plus 12.8B on DCLM.
    - The 12.8B value is the first-node final NLL; the cross-node difference is about 1e-4.
  - The same for the P1T students, also descriptive; note that they had more pretraining tokens.
- **Break-even heal budget:** the smallest budget at which NLL(P300 student) ≤ NLL(1.3B).
  - Interpolate linearly in log budget between the two budgets around it.
  - If it is never reached, report the gap at the largest budget.
- **Total tokens seen by each model:** 300B for the 1.3B; for the students, 300B plus the heal tokens.

## 3. SmolLM3 setup (Part F1)

**Checkpoints:** take these stage-1 revisions from `HuggingFaceTB/SmolLM3-3B-checkpoints`. Tokens are step × 2,359,296; SmolLM3 has 3.08B parameters.

| Label | Revision | Tokens | Tokens per parameter |
|---|---|---|---|
| k6 (short) | `stage1-step-240000` | 566.2B | 184 |
| k20 (long) | `stage1-step-800000` | 1.887T | 613 |
| k64 (curve point) | `stage1-step-2560000` | 6.040T | 1,961 |

- k20/k6 = 3.33, the P1T/P300 ratio, at about the same tokens per parameter as P300 and P1T (175 and 585).
- k64 was used in A2 S; reuse its files if they are kept.
- If a revision is missing, use the nearest stage-1 revision and record its tokens and the new ratio.
- **LR state:** for each checkpoint, read the LR from its saved scheduler or optimizer state, or from the published training config or logs. It should equal the constant peak η_pk = 2e-4.
  - Use the value read as η_pk.
  - If no value can be read, record that and use 2e-4.

**Student:**
- Remove B_S1 exactly as in `A2_breadth_spec.md` v1.0: layers ⌊2L/3⌋ … ⌊11L/12⌋ − 1, which is layers 24–32 for L = 36.
- Keep each remaining layer's RoPE or NoPE setting.
- The teacher is the same checkpoint, unpruned.

**Optimizer and schedule:**
- AdamW with β = (0.9, 0.95), ε = 1e-8, gradient clipping 1.0.
- Weight decay as in the Stage 1 heals, with none on the tied embeddings (as in SmolLM3 pretraining).
- Fresh optimizer state.
- 2% warm-up from 0 to the run's LR, then linear decay to 0.

**Heal data:**
- **Stream:** the FWE heal stream, with the same documents in the same order (seed 2030) as the open-sci-ref FWE heals, re-tokenized with the SmolLM3 tokenizer.
- **Packing:** pack into 4,096-token sequences with intra-document attention masking, and reset the position ids at each document, as in SmolLM3 pretraining.
  - If the stack cannot mask, pack without masking.
  - Flag every result this affects, and continue.

**Evaluation:**
- **Final set:** the A2 S SmolLM3 set: CulturaX final documents, one per sequence, truncated at 4,096 tokens.
- **Tuning set:** the CulturaX tuning documents, prepared the same way.
- **Parent vals:** score the parent vals of k6, k20 and k64 on both sets.
- **Per-sequence NLLs:** store them for every model.

**Overlap audit, as done for FWE on 6 Nov:**
1. Check the FWE heal pool against the tuning and final documents at the level of document content.
2. From each run's actual sample order, record whether any overlapping document was consumed.
3. Report E on the full set and on the set without the overlapping evaluation documents. If every |ΔE| < 0.001, the overlap is immaterial and the full-set E stays primary.

**Keep the raw per-rank sample-order traces of every run** in the backup. Stage 1b's traces were lost.

## 4. Part F2: LR grid and a second data order, at 0.2B

**Grid:** {η_pk/8, η_pk/4, η_pk/2}, which is {2.5e-5, 5e-5, 1e-4} for η_pk = 2e-4.
- η_pk/4 is the value matched to the open-sci-ref heals.
- The heal-to-pretraining batch ratio is 1/9 here, against 1/16 there. With square-root batch scaling, the matched value moves only to about η_pk/3, which the grid brackets.

**Runs:** k6 and k20 at each LR, 0.2B (762 updates), data order seed 2030: 6 runs. Run IDs are `F2-fwe-<k>-BS1-0.2B-lr<lr>-s2030`.

**Primary LR:**
- η* is the common best: the grid LR with the lowest mean tuning NLL over the two arms. Also report each arm's own best.
- Use η* for every later run.

**Edges:**
- **No upward extension.** If η_pk/2 is best, keep η* = η_pk/2 and note that the optimum may lie higher.
  - LRs near η_pk are a different regime: they barely cool a stable checkpoint.
  - For annealed parents, a heal LR near η_pk changed E.
- **Downward extension.** If η_pk/8 is best, add η_pk/16 for both arms (2 runs), and take η* as the best of the four.

**Second data order:** k6 and k20 at η*, 0.2B, data order seed 2031: 2 runs. Run IDs end in `-s2031`.

## 5. Part F3: budgets

- **0.8B** (3,051 updates) at η* for k6 and k20: 2 runs.
- **Automatic 3.2B** (12,207 updates) for k6 and k20: 2 runs.
  - Trigger: |E(0.8B) − E(0.2B)| > max(0.2 × |E(0.2B)|, 0.003).
  - The 0.003 floor keeps noise from triggering it.

## 6. Part F4: the third checkpoint

k64 starts from about twice the one-shot damage of k6 (2.04 against about 1.03 nats in A2 S). A2 Y showed that a student far from repaired understates E, so k64 gets its own trigger.

- **k64** at η*, at 0.2B and 0.8B: 2 runs.
- **Automatic 3.2B** for k64: 1 run.
  - Trigger: D64 (section 7) changes between 0.2B and 0.8B by more than max(0.2 × |D64(0.2B)|, 0.003), or Part F3 ran 3.2B.
  - If Part F3 did not run 3.2B, also run k20 at 3.2B so that D64 has a partner: 1 more run.

## 7. Quantities and readings (fixed now)

**Per run:** parent val, Δ_pre, Δ_post (its sign included), LR, updates, tokens.

**Per budget and LR:**
- E = Δ_post(k20) − Δ_post(k6);
- E_pre = Δ_pre(k20) − Δ_pre(k6);
- the parent gap G = val(k6) − val(k20), and E / G;
- for the curve: E64 = Δ_post(k64) − Δ_post(k6) and D64 = Δ_post(k64) − Δ_post(k20).

**Intervals:** paired bootstrap over documents (10,000 resamples, seed 0) for E, E / G, E64 and D64. These intervals cover evaluation-set noise only.

**Data-order noise:**
- Report |E(s2030) − E(s2031)| at η*, 0.2B.
- Flag it if it exceeds 0.003, against σ = 0.0015 in open-sci-ref (A2 Y).

**Readings, at η*, at the largest budget run (0.8B, or 3.2B if it ran):**
- **FE1, replication:**
  - "persists" if E ≥ +0.008;
  - "erased" if |E| < 0.003;
  - "reversed" if E ≤ −0.008;
  - "weak" otherwise.

  These are the same labels as the main study.
- **FE2, budget:**
  - "flat" if |E(0.8B) − E(0.2B)| < 0.003;
  - otherwise "changing", with its direction.
- **FE3, curve, at the largest budget run for k64:**
  - "grows" if D64 ≥ +0.003;
  - "saturates" if |D64| < 0.003;
  - "falls" if D64 ≤ −0.003.
- **Convergence guard.** Add "not converged" to FE1 or FE3 if its quantity (E, or D64) changed by more than max(20%, 0.003) between the two largest budgets run.

**Descriptive:**
- E at every grid LR, with E at η_pk/4 always shown as the point matched to the open-sci-ref heals;
- E / G next to open-sci-ref's stable (0.51 / 0.81) and annealed (0.41 / 0.50 at 0.2B) values;
- E_pre against E, which shows how much of the one-shot excess healing removes.

## 8. Tests

Every test is recorded and flagged; none stops anything. A failing test marks the results it affects.

| Test | What | Expected |
|---|---|---|
| FT1 | SmolLM3 with no layer removed reproduces the teacher logits; each kept layer keeps its RoPE or NoPE setting | Bitwise in fp32; holds |
| FT2 | Intra-document masking on a synthetic packed batch: no attention across documents, and position ids reset | Holds; recorded |
| FT3 | The k64 parent val on the final set against A2 S | Equal within 1e-6 |
| FT4 | Each run: full LR schedule, data order (raw traces kept), save and reload, as in Stage 1 (T6, T16) | Holds |
| FT5 | The 1.3B: config, parameter count, and the evaluation-code hash against the students' | Recorded; same hash |
| FT6 | The overlap audit (section 3) | Recorded; \|ΔE\| < 0.001 |
| FT7 | The LR state of k6, k20 and k64 (section 3) | Recorded; equals η_pk |

## 9. Budget and outputs

**Budget:** estimates only; report the ledger. There is no cap.

| Part | Estimate (node-hours) |
|---|---|
| F0 | under 0.2 |
| F2 | 2.9, plus 0.7 if the downward extension runs |
| F3 | 2.9, plus about 11.5 if 3.2B runs |
| F4 | 1.8, plus about 5.8 at 3.2B (11.5 if k20 is added) |
| Total | about 8 without the automatic 3.2B runs, up to about 26 with all of them, plus downloads |

The estimates assume about 1.8 node-hours per billion heal tokens for SmolLM3.

**Outputs** go in `results/a2_family_20261107/`:
- `config/`, with this file's SHA-256;
- `manifests/`, with revisions, commits, steps, tokens, LR states, tokenizer and packing details, and document and token counts;
- per-sequence NLL files for every model and run;
- the raw per-rank sample-order traces;
- `runs.jsonl`;
- `tests.json`;
- `overlap_audit/`;
- `decision_family.json`, with every quantity, interval and reading;
- `REPORT.md`, which lists each reading and the full per-run table.
