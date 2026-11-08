# A2 F: the healed length penalty in a second model family (SmolLM3), and a 1.3B from-scratch reference

Version 1.7, 2026-11-09. Written for the agent that runs the A2 experiments.

**Parts F0–F6 are done** under v1.3 and v1.4; the final report came on 8 Nov 21:01. Do not rerun them. Their results, readings and flags stand unchanged.

- **What changed from v1.6 (sent 9 Nov):** run order only. F8 now runs before F7 (owner, 9 Nov 00:25). F8 has the priority; F7 is optional.
  - **If F7 has already started:** pause it at its next saved checkpoint and record the step. Run F8, then resume F7 from that checkpoint with the same data order. FT4's checks apply to the resume.
  - **If any F8 run has already started:** let it finish.
- **What changed from v1.5 (sent 8 Nov):** Part F8 (section 6e) runs A2 Y's cut sizes L3 and L9 at 3.2B on DCLM.
  - **Why:** all four cut sizes are then at one budget. At 3.2B there are only L6 (S1) and L12 so far.
  - **Status:** the owner approved it on 9 Nov 00:09. Like F7, it is an owner-requested extension after the readout.
- **What changed from v1.4 (sent 8 Nov; completed):** Part F7 (section 6d) runs the DCLM stable pair S72/S188 at 3.2B.
  - **Why:** F6's automatic 3.2B fired only on FWE, so on DCLM E_stab has just 0.2B and 0.8B.
  - **Status:** the owner requested it on 8 Nov 23:39, after the final readout. Report it as an owner-requested extension.
  - **No registered reading changes.** This includes FE6 at 0.8B against 0.2B and the joint reading "mixed".
- **What changed from v1.3 (sent 7 Nov; completed):** Part F6 (section 6c). It heals open-sci-ref's stable pair S72/S188 at 0.8B, and at 3.2B if needed, after the SmolLM3 parts. Reason: in the interim report, SmolLM3's E fell from 0.020 (0.2B) to 0.013 (0.8B). Open-sci-ref's stable pair has only been healed at 0.2B, so it is not yet known whether stable-state E falls with budget in open-sci-ref too.
- **What changed from v1.2 (not sent):** Part F5, an automatic 0.8B run on the mixture if Part F2c shows that E depends on the heal data. The mixture stream is now prepared to 0.8B tokens.
- **What changed from v1.1 (not sent):** Part F2c, a small check that heals on a SmolLM3-like data mixture (section 4b). FWE stays the primary heal data.
- **What changed from v1.0 (not sent):**
  - the LR grid is capped at η_pk/2, and E at η_pk/4 is always reported;
  - the optimizer settings are stated;
  - the automatic extensions have a noise floor;
  - the readings have a convergence guard;
  - a second data order is added.
- **This file is the pre-registration.** Record its SHA-256 in every output.
- **It holds every pending part, in run order** (section 1). The pending parts are F8, then F7. The file ends by resuming the A2 D reruns of `A2_depth_spec.md` v1.1.
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

**Pending now (v1.7):**
1. **Pause what is running at its next saved checkpoint.** Record the step. This is the A2 D reruns, or F7 if it has already started under v1.5 or v1.6.
2. **FT10** (section 8), only if this is not wc69, the node that ran A2 Y and F6.
3. **Part F8:** L3 and L9 at 3.2B on DCLM (section 6e). This is the priority.
4. **Part F7:** the DCLM stable pair at 3.2B (section 6d). If it was paused in step 1, resume it from that checkpoint.
5. **The F8 and F7 addendum** to the report (section 9).
6. **Resume the A2 D reruns** per `A2_depth_spec.md` v1.1. They are filler: no claim depends on them, and the owner may stop them at any time.

**Done under v1.3 and v1.4, kept for reference (do not rerun):**
1. **In the background, while the A2 D reruns continue:**
   - download the SmolLM3 checkpoints (section 3);
   - re-tokenize the FWE heal stream and the CulturaX tuning documents with the SmolLM3 tokenizer;
   - build the 0.8B-token mixture stream of Parts F2c and F5 (section 4b).
2. **Pause the A2 D reruns** at their next saved checkpoint once step 1 is done. Record the step.
3. **Part F0:** the 1.3B reference, evaluation only.
4. **Part F1:** SmolLM3 checks FT1–FT4.
5. **Part F2:** LR grid and second data order, then **Part F2c:** the mixture check, then **Part F3:** budgets, then **Part F4:** the third checkpoint, then **Part F5** if triggered.
6. **Part F6:** open-sci-ref's stable pair at larger budgets (section 6c).
   - Download S72 and S188 if they are not on the node. This can run in the background earlier.
   - Run FT9 first.
7. **The report**, covering F0–F6.
8. **Resume the A2 D reruns** per `A2_depth_spec.md` v1.1. They are filler now: no claim depends on them, and the owner may stop them at any time.

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

## 4b. Part F2c: heal-data check on a SmolLM3-like mixture, at 0.2B

**Why:** open-sci-ref healed each parent on its own pretraining corpus. FWE is one part of SmolLM3's stage-1 data, so this check asks whether E changes when SmolLM3 is healed on a sample that follows its stage-1 mixture. FWE stays the primary heal data:
- the open-sci-ref FWE heals used the same documents;
- pruning papers commonly heal on data other than the exact pretraining mix.

**Mixture:**
- **Weights:** use the per-source weights of SmolLM3's published stage-1 configuration. If they are not available, use the blog's stage-1 split:

  | Part | Share | Sources |
  |---|---|---|
  | English web | 73% | FWE and DCLM, half each |
  | Multilingual web | 12% | FineWeb2-HQ in French, Spanish, German, Italian and Portuguese |
  | Code | 12% | The Stack v2 |
  | Math | 3% | FineMath-3+ and InfiWebMath-3+ |

- **Code source:** if The Stack v2 contents need a Software Heritage download, use a code dataset hosted with its contents on Hugging Face. Record which.
- **Size:** prepare 0.8B tokens. Part F2c uses the first 0.2B; Part F5, if it runs, uses all 0.8B. These are small samples per source.
  - Sample documents uniformly within each source with seed 2030.
  - Interleave the sources by weight.
  - Tokenize with the SmolLM3 tokenizer, and pack with intra-document masking as in section 3.
  - For FWE and DCLM, take documents not used in the primary FWE stream.
- **Missing sources:** if a source cannot be obtained, spread its weight over the others in proportion, report it and continue.
- **Overlap:** run the overlap audit of section 3 on every source of the mixture.

**Runs:** k6 and k20 at η*, 0.2B (762 updates): 2 runs. Run IDs are `F2c-mix-<k>-BS1-0.2B-lr<η*>-s2030`.

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

## 6b. Part F5: the mixture at 0.8B (automatic)

- **Trigger:** FE4 (section 7) reads "data-dependent".
- **Runs:** k6 and k20 on the mixture stream at η*, 0.8B (3,051 updates): 2 runs. Run IDs are `F5-mix-<k>-BS1-0.8B-lr<η*>-s2030`.
- **Report:** E(mixture, 0.8B) next to E(FWE, 0.8B), and FE1 on the mixture as a secondary reading.
- **If the trigger does not fire,** skip this part and say so in the report.

## 6c. Part F6: open-sci-ref's stable pair at larger budgets

**Why:** the SmolLM3 checkpoints are all stable, and healing anneals them. The S1 heals of open-sci-ref's stable pair exist only at 0.2B (A2 S): E_stab = +0.011 (DCLM) and +0.015 (FWE). F6 shows whether stable-state E also falls with budget in open-sci-ref, so that SmolLM3 can be compared like for like.

**Arms:**
- S72 and S188 are the stable checkpoints of the 1T runs, at steps 72,000 and 188,000 (≈297B and ≈776B tokens). Each is its own teacher, as in A2 S.
- Use the same revisions as A2 S. Download them if they are not on the node.

**Recipe:** exactly as A2 S Part 2, which is the Stage 1 recipe:
- S1 (layers 16–21 removed);
- LR 1e-3 (= η_pk/4);
- each dataset's own heal stream, seed 2030;
- batch 64 × 4,096 tokens.

**Runs:**
- **FT9, cross-node check:** rerun `ST1-dclm-S72-S1-0.2B-lr1e-3` once on this node. Its final-set val should be within 0.002 of A2 S's value. If it is not, flag it and continue.
- **0.8B (3,051 updates):** S72 and S188 on DCLM and FWE: 4 runs, `ST1-<ds>-<arm>-S1-0.8B-lr1e-3`.
- **Automatic 3.2B (12,207 updates), on at most one dataset:** S72 and S188, 2 runs.
  - Trigger: for a dataset, |E_stab(0.8B) − E_stab(0.2B)| > max(0.2 × |E_stab(0.2B)|, 0.003).
  - If it fires on both datasets, use the one with the larger relative change.

**0.2B values:** use A2 S's (DCLM: Δ_post 0.158411477 and 0.169543452; FWE: 0.146999744 and 0.161780488). Note the node they came from; the cross-node difference has been about 1e-4.

**Overlap:** for the FWE runs, run the overlap audit of section 3 (the 3 known documents) and report the reduced-set E.

**Outcome (8 Nov 21:01):** the 3.2B trigger fired on FWE only.
- FWE: E_stab = 0.0148 / 0.0086 / 0.0050 at 0.2 / 0.8 / 3.2B.
- DCLM: E_stab = 0.0111 / 0.0086 at 0.2 / 0.8B.

## 6d. Part F7: the DCLM stable pair at 3.2B (owner-requested, v1.5)

**Why:** this fills the one missing point of the four open-sci-ref series (annealed or stable, DCLM or FWE) at 3.2B. On DCLM, stable E fell by 0.0025 from 0.2B to 0.8B, inside ±0.003, so FE6 read "flat". With two points, a slower fall cannot be told apart from no fall.

**Runs:** S72 and S188 on DCLM at 3.2B (12,207 updates): 2 runs, `ST1-dclm-S72-S1-3.2B-lr1e-3` and `ST1-dclm-S188-S1-3.2B-lr1e-3`.
- The recipe is exactly that of Part F6 and A2 S Part 2:
  - S1 (layers 16–21 removed);
  - LR 1e-3, 2% warm-up, then linear decay to 0;
  - the DCLM heal stream, seed 2030;
  - batch 64 × 4,096 tokens;
  - each checkpoint is its own teacher.
- Use the same S72 and S188 revisions as F6.

**Same-node comparison:**
- Compare 3.2B with F6's 0.8B DCLM runs.
- If F7 runs on a different node from F6, run FT10 first (section 8). Flag the result if it fails, and continue.

**Report:**
- Δ_pre, Δ_post and the parent vals.
- E_stab at 3.2B with its paired-sequence bootstrap interval, using F6's estimator.
- E_stab / G_stab.
- E_stab / Δ_post(S72).
- The reading FE6-DCLM-3.2B (section 7).
- The full DCLM series at 0.2 / 0.8 / 3.2B next to the FWE series.

## 6e. Part F8: cut sizes L3 and L9 at 3.2B on DCLM (owner-approved, v1.6)

**Why:** this puts every cut size at one budget. The cut-size result rests on two matched comparisons so far:
- L3, L6 and L9 at 0.8B (E = 0.0186 / 0.0208 / 0.0223);
- L6 and L12 at 3.2B (0.0217 / 0.0233).

L12 was far from healed at 0.8B (E = 0.0134). F8 adds L3 and L9 at 3.2B, so that all four cut sizes are at 3.2B.

**Runs:** P300 and P1T, each healed from its own unpruned parent as teacher, at 3.2B (12,207 updates): 4 runs.
- `Y3-dclm-P300-L3-3.2B`, `Y3-dclm-P1T-L3-3.2B`: layers 19–21 removed (21 kept).
- `Y3-dclm-P300-L9-3.2B`, `Y3-dclm-P1T-L9-3.2B`: layers 13–21 removed (15 kept).

**Recipe:** exactly as A2 Y Part 3 (`A2_why_spec.md` v1.0, section 3), which is the Stage 1 recipe:
- forward-KL KD;
- LR 1e-3, 2% warm-up, then linear decay to 0;
- the DCLM heal stream, seed 2030;
- batch 64 × 4,096 tokens;
- the same evaluation sets, code and per-sequence storage.

**Report:**
- Δ_pre, Δ_post and the parent vals.
- E at 3.2B for L3 and L9, with the paired bootstrap interval of A2 Y.
- The four-cut table at 3.2B on DCLM:
  - L3 and L9 from F8;
  - L6 from Stage 1;
  - L12 from A2 Y.

  Record the node of each value. E is a within-node difference, so node offsets of about 1e-4 cancel to first order.
- The change in E from 0.8B to 3.2B for L3 and L9.
- E against the number of layers removed: the least-squares slope per layer and E(12) − E(3).
- The share of the parent gain kept, (G − E) / G, per cut.
- The reading FY3-3.2B (section 7).

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
- **FE4, heal data, at η* and 0.2B:**
  - "robust" if |E(mixture) − E(FWE, s2030)| < 0.003;
  - otherwise "data-dependent", with its direction.

  Also report both E values next to the data-order difference |E(s2030) − E(s2031)|.
- **Convergence guard.** Add "not converged" to FE1 or FE3 if its quantity (E, or D64) changed by more than max(20%, 0.003) between the two largest budgets run.
- **FE6, open-sci-ref's stable pair across budgets.** E_stab = Δ_post(S188) − Δ_post(S72). Read it per dataset, comparing 0.8B with 0.2B:
  - "falls" if E_stab drops by 0.003 or more;
  - "rises" if it increases by 0.003 or more;
  - "flat" otherwise.

  Repeat the reading for 3.2B against 0.8B where 3.2B ran.
- **FE6-DCLM-3.2B (Part F7, v1.5; owner-requested, secondary):**
  - Apply the FE6 rule to DCLM's E_stab at 3.2B against 0.8B: "falls" if it drops by 0.003 or more, "rises" if it increases by 0.003 or more, "flat" otherwise.
  - Also report the change from 0.2B to 3.2B.
  - This reading is secondary. It does not replace FE6 at 0.8B against 0.2B, and it does not change the joint reading below.
- **FY3-3.2B (Part F8, v1.6; owner-approved, secondary, DCLM only):** apply A2 Y's Part 3 rule at 3.2B, with τ = 0.003:
  - "A supported" if E(3) < E(6) < E(9) < E(12);
  - "A not supported" if E(12) − E(3) < τ;
  - "mixed" otherwise.

  A2 Y's rule asks for both datasets, and this reading uses DCLM alone. It does not replace A2 Y's registered verdict at 0.8B ("A not supported").
  - Also label each of L3 and L9 "flat" if its E changed by less than τ from 0.8B to 3.2B, and otherwise "changing", with the direction.
- **Joint reading of the two families at the stable state:**
  - "stable-state effect" if FE6 reads "falls" on both datasets and SmolLM3's FE2 reads "changing" downward. SmolLM3 then behaves like open-sci-ref in the same state.
  - "family difference" if FE6 reads "flat" on both datasets while SmolLM3's E falls.
  - "mixed" otherwise.
- **Descriptive, for F6:**
  - E_stab / G_stab, with G_stab = val(S72) − val(S188);
  - E_stab / Δ_post(S72) at each budget, next to SmolLM3's E / Δ_post(k6) (0.083 at 0.2B, 0.081 at 0.8B). This shows whether stable-state E shrinks in proportion to the remaining damage in both families.

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
| FT8 | The mixture stream of Part F2c: sources, weights used, token count per source, sampling seed | Recorded; shares within 1% of the weights |
| FT9 | Part F6 cross-node check: rerun `ST1-dclm-S72-S1-0.2B-lr1e-3` on this node | Final-set val within 0.002 of A2 S |
| FT10 | Parts F7 and F8, only if they run on a node other than wc69, where A2 Y and F6 ran: rerun `ST1-dclm-S72-S1-0.2B-lr1e-3` there | Final-set val within 0.002 of F6's FT9 rerun (3.066636571) |

## 9. Budget and outputs

**Budget:** estimates only; report the ledger. There is no cap.

| Part | Estimate (node-hours) |
|---|---|
| F0 | under 0.2 |
| F2 | 2.9, plus 0.7 if the downward extension runs |
| F2c | 0.7, plus data preparation on CPU |
| F5 (only if FE4 reads "data-dependent") | 2.9 |
| F6 (open-sci-ref, about 1 node-hour per billion heal tokens) | 3.4: FT9 0.2 plus four 0.8B runs; plus about 6.4 if 3.2B runs |
| F3 | 2.9, plus about 11.5 if 3.2B runs |
| F4 | 1.8, plus about 5.8 at 3.2B (11.5 if k20 is added) |
| Total, F0–F6 (done) | about 12 without the automatic runs, up to about 40 with all of them, plus downloads |
| F7 (v1.5) | about 6.6: F6's FWE 3.2B runs took 3.3 hours each. Add 0.25 if FT10 runs |
| F8 (v1.6) | about 13.2: L3 about 3.7 hours per run (21 layers kept), L9 about 2.9 (15 kept), scaled from S1's 3.3 |
| Total, F8 then F7 | about 20, plus the addendum. F8 alone is about 13 |

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

**For F7 and F8:**
- Append a section "Family v1.7 — F8 and F7" to `REPORT.md`.
- Add their quantities to `decision_family.json` under new keys. Do not change any F0–F6 value.
- With each run, record the SHA-256 of the version it started under:
  - v1.5: `f771ad22ab5f8afb572af2101de300c2a423a95af72b3b01fee3b92259ba35fe`;
  - v1.6: `82f834236e0a23f33cd08ad9817b985d0608aaad684074212e49bdb25f8d97a8`;
  - or this file's.

  Record any pause and resume.
