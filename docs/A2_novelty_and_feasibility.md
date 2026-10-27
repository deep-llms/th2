# A2 deep check: are overtrained models harder to depth-prune? Novelty and feasibility

Date: 2026-10-28. This note supersedes the A2 part of section 5.1 in `IDEA2_deep_dive_gaps_and_projects.md` (version 2.1).

**How it was made:**
- **Novelty search:** an independent agent ran 78 web searches and 9 OpenReview API searches, then read the closest papers in full.
- **Design review:** a second independent agent estimated effect sizes, power, confounds and compute, with scripts in the session scratchpad (`a2_feasibility/`).
- **My own checks:** I verified the two decisive claims myself: the Gromov et al. quotes, and the Catalan-Tatjer et al. abstract.

## 0. Verdict in brief

**Novelty: the exact question is open, but the margin is narrow.**
- **Nobody has done the exact experiment.** We found no work that varies the parent's tokens per parameter (TPP) at a matched learning-rate (LR) state, then removes layers or width and heals.
- **Two well-known papers name it as future work.**
  - Gromov et al. (ICLR 2025): "are scaling-law over-trained or distilled models more difficult to prune?", and pruning "across different pretraining checkpoints".
  - Springer et al. (ICML 2025): "We leave further modifications such as reinforcement learning and pruning to future work."
- **The analogy that motivated A2 has weakened.** Catalan-Tatjer, Ajroldi and Geiping (ICLR 2026) show that for quantization, the damage tracks LR decay rather than training length. Validation loss and quantization error diverge once the LR decays, and this divergence is "largely independent of training data scale".
  - Their method (WSD cooldown branches) is the same as ours. Applying it to pruning will look incremental unless the result differs from theirs.

**Feasibility: as planned, weak; restructured, reasonable.**
- **The plan as written** (three parents at 20/40/80 TPP from one run) gives about a 37% chance of a clean result, and about 20% of an ICML-grade one (design review).
  - The expected effect after healing is about 0.01 nats, and theory does not fix its sign.
  - Artefacts of the healing recipe are about the same size.
- **A restructured design** gives about 60–65% for a clean result and about 30–35% for ICML grade, at a cost of about 16 more node-hours. It adds:
  - a wider TPP range;
  - a second trajectory;
  - an external replication;
  - pure KD healing with a learning rate tuned per parent;
  - LR state as a second factor.
- **My adjustment:** given Catalan-Tatjer et al., I would lower the chance of a clean "harder" result. Overall I estimate ICML grade at about 25–30% and TMLR grade at about 55–60%.

**Decision:**
- **Do not commit to A2 as the headline yet.**
- **Reframe it** as "Overtrained or just annealed? Training length vs LR state in depth pruning with healing".
- **Run the pilot in two phases** with pre-registered GO / REFRAME / STOP rules (section 5):
  - **Phase 0** uses only public open-sci-ref models, with no pretraining: about 23 node-hours. It tests training length (Phase 0A) and, since 28 Oct, the LR state as well (Phase 0B, on intermediate checkpoints of the same runs).
  - **Phase 1** trains our own narrow WSD family, only if Phase 0 shows a signal. Its LR-state part, Phase 1(a), is deferred.

  The terms "narrow model" and "WSD branches" are explained in the appendix.

## 1. Closest work

| Work | What it varies | What it compresses | Healing | Result | Read |
|---|---|---|---|---|---|
| **Catalan-Tatjer, Ajroldi, Geiping** (ICLR 2026) | Open training trajectories (up to 32B parameters, 15T tokens). Controlled runs (160M) with WSD cooldowns branched at 12–82B tokens, plus a cosine replication | Post-training quantization (PTQ) | — | Quantization error stays flat in the stable phase and rises when the LR decays. After cooldown, models reach comparable quantization error. "More training data does not inherently hurt quantization." **This is A2's design template, applied to PTQ, with no pruning** | Full (agent); abstract (me) |
| Kumar et al., Scaling Laws for Precision (ICLR 2025) | D and N, each run with its own cosine schedule | PTQ | — | δ_PTQ ∝ D^0.507 / N^0.344; at fixed N, 4× more data gives about 2× the damage. No pruning | Full |
| Ouyang et al., Low-Bit Quantization Favors Undertrained LLMs (ACL 2025) | Pythia checkpoints (LR state not controlled) | PTQ | — | ΔqLoss ∝ D^0.525. No pruning | Full |
| Springer et al., Overtrained LMs Are Harder to Fine-Tune (ICML 2025) | OLMo-1B checkpoints; controlled 15–90M runs annealed to zero | Gaussian noise, fine-tuning | — | Sensitivity rises with tokens. A larger fine-tuning LR brings the drop earlier and makes it bigger. **Pruning left to future work** | Full |
| **IDEA Prune** (NeurIPS 2026; arXiv 2503.05920) | LR state at equal tokens: the midpoint of a 2T cosine run vs the end of a 1T run | **Width only** (FFN neurons): "We do not prune the depth" | Recovery under one cosine schedule; KD variant | The intermediate start helps MMLU (46.4 vs 38.2), while perplexity is equal (8.88 vs 8.89). The authors attribute this to an LR mismatch on re-warming. No WSD, and TPP is not varied | v1 full |
| P² Law (ACL 2025) | Parent size, pruning ratio, healing tokens | Depth, width, 2:4 | Continued pretraining, LR 2e-5 | L = **L0 +** f(ratio, N0, D). The parent's pre-pruning loss enters only as an additive baseline; parent pretraining tokens are never varied. **A2 tests this assumption** | Full |
| Distillation Scaling Laws (Busbridge et al., ICML 2025) | Teacher quality | — (distillation) | — | The teacher affects the student "only through the resulting teacher cross-entropy". This predicts no TPP effect beyond the parent's loss | Abstract and targeted |
| Gromov et al. (ICLR 2025) | Model families | Depth | QLoRA | Hypothesis that "the smaller models are more overtrained, making parameters less redundant". **Poses A2 as future work** | Targeted (verified) |
| He et al., What Matters in Transformers (TMLR 2026) | MAP-Neo-7B checkpoints every 500B tokens | Attention / MLP / block dropping | None | Block and MLP importance rise during training. LR state not discussed | Targeted |
| Persistent Depth Ordering (2610.01165) | Five public trajectories, one block bypassed at a time | Bypass | None | Mixed: damage rises in 8 of 11 model–domain pairs and falls in OLMo-2 7B | Full |
| Cherilyn & Safaai (2604.23475) | OLMo-2-7B checkpoints | FFN channels (final model only) | None | The share of loss sensitivity held by the top 1% of FFN channels rises from 18.6% to 71.7% over pretraining | Targeted |
| Watts, Li, Goyal, Springer, Raghunathan, Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting (arXiv 2605.02105, May 2026; added 28 Oct from the implementer's review) | Flatness-promoting pretraining: SAM, large LR, **shorter annealing**; 20–150M models, plus SAM mid-training on OLMo-2-1B | 4-bit quantization; post-training (forgetting) | — | Flatter base models lose less after post-training (up to 80% less forgetting) and after 4-bit quantization (40% less on OLMo-2-1B). **No pruning.** It predicts that less-annealed parents may survive perturbation better, which bears on A_init | Abstract |
| Renda, Frankle, Carbin, Comparing Rewinding and Fine-tuning in Neural Network Pruning (ICLR 2020; arXiv 2003.02389) | The retraining schedule after pruning: fine-tuning at a small LR vs weight or LR rewinding | Unstructured and structured pruning (vision, translation) | Retraining | Retraining with the original LR schedule (LR rewinding) recovers more than fine-tuning at a small LR. The healing LR matters, which is why the pilot tunes it per arm and flags grid-edge choices | Known (not re-read) |
| Liew et al., upcycling scaling laws (ICML 2025) | Dense tokens × upcycling tokens (WSD stable checkpoints) | Upcycling to MoE (the reverse of pruning) | Continued training | More parent tokens give a lower start but slower progress. This is an interaction form a reviewer may cite | Targeted |
| Sheared LLaMA, App. F.6 (ICLR 2024) | LLaMA1 (1T) vs LLaMA2 (2T) as the parent; data also differs | Depth and width | Continued pretraining | Pruning from LLaMA2 is "consistently" better, but "not very large" | Targeted |
| Small LLMs: Pruning vs Scratch (2606.14150) | Pruning method, ratio, retraining budget; the parent is fixed at 200B tokens | Depth, width, 2:4 | LM retraining | Parent length not varied | Full |
| Frantar et al., sparse scaling laws (ICLR 2024) | Sparsity × size × data | Unstructured, during training | — | Sparsity "does not appear to interact significantly with the data scaling" | Full |
| The Journey Matters (ICLR 2025) | When pruning starts during pretraining | Unstructured, during training | — | Starting at 25% and ending at 75% of compute is near-optimal | Full |

## 2. What is open, and the honest novelty margin

**Open:**
1. Depth (and width) pruning plus healing across parent TPP, with the LR state matched.
2. LR state as a second factor for depth pruning under WSD. IDEA Prune covers only width pruning, under one cosine schedule.
3. Whether healing erases any difference.
4. A direct test of the assumption that "the parent matters only through its loss" (P² Law; distillation scaling laws).

**What reviewers will cite as already known:**
- Overtrained models are fragile under noise and quantization (Springer; Kumar; Ouyang).
- The LR-decay confound and its control protocol (Catalan-Tatjer).
- The advantage of starting from an unannealed checkpoint, for width pruning (IDEA Prune; in vision: early-bird tickets, LR rewinding).
- The interaction between parent tokens and continued-training tokens (Liew et al., for upcycling).

**How results would be read:**
- A null result would be read as "consistent with Catalan-Tatjer".
- A TPP effect that survives LR matching and healing, in two model families, would be the strong paper.

**Scoop risk:** moderate over the next three months. The question is posed in public by Gromov et al., and the Catalan-Tatjer–Geiping group has the exact protocol.

## 3. Feasibility (design review)

**Effect sizes** for removing 8 of 28 layers at 0.6B:
- **Damage Δ after 1B tokens of KD:** about 0.06–0.25 nats (central about 0.12).
- **The parent's own gain** from 12B to 48B tokens: about 0.17–0.20 nats.
- **The post-healing contrast** Δ(80 TPP) − Δ(20 TPP): plausibly 0.003–0.03 nats. Theory does not fix its sign:
  - a capacity term gives about +0.008 per 4× (Kaplan form) or 0 (Chinchilla form);
  - sensitivity points to "harder";
  - deep layers drifting toward identity points to "easier".
- **For scale:** 4-bit quantization damage would go from about 0.009 to 0.019 nats over this range.

**Noise is not the problem.** Within one trajectory, 8 paired layer sets detect about 0.004–0.005 nats. Buy more layer sets rather than healing seeds.

**The real problems:**
- **Effective n = 3 parents from one trajectory.** With per-parent offsets of about 0.003 nats, which can come from cooldown data, repetition or LR-tuning error, a naive per-set test gives false positives 59% of the time. The parent must be the unit of analysis, with 4–5 or more parents per family.
- **Too narrow a range.** 4× is too narrow for a scaling claim or a convincing null. It also sits below the thousands of TPP where overtraining effects have been seen; the fine-tuning effect appeared near 2,500 TPP for OLMo-1B.
- **Healing-recipe artefacts of the same size as the effect:**
  - **Fresh data:** KD plus LM loss on fresh data adds about +0.01 nats of spurious "harder", because fresh tokens help the 12B-token parent more than the 48B one.
  - **One LR for all parents:** a single healing LR tuned on one parent biases toward "harder".
  - **Different healing fractions:** the healing budget is 8.3%, 4.2% and 2.1% of pretraining across the three parents.
- **Repeated data.** The 80-TPP point repeats data (30B-token English pool).

**Outcome probabilities** (design review, before accounting for Catalan-Tatjer):

| Pattern | Chance it is true | Clean result, as planned | Clean result, restructured | Reception |
|---|---|---|---|---|
| Harder | 0.40 | 0.14 | 0.30 | Best story, if dose-response holds over ≥ 1.5 decades in two families |
| Easier | 0.15 | 0.07 | 0.12 | Surprising; needs a mechanism |
| Difference only before healing | 0.34 | 0.12 | 0.19 | Moderate; TMLR, or ICML with a wide range |
| No difference | 0.11 | 0.04 | 0.05 | Boring |
| **Any clean result** | | **≈ 0.37** | **≈ 0.66** | |
| **ICML grade** | | **≈ 0.20** | **≈ 0.35** | |

**My adjustment:**
- Catalan-Tatjer lowers the prior on "harder at matched annealing": for quantization, the effect was driven by LR decay.
- A "harder" result can still come from the capacity route (a better parent opens a larger teacher–student gap). But reviewers would call that capacity unless the excess over a same-size model trained from scratch also grows.

## 4. Revised design (if the pilot says go)

**Two factors:** TPP × LR state, comparing the stable (pre-decay) checkpoint with the annealed checkpoint at the same token count.

**Range and replication:**
- **A narrow 28-layer family** (d = 512, 88M non-embedding parameters): a WSD trunk to about 27B tokens, with cooldowns at 1.7B, 6.75B and 27B tokens (19 / 77 / 306 non-embedding TPP) on unique data. A second trajectory (new seed) has cooldowns at 1.7B and 6.75B.
- **The 0.6B family:** cooldowns at 3, 6, 12 and 24B tokens. The 48B branch runs only if the pilot passes, and is labelled as repeated data.
- **External:**
  - **open-sci-ref 1.7B** at 300B vs 1T tokens on three datasets (DCLM, FineWeb-Edu, Nemotron-CC-HQ). These are WSD runs with a 20% cooldown and the same peak LR, warm-up and batch size. The checkpoints are public.
  - **Qwen3-0.6B-Base** as a loose anchor; its data and recipe differ.

**Healing:**
- pure forward-KL KD (full logits, or the top 64), on a data slice held out of every pretraining stream;
- the healing LR tuned per parent on a 3-point grid; also report fixed-LR results;
- also report results at a matched ratio of healing to pretraining tokens.

**Analysis:**
- Parents are the unit: fit a mixed model of Δ against log₂ (non-embedding TPP).
- Use 6–8 pre-registered layer sets, with the good stratum as primary, plus each parent's best of K.
- **Metrics:**
  - absolute Δ (primary);
  - the excess over a same-size model trained from scratch at the same tokens;
  - the recovered fraction;
  - the pruned model's gain from extra pretraining relative to the parent's gain.

**Mechanism measurements for every parent:**
- Δ before healing, and after linear repair;
- per-layer bypass profiles, ‖f(x)‖/‖x‖ and Block Influence;
- sensitivity to Gaussian perturbation;
- Δ as a function of healing tokens (0 to 1B; 3B on two parents), to separate the asymptote from the rate.

**Hypotheses and their signatures** (to pre-register):

| Hypothesis | Δ before healing | Δ after healing | Signature |
|---|---|---|---|
| Sensitivity (as in PTQ and fine-tuning) | rises | rises | Gaussian sensitivity rises; the best healing LR falls |
| Capacity / teacher gap | rises or flat | rises | The healing-curve asymptote rises; the excess over scratch rises |
| Identity drift of deep layers | falls for deep sets | falls | ‖f(x)‖/‖x‖ and Block Influence of deep layers fall; the best set moves deeper |
| Healing erases it | differs | flat | The recovered fraction is higher at high TPP |
| LR state, not length | differs between stable and annealed | — | The effect appears between stable and annealed checkpoints, not across TPP |
| Recipe artefact | flat | rises | Disappears under pure KD with a per-parent LR |

**Compute:** about 68 node-hours restructured, vs about 53 as written. Possible cuts:
- cached top-16 logits (−5 h);
- 6 layer sets instead of 8 (−5 h);
- deferring the 48B branch (−16.5 h).

## 5. Pilot in two phases

**The executable protocol is `A2_pilot_spec.md` (version 2.1).** Where it differs from this section, the spec wins. Its differences, each with its reason in the spec:
- Phase 0 uses the open-sci-ref **v0.01** pairs, whose matched recipes are documented in their run names;
- **Phase 0B replaces Phase 1(a)** (28 Oct): the stable-vs-annealed test now runs on public checkpoints of the same runs (section "Phase 0B" below). Phase 1(a) is deferred, unchanged in substance, to `A2_phase1a_deferred_spec.md`;
- the supplementary 100B point is dropped;
- the BI set is chosen on the mean of a dataset's four parents, and may not coincide with another layer set;
- the decision labels are evaluated in a fixed order;
- in STOP's sign clause, a dataset whose |E_d| is below 0.003 does not count as disagreeing;
- REFRAME needs the three pre-healing contrasts E_pre,d to share a sign, and a failed run gives the label INCOMPLETE rather than GREY;
- (version 2.1, after the implementer's review) healing LRs, the BI set and the divergence check are chosen on a tuning set of unused reserved training documents, and the validation split only scores;
- loss-based checks are diagnostics rather than exclusion rules;
- the scored precision of healed students is fixed in advance;
- the causal wording is narrowed: the pilot screens under one parent, teacher and healing recipe.

### Phase 0: public models only, no pretraining (about half a node-day)

**Parents.** All six are open-sci-ref v0.01 models: 1.7B, 24 layers, width 2,048, GPT-NeoX tokenizer (50,304 tokens).
- Each was trained as its own WSD run: 20% cooldown to zero, peak LR 4e-3, 25,000 warm-up steps, about 4.1M tokens per batch, context 4,096.
- The open-sci-ref paper (its Table 4) lists identical settings for the 300B and 1T runs apart from length.
- Tokens per parameter (TPP) below are computed on about 1.61B non-embedding parameters.

| Dataset | 300B tokens (≈ 186 TPP) | 1T tokens (≈ 621 TPP) |
|---|---|---|
| DCLM | `open-sci/open-sci-ref-v0.01-1.7b-dclm-300B-4096` | `open-sci/open-sci-ref-v0.01-1.7b-dclm-1T-4096` |
| FineWeb-Edu | `open-sci/open-sci-ref-v0.01-1.7b-fineweb-edu-1.4t-300B-4096` | `open-sci/open-sci-ref-v0.01-1.7b-fineweb-edu-1.4t-1T-4096` |
| Nemotron-HQ | `open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-300B-4096` | `open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-4096` |

- **No 100B point.** The `laion/open-sci-ref-1.7b-nemotron-hq-100B-4096` model uses RoPE θ = 100k, unlike these parents (10k).
- **Checked on 2026-10-28** (the spec's test T2 repeats the checks):
  - In all three pairs, the Megatron run names in `config.json` match except for the length: global batch 1,008, context 4,096, WSD, LR 4e-3, warm-up 25,000, RoPE θ 10,000.
  - All six repositories hold the same `modeling_opensci.py`, `configuration_opensci.py` and tokenizer files.
- **The repositories also hold intermediate checkpoints**, as branches every 2,000 iterations: the 300B repositories from 2,000 to 72,000, and the 1T repositories from 58,000 to 242,000.
  - In each pair, the `iter_0058000` weights are identical in the two repositories (same SHA-256). So the two runs share one trunk up to that point: the 300B parent is a cooldown branch of the trunk that the 1T run continues.
  - This allows the stable-vs-annealed comparison at 1.7B: the 1T run's checkpoint at 72,000 iterations (297B tokens, LR still at its peak) against the annealed 300B parent (300B tokens). Phase 0B below uses it.

**Cut:** remove 6 of 24 layers (25%), as 3 of 12 two-layer groups (g0 = layers 0–1, …, g11 = layers 22–23). Each dataset gets four fixed sets, used for both its 300B and 1T parents:
1. **Deep block:** g8–g10 (layers 16–21), keeping the last group.
2. **Block Influence:** the 3 groups with the lowest mean BI over the dataset's two parents.
3. **Interleaved:** g3, g6, g9 (layers 6–7, 12–13, 18–19).
4. **Random:** g1, g2, g6, drawn once from g1–g10 with seed 2027.

**Measure before healing** (minutes each):
- Δ before healing, for the 4 sets;
- the single-layer bypass profile (24 layers);
- ‖f(x)‖/‖x‖ and BI per layer;
- the loss increase under small Gaussian weight noise.

**Heal.** Pure forward-KL KD from the unpruned parent. The data is a neutral held-out corpus that no parent trained on: a CulturaX English slice, re-tokenised with GPT-NeoX, with the same tokens for every run.
- **LR:** picked per parent from {3e-5, 1e-4, 3e-4}, on the deep block at 0.03B tokens. Short warm-up, then linear decay to zero.
- **Budgets:**
  - all 4 sets at 0.05B tokens;
  - the deep block and BI set at 0.2B tokens;
  - the deep block on the Nemotron pair at 0.8B tokens, to see whether the gap keeps closing.

**Evaluate:** validation loss on a separate neutral slice (primary), and on each parent's own dataset (secondary).

**Compute:** about 5.7B KD tokens at 1.7B, or roughly 10–12 node-hours at the team's measured throughput. Measurements add about 1 hour. In total, about half a node-day.

**Decision rule.**
- For each dataset d, let E_d = Δ(1T) − Δ(300B) after 0.2B healing tokens, averaged over the deep-block and BI sets, with each parent at its best LR. Let E_pre,d be the same contrast before healing.
- Let E be the mean over the three datasets.

Outcomes:
- **GO to Phase 1(b)** if all of these hold:
  - |E| ≥ 0.008 nats;
  - all three E_d have the same sign;
  - at least 10 of the 12 set × dataset contrasts at 0.05B tokens have that sign.
- **REFRAME** if |E| < 0.003 but |E_pre| ≥ 0.05 nats, with all three datasets' E_pre,d of the same sign: healing erases a real difference that exists before healing.
- **STOP the TPP half of A2** if |E| < 0.003 and either |E_pre| < 0.03 or the E_pre,d differ in sign; or if the datasets disagree in the sign of E_d (ignoring any dataset with |E_d| < 0.003). Phase 0B's LR-state result stands on its own.
- **GREY** (anything in between): run Phase 1(b), whose range is 16× instead of 3.3×.

**Why 0.008 nats:**
- The noise per heal is about 0.001–0.002 nats, and offsets between parent pairs are about 0.003. So the standard error of E is about 0.002, and 0.008 is about 4 standard errors.
- Per doubling of TPP, 0.008 over 3.3× (≈ 0.0046) matches the narrow-family rule (0.015 over 16×, ≈ 0.0038).

### Phase 0B: the LR state on public checkpoints (about 11 node-hours)

Added on 28 Oct, replacing Phase 1(a) in the pilot. Each dataset's 300B and 1T runs form one WSD tree, so four public parents per dataset are available:
- **S72:** the 1T run at step 72,000 (297B tokens, LR still at its peak);
- **P300:** the annealed 300B parent;
- **S188:** the 1T run at step 188,000 (776B tokens, LR still at its peak);
- **P1T:** the annealed 1T parent.

**Arms**, each with an LR grid, then heals with the deep-block and BI sets at 0.2B tokens:
- **S72** from itself;
- **X72:** S72 pruned, then healed from P300;
- **S188** from itself.

S72 and S188 also get the four sets at 0.05B tokens, as the Phase 0 parents do. P300's runs come from Phase 0.

**Questions:**
- **"Prune before you anneal":** compare X72 with P300, with the teacher held fixed, on 6 values (3 datasets × 2 sets) at 0.2B.
  - The label needs a mean of at least 0.01 nats in either direction, with at least 5 of 6 values agreeing.
  - It carries a flag if X72's best healing LR is at the top of the grid. X72 starts behind its teacher by the whole annealing gain, so a capped LR might leave that gap open.
- **TPP at the stable state:** compare S188 with S72 within one run (2.6×), with exactly the rules of Phase 0.
- **Interaction:** whether the TPP effect per unit of log length differs between the annealed pair (Phase 0) and the stable pair (Phase 0B).
  - If the effect appears only after annealing, the damage tracks the anneal, as Catalan-Tatjer et al. found for quantization.
  - The stable pair ends at 776B tokens rather than 1T, which the report must keep in mind.

**What Phase 0 still does not answer:** annealed parents below about 186 TPP, and the Qwen3 architecture with document masking. These are Phase 1's job.

### Phase 1: our own narrow WSD family (about 0.15–0.8 node-day)

**Phase 1(a): deferred.** It is the same stable-vs-annealed test on our own narrow model:
- **Model:** one narrow trunk trained to 6.75B tokens.
- **Comparison:** P77 (annealed) vs S77 (stable checkpoint at the same tokens), with arm X (S77 healed from P77).
- **Heal:** 4 layer sets, pure KD at 0.07B and 0.28B tokens, LR picked per parent.

When this section was first written we believed public checkpoints could not provide this test. The v0.01 intermediate checkpoints can, at 1.7B scale, so Phase 0B took its place in the pilot. Phase 1(a) remains useful later, for a second architecture and exactly matched tokens. Its spec is `A2_phase1a_deferred_spec.md`.

**Phase 1(b): Arm N (runs only if Phase 0 is GO or GREY), the narrow family above.**
- **Layer sets:** 4 pre-registered sets of 4 of 14 groups: the deep block, a Block-Influence set, an interleaved set and one random set.
- **Per parent:**
  - Δ before healing;
  - bypass profiles;
  - a healing-LR grid at 0.07B tokens;
  - pure KD at 0.07B and 0.28B tokens.
- **Trajectory 2:** two sets at 0.28B.
- The comparison of stable and annealed checkpoints is Phase 0B (and, later, Phase 1(a)).

**Arm W:** the 0.6B parent at 12B tokens vs a 3B cooldown branch of the same run, plus Qwen3-0.6B-Base; two sets at 0.2B, two LRs.

**Arm P** has moved to Phase 0.

**Decision rule.** Let E = Δ(306) − Δ(19) in Arm N, averaged over the 4 sets at 0.28B tokens, with each parent at its best LR. Let E_pre be the same contrast before healing.
- **GO** (full restructured A2), if all of these hold:
  - |E| ≥ 0.015 nats;
  - at least 3 of 4 sets agree in sign;
  - trajectory 2 agrees;
  - the 77-TPP point is not reversed;
  - Arm W, or at least 2 of 3 Arm-P datasets, agree.
- **REFRAME** ("healing erases it"), if:
  - |E| < 0.005;
  - every per-set contrast is within ±0.01;
  - |E_pre| ≥ 0.1.

  Then write a lean paper on before vs after healing, framed as a test of P² Law's assumption.
- **STOP**, if |E| < 0.005 and |E_pre| < 0.05, or if signs disagree across sets or trajectories. A2 becomes a section of A1.
- **GREY** (0.005 ≤ |E| < 0.015): add 4 sets and 0.7B-token heals on 2 sets (about 1.5 h), then apply the rule again.
- **Separately:** a large stable-vs-annealed difference is a practical result on its own: "prune before you anneal" for depth pruning under WSD. It extends IDEA Prune from width pruning under cosine to depth pruning under WSD.

## 6. What the outcomes mean for the portfolio

| Pilot result | Project A | Paper |
|---|---|---|
| GO | A2 stays the headline; full restructured design | ICML 2027 possible |
| REFRAME | A2 becomes "pruning damage differs before healing, not after", combined with A1 (healing-budget law) | TMLR |
| STOP | A2 becomes a section of A1. A1 + A3 make a modest paper, and Project B (gated) gains relative weight | TMLR at best; re-plan |
| LR state matters strongly | Add the "prune before you anneal" result in any case | Strengthens any of the above |

## Appendix: terms

### The narrow model

The narrow model is **not** Qwen3 with a smaller embedding. It is a new model, initialised randomly and trained from scratch on our data; no Qwen weights are used.
- **Same as Qwen3-0.6B:** the architecture (RMSNorm, RoPE, QK-norm, GQA attention, SwiGLU MLP, tied embeddings), the Qwen3 tokenizer, and 28 layers.
- **Different:** every layer is half as wide.
- **Embedding:** it shrinks only because the embedding width equals the model width. The vocabulary is unchanged.

| | Qwen3-0.6B shape (our 0.6B parent) | Narrow model |
|---|---|---|
| Layers | 28 | 28 |
| Width | 1,024 | 512 |
| Query heads × head size | 16 × 128 | 4 × 128 |
| KV heads | 8 | 2 |
| MLP width | 3,072 | 1,536 |
| Tokenizer / vocabulary | Qwen3, 151,936 | same |
| Embedding | 155.6M | 77.8M |
| Non-embedding parameters | 440.5M | 88.1M |
| Total | 596M | 166M |

**Why use it:**
- It has the same depth, so "remove 8 of 28 layers" means the same thing as in the 0.6B.
- It is about 3× cheaper per token, so it reaches 306 TPP in a few hours.
- It reaches 306 TPP on unique data. A 0.6B model would need 48B tokens for 80 TPP, more than the 30B-token English pool.

### WSD and cooldown branches, step by step

WSD (warm-up, stable, decay) has three phases:
1. **Warm-up:** the LR rises from 0 to its peak over about the first 1% of steps.
2. **Stable:** the LR stays at the peak for most of training.
3. **Decay (cooldown):** over the last 10–20%, the LR falls linearly to 0. The model "settles" and its loss drops; after this it is *annealed*.

During the stable phase, the model's state does not depend on when training will end. So at any point, you can:
- copy the model (weights and optimiser state);
- run a short cooldown on the copy, which gives a finished model as if training had been planned to stop there;
- let the original continue in the stable phase.

Each copy is a *cooldown branch*.

**For the narrow family:**
- **Trunk:** train at constant LR to 24.3B tokens, saving the model at 1.53B, 6.075B, 6.75B and 24.3B.
- **Branches:**
  - from 1.53B, 0.17B more tokens with the LR falling to 0, gives **P19** (1.7B tokens seen);
  - from 6.075B, 0.675B more gives **P77** (6.75B);
  - from 24.3B, 2.7B more gives **P306** (27B).
- **S77:** the trunk copy saved at 6.75B. It has the same tokens as P77 but is not annealed.

**What this buys:**
- P19, P77 and P306 come from the same run and data stream and are annealed the same way, so they differ **only in training length**.
- S77 and P77 differ **only in whether they are annealed**.

A cosine schedule cannot do this. There, the LR falls throughout, according to the planned total length, so a mid-run checkpoint is not a finished shorter run.

## Sources

- [Training Dynamics Impact Post-Training Quantization Robustness (Catalan-Tatjer, Ajroldi, Geiping; ICLR 2026)](https://arxiv.org/abs/2510.06213v1)
- [The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al.)](https://arxiv.org/html/2403.17887)
- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330)
- [Low-Bit Quantization Favors Undertrained LLMs](https://arxiv.org/abs/2411.17691)
- [Overtrained Language Models Are Harder to Fine-Tune](https://arxiv.org/abs/2503.19206)
- [IDEA Prune](https://arxiv.org/abs/2503.05920)
- [On the Efficiency of Structured Pruning in Small Language Model Pretraining (NeurIPS 2026)](https://neurips.cc/virtual/2026/poster/154468)
- [P² Law](https://arxiv.org/abs/2411.10272)
- [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)
- [Persistent Depth Ordering amid Shifting Block-Bypass Responses](https://arxiv.org/abs/2610.01165)
- [Small LLMs: Pruning vs Training from Scratch](https://arxiv.org/abs/2606.14150)
- [The Journey Matters](https://arxiv.org/abs/2501.12486)
- [open-sci-ref-0.01](https://arxiv.org/html/2509.09009v3)
- [open-sci-ref models on Hugging Face](https://huggingface.co/api/models?search=open-sci-ref&limit=200)
- [open-sci-ref 1.7B Nemotron-HQ 100B model card (laion)](https://huggingface.co/laion/open-sci-ref-1.7b-nemotron-hq-100B-4096)
- Cited from the agents' reads, not read by me:
  - He et al., "What Matters in Transformers";
  - Cherilyn & Safaai (arXiv 2604.23475);
  - Liew et al., upcycling scaling laws (ICML 2025);
  - Sheared LLaMA, App. F.6;
  - Frantar et al., "Scaling Laws for Sparsely-Connected Foundation Models".
