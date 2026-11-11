# DOT-pool pilot: depth-shared expert pools, re-entry and a counterfactual router

Version 1.2, 12 November 2026 (SGT). A new study in Direction 2 (DOT). It does not change any earlier spec.

**Version history.** Nothing has run yet. Each version replaces the previous one, and no arm, run or budget has changed since v1.0.
- **v1.1:**
  - one numbering for middle steps;
  - E1's answer-predicting position, and where E1 places Taylor labels;
  - labels missing from some readings;
  - the batch, tolerance and random-arm criterion of tests T0, T7 and T8.
- **v1.2, answering the agent's questions on v1.0:**
  - the exact balance-loss formula (§2.2);
  - the gradient convention for Taylor labels, and a new test T10 (§2.3, §9);
  - the sign convention of T5 and of the Taylor validity check (§5, §9);
  - backups in three tiers, through owner-approved channels only (§0);
  - record-and-continue covers science, not infrastructure rules (§0).

## 0. How to run this spec

- **This file is complete.** Record its SHA-256 (given in the owner's message) in every config, log and report. If the spec changes later, use the new SHA-256 and keep the history.
- **Record and continue.** Nothing in this spec is a stop condition. If a check fails, a number looks wrong or a budget is overrun:
  - write it in the report's Flags section and say what you did;
  - keep running the queue.
  - The owner may stop runs at any time.
  - This rule covers scientific results and checks. The project's infrastructure rules (ownership, error handling, network policy) stay as they are.
- **A failing test (section 9) is a bug.**
  - Fix it and rerun the test.
  - If it cannot be fixed within 30 minutes, flag it, describe the likely impact, mark the affected arm "suspect" and continue.
- **Time budget: 10–12 node-hours** on the 8×B200 node, from the first timed job of Phase 0 to the final report (section 7).
  - Phase 0 sets the E1 epochs and the E2 token budget from measured throughput, so that the total fits.
  - Writing the code comes before Phase 0 and is not counted. Keep it short: reuse what exists.
- **Back up as you go.** This rule is new, after the A2-Depth and B-DOT losses. Every transfer follows `AGENT_GUIDE.md` and `commands.md`:
  - no outbound connection from the GPU machine;
  - never split a large file to pass a small-file channel.
- **Tier 1, always:** each run's config, metrics, evaluation files, route samples and the manifest.
  - Send them through the project's existing channel for small results, when each run ends.
  - Most of the results survive the loss of the node through this tier alone.
- **Tier 2:** final weights in bf16, without optimizer state. Each is about 0.05–0.5 GB; all 20 runs come to about 6 GB.
- **Tier 3:** E2 pre-decay checkpoints with the full training state: weights, optimizer, scheduler, every RNG state and the data position. Each is about 1.4–3.4 GB; all 10 come to about 30 GB.
- **How Tiers 2 and 3 leave the node:** only through a large-transfer mechanism the owner names. The candidates are:
  - a storage path that is not on the GPU node;
  - a pull initiated from the dev machine;
  - a private Hugging Face repo, uploaded from the controller or the dev machine, never from the GPU machine.
  - Tokens never go into files or Git.
- **Until the owner names a mechanism:** keep Tiers 2 and 3 on the node, record them in the manifest, flag "not yet off-node" and continue. A copy on the same node, such as `_backup`, does not count as off-node.
- **Once a mechanism is named:** test it on one artifact first, then copy in run order and verify SHA-256 after each copy.
- **Manifest, per artifact:** path, size, SHA-256, every location and the time of verification. The report lists every artifact.
- Keep the project's existing conventions for GPU ownership, reminder workers on long jobs, and checks in record-and-continue mode.

## 1. Why this study

**Background.**
- **B-DOT** (344M parameters, 2.5B tokens) showed that pure reordering of layers does not pay: learned routes were no better than random ones.
- **The math** (`DIRECTION2_making_DOT_feasible.md`, section 2) explains why: reordering is a second-order effect, while selecting and reusing modules is first order.

**The motivation for this study** (owner, 12 November):
- A transformer stores a fact at the depth where training needed it.
- In a two-hop query, the bridge entity is resolved in the lower layers, so the second hop must run in the upper layers. Those layers may not hold the fact.
- Information flows upward only: it never returns to the layers that hold the knowledge.
- Evidence: Grokked Transformers, Hopping Too Late, and "Why Knowing Both Hops Is Not Enough".
- Running the same parameters again later in depth ("re-entry") fixes this. Looped models already show it: Loop, Think & Generalize; DiscoLoop.

**DOT-pool.**
- The middle steps draw FFN experts from one pool shared across depth. Any step can therefore reuse any expert: re-entry at expert granularity.
- Unlike a plain loop, the model keeps many parameters.
- B-DOT's own re-entry question (research question 3 of `BDOT_v2_spec.md`) was never run, because B-DOT stopped at Stage 0a.
- The shared pool itself is known (MoEUT, UniPool, MoRE, Dreamer).
- **What would be ours is the router:** it knows the token's path and is trained with a counterfactual signal (Taylor labels).

**Questions.**
- **Q1 (E1, synthetic):** consider two-hop composition of facts that were never composed in training (OOD).
  - Does a routed depth-shared pool generalize like a looped model?
  - Does our router beat an ordinary router and random routing?
- **Q2 (E2, language modeling):** at matched active compute and parameters,
  - does sharing the pool across depth beat a per-layer MoE?
  - does our router beat an ordinary router and random routing?
- **Diagnostics:**
  - Is re-entry used: are experts reused across depth, and does the second hop reuse the experts that hold the fact?
  - Does order matter (order-shuffle control)?
  - Are the Taylor labels informative?

**Not in this study:**
- pure reordering, STOP, forced re-entry and E0;
- two-hop probes mixed into pretraining. At 1.5B tokens each probe fact would be seen only tens of times, far too few for composition to emerge. E1 covers composition.

## 2. Common architecture

All arms use one model class. The dense arm is the special case with no experts.

**Blocks follow Qwen3:**
- pre-norm RMSNorm (eps 1e-6);
- GQA attention with QK-norm (RMSNorm per head on q and k) and RoPE (θ = 10,000);
- SwiGLU FFN;
- no biases;
- input and output embeddings tied, and a final RMSNorm.

**Widths (both studies):**
- d = 512;
- 8 query heads and 4 K/V heads, with head_dim 64;
- dense FFN width 1,536;
- expert width 768 (SwiGLU), with k = 2 experts per token per step. The active FFN width therefore equals the dense width.

**Layout:** n_stem fixed dense layers, then S middle steps, then n_coda fixed dense layers.
- Middle steps are numbered 1 to S everywhere in this spec.
- Every middle step has its own attention (its own Q, K, V, O and norms) in every arm except the looped control.
- Only the FFN of the middle steps differs between arms.

| | E1 | E2 |
|---|---|---|
| Layers | 12 = 2 stem + 8 middle + 2 coda | 24 = 2 stem + 20 middle + 2 coda |
| Vocabulary | 2,240 (2,000 entities, 200 relations, PAD, reserved) | 50,304 (GPT-NeoX-20B, padded) |
| Sequence | one fact per sequence, length 4 | 512 tokens, packed, document-blocked |
| Pool size (shared-pool arms) | 64 experts | 160 experts |
| Per-layer MoE | not used | 8 experts per middle step (160 in all) |

**Initialization:**
- Every linear and embedding weight: normal with std 0.02. Router weights also use std 0.02.
- Norm weights: 1.
- Depth and path embeddings: 0. At the start, the DOT router therefore equals the ordinary router.

### 2.1 The MoE FFN (all expert arms)

- **Input:** x = RMSNorm_ffn,s(h).
  - In the shared-pool arms, add a learned depth embedding d_s to x (as in MoRE). Both the router and the experts see it.
- **Expert m:** a SwiGLU with W_gate, W_up ∈ R^{768×512} and W_down ∈ R^{512×768}.
- **Output:** y = Σ_{j=1..k} k · ĝ_j · E_{m_j}(x), with normalized gates ĝ that sum to 1.
  - The factor k makes equal gates reproduce a dense FFN split into k pieces.
- **No token dropping and no capacity factor.** Every token runs what it chose, so no token's computation depends on later tokens.

### 2.2 Routers

**Ordinary router** (per-layer MoE and shared-ordinary arms; DeepSeek-V3 style):
- **Scores:** σ_m = sigmoid(w_{s,m} · x), computed in fp32.
- **Selection:** the top-k of σ_m + β_m. The bias β is used for selection only, never in the gates.
- **Gates:** ĝ_j = σ_{m_j} / Σ_{selected} σ.
- **Loss-free balancing:** after each optimizer step, update β_m ← β_m + 1e-3 · sign(mean load − load_m). The loads come from that step's all-reduced counts.
  - Shared pool: one β ∈ R^N, with counts summed over all middle steps (pool-level balancing; UniPool found that per-layer balancing of a shared pool is worse).
  - Per-layer MoE: one β per layer.
  - Never update β inside a forward pass.
- **Auxiliary loss:** a sequence-wise balance loss with α = 1e-4, at the same level as β (pool or layer).
  - **Sequence:** one row of input, i.e. one fact in E1 or one packed 512-token row in E2. PAD positions are excluded here and from the β counts.
  - **Terms:** D is the number of routing decisions in scope, N the number of experts and k = 2.
    - f_m = N · count_m / (k · D), where count_m is the number of decisions whose top-k (by σ + β) includes m. No gradient flows through the counts.
    - P_m = the mean over decisions of σ_m / Σ_{all experts} σ. It does not contain β.
    - L_bal(sequence) = α · Σ_m f_m · P_m.
  - **Shared pool:** one loss per sequence, over all of its token × middle-step decisions.
  - **Per-layer MoE:** one loss per layer, then the mean over layers, not the sum. Both arms then regularize at the same strength (L_bal = α at perfect balance).
  - **Batch:** average over the sequences of the global batch, with the same normalization across microbatches and DDP ranks.
  - The loss changes neither the forward routing nor the attention mask.
  - It applies to the ordinary and DOT routers, not to random routing.

**DOT router (ours)** = the ordinary router plus two additions:
1. **Path awareness.**
   - The router input is x + p_s, with p_s = Σ_j ĝ_{s−1,j} · P[m_{s−1,j}] for s ≥ 2 and p_1 = 0.
   - P ∈ R^{N×512} is a learned table of expert embeddings. Gradients flow through ĝ and P.
   - Only the router sees p_s; the experts do not.
2. **Taylor labels** (section 2.3).

**Random routing:**
- k distinct experts are drawn uniformly at random for each token and step, in both training and evaluation (seeded).
- ĝ = 1/k.
- No router parameters and no balancing.

### 2.3 Taylor labels (counterfactual router signal)

- **Labeled tokens:** each training step, sample 2% of tokens independently at each middle step (seeded).
  - In E1, the loss sits only on the answer, so label only the answer-predicting position (section 4), in 10% of sequences per middle step.
- **Forward pass:**
  - For each labeled token at step s, take the C = 16 candidates with the highest σ + β. They include the k selected experts.
  - Compute u_m = E_m(x) for all C candidates without gradient.
  - Store the u_m and the executed output y, both detached.
- **Backward pass:** a hook on the FFN output y_s stores g = ∂L/∂y_s for the labeled tokens.
  - **Convention:** g is the gradient of the summed task NLL over valid targets, not of its mean.
    - If the backward uses a mean loss or gradient-accumulation scaling, multiply the captured gradient back by that factor.
    - The gradients used for the update do not change.
  - **Take g from the main backward pass**; do not run a second backward.
    - The Taylor router loss uses detached inputs, so it contributes nothing to g.
    - The balance loss (α = 1e-4) contributes a negligible amount. This is accepted and stated in the report.
- **Scores:**
  - c_m = −⟨g, k·u_m − y⟩, the first-order predicted loss change if the step's FFN output were k·u_m, with the sign flipped. A higher c_m therefore predicts a larger drop in loss.
  - Standardize the scores over the C candidates: subtract the mean, then divide by the population standard deviation (unbiased = False) + 1e-6. Set q = softmax(c_std).
  - **Log per step:** the median and the 5th percentile of std(c) over labeled tokens, and the share of tokens with std(c) < 1e-4. Labels near uniform because of the epsilon show up there.
- **Router loss:**
  - At the next optimizer step, on the stored detached router inputs (x, plus the previous step's expert ids and gates, which are needed for p_s):
    L_T = 0.1 · mean CE(q, softmax(z_C)).
  - z_C are the router's pre-sigmoid logits for the C candidates, computed with the current weights.
  - Gradients reach only the router weights w_s and the table P.
- **Why this signal:** the labels use the gradient of the executed path, so they respect the router's own later choices.
- **Cost:** about 7% more forward FLOPs, in training only. Report the measured time.

## 3. Arms

**E1: 5 arms × 2 seeds = 10 runs**

| Arm | Middle FFN | Router | Role |
|---|---|---|---|
| E1-D | dense, one per step | — | baseline: should fail OOD |
| E1-L | looped control: a 4-layer block (attention and FFN) run twice between the fixed stem and coda | — | positive control (Loop, Think & Generalize; Grokked's sharing) |
| E1-SO | shared pool of 64 experts | ordinary | |
| E1-DOT | shared pool of 64 experts | DOT | ours |
| E1-SR | shared pool of 64 experts | random | chance control |

**E2: 5 arms × 2 seeds = 10 runs**

| Arm | Middle FFN | Router | Role |
|---|---|---|---|
| E2-D | dense, one per step | — | baseline |
| E2-PL | per-layer MoE: 8 experts of its own per middle step | ordinary | standard MoE |
| E2-SO | shared pool of 160 experts | ordinary | UniPool-like |
| E2-DOT | shared pool of 160 experts | DOT | ours |
| E2-SR | shared pool of 160 experts | random | chance control |

**Parameter counts.** All are non-embedding counts; verify them within 0.1%.

| Arm | Total | Active per token |
|---|---|---|
| E1-D | 37.76M | 37.76M |
| E1-L | 25.18M unique | 37.76M executed |
| E1-SO | 94.65M | 38.03M |
| E1-DOT | 94.69M | 38.06M |
| E1-SR | 94.39M | 37.77M |
| E2-D | 75.53M | 75.53M |
| E2-PL | 217.17M | 75.61M |
| E2-SO | 218.73M | 77.17M |
| E2-DOT | 218.81M | 77.26M |
| E2-SR | 217.09M | 75.54M |

- Embeddings add 1.15M in E1 and 25.76M in E2.
- Forward FLOPs per token for E2-D at sequence 512: 215M.
- Routers add at most 3.3M per token. Taylor labels add about 7%, in training only.
- Report measured tokens/s, but compare quality at matched tokens and active compute.

## 4. E1: two-hop composition (synthetic)

**Data** (as in `BDOT_v2_spec.md` section 10.1, unchanged):
- **Facts:** 2,000 entities and 200 relations, with 20 outgoing relations per entity and targets drawn uniformly at random. This gives 40,000 atomic facts (a, r) → b. KG seed 2026; one KG for all runs.
- **Split:** the atomic facts are split at random, 95% in-distribution (ID) and 5% OOD.
- **Training set:**
  - all 40,000 atomic facts;
  - 273,600 two-hop facts (a, r1, r2) → c whose two hops are both ID (φ = 7.2 per ID atomic fact), sampled without replacement and excluding the dev and test items.
- **Dev and test sets:**
  - ID: 2,000 unseen ID compositions, split into 1,000 dev and 1,000 test;
  - OOD: all compositions whose two hops are both OOD (about 2,000), split into dev and test halves.
- **Format:** one token per entity or relation. Atomic facts are `a r b PAD`; two-hop facts are `a r1 r2 c`. The loss is on the answer token only.
- **Answer-predicting position:** the last relation token (r in an atomic fact, r2 in a two-hop fact), whose output predicts the answer.
- **Accuracy:** the argmax over the full vocabulary at the answer-predicting position equals the answer.

**Training:**
- AdamW (β 0.9, 0.95; weight decay 0.1, with the same parameter groups as E2), bf16 autocast.
- Batch of 4,096 facts. One epoch is one pass over the 313,600 training facts in a fresh random order: 77 steps, the last one partial.
- 1,000 warm-up steps, then a constant LR. The constant LR is what lets Phase 4 continue a run without changing any schedule.
- **LR:** chosen from {3e-4, 1e-3} by E1-D's ID-dev accuracy after 500 epochs (Phase 0), then used for every arm.
- **Length:** N_E1 epochs, set in Phase 0 (section 7), between 2,000 and 10,000 and the same for every arm.
- Log ID-dev and OOD-dev accuracy every 50 epochs.
- Save a checkpoint every 1,000 epochs (keep the last two) and at the end.

**Metrics** (all at the final epoch):
- ID-test and OOD-test accuracy;
- accuracy on all trained facts, and on the OOD atomic facts alone (both should be near 100%);
- the first epoch at which OOD-dev reaches 50%, or "never".

**Re-entry analysis** (E1-SO, E1-DOT, E1-SR):
- For each OOD-test item with bridge b, take two sets of experts:
  - A: the experts chosen at the answer-predicting position in middle steps 5–8 of the two-hop query;
  - B: the experts chosen at the answer-predicting position in middle steps 1–4 of the atomic query (b, r2).
- Overlap = |A ∩ B| / |A ∪ B|.
- Excess overlap = mean overlap − mean overlap with randomly re-paired atomic queries, with a 95% bootstrap CI over items.
- Report it separately for correctly and wrongly answered items.

## 5. E2: language modeling

**Data:**
- **Source:** DCLM-baseline with the GPT-NeoX-20B tokenizer, the one open-sci-ref uses (50,277 tokens, padded to 50,304).
  - Reuse A2's tokenized DCLM if present. Check the tokenizer identity and the document boundaries (EOS).
  - Otherwise tokenize a fresh subset.
- **Train:** the first T tokens, with T from Phase 0 (1.0–2.0B).
  - Pack into 512-token sequences with EOS between documents.
  - Use document-blocked attention (FlashAttention varlen) and reset positions at each document.
  - One fixed order (seed 0) for all runs.
- **Validation:** 10M tokens from held-out documents, disjoint from train and packed the same way, plus a fixed 1M-token subset for periodic evaluation.

**Training:**
- **Optimizer:** AdamW (β 0.9, 0.95; eps 1e-8).
  - Weight decay 0.1 on matrices and embeddings; none on norms or on the depth and path embeddings.
  - Gradient clip 1.0.
  - bf16 autocast, with fp32 master weights and fp32 router math.
- **Batch:** 262,144 tokens (512 sequences), with no accumulation if memory allows. Steps: 3,815, 4,768, 5,722 or 7,629 for T = 1.0, 1.25, 1.5 or 2.0B tokens.
- **Schedule:** WSD — 200 warm-up steps, then constant, then linear decay to 0 over the last 20%.
  - Save a checkpoint just before the decay; it can be resumed for longer runs later.
  - Save another at the end.
- **Peak LR:** from Phase 0's mini-sweep over {1e-3, 2e-3, 4e-3}, at 0.25B tokens with the same schedule shape.
  - E2-D's sweep sets E2-D's LR.
  - E2-SO's sweep sets the LR of all four expert arms.
  - If two LRs are within 0.005 nats, take the lower one.
- **Seeds:** model seeds 0 and 1; the data order is fixed.
- **Evaluation during training:** every 250 steps on the 1M subset, and the full 10M validation set at the pre-decay checkpoint and at the end.

**Evaluation at the end:**
- **Validation loss:** the main metric.
- **Downstream (reported only, no reading):** lm-eval (commit 13686ed, as in A2) through a thin wrapper, zero-shot.
  - Tasks: hellaswag (acc_norm), piqa (acc_norm), arc_easy (acc), lambada_openai (acc).
  - Context at most 512 tokens, truncated on the left.
- **Order-shuffle** (E2-SO, E2-DOT):
  - First record each token's expert ids and gates at every middle step.
  - Then rerun with each sequence's middle steps permuted: at step s, the token uses the expert set recorded at step π(s), with one random π per sequence.
  - Attention, norms and depth embeddings stay with their step.
  - Use 3 permutations on 1M tokens. Cost = loss(shuffled) − loss(normal).
- **Re-entry statistics** (E2-SO, E2-DOT, E2-SR):
  - distinct experts per token over the 20 steps (out of 40 selections);
  - reuse rate: the share of selections of an expert the same token already chose at least 2 steps earlier;
  - each expert's entropy over steps.
  - E2-SR gives the chance level.
- **Expert usage** (E2-PL, E2-SO, E2-DOT):
  - load per expert over the last 100 steps;
  - dead experts (load under 10% of the mean);
  - max/mean load.
- **Taylor validity** (E2-DOT, final checkpoint):
  - Use 256 validation tokens, middle steps {2, 10, 18} and C = 16 candidates.
  - For each candidate, replace that token's FFN output at that step by k·u_m and recompute the sequence.
  - **Sign convention:** ΔL = L_counterfactual − L_normal, computed on the summed loss over positions ≥ t. A positive ΔL means the candidate makes the loss worse.
  - Report the median over tokens of Spearman(c_m, −ΔL), which is expected to be positive, and how often the router's top choice is the best candidate by −ΔL.
  - Store ΔL itself and this sign convention in the output.

## 6. Readings

These are fixed now. They are labels only, never stop rules.

**E1** (OOD-test accuracy, mean of 2 seeds; also report each seed):
- **E1-R0, setup check:** E1-D reaches ID-test ≥ 90% and OOD-test < 20%, and E1-L reaches OOD-test ≥ E1-D + 20 points.
  - If E1-L fails, label E1 "inconclusive: composition did not emerge within the budget". R1–R3 are still reported.
- **E1-R1, pool vs loop:** E1-SO vs E1-L. "Matches" within 10 points; otherwise "better" or "worse".
- **E1-R2, our router:** E1-DOT − E1-SO. ≥ +10 points "helps"; ≤ −10 "hurts"; otherwise "no clear difference".
- **E1-R3, learned vs random:** E1-SO − E1-SR and E1-DOT − E1-SR. ≥ +10 points "learned routing matters"; ≤ −10 "random better"; otherwise "no clear difference".
- **E1-R4, re-entry:** an excess overlap whose CI lies above 0 is labelled "re-entry observed".

**E2** (validation loss after the decay, mean of 2 seeds; seed spread = |seed 0 − seed 1| for each arm):
- **E2-R0, MoE sanity:** E2-PL ≤ E2-D − 0.02. Otherwise flag "MoE implementation or recipe suspect".
- **E2-R1, sharing across depth:** E2-SO vs E2-PL. At least 0.01 lower: "sharing helps". Within 0.01: "no clear difference". At least 0.01 higher: "sharing hurts".
- **E2-R2, our router** (Δ = E2-SO − E2-DOT, positive when DOT is lower):
  - Δ ≥ 0.01: "router helps";
  - 0.003 ≤ Δ < 0.01: "small";
  - |Δ| < 0.003: "no effect";
  - −0.01 < Δ ≤ −0.003: "small, worse";
  - Δ ≤ −0.01: "router hurts".
- **E2-R3, learned vs random:** E2-SR − E2-SO and E2-SR − E2-DOT. ≥ 0.01 "learned routing helps"; ≤ −0.01 "random better"; otherwise "no clear difference".
- **E2-R4, order:** a shuffle cost of at least 0.01 is labelled "order matters"; below 0.01, "order matters little".
- **E2-R5, Taylor labels:** a median Spearman of at least 0.3 is labelled "informative"; below 0.3, "weak".
- **Noise rule:** if a difference passes its threshold but is smaller than twice the larger seed spread of the two arms, add "within seed noise".
- **Downstream:** table only, no label.

## 7. Phases, budgets and run order

**Phase 0, setup and calibration** (target ≤ 1.5 h)
1. Run tests T0–T10 (section 9). If the owner has named a large-transfer mechanism, also test it on one small artifact (§0).
2. Prepare data: the E1 KG and splits, the E2 training shards (≥ 2.0B tokens) and the 10M-token validation set.
3. **E1 LR pick:** E1-D at 3e-4 and at 1e-3, to 500 epochs. Run this on 2 GPUs, in parallel with step 4 where possible.
4. **Throughput:**
   - each E2 arm for 100 steps, after 20 warm-up steps;
   - each E1 arm for 2,000 steps.
   - Record tokens/s or steps/s, and which expert kernel was used.
5. **E2 LR mini-sweep:** E2-D and E2-SO, each at {1e-3, 2e-3, 4e-3} for 0.25B tokens: 6 short runs.
6. **Set the budgets:**
   - N_E1: the largest multiple of 500 in [2,000, 10,000] such that the slowest E1 run, at its measured speed and with its GPU sharing, finishes within 3.0 h.
   - T: the largest of {1.0, 1.25, 1.5, 2.0}B tokens such that the 10 E2 runs, at their measured speeds plus 5% for evaluation and checkpoints, fit in 6.0 h.
   - If even the minimum does not fit, use the minimum and flag it.
   - Report the chosen N_E1, T and LRs, with the measurements behind them.

**Phase 1, E1** (≤ 3.0 h)
- All 10 runs in parallel:
  - the 6 expert-arm runs on one GPU each;
  - the 4 runs of E1-D and E1-L, both seeds, two per GPU.
- The agent may rebalance if the measured speeds suggest a better packing.

**Phase 2, E2** (≤ 6.0 h)
- Runs go in sequence, each on all 8 GPUs (data parallel):
  - seed 0 of E2-D, E2-PL, E2-SO, E2-DOT and E2-SR;
  - then seed 1 in the same order.
- Every arm therefore has one seed before any arm gets its second.
- Back up each run when it ends.

**Phase 3, evaluation and report** (≤ 0.75 h)
- E2 evaluations: downstream for all 10 models in parallel, then order-shuffle, re-entry, expert usage and Taylor validity.
- E1 re-entry analysis.
- The report.

**Phase 4, only if time remains**
- Condition: total elapsed time is under 11.5 h, and E1-L's OOD-dev accuracy was still rising over its last 500 epochs.
- Then resume all 10 E1 runs from their final checkpoints (constant LR, no schedule change) for the same number of extra epochs, ending by 12 h.
- Re-evaluate, and report both endpoints.

| Phase | Target time |
|---|---|
| 0: tests, data, LR picks, throughput | ≤ 1.5 h |
| 1: E1 | ≤ 3.0 h |
| 2: E2 | ≤ 6.0 h |
| 3: evaluation and report | ≤ 0.75 h |
| Total | about 10.5–11.5 h (at most 12 h with Phase 4) |

Rough expectations, to be replaced by the Phase 0 measurements:
- E1 expert arms: about 0.5–1 h per 2,000 epochs.
- E2-D: about 10–20 min per 1.5B tokens.
- E2 expert arms: about 1.5–3× the time of E2-D.

## 8. Outputs

Put everything in `results/dot_pool_20261112/`.

**Per run:**
- `config.json` (with the spec SHA-256);
- `metrics.jsonl`;
- `final_eval.json`;
- `routes_sample.npz`: for E2, the expert ids and gates on the first 64 validation sequences; for E1, all dev and test items plus the atomic queries (b, r2) of the OOD-test items, which the re-entry analysis needs;
- checkpoints, backed up as in section 0;
- entries in the study-wide `manifest.json` (section 0).

**E1:**
- `e1_summary.csv`, with columns arm, seed, ID-test, OOD-test, trained-fact accuracy, OOD-atomic accuracy and epoch_OOD50;
- learning curves (PNG);
- `e1_reentry.csv`.

**E2:**
- `e2_summary.csv`, with columns arm, seed, final validation loss, pre-decay validation loss, tokens/s, wall-clock and parameter counts;
- learning curves (PNG);
- `downstream.csv`, `shuffle.csv`, `reentry.csv`, `usage.csv` and `taylor_validity.csv`.

**`REPORT.md`:**
- the spec SHA-256;
- the environment (GPUs, PyTorch version, kernels);
- Flags;
- phase timings;
- the chosen budgets (N_E1, T, LRs) and the measurements behind them;
- the parameter and FLOP table;
- the E1 and E2 tables;
- every reading with its label;
- the artifact list with backup locations and SHA-256.

## 9. Implementation and tests

**Implementation:**
- **One codebase.** Each arm is a config file, so adding a variant later means adding a config.
- **Training loop:** plain PyTorch with DDP (not the Hugging Face Trainer, whose overhead B-DOT measured), bf16 autocast, FlashAttention varlen, and torch.compile where it works.
- **Experts:** grouped GEMM (`torch._grouped_mm`) if the node's PyTorch has it. Otherwise sort tokens by expert and loop. Record which one was used.
- B-DOT code may be reused if it is still available. The uploaded `model.py` (design v1) is for reference only.

**Tests.** Report pass or fail, with values.
- **T0 Qwen3 equivalence:** the dense arm's logits match Hugging Face `Qwen3ForCausalLM` with the same config and weights (causal attention, no packing, the same attention kernel on both sides): within 1e-4 in fp32 and 1e-3 relative in bf16.
- **T1:** parameter counts match section 3 within 0.1%.
- **T2 causality:** changing token t+1 never changes the outputs at positions ≤ t. Check every arm, on packed sequences with document boundaries.
- **T3 routing:**
  - each token gets k distinct experts per step;
  - the gates sum to 1 before the factor k;
  - the β update reduces load imbalance on a skewed synthetic input over 100 steps.
- **T4 shared pool:** an expert used at two steps receives the sum of both gradients (compare with an unshared copy).
- **T5 Taylor labels:** on a tiny model, over 64 tokens, report Spearman(c_m, −ΔL), with the sign convention of §5 (ΔL = L_counterfactual − L_normal, from finite differences). It is expected to be positive. Store ΔL and the sign convention with the result.
- **T6 random routing:** selection is uniform (chi-square p > 0.01 over 10^6 draws) and reproducible at evaluation with a fixed seed.
- **T7 overfit:** each arm drives the loss on one small fixed batch below 0.1 within 300 steps.
  - The batch is 8 sequences of 512 tokens for E2 and 256 facts for E1.
  - The random-routing arms draw new routes at every step, so for them it is enough that the loss falls below half its initial value.
- **T8:** the order-shuffle with the identity permutation reproduces the normal loss to 1e-4, which allows for kernel nondeterminism.
- **T9 resume:** saving and reloading reproduces the next 10 steps' losses within 1e-4.
- **T10 Taylor labels and microbatch split:** with the same data and weights, a different microbatch split gives the same labels: Spearman ≥ 0.999 for each labeled token, and the same top candidate except for ties.

## 10. Sources

**Our notes:**
- `DIRECTION2_making_DOT_feasible.md`: section 2 (the math), section 4 (DOT-pool), section 5 (E0–E2).
- `DIRECTION2_dynamic_order_revisited.md` (the B-DOT results and their reading).
- `BDOT_v2_spec.md`: section 3.7 (router lessons) and section 10.1 (the two-hop task).

**Papers:**
- Composition and re-entry:
  - [Grokked Transformers are Implicit Reasoners (NeurIPS 2024)](https://arxiv.org/abs/2405.15071)
  - [Loop, Think, & Generalize (arXiv 2604.07822)](https://arxiv.org/abs/2604.07822)
  - [DiscoLoop (arXiv 2607.00341)](https://arxiv.org/abs/2607.00341)
  - [Why Knowing Both Hops Is Not Enough (arXiv 2608.07261)](https://arxiv.org/abs/2608.07261)
  - [Hopping Too Late (arXiv 2406.12775)](https://arxiv.org/abs/2406.12775)
  - [Reasoning with Latent Thoughts (Saunshi et al., ICLR 2025)](https://arxiv.org/abs/2502.17416)
- Shared pools:
  - [UniPool (arXiv 2605.06665)](https://arxiv.org/abs/2605.06665)
  - [MoRE (arXiv 2609.18176)](https://arxiv.org/abs/2609.18176)
  - [MoEUT (arXiv 2405.16039)](https://arxiv.org/abs/2405.16039)
  - [Dreamer (arXiv 2601.21582)](https://arxiv.org/abs/2601.21582)
- Routing and balancing:
  - [DeepSeek-V3 (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
  - [Auxiliary-loss-free load balancing (arXiv 2408.15664)](https://arxiv.org/abs/2408.15664)
  - [RMoE (arXiv 2408.06793)](https://arxiv.org/abs/2408.06793)
  - [Default MoE / dense backpropagation (arXiv 2504.12463)](https://arxiv.org/abs/2504.12463)
  - [ProxylessNAS (arXiv 1812.00332)](https://arxiv.org/abs/1812.00332)
- Model and tokenizer: [open-sci-ref-0.01 (arXiv 2509.09009)](https://arxiv.org/abs/2509.09009)
