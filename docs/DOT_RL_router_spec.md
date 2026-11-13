# DOT-RL pilot: learning paths through a layer library with RL

Version 1.2, 14 November 2026 (SGT). A new study in Direction 2 (DOT). It follows `DOT_pool_pilot_spec.md` v1.5 and does not change it.

**Version history** (nothing has run yet):
- **v1.2:** review fixes; the design is unchanged.
  - Cites the closest prior architecture, Library of Layers (LoL), and corrects the scope note on gradient-trained routers (§1).
  - ID-dev and ID-test are excluded from the training compositions, as in E1 (§2).
  - States the order of operations within a step (§5) and how N and WS-RL's switch are fixed while runs continue (§8).
  - Adds the sanity and timing readouts used in E1b: ID-test for every arm, accuracy on trained facts and on OOD atomic facts, the first epoch at which OOD-dev reaches 50%, and the OOD-dev change over the last 500 epochs (§6, §7, §9).
  - Adds a path comparison between training and OOD questions (§6), and small clarifications (router norm and betas, T2, T7).
- **v1.1:** the main test is now joint learning, where the layers keep learning under the router's paths, in two forms: from scratch (J-RL) and after a random-path warm-up (WS-RL).
  - Post-training on frozen layers is kept only as a cheap diagnostic (FR).
  - **Why:** frozen layers let RL choose among paths the base already supports, but never learn a new order. This matches what RL post-training seems to do to LLMs: mostly sharpen what the base model can already do.
  - RL is made cheap enough to run at the same epochs as the other arms: the layers train on one sampled path per example, and the router learns from forward-only group rollouts.
  - The router is rewarded only on held-out compositions, never on training data. Its advantages use the group mean as the baseline and one batch-level scale, which suits a dense reward (§5).
- **v1.0:** frozen post-training was the main test and joint RL was secondary. Never ran.

## 0. How to run this spec

- **This file is complete.** Record its SHA-256 (given in the owner's message) in every config, log and report. If the spec changes later, use the new SHA-256 and keep the history.
- **Record and continue.** Nothing in this spec is a stop condition.
  - If a check fails, a number looks wrong or a budget is overrun, write it in the report's Flags section, say what you did, and keep running the queue.
  - The owner may stop runs at any time.
  - This rule covers scientific results and checks. The project's infrastructure rules (ownership, error handling, network policy) stay as they are.
- **A failing test (§10) is a bug.** Fix it and rerun the test. If it cannot be fixed within 30 minutes, flag it, describe the likely impact, mark the affected arm "suspect" and continue.
- **Time budget: about 12 node-hours** on the 8×B200 node, from the first timed job to the final report (§8). Writing the code comes before and is not counted. Extend the DOT-pool codebase; do not start a new one.
- **Backups follow the three tiers of `DOT_pool_pilot_spec.md` §0**, through owner-approved channels only.
  - Router checkpoints are tiny (kilobytes), so they belong to Tier 1 and leave the node with the results.
  - Until the owner names a large-transfer mechanism, model weights and full training state stay on the node, flagged "not yet off-node".

## 1. Why this study

**Where DOT stands** (DOT-pool v1.5, E1b, two-hop composition, 4,500 epochs):
- **The looped control works.** A 4-layer block run twice reached 41% and 49% OOD-test. A dense 8-layer model reached 0.1%.
- **Routed shared expert pools barely did.** The ordinary router reached 0.0% and 0.2%; our DOT router reached 7.5% and 5.7%.
  - DOT's excess expert overlap disappeared against the same-relation baseline (E1-R4b, seed 1).
  - 88–91% of its Taylor labels were near uniform (std below 1e-4) at the end of training.
- **A possible reason, not tested:** the routing signals from gradients were too weak to show which alternative choice would have been better. RL measures the effect of whole alternative paths directly.

**What the literature says about routers over layers and RL routers:**
- **The closest architecture is Library of Layers (LoL, ICLR 2027 submission).** A router picks each next layer from a shared library of whole transformer layers and is trained end-to-end with the model.
  - LoL is competitive on language modeling and learns input-dependent layer orders on sudoku and mazes.
  - Letting positions of one sequence choose different layers mixes attention states across layers; LoL found that this made training unstable, so it routes per window.
- REINFORCE-style estimators have worked poorly for MoE expert routing (SparseMixer, citing Kool et al. 2021).
- Routers trained jointly with their modules tend to collapse and specialize badly (Rosenbaum et al. 2019; Mittal et al. 2022).
- RL has been used for small, structured choices, such as the stopping depth of a looped transformer. There it often helped, but it could also lock in a suboptimal computation (RL-Halting, 2026).
- One study finds that RL post-training of LLMs mostly sharpens what the base model can already do, rather than creating new abilities (Yue et al. 2025); this is debated.

**Design choices that follow from this:**
1. **Route whole layers, not experts.** In E1b, re-running whole layers composed (45%), while routed expert pools barely did (0.1% and 6.6%). So the router chooses among 4 full transformer layers at each of 8 steps, as in LoL, but the router is trained by RL.
2. **Let the layers keep learning under the router's paths.** A router over frozen layers can only select; it cannot teach the layers a new order. The main arms therefore train layers and router together:
   - **J-RL:** from scratch.
   - **WS-RL:** after a warm-up with uniformly random paths, which trains every layer on the outputs of every other layer. This is the owner's earlier "warm up, then route" idea (D2), here tested on composition.
   - **FR (diagnostic):** a router over the frozen random-path model. It measures how much selection alone can do.
3. **Use the stable parts of LLM RL:** group baselines (GRPO, RLOO), a KL to the reference path distribution, on-policy updates, and verifiable dense rewards. No critic.
4. **Make RL cheap.** The layers train on one path per example, sampled from the current policy. The router learns from group rollouts that only run forward. In compute this is about 1.2× a random-path run, instead of about 4× if every rollout also trained the layers. Wall-clock may be higher, so N is set from measured speed (§8), and all arms run the same N.
5. **Reward the router only on held-out compositions.**
   - Unseen in-distribution (ID) compositions do not reward the OOD skill: the dense model scores 99% ID-test and 0.1% OOD-test.
   - So a separate set of atomic facts ("OOD-RL") is never composed in training. The router is rewarded only on compositions of those facts, and they never train the layers. Testing uses a third, disjoint set ("OOD-test").
   - The router gets no reward on training data. Otherwise it could lock onto whichever path the layers have memorized best, the kind of collapse seen in jointly trained routers. The split follows architecture search, where the architecture learns on held-out data and the weights on training data (DARTS).

**Questions:**
- **Q1:** can routing learned jointly with RL discover paths that compose OOD, from scratch (J-RL) or after a warm-up (WS-RL)?
- **Q2:** does the warm-up help?
- **Q3:** how do the RL arms compare with the looped control and with random paths at the same epochs?
- **Q4 (diagnostic):** with frozen random-path layers, does any path compose (headroom), and how much can a router find by selection alone? Does per-question routing matter there?

**Fairness note:** the RL routers receive rewards on OOD-RL compositions; the looped control and the random-path arm do not. The comparison with looping is therefore about whether learned routing can find re-entry when given composition rewards, not about beating looping on equal data.

**Not in this study:** expert pools, Taylor labels, language modeling, and STOP or halting actions. No gradient-trained router either: E1b tested such routers only over FFN expert pools. A gradient-trained router over whole layers, as in LoL, is a natural follow-up if RL routing works.

## 2. Data (E1-RL split)

**Knowledge graph:** as in `DOT_pool_pilot_spec.md` §4. 2,000 entities, 200 relations, 20 outgoing relations per entity, targets uniform at random, 40,000 atomic facts. KG seed 2026.

**Split of the atomic facts,** at random:
- **ID (85%):** 34,000 facts.
- **OOD-RL (10%):** 4,000 facts.
- **OOD-test (5%):** 2,000 facts.

All 40,000 atomic facts are trained. Only ID facts ever appear in the compositions used to train the layers.

**Two-hop sets.** Draw ID-dev and ID-test first; the training compositions are then sampled from the remaining ID compositions.

| Set | Content | Use |
|---|---|---|
| ID-dev / ID-test | 1,000 / 1,000 compositions, both hops ID | evaluation |
| Training compositions | 244,800 (φ = 7.2 × 34,000), both hops ID, sampled without replacement, excluding ID-dev and ID-test | train the layers |
| RL-train / RL-dev | all compositions with both hops OOD-RL (about 8,000), split 90 / 10 at random | router rewards / router monitoring, and model selection in FR |
| OOD-dev / OOD-test | all compositions with both hops OOD-test (about 2,000), split in halves | evaluation only |

**Format** as in E1:
- One token per entity or relation; atomic `a r b PAD`, two-hop `a r1 r2 c`; the loss is on the answer only.
- The answer-predicting position is the last relation token.
- Accuracy is the argmax over the full vocabulary (2,240) at that position.

An epoch is one pass over the 284,800 training facts: 70 steps at batch 4,096, the last one partial.

## 3. Model: a library of 4 layers and per-question paths

**Layers:** 4 transformer layers as in E1b (Qwen3 blocks):
- d = 512, 8 query and 4 K/V heads (head_dim 64), FFN 1,536;
- no stem and no coda; tied embeddings and a final RMSNorm;
- 12.59M unique non-embedding parameters.

**Path:** 8 steps. At step s (1..8), every position of the question runs the same layer a_s ∈ {1, 2, 3, 4}.
- One path per question means no mixing of attention states across layers inside a question, the problem LoL avoids with per-window routing. A question here is a single short window.
- The loop path is 1-2-3-4-1-2-3-4.
- Batching is free: for example, group the questions by their chosen layer at each step, or run all four layers on the whole batch and keep each question's chosen output. Use the faster one if both are tried; T13 and T14 must pass.

**Policy:**
- z_s = W_s · RMSNorm(sg(h_s)) + E[a_{s−1}] + b_s, and π_s = softmax(z_s).
  - h_s is the hidden state at the answer-predicting position entering step s (the embedding for s = 1).
  - sg is stop-gradient: the policy never sends gradients into the layers.
  - This RMSNorm has no learnable scale.
  - W_s ∈ R^{4×512}, E ∈ R^{5×4} (index 0 = start), b_s ∈ R^4, all initialized at 0, so the policy starts uniform.
- **Fixed variant (FR only):** W_s ≡ 0. The policy then depends only on the step and the previous layer: one path distribution for every question.
- **No answer leakage:** the policy reads only the answer-predicting position, which under causal attention never sees the answer token (test T2).

## 4. Arms

All four main arms train from scratch with the E1b recipe (`DOT_pool_pilot_spec.md` §4, as fixed for E1b in §7, Phase 5): AdamW with weight decay 0.1, batch 4,096, 1,000 warm-up steps, then a constant LR of 1e-3. They all run the same N epochs (§8). Seeds 0 and 1, one GPU per run: 8 GPUs.

| Arm | Path used to train the layers | Router |
|---|---|---|
| L | fixed loop path 1-2-3-4-1-2-3-4 | none (positive reference: E1b's looped control, retrained on this split) |
| P-rand | a fresh uniformly random path for every example, each step independent | none |
| J-RL | one path per example, sampled from the current policy | RL from epoch 0 (§5) |
| WS-RL | random paths, as P-rand, up to the switch epoch (N/3 rounded down to a multiple of 500, at least 500); then as J-RL | RL from the switch epoch |

**Diagnostic FR** (frozen router), on each P-rand seed's final checkpoint, with the layers frozen:
- FR-adaptive (full policy) and FR-fixed (W_s ≡ 0);
- seeds 0 and 1: 4 small runs.

## 5. RL details

**Group rollouts for the router** (J-RL, and WS-RL after the switch):
- Each optimizer step draws 512 RL-train questions, without replacement within each pass over RL-train. Training examples never reward the router.
- For each question q, sample G = 4 paths from π and run the layers forward only, with no gradient to the layers.
- **Reward:** r_{q,g} = log p(answer | q, a^(g)), clipped below at −20.
- **Advantage:** subtract the group mean, d_{q,g} = r_{q,g} − mean_g r_{q,·}; then divide by one scale for the whole batch, A_{q,g} = d_{q,g} / (std of all d in the batch + 1e-6), with the population std.
  - GRPO divides by each group's own std instead. With a dense reward, that would blow tiny differences between near-identical paths up to full size. Group mean plus batch-level std is the "REINFORCE++ with baseline" estimator; Dr. GRPO drops the per-group std for a related reason.
  - A group whose G paths all coincide gets A = 0.
- **Router loss:** L_π = −mean_{q,g} [ A_{q,g} Σ_s log π_s(a_s^(g)) ] + β · mean_{q,g} Σ_s KL(π_s ‖ U), with π_s taken along each sampled path.
  - KL(π_s ‖ U) is computed exactly over the 4 layers.
  - U is uniform over the 4 layers, the path distribution of the warm-up and of P-rand; it plays the role of the reference policy in LLM post-training.
  - β = 0.04.
- **Router optimizer:** AdamW with the layers' betas (0.9, 0.95), LR 1e-4 (0.1× the layers'), no weight decay. One update per step, on-policy, with no importance ratios or clipping.

**Layers:** CE loss on all 4,096 training examples of the step, each run through one path sampled from the current policy.
- Early in J-RL, the policy is near uniform, so the layers see random paths, as in P-rand. As the policy sharpens, they specialize to its paths. This is how a new order can be learned.
- RL-train questions never give the layers a gradient.

**Order within a step:** the layers' forward pass and the rollouts use the same weights. Then both optimizers step.

**Logging, every 100 updates:**
- the mean reward on RL-train questions and the training CE loss;
- the batch std of the centered rewards d;
- policy entropy per step;
- the share of questions whose G paths are all identical;
- the share of the loop path among greedy paths.

**Collapse check:** flag any run whose mean policy entropy falls below 0.2 nats per step within 500 epochs of the router starting. This can also be fast convergence to a good path, so report which path it is.

**FR (frozen diagnostic):**
- Router parameters only. The layers are frozen; record their hashes before and after.
- AdamW, LR 1e-3; batch 256 RL-train questions × G = 8; up to 3,000 updates; same reward, advantage and KL.
- Every 100 updates, evaluate RL-dev greedy accuracy. The selected router is the checkpoint with the best RL-dev greedy accuracy (earlier on ties).
- Report OOD-test for the selected and for the last checkpoint.

## 6. Evaluation

At the end, on ID-test, OOD-test and RL-dev:

| Arm | Paths evaluated |
|---|---|
| L | loop path |
| P-rand | (a) mean over 8 seeded random paths per question; (b) loop path; (c) oracle best-of-8: share of questions where at least one of the 8 random paths answers correctly (headroom) |
| J-RL, WS-RL | greedy path; also the mean over 8 sampled paths |
| FR-adaptive, FR-fixed | greedy path; also the mean over 8 sampled paths |

**Sanity readouts** (every arm and evaluation mode), as in E1b: accuracy on all trained facts, and on the OOD-RL and OOD-test atomic facts alone. Both should be near 100%; otherwise OOD composition cannot succeed.

**Curves:** log ID-dev and OOD-dev accuracy every 50 epochs for every main arm: greedy paths for the RL arms, random and loop paths for P-rand.
- For the RL arms, also log RL-dev accuracy and mean reward. A growing gap between RL-train and RL-dev reward means the router is overfitting its reward set.
- For every main arm, report the first epoch at which OOD-dev reaches 50% (or "never") and the OOD-dev change over the last 500 epochs. Still rising means N was too short for that arm.

**Path analysis** (J-RL, WS-RL, FR), on OOD-test:
- the 10 most frequent greedy paths and their shares, separately for correct and wrong answers;
- the share of the loop path;
- the share of "repeat" paths, where steps 5–8 use the same layer sequence as steps 1–4;
- whether atomic and two-hop questions take different paths;
- whether training compositions, RL-train and OOD-test questions take the same paths. The router is rewarded only on OOD-RL questions, while the layers learn on the paths of training questions;
- for J-RL and WS-RL, how these change over training (every 500 epochs).

## 7. Readings

These are fixed now. They are labels only, never stop rules. The values are OOD-test accuracy, the mean of 2 seeds; also report each seed.

- **RL-R0, setup:**
  - L ≥ 20%: "positive control works". Otherwise flag "inconclusive: composition did not emerge within the budget".
  - Flag any main arm whose ID-test is under 90%: "did not fit ID".
- **RL-R1, joint RL against random paths:** for each of J-RL and WS-RL, Δ = arm greedy − P-rand random-path mean.
  - Δ ≥ +10 points: "learned routing helps";
  - +3 ≤ Δ < +10: "small";
  - Δ < +3: "no effect".
- **RL-R2, warm-up:** WS-RL − J-RL. ≥ +10 points: "warm-up helps"; ≤ −10: "warm-up hurts"; otherwise "no clear difference".
- **RL-R3, against looping:** the better of J-RL and WS-RL against L. Within 10 points: "matches looping with composition rewards"; ≥ +10: "beats"; ≤ −10: "worse".
- **RL-R4, selection-only ceiling** (FR):
  - FR-adaptive greedy − P-rand random-path mean, with the labels of RL-R1.
  - Also report P-rand's oracle best-of-8 minus its random-path mean. If the oracle is under 10%, label "little headroom: hardly any path composes".
- **RL-R5, per-question routing** (FR): FR-adaptive − FR-fixed ≥ +5 points: "per-question routing matters". Otherwise "one good path suffices".
- **RL-R6, joint vs selection-only:** the better joint arm against FR-adaptive, with the labels of RL-R3. This answers whether letting the layers learn new orders matters.
- **Noise rule:** if a difference passes its threshold but is smaller than twice the larger seed spread of the two arms, add "within seed noise".

## 8. Phases, budgets and run order

**Phase 0** (≤ 0.5 h): tests (§10), data generation and the split counts.

**Phase 1, main arms** (≤ 10.5 h): all 8 runs start together, one GPU each.
- **N epochs:** the largest multiple of 500, up to 4,500, such that J-RL, at its speed over its first 200 epochs, finishes within 10.5 h.
  - If fewer than 2,000 fit, use 2,000 and flag it.
  - Fix N as soon as J-RL has run 200 epochs. The other runs keep going meanwhile.
  - WS-RL keeps its random-path warm-up until N is fixed. If it is then already past the switch epoch, it switches at once; flag the actual switch epoch.
- The other arms are cheaper and run the same N.
- Checkpoints every 500 epochs (keep the last two) and at the end, with full state.

**Phase 2, FR diagnostic** (≤ 1.5 h): starts as soon as each P-rand seed finishes, on the GPU it freed.

**Phase 3, evaluation and report** (≤ 0.5 h).

| Phase | Target |
|---|---|
| 0: tests and data | ≤ 0.5 h |
| 1: L, P-rand, J-RL, WS-RL | ≤ 10.5 h |
| 2: FR diagnostic | ≤ 1.5 h, overlapping Phase 1 |
| 3: evaluation and report | ≤ 0.5 h |
| Total | about 12 h |

## 9. Outputs

Put everything in `results/dot_rl_20261114/`.

- **Per run:**
  - `config.json` (with the spec SHA-256);
  - `metrics.jsonl`;
  - `final_eval.json`;
  - router checkpoints (Tier 1);
  - greedy paths for every evaluation item.
- **Summaries and curves:**
  - `summary.csv`: arm, seed, evaluation mode, epochs or updates, ID-test, OOD-test and RL-dev accuracy, trained-fact accuracy, OOD-RL and OOD-test atomic accuracy, the first epoch at which OOD-dev reaches 50%, and the OOD-dev change over the last 500 epochs;
  - `fr_curves.csv`: RL-dev reward and accuracy every 100 updates;
  - `paths.csv`;
  - `oracle.csv`;
  - curves (PNG).
- **`REPORT.md`:**
  - the spec SHA-256;
  - the environment;
  - Flags;
  - phase timings;
  - the chosen N and switch epoch, with the measurements behind them;
  - split counts;
  - tables;
  - every reading with its label;
  - the artifact list with locations and SHA-256.

## 10. Implementation and tests

**Implementation:** extend the DOT-pool codebase.
- One model class: a library of layers and a path.
- One policy class with an adaptive and a fixed variant.
- One RL module: rollouts, advantages and the router loss.
- Each arm is a config file.

**Tests** (report pass or fail, with values):
- **T1:** unique non-embedding parameters are 12.59M for every arm (plus the policy where present), within 0.1%.
- **T2 no answer leakage:** with the same sampling seed, changing the answer token or any later token changes neither the routing decisions nor the prediction at the answer-predicting position.
- **T7 overfit**, with the codebase's T7 settings: on 256 fixed facts, each of these brings the loss below 0.1 within 300 steps:
  - L;
  - P-rand with its paths fixed per fact;
  - J-RL with its policy frozen at uniform and its paths fixed per fact.
- **T9 resume:** saving and reloading reproduces the next 10 steps' losses within 1e-4.
- **T11 RL sanity:** with a toy reward equal to the number of steps where a_s = ((s − 1) mod 4) + 1, the fixed policy reaches at least 90% probability on the loop path within 1,000 updates (batch 256 × G = 8; LR 1e-2 for this test only, since it checks the code, not the settings).
- **T12 frozen layers:** in FR, the layer parameter hashes are identical before and after training.
- **T13 path sampling:**
  - P-rand's per-step choices are uniform (chi-square p > 0.01 over 10^6 draws);
  - every position of a question runs the same layer at each step.
- **T14 forward-only rollouts:** the router loss gives the layers exactly zero gradient, and RL-train questions give the layers no gradient at all.

## 11. Sources

**Our notes and specs:**
- `DOT_pool_pilot_spec.md` v1.5 and its E1b results (`RESULTS_VERIFIED_NODE.md`).
- `DIRECTION2_dynamic_order_revisited.md` (D2, "warm up, then route").
- `DIRECTION2_making_DOT_feasible.md`.

**Papers:**
- Routing over layers and modularity:
  - Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime (ICLR 2027 submission; in our uploads): the closest prior architecture.
  - Random Recursive Models (ICLR 2027 submission; in our uploads): uniform random layer choice from a shared pool, the closest prior to P-rand.
  - [Routing Networks and the Challenges of Modular and Compositional Computation](https://arxiv.org/abs/1904.12774)
  - [Is a Modular Architecture Enough? (NeurIPS 2022)](https://arxiv.org/abs/2206.02713)
  - [Modular Deep Learning (survey)](https://arxiv.org/abs/2302.11529)
  - [Sparse Backpropagation for MoE Training (SparseMixer)](https://arxiv.org/abs/2310.00811)
  - [Soft Merging of Experts with Adaptive Routing (SMEAR)](https://arxiv.org/abs/2306.03745)
- Adaptive depth and layer routing:
  - [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping (RL-Halting)](https://arxiv.org/abs/2606.29983)
  - [Dr.LLM: Dynamic Layer Routing in LLMs (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/774164b966cc277c82a960934445140d-Paper-Conference.pdf)
- RL for LLM post-training:
  - [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
  - [Back to Basics: REINFORCE-style optimization for RLHF (RLOO)](https://arxiv.org/abs/2402.14740)
  - [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)
  - [Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/abs/2503.20783)
  - [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
  - [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- Architecture search:
  - [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- Composition and looping:
  - [Grokked Transformers are Implicit Reasoners (NeurIPS 2024)](https://arxiv.org/abs/2405.15071)
  - [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822)
