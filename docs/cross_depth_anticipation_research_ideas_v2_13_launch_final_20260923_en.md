# Cross-Depth Anticipation

**Research Idea Inventory, Re-evaluation, and Cheap Kill Tests**
**22 September 2026**

**Status:** research notebook + locked Phase-1/2 implementation contract for cheap frozen-model probes; not authorization for large-scale pretraining.

> **Main decision:** drop the original frozen CCM formulation as the primary research direction. The most promising replacement is to approximate deep-past → shallow-future feedback in one pass by distilling only the correction or functionality induced by privileged deep feedback.

# 1. Executive decision

After re-checking novelty and the closest antecedents, the broad idea of "predict a latent/feature and inject it back" is no longer sufficiently novel: CoCoMix, NCP-ArchPreview, KV Prediction, FusedKV, NextLat, and multiple distillation lines already cover this area fairly broadly. The narrower gap worth pursuing is to use a one-pass prediction to approximate the benefit of deep-past → shallow-future feedback, which currently often requires recurrence, multi-pass execution, Jacobi/fixed-point methods, or more complex deep routing paths.

Current recommendation: do **not** train for 5B/10B tokens. First run a privileged-reference probe study on a frozen pretrained model. Only move to pretraining if privileged deep feedback clearly helps and the one-pass predictor recovers a meaningful fraction of that gain.

## 1.1 Portfolio summary

| Idea | Role | Novelty risk | Technical risk | Decision |
|---|---|---:|---:|---|
| Privileged Cross-Depth Correction (PCC) | Primary architecture | Medium | Medium | KEEP - priority #1 |
| Query-conditioned Communication Correction | Richer functional variant | Medium-high | High | KEEP - second stage |
| Predictable Functional Bottleneck | Target design / teacher representation | High as standalone | Medium | KEEP as component |
| Directional / Orthogonal Correction | Target ablation | Medium-high | Medium-high | KEEP as ablation |
| Representation Readiness Gating | Safety / alternative mechanism | Plausibly lower | Medium | PROBE only |
| Sparse / Event-Triggered Feedback | Efficiency wrapper | High as standalone | Medium | KEEP as wrapper |

# 2. Ground rules to avoid repeating the CCM mistake

- Do not use a persistent hidden-state artifact created by an old checkpoint and inject it into a backbone that has drifted into a new representation space.

- The teacher target must be recomputed from the current model/checkpoint itself; if the teacher is used only for supervision, the target must be detached / stop-gradient.

- Do not choose a target because its reconstruction loss is low. The target must demonstrate functional utility through held-out LM loss or attention/communication behavior.

- Every proposal must have a privileged/oracle-style reference and a cheap kill test before any from-scratch pretraining.

- Do not call a component novel if it already exists independently in the literature. Novelty must lie in the structural role / problem solved by the combination.

- The model must start exactly or nearly exactly from Base: use a zero-initialized correction/output branch or a gate initialized to no-op.

- Any extra teacher privilege must have a reachability test: the teacher may know something the student cannot infer from the causal shallow prefix, and the student should not be forced to mimic the unreachable part.

# 3. Re-evaluation of the self-generated ideas

## 3.1 Privileged Cross-Depth Correction (PCC) - KEEP, priority #1

**Idea:** on a small subset of training batches, run a privileged path that lets shallow future tokens use true deep representations of past tokens. Compare the hidden state at the same shallow layer between the privileged path and the normal one-pass path. Distill only the correction induced by the privilege.

**Teacher target:**

$$
\delta_t = \operatorname{stopgrad}\left(h_t^{(s,\mathrm{priv})} - h_t^{(s,\mathrm{base})}\right)
$$

**Student one-pass predictor:**

$$
\hat{\delta}_t = C_\phi\left(H_{\le t}^{(s,\mathrm{base})}\right)
$$

**Use path:**

$$
h_t^s \leftarrow h_t^{(s,\mathrm{base})} + \hat{\delta}_t
$$

- **Strength:** the teacher and Base targets lie at the same layer, same checkpoint, and same coordinate system. There is no stale-state mismatch of the CCM kind.

- **Independent grounding:** Full-bandwidth / WhiteMatter show that privileged deep feedback can be useful; Privileged Foresight Distillation shows that distilling a teacher-minus-base correction is a reasonable target in another domain.

- **Plausible novelty:** in language modeling, use correction prediction to approximate deep-past → shallow-future feedback in a single parallel pass.

- **Main risks:** the teacher advantage may not be predictable from the shallow prefix; the predictor may learn noisy corrections; the privileged path may be too expensive.

- **Safeguard:** zero-initialized output. Confidence/reachability gating and advantage-weighted supervision are later ablations; the locked first probe does not enable them.

**Cheap kill test:**

- Use a frozen pretrained LM. Create a privileged-reference path. If the privileged path does not improve NLL reliably, abandon the entire family.

- If the privileged reference helps but the correction predictor does not recover a meaningful gain, abandon the one-pass approximation.

## 3.2 Query-conditioned Communication Correction - KEEP, but only after PCC

Instead of making a source token predict one fixed correction vector, predict the parameters of a function describing how the deep source would respond to each future query. This is a functional version of deep feedback and is more natural when the same source token is useful in different ways to different queries.

**Teacher functional residual:**

$$
\Delta m_{t\leftarrow j}
=
\operatorname{stopgrad}
\left(
m_{t\leftarrow j}^{\mathrm{deep}}
-
m_{t\leftarrow j}^{\mathrm{shallow}}
\right)
$$

The student can generate a low-rank query-response operator from the source token's shallow state, and the future query then uses this operator to produce a correction.

- **Strength:** does not reconstruct arbitrary hidden coordinates; directly models communication behavior.

- **Grounding:** MiniLM-style relation distillation, K/V functional compression, and evidence that K and V play different roles across depth.

- **Risk:** pairwise token-query supervision may be very expensive; overlap with WhiteMatter/MiniLM/KV-compression is higher; implementation is more complex.

- **Decision:** do not use as the first experiment. Promote it only if PCC shows a privileged-reference gain from cross-depth correction but the vector correction is too query-agnostic.

## 3.3 Representation Readiness Gating - KEEP as a probe/control

This is the dual solution to deep feedback: instead of forcing deep information to become available earlier, the model learns when a past token representation is still "not mature" and reduces shallow future queries' tendency to read it too early.

Example attention gating:

$$
\operatorname{score}_{t,j}^{\ell}
\leftarrow
\operatorname{score}_{t,j}^{\ell}
+
\log(r_j^\ell + \epsilon)
$$

- **Strength:** can initialize \(r=1\), so the model starts exactly as a standard Transformer; it does not inject noisy content.

- **Motivation:** Racing Thoughts shows that timing/contextualization errors are real.

- **Risk:** suppressing early information may make the model worse rather than compensate for missing information; the readiness teacher is difficult to define; query-dependent readiness increases cost.

- **Novelty:** I have not found an exact learned depth-maturity gate in a causal LM, but this would need another audit if the direction is promoted in priority.

- **Decision:** cheap probe, not the main paper direction at this stage.

## 3.4 Predictable Functional Bottleneck - KEEP as target design, NOT as standalone novelty

The generic idea \(E(h_{\mathrm{deep}})\rightarrow z\) is too close to TED, CoCoMix, NCP, NextLat, SAE/crosscoder, and latent distillation. However, it remains very useful as a module for defining a compact target inside PCC.

- \(z\) must be anchored by a functional loss: future-token utility, outgoing-attention behavior, tail-logit fidelity, or privileged correction.

- Do not train \(E\) using only reconstruction of \(h_{\mathrm{deep}}\).

- Predictor \(P\) should only learn to match \(\operatorname{stopgrad}(z)\); the alignment loss must not pull \(E\) toward a code that is easy to predict but useless.

- If \(E\) and \(P\) are trained jointly, task-utility / variance constraints are required to prevent collapse or collusion.

## 3.5 Directional / Orthogonal Correction - KEEP as a new ablation

Recent work on directional decomposition separates a residual update into a component parallel to the current hidden state and a component orthogonal to it. That work observes that the perpendicular component is more strongly associated with directional/semantic change and is more sensitive to perturbation, whereas the parallel component is often more robust.

For PCC correction \(\delta_t\), decompose:

$$
\delta_{\parallel}
=
\frac{\langle \delta,h\rangle}{\|h\|^2} h
$$

$$
\delta_{\perp}
=
\delta-\delta_{\parallel}
$$

Ablation: make the predictor learn only \(\delta_\perp\), or use a weighted loss that prioritizes perpendicular error.

- **Attraction:** has a new geometric basis and may reduce target redundancy.

- **Risk:** recent work emphasizes that the effect depends on the representation space; the residual-space perpendicular component is very sensitive, so prediction error may be especially harmful.

- **Standalone novelty is low** because orthogonal residual / directional decomposition already exists. Use only as a target ablation for PCC.

## 3.6 Sparse / Event-Triggered Feedback - KEEP as an efficiency/safety wrapper

Recent work repeatedly shows that more cross-layer access is not automatically better. Therefore correction should not necessarily be active for every token.

- **Teacher-side advantage gate:** distill only when the privileged teacher improves token loss or another functional metric over Base.

- **Student-side confidence gate:** the predictor outputs both correction and confidence; if confidence is low, it becomes a no-op.

- The gate can be regularized toward a target activation rate or implemented as budgeted Top-k per segment.

- Do not treat sparsity/event-triggering as the main novelty; treat it as a mechanism to reduce noise, bandwidth, and risk.

# 4. List of reasonable ideas to test

## A. Privileged Cross-Depth Correction (primary)

A privileged teacher uses true deep-past feedback; the one-pass student predicts the same-layer correction induced by that feedback.

**First test:** Frozen-model privileged reference → correction predictor → tiny pilot.

**Why keep it:** Highest chance of providing a functional target while avoiding coordinate mismatch.

## B. Attention-Output / Same-Layer Communication Correction

Instead of distilling a correction on the full residual \(h^s\), distill the correction on the attention output or the residual contribution of a shallow block.

**First test:** Compare delta on the residual stream vs. delta on the attention-branch output.

**Why keep it:** May target communication more directly and place less pressure on MLP/self information.

## C. Query-Conditioned Communication Operator

The source token's shallow state predicts a low-rank operator; a future query obtains a query-specific correction.

**First test:** Only try this if A/B show that correction depends strongly on the query.

**Why keep it:** Expressive, but systems/novelty overlap is high.

## D. Functional Bottleneck inside PCC

Compress the teacher correction through \(E\) into a 64/128-D code; \(E\) is trained using privileged/tail/future utility, and the student predicts the code.

**First test:** Privileged-reference code quality first, then predictability.

**Why keep it:** Reduces bandwidth and target complexity; standalone novelty is low.

## E. Directional Innovation Target

Predict only the perpendicular part of the privileged correction, or weight perpendicular error more strongly.

**First test:** Ablation on a frozen model.

**Why keep it:** Grounded by 2026 directional-decomposition evidence, but caution is required because perpendicular errors are highly fragile.

## F. Readiness Gating

Predict whether a past-token representation at depth \(\ell\) is mature enough for future queries to read, and gate attention exposure accordingly.

**First test:** Frozen-model intervention + learned gate.

**Why keep it:** Does not provide missing information, but may reduce racing errors and is very safe when initialized to 1.

## G. Sparse/Advantage-Gated PCC

Apply corrections only to tokens with high teacher advantage and high student confidence.

**First test:** Layer on top of A after the privileged-reference study.

**Why keep it:** High practical potential, but not independent novelty.

# 5. Directions that should not be prioritized

**Raw \(h_{\mathrm{deep}}\) regression:** already crowded, coordinate-sensitive, and the EAGLE family provides a warning about feature-prediction bottlenecks.

**Raw \(h_{\mathrm{deep}}-h_{\mathrm{shallow}}\) regression as the headline:** meaningful as a residual, but still a coordinate-level target; keep it as a baseline.

**Generic SAE/concept extraction → predict → inject:** CoCoMix/NCP already cover the broad idea; the utility of a reconstruction feature is not guaranteed.

**Predict full deep K/V:** high overlap with KV Prediction/FusedKV; likely overkill.

**Fixed layer-2 → layer-16 skip/gate:** DenseFormer/AttnRes/SATFormer/Depth-Attention already generalize this more strongly.

**Frozen offline hidden memory:** representation-staleness/co-adaptation risk; this is the lesson from CCM.

# 6. Loss design: match the loss to the target

Do not search for one universal loss. The loss must reflect the semantics of the target. LM loss always remains the ultimate functional anchor.

| Target | Primary loss | Role | Warning |
|---|---|---|---|
| Same-layer privileged correction | Normalized SmoothL1 in the locked first probe; cosine only as a later ablation | Predict vector correction | First probe does not use advantage gating; add it only as a later ablation |
| Privileged final behavior | Temperature KL | Functional teacher consistency | Do not let KL pull the teacher; detach teacher |
| Attention routing correction | KL on attention distributions/logits | Preserve routing | Head/layer alignment must be explicit |
| Attention/message correction | SmoothL1/cosine on message/output | Preserve delivered content | Do not confuse received output with outgoing source message |
| Functional bottleneck | Task/future CE or KL + bottleneck regularization | Define useful \(z\) | Extractor must not optimize only for easy matching |
| Student-to-bottleneck | SmoothL1/cosine or contrastive | Predict \(z\) | Select by downstream LM gain, not reconstruction score |
| Directional correction | Weighted perpendicular/parallel error | Emphasize direction-changing part | Space-dependent; can destabilize |

## 6.1 Default gradient routing

To avoid teacher/student collusion, teacher-derived targets must be detached. If there is a learnable extractor \(E\), \(E\) should receive gradients only from the functional-utility objective of the teacher representation, not from the student alignment loss. The student predictor receives alignment gradients plus LM gradients through the use path.

Recommended schematic:

$$
L_{\mathrm{total}}
=
L_{\mathrm{LM}}
+
\lambda_{\mathrm{corr}}
L_{\mathrm{corr}}
\left(
\mathrm{student},
\operatorname{stopgrad}(\mathrm{teacher\ correction})
\right)
+
\lambda_{\mathrm{func}}
L_{\mathrm{func}}(\mathrm{teacher\ extractor})
$$

# 7. Research funnel before pretraining

## Phase 0 - Exact novelty freeze

- Freeze terminology and claims; re-check Full-bandwidth, WhiteMatter, PFD, CoCoMix/NCP, KV Prediction/FusedKV, and recent cross-layer papers.

- Do not call "feature prediction", "deep feedback", or "latent injection" novel. The claim should be restricted to one-pass distillation of privileged cross-depth correction if no exact precedent appears.

## Phase 1 - Privileged-feedback reference test

- Use a frozen pretrained LM.

- Create a Base path and an expensive privileged deep-feedback path on the same batch/model.

- Measure per-token and overall NLL gain, gate/attention changes, and whether gain concentrates in particular token categories/depths.

- The first probe uses one pinned model/dataset. Before scaling beyond a tiny pilot, replicate the privileged-reference gain on at least one additional checkpoint/model size or dataset.

## Phase 2 - Target comparison

- same-layer residual correction (PCC default)

- same-layer attention-output correction

- privileged logit correction

- compressed functional bottleneck

- perpendicular-only correction

Choose the target by held-out LM benefit under privileged-reference access, not by MSE.

## Phase 3 - One-pass predictor

- Train the predictor from the shallow prefix only; initially keep the backbone frozen.

- Measure the fraction of privileged-reference gain recovered by the predicted correction.

- Use a zero-init/no-op branch. Confidence gating is a later ablation, not part of the locked first probe.

- If the predictor reconstructs well but does not improve LM loss, reject the target.

## Phase 4 - Tiny from-scratch pilot

- Only after Phases 1-3 pass.

- Use a small model / short budget; compare Base, PCC, PCC-no-teacher-loss, shuffled/wrong correction, and a privileged reference if feasible.

- Report quality per FLOP/wall-clock; extra teacher batches must be counted.

## Phase 5 - Scale only after replication

Do not scale because one seed shows a gain. Replicate the mechanism, then increase model size/tokens. Extra compute should be compared against simply training Base longer.

# 8. Current decision matrix

| Idea | Grounding | Novelty | Parallel-friendly | Cheap kill-test | Priority |
|---|---|---|---|---|---|
| PCC | Strong | Plausible narrow | Yes at inference/student; teacher subset expensive | Yes | 1 |
| Attention-output PCC | Strong-medium | Plausible narrow | Yes | Yes | 2 |
| Query-response correction | Medium-strong | Narrow/overlap risk | Potentially | Moderate | 3 |
| Functional bottleneck | Strong | Not standalone | Yes | Yes | Component |
| Directional correction | New evidence | Not standalone | Yes | Yes | Ablation |
| Readiness gating | Medium | Plausible | Yes | Yes | Probe |
| Sparse/advantage gating | Strong design principle | Not standalone | Yes | Yes | Wrapper |

# 9. Execution recommendation

If only one experiment is run next, choose the frozen-model Privileged Cross-Depth Correction privileged-reference study. Do not begin with a complex feature extractor. First establish that true deep feedback creates a useful same-layer correction and that this correction can be predicted from the shallow causal prefix.

After privileged-reference PCC passes, the recommended expansion order is:
1. compare residual vs. attention-output correction;
2. add advantage/confidence gating;
3. test a compressed functional bottleneck;
4. only then consider query-conditioned communication operators or directional targets.

**Core scientific question:** Can the benefit of exact deep-past → shallow-future communication be compressed into a predictable, current-checkpoint, one-pass correction without introducing stale representations or recurrent training?

# 10. Selected research basis

- [Full-bandwidth Transformer (2026)](https://arxiv.org/abs/2608.08888)
- [WhiteMatter: all-depth cross-layer KV connectivity (2026)](https://arxiv.org/abs/2608.18486)
- [Racing Thoughts: contextualization timing errors (NAACL 2025)](https://aclanthology.org/2025.naacl-long.155/)
- [Privileged Foresight Distillation (2026)](https://arxiv.org/abs/2604.25859)
- [Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation (2026)](https://arxiv.org/abs/2606.10385)
- [Jump to Conclusions: early-to-late representation mapping (2024)](https://aclanthology.org/2024.lrec-main.840/)
- [TED: task-aware filtered hidden distillation (ICML 2023)](https://proceedings.mlr.press/v202/liang23j.html)
- [MiniLMv2: multi-head self-attention relation distillation](https://arxiv.org/abs/2012.15828)
- [LLM Pretraining with Continuous Concepts (CoCoMix)](https://arxiv.org/abs/2502.08524)
- [NCP-ArchPreview (2026)](https://arxiv.org/abs/2609.10715)
- [KV Prediction (ICLR 2025)](https://machinelearning.apple.com/research/kv-prediction)
- [FusedKV (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html)
- [Delta Attention Residuals (2026)](https://arxiv.org/abs/2605.18855)
- [Disentangling Representation Evolution in Transformers through Directional Decomposition (2026)](https://arxiv.org/abs/2609.15975)
- [Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks (arXiv:2505.11881)](https://arxiv.org/abs/2505.11881)

# 11. Codex-ready implementation contract: first PCC probe

This section is the executable default for Codex. Earlier sections remain the research inventory. Where an earlier section leaves multiple reasonable choices open, this section wins for the first experiment. Do not substitute a different feedback mechanism, layer pair, loss, masking rule, or target without creating a new experiment version.

## 11.1 Goal

Test one narrow premise before any from-scratch pretraining: whether true deep-past information can improve a frozen model's shallow future computation, and whether a one-pass shallow-only branch can recover a meaningful fraction of that privileged gain.

- This is a mechanism probe, not a paper-scale training run.

- No persistent memory is created. All deep teacher states come from the current frozen model and current batch.

- No future-token leakage is allowed. A token at position \(t\) may only read source positions \(j<t\) that are also visible under the backbone attention mask for the same model-context input.

## 11.2 Locked default backbone and data

**Model-config check (verified 2026-09-22):** Codex must assert `num_hidden_layers=28`, `hidden_size=1024`, `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`, `rms_norm_eps=1e-6`, `rope_theta=1,000,000`.

| Item | Locked default |
|---|---|
| Backbone | `Qwen/Qwen3-0.6B-Base` pretrained weights, revision `ddc928429ed09d9ad603fd762053d0434c15e865`; frozen throughout the probe |
| Depth | 28 Transformer blocks |
| Sequence length | 2048 tokens |
| Shallow hook \(s\) | screen candidates after blocks 4 or 8 residual additions, before the next block input RMSNorm; full-run \(s\) is selected by Section 11.2.1 |
| Deep source hook \(d\) | screen candidates after blocks 16, 20, or 24 residual additions, before the next block input RMSNorm; full-run \(d\) is selected by Section 11.2.1 |
| Tokenizer | Tokenizer from the same pinned model revision. Reuse the existing tokenization/EOS policy from the fixed CulturaX preprocessing pipeline exactly; do not introduce a new BOS/EOS policy for this probe. All compared arms receive token-identical model-context inputs. |
| Data | Reuse the exact existing user-selected CulturaX English subset. Split it once into fixed train/dev/test data using the existing pipeline; all compared methods use exactly the same splits. |
| Packing | Packing or concatenating multiple documents is allowed. Use one fixed preprocessing/context-construction pipeline for all arms. The auxiliary branch must obey the same allowed-context/attention mask and position IDs as the backbone. |
| Probe train | 610 updates × 32,768 non-padding input tokens/update = 19,988,480 input tokens, taken from the same fixed training stream for every full-run arm |
| Probe dev | 10,000,000 non-padding input tokens from the same fixed dev split for every arm |
| Probe test | 10,000,000 non-padding input tokens from the same fixed test split for every arm; locked until all dev decisions are frozen |
| Layer-pair screen train | First 128 updates / 4,194,304 tokens from the same fixed training stream for every candidate pair |
| Layer-pair screen dev | A fixed 2,000,000-token slice of the same dev split, identical for every candidate pair |

If the exact pretrained checkpoint is unavailable on the execution node, stop and record the issue; do not silently replace it with a from-scratch checkpoint or another model family.

**Data fairness rule:** all compared methods must train on the same fixed CulturaX training data, with the same data-order seed, the same number of training tokens/updates, and the same tokenizer/preprocessing/context construction. All methods must be evaluated on the same fixed dev/test sequences.

Packing or concatenating documents is allowed. Whatever the existing pipeline does—e.g. EOS-separated concatenation, ordinary causal packing, or block-isolated packing—must be applied identically to every arm.

No new manifest/hash/deduplication system is required for this probe. Existing bookkeeping may be reused if convenient, but it must not become a prerequisite or alter the selected CulturaX subset.

Microbatch size and gradient accumulation are systems choices and may vary with hardware, but each optimizer update must contain exactly 32,768 non-padding input tokens globally and preserve identical ordered examples across matched arms.

**Seeds:** `data_order_seed=20260922`; `screen_adapter_init_seed=1701`; `full_adapter_init_seed=2901`; `bootstrap_seed=20260922`. Matched arms must use the same data-order seed and paired initialization seed where architectures match.

**Execution precision:** frozen backbone and adapter forward/backward in bf16; LM loss/correction-loss reductions in fp32; AdamW moments in fp32; adapter dropout=0.

### 11.2.1 Pre-registered layer-pair screen

Do not treat \((s=4,d=20)\) as scientifically privileged. Before the full probe, screen four fixed pairs:

$$
(s,d)\in\{(4,16),(4,20),(8,20),(8,24)\}
$$

For each pair, train Privileged-Deep and matched Shallow-ExtraAttn for exactly 128 updates / 4,194,304 tokens using paired adapter initialization and paired ordered batches. Use the same optimizer recipe as the full adapter run but with a 7-update linear warmup and cosine decay over 128 updates. Evaluate once on `D_screen_dev`.

$$
\operatorname{DeepSourceAdv}(s,d)
=
\operatorname{NLL}(\mathrm{Shallow\mbox{-}ExtraAttn})
-
\operatorname{NLL}(\mathrm{Privileged\mbox{-}Deep})
$$

On `D_screen_dev`, run a 1,000-replicate paired bootstrap over the fixed evaluation sequences/contexts with seed `20260922`. Use the same resampled sequence indices for every compared arm. A candidate pair is eligible only if:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Privileged\mbox{-}Deep}}
-
\operatorname{NLL}_{\mathrm{Base}}
\right)
<0
$$

and

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Privileged\mbox{-}Deep}}
-
\operatorname{NLL}_{\mathrm{Shallow\mbox{-}ExtraAttn}}
\right)
<0.
$$

Among eligible pairs, select the pair with the largest point `DeepSourceAdv`; break ties by lower Privileged-Deep NLL, then shallower \(s\), then lower \(d\). Freeze the pair before the full probe.

If no pair satisfies both bootstrap eligibility criteria, stop before the full probe and report a negative screen. Do not add new layer pairs after seeing results.

## 11.3 Named tensors and causal semantics

For a model-context segment \(x_0,\ldots,x_{T-1}\):

$$
H^s=[h_0^s,\ldots,h_{T-1}^s]
$$

at the currently evaluated/selected shallow hook \(s\), and

$$
H^d=[h_0^d,\ldots,h_{T-1}^d]
$$

at the currently evaluated/selected deep hook \(d\).

During the layer-pair screen, Pass A may capture all candidate hooks \(\{4,8,16,20,24\}\) in one clean forward, but each adapter pair may consume only its declared \(H^s\) and \(H^d\) tensors.

- Both are residual-stream states after the named block and before the next block's input RMSNorm.

- All teacher source states \(H^d\) must come from a clean memory/feedback-free pass and are detached before privileged use.

- Privileged attention uses a strict-past mask \(j<t\). It must not use \(j=t\). The same-token deep state is deliberately forbidden because it is not available in a one-pass causal system.

- For any query position with no allowed strict-past source under the backbone mask (for example the first position of a context, or the first position of an isolated packed segment), do not softmax an all-masked row. Define the auxiliary attention output and correction to be exactly zero.

- Auxiliary key/value visibility must match the backbone's fixed context construction. A source position \(j\) is usable only if \(j<t\) and the backbone attention mask for that model-context sequence allows \(t\) to attend to \(j\).

## 11.4 Pass A: clean frozen-backbone pass

Run the frozen backbone exactly as the original pretrained model, with no extra branch. Capture \(H^s\) and \(H^d\) at the explicit hooks and compute Base logits/NLL.

Required outputs per batch:

- detached \(H^s\) for the shallow query/input state;

- detached \(H^d\) for privileged teacher source states;

- Base logits and per-target-token losses;

- the exact backbone attention mask / allowed-context representation and `position_ids` needed to reproduce the same causal visibility in the auxiliary branch.

Pass-A logits must match an unmodified vanilla forward within the project bf16 tolerance before any privileged-reference or student experiment can run.

## 11.5 Privileged-Deep reference module (arm label: Privileged-Deep)

The first privileged reference is a learned strict-past cross-attention adapter. It is intentionally simple and uses the same shallow query state but true deep source states from Pass A.

**Normalize:**

$$
q^{\mathrm{in}}_t
=
\operatorname{RMSNorm}_q(h_t^s)
$$

$$
kv^{\mathrm{in}}_j
=
\operatorname{RMSNorm}_{kv}(h_j^d)
$$

Project to an inner width \(r=256\) using 2 heads × 128 dimensions.

Apply the pinned Qwen RoPE convention to each auxiliary 128-d Q/K head using the exact `position_ids` used by the backbone for the same model-context sequence. Do not impose an extra reset rule beyond the existing preprocessing pipeline. Use identical positional treatment in Privileged-Deep and Shallow-ExtraAttn/Student branches.

$$
Q_t=W_Q q_t^{\mathrm{in}},
\qquad
K_j=W_K kv_j^{\mathrm{in}},
\qquad
V_j=W_V kv_j^{\mathrm{in}}.
$$

**Strict-past multi-head attention:**

For each 128-d head use standard scaled dot-product attention:

$$
\operatorname{logits}=QK^\top/\sqrt{128}
$$

Apply the intersection of the backbone attention mask and the strict-past mask, softmax over allowed sources, then weight \(V\). Concatenate the 2 heads to recover the 256-d \(A_t\). Compute softmax/reductions in fp32 if needed for numerical stability.

$$
A_t
=
\operatorname{MHA}
\left(
Q_t,
\{K_j,V_j:j<t\ \text{and}\ \operatorname{backbone\_mask}(t,j)=\mathrm{allow}\}
\right).
$$

**Scalar gate and correction:**

$$
g_t
=
\sigma(w_g^\top q_t^{\mathrm{in}}+b_g),
\qquad
b_g=-2
$$

$$
c_t^{\mathrm{priv}}
=
g_t W_OA_t
$$

$$
h_t^{(s,\mathrm{priv})}
=
h_t^s+c_t^{\mathrm{priv}}.
$$

- \(W_Q/W_K/W_V\) use `bias=False` and normal `std=0.02` initialization. Initialize gate vector \(w_g\) to zeros and scalar bias \(b_g=-2\).

- \(W_O\) uses `bias=False` and is initialized to all zeros so the privileged model initially equals Base.

- RMSNorm scales initialize to 1; no bias in \(W_O\).

- Frozen backbone parameters never enter any optimizer group.

- \(H^d\) is detached. Gradients may flow through the frozen tail computation to the privileged-reference module, but not into backbone parameters or Pass-A source states.

## 11.6 Efficient privileged-reference training/evaluation path

Do not rerun blocks 1..\(s\). Reuse detached \(H^s\) from Pass A, add \(c_t^{\mathrm{priv}}\), then rerun only blocks \((s+1)..28\) plus final norm/head with frozen weights.

Train only the privileged-reference module for exactly **610 updates / 19,988,480 non-padding input tokens** at global batch **32,768 tokens/update**.

Optimizer:

- AdamW
- betas = `(0.9, 0.95)`
- eps = `1e-8`
- peak LR = `3e-4`
- 31-update linear warmup
- cosine decay to `3e-5`
- weight decay = `0.01` on matrices
- weight decay = `0` on norms/bias/gate bias
- global grad clip = `1.0`

Primary privileged-reference target is ordinary causal-LM NLL. There is no hidden-state reconstruction loss in privileged-reference training.

The full Privileged-Deep run starts from the fresh `full_adapter_init_seed=2901` initialization; never continue from a screened adapter checkpoint.

After privileged-reference training, freeze the privileged-reference module before student training.

## 11.7 Exact PCC target

For this locked first probe, the PCC target is the privileged branch's same-layer correction itself, not \(h^d\), not \(h^d-h^s\), and not a post-tail state.

$$
\delta_t^*
=
\operatorname{stopgrad}(c_t^{\mathrm{priv}})
$$

Because \(c_t^{\mathrm{priv}}\) lives in the selected shallow-layer \(s\) residual coordinate system and is the exact intervention that produced the privileged path, this avoids cross-layer coordinate subtraction.

- Only target-bearing positions are included in correction losses.

- Optional teacher-advantage weights may be logged, but advantage gating is **NOT** enabled in the first locked run; it is a later ablation.

## 11.8 Student one-pass predictor

Use an auxiliary causal self-attention branch with the same inner width (256) and 2 heads × 128 dimensions, but it may use only shallow states \(H^s\).

**Student source/query inputs:**

$$
\bar q_t
=
\operatorname{RMSNorm}_{qs}(h_t^s)
$$

$$
\bar s_j
=
\operatorname{RMSNorm}_{ss}(h_j^s)
$$

**Student attention:**

$$
Q'_t=W'_Q\bar q_t,
\qquad
K'_j=W'_K\bar s_j,
\qquad
V'_j=W'_V\bar s_j
$$

$$
A'_t
=
\operatorname{MHA}
\left(
Q'_t,
\{K'_j,V'_j:j<t\ \text{and}\ \operatorname{backbone\_mask}(t,j)=\mathrm{allow}\}
\right)
$$

$$
g'_t
=
\sigma({w'_g}^\top\bar q_t-2)
$$

$$
\hat\delta_t
=
g'_t W'_O A'_t
$$

$$
h_t^{(s,\mathrm{student})}
=
h_t^s+\hat\delta_t.
$$

- Student \(W'_Q/W'_K/W'_V\) use `bias=False` and normal `std=0.02`; RMSNorm scales initialize to 1; \(w'_g\) initializes to zeros with gate bias -2; \(W'_O\) uses `bias=False` and initializes to all zeros. Attaching the untrained student branch must exactly reproduce Base logits.

- The student branch has no access to \(H^d\), privileged K/V, privileged corrections from other batches, or any persistent deep-state cache.

- At autoregressive inference, cache only the student's own shallow \(K'/V'\) states exactly as an ordinary extra attention branch would.

## 11.9 Student losses and gradient routing

Before student optimization, run frozen Privileged-Deep on the first deterministic **1,000,000 input tokens** of probe-train with no optimizer updates and compute one scalar teacher-correction RMS:

$$
\sigma_\delta
=
\sqrt{
\frac{
\sum_t\|\delta_t^*\|_2^2
}{
N_{\mathrm{target}}\,d_{\mathrm{model}}
}
}
$$

Freeze \(\sigma_\delta\) for every student arm.

**Frozen privileged correction target:**

$$
\delta_t^*
=
\operatorname{stopgrad}(c_t^{\mathrm{priv}})
$$

**Correction alignment:**

$$
L_{\mathrm{corr}}
=
\operatorname{mean}_{\text{eligible target positions, hidden dims}}
\operatorname{SmoothL1}
\left(
\frac{\hat\delta_t}{\sigma_\delta},
\frac{\delta_t^*}{\sigma_\delta};
\beta=1.0
\right)
$$

Eligible target positions are the same non-padding causal-LM prediction positions used by \(L_{\mathrm{LM}}\). Positions with no allowed strict-past source remain eligible but have privileged/student correction exactly zero by construction.

**Functional LM loss:**

$$
L_{\mathrm{student}}
=
L_{\mathrm{LM}}
+
\lambda_{\mathrm{corr}}L_{\mathrm{corr}}
$$

Locked first-run:

$$
\lambda_{\mathrm{corr}}=1.0
$$

after \(\sigma_\delta\) normalization.

- Privileged-reference parameters are frozen during student training.

- Backbone remains frozen during this probe.

- No gradient from \(L_{\mathrm{corr}}\) may enter privileged-reference targets or Pass-A deep states.

- LM gradients are allowed through the student branch and frozen tail activations to student parameters.

- If \(\sigma_\delta\) is non-finite or \(\le 10^{-8}\), stop because the privileged reference is producing no usable correction. Otherwise do not retune \(\lambda_{\mathrm{corr}}\) based on observed losses in this locked run.

**Student-arm schedule:**

- exactly 610 updates / 19,988,480 input tokens
- global batch = 32,768
- AdamW betas = `(0.9, 0.95)`
- eps = `1e-8`
- peak LR = `3e-4`
- 31-update warmup
- cosine decay to `3e-5`
- weight decay = `0.01` on matrices
- weight decay = `0` on norms/biases/gate biases
- grad clip = `1.0`
- identical ordered batches across matched arms

**Development/final-use rule:** after all full-run arms finish, evaluate them first on `D_probe_dev` and apply the exploratory go/no-go criteria there. If any primary gate fails, stop and leave `D_probe_test` untouched. If all primary gates pass, freeze all interpretations and evaluate once on `D_probe_test` with no further tuning, layer changes, loss changes, or retraining.

## 11.10 Required controls

**Fairness requirement:** for each screened/full-run pair, Privileged-Deep and Shallow-ExtraAttn use byte-identical Q/K/V/O/gate/RMSNorm initialization where shapes match; only the source tensor differs (\(H^d\) vs. \(H^s\)). Shallow-ExtraAttn and Student-PCC use one shared byte-identical student initialization.

| Arm | Definition |
|---|---|
| Base | Frozen pretrained backbone, no auxiliary branch. |
| Privileged-Deep | Frozen backbone + trained privileged deep-source adapter; privileged reference, not a mathematical upper bound. |
| Shallow-ExtraAttn | Same one-pass student architecture, shallow \(H^s\) as K/V source, trained only by LM loss (`lambda_corr=0`). This is the matched control for whether an extra shallow attention branch alone explains gains. |
| Student-PCC | Same student architecture with `lambda_corr=1.0` and frozen privileged correction targets. |
| Target-Permuted (conditional required control) | If Student-PCC passes the dev PCC + distillation gates, train this arm before touching `D_probe_test`. Use the same student initialization/data order/schedule and same LM + correction loss, but deterministically permute \(\delta^*\) across different examples/positions within coarse buckets of relative-position decile and teacher-correction-norm decile (fixed seed `20260922`). This preserves coarse position/magnitude marginals while destroying target semantics. |

## 11.11 Metrics and statistics

- Overall target-token-weighted causal NLL and perplexity.

- Privileged gain:

$$
\operatorname{NLL}(\mathrm{Base})
-
\operatorname{NLL}(\mathrm{Privileged\mbox{-}Deep})
$$

- Student gain:

$$
\operatorname{NLL}(\mathrm{Base})
-
\operatorname{NLL}(\mathrm{Student\mbox{-}PCC})
$$

- Deep-source advantage:

$$
\operatorname{NLL}(\mathrm{Shallow\mbox{-}ExtraAttn})
-
\operatorname{NLL}(\mathrm{Privileged\mbox{-}Deep})
$$

- Distillation-specific gain:

$$
\operatorname{NLL}(\mathrm{Shallow\mbox{-}ExtraAttn})
-
\operatorname{NLL}(\mathrm{Student\mbox{-}PCC})
$$

- Target-semantics gain:

$$
\operatorname{NLL}(\mathrm{Target\mbox{-}Permuted})
-
\operatorname{NLL}(\mathrm{Student\mbox{-}PCC})
$$

reported when the conditional control is run.

- Operational recovery ratio:

$$
\rho
=
\frac{\mathrm{Student\ gain}}{\mathrm{Privileged\ gain}}
$$

reported only when Privileged gain \(>0\); this is recovery relative to the specific Privileged-Deep reference, not a true upper bound.

- Mean/median/p90 \(\|\delta^*\|\) and \(\|\hat\delta\|\); cosine\((\hat\delta,\delta^*)\).

- Privileged/student gate statistics.

- Training/eval tokens/s and peak accelerator memory.

- Paired bootstrap over the fixed evaluation sequences/contexts: resample complete evaluation sequences with replacement and keep the same resampled sequence indices for every compared arm. For each bootstrap replicate, compute NLL as total summed token loss divided by total target-token count across the resampled sequences (do not average per-sequence NLLs). Use 2,000 resamples on `D_probe_dev` and 5,000 on locked `D_probe_test`; seed=`20260922`; report 95% CIs for all primary NLL contrasts.

## 11.12 Exploratory go/no-go criteria

These are engineering/research gates for deciding whether to invest in a tiny from-scratch pilot; they are not final-paper preregistration.

- `D_probe_dev` Privileged gate:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Privileged\mbox{-}Deep}}
-
\operatorname{NLL}_{\mathrm{Base}}
\right)
<0
$$

and point improvement \(\ge 0.0005\) nats/target-token.

- `D_probe_dev` Deep-source gate:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Privileged\mbox{-}Deep}}
-
\operatorname{NLL}_{\mathrm{Shallow\mbox{-}ExtraAttn}}
\right)
<0.
$$

If this fails, the evidence supports an extra-attention effect rather than privileged deep information.

- `D_probe_dev` PCC gate:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Student\mbox{-}PCC}}
-
\operatorname{NLL}_{\mathrm{Base}}
\right)
<0.
$$

- `D_probe_dev` Distillation gate:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Student\mbox{-}PCC}}
-
\operatorname{NLL}_{\mathrm{Shallow\mbox{-}ExtraAttn}}
\right)
<0.
$$

- **Conditional target-semantics gate:** if PCC + distillation gates pass, train/evaluate Target-Permuted on `D_probe_dev` and require:

$$
\operatorname{upper95CI}
\left(
\operatorname{NLL}_{\mathrm{Student\mbox{-}PCC}}
-
\operatorname{NLL}_{\mathrm{Target\mbox{-}Permuted}}
\right)
<0
$$

before unlocking `D_probe_test`.

- `D_probe_dev` Recovery gate:

$$
\rho \ge 0.25.
$$

- **Locked-test rule:** unlock `D_probe_test` only after Privileged, deep-source, PCC, distillation, recovery, and (when triggered) target-semantics gates pass on `D_probe_dev`. Then evaluate once with no post-test method change.

- If the full selected configuration fails, do not scale. Interpret it together with the pre-registered layer-pair screen. If no screened pair showed a deep-source advantage, abandon this family. If Student-PCC does not beat Shallow-ExtraAttn, the correction-distillation mechanism is unsupported even if an extra attention branch helps.

## 11.13 Required acceptance tests before running probe data

1. **Vanilla equivalence:** Pass A equals the unmodified model within bf16 tolerance.

2. **Zero-init equivalence:** untrained Privileged-Deep and Student branches produce Base logits.

3. **Strict-past causality:** teacher/student attention at \(t\) has zero mass for \(j\ge t\).

4. **Future-token perturbation:** changing tokens after \(t\) cannot change \(c_t^{\mathrm{priv}}\) or \(\hat\delta_t\).

5. **Context-mask equivalence:** the auxiliary branch uses exactly the same allowed past-token visibility as the backbone for the fixed preprocessing pipeline, with the additional strict-past condition \(j<t\).

6. **Clean-source guarantee:** every \(H^d\) used by Privileged-Deep comes from Pass A, never from the privileged rerun.

7. **Backbone frozen:** no backbone parameter has `requires_grad=True` in optimizer groups and no backbone parameter changes after an update.

8. **Teacher detach:** student correction loss produces zero gradient into privileged-reference parameters and \(H^d\).

9. **Target-permutation wiring:** the permutation is deterministic, maps to a different example/position, preserves the declared coarse position/norm buckets, and does not accidentally preserve the original target index.

10. **Hook correctness:** every candidate/selected hook \(\{4,8,16,20,24\}\) used by the protocol is captured after the full block residual additions and before the next block RMSNorm.

11. **Auxiliary RoPE correctness:** each 128-d auxiliary Q/K head uses the same Qwen rotary convention and exact `position_ids` as the backbone for the same model-context input.

12. **Empty-source safety:** for every query position with no source allowed by the intersection of the backbone mask and strict-past mask, privileged/student auxiliary attention outputs and corrections are exactly zero and contain no NaN/Inf.

13. **Correction-scale correctness:** \(\sigma_\delta\) matches a direct fp64 reference on a tiny batch and is reused identically across student arms.

14. **Preprocessing equivalence:** all matched arms receive token-identical model-context sequences from the existing fixed CulturaX pipeline, including identical EOS behavior, packing/concatenation, attention masks, and `position_ids`.

15. **Paired-batch correctness:** matched arms consume identical tokenized model-context sequences in identical order and identical optimizer-update grouping.

16. **Full-run freshness:** the full-run adapters are freshly initialized with `full_adapter_init_seed=2901` and no screened trained adapter checkpoint is loaded.

17. **Data fairness:** all matched arms use the same fixed train data, same data-order seed, same token/update budget, and the same fixed dev/test evaluation sequences.


## 11.14 Artifacts Codex must save

- Pinned model/tokenizer revision/config and code commit.

- The fixed train/dev/test data specification (or existing split files/config), the fixed data-order seed, and the exact preprocessing/packing/concatenation configuration used by all arms.

- Privileged-Deep module checkpoint and config.

- Shallow-ExtraAttn, Student-PCC, and any optional Target-Permuted diagnostic checkpoints/configs.

- Per-evaluation-sequence summed loss/count files for paired bootstrap.

- Correction/gate diagnostic summaries.

- Throughput and peak-memory logs.

- A machine-readable decision JSON with all go/no-go criteria and pass/fail outcomes.

# 12. Review notes: what was ambiguous in v1 and is now resolved

| v1 ambiguity | v2 resolution |
|---|---|
| Privileged teacher definition | v1 did not specify how true deep-past feedback is injected. Current spec locks a strict-past cross-attention adapter using clean Pass-A deep source states from the selected \(d\). |
| Circularity | v1 could be read as using deep states generated after privileged feedback. Current spec forbids this; privileged source states always come from the clean feedback-free Pass A. |
| Same-token leakage | v1 did not explicitly forbid using \(h_t^d\) to modify \(h_t^s\). v2 uses \(j<t\) only. |
| Hook coordinates | v1 used generic \(h^s/h^d\). Current spec screens fixed candidate pairs and locks post-block residual hooks before the next RMSNorm. |
| Student architecture | v1 said \(C_\phi(H_{\le t}^s)\) but did not define \(C_\phi\). Current spec locks a width-256, 2-head × 128 auxiliary strict-past causal attention branch. |
| Loss | Current spec locks teacher-RMS-normalized SmoothL1 correction loss plus ordinary LM loss; raw cosine loss was removed to avoid instability near zero corrections. |
| Controls | Current spec requires Shallow-ExtraAttn as the matched extra-attention control and conditionally requires Target-Permuted to test whether correct privileged-target semantics matter. |
| Statistics | v1 had qualitative kill tests. Current spec uses paired bootstrap over the same fixed evaluation sequences and explicit exploratory go/no-go thresholds. |

# 13. Codex execution order

1. Implement explicit hooks, auxiliary RoPE using the backbone's exact `position_ids`, strict-past masking intersected with the backbone attention mask, and all acceptance tests.

2. Reuse the exact existing user-selected CulturaX subset and fixed train/dev/test split. Reuse one deterministic training stream with the locked data-order seed and the existing fixed preprocessing/context-construction configuration; every matched arm must consume the same tokenized stream and token budget.

3. Verify Pass-A vanilla equivalence and all candidate-hook captures on the pinned Qwen3-0.6B-Base revision.

4. Run the four-pair layer screen. For every pair train Privileged-Deep and matched Shallow-ExtraAttn for 128 updates; bootstrap `D_screen_dev` and select/freeze a pair only by the preregistered eligibility/tie-break rule.

5. If the layer screen is negative, stop and write results. Do not add new layer pairs post hoc.

6. For the selected pair, start from fresh `full_adapter_init_seed=2901`, train full Privileged-Deep for 610 updates; freeze it; compute \(\sigma_\delta\).

7. Train full Shallow-ExtraAttn and Student-PCC for 610 updates with paired data order and initialization.

8. Evaluate Base, Privileged-Deep, Shallow-ExtraAttn, and Student-PCC on `D_probe_dev`; run the 2,000-replicate bootstrap and apply Privileged/deep-source/PCC/distillation/recovery gates.

9. If PCC + distillation gates pass, train Target-Permuted with the preregistered permutation rule; evaluate it on `D_probe_dev` and apply the target-semantics gate.

10. If any required dev gate fails, stop and leave `D_probe_test` untouched. If all required gates pass, freeze interpretation and unlock `D_probe_test`.

11. Evaluate Base, Privileged-Deep, Shallow-ExtraAttn, Student-PCC, and Target-Permuted once on locked `D_probe_test`; run the 5,000-replicate confirmatory bootstrap and write decision JSON.

12. Only if dev gates pass and locked-test results remain directionally consistent, draft a separate tiny-from-scratch pilot specification. Do not auto-launch it.
