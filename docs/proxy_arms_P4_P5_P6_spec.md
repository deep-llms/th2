# Proxy Arms P4, P5, P6 — Architecture and Implementation Specification

**Revision 3 — 6 October 2026.** This document specifies three new proxy arms, **P4**, **P5** and **P6**, for the from-scratch Qwen3-0.6B-style screen. Their single goal is to **beat arm A**. It is an implementation contract: an implementation that deviates from Sections 3–7 is a different experiment and must be labelled as such.

Revision 3 adds the P1 run with gate initialized to 1, and introduces **estimator isolation**:
- **Isolation option.** The injection path uses `sg(ŷ)`, so only the auxiliary loss trains the estimator.
- **New arm P4-iso.** P4 with isolation.
- **P5 default.** Isolation is on.
- **Run order.** Revised.
- **New test T10.**

Revision 2 (retained) changes:
- **New requirement:** the backbone initialization must be identical to arm A, with a new test (T9).
- **Additional A seed:** varies initialization only.
- **T2:** stated per block.
- **T4:** uses a tolerance.
- **Block-routing cost:** corrected, and a no-recompute option added.
- **Overhead figure:** corrected to 12%.
- **Loss accumulation rule:** marked as new in this document.

Components shared with P1/P3 — base model, document isolation, the P1 target, target normalization, the auxiliary-loss schedule and the integration-test conventions — are defined in the P1/P3 specification, Revision 7 [1]. They are referenced here, not repeated, except where this document changes them.

## 1. Motivation from the P1/P3 screen

Results at 2,500 steps, one seed, dense-SDPA document isolation [2]:

| Model | Validation LM loss ↓ |
|---|---:|
| A | 3.477173 |
| P1-block | 3.477280 |
| P3-block | 3.478909 |
| P3-block, proxies disabled at evaluation | 3.479060 |

**The proxies were effectively unused.**
- In P1-block, the mean |α| was 0.001–0.0034 at step 2,420, so the proxy term contributed roughly 0.1–0.3% of the proxy K/V input.
- Disabling P3's proxies changed loss by only 0.00015.

**The estimator nevertheless learned.** P1's final auxiliary loss was about 0.412, i.e. a mean cosine of about 0.59 between predictions and the normalized target.

**Diagnosis: zero-initialized gate.** The gradient of α measures the benefit of a small amount of `ŷ` *given the current weights*. The heads had never learned to read `ŷ` while α ≈ 0, so that benefit stayed near zero, and α never grew. A zero-initialized gate therefore never tested whether the anticipated information is useful.

**Follow-up: P1 with α initialized to 1** [2]. This is the same P1 design (injection into keys **and** values of 2 of 8 KV groups), with gates starting at 1.

| Model | Validation LM loss ↓ | Final auxiliary loss | Mean cosine |
|---|---:|---:|---:|
| A | 3.47717 | — | — |
| P1, α init 0 | 3.47728 | 0.41191 | ≈ 0.59 |
| P1, α init 1 | 3.48232 | ≈ 0.5590 | ≈ 0.44 |
| P1, α init 1, proxies disabled at evaluation | 3.52501 | — | — |

- **The proxy is used and kept.** Mean |α| at step 2,500 was 0.984 (per-layer range 0.975–1.003), and disabling the proxies raised loss by 0.043.
- **The arm is nevertheless worse than A, by 0.005.**
- **The estimator drifted away from its target.** Mean cosine fell from about 0.59 to about 0.44. With α = 1, the LM loss sends a strong gradient into the estimator through the injection path, while the auxiliary loss carries weight only 0.1. The estimator therefore becomes a general feature extractor for the proxy heads rather than a predictor of upper-block outputs. Targets also shift as the backbone adapts, which may contribute.

What this run actually tested was "extra nonlinear K/V features in 25% of heads", which hurt. It did **not** test whether *anticipated deep information* is useful. Two likely causes of the harm:
1. **Injection into keys** perturbs which tokens the proxy heads select.
2. **Estimator drift:** the injected content is no longer a deep-state prediction.

P4 (values only, keys native) addresses the first cause. **Estimator isolation** (Section 2.4) addresses the second.

**Design principles for P4–P6:**
1. **Forced use.** Every injection gate starts **non-zero**, so the network co-adapts to the proxy from step 0.
2. **Additive injection only.** Native computation is kept and the proxy is added to it, so no native capacity is removed.
3. **Broader application** than P1/P3, because per-token anticipation is a modest mechanism and needs to act on many heads to show an effect.
4. **Low wall-clock overhead** (Section 6), because P1 ran about 12% slower than A (98.6 vs 88 training minutes) for about 0.9% extra FLOPs.
5. **Keys untouched and, where specified, the estimator isolated**, so that the injected content stays a deep-state prediction and attention routing is not perturbed.

All three arms use **per-token** estimators. They can anticipate what upper blocks will compute for the same token, typically 1–4 blocks before that computation happens, but not cross-token context. Expected effects are modest.

## 2. Shared definitions

### 2.1 Base model, attention and indexing

These are as in [1, §2]:
- Qwen3-0.6B-style decoder: 28 blocks, `d = 1024`, 16 query heads, 8 KV heads, head dimension 128.
- **Dense SDPA document isolation**, with `position_ids` reset at every document start.
- Notation `u_ℓ`, `a_ℓ`, `m_ℓ`, `h_ℓ` as in [1, §2.2].

Blocks are numbered **from 1**. The proxy layer set is

$$
\mathcal{P} = \{2, 4, 6, \dots, 24\}\quad(12\ \text{blocks}),
$$

which corresponds to **`model.layers[1], [3], …, [23]`** in 0-indexed code. Verify this mapping in the implementation (test T0).

### 2.2 Per-token estimator (P4, P6)

As P1 [1, §4.1], with default (non-zero) initialization:

$$
\hat y_\ell(j) = W_{2,\ell}\,\mathrm{SiLU}\big(W_{1,\ell}\,u_\ell(j)\big),\qquad
W_1\in\mathbb{R}^{256\times 1024},\; W_2\in\mathbb{R}^{1024\times 256}.
$$

### 2.3 Target, normalization and auxiliary loss

These are unchanged from P1:
- **Target** [1, §4.2]: the normalized sum of the MLP outputs of the next four blocks,

$$
y_\ell(j) = \mathrm{sg}\Big(\mathcal{N}_\ell\Big(\sum_{i=\ell}^{\ell+3} m_i(j)\Big)\Big).
$$

- **Normalization** `N` [1, §6]: per-channel running standardization, relative variance floor, clipping at ±10, two-pass initialization, statistics frozen during recomputation and evaluation, and `target_version` recorded. For these arms set `target_version = "p4p6-r1"`.
- **Loss** [1, §4.3, §7]: cosine, averaged over all tokens of the micro-batch and then over proxy layers. **New in this document:** the auxiliary loss is scaled exactly like the LM loss under gradient accumulation and data parallelism (same division by the number of micro-batches, same gradient averaging across ranks).

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda(s)\cdot\frac{1}{|\mathcal{P}|}\sum_{\ell\in\mathcal{P}}\mathrm{mean}_j\Big(1-\cos\big(\hat y^{\,\text{aux}}_\ell(j),\,y_\ell(j)\big)\Big),\qquad
\lambda(s) = 0.1\,\min(s/250,\,1).
$$

- **Routing: block** (the default). The auxiliary gradient reaches only the estimator parameters. The LM-loss gradient flows normally everywhere. Two equivalent implementations are allowed:
  - **(a) Recompute:** compute `ŷ^aux` with the estimator on `sg(u_ℓ)`. This costs one extra estimator forward per proxy layer during training.
  - **(b) No recompute (preferred):** a custom autograd function returns the forward `ŷ_ℓ` for the auxiliary loss. In backward, it routes the auxiliary gradient to the estimator weights only, using the saved forward activations, and passes zero to `u_ℓ`.

  Both must give the same parameter gradients (test T4).
- The learning-rate schedule is identical to arm A.

**Gate convention for all arms.** Gates `α_ℓ ∈ ℝ^d` are per-channel, learnable, excluded from weight decay, and initialized **non-zero** as each arm specifies.

`RMSNorm_0(x) = x / sqrt(mean(x²) + ε)`, with `ε = 1e-6` and no learned weight.

### 2.4 Estimator isolation (`isolate_estimator`)

Every arm injects a quantity `ŷ^inj_ℓ` defined by a per-arm flag:

$$
\hat y^{\,\text{inj}}_\ell = \begin{cases}\mathrm{sg}(\hat y_\ell) & \texttt{isolate\_estimator = true}\\ \hat y_\ell & \texttt{isolate\_estimator = false}\end{cases}
$$

**With isolation on**, gradients are routed as follows:

| Parameters | LM loss | Auxiliary loss |
|---|---|---|
| Estimator (or stream) | **No** | Yes |
| Gates `α`, projections that read the injection (e.g. `W_V`), rest of the model | Yes | No (block routing) |

- The injected content therefore stays a **prediction of upper-block outputs**, which is the quantity the proxy hypothesis is about, and cannot drift into an arbitrary feature extractor as happened in P1 with α = 1 (Section 1).
- The model still learns *how much* and *how* to use it, through `α` and the downstream weights.
- **Optimization note.** With isolation, the auxiliary loss is the estimator's only training signal. Under Adam the size of `λ` barely changes the estimator's update magnitude, because Adam normalizes per-parameter gradient scale. The ramp of `λ` (which is 0 only at step 0) therefore does not stall the estimator.

**With isolation off**, both losses train the estimator, as in P1.

Record `isolate_estimator` in the run configuration alongside `target_version`, and refuse to resume across a change of this flag.

## 3. P4 — Anticipatory values in all heads

### 3.1 Idea

In every proxy layer, keep queries and keys native and add the anticipated content to the **values of all eight KV groups**. Attention still selects tokens as in the base model; the proxy enriches only *what is read out*.

### 3.2 Definition

For ℓ ∈ 𝒫:

$$
q = W_Q\,u_\ell,\qquad k = W_K\,u_\ell,\qquad
v = W_V\,\big(u_\ell + \alpha_\ell\odot\mathrm{RMSNorm}_0(\hat y^{\,\text{inj}}_\ell)\big),\qquad
\alpha_\ell = \mathbf{1}\ \text{at initialization},
$$

with `ŷ^inj` as in Section 2.4. Two arms share this definition:

| Arm | `isolate_estimator` |
|---|---|
| **P4** | false |
| **P4-iso** | **true** |

- QK-norm, RoPE, the document mask and the SDPA call are unchanged.
- **Implementation:** form `z_v = u + α ⊙ RMSNorm_0(ŷ^inj)` and call the existing `v_proj` once on `z_v`. No extra projection is needed.
- With all `α = 0`, the block is exactly the vanilla block (test T1).
- **Early steps.** With `α = 1` the injected term has the same RMS as `u` from step 0. During the λ ramp the estimator is still nearly untrained, so the injection is close to noise. This is acceptable: the whole model is untrained at that point, and the learning-rate warm-up keeps early changes to `α` small, so `α` is not driven to zero before the estimator becomes informative. Log |α| from step 0 to confirm.

### 3.3 Rationale and evidence

- Because the head keeps `W_V u`, native capacity is preserved and the proxy is purely additive. This is why P4 applies to all heads, unlike P1's two groups.
- Keys stay native, which removes the routing perturbation that P1 with α = 1 introduced. **P4-iso** additionally keeps the injected content a deep-state prediction, removing the estimator drift seen in that run. P4-iso is therefore the cleanest test of the proxy hypothesis.
- In the Latent Recurrent Transformer ablations, injecting into values outperformed injecting into keys, and combined KV-plus-residual injection was best [3]. That evidence concerns recurrent memory of the previous token's deep state, not per-token anticipation, so it motivates P4 rather than proving it.

### 3.4 Cost

The estimator costs `2 · 1024 · 256 ≈ 0.524 M` MAC per proxy layer, or **6.29 M MAC/token** for 12 layers (about 0.9% of a ~0.71 G MAC/token forward pass). Elementwise operations are negligible. In training, block routing adds either a full extra estimator forward (0.524 M MAC per proxy layer, implementation (a)) or nothing (implementation (b)); see Section 2.3.

## 4. P5 — Lookahead stream with P4 injection

### 4.1 Idea

Replace P1's independent per-layer estimators with **one narrow residual stream** (width 256) that runs through the proxy layers. Each proxy layer refines the previous layer's estimate instead of predicting from scratch, so estimation capacity accumulates with depth. The aim is to raise prediction quality above P1's cosine of about 0.59.

### 4.2 Definition

The stream `a_ℓ ∈ ℝ^{256}`, per token, for ℓ ∈ 𝒫:

$$
a_2 = W_{d,2}\,u_2,\qquad
a_\ell = a_{\ell-2} + \Delta_\ell\big([\,W_{d,\ell}\,u_\ell;\ a_{\ell-2}\,]\big)\quad(\ell = 4, 6, \dots, 24),
$$

$$
\hat y_\ell = W_{\text{up},\ell}\,a_\ell .
$$

- **Shapes:** `W_{d,ℓ} ∈ ℝ^{256×1024}` is **separate for each layer**. `Δ_ℓ` is `Linear(512→256) → SiLU → Linear(256→256)`. `W_{up,ℓ} ∈ ℝ^{1024×256}`. No biases; default initialization.
- **Injection:** exactly as P4 (Section 3.2), using `ŷ^inj_ℓ`: values of all eight KV groups, `α_ℓ = 1` at initialization.
- **Isolation: on by default** (`isolate_estimator = true`). P5's purpose is to improve *prediction* quality, which is only meaningful if the stream is trained as a predictor. With isolation the LM loss does not reach `W_d`, `Δ` or `W_up`.
- **Target and loss:** as Section 2.3, applied to each `ŷ_ℓ`.
- **Block routing:** the auxiliary gradient must reach only `W_d`, `Δ` and `W_up` (across all layers of the stream), never the backbone. Two options:
  - implementation (a): recompute the entire stream from `sg(u_ℓ)` inputs;
  - implementation (b): wrap the stream so that the auxiliary gradient into each `u_ℓ` is zero.
- The stream is per-token: it mixes information across depth, never across positions, so it cannot leak across documents.

### 4.3 Cost

| Component (per proxy layer) | MAC/token |
|---|---:|
| `W_{d,ℓ}` | 0.262 M |
| `Δ_ℓ` (absent at ℓ = 2) | 0.197 M |
| `W_{up,ℓ}` | 0.262 M |

The total is **8.45 M MAC/token** for 12 layers, about 1.2% of the forward pass.

### 4.4 When to run

Run P5 after P4-iso. If P4-iso shows that isolated deep-state predictions help, P5 tests whether better predictions help more. P5 only improves estimate quality and does not overcome the per-token limit of Section 1.

## 5. P6 — Anticipatory residual injection

### 5.1 Idea

Change no head. Before each proxy block, add the anticipated upcoming MLP output **directly to the residual stream**. Queries, keys, values and the MLP of block ℓ and of every later block then see it, and later tokens see it through attention.

### 5.2 Definition

For ℓ ∈ 𝒫, with `ŷ_ℓ` computed from `u_ℓ = InputNorm_ℓ(h_{ℓ−1})` **before** injection:

$$
\tilde h_{\ell-1} = h_{\ell-1} + \alpha_\ell\odot\mathrm{RMSNorm}_0(\hat y^{\,\text{inj}}_\ell)\cdot \mathrm{sg}\big(\mathrm{rms}(h_{\ell-1})\big),\qquad
\alpha_\ell = 0.1\cdot\mathbf{1}\ \text{at initialization},
$$

where `rms(h) = sqrt(mean_c h_c² + ε)` is per token. Block ℓ then runs on `h̃_{ℓ−1}`, including its input normalization.

- **Isolation: off by default** (`isolate_estimator = false`). P6 is the arm aimed most directly at beating A, and it is allowed to use the estimator as an LM-trained module (Section 5.3). `P6-iso` (isolation on) is an optional later variant.

- The injection is scaled **relative to the residual**: it starts at 10% of the residual magnitude at every depth. That is large enough to avoid the zero-gate failure and small enough not to disrupt the residual stream.
- The scale factor is detached, so it acts as a pure magnitude reference.
- With all `α = 0`, the model is exactly vanilla (test T1).
- Targets `m_i` (i ≥ ℓ) are computed from the injected residual stream, as produced by the forward pass, and are detached.

### 5.3 Interpretation

`ŷ_ℓ` is a function of `h_{ℓ−1}`, so P6 is equivalent to inserting **12 narrow MLP sublayers** (width 256) whose outputs are trained to anticipate the upcoming MLP outputs. Some of any gain may come from the added capacity (about 0.9% FLOPs). For this screen, where the goal is to beat A, that is acceptable; any later claim must keep the distinction.

Upper MLPs may learn to reduce their outputs to compensate for the anticipated contribution, which shifts the target. Targets are detached, so this cannot destabilize training; monitor cosine and |α| over time (Section 8).

### 5.4 Cost

Estimator as P4: **6.29 M MAC/token**. Per-token RMS and elementwise operations are negligible.

## 6. Implementation efficiency (required before runs)

P1 used about 0.9% extra FLOPs but ran about 12% slower than A (98.6 vs 88 training minutes). With time-matched comparisons (Section 7), overhead directly reduces the chance of beating A. Before screening:

- **Profile one P1-style step** and report the time split: estimator forward/backward, auxiliary recomputation, target construction, normalization statistics (including the all-reduce), and the remaining model.
- **Reduce overhead** with:
  - block routing by implementation (b), with no recomputation (Section 2.3);
  - `torch.compile` on the estimator, injection and loss path;
  - batching the per-layer statistics accumulation and doing a single all-reduce per optimizer step for all buffers;
  - computing target window sums incrementally over the 12 layers instead of re-summing;
  - fp32 only where the reference specification requires it (statistics, cosines, sums).
- **Target:** at most 3% wall-clock overhead versus A for P4 and P6, measured as seconds per update. If this is not reachable, report the remaining breakdown before launching.

## 7. Screen protocol and success criterion

**Runs**, in order:
1. **P4-iso:** the cleanest test of the proxy hypothesis.
2. **P6:** the arm aimed most directly at beating A.
3. **P5:** only if P4-iso shows promise.

**P4** (not isolated) and **P6-iso** are optional follow-ups. Each run uses one seed and 2,500 optimizer steps, with data order, seed, schedule and execution settings identical to arm A.

**Identical backbone initialization (required).** Every backbone parameter at step 0 must be bitwise identical to arm A's:
1. Initialize the backbone first, with A's seed and in A's order.
2. Initialize the new modules (estimators, stream, gates) from a **separate** random generator, so they do not consume the backbone's random stream.

Otherwise each arm effectively uses a different seed, and differences of order 1e-3 could be initialization noise. This was a suspected cause of P3-block's +0.0017 gap, since disabling its proxies changed loss by only 0.00015. Test T9.

**Seed spread of A:** train **one additional run of A** with a different **initialization** seed and the **same data order**. This matches the comparison setting, in which arms share A's data order but follow different training trajectories. Without it, differences of order 1e-3 cannot be interpreted.

**Comparisons**, all on the same fixed held-out set:

| Reference | Definition |
|---|---|
| **A, token-matched** | A at step 2,500 (existing dense-SDPA isolated run) |
| **A, time-matched** | A resumed from its step-2,500 checkpoint, with the same schedule, to step `round(2500 · t_arm / t_A)`, where `t` is measured training time excluding preprocessing. For example, P1's 98.6 vs 88 minutes gives step 2,801 |

**Success:** the arm beats **time-matched A** by more than A's seed-to-seed spread. Beating token-matched A by more than the spread marks an arm as promising; optimize its overhead before concluding.

Controls (λ = 0 arms, compute-matched vanilla) are **deferred** until an arm beats A.

## 8. Logging and evaluation

**Every logging interval:**
- LM loss, gradient norms, seconds per update;
- per proxy layer: auxiliary loss, cosine, and mean |α|;
- normalization diagnostics as in [1, §10]: clip fraction, floored channels, median `σ²`, lag ratios.

**Evaluation**, on the held-out set:
- LM loss;
- **proxy-disabled ablation:** the same checkpoint with every `α = 0`.

Read α together with the ablation:

| Observation | Meaning |
|---|---|
| \|α\| stays near or above its initial value, and disabling hurts | The model uses the proxy |
| \|α\| decays toward 0, and disabling has no effect | The model actively rejects the proxy. This is real negative evidence, unlike the zero-gate runs |
| \|α\| stays, disabling hurts a lot, yet the arm is worse than A | The model has come to depend on the proxy path, but the arm is harmful overall (the pattern of P1 with α = 1). Check the auxiliary cosine: a fall over training (non-isolated arms) indicates estimator drift |

For isolated arms the auxiliary cosine should not fall the way P1's did (0.59 → 0.44). If it does, the targets themselves are shifting, not the estimator drifting.

## 9. Integration tests

Conventions as in [1, §9]: fp32 where tolerances are tight, normalization buffers frozen except where stated, and **stop and report** on any failure without modifying this specification.

| ID | Test | Pass criterion |
|---|---|---|
| T0 | Proxy-layer placement | Modules attached exactly to `model.layers[1], [3], …, [23]` (1-indexed blocks 2, …, 24) |
| T1 | **α = 0 equivalence:** set every α to 0 (P4, P4-iso, P5, P6) and compare with vanilla using identical shared weights | Logits equal; fp32 max abs diff ≤ 1e-5 |
| T2 | **P4/P4-iso/P5 path isolation, per block:** for each proxy block, feed the **same input** `h_{ℓ−1}` to the proxy block (random non-zero α) and to the vanilla block with identical weights | Queries and keys bitwise identical; only values differ. (Across the whole model, later blocks' inputs legitimately differ once an earlier block injects) |
| T3 | Cross-document leakage, as [1, T3] | Gradients w.r.t. document-1 inputs exactly 0; document-2 activations, targets and P5 stream values unchanged under document-1 perturbation |
| T4 | **Block-routing equivalence:** (i) for implementation (a), the recomputed estimator or stream output vs the forward output on the same inputs; (ii) parameter gradients of the auxiliary loss under implementations (a) and (b) | (i) and (ii) agree within fp32 max relative error ≤ 1e-6; the auxiliary gradient into every `u_ℓ` is exactly 0 |
| T5 | Gradient routing at initialization (backpropagate the LM loss and the auxiliary loss separately) | Auxiliary loss finite; α receives non-zero LM gradient; the auxiliary loss alone gives exactly zero gradient on backbone parameters |
| T6 | Targets detached; normalization tests [1, T6, T9] | As in [1] |
| T7 | **P6 injection scale at initialization:** per layer, RMS of the injected term ÷ RMS of `h_{ℓ−1}` | Between 0.09 and 0.11 |
| T8 | Compute and throughput | MAC counts reported; seconds/update vs A reported with the profile breakdown of Section 6 |
| T9 | **Backbone initialization:** instantiate each arm and arm A with the same seed; compare all backbone parameters at step 0 | Bitwise identical; new-module initialization does not change when the backbone seed is fixed and the module seed changes, and vice versa |
| T10 | **Isolation routing** (`isolate_estimator = true`; P4-iso, P5, and P6-iso if run): backpropagate the LM loss and the auxiliary loss separately at initialization | LM loss alone: exactly zero gradient on every estimator/stream parameter, and non-zero gradient on `α` and `W_V` (or the residual path for P6-iso). Auxiliary loss alone: exactly zero gradient on `α`, `W_V` and all backbone parameters, and non-zero gradient on the estimator/stream |

## 10. Defaults

| Parameter | P4-iso | P4 | P5 | P6 |
|---|---|---|---|---|
| Run order | 1 | optional | 3 | 2 |
| Proxy layers | {2, …, 24} | {2, …, 24} | {2, …, 24} | {2, …, 24} |
| Injection site | Values, all 8 KV groups | Values, all 8 KV groups | Values, all 8 KV groups | Residual before block ℓ |
| Estimator | Per-layer MLP 1024→256→1024 | same | Lookahead stream, width 256 | Per-layer MLP 1024→256→1024 |
| `isolate_estimator` | **true** | false | **true** | false (P6-iso optional) |
| Gate init | α = 1 | α = 1 | α = 1 | α = 0.1 (relative to residual RMS) |
| Target | Σ MLP outputs of blocks ℓ…ℓ+3, normalized [1, §6] | same | same | same |
| Aux loss / λ | Cosine; λ = 0.1 after 250-step ramp | same | same | same |
| Routing | block, implementation (b) preferred | same | block, (b) preferred; (a) = full-stream recompute | block, (b) preferred |
| Backbone init | Bitwise identical to A (separate RNG for new modules) | same | same | same |
| Extra MAC/token | 6.29 M | 6.29 M | 8.45 M | 6.29 M |
| `target_version` | p4p6-r1 | p4p6-r1 | p4p6-r1 | p4p6-r1 |

## 11. Scope

- This document covers only the P4–P6 screen. P1/P3 remain defined by [1]. P1 with α initialized to 1 has been run (Section 1); further P1/P3 variants are outside this document.
- All estimators are per-token. Arms with cross-token estimators (for example P3-style moving sums with forced injection) are a possible later revision.
- Decode behavior is unchanged in kind from the base model: P4 and P6 each add one small MLP per proxy layer per token, and P5 adds the stream update.

## Sources

[1] `proxy_heads_P1_P3_spec_v3.md`, Revision 7 — base model, document isolation, P1 target, normalization (Section 6), auxiliary-loss schedule and integration-test conventions.

[2] P1/P3 screen reports, 5–6 October 2026 — validation losses; P1 (α init 0) gate magnitudes (mean |α| 0.001–0.0034 at step 2,420), final auxiliary loss 0.41191; training times A 88 min and P1 98 min 35 s. P1 with α initialized to 1: validation loss 3.48232, 3.52501 with proxies disabled, final auxiliary loss ≈ 0.5590, mean |α| 0.98370 at step 2,500 (per-layer 0.97468–1.00298), runtime 99 min 37 s.

[3] Huang et al., *Latent Recurrent Transformer*, arXiv 2605.26797 — injection-site ablation: value injection outperformed key injection; combined KV-plus-residual injection was best (recurrent memory of the previous token's deep state).

[4] Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP 2021 — background for per-token anticipation of MLP content.
