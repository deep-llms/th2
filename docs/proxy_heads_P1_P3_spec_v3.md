# Proxy Attention Heads (P1, P3) — Architecture and Implementation Specification

**Revision 4 — 4 October 2026.** This document specifies two proxy-head architectures, **P1** and **P3**, for the from-scratch Qwen3-0.6B-style experiments, together with the screening protocol used to test them. It is an implementation contract: an implementation that deviates from Sections 3–8 is a different experiment and must be labelled as such.

Revision 4 aligns the specification with the existing training setup:
- **Attention:** switched from FA4 varlen to **SDPA with a dense document mask**, chosen for robustness because the FA4 package is a beta release (Section 2.3).
- **Positions:** `position_ids` are **reset to 0 at every document start**, as in the current setup (Section 2.3).
- **P3 resets:** stated explicitly to coincide with attention isolation, so no component of the model carries information across documents.

Changes retained from Revision 3:
- It replaces both the dense `T×T` decay matrix and the random target sketch of Revision 2 with an **exact two-level chunkwise scan**. This scan gives the same values as the dense form, keeps P3 targets at full width, and is about 10× cheaper.
- It recomputes the P3 compute budget and the V3 widening accordingly.
- It corrects the role of arm A.

It also fixes `ε`, specifies how `μ` is initialized under data parallelism, and lists the boundary cases the scan tests must cover. Further clarifications in this revision:
- stop and report if the codebase differs from Section 2 or if any integration test fails;
- running buffers `μ` are updated in λ = 0 arms as well.

## 1. Research question

Earlier experiments establish that **real** deep states of past tokens help shallow computation when an extra pass supplies them. In Probe 2, Deep beat a matched Shallow branch by about 0.007 nats/token in two seeds, token-matched but not compute-matched [2, §5]. In contrast, single-pass estimators built from linear projections of a shallow state tied or lost to an extra-branch-only control [1]. The present experiments ask a narrower question:

> In a single, fully parallel forward pass with no teacher, does replacing a small fraction of native KV groups with **proxy** KV groups — whose keys and values carry a learned estimate of deeper-layer information — improve language modelling at matched parameters and FLOPs?

Two design decisions keep the comparison clean:

- **Replace, do not add.** Proxy KV groups take the place of native KV groups, so the head count is unchanged. The only added compute is the small estimator, and a widened-MLP vanilla control matches it.
- **λ = 0 control.** Every proxy architecture is also trained without its auxiliary loss. A gain of the λ > 0 arm over this control is attributable to the *anticipation training signal*, not to the extra nonlinearity in the K/V path.

The two variants test different hypotheses, both untested:

| Variant | What the estimator anticipates | Hypothesis | Context in estimator |
|---|---|---|---|
| **P1** Lookahead-MLP | Sum of the MLP outputs of the next `k` blocks, per token | Upper MLPs largely retrieve token-associated content (MLPs act as key–value memories [4]); a small per-token MLP can predict part of it | No |
| **P3** Context-summary | Exponential moving sums, at three timescales, of deep-band states | The slowly varying part of deep states (topic, entities, discourse) is easier to estimate as a time average | Yes, through exponential moving sums |

P3's estimator uses multi-timescale exponential decay, a mechanism shown to help in RetNet, where both decay and multiple decay rates improved language modelling in ablations [6]. Its novelty here lies only in the target and in its use inside replaced heads.

## 2. Base model, data, and attention

### 2.1 Base architecture

Qwen3-0.6B-style decoder as in the existing codebase:
- 28 blocks (1-indexed), hidden size `d = 1024`;
- 16 query heads and 8 KV heads (GQA: 2 query heads per KV group), head dimension 128;
- pre-norm RMSNorm, QK-norm, RoPE;
- SwiGLU MLP with intermediate size `I` (3072 in Qwen3-0.6B).

Training uses sequence length `T = 2048`, micro-batch 16 sequences × gradient accumulation 4 on 8 GPUs, i.e. 1,048,576 tokens per optimizer step [3]. Everything not specified here is unchanged from the vanilla arm.

**If the codebase differs from these assumptions — for example q/k/v biases, QK-norm placement relative to RoPE, KV-head layout in the projection weights, or block structure differing from Section 2.2 — stop and report the difference before implementing. Do not adapt this specification silently.**

### 2.2 Notation

Within block ℓ:

$$
u_\ell = \mathrm{InputNorm}_\ell(h_{\ell-1}),\qquad
a_\ell = \mathrm{o\_proj}\big(\mathrm{Attn}(q,k,v)\big),\qquad
r_\ell = h_{\ell-1} + a_\ell,
$$

$$
m_\ell = \mathrm{MLP}_\ell\big(\mathrm{PostAttnNorm}_\ell(r_\ell)\big),\qquad
h_\ell = r_\ell + m_\ell ,
$$

with `h_0` the token embeddings.
- **`u_ℓ` is the tensor already used for the native q/k/v projections.**
- **`m_ℓ` is the MLP sublayer output before the residual addition.**
- `sg(·)` denotes `detach()`.
- Token index `j` runs over all tokens of a micro-batch.
- `ε = 1e-6` wherever it appears.

### 2.3 Document isolation

All arms, including the vanilla controls, use **SDPA with a dense document mask** (`torch.nn.functional.scaled_dot_product_attention` with a boolean `attn_mask`). This configuration was validated on the full model [3]:
- dense SDPA isolation and FA4 varlen isolation differed in loss by 0.000042, and both had about 1.23% gradient error against FP32, which is normal for bf16;
- isolation produced exactly zero cross-document output leakage and activation gradients;
- cost: 2.425 s per optimizer step, versus 2.062 s for fast causal SDPA without isolation and 2.072 s for FA4 varlen isolation. The roughly 17% slowdown is accepted in exchange for avoiding the beta FA4 package.

**Mask.**
- Boolean, shape `[B, 1, T, T]`, where `True` means "may attend":

$$
\mathrm{mask}[b,0,i,j] = \big(j\le i\big)\ \wedge\ \big(\mathrm{doc\_id}[b,i]=\mathrm{doc\_id}[b,j]\big).
$$

- Causality is inside the mask, so call SDPA with `is_causal=False`; PyTorch does not allow combining an explicit mask with `is_causal=True`.
- Every row keeps at least its diagonal entry, so no row is fully masked and no NaNs arise.
- Build the mask **once per micro-batch** and reuse it in every layer. It occupies about 67 MB for 16 sequences.
- **GQA:** use the same mechanism as the vanilla arm, either `repeat_kv` (query head `i` reads KV head `i // 2`) or `enable_gqa=True`. All arms must use the same SDPA backend and GQA handling.

**Rules.**
- **Boundaries.** The EOS token belongs to the document it ends; the next token starts a new document. The first token of every 2048-token sequence starts a new document segment.
- **Positions.** Reset `position_ids` to 0 at the first token of every document, as in the current setup, and use the same reset rule in all arms. Under isolation every query–key pair lies within one document and RoPE scores depend only on relative position, so this is mathematically equivalent to not resetting. It is kept for consistency with the existing baseline. Derive `position_ids` from the same `doc_id` as the mask.
- **Metadata.** Compute a per-sequence `doc_id[B, T]` **once per micro-batch**. Derive from it both the attention mask and the P3 scan masks (Section 5.3), and reuse them in every layer.
- **P3 consistency.** P3's moving sums reset at exactly the same document boundaries (the same `doc_id`). Under isolation, no component of the model carries information across documents: attention cannot see earlier documents, and P3 summaries restart at every document. The resets are therefore required, not optional.
- **Comparability.** Earlier A–G runs used SDPA without isolation, so they are **not** comparable to runs under this specification. Arm A is retrained here (Section 8).

## 3. Common proxy-head mechanism

### 3.1 Proxy layers

P1 with lookahead `k` uses the proxy layer set

$$
\mathcal{P}_k = \{\ell \text{ even} : 2 \le \ell \le 29 - k\},
$$

so every target block index is ≤ 28. With the default `k = 4` this is {2, 4, …, 24}, i.e. 12 blocks. **P3 always uses {2, 4, …, 24}**, independent of `k`.

### 3.2 Proxy KV groups

In each proxy layer, the **last `G_p` KV groups** (0-based KV indices `8−G_p, …, 7`) are proxy groups. The default is `G_p = 2`, i.e. 4 of 16 query heads or 25% of the layer's attention; 1 and 4 are also allowed.

The query heads mapped to those groups are the proxy query heads. Under the standard GQA convention, query head `i` uses KV head `i // 2`; **the implementation must verify the convention its kernel uses** (test T2). Queries are unchanged for all heads.

### 3.3 Keys and values

Native groups compute K/V from `u_ℓ` exactly as before. Proxy groups compute K/V from the modified input `ẑ_ℓ`, using the **same projection weights**, namely the rows of `k_proj` and `v_proj` that belong to the proxy groups:

$$
k^{\text{prx}}_j = W^{\text{prx}}_K\,\hat z_\ell(j),\qquad
v^{\text{prx}}_j = W^{\text{prx}}_V\,\hat z_\ell(j).
$$

- No new projection matrices are introduced. Total K/V projection cost is unchanged, because rows are only re-routed.
- Because proxy groups are the last groups, `W^prx` is the trailing `G_p · 128` rows of each weight and `W^nat` the leading rows.
- Native and proxy keys are **concatenated in group order**. QK-norm and RoPE are then applied exactly as in the base model, followed by the single SDPA call with the shared document mask on the full q/k/v tensors (Appendix A).

### 3.4 Gated injection

$$
\hat z_\ell(j) = u_\ell(j) + \sum_{c} \alpha_{\ell,c}\odot \mathrm{RMSNorm}_0\big(\hat y_{\ell,c}(j)\big),
\qquad \mathrm{RMSNorm}_0(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}} ,
$$

where the mean is over channels and the normalization has no learned weight.

- P1 has a single term. P3 has one term per timescale, `c = 1, 2, 3`.
- **`α_{ℓ,c} ∈ ℝ^d` is learnable, initialized to 0, and excluded from weight decay.**
- Estimator weights use the default (normal) initialization, so the estimates are non-zero at step 0 and `α` receives gradient immediately. **Do not zero-initialize estimator weights**: the cosine losses of Sections 4.3 and 5.5 would be undefined.
- With all `α = 0`, the model computes exactly the vanilla function (test T1).

## 4. P1 — Lookahead-MLP heads

### 4.1 Estimator

$$
\hat y_\ell(j) = W_{2,\ell}\,\mathrm{SiLU}\big(W_{1,\ell}\,u_\ell(j)\big),\qquad
W_1\in\mathbb{R}^{256\times 1024},\; W_2\in\mathbb{R}^{1024\times 256}
$$

(PyTorch `[out, in]` shapes), no biases. Injection: `ẑ_ℓ = u_ℓ + α_ℓ ⊙ RMSNorm_0(ŷ_ℓ)`.

### 4.2 Target

The target is the sum of the MLP sublayer outputs of the next `k` blocks, starting with block ℓ, whose MLP runs after its attention:

$$
y_\ell(j) = \mathrm{sg}\Big(\mathcal{N}^{\text{P1}}_\ell\Big(\sum_{i=\ell}^{\ell+k-1} m_i(j)\Big)\Big),\qquad k = 4 .
$$

`N` is defined in Section 6. Accumulate the window sum in fp32 from detached `m_i`.

### 4.3 Auxiliary loss

$$
\mathcal{L}^{(\ell)}_{\text{aux}} = \mathrm{mean}_j\Big(1-\cos\big(M\odot \hat y^{\,\text{aux}}_\ell(j),\; y_\ell(j)\big)\Big),
$$

computed in fp32, with `M` the channel mask of Section 6. **`ŷ^aux` is the raw estimator output, not multiplied by `α`.**

### 4.4 Gradient-routing variants

- **block:** `ŷ^aux = W_2 SiLU(W_1 sg(u_ℓ))`, a recomputation used only for the loss. The auxiliary gradient reaches only `W_1` and `W_2`.
- **flow:** `ŷ^aux = ŷ_ℓ` from the forward pass. The auxiliary gradient also reaches the backbone through `u_ℓ`. This mirrors NextLat, where losses on detached latent targets shape backbone representations [7].

Both variants are screened for P1 because the evidence conflicts. Shaping the backbone can help [7], while feature-matching constraints have limited gains or hurt at scale in other settings [8, 9].

In both variants, the LM-loss gradient flows normally through `ẑ_ℓ`: into the estimator via `α`, and into the backbone.

## 5. P3 — Context-summary heads

### 5.1 Feature MLP

$$
\phi_\ell(j) = W_{2,\ell}\,\mathrm{SiLU}\big(W_{1,\ell}\,u_\ell(j)\big)\in\mathbb{R}^{255},\qquad
W_1\in\mathbb{R}^{256\times 1024},\; W_2\in\mathbb{R}^{255\times 256},
$$

no biases, default initialization. The output is split into three 85-dimensional parts `φ = [φ⁽¹⁾; φ⁽²⁾; φ⁽³⁾]`, one per timescale.

### 5.2 Exponential moving sums with document reset

The decays are fixed buffers: **`γ = (0.5, 0.9, 0.99)`**, with effective windows `1/(1−γ)` of about 2, 10 and 100 tokens.

The reference semantics, which is also the decode rule, for any input sequence `x` and decay `γ`:

$$
\mathrm{EMS}_\gamma(x)(j) = \gamma\,\mathrm{EMS}_\gamma(x)(j-1) + (1-\gamma)\,x(j),\qquad
\mathrm{EMS}_\gamma(x)(j-1) := 0 \ \text{if } j \text{ starts a document}.
$$

Equivalently, with `s(j)` the first token of `j`'s document,

$$
\mathrm{EMS}_\gamma(x)(j) = \sum_{i=s(j)}^{j} (1-\gamma)\,\gamma^{\,j-i}\,x(i).
$$

This is an exponentially weighted **sum** over all earlier tokens of the document, including token `j`. Its weights total `1 − γ^{j−s(j)+1}`, which is below 1 near the start of a document, so it is not renormalized. The estimator and its target use the same operator, so this affects both identically.

The P3 states are

$$
e^{(c)}_\ell = \mathrm{EMS}_{\gamma_c}\big(\phi^{(c)}_\ell\big),\qquad c = 1,2,3 .
$$

### 5.3 Computing EMS in parallel (exact two-level chunkwise scan)

Use chunk size `C = 64`; `T = 2048` gives 32 chunks per sequence. Write a position as `t = κC + τ` (chunk `κ`, offset `τ ∈ [0, C)`). Let `end_κ = κC + C − 1`, and `doc(·)` the document id. All steps are vectorized over batch and chunks, computed in fp32, and involve no Python loop over positions or chunks.

**Step 1 — intra-chunk sums.** For each chunk, `Y_κ = L_κ X_κ`, where `X_κ ∈ ℝ^{C×n}` is the chunk's input and

$$
(L_\kappa)_{\tau,\sigma} = \begin{cases}(1-\gamma)\,\gamma^{\,\tau-\sigma} & \sigma\le\tau \ \text{and}\ \mathrm{doc}(\kappa C+\sigma)=\mathrm{doc}(\kappa C+\tau),\\ 0 & \text{otherwise.}\end{cases}
$$

**Step 2 — chunk-end states.** With `S_κ = EMS(x)(end_κ)` and `Y^end_κ = Y_κ[C−1]`,

$$
S_\kappa = \sum_{\kappa'\le\kappa} \gamma^{\,C(\kappa-\kappa')}\;\mathbb{1}\big[\mathrm{doc}(\mathrm{end}_\kappa)=\mathrm{doc}(\mathrm{end}_{\kappa'})\big]\;Y^{\text{end}}_{\kappa'} ,
$$

a `32 × 32` masked decay matrix applied to the chunk-end values.

**Step 3 — combine.** For `t = κC + τ`,

$$
\mathrm{EMS}_\gamma(x)(t) = Y_\kappa[\tau] + \gamma^{\,\tau+1}\;\mathbb{1}\big[\mathrm{doc}(t)=\mathrm{doc}(\kappa C-1)\big]\;S_{\kappa-1},
\qquad S_{-1} := 0 .
$$

These three steps are exact. Documents are contiguous, so the indicator in Step 2 is zero precisely when a document starts between the two chunk ends, and the indicator in Step 3 is zero precisely when a document starts within chunk `κ` at or before `t`.

**Further requirements:**
- Build the decay coefficients as `exp(Δ · log γ)` on non-negative offsets `Δ`; **never** use `γ^{-i}` scaling, which overflows.
- The masks depend only on `doc_id` and `γ`. Build them **once per micro-batch** and reuse them in every proxy layer and for the targets.
- **Reference implementation for tests:** the dense matrix `D_{j,i} = (1−γ)γ^{j−i}` for `i ≤ j` and `doc(i) = doc(j)`, else 0, with `EMS = D x`. It must agree with the chunkwise result and with a sequential loop (test T4). Do not use the dense form for training.

### 5.4 Projection and injection

$$
\hat y^{(c)}_\ell(j) = P_{\ell,c}\, e^{(c)}_\ell(j),\qquad P_{\ell,c}\in\mathbb{R}^{1024\times 85}\ \ (\text{no bias}),
$$

$$
\hat z_\ell(j) = u_\ell(j) + \sum_{c=1}^{3}\alpha_{\ell,c}\odot \mathrm{RMSNorm}_0\big(\hat y^{(c)}_\ell(j)\big).
$$

Applying `P` after the moving sum preserves each part's temporal kernel, because `P · EMS(φ) = EMS(P φ)`. The P3 auxiliary loss is therefore well defined at full width.

### 5.5 Target and auxiliary loss

**Deep band.** For proxy layer ℓ, `B_ℓ = [ℓ+2, min(ℓ+6, 28)]`, inclusive (5 blocks for ℓ ≤ 22, 3 blocks for ℓ = 24). The deep-band state is

$$
\bar D_\ell(j) = \frac{1}{|B_\ell|}\sum_{b\in B_\ell}\mathcal{N}^{\text{P3}}_b\big(h_b(j)\big).
$$

Averaging several layers follows data2vec, where targets averaged over the top layers outperformed single-layer targets [5].

**Target**, at full width (1024), using the same operator and masks as the estimator:

$$
m^{(c)}_\ell = \mathrm{EMS}_{\gamma_c}\big(\mathrm{sg}(\bar D_\ell)\big),\qquad c = 1,2,3 .
$$

**Loss**, in fp32:

$$
\mathcal{L}^{(\ell)}_{\text{aux}} = \frac{1}{3}\sum_{c=1}^{3}\mathrm{mean}_j\Big(1-\cos\big(M\odot\hat y^{(c),\text{aux}}_\ell(j),\;m^{(c)}_\ell(j)\big)\Big).
$$

**Routing.**
- **block** (default): recompute `φ` from `sg(u_ℓ)`, then the scan and `P_c`, for the loss path only. The auxiliary gradient reaches only `W_1`, `W_2`, `P_c`.
- **flow** (optional): use `ŷ⁽ᶜ⁾` from the forward pass.

**Memory.** Compute each layer's targets and loss immediately after that layer's band states are available, then free them. One layer's targets occupy about 0.4 GB in fp32 for 16 sequences × 3 timescales.

### 5.6 Decode

Keep, per proxy layer and sequence, three 85-dimensional states. For each new token compute `φ`, update `e⁽ᶜ⁾ ← γ_c e⁽ᶜ⁾ + (1−γ_c) φ⁽ᶜ⁾`, and reset the states to 0 when a new document starts. Proxy K/V enter the KV cache like native K/V.

## 6. Target normalization

Every raw target quantity `x` is normalized before use. For P1 this is each window sum; for P3 it is each block output `h_b`:

$$
\mathcal{N}(x) = \frac{M\odot(x-\mu)}{\sqrt{\mathrm{mean}_{\text{unmasked}}\big((x-\mu)^2\big)+\epsilon}} ,
$$

where the mean in the denominator is over unmasked channels, per token.

**Centering `μ`.** A per-channel running mean of the raw quantity, detached, with one buffer per (target type, layer): one per proxy layer for P1, one per block `b ∈ [4, 28]` for P3. It removes the token-independent component, so the cosine rewards token-specific content.
- **Initialization:** before step 1, run one no-grad forward pass on each rank's first micro-batch, all-reduce the per-channel means across data-parallel ranks, and set `μ` to the result.
- **Updates:** once per **optimizer step**. Accumulate per-channel sums and token counts over all micro-batches of the step, all-reduce across ranks, and update `μ ← 0.99 μ + 0.01 μ_step`.
- **Within a step:** all micro-batches use the `μ` from before the step.
- Flag: `target_centering`, default on.

**Mask `M`.** A binary channel mask that excludes massive-activation channels. Compute it once on a trained checkpoint (the old arm A suffices):
- over about 1M tokens, compute the per-channel mean |value| of every block output `h_b`;
- exclude every channel that exceeds 20× the median per-channel value in any layer;
- apply the union of excluded channels to all targets and, as in Sections 4.3 and 5.5, to all predictions inside the cosine.

If no checkpoint is available, use an empty mask and log this choice.

## 7. Training objective and controls

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda(s)\cdot\frac{1}{|\mathcal{P}|}\sum_{\ell\in\mathcal{P}}\mathcal{L}^{(\ell)}_{\text{aux}} ,
$$

where `s` is the optimizer step.

- **`λ_max = 0.1`.** The schedule is defined in **absolute optimizer steps**: linear from 0 to `λ_max` over the first 250 steps, then constant. A decay phase for longer confirmation runs is a config option, off by default.
- **λ = 0 control:** the proxy modules are present and trained only through the LM loss; skip the auxiliary backward. Compute auxiliary cosines under `no_grad` at the logging interval. **The running buffers `μ` are initialized and updated every optimizer step exactly as in λ > 0 arms**, so logged cosines are comparable across arms.
- **Precision:** model compute in bf16, as in the vanilla arm. Scans, window sums, normalization statistics and cosines in fp32.
- **Optimizer groups:** estimator weights (`W_1`, `W_2`, `P_c`) go in the standard weight-decay group; gates `α` get no weight decay. All use the same LR schedule. Buffers (`μ`, `γ`, `M`) are saved in checkpoints.

## 8. Arms, compute matching, and screening protocol

| Arm | Architecture | Auxiliary loss |
|---|---|---|
| **A** | Vanilla, document isolation | — |
| **V1** | Vanilla, MLP widened to match P1 | — |
| **V3** | Vanilla, MLP widened to match P3 | — |
| **P1-λ0** | P1, `k = 4` | none (λ = 0) |
| **P1-block** | P1, `k = 4` | block routing |
| **P1-flow** | P1, `k = 4` | flow routing |
| **P3-λ0** | P3 | none (λ = 0) |
| **P3-block** | P3 | block routing |

**Screening:** 2,500 optimizer steps and 3 seeds per arm, with identical data order across arms for a given seed. Add seeds if the seed spread of A exceeds the smallest difference of interest.

**Compute matching** counts forward multiply-accumulates (MAC) per token of the *architecture*. Training-only auxiliary computation (targets, block recomputation) is not matched; it is reported as seconds per update.

| Item | Extra MAC/token per proxy layer | × 12 layers |
|---|---:|---:|
| P1 estimator (`1024·256 + 256·1024`) | 0.524 M | 6.29 M |
| P3 estimator (`1024·256 + 256·255 + 3·85·1024`) | 0.589 M | 7.06 M |
| P3 chunkwise scan (`≈ 3 · C · 85`, `C = 64`) | 0.016 M | 0.20 M |
| **P3 total** | 0.605 M | **7.26 M** |

For reference, the vanilla forward pass costs roughly 0.7 G MAC/token:
- about 0.44 G in the projections and MLPs;
- about 0.16 G in the LM head for a ~152k vocabulary;
- at most about 0.12 G in attention scores.

The implementation must recompute these numbers from the actual config.

**V widening.** Widen the MLP intermediate size of **all 28 blocks** by

$$
\Delta = \frac{\text{extra}}{3\cdot d\cdot 28} = \frac{\text{extra}}{86{,}016},
$$

rounded to the nearest multiple of 8: **+72 for V1, +88 for V3**. Report the residual mismatch, which should stay below 0.2% of total MAC. Parameter counts also match closely: P1 adds 6.29 M parameters and V1 6.19 M; P3 adds 7.06 M and V3 7.57 M.

**Throughput reference:** A runs at ≈ 2.425 s per optimizer step on 8×B200 with dense SDPA isolation [3]. Report seconds per update for every arm.

## 9. Integration tests

All tests must pass before any training run. Use fp32 where tolerances are tight, and freeze the running buffers `μ` during tests.

**If any test fails, stop and report the failure with its numbers. Do not modify this specification, loosen tolerances, or start training to work around a failure.**

| ID | Test | Pass criterion |
|---|---|---|
| T1 | Init equivalence: proxy model with all `α = 0` vs vanilla with identical shared weights | Logits equal; fp32 max abs diff ≤ 1e-5 |
| T2 | GQA mapping: set `α` to random non-zero values | Outputs of non-proxy query heads unchanged vs `α = 0`; proxy-mapped heads change |
| T3 | Cross-document leakage: ≥ 2 documents in one sequence; loss (LM + aux) on document 2 only | Gradients w.r.t. document-1 inputs exactly 0; perturbing document-1 inputs leaves all document-2 activations, P3 states and targets unchanged |
| T4 | EMS correctness: chunkwise vs dense reference vs sequential loop | Max relative error ≤ 1e-5 (fp32) in all of the following cases: a document starting at a chunk start; one starting mid-chunk; one spanning several chunks; a length-1 document; several documents within one chunk; a single document filling the sequence |
| T5 | Auxiliary sanity at step 0 | Aux loss finite; `α` has non-zero gradient from the LM loss; block routing gives exactly zero aux gradient on backbone parameters; flow routing gives non-zero |
| T6 | Targets detached | Every target tensor has `requires_grad == False` |
| T7 | λ = 0 control | Aux contributes no gradient; logged cosines computed under `no_grad` |
| T8 | Compute and throughput | Parameter and MAC counts of V1, V3, P1, P3 reported; V mismatch < 0.2%; seconds per update reported |

## 10. Logging and evaluation

**Every logging interval:**
- LM training loss and gradient norms;
- for each proxy layer: auxiliary loss, cosine (per timescale for P3), and mean |α| (per timescale for P3);
- seconds per update.

**Evaluation**, on a fixed held-out set processed identically for all arms:
- LM loss;
- **reliance ablation:** evaluate the same checkpoint with all `α` set to 0 and report the loss increase;
- optionally, attention mass on proxy versus native heads, measured on a small evaluation batch through an eager (non-fused) attention path that returns attention weights, using the same document mask. SDPA itself does not return weights.

**Success criterion:** a λ > 0 proxy arm beats **both** its λ = 0 control **and** its compute-matched V arm by more than the seed-to-seed spread. Arm A is the unwidened reference and the source of the seed-variance estimate. It is not comparable to the earlier non-isolated runs.

## 11. Optional pre-step: offline estimability check for P1

On a trained checkpoint (the old arm A suffices):
1. Collect `u_ℓ` and P1 targets for ℓ ∈ {2, 8, 14, 20} and `k ∈ {1, 2, 4, 8}`, respecting `ℓ + k − 1 ≤ 28`, over about 20M tokens. Normalize targets with `μ` estimated from the collected data, and hold out 10%.
2. Fit three predictors with the masked cosine loss: (a) a constant mean direction, (b) a linear map, (c) an MLP `1024 → 256 → 1024`.
3. Report the held-out cosine of each.

**Decision:** if (c) − (b) < 0.02 and both are close to (a), deprioritize P1. Otherwise use the `k` with the largest (c) − (b) margin.

## 12. Implementation order

1. Document metadata (`doc_id`), the dense document mask, and SDPA attention for all arms; retrain A.
2. Config flags and the proxy K/V path of Section 3 with `α = 0`; pass T1 and T2.
3. P1 estimator, target capture (detached `m_i`), normalization `N`, auxiliary loss, both routings; pass T3 and T5–T7.
4. The EMS operator (chunkwise plus dense reference); pass T4. Then the P3 feature MLP, projection, targets and loss; pass T3 and T5–T7.
5. V1 and V3 widening; pass T8.
6. Screening runs (Section 8).

## 13. Defaults

| Parameter | Default | Options |
|---|---|---|
| Proxy layers | P1: `𝒫_k` = even ℓ, 2 ≤ ℓ ≤ 29 − k (k = 4: {2, …, 24}); P3: {2, …, 24} | — |
| Proxy KV groups `G_p` | 2 of 8 (last groups) | 1, 4 |
| P1 hidden width | 256 | — |
| P1 lookahead `k` | 4 | 1, 2, 8 |
| P3 feature dim | 255 = 3 × 85 | — |
| P3 decays `γ` | (0.5, 0.9, 0.99), fixed | — |
| P3 deep band `B_ℓ` | [ℓ+2, min(ℓ+6, 28)], inclusive | — |
| EMS chunk size `C` | 64 | divisors of T |
| `λ_max` | 0.1 | 0.03, 0.3 |
| λ schedule | 250-step linear warm-up, then constant | decay (off) |
| Routing | P1: block and flow; P3: block | P3 flow |
| Target centering | On; momentum 0.99; updated per optimizer step; all-reduced | Off |
| Mask threshold | 20 × median per-channel mean \|h\| | — |
| Gates `α` | ℝ^d, zero init, no weight decay | — |
| `ε` | 1e-6 | — |
| Attention | SDPA, dense boolean mask `[B,1,T,T]` = causal ∧ same document; `is_causal=False` | — |
| `position_ids` | Reset to 0 at each document start | — |

## 14. Scope and non-goals

- This specification covers only the screening experiment. Confirmation runs (10k steps), λ decay schedules, and the P2 and P4 variants are out of scope.
- Decode-time behavior (Section 5.6) is specified for completeness; screening needs only training and held-out loss.
- Learnable or input-dependent decays for P3 are not part of this revision. With fixed `γ`, the target's temporal kernel stays identical to the estimator's.
- The P3-λ0 arm is itself a hybrid with an EMS-based K/V path. Differences between P3-λ0 and V3 describe that hybrid, not deep anticipation.

## Appendix A. Reference pseudocode

```python
def attn_with_proxy(h_prev, layer, ctx):
    u = layer.input_layernorm(h_prev)                          # [B, T, d]
    q = layer.q_proj(u)                                         # all query heads
    z = layer.proxy.build_z(u, ctx)                             # P1 or P3, [B, T, d]

    n_nat = (8 - G_p) * 128                                     # rows of native KV groups
    Wk, Wv = layer.k_proj.weight, layer.v_proj.weight          # [8*128, d]
    k = torch.cat([F.linear(u, Wk[:n_nat]), F.linear(z, Wk[n_nat:])], dim=-1)
    v = torch.cat([F.linear(u, Wv[:n_nat]), F.linear(z, Wv[n_nat:])], dim=-1)

    q, k, v = to_heads(q, 16), to_heads(k, 8), to_heads(v, 8)   # [B, T, H, 128]
    q, k = layer.q_norm(q), layer.k_norm(k)                     # QK-norm as in base model
    q, k = apply_rope(q, k, ctx.position_ids)                   # position_ids reset per document
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))            # [B, H, T, 128]
    k, v = repeat_kv(k, 2), repeat_kv(v, 2)                      # same GQA handling as vanilla
    o = F.scaled_dot_product_attention(q, k, v,
                                       attn_mask=ctx.attn_mask,  # [B,1,T,T], causal & same doc
                                       is_causal=False)
    return layer.o_proj(o.transpose(1, 2).reshape(B, T, -1))


# P1 build_z
def build_z_p1(u, ctx):
    y_hat = W2(silu(W1(u)))
    ctx.save(layer_idx, u=u, y_hat=y_hat)          # for the aux loss
    return u + alpha * rmsnorm0(y_hat)


# P3 build_z
def build_z_p3(u, ctx):
    phi = W2(silu(W1(u))).view(B, T, 3, 85)
    z = u
    for c in range(3):
        e_c = ems(phi[:, :, c].float(), gamma[c], ctx.doc_masks)   # Section 5.3
        y_c = P[c](e_c.to(u.dtype))
        z = z + alpha[c] * rmsnorm0(y_c)
    ctx.save(layer_idx, u=u)
    return z
```

`ems(x, γ, masks)` implements Section 5.3 (Steps 1–3). `ctx.doc_masks` holds the per-micro-batch intra-chunk masks, chunk-end masks and carry indicators, built once from `doc_id`.

## Sources

[1] `cross_depth_anticipation_research_handoff_v4_results.md` — from-scratch arms A–G and bottleneck variants; single-pass estimators tied or lost to the extra-branch control.

[2] `early_deep_feedback_probes_summary_v2.md` — Probes 1–3; Probe 2 joint-training result (Deep vs Shallow, two seeds).

[3] `/disk/thuat/deep2shallow/docs/CURRENT_TASK.md` — document-isolation benchmark (dense SDPA and FA4 varlen) and correctness checks (8×B200, Qwen3-0.6B, sequence 2,048).

[4] Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP 2021 (background; not re-verified for this revision).

[5] Baevski et al., *data2vec*, 2022 — averaged top-layer targets outperform top-layer targets; FFN outputs make better targets than FFN plus residual.

[6] Sun et al., *Retentive Network*, arXiv 2307.08621 — per-head multi-scale decay; ablations show that both decay and multiple decay rates improve language modelling.

[7] Teoh et al., *Next-Latent Prediction Transformers Learn Compact World Models*, arXiv 2511.05963 — detached latent targets; Smooth L1 plus token-space KL through a frozen head.

[8] Li et al., *EAGLE-3* — the feature-prediction loss acted as a constraint that limited gains from scaling data; it was removed in favor of direct token prediction with multi-layer feature fusion.

[9] Wu and Tu, *Layer-Condensed KV Cache*, ACL 2024 — an MSE loss on KVs helped with small data but hurt with large data.
