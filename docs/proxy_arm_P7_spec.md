# Proxy Arm P7 — Proxy-Memory Attention: Architecture and Implementation Specification

**Revision 2 — 6 October 2026.** This document specifies arm **P7** for the from-scratch Qwen3-0.6B-style screen. Its goal, like that of P4–P6, is to **beat arm A**. It is an implementation contract: an implementation that deviates from Sections 3–7 is a different experiment and must be labelled as such.

Revision 2 changes:
- the relational-loss quantity is renamed so that it cannot be confused with the detached proxy `p`;
- the query-sampling rule is completed;
- T2 uses a tolerance instead of bitwise equality;
- it is stated that proxy-head queries use the native `q_norm` and RoPE;
- the step-1,000 cosine threshold is marked as a heuristic trigger;
- boolean mask semantics, the P7-kq value layout, the parameter count and the KV-cache overhead are stated explicitly.

Shared components are defined in two earlier documents and are referenced, not repeated, except where this document changes them:
- **P1/P3 specification, Revision 7 [1]:** base model, dense-SDPA document isolation, notation, target normalization, integration-test conventions.
- **P4–P6 specification, Revision 3 [2]:** auxiliary-loss schedule and accumulation rule, block routing, estimator isolation, identical backbone initialization, efficiency requirements, and the screen protocol.

## 1. Motivation

All earlier proxy arms add the proxy into an existing representation. The results [2, §1]:
- **Zero-initialized gate (P1, P3):** the proxies were never used.
- **Gate initialized to 1 (P1, keys and values of 2 of 8 KV groups):** the proxy was used and kept (mean |α| ≈ 0.98; disabling it raised loss by 0.043), yet the arm was **worse than A by 0.005**. The estimator also drifted away from its target (mean cosine 0.59 → 0.44), because the LM gradient reached it through the injection path.

**P7 changes how the proxy is used.** In selected heads, each past token contributes **two memory entries** to a single softmax: its native entry, computed from `u`, and a separate **proxy entry**, computed from a prediction of what deeper blocks will add for that token. Consequences:
- **No native capacity is lost**, because native entries remain.
- **Each query decides per token** whether to read the native representation or the predicted deep one. Nothing is forced by addition.
- **The proxy stays a deep-state prediction**, because the estimator is isolated [2, §2.4].

P7 is the single-pass, predicted-KV counterpart of attending to upper-layer KVs, as in Layer-Condensed KV Cache [3] and the cross-layer KV-sharing study [4]. Those approaches require iterative training, because real upper-layer KVs of past tokens are not available in the same parallel pass.

## 2. Shared definitions

- **Base model, attention, notation** [1, §2]. Qwen3-0.6B-style, 28 blocks, `d = 1024`, 16 query heads, 8 KV heads, head dimension 128. Dense SDPA document isolation; `position_ids` reset at each document start. `u_ℓ`, `a_ℓ`, `m_ℓ`, `h_ℓ` as defined there.
- **Proxy layers.** `𝒫 = {2, 4, …, 24}` (1-indexed blocks), i.e. `model.layers[1], [3], …, [23]` in code.
- **Normalization** [1, §6]: per-channel running standardization, floor, clipping, two-pass initialization, statistics frozen during recomputation and evaluation. One buffer pair per proxy layer. `target_version = "p7-r1"`.
- **λ schedule and accumulation** [2, §2.3]: `λ(s) = 0.1 · min(s/250, 1)`. The auxiliary loss is scaled exactly like the LM loss under gradient accumulation and data parallelism. The learning-rate schedule is identical to A.
- **Estimator isolation: on** [2, §2.4]. The estimator receives **only** the auxiliary gradient, and the auxiliary gradient never reaches the backbone (block routing, implementation (a) or (b) of [2, §2.3]).
- **Backbone initialization** is bitwise identical to A, with a separate random generator for new modules [2, §7].
- `RMSNorm_0(x) = x / sqrt(mean(x²) + ε)`, with `ε = 1e-6` and no learned weight.

## 3. Architecture

### 3.1 Proxy groups and heads

In each proxy layer, the **last `G_p = 2` KV groups** (0-based KV heads 6 and 7) are proxy groups. Under the standard GQA mapping (query head `i` uses KV head `i // 2`), these serve **query heads 12–15**. The implementation must verify the mapping its kernel uses (test T0). All other heads are unchanged.

### 3.2 Estimator: causal convolution plus MLP

For each proxy layer ℓ and token `j`:

$$
x_j = W_{\text{in}}\,u_\ell(j)\in\mathbb{R}^{256},\qquad
c_j = \sum_{i=0}^{3} w_i\odot x_{j-i}\cdot\mathbb{1}\big[j-i\ge 0\ \wedge\ \mathrm{doc}(j-i)=\mathrm{doc}(j)\big],
$$

$$
\hat y_\ell(j) = W_{\text{out}}\,\mathrm{SiLU}(c_j),\qquad
p_j = \mathrm{RMSNorm}_0\big(\mathrm{sg}(\hat y_\ell(j))\big).
$$

- `W_in ∈ ℝ^{256×1024}`, `W_out ∈ ℝ^{1024×256}`, and depthwise kernel weights `w_0, …, w_3 ∈ ℝ^{256}`. No biases; default (non-zero) initialization.
- The convolution is causal, so it uses tokens `j−3 … j`. **Taps from a previous document or before the sequence start are zero.** It provides local context cheaply, in the spirit of the short causal convolution in Mamba [6].
- `sg` implements isolation: the LM loss cannot reach the estimator through `p`.

### 3.3 Proxy entries

For each proxy group, the proxy entry of token `j` is

$$
k^{p}_j = \mathrm{RoPE}_j\Big(\mathrm{KNorm}^{p}\big(W^{p}_K\,p_j\big)\Big),\qquad
v^{p}_j = W^{p}_V\,p_j,
$$

- `W^p_K, W^p_V ∈ ℝ^{(G_p·128)×1024}` are **new** projections without biases, separate from the native `k_proj` and `v_proj`, because `p` lives in standardized-target coordinates and not in the coordinates of `u`.
- `KNorm^p` is a **separate** per-head RMSNorm over the head dimension, with learned weight initialized to 1 like the native `k_norm`.
- RoPE uses token `j`'s position, the same position as its native entry.

Native entries of the proxy groups are unchanged: `k_j = RoPE_j(KNorm(W_K u_ℓ(j)))` and `v_j = W_V u_ℓ(j)`, using the native rows for those groups.

### 3.4 Attention of proxy heads

For a proxy query head at token `t`, with visible set `V(t) = {j ≤ t : doc(j) = doc(t)}`:

$$
\mathrm{out}_t = \sum_{j\in V(t)} \pi^{\text{nat}}_{t,j}\, v_j \;+\; \sum_{j\in V(t)} \pi^{p}_{t,j}\, v^{p}_j ,
$$

$$
(\pi^{\text{nat}}_{t,\cdot},\ \pi^{p}_{t,\cdot}) = \mathrm{softmax}\Big(\big[\,q_t^{\top}k_j/\sqrt{128}\,\big]_{j\in V(t)}\ \Vert\ \big[\,q_t^{\top}k^{p}_j/\sqrt{128}\,\big]_{j\in V(t)}\Big).
$$

There is **one softmax over both entry sets**, so native and proxy entries compete directly. Queries are the native queries: `q_t = RoPE_t(QNorm(W_Q u_ℓ(t)))`, with the base model's `q_norm` and RoPE. The outputs pass through the unchanged `o_proj`.

**Implementation.** Use two SDPA calls per proxy layer, then concatenate head outputs in the original head order:
1. **Query heads 0–11:** standard call, native K/V, mask `[B, 1, T, T]`.
2. **Query heads 12–15:** K and V concatenated along the sequence axis, `[native ; proxy]` of length `2T`, with mask `[B, 1, T, 2T] = [M | M]`, where `M` is the standard causal-and-same-document mask.

Use the same SDPA backend and GQA handling as the vanilla arm.

**Mask semantics.** With a boolean SDPA mask, `True` means "may attend". "Masking proxy entries" (`−∞` logits) means setting the **proxy half of the mask to `False`**. Every row keeps at least its own native entry, so no row is fully masked.

**Equivalence.** If all proxy entries are masked, the layer computes exactly the vanilla function (test T1).

## 4. Target and loss

### 4.1 Target

The target is the increment of the next four blocks:

$$
y_\ell(j) = \mathrm{sg}\Big(\mathcal{N}_\ell\big(h_{\ell+3}(j) - h_{\ell-1}(j)\big)\Big).
$$

The subtraction is done in fp32 on detached tensors.
- The increment excludes what is already at layer ℓ − 1, which the native entries provide. Proxy entries are meant to carry what deeper blocks *add*, including their attention outputs.
- The increment includes block ℓ's own attention output, which reads proxy entries. The target therefore depends indirectly on the proxies. Targets are detached, so this creates no gradient loop.

### 4.2 Auxiliary loss

Per proxy layer, all terms are computed in fp32.

**Token-level alignment:**

$$
\mathcal{L}_{\cos} = \mathrm{mean}_j\Big(1-\cos\big(\hat y_\ell(j),\,y_\ell(j)\big)\Big),
$$

averaged over all tokens of the micro-batch.

**Relational alignment.** Attention uses similarities between tokens, so the proxies must preserve the *geometry* of the deep increments, as in relation distillation [5].
- **Eligible queries:** positions that have at least one earlier token in the same document.
- **Sampling:** for each sequence, draw `S = 256` query positions uniformly **without replacement** from the eligible positions. If fewer than 256 are eligible, use all of them. Use a dedicated random generator seeded from the global step (and the sequence index), separate from data and model randomness, so runs are reproducible.
- For each sampled query `i`, define the candidate set `C(i) = {j < i : doc(j) = doc(i)}`. **Exclude `j = i`**: self-similarity would dominate the distribution.
- Let `ȳ = y/‖y‖`, the normalized target, and let `ŷ̄ = ŷ/‖ŷ‖` be computed **from the estimator output `ŷ`, with gradient**. **Do not compute it from the detached proxy `p`**, or the relational loss would not train the estimator. Then

$$
P^{\text{tgt}}_{i,\cdot} = \mathrm{softmax}_{j\in C(i)}\Big(\frac{\bar y_i^{\top}\bar y_j}{\tau_r}\Big),\qquad
P^{\hat y}_{i,\cdot} = \mathrm{softmax}_{j\in C(i)}\Big(\frac{\bar{\hat y}_i^{\top}\bar{\hat y}_j}{\tau_r}\Big),\qquad \tau_r = 0.1,
$$

$$
\mathcal{L}_{\text{rel}} = \mathrm{mean}_{i}\ \mathrm{KL}\big(P^{\text{tgt}}_{i,\cdot}\ \Vert\ P^{\hat y}_{i,\cdot}\big).
$$

**Combined:**

$$
\mathcal{L}^{(\ell)}_{\text{aux}} = \mathcal{L}_{\cos} + 0.5\,\mathcal{L}_{\text{rel}},\qquad
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda(s)\cdot\frac{1}{|\mathcal{P}|}\sum_{\ell\in\mathcal{P}}\mathcal{L}^{(\ell)}_{\text{aux}} .
$$

**Gradient routing.**
- The auxiliary loss trains **only** the estimator (`W_in`, the convolution, `W_out`).
- The LM loss trains everything **except** the estimator: `W^p_K`, `W^p_V`, `KNorm^p` and the backbone.
- With isolation, Adam makes the estimator's update size largely independent of `λ`. The ratio between `L_cos` and `L_rel` is what matters.

## 5. Optional variants

| Variant | Change | When to run |
|---|---|---|
| **P7-kq** | Proxy entries use `k^p` from `p` but **native values** `v_j` (`W^p_V` unused); keys concatenated `[k ; k^p]`, values `[v ; v]` | To separate routing by predicted deep features from reading predicted content |
| **P7-ems** | Estimator adds long context: `c_j ← c_j + EMS_{0.9}(x)(j) + EMS_{0.99}(x)(j)`, with document resets, using the EMS operator of [1, §5.2–5.3] | If mean cosine < 0.3 at step 1,000 |
| **P7-mlp** | Target replaced by the P1 target, the normalized sum of MLP outputs of blocks ℓ…ℓ+3 [1, §4.2] | If mean cosine < 0.3 at step 1,000 |
| **Logit bias** | Learnable scalar `b_ℓ` (init 0, no weight decay) added to all proxy logits, via a float mask `[B, 1, T, 2T]` | Only if the SDPA backend supports gradients through `attn_mask`; otherwise omit |

## 6. Cost

| Component (per proxy layer, `G_p = 2`) | MAC/token |
|---|---:|
| Estimator: `W_in`, `W_out` (convolution ≈ 0.001 M) | 0.525 M |
| `W^p_K`, `W^p_V` (1024 → 256 each) | 0.524 M |
| Attention over the extra proxy entries (4 query heads; ≤ 1,024 visible entries on average) | ≤ 1.05 M |
| **Total** | **≈ 2.1 M** |

For 12 proxy layers this is **≈ 25 M MAC/token**, about 3.5% of the ~0.71 G MAC/token forward pass, roughly four times P4/P6.

**Parameters:** about 1.05 M per proxy layer (`W_in`, `W_out`, `W^p_K`, `W^p_V`, plus about 1k for the convolution and `KNorm^p`), so **≈ 12.6 M added parameters** in total.

**Training-only additions:** the relational loss costs about 0.26 M MAC/token per layer (256 sampled queries × up to 2,048 candidates × 1,024 dimensions, per 2,048-token sequence); the cosine loss and target construction are negligible.

**Wall-clock.** The extra SDPA call uses the dense-mask path with `2T` keys. Given that P1 (0.9% FLOPs) ran about 12% slower than A, P7 may exceed 15% without optimization. Apply [2, §6], measure seconds per update in the integration tests (T10), and report the profile before launching the screen.

## 7. Screen protocol

As in [2, §7]:
- one seed, 2,500 optimizer steps, settings identical to A;
- backbone initialization identical to A;
- comparison against **token-matched A** and **time-matched A**;
- interpretation using the additional seed of A from the P4–P6 screen.

**Success:** P7 beats time-matched A by more than A's seed spread.

**Mid-run check at step 1,000:** report the mean cosine per layer and the attention mass on proxy entries (Section 8). If the mean cosine is below 0.3, queue P7-ems or P7-mlp. The 0.3 threshold is a heuristic trigger, not a validated cut-off; for reference, P1's per-token estimator reached about 0.59 on its easier MLP-sum target.

## 8. Logging and evaluation

**Every logging interval:**
- LM loss, gradient norms, seconds per update;
- per proxy layer: `L_cos`, `L_rel`, mean cosine;
- normalization diagnostics [1, §10];
- `b_ℓ` if the logit bias is enabled.

**Evaluation**, on the fixed held-out set:
- LM loss;
- **proxy-masked ablation:** the same checkpoint with every proxy entry masked (`−∞`);
- **attention mass on proxy entries:** per layer, on a small evaluation batch (for example 8 sequences), through an eager attention path that returns weights. Report the mean over proxy query heads and tokens of `Σ_j π^p_{t,j}`.

Reading the diagnostics together:

| Observation | Meaning |
|---|---|
| Substantial proxy mass, and masking proxies hurts | The model uses the proxy entries |
| Proxy mass near 0, and masking has no effect | The model ignores the proxies; the predictions are not useful to attention |
| Proxies used, but the arm is worse than A | The proxies are used but net harmful, or the overhead is not repaid |

## 9. Integration tests

Conventions as in [1, §9]: fp32 where tolerances are tight, normalization buffers frozen except in normalization tests, and **stop and report** on any failure without modifying this specification.

| ID | Test | Pass criterion |
|---|---|---|
| T0 | Placement and mapping | Modules on `model.layers[1], [3], …, [23]`; proxy groups are KV heads 6–7, serving query heads 12–15 under the kernel's actual GQA mapping |
| T1 | **Proxy-masked equivalence:** mask every proxy entry with `−∞` and compare with vanilla using identical shared weights | Logits equal; fp32 max abs diff ≤ 1e-5 |
| T2 | **Non-proxy heads unaffected, per block:** same input `h_{ℓ−1}` to the P7 block and to the vanilla block | Outputs of query heads 0–11 before `o_proj` equal within fp32 max abs diff ≤ 1e-6 (not necessarily bitwise: they come from a separate SDPA call). Their queries, keys and values are bitwise identical |
| T3 | **Cross-document leakage:** ≥ 2 documents per sequence; loss (LM + `L_cos` + `L_rel`) on document 2 only | Gradients w.r.t. document-1 inputs exactly 0; perturbing document-1 inputs leaves document-2 activations, `p`, targets and relational distributions unchanged |
| T4 | **Mask `[T, 2T]`** on random document layouts | The proxy half equals the native half; proxy entry `j` is visible to query `t` iff `j ≤ t` and both are in the same document |
| T5 | **Convolution causality and resets** | `c_j` is unchanged when `x_{j+1}` is perturbed, and when any token of a previous document is perturbed; first tokens of a document use only in-document taps |
| T6 | **Isolation routing:** backpropagate the LM loss and the auxiliary loss separately at initialization | LM alone: exactly zero gradient on `W_in`, the convolution and `W_out`; non-zero on `W^p_K`, `W^p_V`, `KNorm^p`. Auxiliary alone: exactly zero gradient on all backbone parameters and on `W^p_K`, `W^p_V`, `KNorm^p`; non-zero on the estimator |
| T7 | **Relational loss** | Candidate sets exclude `j = i` and other documents; queries without candidates are excluded; sampled positions are distinct (no replacement), and all eligible positions are used when fewer than 256; with a fixed step the sampled positions are identical across reruns; loss finite; its gradient reaches the estimator (non-zero), confirming `ŷ̄` is computed from `ŷ` and not from the detached `p` |
| T8 | Targets detached; normalization [1, T6, T9] | As in [1] |
| T9 | Backbone initialization [2, T9] | Bitwise identical to A |
| T10 | Compute and throughput | MAC counts reported; seconds/update vs A, with the profile breakdown of [2, §6] |

## 10. Defaults

| Parameter | Value |
|---|---|
| Proxy layers | {2, 4, …, 24} (code: `layers[1], [3], …, [23]`) |
| Proxy groups `G_p` | 2 (KV heads 6–7; query heads 12–15) |
| Attention | One softmax over native and proxy entries; same RoPE positions, causal and document masks |
| Proxy projections | New `W^p_K`, `W^p_V` (1024 → 256), separate `KNorm^p` (init 1) |
| Estimator | `W_in` 1024→256, depthwise causal convolution (kernel 4, document-masked), SiLU, `W_out` 256→1024; isolated |
| Target | `N(h_{ℓ+3} − h_{ℓ−1})`, per-channel running standardization [1, §6] |
| Loss | `L_cos + 0.5 · L_rel`; `τ_r = 0.1`; 256 sampled queries per sequence; candidates `j < i` in the same document |
| λ | `0.1 · min(s/250, 1)` |
| Routing | Block; implementation (b) preferred [2, §2.3] |
| Backbone initialization | Bitwise identical to A |
| Extra MAC/token | ≈ 25 M (≈ 3.5%) |
| `target_version` | `p7-r1` |
| Optional | P7-kq, P7-ems, P7-mlp, logit bias `b_ℓ` |

## 11. Scope

- This document covers only P7 and its optional variants. P1/P3 are defined by [1] and P4–P6 by [2].
- Controls (λ = 0, compute-matched vanilla) are deferred until an arm beats A, as for P4–P6.
- Decode: per token and proxy layer, compute `p_t` (estimator with a 3-token convolution state), append the proxy entry to a second KV cache for the proxy groups, and attend over both caches. The second cache doubles KV memory for 2 of 8 groups in 12 of 28 layers, i.e. about +10.7% total KV-cache memory.

## Sources

[1] `proxy_heads_P1_P3_spec_v3.md`, Revision 7 — base model, document isolation, notation, EMS operator, target normalization (Section 6), logging and test conventions.

[2] `proxy_arms_P4_P5_P6_spec.md`, Revision 3 — screen results for P1/P3 and P1 with α initialized to 1; auxiliary-loss schedule and accumulation rule; block routing; estimator isolation; identical backbone initialization; efficiency requirements; screen protocol.

[3] Wu and Tu, *Layer-Condensed KV Cache for Efficient Inference of Large Language Models*, ACL 2024 — queries of all layers attend to top-layer KVs; trained with iterative parallel passes.

[4] Wu, Wu and Tu, *A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference*, NAACL 2025 — configurations in which queries attend to upper-layer KVs perform well but need additional training cost.

[5] Wang et al., *MiniLM* (NeurIPS 2020) and *MiniLMv2* (Findings of ACL 2021) — distillation of self-attention distributions and relations.

[6] Gu and Dao, *Mamba*, 2023 — short causal depthwise convolution before the sequence-mixing operator.
