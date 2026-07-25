# Context-Routed Compositional Embeddings for Generative LMs — Design Document

This document specifies a new research direction: replacing the input embedding
layer of a decoder-only LM (Qwen3-0.6B, from scratch) with a **context-conditioned
sparse composition over a shared anchor codebook**. It captures the method, its
novelty positioning, the lessons carried from the ADE reimplementation, and the
experiment plan.

> **How to read this, depending on why you're here.**
> - **Implementing / running the experiments:** §1 (orientation) → **§3, §7, §10** →
>   `compositional_lm_pseudocode.md` (the code spec). Start at §10, which is the build order.
> - **Writing the paper / defending it:** add **§2** (positioning), **§5** (the collapse
>   argument), **§8** (fair-comparison protocol), **§9** (risks). These are argumentation, not
>   instructions — an implementer can skip them.
> - **§4 and §6** are rationale. Two operational bits hide there: keep `θ@A` **dense**
>   (§4 — a top-k gather measured slower), and Original-ANT needs a **per-coordinate**
>   proximal (§6.1), not a constant threshold.

---

## 1. Core Idea

Standard LMs use a flat lookup table `E ∈ ℝ^{N×d}` (one independent vector per
token). We instead build each token's embedding as a **sparse weighted sum over
a small shared codebook of `K` anchor vectors** `A ∈ ℝ^{K×d}`, where the
selection of anchors is **conditioned on context**:

```
e_i = θ_i · A ,   θ_i sparse ,   θ_i = f(token_i, context_i)
```

- **Shared anchors** `A`: `K ≪ N` reusable semantic atoms.
- **Sparse selection** `θ_i`: each token uses only a few anchors.
- **Context-conditioned**: the same token activates *different* anchors in
  different sentences ("river bank" vs. "bank account") — resolved at the input,
  before the main transformer.

The bet: forcing tokens to be sparse, context-dependent compositions of shared
atoms is a useful inductive bias (parameter sharing across the vocabulary,
polysemy handled at entry, graceful vocabulary scaling), pretrained end-to-end
with next-token prediction.

---

## 2. Why This Is Novel (positioning against prior work)

Every ingredient exists in isolation; the **specific conjunction is unclaimed**:

| Ingredient | Prior work | What they lack |
|---|---|---|
| Sparse anchor composition `E=T·A` | **Anchor & Transform (ICLR 2021)** | static `T` (no context), LSTM not decoder-only |
| Anchors + attention reweighting | **ADE (2026)** | encoder + distilled, classification only; lists generative as open future work |
| Compositional embedding in decoder-only LM from scratch | **MultiHashFormer (2026)** | *fixed random hashing*, not learned semantic anchors |
| Shared cross-lingual codebook | **MUL/VQ-CA (AAAI 2024)** | hard 1-of-K symbols, not sparse graded composition |
| Context selects among senses | multi-sense embeddings (2010–2016) | per-word, not shared, not end-to-end |
| Soft mixture avoids routing collapse | **Soft-MoE (ICLR 2024)** | experts are MLPs on hidden states, dense softmax (no sparsity) |

**The empty cell = our contribution:** a *context-conditioned, differentiable
router* that emits a *sparse* weighting over a *shared* anchor codebook to
*construct the input embedding per occurrence*, trained end-to-end in a
from-scratch decoder-only LM.

### Must-beat competitors
- **MultiHashFormer (arXiv:2606.28057)** — closest *setting* (decoder-only, from
  scratch, 100M/1B/3B, beats standard transformers, constant footprint under
  multilingual vocab expansion). Differentiator: **learned semantic anchors +
  context routing** vs. their fixed hashing. Aim for a head-to-head at matched scale.
- **ADE (arXiv:2604.24940)** — closest *mechanism* (anchors + attention). We do
  generative, from-scratch, no teacher — solving their stated open problem.
- **Original ANT (Liang et al. 2021, free `T` + YOGI/proximal)** — the closest prior method
  to our static rung; **must-try baseline** (§3 rung 1 box). Proxy evidence: ours reaches
  **nnz≈5.8 at ppl≈141 with 10M params**, while YOGI-ANT **cannot go below nnz≈52** and needs
  **155.8M** — **~9× sparser, 15× smaller**. (No matched-sparsity ppl comparison exists: YOGI-ANT
  never reaches nnz≈5, so there is no operating point to compare at.) Confirm at scale.
- **Standard Qwen3-0.6B** (standard embedding, from scratch) — the baseline;
  matched per the §8 protocol (identical backbone, honest param+FLOP deltas, frontier).

### Must-cite to distinguish
Soft-MoE (routing math), Product-Key Memory (context-sparse memory read — but at
FFN, not input construction), A&T (base method), Shu & Nakayama compositional
codes (static codes), Adaptive Input (frequency low-rank), Byte Latent
Transformer (the "eliminate the vocabulary" school).

### The objection we must answer up front
*"Self-attention already makes representations context-dependent (ADE's SAT does
this after a static lookup) — why route at the input?"*
Answer with the **decisive experiment**: static selection vs. context-routed
selection, dictionary and downstream stack held fixed, showing the **same token
provably activates different anchors across contexts** (an experiment ADE never
ran). Theoretical backing: Arora et al. (TACL 2018) — word senses live in linear
superposition recoverable by sparse coding over ~2000 atoms, k≈5.

---

## 3. Method — the composition ladder (implementation spec)

> **Implement from [`compositional_lm_pseudocode.md`](compositional_lm_pseudocode.md)**,
> not from this prose. That file is the authoritative, shape-annotated reference (mirrors
> the verified test harness); this section gives the rationale behind it. If the two ever
> disagree, the pseudocode wins.

Every rung is a standalone embedding module: it takes `token_ids ∈ ℤ^{B×L}` (plus the
doc-mask) and returns `(e ∈ ℝ^{B×L×d}, θ)`, which the harness feeds to the standard Qwen3
backbone as `inputs_embeds`. (It is **not** installed as `embed_tokens` — that signature
could carry neither θ nor the mask; see pseudocode §7d.) Only this module changes across
arms; the backbone and (input-only setting) the dense output head are identical.

### 3.1 Shared components (used by rungs 1–4)
- **Codebook** `A ∈ ℝ^{K×d}` (K≈4096) — the learned shared anchor vectors.
- **Base token table** `X ∈ ℝ^{N×d_x}` (d_x≈128–256) — a small learned per-token
  vector. This is the **only per-token parameter** in the compositional rungs and
  the source of token identity for the router. (`N·d_x ≪ N·d` → the input side is
  smaller than the standard table.)
- **Selection score** — given a per-token **source** vector `u_i` (what differs
  across rungs), score all K anchors (with **QK-norm** on query & key) and sparsify:
  ```
  q_i = RMSNorm(u_i · W_q) ,  k_j = RMSNorm(A_j · W_k)   # QK-norm, exactly as Qwen3 attention
  s_i = γ · (q_i · kᵀ) / √d_k         ∈ ℝ^K   # W_q: d_src×d_k , W_k: d×d_k  (shared across tokens)
  θ_i = α-entmax(s_i)                 ∈ ℝ^K   # mostly-zero weights, Σ_j θ_ij = 1
  ```
  **Init = Qwen3's** (`N(0, 0.02)` on every weight/table, biases 0). ⚠️ **QK-norm is
  required with that init:** the raw score is a 4-fold product of 0.02-scale matrices → std
  ≈ 5.9e-5 → θ uniform → γ would need ≈30 000 to sparsify (verified). RMSNorm on q,k rescales
  them to unit RMS → scores O(1), and θ is non-uniform *from init*.

  **γ = 1 (default) — a plain constant, not a Parameter; at 1 it is a no-op.**
  ⚠️ **γ is a sparsity↔quality knob, not a free win.** Converged nnz / held-out ppl / dead-rate
  (proxy, 1.2–1.5k steps): γ=3 → 1.1/396/9.0% · γ=1.5 → 2.2/359/3.4% · **γ=1 → 4.6–5.5/337/0.8%**
  · γ=0.75 → 10.1/332/0.1% · γ=0.5 → 19/325/0.0% · γ=0.25 → 64.6/319/0.0%. **ppl and dead-rate
  improve monotonically as γ falls**, with no turning point down to 0.25 — so γ=1 is *not* the
  ppl optimum (an earlier sweep tested only {1, 1.5, 3}, making γ=1 its boundary).
  **Why γ=1 regardless:** it is the operating point that keeps selection **graded and sparse
  (~5)** — the regime this paper is about, what the Arora k≈5 framing (§2) assumes, and what
  V0/V1's `max_k=16` requires. γ=0.5 gives nnz≈19 and γ=0.25 nnz≈65 (6% dense), forfeiting the
  sparse-composition thesis and truncating V0/V1. **Cost, stated honestly: ≈+3.6% ppl vs γ=0.5.**
  Re-confirm the calibration at real scale (K=4096, full vocab, 30B tokens).
  **Monitoring:** log the **converged** avg-nnz, never the init value — the two move in
  *opposite* directions with γ, so any "5–10" figure quoted anywhere is an init reading. Expect
  ≈4.6–5.5; healthy is graded selection in the **~3–10 band, stable over training**; do not read
  "<5" as a shortfall. Only two alarms: **collapse toward 1** (the observed failure mode →
  *lower* γ) and **drift toward K** (→ raise γ slightly).
  ⚠️ **Three things never to do:** (a) a **per-token entropy penalty** — it sharpens θ toward
  one-hot, *accelerating* the collapse, and the bonus direction is unstable; (b) an **L1 penalty**
  — entmax output lies on the simplex, so `‖θ‖₁≡1` and its gradient is **exactly 0** (verified);
  (c) a **learnable or scheduled γ** — a free γ drifts *down*, since a denser θ is a weakly more
  expressive embedding. (The *batch-usage* entropy in §7.3 is a different quantity and **is**
  permitted.) entmax's α is the other available sparsity knob.
- **Sparse operator** `α-entmax`, α=1.5, over the K anchors (last dim). Exact
  zeros, variable cardinality, differentiable **on its support** — nonzero gradient
  reaches only near-boundary anchors, **not** a revival guarantee (a fully-dead anchor
  gets ≈0 gradient; see §5, do not overclaim this). Formula:
  `entmax_α(s)=argmax_{p∈Δ} p·s+H_α(p)` → `p_j=[(α−1)s_j−τ]_+^{1/(α−1)}`, τ chosen
  so `Σ_j p_j = 1`. Use the `entmax` PyTorch package. (Gumbel-softmax / hard
  top-k are ablations.)
- **Multi-head selection (`H` heads — UNCONFIRMED (confounded proxy, see below), `H=1` is the
  default and the only implemented path).** Instead of one entmax over all `K` anchors, split the codebook into `H`
  disjoint sub-codebooks of `K/H` and run `H` routers in parallel, each with its own
  `W_q^h, W_k^h`, then **average** the per-head results (`1/H`, not a bare sum — a bare sum
  would scale `‖e‖` by `H`; see the pseudocode call-site warning):
  ```
  A viewed as (H, K/H, d) ;  θ_h = α-entmax(γ·(RMSNorm(u W_q^h)·RMSNorm(A_h W_k^h)ᵀ)/√d_k)
  e = (1/H) · Σ_h  θ_h @ A_h
  ```
  **Cost at `H=4`: +221K params (~0.9% of the embedding) and +<1% of embedding FLOPs** — the
  codebook is untouched at `K×d`; only the router projections are ×H. The full accounting, and
  the two traps in it (count **both** `W_q` and `W_k` — they differ 8× in size; the **query**
  projection is the one non-neutral FLOP term), is in pseudocode §0.1. Near-enough
  capacity-neutral to need no special §8 defense — provided the real numbers are reported.
  **The hypothesis** (not yet established — see the confound below) is that the token composes
  along `H` *independent* axes instead of one flat simplex, and — for V2 — context gets `H`
  independent knobs instead of one. `H=1` recovers the single-head selection as a *functional
  form* (a freshly-initialized `H=1` module has its own weights — pseudocode invariant 6).
  **Proxy evidence — ⚠️ CONFOUNDED, do not rely on it yet.** `probe9.py` (2.4k steps, K=1024,
  wikitext-2 held-out) gave H=4 **−4% ppl** vs H=1 at nnz≈15 (≈3.7/head), 0% dead. But **both
  arms ran at the same γ=1**, and each head owns its own entmax simplex, so H=4 mechanically
  carries ~4× the support (**nnz 6.8 → 15**). The arms therefore differ in *two* variables —
  head structure **and** anchors-per-token — and the γ sweep above shows that moving nnz
  5.5→19 **by lowering γ alone, at H=1**, already buys **−3.6%** with 0% dead. So the current
  evidence **cannot separate "H independent axes" from "more anchors per token."**
  **Required control before any claim:** `H=1` with γ tuned to match H=4's nnz (≈15), same
  steps/seed/data. If H=4 still wins at matched nnz, the structural claim stands and justifies
  the +221K params; if they tie, prefer **lower γ** — it is genuinely free (no extra params,
  no extra FLOPs, and with dense `θ@A` nnz changes neither) and multi-head should be dropped.
  Grounded in product-key memory / product-quantization / RVQ. ⚠️ Build it to *run the matched-nnz control*, not because
  it is established.
- **Causality (critical — this is a decoder-only LM):** `e_i` must depend only on
  tokens `≤ i`. Every context mechanism below (LocalEnc, anchor SAT) is **causal
  at the token level**: position i may use tokens ≤ i, never > i. (ADE's SAT was
  bidirectional because it was an encoder; here it must be masked causally.)

### 3.2 The rungs (each adds ONE mechanism; V2 is the contribution)

**(0) Standard** — ordinary Qwen3 embedding.
- Params: `E ∈ ℝ^{N×d}`.  Forward: `e = E[token_ids]` → (B,L,d).  Reference arm.

**(1) ANT (ours) — static compositional, entmax-router (context-free selection).**
- Source `u_i = X[w_i]` — the token's own base vector, **no context**.
- Forward: `x = X[token_ids]` (B,L,d_x) → `s = γ·RMSNorm(x·W_q)·RMSNorm(A·W_k)ᵀ/√d_k`
  (B,L,K) — the §3.1 **QK-normed** score with `u=x` → `θ = α-entmax(s)` (B,L,K) →
  `e = θ @ A` (B,L,d).
- Causal by construction (depends only on token i). **Question:** does building
  the embedding from a shared codebook (vs a free table) preserve perplexity?
- **This is OUR reformulation of ANT** — a *learned router* (`X`→score→entmax) instead of a
  free per-token weight matrix, with sparsity from **entmax** instead of a proximal operator.

**Baseline — Original ANT (Liang et al. 2021) — MUST TRY (the prior work we improve on).**
The literal published method, to compare head-to-head against rung (1):
- `e = T[token] @ A`, where **`T ∈ ℝ^{N×K}` is a FREE, non-negative per-token weight matrix**
  (each token's weights learned *directly*, not routed), made sparse by an **L1 proximal**.
- Optimizer: **YOGI + per-coordinate proximal on `T`** (`threshold = λ·step_size/denom`,
  soft-threshold + clamp≥0), **AdamW on the backbone**; λ warmup-then-ramp.
  **Code: pseudocode §7c** (full `Yogi` + `OriginalANT` + the λ ramp).
- **Proxy result (K=1024, 1.2k steps, same backbone/seed) — our ANT Pareto-dominates it:**
  ours nnz≈**5.8**, ppl≈**141**, **10M** params; YOGI-ANT floors at **nnz≈52** (the adaptive
  proximal protects frequent anchors — can't reach ~5), best ppl≈149 only when *dense* (nnz 241),
  and needs **155.8M** params (free `T`). So ours is **~9× sparser (52/5.8 = 8.97), 15× smaller
  (155.8/10), and needs no λ tuning.** ⚠️ **Do not claim a ppl multiple.** The only ppl pair
  above is 141 vs 149 (≈5%), and since YOGI-ANT never reaches nnz≈5 there is **no
  matched-sparsity point** at which to compare — state the result as sparsity + params, which
  is already strong. (To get a real matched comparison, run ours *at nnz≈52* and report both.)
  Confirm at real scale — this is a standalone contribution *independent of V2*.

**(2) V0 — static selection + causal anchor-level SAT (ADE-style).**
Selection is the *static* θ_i from rung (1); the added mechanism is that the
selected anchor **vectors** are contextualized by attention before aggregation.
1. Static select (rung 1) → keep the top-`max_k` active anchors/token:
   `idx ∈ ℤ^{B×L×max_k}` (anchor indices), `β ∈ ℝ^{B×L×max_k}` (their θ weights,
   0-padded). **`max_k` must be ≥ the entmax support** (# nonzero θ) of every token,
   else the smallest active anchors are silently truncated: `Σβ<1`, part of the
   embedding is dropped, and (with a normalize-to-1 aggregation) a magnitude confound
   is introduced. **Log the per-batch truncation rate** `mean(support > max_k)`; if it
   is non-trivial, raise `max_k` (preferred) or sparsify harder via entmax α / a slightly
   higher γ — **not** an entropy penalty (§3.1: it accelerates the nnz→1 collapse).
2. Gather vectors: `a = A[idx]` → (B,L,max_k,d); reshape to anchor sequence
   `(B, L·max_k, d)`.
3. **GPE:** add a positional encoding indexed by **token position** — all `max_k`
   anchors of token i share position i. Use a **sinusoidal / relative** encoding,
   **not a learned absolute table** (a learned `L×d` GPE caps context length and breaks the
   length-generalization the backbone's RoPE gives). ⚠️ **Scale it by a learnable `gpe_scale`
   (init ≈ 0.02):** a raw sinusoid (norm ≈ 22) swamps the Qwen-0.02-init anchor vectors
   (norm ≈ 0.66, 34×) → the SAT would be position-dominated at init; the scale makes GPE
   comparable to the anchor magnitude and lets the model tune positional strength.
4. **Causal anchor SAT:** **one attention-only layer** (Q/K/V/O projections, no
   FFN — as in ADE) over the `L·max_k` anchor sequence; mask so an anchor of token
   i attends only to anchors of tokens **≤ i** (plus a padding mask over unused
   `max_k` slots). Output `ã` → (B,L,max_k,d). Cost **`O((L·max_k)²)`**. ⚠️ **QK-norm
   (RMSNorm on Q,K):** at 0.02 init the raw Q·K logits are ~2e-4 → *perfectly uniform,
   content-blind* attention (verified) → the SAT does nothing at init and the V0/V1 ablation
   is confounded; QK-norm makes logits O(1) (same fix as the router, §3.1).
5. Aggregate with the **static** β:
   - **β post-SAT (default):** `e_i = Σ_m β_{i,m} · ã_{i,m}` (padding `β=0` drops out).
   - **β pre-SAT (variant):** scale before step 4 (`a ← β·a`), then
     `e_i = Σ_{m∈real} ã_{i,m}` — **sum over real slots only** (padding slots have
     `β=0` but the SAT still gives them nonzero `ã`, so a plain sum would leak them).
     ⚠️ The SAT's QK-norm normalizes β out of the attention Q/K (per-slot RMSNorm), so
     β-pre affects only the SAT **values**, not the scores — still valid (β weights value
     contributions; verified it still influences the output), just narrower than pre-QK-norm.
- **Question:** does anchor-level context help, and is it worth `O((L·max_k)²)`?

**(3) V1 — V0 + context-dependent aggregation weight.**
Steps 1–4 identical to V0; step 5 **modulates** the static β with a **context weight
α** by attention-pooling token i's own contextualized anchors (intra-token,
`O(L·max_k)`). Compute an unnormalized context score per slot:
- **Variant A (shared query / CLS):** one learned `q_cls ∈ ℝ^d`; `z_{i,m}=q_cls·ã_{i,m}`.
- **Variant B (content query):** `q_i=mean_{m∈real} ã_{i,m}` (mean over **real slots
  only**); `z_{i,m}=q_i·ã_{i,m}`.

Then form the aggregation weight by **combining** β and α (do **not** replace β), and
rescale so the per-token total weight equals V0's `Σβ` (**not** 1) — this makes V1
reduce to V0 *exactly* when α is uniform, so the ablation is confound-free:
```
z_{i,m} ← z_{i,m}.masked_fill(padding, NEG)         # NEG=finfo(dtype).min; NOT −1e9 (fp16-unsafe)
α_{i,m} = softmax_m(z_{i,m})                        # context weight (Σ_m α = 1 over real)
w_{i,m} = β_{i,m} · α_{i,m}                         # β keeps the selector in the graph
w_{i,m} ← w_{i,m} · Σ_m β_{i,m} / (Σ_m w_{i,m} + ε) # preserve Σβ (NOT normalize to 1)
e_i     = Σ_m w_{i,m} · ã_{i,m}
```
- **Why combine, not replace (critical — a pure-α V1 does not train):** the θ *values*
  β are the **only differentiable path** from the loss back to the selection head
  (`X`, `W_q`, `W_k`, γ) — the top-`max_k` *indices* are non-differentiable. If step 5
  aggregates by α alone, β never appears in the forward, so **the selector receives no
  task gradient and is frozen at init** (verified empirically: `∂loss/∂X = None`).
  Keeping the factor `β·α` restores the selector gradient while α adds the context
  modulation. This is the *only* change from a naïve V1.
- **Why preserve Σβ, not normalize to 1 (clean ladder):** under **uniform α**
  (`α_{i,m}=1/n_real`), the preserve-Σβ rescale gives `w=β` exactly, so **V1 collapses
  to V0** and any measured V1−V0 gap is attributable *only* to context, not to a
  magnitude change. Normalizing `w` to sum 1 instead would rescale every token whose
  `Σβ<1` (i.e. whenever `max_k` truncates some active anchors — see the `max_k` note in V0
  step 1, above),
  injecting a magnitude confound into the residual stream (RMSNorm does **not** remove
  it: the raw embedding rides the residual un-normalized). Verified: preserve-Σβ ⇒
  `V1(uniform α)==V0` exactly; sum-to-1 ⇒ equal only when `Σβ=1`.
- **Padding mask:** mask padding slots (`β=0`) in the α-softmax so they don't steal
  probability mass from real slots. α-entmax guarantees ≥1 active anchor/token, so no
  α-pool row is ever all-padding and literal −∞ is technically safe — but use
  **`NEG = torch.finfo(dtype).min`** as a cheap defensive choice (an all-padding row would
  make softmax `0/0 = NaN`; `NEG` degrades to a uniform-over-padding weight instead, which
  β·α then zeros anyway). ⚠️ **Not a literal `−1e9`:** it raises an overflow error on fp16
  tensors (verified), crashing low-precision training; `finfo.min` is dtype-correct. (β·α
  already zeros padding in the *final* weight since β=0 there — the mask only keeps the
  softmax normalization clean.)
- **Question:** does context-weighting the aggregation add anything over V0?

**(4) V2 — context-conditioned SELECTION (the contribution).**
The selection **source** becomes context-mixed, so *which anchors are selected*
depends on neighbors — and no anchor-level SAT is needed.
```
c_i = LocalEnc(X[w_{≤i}])          # CAUSAL context mixer over TOKENS → (B,L,d_c)
s_i = §3.1 QK-normed score with u=c_i  # γ·(RMSNorm(c_iW_q)·RMSNorm(AW_k)ᵀ)/√d_k → (B,L,K)
θ_i = α-entmax(s_i)                # (B,L,K)
e_i = θ_i @ A                      # (B,L,d)
```
Only the **LocalEnc** (how `c_i` is built) changes across V2 variants; everything
downstream (`s_i, θ_i, e_i`) is identical. Fix the two secondary knobs the same for all
LocalEnc variants so the comparison is single-axis: **score head = W_q/W_k**, **init =
residual zero-init** (below). Cost of the shared parts: router `O(L·K·d)` (each token
cross-attends the fixed K-codebook; see §4). **No `O((L·max_k)²)`** (contrast V0/V1).

**LocalEnc variants — run in this ORDER (decide with attn, then optimize with conv):**

| Variant | LocalEnc | Window | Cost | Role |
|---|---|---|---|---|
| **V2-attn** *(primary)* | **1 causal self-attention layer** over tokens | full history | `O(L²·d_c)` | **decisive arm:** strongest context available, so a null result cannot be blamed on the encoder |
| **V2-conv** *(efficiency fallback)* | stack of **2–3 dilated causal Conv1d** (dilations 1,2,4, k=3, channel-mixing) | ~15 tok | `O(L·d_c²·k)` **linear** | run *only if V2-attn wins*: does a linear-cost encoder recover the gain? |
| **V2-conv-lite** | **single** kernel-3 causal Conv1d | ~3 tok | `O(L·d_c²·k)` linear | context-depth floor (how little context suffices?) |

- **Why V2-attn is primary (reversed from an earlier draft — decide first, optimize
  second).** A proxy probe (2.4k steps, K=1024, 4-head selection, wikitext-2 held-out)
  showed the *encoder was never the bottleneck*: upgrading the LocalEnc from the ~15-token
  dilated conv to full causal attention still left context worth only **~1%** ppl over an
  identical **context-free** arm. (The same probe's −4% for multi-head is **confounded with
  nnz** — see §3.1 — so treat only the ~1% context figure as informative here; the point
  stands either way: context stayed small *with the strongest encoder*.) Sparsity was healthy
  throughout (nnz≈15, 0% dead), so the flat result is
  **not** collapse and **not** a weak window. Conclusion: give V2 the *strongest* context
  when deciding whether context-routed selection works at all — if it fails even with full
  attention, it fails regardless of encoder. Cost is not the obstacle: the LocalEnc attends
  over **`L` tokens at `d_c≈d_x`**, *not* over anchors — there is **no `O((L·max_k)²)`**
  term (contrast V0/V1), and it is a small fraction of the backbone.
- **Then fall back for efficiency.** Only *after* V2-attn beats ANT, run V2-conv to check
  whether a **linear-in-L** encoder recovers most of the gain. That restores the "cheap by
  construction" selling point and denies the "you just prepended a transformer layer"
  objection — but as a *second* result, not a precondition. **Regular (channel-mixing)
  conv, not depthwise** — channel mixing is what lets a neighbor change *which* anchor is
  selected.
- **Document boundaries (now mandatory, since V2-attn is primary):** LocalEnc is causal but
  not document-aware. In packed pretraining it reads across pack boundaries — the conv
  variants leak ≤ (receptive field−1) tokens (benign), but **V2-attn leaks the entire
  previous document**. If the backbone uses a document/reset attention mask, you **must**
  apply the **same** mask inside V2-attn's LocalEnc. Keep this identical across all arms so
  it cannot explain any gap.
- **V2-attn carries the full §8 burden:** because the primary arm now adds an attention
  layer, the param+FLOP accounting and the **isolation control** (§8) are load-bearing, not
  optional — they are what separate "context drives *selection*" from "you added capacity."

**Score head (secondary knob — how `s_i` is formed; default W_q/W_k for all above). All keep
QK-norm / input-norm so scores are O(1) at Qwen init (§3.1):**
  - **W_q/W_k (default):** as in the box (QK-norm on q,k; key derived from A, flexible).
  - **A-only:** `s_i = γ·RMSNorm(c_i)·RMSNorm(A)ᵀ/√d` (key = A directly; needs `d_c = d`; drops
    W_q/W_k — **not** combinable with residual zero-init's `d_c = d_x` unless `d_x = d`; standalone).
  - **W_router:** `s_i = γ·RMSNorm(c_i)·W_router`, `W_router ∈ ℝ^{d_c×K}` (free, not tied to A;
    norm the source so γ stays sane).

**Init — residual zero-init (recommended default on every V2 variant):** make LocalEnc a
**residual with zero-initialized output** so `c_i = X[w_i] + Δ(X[w_{≤i}])`, `Δ ≡ 0` at
init (requires `d_c = d_x`; zero-init the last conv / the attn output projection). At
init `c_i = X[w_i]`, so selection is **context-free — the same form as ANT**; training
then grows the context correction `Δ`. Any gain as `Δ` grows is provably from context →
the cleanest static-vs-routed ablation. (This is the mechanism formerly labelled "V2b".)

**(5) V3 — layered routing (deferred).** Re-run the V2 selection at 2–3 transformer
depths using the hidden state as source (`c ← h^ℓ`), adding `θ^ℓ @ A` as a residual
to `h^ℓ`. Add **only** if the de-risk shows V2's input-time selection is too
shallow; note it drifts toward memory-augmentation.

---

## 4. Efficiency (why V2 is cheap by construction)

The `O((L·max_k)²)` cost comes *only* from anchor↔anchor self-attention over the
flattened `L·max_k` sequence (V0/V1/ADE). V2 never builds it:

| Operation | Attends | Cost | Quadratic in L? |
|---|---|---|---|
| V2 router (score + `θ@A`) | token context → fixed K-codebook | `O(L·K·d)` | No |
| V1 intra-token pool | one pooling query → a token's max_k anchors | `O(L·max_k)` | No |
| main transformer | token ↔ token (after aggregation) | `O(L²)` | same as baseline |
| V0/V1 anchor SAT | anchors ↔ anchors across tokens | `O((L·max_k)²)` | **Yes** ❌ |

The V2 router is `O(L·K·d)` (scoring `L·K·d_k` + dense `θ@A` `L·K·d`), **linear in L** and small
in absolute terms — measured at **<1% of the 28-layer backbone's FLOPs** at 0.6B (measured;
the full V2 embed fwd+bwd was ≈23 ms at B=8,L=1024, of which `θ@A`≈9 ms). Dense `θ@A` is the
*right* implementation — a sparse top-k gather was measured **slower and heavier** (materializes
`(B,L,max_k,d)`, bandwidth-bound), so do not "optimize" it into a gather.

Key insight: **cross-attention to a fixed codebook (linear) replaces
self-attention over a long sequence (quadratic).** V2 gets context-dependence by
contextualizing at the *token* level (LocalEnc) and carrying it into the
embedding via *selection* — `max_k²` cheaper than V0/V1's anchor SAT. Note the LocalEnc
attends over **`L` tokens**, never over anchors: the primary **V2-attn** costs
`O(L²·d_c)` at the small `d_c≈d_x`, and the **V2-conv** fallback is **linear** `O(L·d_c²·k)`
— either way there is **no `O((L·max_k)²)`** term. If K is huge,
product-key routing reduces the router to `O(L·√K)`.

---

## 5. The Collapse Question (state it precisely — do NOT overclaim)

Claim: anchor composition is more collapse-robust than hard MoE/VQ routing.
**True, but softer than a guarantee for the *routed* (V2) variant.** Get the
gradient argument exactly right — a reviewer who knows sparsemax/entmax Jacobians
will check it.

- **Genuinely better than hard top-k / argmax:** α-entmax has a *soft, learned
  threshold*, so anchors **near the selection boundary get nonzero gradient** and
  can be pulled back in. Hard top-k / argmax gives **zero gradient to every
  unselected** expert (0 a.e.) — dropped experts get no signal at all. Marginal
  anchors revive under entmax but not under top-k. (Same reason Soft-MoE /
  Gumbel-VQ beat hard routing.)
- **The honest limit:** α-entmax's Jacobian is **zero on the far-below-threshold
  set**, so an anchor selected by *no* token gets ≈0 gradient through **both**
  paths (value `θ@A` *and* key `A·W_k` — both route through the zeroed entmax
  output) and can die — the SAE dead-latent phenomenon (64–90% dead without
  mitigation). Routing does **not** inherit ANT's *free-weight* revival: in
  original ANT `T_ij` is a free parameter with gradient `⟨g_i,A_j⟩` regardless of
  its value; once θ is produced by router+entmax, dead entries lose that signal.
- **The real robustness — vocabulary-wide sharing:** each anchor is scored by the
  *entire vocabulary* every step, so it dies only if below threshold for *all*
  tokens — far less likely than an MoE expert dropped by one router's argmax. This
  sharing, not a revival guarantee, is the defensible robustness.
- **What the load-balancing loss can and cannot do (get this right — verified on the
  built module):** the Switch-style loss `K·Σ_j usage_j·weight_j` shares the entmax
  Jacobian's blind spot. Its gradient to the selector is **exactly 0 when θ is uniform**
  (the upstream grad `K·usage/(BL)` is a *constant* vector and entmax's Jacobian columns
  sum to 0, annihilating it), and **0 for a fully-dead anchor** (below threshold for all
  tokens → zeroed Jacobian, the same limit as the value/key paths above). So load-balance
  **cannot bootstrap diversity from a uniform/collapsed state and cannot revive a dead
  anchor** — it only keeps *already-used, marginal* anchors from dying. **QK-norm resolves
  the uniform-start case:** with Qwen init + QK-norm (§3.1) scores are O(1) so θ is
  non-uniform *from init* (verified in the real Qwen3 module — an **init** measurement at
  γ=4 gave nnz≈3–4; that is *not* a γ recommendation, keep γ=1 per §3.1), so load-balance has
  real gradient throughout and never needs to bootstrap. Division of labor: **QK-norm + init**
  give a non-uniform start, **load-balance** keeps usage balanced, and **dead-anchor
  resampling** is the only thing that revives a truly-dead anchor.
- **How to defend it:** report an empirical **dead-anchor rate**; use the division of labor
  above (QK-norm+init for a non-uniform start, load-balancing loss to keep usage balanced,
  dead-anchor resampling to revive); use α-entmax (soft boundary) over hard top-k.
  Frame the contribution as *"soft-threshold selection + vocabulary-wide sharing + a light
  load-balancing loss,"* **not** a collapse-free guarantee.
  Cite: Gao 2024 (scaling SAEs), Templeton 2024, Bricken 2023, Ayonrinde 2024
  (feature-choice SAEs, 0% dead), Soft-MoE, Gumbel-VQ.

---

## 6. Lessons Carried from the ADE Reimplementation

1. **Sparsity needs a per-coordinate proximal (YOGI-style `λ·step/(√v+ε)`).** A
   constant `lr·λ` threshold (SGD/Adam) either yields no sparsity or collapses
   everything; SGD couldn't even learn. α-entmax sidesteps the hand-tuned proximal.
2. **Cosine reconstruction loss + proximal is unstable** (gradient scales `1/‖a‖`,
   blows up as weights shrink). End-to-end **LM loss** avoids this entirely — the
   reason generative-from-scratch is the right home (a simple model on a hard
   objective gets zero gradient and the proximal zeros all anchors).
3. **Anchor-level SAT is the latency killer** (expanded sequence → 6.9× slower).
   V2 avoids it by design. Also: keep embedding-vs-layer param savings separate
   (ADE's headline conflated them), and disclose any latency tradeoff.

---

## 7. Experiment Plan (8×H200, ~30B tokens ≈ 1 day per Qwen3-0.6B run)

### 7.1 Shared setup (identical across all arms)
- **Backbone: Qwen3-0.6B architecture, trained FROM SCRATCH (random init, NOT
  pretrained weights).** Config: vocab **151,936**, d=1024, **28 layers**, GQA
  (16 query / 8 KV heads), SwiGLU (intermediate 3072), RoPE, RMSNorm. Instantiate
  via `Qwen3ForCausalLM(Qwen3Config(...))` — **do not** `from_pretrained`.
  - *Controlled comparison, not a match to real Qwen3-0.6B:* the released weights
    saw trillions of tokens; ours (~30B, from scratch) is far weaker. The baseline
    is "standard-embedding Qwen3-0.6B, from scratch, same data" — the released
    Qwen3-0.6B is only a **reference ceiling**, never the thing we beat.
  - *De-risk uses the same Qwen3-0.6B config with a short run (~5B tokens)* so the
    de-risk ladder (§7.2) finishes fast; the full comparison uses the full ~30B tokens.
- **Tokenizer:** Qwen3-0.6B's own tokenizer (vocab 151,936).
- **Corpus:** standard from-scratch LM data — a FineWeb / FineWeb-Edu slice. Same
  data, tokens, steps, seed for **every** arm.
- **The 152K vocab makes the embedding 26% of params (155.6M of 596M tied; verified by
  instantiating the config)** — the
  large-vocab-era compression motivation is concrete and strong here.
- **Output head + the tied-embedding subtlety:** Qwen3 **ties** input=output.
  - For the **de-risk**, set `tie_word_embeddings=False`, give **all arms an
    identical dense `lm_head` (N×d)**, and change only the embedding module
    → clean isolation of the *input* idea (setting A1).
  - **Caveat:** with a tied model, input-only compression is a *scientific* test,
    **not** a parameter win — the equally-large output table remains. The real 26%
    compression prize needs the **output side** compressed too (phase 2, §9).
- **Training:** AdamW, cosine LR schedule; identical for all arms.

### 7.2 Arms — the ablation ladder (train in this order)
Same backbone + same dense output head; only the embedding module changes. Each arm is
defined precisely in §3 — the table just names it and states what it answers.

| Arm | Definition | Question it answers |
|---|---|---|
| **Standard** | §3 rung (0) | reference perplexity |
| **Original ANT** (must-try) | free `T` + YOGI/proximal (§3 rung 1 baseline box) | prior work — does our entmax router beat it (sparsity, ppl, params)? |
| **ANT (ours)** | §3 rung (1), entmax router | does *our* compositional embedding *alone* hold perplexity? |
| **ANT (ours) `H`-ablation — at MATCHED nnz** | §3 rung (1): `H=4` vs `H=1` **with γ tuned so both hit nnz≈15**, all else fixed. (From the §3.1 sweep, `H=1` hits nnz≈15 between γ=0.75→10.1 and γ=0.5→19 — bisect there, then verify the two arms' measured nnz match before comparing ppl.) | does multi-head help *structurally*, or was the proxy's −4% just more anchors? ⚠️ The proxy ran both at γ=1 (nnz 6.8 vs 15) and is **confounded** — this control is what decides whether multi-head (+221K params) is kept at all |
| **V0** (variants: β pre-/post-SAT) | §3 rung (2) | does anchor-level context help, at what `O((L·max_k)²)` cost? |
| **V1** (variants: shared learned query `q_cls` vs mean/content query) | §3 rung (3) | does context-weighting the aggregation add over V0? |
| **V2-attn** *(primary)* | §3 rung (4), 1 causal-attn LocalEnc, residual zero-init, W_q/W_k head | **the contribution:** does context-routed selection help *at all*, given the strongest context? |
| **V2-conv** *(efficiency fallback)* | §3 rung (4), dilated causal conv LocalEnc (run only **after** V2-attn wins) | does a **linear-in-L** encoder recover the gain? (restores "cheap by construction") |
| **V2-conv-lite** | §3 rung (4), single kernel-3 conv LocalEnc | context-depth floor: how little context suffices? |
| *V2 head sub-ablation* | on the winning LocalEnc: swap W_q/W_k → A-only, W_router | how much does the score-head parameterization matter? |
| **V2 load-balance A/B** | §3 rung (4) at `lambda_div=1e-2` vs `lambda_div=0` | does the load-balance loss earn its place? ⚠️ **Note the default: `load_balance` is ON (`lambda_div=1e-2`) for EVERY arm** (pseudocode §6) — so this arm is the *ablation that turns it off*, not an arm that adds it. A proxy A/B showed it matters for V2 (13.8%→0.3% dead) and is ~free on ppl |
| **Isolation control** (§8; **code: pseudocode §5.1b**) | **ANT embedding + LocalEnc output added as a residual** (context is computed and used, but does **not** drive selection). ⚠️ Needs `W_ctl : d_c×d` to lift context to `d` → **~+123K params vs V2**; report it, the arms are not exactly matched | proves V2's gain is *selection*, not just having a context encoder |

### 7.3 Metrics (operational — how to compute each; see pseudocode §7b for two details)
- **Perplexity:** held-out set. Report overall **and frequency-stratified**:
  bucket the vocab into deciles by training-corpus frequency, report ppl per
  decile. *Expected distinctive win: V2 helps the rarest deciles most* (shared
  anchors give rare tokens statistical strength).
- **Dead-anchor rate:** over ~1M held-out tokens, `dead = (# anchors with θ=0 for
  every token) / K`. Also log the anchor-usage histogram (Gini/entropy of usage).
  ⚠️ **With multi-head (`H>1`), log dead-rate PER HEAD as well as globally.** Sub-codebooks
  are disjoint, so each anchor is reachable by exactly one head — a single collapsing head
  silently kills its whole `K/H` block, which a global average can mask (e.g. one dead head
  out of 4 reads as a merely-25% global dead-rate).
- **Anti-collapse loss (ON by default at `lambda_div=1e-2`; the §7.2 arm turns it OFF to
  test it):** load-balancing style — encourage uniform
  average anchor usage across a batch, e.g. `L_div = K · Σ_j (ū_j · m̄_j)` where
  `ū_j` = mean selection weight of anchor j and `m̄_j` = mean fraction of tokens
  selecting j (Switch-Transformer form), or a simple entropy bonus on batch usage.
  ⚠️ **Do not confuse this with the entropy term §3.1 forbids — they act on different
  distributions.** *Forbidden:* entropy of the **per-token** θ_i (over anchors, within one
  token) — penalizing it sharpens θ toward one-hot and accelerates the nnz→1 collapse, and
  the bonus direction was measured unstable. *Allowed here:* entropy of the **batch-average
  anchor usage** `ū` (over anchors, aggregated across all tokens) — a *balance* term, not a
  *per-token sparsity* term; it is maximized by spreading usage across anchors and says
  nothing about how sparse any individual token is. Only the second belongs in `L_div`.
- **Polysemy probe (the decisive analysis):**
  1. pick ~50 polysemous tokens; collect ~100 occurrences each across varied contexts;
  2. record the active anchor set `supp(θ_i)` per occurrence;
  3. metric = **mean pairwise Jaccard distance between a token's anchor sets across
     contexts** — 0 ⇒ selection never changes (no context-dependence), higher ⇒
     context changes selection. **ANT** is **exactly 0** by construction;
     **V2** must be clearly > 0.
  4. bonus: cluster occurrences by known sense, show anchor sets are sense-consistent.
- **Context-depth check:** run the window ladder **downward from the primary**: **V2-attn
  (full history) → V2-conv (~15 tok) → V2-conv-lite (~3 tok)**. If V2-attn ≫ V2-conv,
  selection is context-depth-limited ⇒ V3 (layered routing) is warranted. If V2-conv ≈
  V2-attn, ship the conv (linear cost, same gain). ⚠️ A proxy probe already showed the
  attn↔conv gap is *small* — so do **not** read a V2 null result as "window too narrow."

### 7.4 Go / No-go criteria
- **Go:** V2 ppl ≈ Standard, **V2 clearly better than ANT (ours)** — especially on the
  **rare/low-frequency deciles** (§7.3) — polysemy-Jaccard clearly > 0, dead-anchor rate
  manageable (≲30%, or fixed by anti-collapse), and V2 > Isolation-control
  [selection beats additive context].
  ⚠️ **"V2 > ANT" is non-negotiable and is the criterion most at risk.** A proxy probe found
  V2 ≈ ANT on aggregate ppl with a *healthy* selection (nnz≈15, 0% dead) and a **stronger**
  attention LocalEnc — i.e. the cheap explanations (collapse, narrow window, narrow selection
  channel) are already ruled out. Beating Standard while merely *tying* ANT is **not** a Go:
  it would mean the context-routing — the entire contribution — does nothing, and the paper
  falls back to ANT (ours) as the result.
- **No-go / rethink:** V2 ≈ ANT on ppl **and** polysemy-Jaccard ≈ 0 (routing does
  nothing), or **both** V2 and ANT ≪ Standard (sparse composition too weak — a
  codebook/sparsity problem, not a routing problem).
- **The awkward middle case — plan for it:** V2 ≈ ANT on ppl **but** polysemy-Jaccard > 0
  (context *does* change the selection, it just doesn't *help*). This is the outcome the
  proxy actually produced, and neither branch above covers it. Resolve it with the
  **frequency-stratified** ppl: if V2 wins the rare deciles, the effect is real but
  long-tail-localized (report it that way); if V2 ties ANT *even there*, treat it as
  **No-go for V2** and pivot the paper to ANT (ours) vs Original ANT.

### 7.5 Full comparison (~days, Qwen3-0.6B, ~30B tokens)
Arms: **V2** (best de-risk config) · standard Qwen3-0.6B · **MultiHashFormer** ·
**Original ANT** (free `T` + YOGI/proximal, pseudocode §7c) · **ANT (ours)** (static, §3 rung 1),
matched per §8. ⚠️ **MultiHashFormer is the one arm these documents do not specify** — it is
external prior work (arXiv:2606.28057); use the authors' released implementation, or drop it
and say so. Every other arm is fully spec'd here. Report:
- perplexity overall + frequency-stratified;
- zero-shot downstream (LAMBADA, HellaSwag, ARC-easy, …);
- **params + FLOPs for every arm**, and the quality-vs-params & quality-vs-FLOPs
  **frontier** across 2–3 scales;
- analyses: polysemy probe, anchor-usage / interpretability (which tokens share
  anchors), dead-anchor rate.

### 7.6 Config knobs to fix up front
- **Run scale — fix these first; they must be IDENTICAL across every arm.** Nothing in the
  method depends on the exact values, but the comparison does, so pin them and record them:
  - **`L` (sequence length) = 2048.** Sets `max_position_embeddings` and the LocalEnc's
    `O(L²·d_c)` cost. (The proxy probes used L=256 and the FLOP bench L=1024 — neither is the
    run config.)
  - **Tokens per optimizer step ≈ 0.5M** (e.g. global batch 256 × L=2048), via grad
    accumulation as the hardware requires.
  - **`STEPS` = token-budget ÷ tokens-per-step** — ≈**57k** steps for the ~30B-token full run,
    ≈**10k** for the ~5B-token de-risk (§7.1).
  - **LR 3e-4, AdamW (wd 0.01), cosine decay, warmup ≈1–2% of `STEPS`** (~1000 steps at 57k),
    grad-clip 1.0. Identical for every arm (Original-ANT additionally uses YOGI on its
    embedding — pseudocode §7c).
  - **Precision:** bf16 autocast, but ⚠️ **entmax must run in fp32** (pseudocode §0.1).
- **Scale: Qwen3-0.6B only, for now.** Everything through §7.4's go/no-go runs at the single
  §7.1 config. The multi-scale frontier (§7.5/§8) is **deferred until V2 passes go/no-go** —
  pick the scales then. Two constraints to remember if you get there: hold the **vocab fixed
  at 151,936** (the compression claim is *about* a large vocab), and keep `K`/`d_x`/`d_k`/γ/`H`
  fixed, or the frontier confounds scale with retuning.
- **K** (codebook size) ≈ 4096.
- **d_x** (base table dim) ≈ 128–256; **d_c** (LocalEnc/context dim) **= d_x**
  (equality is *required* by the residual zero-init default; only the A-only head
  needs `d_c = d` instead); **d_k** (router key dim) ≈ 64–128.
- **max_k** (V0/V1 anchor cap per token) ≈ 12–16.
- **Init = Qwen3's** `N(0, 0.02)` on every weight/table (biases 0), **except** residual-Δ
  conv/attn outputs (zero-init). **QK-norm (RMSNorm on router q,k)** is mandatory with this
  init (§3.1) — it makes scores O(1).
- **γ = 1 (default, a NO-OP)** — the score is just the QK-normed similarity; γ is a constant,
  not a parameter. γ is a **sparsity↔quality knob** (full sweep in §3.1): ppl and dead-rate
  keep *improving* as γ falls (γ=0.25 → ppl 319, nnz 65), so γ=1 is **not** the ppl optimum —
  it is chosen because it lands the **graded ~5 band** the thesis requires, at ≈+3.6% ppl vs
  γ=0.5. Re-check this calibration at real scale. **Do NOT** add L1 (`‖θ‖₁≡1`, zero grad)
  or a per-token entropy penalty (it sharpens further — counter-productive; §5/review).
- **LocalEnc (chosen): 1 causal self-attention layer over tokens (V2-attn)** — decide with
  the *strongest* context so a null result can't be blamed on the encoder (proxy probe: the
  encoder was never the bottleneck; the attn-vs-conv gap is small and context was worth only
  ~1% even with full attention). The **2–3 dilated causal Conv1d** stack (dilations 1,2,4, kernel 3,
  channel-mixing) is the **linear-cost efficiency fallback**, run *after* V2-attn wins.
  ⚠️ With attn primary, the document/reset mask inside the LocalEnc is **mandatory** (§3.2).
- **V2 init:** residual LocalEnc with zero-init output (at init `c=X`, Δ=0 → context-free
  ANT-*form* selection; not numerically == a separate ANT run unless `W_q` is shared) —
  the default on every V2 variant (V2-attn / V2-conv / V2-conv-lite).
- **Selection heads `H`** (§3.1 multi-head): `H=1` is the current reference implementation and
  the **default**; `H=4` is *unconfirmed* pending the matched-nnz control (§3.1). Fix one `H`
  across **ANT and V2**. ⚠️ **V0/V1 are exempt and run at `H=1`** — multi-head × V0/V1 is
  deliberately unspecified (their `topk(max_k)` over a head-concatenated θ can draw all slots
  from one head; pseudocode §0.1). If `H=4` is adopted for ANT/V2, note that V0/V1 are then
  not head-matched to them, and compare V0/V1 only against the `H=1` ANT.
- Output stays A1 (input-only, dense `lm_head`) until the input side works;
  compress/tie the output only afterward (§9).

---

## 8. Fair-Comparison Protocol (defuse "you just added capacity")

Two distinct reviewer objections; each needs its own control. Do **not** tune
`d_ff` (or any single knob) to force an exact param match — it confounds FFN
capacity and produces non-standard values that read as cherry-picking.

### Objection 1 — "you added parameters / compute / depth"
- **Keep the backbone standard and identical** for all arms (same layers, width,
  `d_ff = 4d`, head count). Change **only** the embedding module. Any difference
  is then attributable to the embedding.
- **Report exact param AND FLOP deltas** (never just one axis — LocalEnc/router
  add FLOPs even when they save params).
- **Lean on the fact that the compositional input is usually *smaller***
  (`N·d_x + K·d + router ≪ N·d`, since `d_x ≪ d`; strongest for a large vocab like
  Qwen3's 152K). *Winning with equal-or-fewer params on an identical backbone is
  the strongest framing* and needs no contortion.
- **If V2 ends up larger, add the extra budget to the *baseline*** (a bit more
  width or a standard layer) — never handicap V2 by removing one of its layers.

### Objection 2 — "it's just the extra context-processing, not your selection"
The param-match alone does not answer this. Use the **isolation ablation**:
- **V2 (ours):** LocalEnc context `c_i` **drives anchor selection**.
- **Control (identical capacity):** *same* LocalEnc, same params/FLOPs, but its
  output is **added as a residual** to a *static* embedding (context is processed
  and used — just not for selection).
- **Code:** pseudocode **§5.1b** (zero-init `W_ctl`, so the control starts as plain ANT;
  use the *identical* LocalEnc variant and doc-mask as the V2 arm under test).
- The arms are close in params/FLOPs but **not exactly matched**: the control needs
  `W_ctl : d_c×d` (131K) where V2 spends `W_q : d_c×d_k` (8.2K) — **~+123K in the control's
  favour**. Report it. If V2 wins *despite* the control having slightly more capacity, say
  so — the result is stronger that way, and the honest number survives review.
- The **only** substantive difference is whether context *selects anchors*
  vs. *is added in*. If V2 wins, the gain is
  **provably from context-conditioned selection** — this single ablation answers
  both this objection and the "why-not-self-attention" question. (This is the
  "Isolation control" arm in §7.2.)

### Most robust: report the frontier, not a single point
Train both families at **2–3 natural scales** (vary layers/width in standard
steps) and plot **quality vs. params** *and* **quality vs. FLOPs**. Dominance on
the curve at every budget is immune to "you picked a favorable operating point"
and is what scaling-law reviewers expect.

---

## 9. Open Questions / Risks

1. **Does sparse composition preserve LM quality?** Each token embedding is a sum
   of only ~5 shared anchors (≈4.6–5.5 converged at γ=1; ~3–10 band, §3.1) out of
   K=4096 — the constraint is *sparsity*,
   not rank (K>d, so the composed matrix can be full-rank d). Factorized
   embeddings can work (ALBERT), but this *sparse* regime for generative input is
   untested. Note α-entmax also **normalizes θ to sum 1**, so `e` is a *convex*
   combination of anchors (more restrictive than ANT's conic weights); **learnable
   anchor magnitudes** should absorb this (the backbone's RMSNorm does **not** — it
   normalizes each block's input, not the residual stream the raw embedding rides on;
   consistent with §3 V1 and the verified fact that Qwen3 uses the embedding unscaled).
   ⚠️ **Multi-head does NOT relax this** — with `e = (1/H)Σ_h θ_h @ A_h` the effective weight
   on anchor j is `θ_j/H`, which still sums to exactly 1 over all K, so `e` remains a convex
   combination. Multi-head widens *which* combinations are reachable, not the convexity
   constraint itself. If convexity proves limiting, an unnormalized sparse operator
   (e.g. threshold-ReLU) is the fallback — that is the only lever here.
2. **Does context-routing beat static + deep transformer?** — the core empirical
   bet (the "why-not-self-attention" objection).
3. **Base-representation dilemma:** the per-token base table `X (N×d_x)` (used by
   every compositional rung) must be small (`d_x ≪ d`) for compression yet
   expressive enough to route to the right anchors — an untested tradeoff.
4. **Collapse / dead anchors** — a *throughout-training* risk (an anchor drifting below
   threshold for all tokens), not an early-init one (QK-norm gives a non-uniform start, §5);
   monitor the dead-anchor rate, use the load-balance loss + resampling.
5. **Output-layer compression** is where the *real* prize is for a **tied** model
   like Qwen3 — the 26% tied table serves both input and output, so input-only
   compression alone yields no param win. It is the harder half (context-dependent
   selection needs a *static* output composition per candidate token — you can't
   context-route candidates you haven't generated), the part MultiHashFormer
   engineered around. Phase 2.
6. **Freshness:** verify no v2 of the ADE / MultiHashFormer 2026 preprints
   extends into this exact setting before submission.

---

## 10. Build order

These two documents are self-contained: everything below is specified in one of them. Build
in this order — each step is verifiable before the next.

1. **Embedding rungs** — pseudocode §1–§5 (Standard, ANT, V0, V1, V2 + LocalEnc variants),
   §5.1b (isolation control), §0.1 (`select`, and `select_mh` for the `H`-ablation).
2. **Invariant tests** — pseudocode §7, as real tests, on tiny dims. **Do this before the
   training harness.** They catch the traps marked ⚠️ (causality, V1's β·α gradient path,
   preserve-Σβ, `H=1 == select()`, the `theta @ A == H·e` factor) in seconds.
3. **Losses** — pseudocode §6 (`load_balance` only; no entropy/L1 term, and `entropy()` is a
   logged diagnostic).
4. **Harness** — pseudocode §7d (backbone instantiation, packed data + doc/reset mask,
   train loop) and §7c (the Original-ANT arm: `Yogi` + per-coordinate proximal).
5. **Eval** — §7.3 + pseudocode §7b (frequency-stratified ppl with per-bucket `n`,
   dead-anchor rate incl. per-head, polysemy-Jaccard).
6. **Arms** — §7.2 in order, then the §7.4 go/no-go, before committing H200-days to §7.5.

**Two experiments are obligations, not options** — both currently block claims in this doc:
- **matched-nnz `H`-ablation** (§7.2): the proxy's −4% for multi-head is confounded with nnz.
  This decides whether multi-head is kept at all.
- **γ sweep including γ<1** (§3.1): γ=1 is not the ppl optimum; the recommendation rests on
  the sparsity goal, and that needs to hold at real scale.

**Facts already established at proxy scale** (folded into §3.1/§5 — do not re-derive):
(1) with Qwen's 0.02 init the router/attention dot products vanish (~5.9e-5) → **QK-norm is
required** on every scorer (router, anchor-SAT, V2-attn) to get O(1) scores; (2) load-balance
has zero gradient at uniform θ — but QK-norm makes θ non-uniform from init, so that failure
mode no longer bites; (3) input-side compression at 0.6B is **6.56×** (155.6M → 23.7M).
