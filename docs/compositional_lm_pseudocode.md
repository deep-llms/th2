# Compositional Embeddings — Implementation Pseudocode

Authoritative, shape-annotated pseudocode for `compositional_lm_design.md` §3.
**Implement from this file.** It covers every rung (§1–§5), the isolation control (§5.1b),
losses (§6), invariants (§7), eval details (§7b), the Original-ANT baseline arm (§7c), and
the training harness (§7d). What it does *not* cover — **which** arms to run, in what order,
and the go/no-go criteria — is design §7.2/§7.4/§7.6.

Every variant here mirrors code that passed the verification suite (causality, ladder
isolation, gradient completeness). `⚠️` marks a line whose *exact* form is a correctness
requirement — each one was a real bug when written the "obvious" way (see the note at
its side). PyTorch-flavored; `B`=batch, `L`=seq len, `N`=vocab, others in §0.

Each rung is a standalone module `forward(ids, doc_mask=None) -> (e, theta)`, called
**explicitly by the harness** (§7d) — *not* installed as `embed_tokens`. The harness sets
`lm.model.embed_tokens = None` and feeds the result in via `inputs_embeds=e`, which is what
lets it also take `theta` for the aux losses (§6) and thread the doc-mask (§5.1). The
backbone (Qwen3-0.6B) and dense `lm_head` are identical across all arms; **only this
module changes.**

> **Verified against `transformers` Qwen3 source:** the model uses the embedding output
> **UNSCALED** (`hidden_states = inputs_embeds`) — do **NOT** multiply by `√d` (that's
> Gemma, not Qwen3); return raw `(B,L,d)`. Set **`tie_word_embeddings=False`** (a composed
> module has no single `.weight` to tie to `lm_head`) and give a separate dense `lm_head`.
> Default `pad_token_id=None`, so there's no `padding_idx` row to keep at zero.

---

## 0. Dimensions & shared parameters

```python
# dims (see design §7.6)
N   = 151936     # vocab
d   = 1024       # model / anchor dim  (backbone hidden size)
K   = 4096       # codebook size (# anchors)
d_x = 128        # base-table dim   (d_x << d  -> input compression)
d_c = d_x        # context dim      ⚠️ MUST equal d_x for the residual zero-init default
d_k = 64         # router key dim
max_k = 16       # V0/V1 only: anchors kept per token  ⚠️ must be >= entmax support (§ note)

# learned parameters shared across rungs
A  : Parameter(K,   d)     # the anchor codebook (shared semantic atoms)
X  : Parameter(N,   d_x)   # base token table — the ONLY per-token param; token identity
W_q: Parameter(d_src, d_k) # query proj;  d_src = d_x (ANT) or d_c (V2)
W_k: Parameter(d,   d_k)   # key proj (keys derived from A)
q_norm, k_norm : RMSNorm(d_k)  # QK-norm on router query & key (exactly Qwen3's attention)
gamma : scalar = 1.0       # score temperature. DEFAULT 1.0 = NO-OP: a plain constant (not a
                           # Parameter, no gradient) -> score is just the QK-normed similarity.
                           # KEEP γ=1: holds converged nnz≈4.6–5.5; higher γ collapses it. γ is a
                           # SPARSITY<->QUALITY knob, not a ppl optimum — see the γ box (§0.1).

# INIT = Qwen3's: every weight/table ~ N(0, initializer_range=0.02), biases 0, RMSNorm weight=1;
#        EXCEPT the deliberate zero-inits (residual-Δ conv/attn output — §5) which stay 0.
#        ⚠️ Conv1d default init is kaiming, NOT normal — re-init conv weights to N(0,0.02) to match.
```

### 0.1 Selection score + sparse operator (used by every rung)

```python
def select(u, W_q):                      # u: (B,L,d_src)  ->  theta: (B,L,K)
    # scaled query·key similarity (keys = A projected), with QK-norm on q and k
    q = q_norm(u @ W_q)                                 # (B,L,d_k) unit-RMS  (QK-norm)
    k = k_norm(A @ W_k)                                 # (K,  d_k) unit-RMS  (QK-norm)
    s = gamma * (q @ k.T) / sqrt(d_k)                  # (B,L,K)  scores are O(1) at init
    theta = entmax15(s.float(), dim=-1)                # (B,L,K) sparse, rows sum to 1
    return theta.to(u.dtype)                            # back to model dtype for θ@A
    # ⚠️ QK-NORM IS REQUIRED with Qwen's N(0,0.02) init: without it the 4-fold product
    #    (u·W_q·W_k·A) of 0.02-scale matrices gives scores ~5.9e-5 -> θ UNIFORM -> γ would
    #    need ~30000 to sparsify (verified). q_norm/k_norm (RMSNorm = Qwen3's attention
    #    QK-norm) rescale q,k to unit RMS -> scores O(1) -> γ SANE (keep γ=1; see the γ box
    #    below — do NOT tune γ by the INIT nnz, the converged nnz moves the opposite way).
    #    BONUS: θ is non-uniform FROM INIT, so the dense-at-init / load-balance-can't-bootstrap
    #    problem (§6) no longer bites. (A-only head: RMSNorm u and A directly, dim d.)
    # ⚠️ COMPUTE ENTMAX IN fp32 (the .float()). In bf16 the Σθ=1 normalization drifts to
    #    ~[0.985,1.025] (±2.5%, verified) — it does NOT NaN or lose sparsity, but the
    #    "convex combination / Σθ=1" invariant that V1's preserve-Σβ and the collapse
    #    analysis rely on only holds in fp32. entmax is not a standard autocast-upcast op,
    #    so under naive autocast it would run in bf16 — cast explicitly. (fp16 drift is
    #    smaller ~0.3% but still cast for consistency.) Scores are only B·L·K — cheap in fp32.
    # entmax15 = α-entmax with α=1.5 (from the `entmax` package): exact zeros,
    # variable # of nonzeros per row, differentiable ON ITS SUPPORT.
    # ⚠️ Differentiable-on-support is NOT a revival guarantee: an anchor that is below
    #    threshold for ALL tokens gets ~0 gradient (value path θ@A AND key path A·W_k
    #    both route through the zeroed entmax output). Keep-alive comes from the
    #    load-balance loss + vocab-wide sharing, NOT from entmax itself.
```

**Multi-head selection (`H` heads) — optional and UNCONFIRMED; `H=1` == `select()` above and is
the default.** ⚠️ The −4% proxy result is **confounded with nnz** (both arms ran γ=1, so H=4
carried ~4× the support); design §3.1 specifies the matched-nnz control that decides whether
this is kept. Implement it to *run that control*, not because it is established.

```python
# Split the codebook into H DISJOINT sub-codebooks of K/H and run H routers in parallel.
# ⚠️ CODEBOOK-param-neutral and FLOP-neutral IN THE DOMINANT TERMS (what defuses "you just added
#    capacity" in design §8) -- but be precise, §8 demands EXACT deltas:
#      params : codebook still K×d EXACTLY; only the router projections are ×H.
#               ⚠️ COUNT BOTH, and note they are DIFFERENT sizes: W_q is d_src×d_k (128*64
#               = 8.2K) but W_k is d×d_k (1024*64 = 65.5K) -- W_k is 8x bigger, so counting
#               W_q alone undercounts by ~9x. At H=4 the delta is
#                   (H-1)*(d_src*d_k + d*d_k) = 3*(8.2K + 65.5K) = +221K params
#               = ~5% of the 4.19M codebook, or ~0.9% of the ~23.7M full compositional
#               embedding (design §10). Small -- but report it, and count BOTH projections.
#      FLOPs, per token, EXACTLY neutral:
#               scoring   H·(K/H)·d_k = K·d_k          <- identical to H=1
#               value     H·(K/H)·d   = K·d            <- identical to H=1
#               key proj  H·(K/H)·d·d_k = K·d·d_k      <- identical to H=1 (amortized over batch)
#      ⚠️ NOT neutral: the QUERY projection u@W_q_mh[h] runs ONCE PER HEAD -> H·d_src·d_k
#         (vs d_src·d_k at H=1). It is the ONLY H× term. Size it before claiming anything:
#         at K=4096,d=1024,d_k=64,d_src=128 the per-token embed FLOPs are value 4.19M,
#         scoring 262K, query proj 8.2K -> H=4 raises query proj to 32.8K, i.e. from ~0.18%
#         to ~0.73% of the embedding (which is itself <1% of the backbone, design §4).
#         So: report it as "+<1% of embedding FLOPs", NOT as "identical". Do not overclaim.
# ⚠️ DISTINCT TENSORS from the §0 single-head W_q/W_k (different shape: leading H dim).
#    Do NOT reuse the §0 names — `select()` above is called with the (d_src,d_k) W_q and
#    would break on a 3-D one. Multi-head REPLACES select() entirely; it does not wrap it.
W_q_mh: Parameter(H, d_src, d_k)   # per-head query proj
W_k_mh: Parameter(H, d,     d_k)   # per-head key proj

def select_mh(u):                   # u:(B,L,d_src) -> e:(B,L,d), theta:(B,L,K)
    A_h = A.view(H, K // H, d)      # disjoint sub-codebooks
    e, thetas = 0, []
    for h in range(H):
        q = q_norm(u @ W_q_mh[h])                       # (B,L,d_k)
        k = k_norm(A_h[h] @ W_k_mh[h])                  # (K/H,d_k)
        s = gamma * (q @ k.T) / sqrt(d_k)               # (B,L,K/H)
        th = entmax15(s.float(), dim=-1).to(u.dtype)    # ⚠️ SAME fp32 entmax as select()
        e = e + th @ A_h[h]
        thetas.append(th)
    return e / H, cat(thetas, dim=-1)                   # ⚠️ /H keeps ‖e‖ comparable to H=1
    # ⚠️ theta is the CONCATENATION over heads -> (B,L,K), so load_balance(theta) (§6) and the
    #    nnz/dead diagnostics work UNCHANGED. Note Σθ = H (not 1) across the concat: each head
    #    is its own simplex. The convex-combination/Σθ=1 invariant holds PER HEAD, and e is the
    #    MEAN of H convex combinations -> still a convex combination over all K (weights th_j/H),
    #    so multi-head does NOT relax the convexity constraint (design §9).
    # ⚠️ MULTI-HEAD x V0/V1 IS NOT SPECIFIED HERE — do not combine without re-deriving. V0/V1
    #    do `theta.topk(max_k)` over the FULL (B,L,K) vector; on a head-concatenated theta that
    #    topk can draw all max_k slots from ONE head, silently destroying the per-head structure
    #    (and Sum(beta) is then no longer per-head-normalized, breaking V1's preserve-Sum(beta)).
    #    Multi-head is specified for ANT and V2 only. A correct V0/V1 version needs a PER-HEAD
    #    topk (max_k/H each) and per-head Sum(beta) preservation — derive it before using it.
    # ⚠️ Disjoint sub-codebooks mean an anchor is reachable by exactly ONE head — a head that
    #    collapses takes its K/H anchors with it. Log dead-rate PER HEAD, not just globally.

# ⚠️ CALL SITES CHANGE. select_mh returns (e, theta); select() returns theta only and the
#    caller does `e = theta @ A`. So multi-head rungs become ONE call, not two lines:
#        ant : e, theta = select_mh(X[token_ids])        # was: theta=select(...); e=theta@A
#        v2  : e, theta = select_mh(c)                   # c = x + localenc(x), as before
# ⚠️ DO NOT then also compute `theta @ A` — it does NOT error and is NOT garbage, it returns
#    EXACTLY H*e (the concat weights hit the same-ordered A, giving Sum_h theta_h @ A_h = H*e).
#    A silent H-times-too-large embedding is the single easiest bug to introduce here.
```

**Sparsity is set by γ (and entmax α), NOT by a penalty term. Keep γ LOW (=1).**

```python
# ⚠️ EMPIRICAL (proxy probes, 1.2-1.5k steps, held-out): γ sets the CONVERGED support AND is a
#    SPARSITY<->QUALITY TRADEOFF — it is NOT a free win. Full sweep (nnz / ppl / dead):
#      γ=3 -> 1.1/396/9.0%   γ=1.5 -> 2.2/359/3.4%   γ=1 -> 4.6-5.5/337/0.8%
#      γ=0.75 -> 10.1/332/0.1%   γ=0.5 -> 19/325/0.0%   γ=0.25 -> 64.6/319/0.0%
#    ⚠️ ppl and dead-rate keep IMPROVING as γ falls, with no turning point down to 0.25 — so
#    γ=1 is NOT the ppl optimum (an earlier sweep tested only {1,1.5,3}, i.e. γ=1 was its
#    boundary). Keep γ=1 because it lands the GRADED ~5 BAND the thesis needs (and V0/V1's
#    max_k=16 requires), at a stated cost of ~+3.6% ppl vs γ=0.5. Do NOT control nnz with a
#    loss term:
#  - Do NOT add an ENTROPY PENALTY (L_ent = +lambda*entropy): it minimizes entropy toward
#    one-hot, ACCELERATING collapse (wrong direction); the useful direction (entropy BONUS,
#    -lambda) is UNSTABLE — it killed polysemy at γ=3 and ran away to nnz~130 at γ=1.5. Drop it.
#  - Do NOT add an L1 penalty: entmax output is on the simplex (Σθ=1) -> ||theta||_1 == 1
#    IDENTICALLY -> gradient wrt scores is EXACTLY 0 (verified). L1 does nothing.
def entropy(theta):                             # kept only as a DIAGNOSTIC to LOG, not a loss
    p = theta.clamp_min(1e-9)
    return -(p * p.log()).sum(-1).mean()
# ⚠️ Real-scale nnz (K=4096, full vocab, 30B tokens) still TBD by the de-risk — but the proxy
#    says graded selection (~5) is reachable and preferred at γ=1, not something to force.
# Always LOG avg-nnz = (theta>0).sum(-1).mean() and the dead-anchor rate throughout training.
```

### 0.2 Masking constant + causality principle

```python
# ⚠️ DTYPE-SAFE MASK FILL. Never a literal -1e9. Compute the fill from the ACTUAL tensor
#    being masked (its dtype), via this helper — used at every masked_fill below:
def neg(t):                                     # t = the score tensor about to be masked
    return torch.finfo(t.dtype).min             # -3.4e38 (fp32/bf16), -65504 (fp16)
#   A literal -1e9 RAISES "value cannot be converted to at::Half without overflow" in fp16
#   (verified) -> crashes fp16 training; bf16/fp32 tolerate it but neg() is uniformly correct.
#   neg() also keeps an all-masked row's softmax finite (uniform), preserving the 0/0-NaN guard.
#   (Must be per-tensor, not a single constant — different masked tensors may differ in dtype.)

# INVARIANT:  e_i must depend only on tokens <= i.
# Test it: perturb token j; embeddings at positions 0..j-1 must be BITWISE identical,
#          and position j itself must change. (Both halves matter — see verify_adversarial.)
# Every context mechanism below (LocalEnc conv/attn, anchor-SAT) is masked causally.
```

---

## 1. Standard (rung 0) — reference

```python
E : Parameter(N, d)
def standard(token_ids):            # (B,L) -> (B,L,d)
    return E[token_ids]
```

## 2. ANT (rung 1) — static compositional, context-free selection

```python
def ant(token_ids):                 # (B,L) -> (B,L,d)
    u = X[token_ids]                # (B,L,d_x)  source = token's own base vector, NO context
    theta = select(u, W_q)          # (B,L,K)    W_q is (d_x, d_k) here
    e = theta @ A                   # (B,L,d)
    return e
    # Causal by construction: e_i depends only on token i. Baseline for "does a shared
    # codebook (vs a free table) hold perplexity?"  Polysemy-Jaccard is EXACTLY 0 here.
```

---

## 3. V0 (rung 2) — static selection + causal anchor-level SAT (ADE-style)

Selection is still the static ANT θ; the new mechanism is that the **selected anchor
vectors** are contextualized by attention (across tokens) before aggregation.

```python
# extra params: Wq_sat,Wk_sat,Wv_sat,Wo_sat : (d,d)  (the anchor-SAT projections);
#   sat_q_norm, sat_k_norm : RMSNorm(d)  (QK-norm for the SAT); gpe_scale : scalar (learnable).
#   ⚠️ Wq_sat is DISTINCT from the §0 selection proj `W_q` (d_src×d_k) — different tensor,
#      different shape; the case-only name W_q vs Wq_sat is intentional, don't conflate.
#   GPE : sinusoidal, shape (L,d), SCALED by a learnable scalar gpe_scale (init = 0.02).
#   ⚠️ A raw sinusoid (norm ~22) SWAMPS the 0.02-init anchor vectors (norm ~0.66, 34x) ->
#      the SAT would be POSITION-dominated at init. gpe_scale (init=std) makes GPE comparable
#      to the anchor magnitude at init and lets the model tune positional strength.

def anchor_sat(a, real):            # a:(B,L,max_k,d) gathered anchors, real:(B,L,max_k) -> (B,L,max_k,d)
    a = a + gpe_scale * GPE.view(1, L, 1, d)          # ⚠️ sinusoidal, indexed by TOKEN pos i (all
                                                      #    max_k slots share i); scaled (see above);
                                                      #    NOT a learned L×d table (that caps length)
    S = L * max_k
    seq = a.reshape(B, S, d)
    tok = arange(S) // max_k                          # which token each flat slot belongs to
    causal = tok[:,None] >= tok[None,:]               # (S,S) attend to anchors of tokens <= i
    keep   = causal[None] & real.reshape(B,1,S)       # (B,S,S) also drop padding KEYS
    Q,Kk,V = sat_q_norm(seq@Wq_sat), sat_k_norm(seq@Wk_sat), seq@Wv_sat   # ⚠️ QK-norm (see below)
    att = (Q @ Kk.transpose(1,2)) / sqrt(d)
    att = att.masked_fill(~keep, neg(att)).softmax(-1)   # ⚠️ neg(att) dtype-safe (§0.2), not -inf/-1e9
    # ⚠️ QK-norm on the SAT (sat_q_norm/sat_k_norm : RMSNorm(d)) is REQUIRED: at 0.02 init the
    #    Q·K logits are ~0.0002 -> PERFECTLY uniform (content-blind) attention (verified). Norm -> O(1).
    return ((att @ V) @ Wo_sat).reshape(B, L, max_k, d)   # cost O((L·max_k)^2) — the reason V2 exists

def v0(token_ids, mode="post"):     # (B,L) -> (B,L,d)
    theta = select(X[token_ids], W_q)                 # (B,L,K) static selection (selection W_q!)
    beta, idx = theta.topk(max_k, dim=-1)             # (B,L,max_k) weights + anchor ids
    real = beta > 0                                   # (B,L,max_k) True=active, False=pad
    #  ⚠️ if support(theta) > max_k the smallest actives are DROPPED (Σβ<1). Log
    #     truncation rate = mean(support > max_k); keep it ~0 by max_k >= support.
    a = A[idx]                                        # (B,L,max_k,d) gather anchor vectors
    if mode == "pre":
        a = a * beta.unsqueeze(-1)                    # β scaled in BEFORE the SAT (variant)
        # ⚠️ note: the SAT's QK-norm normalizes β out of the attention Q/K (per-slot RMSNorm),
        #    so β-pre affects only the SAT VALUES, not the attention scores. Still a valid
        #    variant (β weights value contributions); just narrower than pre-QK-norm.
    a_ctx = anchor_sat(a, real)                       # (B,L,max_k,d) contextualized anchors

    # ---- aggregate max_k contextualized anchors -> one embedding per token ----
    if mode == "post":
        return (beta.unsqueeze(-1) * a_ctx).sum(2)    # β applied AFTER SAT; padding (β=0) drops out
    else:  # "pre"
        return (a_ctx * real.unsqueeze(-1)).sum(2)    # ⚠️ sum REAL slots only — padding got β=0
                                                      #    pre-SAT but SAT gives it nonzero a_ctx,
                                                      #    so a plain .sum(2) would leak it
```

*Isolation fact (verified):* with the SAT set to identity, `GPE=0`, and `max_k=K`, V0
reduces to ANT **exactly** — so any V0−ANT gap is attributable only to the SAT.

---

## 4. V1 (rung 3) — V0 + context-dependent aggregation weight

Steps 1–4 identical to V0. Step 5 **modulates** the static β with a context weight α
(intra-token pooling, `O(L·max_k)`). This rung is where the "obvious" implementation has
**two** real bugs — both fixed below.

```python
# extra param (variant A only): q_cls : Parameter(d)

def v1(token_ids, query="content"):      # (B,L) -> (B,L,d)
    # --- steps 1–4: identical to V0, producing beta, real, a_ctx ---
    theta = select(X[token_ids], W_q)
    beta, idx = theta.topk(max_k, dim=-1); real = beta > 0
    a_ctx = anchor_sat(A[idx], real)     # (B,L,max_k,d)  same helper as V0 (raw anchors, no β-pre)

    # --- step 5: context weight α, then COMBINE with β (do not replace) ---
    if query == "cls":
        q = q_cls.view(1,1,1,d)                        # shared learned query
    else:  # "content": mean over REAL slots only
        q = (a_ctx * real.unsqueeze(-1)).sum(2, keepdim=True) \
            / real.sum(2, keepdim=True).clamp_min(1).unsqueeze(-1)   # ⚠️ real-only mean

    z = (q * a_ctx).sum(-1)
    z = z.masked_fill(~real, neg(z))                   # ⚠️ neg(z) dtype-safe (§0.2), mask padding slots
    alpha = z.softmax(-1)                               # (B,L,max_k)  context weight, Σ=1

    w = beta * alpha                                    # ⚠️ BUG-FIX #1: keep β. If you aggregate
    #   by alpha alone, β never enters the forward, the top-k indices are non-diff, and the
    #   ENTIRE selector (X, W_q, W_k, gamma) gets ZERO task gradient — verified ∂loss/∂X = None.
    #   Multiplying by β is the only differentiable path back to the selector.

    w = w * beta.sum(-1, keepdim=True) / (w.sum(-1, keepdim=True) + 1e-9)
    #   ⚠️ BUG-FIX #2: rescale so Σw = Σβ (NOT normalize to 1). This makes V1 reduce to V0
    #   EXACTLY when alpha is uniform, so a V1−V0 gap is pure context effect, not a magnitude
    #   change. (Normalizing to 1 injects a per-token magnitude confound that rides the
    #   residual stream un-normalized — RMSNorm does not remove it.)

    e = (w.unsqueeze(-1) * a_ctx).sum(2)               # (B,L,d)
    return e
```

---

## 5. V2 (rung 4) — context-conditioned SELECTION (the contribution)

The selection **source** becomes context-mixed, so *which* anchors are selected depends
on neighbors — and **no anchor-SAT is needed** (router is `O(L·K·d)`, linear in L). Only the
`LocalEnc` differs across V2 variants; everything downstream is identical.

```python
# ⚠️ `localenc` (and V0's `mode`, V1's `query`) are BOUND AT CONSTRUCTION by build_arm (§7d),
#    NOT passed per call — every arm's runtime signature is forward(ids, doc_mask=None).
def v2(token_ids, doc_mask=None):       # (B,L) -> (B,L,d)
    x = X[token_ids]                    # (B,L,d_x)
    c = x + localenc(x, doc_mask)       # ⚠️ THREAD THE MASK: localenc_attn needs it (§5.1); the
                                        #    conv variants ignore it (their window is local).
    #                                   # (B,L,d_c)  ⚠️ RESIDUAL, and localenc is ZERO-INIT at
    #   ⚠️ here `localenc` == Δ ONLY (the residual add is EXPLICIT here). The design doc §3.2
    #      folds the residual INTO its "LocalEnc" (its c = LocalEnc(X) = X + Δ) — so do NOT
    #      also add x inside localenc, or you double-count x (c = 2x + Δ). Follow THIS file.
    #   its output -> at init c == x EXACTLY (Δ=0, verified), so selection is context-free
    #   (ANT-FORM). Training grows the context correction, so any gain is PROVABLY from
    #   context. Needs d_c == d_x. (Numerically identical to a given ANT run only if they
    #   SHARE W_q — across independent arms each has its own W_q, so test `c == X`, not `==ANT`.)
    theta = select(c, W_q)             # (B,L,K)   W_q is (d_c, d_k) here
    e = theta @ A                      # (B,L,d)
    return e
```

### 5.1 LocalEnc variants — the V2 arms, in RUN ORDER

Run order: **V2-attn first (decide), then V2-conv (optimize).** The encoder was never the
bottleneck in the proxy probe — upgrading conv→attn moved ppl far less than widening the
*selection* channel — so use the strongest context to decide whether context-routing works
at all, and only then check whether a linear-cost encoder recovers the gain.

```python
# extra params (V2-attn): Wq_a,Wk_a,Wv_a,Wo_a : (d_x,d_x); qa_norm,ka_norm : RMSNorm(d_x).
#   Wo_a is ZERO-init (the residual Δ=0); the rest ~N(0,0.02).  zero_init(t) := t.zero_()
#   GPE (V0/V1 only) := standard sinusoidal table, shape (L,d), built once, not learned.
# ---- V2-attn (PRIMARY): 1 causal self-attn layer over TOKENS, full history ----
# Decide with the strongest context: if V2 fails HERE, it fails regardless of encoder.
# NOTE: attends over L TOKENS at d_c (~d_x) — never over anchors. No O((L*max_k)^2) term.
def localenc_attn(x, doc_mask=None):    # (B,L,d_x) -> (B,L,d_c), CAUSAL
    #   doc_mask: (B,L,L) bool, True = "key j is visible to query i" — the SAME mask the
    #   backbone uses for packed sequences. ⚠️ MANDATORY when packing: without it this attn
    #   leaks the ENTIRE previous document (conv only leaks <= rf-1 tokens). Pass it through
    #   from the harness (§7d); None is only valid for one-document-per-block data.
    Q,Kk,V = qa_norm(x@Wq_a), ka_norm(x@Wk_a), x@Wv_a   # ⚠️ QK-norm (RMSNorm d_x) — same reason
    m = (arange(L)[:,None] >= arange(L)[None,:])           # causal    as the SAT/router: 0.02 init
    if doc_mask is not None: m = m & doc_mask              # causal AND same-document
    scores = (Q@Kk.transpose(1,2))/sqrt(d_x)               #  -> ~0 logits -> uniform without it
    att = scores.masked_fill(~m, neg(scores)).softmax(-1)   # neg(scores) dtype-safe (§0.2)
    return (att @ V) @ Wo_a             # ⚠️ zero-init Wo_a for the residual; O(L^2·d_c) cost

# ---- V2-conv (EFFICIENCY FALLBACK, run only AFTER V2-attn wins): LINEAR in L ----
# regular (channel-mixing) conv, NOT depthwise — channel mixing is what lets a neighbor
# change WHICH anchor is picked. dilations grow the window cheaply.
conv1 = Conv1d(d_x, d_c, k=3, dilation=1)
conv2 = Conv1d(d_c, d_c, k=3, dilation=2)
conv3 = Conv1d(d_c, d_c, k=3, dilation=4);  zero_init(conv3)   # ⚠️ zero-init LAST layer -> Δ=0

def localenc_conv(x):                   # (B,L,d_x) -> (B,L,d_c), CAUSAL
    h = x.transpose(1,2)                # (B,d_x,L)
    for conv, dil in [(conv1,1),(conv2,2),(conv3,4)]:
        h = conv(F.pad(h, ((3-1)*dil, 0)))   # ⚠️ F.pad last-dim (left=(k-1)*dilation, right=0) => causal
    return h.transpose(1,2)
    # receptive field = 1 + (k-1)*Σdilations = 1+2*7 = 15 tokens (verified causal + rf=15).

# ---- V2-conv-lite: single k=3 conv (~3-token window) — the context-DEPTH floor ----
conv_single = Conv1d(d_x, d_c, k=3);  zero_init(conv_single)   # zero-init for residual (Δ=0)
def localenc_conv_lite(x):
    return conv_single(F.pad(x.transpose(1,2), (2,0))).transpose(1,2)   # left-pad 2 => causal, rf=3
```

### 5.1b Isolation control (design §8, Objection 2) — a REQUIRED arm, spec'd here

The control that proves V2's gain comes from *selection*, not from merely having a context
encoder. Same LocalEnc, but its output is **added to a static ANT embedding** instead of
driving the router.

```python
# extra param: W_ctl : Parameter(d_c, d)   # lifts context d_c -> d so it can be ADDED to e
#   ⚠️ zero-init W_ctl (same residual logic as LocalEnc): at init the control == plain ANT.

def isolation_control(token_ids, doc_mask=None):   # (B,L) -> (B,L,d)   (localenc bound at ctor)
    x     = X[token_ids]                        # (B,L,d_x)
    theta = select(x, W_q)                      # ⚠️ STATIC selection — source is x, NOT c.
    e     = theta @ A                           # (B,L,d)  the ANT embedding
    ctx   = localenc(x, doc_mask)               # ⚠️ SAME encoder, config AND mask as the V2 arm
    return e + ctx @ W_ctl, theta               # context is computed and used, but never selects
```

⚠️ **Param accounting — the arms are NOT automatically matched.** The control needs
`W_ctl : d_c×d` (128*1024 = **131K**) to lift context into `d`; V2 spends only `W_q : d_c×d_k`
(**8.2K**) on the same context. So the control is **~+123K params vs V2** at the §0 dims.
That is small, but design §8 demands exact deltas — **report it**, and do not claim the two
arms are bit-for-bit matched. (If V2 wins *despite* the control having slightly more
capacity, the result is only stronger; state it that way.)

⚠️ Use the **identical LocalEnc variant** (attn/conv) and the identical doc-mask as the V2
arm being tested, or the comparison is confounded.

### 5.2 Score-head variants (secondary knob; default = W_q/W_k above)

Each variant **replaces the `theta = select(c, W_q)` line inside `v2()`** with its own
score `s` followed by `theta = entmax15(s.float(), dim=-1).to(c.dtype)` — **same fp32
entmax as `select()`** (§0.1); do not drop the `.float()`. `select()` bundles the default
head; the others compute `s` directly:

All heads keep the **QK-norm / input-norm** so scores are O(1) at Qwen init (§0.1):
```python
# default   : exactly select() — q=q_norm(c@W_q), k=k_norm(A@W_k), s=gamma*(q@k.T)/sqrt(d_k)
# W_router  : s = gamma * (in_norm(c) @ W_router)   # W_router:(d_c,K); RMSNorm the SOURCE.
#             ⚠️ No key-norm (W_router rows aren't unit) -> DENSER than the default head at
#                equal γ (measured nnz≈44 vs 6 at γ=3, K=4096). Give this arm its OWN, higher
#                γ to hit the avg-nnz target. c is d_c-dim -> residual c=x+localenc(x) unchanged.
# A-only    : q=qA_norm(c), k=kA_norm(A); s = gamma * (q @ k.T) / sqrt(d)   # key = A directly.
#   ⚠️ qA_norm/kA_norm are SEPARATE RMSNorm(d) modules — NOT the default head's RMSNorm(d_k)
#      (different dim: A-only works in dim d, the default in d_k). Don't reuse the d_k norms.
#   ⚠️ needs c of dim d (A is K×d), but the residual c = x + localenc(x) is d_x-dim, so A-only
#   does NOT drop into v2() as-is. Run it standalone with a d-dim context, EITHER:
#     (i)  set d_x = d (no input compression — a pure head ablation), keep residual init; OR
#     (ii) build c = localenc_to_d(x)  [LocalEnc mapping d_x -> d], NON-residual — which
#          forfeits the clean Δ=0 init (c ≠ X at init), so it is NOT a rung on the ANT ladder.
```

---

## 6. Losses

⚠️ **Surface `theta` from the forward.** The rungs above are written `return e` for
brevity, but the aux losses need the **full pre-topk** selection `theta` (B,L,K) — so each
compositional rung must actually `return e, theta` (V0/V1: the `select(...)` output
*before* `.topk`; V2: `select(c)`). `Standard` has no `theta` (`theta=None`), and the aux
losses simply don't apply to that arm (`L_spa = L_div = 0`).

```python
def total_loss(logits, targets, theta, lambda_div=1e-2):   # theta: (B,L,K) or None
    L_lm = cross_entropy(logits, targets)           # next-token prediction (the real objective)
    if theta is None:                               # Standard arm — no composition, no aux loss
        return L_lm, {"lm": L_lm.detach()}
    L_div = load_balance(theta)                     # keep anchors alive. See §5.
    return L_lm + lambda_div * L_div, {"lm": L_lm.detach(), "div": L_div.detach(),
                                       "ent": entropy(theta).detach()}   # ent = DIAGNOSTIC only
    # ⚠️ There is deliberately NO entropy/L1 knob in the signature. An entropy penalty
    #    accelerates the nnz->1 collapse (§0.1) and L1 is inert on the simplex, so neither
    #    belongs in the loss — exposing a weight for them is a footgun. load_balance is the
    #    only aux term, and even it is optional (test an arm with lambda_div=0). L_lm is the
    #    objective. ⚠️ Returns (loss, logs) — the §7d train loop unpacks both.

def load_balance(theta):                            # Switch-Transformer style
    usage  = (theta > 0).float().mean(dim=(0,1))    # (K,) fraction of tokens using anchor j (hard)
    weight = theta.mean(dim=(0,1))                  # (K,) mean selection weight (differentiable)
    return K * (usage * weight).sum()               # minimized at uniform usage; grad flows via
    #  `weight` (usage is a non-diff count, exactly like Switch's f_i).
# ⚠️ VANISHES AT UNIFORM θ (math fact): at uniform usage the upstream grad K·usage/(BL) is a
#    CONSTANT vector and entmax's Jacobian columns sum to zero -> it annihilates any constant,
#    so ∂L_div/∂selector is EXACTLY 0. Hence load-balance cannot BOOTSTRAP diversity from a
#    fully uniform/collapsed state — it only balances already-non-uniform usage.
# ✅ RESOLVED HERE by QK-norm (§0.1): scores are O(1) so θ is non-uniform FROM INIT (verified
#    at INIT in the real Qwen3 module; γ=4 there gave nnz≈3–4/256 — an init measurement only,
#    NOT a recommended γ: keep γ=1, see §8) -> the model never sits at uniform ->
#    load-balance has real gradient throughout. (Pre-QK-norm a tiny-init router WAS uniform
#    and this bit; QK-norm removes that failure mode.)
```

---

## 7. Invariants to assert in a unit test (all verified in scratchpad harness)

```python
# 1. CAUSALITY   perturb token j -> e[:, :j] bit-identical, e[:, j] changed.  (every rung)
# 2. LADDER      V2 init: c == X exactly (Δ=0; == ANT only if the whole selector — W_q,W_k,
#                q_norm,k_norm — is shared) ;
#                V0(SAT=id, GPE=0, max_k=K) == ANT ;  V1(uniform α) == V0.
# 3. GRADIENT    ∂loss/∂{A, X, W_q, W_k, QK-norms, LocalEnc} all finite & NONZERO (esp. V1 via
#                β). (gamma only if it's a learnable Parameter — as a fixed buffer it has none.)
# 4. SPARSITY    QK-norm -> avg-nnz is non-uniform from init (γ=1 -> ~27/4096 at init, which
#                CONVERGES to ~4.6-5.5 — see §8: init nnz and converged nnz move OPPOSITE ways
#                with γ, so assert on the converged value, not the init one). Log it.
# 5. BATCH       perturbing batch row b leaves all other rows unchanged.
# 6. MULTI-HEAD  select_mh at H=1 == select() EXACTLY -- but ONLY when the weights are copied
#                across (W_q_mh[0]=W_q, W_k_mh[0]=W_k); they are DIFFERENT Parameters, so a
#                fresh H=1 module is NOT bitwise-equal to a fresh select(). Assert on a
#                weight-copied pair. Also: theta is (B,L,K) with Sum(theta)==H and each
#                per-head row summing to 1; codebook param count is independent of H.
#                (Only if multi-head is enabled.)
```

---

## 7b. Two eval details design §7.3 leaves ambiguous

- **Frequency deciles:** bucket by **type rank** (sort vocab by train count, split into 10).
  **Report the eval-token count `n` per bucket next to the ppl** — the rarest buckets can be
  so sparse that ppl ≈ `exp(ln V)` (random). A proxy run hit exactly that. Report those as
  *unmeasured*, not as a V2-vs-ANT result.
- **Polysemy-Jaccard:** distance = `1 − |supp_a ∩ supp_b| / |supp_a ∪ supp_b|`, meaned over
  all occurrence pairs of a token, then over tokens. **Freeze one token list and reuse it for
  every arm** (the metric is only comparable on an identical set). Sanity check: **ANT must
  score exactly 0** — nonzero means the harness is wrong. With `H>1`, `supp(θ)` is the union
  across heads, which the concatenated `(B,L,K)` θ already gives.

---

## 7c. Original-ANT baseline arm (design §3.2) — the prior work we compare against

Required §7.2 arm. Unlike our rungs it is **not** a router: `T` is a free per-token weight
matrix made sparse by a **per-coordinate proximal** inside a **YOGI** step. A plain
`lr·λ` threshold (SGD/Adam) does not work — that was the round-1 lesson (design §6).

```python
class OriginalANT(nn.Module):
    def __init__(self, N, K, d):
        super().__init__()
        self.A = nn.Parameter(randn(K, d) * 0.02)
        self.T = nn.Parameter(empty(N, K).uniform_(0, 0.02))   # free, NON-NEGATIVE init
        self.T._sparse_param = True                            # flag: proximal applies to T only
    def forward(self, ids):
        w = self.T[ids]                    # (B,L,K) per-token weights
        return w @ self.A, w               # e, theta-equivalent

class Yogi(Optimizer):                     # Adam-family with additive-sign second moment
    def __init__(self, params, lr=1e-2, betas=(0.9,0.999), eps=1e-3, v_init=1e-6):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps)); self.v_init = v_init
    def step(self, l1_penalty=0.0):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                grad, st = p.grad.data, self.state[p]
                if not st:
                    st['step'] = 0; st['exp_avg'] = zeros_like(p.data)
                    st['exp_avg_sq'] = zeros_like(p.data) + self.v_init
                m, v = st['exp_avg'], st['exp_avg_sq']
                b1, b2 = g['betas']; st['step'] += 1
                m.mul_(b1).add_(grad, alpha=1-b1)
                g2 = grad*grad
                v.add_((v - g2).sign_() * g2, alpha=b2-1)      # ⚠️ YOGI's sign rule, not Adam's
                denom = v.sqrt().add_(g['eps'])
                bc1, bc2 = 1-b1**st['step'], 1-b2**st['step']
                step_size = g['lr'] * sqrt(bc2) / bc1
                p.data.addcdiv_(m, denom, value=-step_size)
                if l1_penalty > 0 and getattr(p, '_sparse_param', False):
                    thr = l1_penalty * (step_size / denom)     # ⚠️ PER-COORDINATE threshold
                    p.data = (p.data > thr).float() * (p.data - thr)   # soft-threshold + clamp>=0
```

**Wiring:** `Yogi` on the embedding (`lr=1e-2`), **AdamW on the backbone** (same LR/schedule as
every other arm). λ is **0 during warmup, then ramps linearly** to its target:
```python
def lam_at(step, LAM, WARMUP, STEPS):
    return 0.0 if step < WARMUP else LAM * (step-WARMUP) / max(1, STEPS-WARMUP)
# each step:  emb_opt.step(l1_penalty=lam_at(...));  bb_opt.step()
```
⚠️ λ needs a sweep (this arm's sparsity is *tuned*, ours is not — that asymmetry is the point).
⚠️ `T` is `N×K` = **155.8M** params at the §0 dims; budget memory for it.

---

## 7d. Experiment harness (design §7.1) — backbone, data, loop

```python
# ---- deps ----  torch>=2.4 (nn.RMSNorm), transformers, datasets, entmax
# ---- backbone: FROM SCRATCH, never from_pretrained ----
cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, num_hidden_layers=28,
                  num_attention_heads=16, num_key_value_heads=8,      # GQA 16/8
                  intermediate_size=3072, max_position_embeddings=L,
                  tie_word_embeddings=False)                          # ⚠️ untied dense lm_head
lm = Qwen3ForCausalLM(cfg); lm.model.embed_tokens = None   # unused: we pass inputs_embeds

# ---- data: same tokens/order/seed for EVERY arm ----
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")               # vocab 151936
# FineWeb / FineWeb-Edu slice, packed to fixed-length L blocks; hold out a fixed eval split.
# ⚠️ If packing multiple documents per block, build the doc/reset mask HERE. Both the backbone
#    and V2-attn's LocalEnc must see the SAME visibility — but they want DIFFERENT FORMATS:
#      doc_mask : (B,L,L) bool, True = "key j visible to query i"   -> for localenc_attn (§5.1)
#      HF wants : (B,1,L,L) ADDITIVE float mask (0 keep, -inf drop) -> for Qwen3ForCausalLM.
#    ⚠️ Passing the (B,L,L) bool straight to HF as `attention_mask=` is WRONG — HF's 2-D
#    `attention_mask` is a (B,L) PADDING mask, so a 3-D bool is either rejected or misread.
#    Derive one from the other; do not build them independently:
def hf_mask(doc_mask, dtype):        # (B,L,L) bool -> (B,1,L,L) additive float
    causal  = torch.ones(L, L, dtype=torch.bool, device=doc_mask.device).tril()
    visible = doc_mask & causal      # ⚠️ INCLUDE CAUSALITY — see the warning below
    z = torch.zeros(visible.shape, dtype=dtype, device=doc_mask.device)
    return z.masked_fill(~visible, torch.finfo(dtype).min)[:, None]    # same fill as neg() (§0.2)
# ⚠️ A 4-D attention_mask is treated by recent `transformers` as the COMPLETE mask — HF does
#    NOT add the causal triangle on top of it. So the causal part must be baked in here, as
#    above. This is version-dependent, and getting it wrong silently breaks causality (the
#    backbone would attend to future tokens). VERIFY IT: run the §0.2 causality probe through
#    the FULL model (perturb token j -> logits at positions < j must be bitwise identical).
#    Do this once, before any training run.
# (localenc_attn applies its own triangular `m` and ANDs doc_mask, so it is already correct;
#  doc_mask itself carries only the DOCUMENT-boundary part.)

# ---- arm contract ----
# build_arm(name) returns a module whose forward(ids, doc_mask=None) -> (e, theta).
# ⚠️ EVERY arm returns the 2-tuple, including Standard (§1), which returns (E[ids], None).
#    theta=None is what routes the Standard arm past the aux loss (§6).
# ⚠️ Variant knobs are BOUND AT CONSTRUCTION, never passed per call: V2/isolation-control's
#    `localenc`, V0's `mode`, V1's `query`. The §1-§5 snippets show them as extra args for
#    readability; build_arm must close over them so every arm has the SAME runtime signature.
#    (Otherwise `embed(x, doc_mask)` would pass the mask into v2's `localenc` slot.)
embed = build_arm(arm_name)          # rungs §1-§5, isolation control §5.1b, Original-ANT §7c

# ---- optimizers: ONE arm is different ----
IS_ORIG_ANT = (arm_name == "original_ant")
bb_opt = AdamW(lm.parameters(), lr=3e-4, weight_decay=0.01)
if IS_ORIG_ANT:
    emb_opt = Yogi(embed.parameters(), lr=1e-2)        # §7c — proximal lives inside .step()
else:
    emb_opt = AdamW(embed.parameters(), lr=3e-4, weight_decay=0.01)
scheds = [CosineAnnealingLR(o, STEPS) for o in (bb_opt, emb_opt)]  # one per optimizer,
                                                  # + warmup; identical schedule shape for both

# ---- train loop (identical across arms; only `embed` and the emb optimizer change) ----
for step in range(STEPS):
    x, doc_mask = next_batch()                    # (B,L), (B,L,L) bool or None
    e, theta = embed(x, doc_mask)                                     # bool form -> LocalEnc
    logits = lm(inputs_embeds=e,
                attention_mask=hf_mask(doc_mask, e.dtype) if doc_mask is not None else None
                ).logits                                              # ⚠️ 4-D additive form -> HF
    # ⚠️ Original-ANT is the LITERAL prior method — its sparsity is the proximal (§7c), NOT our
    #    load-balance. Pass lambda_div=0 for it, or you contaminate the baseline (and confound
    #    the "our ANT vs Original ANT" comparison). w still surfaces for nnz/dead logging.
    loss, logs = total_loss(logits[:, :-1], x[:, 1:], theta,
                            lambda_div=0 if IS_ORIG_ANT else 1e-2)    # §6
    bb_opt.zero_grad(); emb_opt.zero_grad(); loss.backward()
    clip_grad_norm_(lm.parameters(), 1.0)         # ⚠️ backbone and embedding are clipped
                                                  #    SEPARATELY (not as one joint norm) so that
                                                  #    Yogi's params can be left unclipped — its
                                                  #    proximal assumes the raw step size (§7c).
                                                  #    Identical for every non-Original-ANT arm,
                                                  #    so parity holds; state it in the writeup.
    if IS_ORIG_ANT: emb_opt.step(l1_penalty=lam_at(step, LAM, WARMUP, STEPS))   # §7c ramp;
                                                  #    LAM from the §7c sweep, WARMUP per §7.6
    else:           clip_grad_norm_(embed.parameters(), 1.0); emb_opt.step()
    bb_opt.step(); [s.step() for s in scheds]
    if step % LOG_EVERY == 0:
        log(avg_nnz=(theta>0).sum(-1).mean(), dead=..., **logs)       # §7b, per-head if H>1
```

⚠️ **Arm parity is the whole experiment.** Same data order, steps, seed, LR schedule, and
backbone init for every arm; change **only** `embed`. Any deviation must be reported (§8).

---

## 8. Implementation notes (what this spec leaves to you)

The math above is exact and verified. These are the choices left to the implementer —
init and γ affect *training*, the `embed_tokens` note is an *integration* detail, and the
last is a *process* recommendation. None change module correctness, but each is easy to get
wrong:

- **Parameter init = Qwen3's** (§0): every weight/table `~ N(0, 0.02)`, biases 0, RMSNorm
  weight 1, **except** the residual-Δ conv/attn output which stays zero-init. Re-init
  `Conv1d` weights to `N(0,0.02)` (their default is kaiming). QK-norm (§0.1) is what makes
  this init usable — without it, `N(0,0.02)` on the router gives ~5.9e-5 scores and a dead
  (uniform) router.
- **γ = 1, fixed (§0.1 + §6). ⚠️ Do NOT pick γ from the *init* nnz — that is the trap.**
  With QK-norm the scores are O(1) at init, and the *init* avg-nnz reads γ≈1 → ≈27/4096,
  γ≈2 → ≈5–10, γ≈4 → ≈4. Tuning γ to hit "nnz 5–10 at init" therefore lands you at γ≈2–4 —
  and the **converged** nnz then **collapses** (proxy probe: γ=3 → ≈1.1, γ=1.5 → ≈2.0,
  while **γ=1 holds ≈4.6–5.5**). Init nnz and converged nnz move in **opposite** directions
  with γ; only the converged one matters. ⚠️ Lower γ is *better* on ppl (§0.1 sweep) — γ=1 is
  the **sparsity** choice, not the quality optimum.
  **So: set γ = 1 and leave it** (a plain constant, a no-op — not a Parameter, no schedule),
  and **monitor avg-nnz throughout training**, not at init. (Contrast: *without* QK-norm the
  same init needed γ≈30000 — do not do that.) Because QK-norm gives non-uniform θ from init,
  the load-balance loss has real gradient immediately (§6) — no bootstrap problem.
- **Do NOT install the rung as `embed_tokens`; no shim.** Set `lm.model.embed_tokens = None`,
  call `e, theta = embed(ids, doc_mask)` yourself, and pass `inputs_embeds=e` (§7d). An
  `embed_tokens(ids) -> tensor` signature can carry neither θ (needed by §6) nor the doc-mask
  (needed by §5.1), so routing through it forces a side-channel for both.
- **Write §7's invariants as real tests first.** Every rung's forward here was validated
  against them (11/11 passing in the originating repo), and they are the cheapest way to
  catch the traps this file marks with ⚠️ — causality, the V1 β·α gradient path, preserve-Σβ,
  the `H=1 == select()` equivalence, and the `theta @ A == H·e` factor. Build them before the
  training harness; they run in seconds on tiny dims (e.g. `N=99, d=32, K=16, d_x=8, d_k=4`).
