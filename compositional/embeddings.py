import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neg(t):
    """Dtype-safe large negative for masked_fill. Never a literal -1e9."""
    return torch.finfo(t.dtype).min


def _sinusoidal_pe(max_len, d):
    """Standard sinusoidal positional encoding table, shape (max_len, d)."""
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# ALBERT-style low-rank factorization (Lan et al. 2020) — no-routing control
# ---------------------------------------------------------------------------

class LowRankEmbed(nn.Module):
    """Factorized embedding: V x r lookup then r -> d linear projection.

    The parameter-matched control for the compositional arms: same low-rank
    token table dimension (r = d_x = 128) but no anchors and no routing.
    Isolates whether anchor routing adds anything beyond a plain linear
    bottleneck. Matches HF ALBERT: Linear with bias, normal(0.02) init.

    Forward returns (e, None): no theta, so the trainer logs no routing
    metrics and the div loss is skipped.
    """

    def __init__(self, vocab_size, embed_dim, rank=128):
        super().__init__()
        self.X = nn.Parameter(torch.randn(vocab_size, rank) * 0.02)
        self.proj = nn.Linear(rank, embed_dim, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids, doc_mask=None):
        return self.proj(self.X[input_ids]), None


# ---------------------------------------------------------------------------
# Original ANT (Liang et al. 2021)
# ---------------------------------------------------------------------------

class OriginalANT(nn.Module):
    """Free per-token weight matrix + shared codebook.

    Forward: e = T[token_ids] @ A
    Sparsity from YOGI's per-coordinate proximal applied to T externally.
    """

    def __init__(self, vocab_size, codebook_size, embed_dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(codebook_size, embed_dim) * 0.02)
        self.T = nn.Parameter(torch.empty(vocab_size, codebook_size).uniform_(0, 0.02))

    def forward(self, input_ids, doc_mask=None):
        w = self.T[input_ids]  # (B, L, K)
        e = w @ self.A  # (B, L, d)
        return e, w

    def sparse_params(self):
        return [self.T]

    def non_sparse_params(self):
        return [self.A]


# ---------------------------------------------------------------------------
# ANT (ours) — rung 1: entmax-routed, context-free
# ---------------------------------------------------------------------------

class ANTEmbed(nn.Module):
    """Entmax-routed compositional embedding (context-free selection).

    Forward: u = X[ids] -> theta = select(u) -> e = theta @ A
    Base class for V0, V1, V2 which reuse _select() and the shared parameters.

    When num_heads > 1, the codebook is split into H disjoint sub-codebooks and
    H independent routers run in parallel. ANT and V2 support this; V0/V1 do not
    (topk over concatenated theta breaks per-head structure).
    """

    def __init__(self, vocab_size, codebook_size, embed_dim, d_x=128, d_k=64,
                 gamma=1.0, num_heads=1):
        super().__init__()
        self.d_k = d_k
        self.gamma = gamma
        self.num_heads = num_heads

        self.A = nn.Parameter(torch.randn(codebook_size, embed_dim) * 0.02)
        self.X = nn.Parameter(torch.randn(vocab_size, d_x) * 0.02)
        self.q_norm = nn.RMSNorm(d_k)
        self.k_norm = nn.RMSNorm(d_k)

        if num_heads == 1:
            self.W_q = nn.Parameter(torch.randn(d_x, d_k) * 0.02)
            self.W_k = nn.Parameter(torch.randn(embed_dim, d_k) * 0.02)
        else:
            assert codebook_size % num_heads == 0, \
                f"K={codebook_size} must be divisible by num_heads={num_heads}"
            self.W_q_mh = nn.Parameter(torch.randn(num_heads, d_x, d_k) * 0.02)
            self.W_k_mh = nn.Parameter(torch.randn(num_heads, embed_dim, d_k) * 0.02)

    def _select(self, u):
        """Single-head selection. Returns theta only; caller does e = theta @ A."""
        q = self.q_norm(u @ self.W_q)                      # (B, L, d_k)
        k = self.k_norm(self.A @ self.W_k)                 # (K, d_k)
        s = self.gamma * (q @ k.T) / math.sqrt(self.d_k)   # (B, L, K)
        theta = entmax15(s.float(), dim=-1)                 # fp32 for Sigma=1
        return theta.to(u.dtype)

    def _select_mh(self, u):
        """Multi-head selection. Returns (e, theta) directly.

        Each head routes over a disjoint K/H sub-codebook. theta is the
        concatenation across heads (B, L, K) with per-head rows summing to 1.
        e = (1/H) * sum_h (theta_h @ A_h) — do NOT also compute theta @ A.
        """
        H = self.num_heads
        K_h = self.A.size(0) // H
        d = self.A.size(1)
        A_h = self.A.view(H, K_h, d)

        e = 0
        thetas = []
        for h in range(H):
            q = self.q_norm(u @ self.W_q_mh[h])                    # (B, L, d_k)
            k = self.k_norm(A_h[h] @ self.W_k_mh[h])               # (K/H, d_k)
            s = self.gamma * (q @ k.T) / math.sqrt(self.d_k)        # (B, L, K/H)
            th = entmax15(s.float(), dim=-1).to(u.dtype)
            e = e + th @ A_h[h]
            thetas.append(th)

        return e / H, torch.cat(thetas, dim=-1)

    def _embed(self, u):
        """Route u through single- or multi-head selection. Returns (e, theta)."""
        if self.num_heads == 1:
            theta = self._select(u)
            return theta @ self.A, theta
        return self._select_mh(u)

    def forward(self, input_ids, doc_mask=None):
        u = self.X[input_ids]       # (B, L, d_x)
        return self._embed(u)


# ---------------------------------------------------------------------------
# V0 — rung 2: static selection + causal anchor-level SAT
# ---------------------------------------------------------------------------

class V0Embed(ANTEmbed):
    """ANT selection + causal self-attention over the expanded anchor sequence.

    Cost O((L * max_k)^2) — the reason V2 exists.

    Args:
        mode: "post" = beta applied after SAT (default);
              "pre"  = beta scales anchors before SAT.
    """

    def __init__(self, vocab_size, codebook_size, embed_dim, d_x=128, d_k=64,
                 gamma=1.0, max_k=16, mode="post", max_pe_len=8192, **kwargs):
        super().__init__(vocab_size, codebook_size, embed_dim, d_x, d_k, gamma,
                         num_heads=1)
        self.max_k = max_k
        self.mode = mode

        d = embed_dim
        self.Wq_sat = nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wk_sat = nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wv_sat = nn.Parameter(torch.randn(d, d) * 0.02)
        self.Wo_sat = nn.Parameter(torch.randn(d, d) * 0.02)
        self.sat_q_norm = nn.RMSNorm(d)
        self.sat_k_norm = nn.RMSNorm(d)

        self.gpe_scale = nn.Parameter(torch.tensor(0.02))
        self.register_buffer("gpe", _sinusoidal_pe(max_pe_len, d))

    def _anchor_sat(self, a, real):
        """Causal self-attention over the expanded anchor sequence.

        Args:
            a: (B, L, max_k, d) gathered anchor vectors.
            real: (B, L, max_k) True=active, False=padding.
        Returns:
            (B, L, max_k, d) contextualized anchors.
        """
        B, L, max_k, d = a.shape
        S = L * max_k

        a = a + self.gpe_scale * self.gpe[:L].view(1, L, 1, d)

        seq = a.reshape(B, S, d)

        tok = torch.arange(S, device=a.device) // max_k
        causal = tok[:, None] >= tok[None, :]           # (S, S)
        keep = causal.unsqueeze(0) & real.reshape(B, 1, S)  # (B, S, S)

        Q = self.sat_q_norm(seq @ self.Wq_sat)
        Kk = self.sat_k_norm(seq @ self.Wk_sat)
        V = seq @ self.Wv_sat

        att = (Q @ Kk.transpose(1, 2)) / math.sqrt(d)
        att = att.masked_fill(~keep, _neg(att)).softmax(-1)

        return ((att @ V) @ self.Wo_sat).reshape(B, L, max_k, d)

    def forward(self, input_ids, doc_mask=None):
        u = self.X[input_ids]
        theta = self._select(u)                          # (B, L, K) full pre-topk

        beta, idx = theta.topk(self.max_k, dim=-1)       # (B, L, max_k)
        real = beta > 0

        a = self.A[idx]                                   # (B, L, max_k, d)

        if self.mode == "pre":
            a = a * beta.unsqueeze(-1)

        a_ctx = self._anchor_sat(a, real)

        if self.mode == "post":
            e = (beta.unsqueeze(-1) * a_ctx).sum(2)
        else:
            e = (a_ctx * real.unsqueeze(-1).float()).sum(2)

        return e, theta


# ---------------------------------------------------------------------------
# V1 — rung 3: V0 + context-dependent aggregation weight
# ---------------------------------------------------------------------------

class V1Embed(V0Embed):
    """V0 + context-modulated aggregation: w = beta * alpha, preserve Sigma(beta).

    Args:
        query: "content" = mean-pool real slots as query (default);
               "cls"     = shared learned query vector.
    """

    def __init__(self, vocab_size, codebook_size, embed_dim, d_x=128, d_k=64,
                 gamma=1.0, max_k=16, query="content", max_pe_len=8192):
        super().__init__(vocab_size, codebook_size, embed_dim, d_x, d_k, gamma,
                         max_k, mode="post", max_pe_len=max_pe_len)
        self.query_mode = query

        if query == "cls":
            self.q_cls = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def forward(self, input_ids, doc_mask=None):
        u = self.X[input_ids]
        theta = self._select(u)                          # (B, L, K)

        beta, idx = theta.topk(self.max_k, dim=-1)
        real = beta > 0

        a_ctx = self._anchor_sat(self.A[idx], real)      # raw anchors, no beta-pre

        # Context weight alpha
        if self.query_mode == "cls":
            q = self.q_cls.view(1, 1, 1, -1)             # (1, 1, 1, d)
        else:
            n_real = real.sum(-1, keepdim=True).clamp_min(1).unsqueeze(-1)  # (B, L, 1, 1)
            q = (a_ctx * real.unsqueeze(-1)).sum(2, keepdim=True) / n_real  # (B, L, 1, d)

        z = (q * a_ctx).sum(-1)                           # (B, L, max_k)
        z = z.masked_fill(~real, _neg(z))
        alpha = z.softmax(-1)                              # (B, L, max_k)

        w = beta * alpha
        w = w * beta.sum(-1, keepdim=True) / (w.sum(-1, keepdim=True) + 1e-9)

        e = (w.unsqueeze(-1) * a_ctx).sum(2)              # (B, L, d)
        return e, theta


# ---------------------------------------------------------------------------
# LocalEnc variants for V2
# ---------------------------------------------------------------------------

class LocalEncAttn(nn.Module):
    """1 causal self-attention layer over tokens. Returns the residual delta only."""

    def __init__(self, d_x):
        super().__init__()
        self.d_x = d_x
        self.Wq_a = nn.Parameter(torch.randn(d_x, d_x) * 0.02)
        self.Wk_a = nn.Parameter(torch.randn(d_x, d_x) * 0.02)
        self.Wv_a = nn.Parameter(torch.randn(d_x, d_x) * 0.02)
        self.Wo_a = nn.Parameter(torch.zeros(d_x, d_x))    # zero-init -> delta=0
        self.qa_norm = nn.RMSNorm(d_x)
        self.ka_norm = nn.RMSNorm(d_x)

    def forward(self, x, doc_mask=None):
        B, L, d = x.shape
        Q = self.qa_norm(x @ self.Wq_a)
        Kk = self.ka_norm(x @ self.Wk_a)
        V = x @ self.Wv_a

        m = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        if doc_mask is not None:
            m = m.unsqueeze(0) & doc_mask

        scores = (Q @ Kk.transpose(1, 2)) / math.sqrt(d)
        scores = scores.masked_fill(~m, _neg(scores))
        att = scores.softmax(-1)

        return (att @ V) @ self.Wo_a


class LocalEncConv(nn.Module):
    """2-3 dilated causal Conv1d (dilations 1,2,4), receptive field ~15 tokens.
    Returns the residual delta only."""

    def __init__(self, d_x):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_x, d_x, kernel_size=3, dilation=1),
            nn.Conv1d(d_x, d_x, kernel_size=3, dilation=2),
            nn.Conv1d(d_x, d_x, kernel_size=3, dilation=4),
        ])
        self.dilations = [1, 2, 4]

        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)
        # Zero-init last layer -> delta=0 at init
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, x, doc_mask=None):
        h = x.transpose(1, 2)                              # (B, d_x, L)
        for conv, dil in zip(self.convs, self.dilations):
            h = conv(F.pad(h, ((3 - 1) * dil, 0)))         # causal left-pad
        return h.transpose(1, 2)


class LocalEncConvLite(nn.Module):
    """Single k=3 causal Conv1d (~3-token window).
    Returns the residual delta only."""

    def __init__(self, d_x):
        super().__init__()
        self.conv = nn.Conv1d(d_x, d_x, kernel_size=3)
        nn.init.zeros_(self.conv.weight)                    # zero-init -> delta=0
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, doc_mask=None):
        h = x.transpose(1, 2)                              # (B, d_x, L)
        h = self.conv(F.pad(h, (2, 0)))                    # causal left-pad 2
        return h.transpose(1, 2)


# ---------------------------------------------------------------------------
# V2 — rung 4: context-conditioned SELECTION (the contribution)
# ---------------------------------------------------------------------------

class V2Embed(ANTEmbed):
    """Context-routed selection: LocalEnc contextualizes tokens, router selects
    different anchors for the same token in different contexts.

    c = x + localenc(x)  (residual, delta=0 at init -> starts as ANT)
    theta = select(c)    (context-conditioned)
    e = theta @ A

    No anchor-level SAT needed. Cost O(L*K*d), linear in L.
    """

    def __init__(self, vocab_size, codebook_size, embed_dim, d_x=128, d_k=64,
                 gamma=1.0, num_heads=1, localenc="attn"):
        super().__init__(vocab_size, codebook_size, embed_dim, d_x, d_k, gamma,
                         num_heads=num_heads)

        if localenc == "attn":
            self.localenc = LocalEncAttn(d_x)
        elif localenc == "conv":
            self.localenc = LocalEncConv(d_x)
        elif localenc == "conv_lite":
            self.localenc = LocalEncConvLite(d_x)
        else:
            raise ValueError(f"Unknown localenc: {localenc}")

    def forward(self, input_ids, doc_mask=None):
        x = self.X[input_ids]                               # (B, L, d_x)
        c = x + self.localenc(x, doc_mask)                  # residual, delta=0 at init
        return self._embed(c)


# ---------------------------------------------------------------------------
# Isolation control (design section 8, Objection 2)
# ---------------------------------------------------------------------------

class IsolationControlEmbed(ANTEmbed):
    """Same LocalEnc as V2, but context is added to a static ANT embedding
    instead of driving the router. Proves V2's gain is from selection,
    not from merely having a context encoder.

    e = (theta @ A) + localenc(x) @ W_ctl
    theta is static (from x, NOT from c).
    """

    def __init__(self, vocab_size, codebook_size, embed_dim, d_x=128, d_k=64,
                 gamma=1.0, num_heads=1, localenc="attn"):
        super().__init__(vocab_size, codebook_size, embed_dim, d_x, d_k, gamma,
                         num_heads=num_heads)

        if localenc == "attn":
            self.localenc = LocalEncAttn(d_x)
        elif localenc == "conv":
            self.localenc = LocalEncConv(d_x)
        elif localenc == "conv_lite":
            self.localenc = LocalEncConvLite(d_x)
        else:
            raise ValueError(f"Unknown localenc: {localenc}")

        self.W_ctl = nn.Parameter(torch.zeros(d_x, embed_dim))  # zero-init

    def forward(self, input_ids, doc_mask=None):
        x = self.X[input_ids]                               # (B, L, d_x)
        e, theta = self._embed(x)                           # STATIC selection
        ctx = self.localenc(x, doc_mask)
        return e + ctx @ self.W_ctl, theta
