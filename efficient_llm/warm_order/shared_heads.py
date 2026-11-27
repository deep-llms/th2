"""Cross-layer shared attention heads in place of DOT's separate T layer (docs/DOT_SHARED_HEADS_20261125.md).

User (25 Nov): share attention heads between layers (added to or replacing own heads) instead of a separate T layer;
the shared heads are the only difference from the base transformer. A shared head bank (q, k, v projections and q/k
norms, one set for the whole model) is applied by every block application to that block's own normalised input.
Modes, i.e. how the shared heads' output reaches the residual stream:

  replace : the last GQA group of every block (heads // kv_heads q heads + 1 kv head) IS the shared bank; each block's
            own o_proj reads all heads. Same compute as the base transformer, fewer parameters.
  concat  : all own heads plus one extra shared GQA group, concatenated; each block's o_proj gains columns for the
            shared group (zero init: starts as the base model). Concat followed by a linear map equals a weighted
            sum with matrix weights, so this also covers "concat then project".
  vadd    : one shared head (1 q + 1 kv); its output is added to every own head's output with a per-block, per-head
            weight (zero init), then the block's own o_proj.
  oadd    : one extra shared GQA group with a shared o_proj; its output is added to the block's attention output
            with a per-block, per-channel gate (zero init).

Order of block applications: 'dense' (B0 B1 B2 B3) or the looped schedules of loop_baselines (cycle, prc).
"""
import torch
from torch import nn
from torch.nn import functional as F

from efficient_llm.dot_pool.model import RMSNorm, attend

from .loop_baselines import BlockSchedule

MODES = ('replace', 'concat', 'vadd', 'oadd')


def _trimmed(linear, rows):
    """A bias-free Linear keeping the first `rows` output rows of `linear` (same values)."""
    out = nn.Linear(linear.in_features, rows, bias=False)
    with torch.no_grad():
        out.weight.copy_(linear.weight[:rows])
    return out


class SharedHeadSchedule(BlockSchedule):
    def __init__(self, config, arm, spec):
        super().__init__(config, arm, spec)  # identical base init (embed, B0..B3, head) to the other arms of the seed
        c, hd = config, config.head_dim
        self.mode = spec['shared']['mode']
        if self.mode not in MODES:
            raise ValueError('unknown shared-head mode')
        if c.kv_heads < 2 and self.mode == 'replace':
            raise ValueError('replace needs at least two kv groups')
        g = c.heads // c.kv_heads  # q heads per kv head (one GQA group)
        n_q = 1 if self.mode == 'vadd' else g
        self.sh_q = nn.Linear(c.d, n_q * hd, bias=False)
        self.sh_k = nn.Linear(c.d, hd, bias=False)
        self.sh_v = nn.Linear(c.d, hd, bias=False)
        self.sh_q_norm, self.sh_k_norm = RMSNorm(hd), RMSNorm(hd)
        if self.mode == 'replace':
            # the bank starts as block 0's last group; every block keeps its other groups (same values as dense4)
            first = self.blocks[0].self_attn
            with torch.no_grad():
                self.sh_q.weight.copy_(first.q_proj.weight[-g * hd:])
                self.sh_k.weight.copy_(first.k_proj.weight[-hd:])
                self.sh_v.weight.copy_(first.v_proj.weight[-hd:])
            for block in self.blocks:
                a = block.self_attn
                a.q_proj = _trimmed(a.q_proj, (c.heads - g) * hd)
                a.k_proj = _trimmed(a.k_proj, (c.kv_heads - 1) * hd)
                a.v_proj = _trimmed(a.v_proj, (c.kv_heads - 1) * hd)
        else:
            for module in (self.sh_q, self.sh_k, self.sh_v):
                nn.init.normal_(module.weight, std=.02)
        if self.mode == 'concat':
            self.sh_o_cols = nn.ModuleList([nn.Linear(g * hd, c.d, bias=False) for _ in self.blocks])
            for module in self.sh_o_cols:
                nn.init.zeros_(module.weight)
        if self.mode == 'vadd':
            self.sh_w = nn.Parameter(torch.zeros(len(self.blocks), c.heads))
        if self.mode == 'oadd':
            self.sh_o = nn.Linear(g * hd, c.d, bias=False)
            nn.init.normal_(self.sh_o.weight, std=.02)
            self.sh_gate = nn.Parameter(torch.zeros(len(self.blocks), c.d))
        for block in self.blocks:
            if block.self_attn.c.backend not in ('sdpa', 'fused', 'fa4'):
                raise ValueError('unknown attention backend')

    @staticmethod
    def _attend(a, q, k, v, pos, segments):
        """Attention.forward after the projections and q/k norms: rotary, then the shared causal-within-document kernel."""
        b, t, n_q, hd = q.shape
        angles = pos.float()[..., None] * a.inv
        angles = torch.cat([angles, angles], -1)[:, :, None, :]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

        def rotary(z):
            return z * cos + torch.cat([-z[..., hd // 2:], z[..., :hd // 2]], -1) * sin

        return attend(rotary(q), rotary(k), v, segments, a.c.backend)

    def _attention(self, b, u, pos, segments):
        a, hd = self.blocks[b].self_attn, self.c.head_dim
        B, T = u.shape[:2]
        q = a.q_norm(a.q_proj(u).view(B, T, -1, hd))
        k = a.k_norm(a.k_proj(u).view(B, T, -1, hd))
        v = a.v_proj(u).view(B, T, -1, hd)
        sq = self.sh_q_norm(self.sh_q(u).view(B, T, -1, hd))
        sk = self.sh_k_norm(self.sh_k(u).view(B, T, 1, hd))
        sv = self.sh_v(u).view(B, T, 1, hd)
        if self.mode in ('replace', 'concat'):  # one attention call; the shared group is the last GQA group
            out = self._attend(a, torch.cat([q, sq], 2), torch.cat([k, sk], 2), torch.cat([v, sv], 2), pos, segments)
            if self.mode == 'replace':
                return a.o_proj(out)
            n_own = q.shape[2] * hd
            return a.o_proj(out[..., :n_own]) + self.sh_o_cols[b](out[..., n_own:])
        own = self._attend(a, q, k, v, pos, segments)
        shared = self._attend(a, sq, sk, sv, pos, segments)
        if self.mode == 'vadd':
            w = self.sh_w[b].to(own.dtype).view(1, 1, -1, 1)
            return a.o_proj((own.view(B, T, -1, hd) + w * shared.view(B, T, 1, hd)).flatten(2))
        return a.o_proj(own) + self.sh_gate[b].to(own.dtype) * self.sh_o(shared)

    def _run_block(self, b, h, pos, segments):
        block = self.blocks[b]
        h = h + self._attention(b, block.input_layernorm(h), pos, segments)
        return h + block.mlp(block.post_attention_layernorm(h))
