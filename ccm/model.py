"""Upstream Qwen3 with an explicit block-2 reader; no rewritten attention/MLP."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from .contracts import require, HOOKS, digest_json
from .keys import safe_gather


def pilot_config(path):
    c = Qwen3Config.from_pretrained(path, local_files_only=True)
    expected = dict(model_type="qwen3", vocab_size=151936, hidden_size=1024,
                    intermediate_size=3072, num_hidden_layers=28,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                    rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0.0,
                    tie_word_embeddings=True, hidden_act="silu")
    for field, value in expected.items():
        require(getattr(c, field) == value, f"Base config field mismatch: {field}")
    theta = getattr(c, "rope_theta", None)
    if theta is None:
        theta = c.rope_parameters["rope_theta"]
    require(theta == 1000000, "Unexpected Qwen RoPE theta")
    c.num_hidden_layers = 12
    c.max_window_layers = 12
    c.use_sliding_window = False
    c.layer_types = ["full_attention"]*12
    c.use_cache = False
    c._attn_implementation = "sdpa"
    return c


class Reader(nn.Module):
    def __init__(self, width, eps=1e-6, gate_width=128):
        super().__init__()
        self.norm_q = Qwen3RMSNorm(width, eps)
        self.norm_m = Qwen3RMSNorm(width, eps)
        self.wq = nn.Linear(width, gate_width, bias=False)
        self.wk = nn.Linear(width, gate_width, bias=False)
        self.wv = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.wq.weight, std=0.02)
        nn.init.normal_(self.wk.weight, std=0.02)
        nn.init.zeros_(self.wv.weight)

    def forward(self, r, m):
        mh = self.norm_m(m)
        q, k = self.wq(self.norm_q(r)), self.wk(mh)
        a = torch.sigmoid((q.float()*k.float()).sum(-1)/math.sqrt(q.shape[-1])+self.bias.float())
        contribution = a.unsqueeze(-1).to(r.dtype)*self.wv(mh).to(r.dtype)
        return contribution, a


class MemoryLM(nn.Module):
    def __init__(self, config, arm="base", table=None, slots=0, reader_seed=100017, table_seed=300017):
        super().__init__()
        require(config.num_hidden_layers >= 3, "Need a read block and deeper writer blocks")
        require(arm in ("base", "grad", "shallow", "contextual", "delta", "isolated", "shuffled"), "Unknown arm")
        self.backbone = Qwen3ForCausalLM(config)  # from config: NEVER pretrained weights
        self.arm = arm
        self.reader = None
        if arm != "base":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(reader_seed)
                self.reader = Reader(config.hidden_size, config.rms_norm_eps)
            if arm == "grad":
                require(slots > 0, "Grad requires a vocabulary capacity")
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(table_seed)
                    self.table = nn.Parameter(torch.empty(slots, config.hidden_size))
                    nn.init.normal_(self.table, std=0.02)
            else:
                require(table is not None and table.ndim == 2 and table.shape[1] == config.hidden_size,
                        "Frozen table shape mismatch")
                self.register_buffer("table", table.detach().clone().to(torch.bfloat16))
        self._slots = None
        self._capture = False
        self._states = {}
        self.gates = None
        self.backbone.model.layers[1].register_forward_hook(self._read_hook)
        self.backbone.model.norm.register_forward_pre_hook(self._writer_hook)

    def _read_hook(self, module, args, output):
        require(isinstance(output, torch.Tensor), "Unsupported upstream Qwen block API")
        if self._capture:
            self._states["r2_pre_memory"] = output
        u = output
        if self.reader is not None:
            require(self._slots is not None and self._slots.shape == output.shape[:2], "Missing/misaligned slot IDs")
            m = safe_gather(self.table, self._slots).to(output.dtype)
            contribution, a = self.reader(output, m)
            # Defense in depth: strict zero miss independent of reader numerics.
            contribution = contribution.masked_fill((self._slots < 0).unsqueeze(-1), 0)
            u = output+contribution
            self.gates = a.detach()
        if self._capture:
            self._states["u2_post_memory"] = u
        return u

    def _writer_hook(self, module, args):
        if self._capture:
            self._states["r12_pre_final_norm"] = args[0]

    def set_phase(self, phase):
        require(phase in ("common", "compile", "stage1", "stage2", "eval"), "Online self-writing is not implemented in pilot v1.")
        require(phase != "common" or self.arm == "base", "Common training must be memory-free")
        for p in self.backbone.parameters():
            p.requires_grad_(phase in ("common", "stage2"))
        if self.reader is not None:
            for p in self.reader.parameters():
                p.requires_grad_(phase in ("stage1", "stage2"))
            if self.arm == "grad":
                self.table.requires_grad_(phase in ("stage1", "stage2"))
        if phase in ("compile", "eval"):
            self.eval()

    def forward(self, input_ids, attention_mask, position_ids, slots=None, targets=None,
                capture=False, return_logits=False, loss_chunk=128):
        require(loss_chunk > 0, "Positive loss chunk required")
        self._slots, self._capture, self._states, self.gates = slots, capture, {}, None
        h = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, use_cache=False).last_hidden_state
        result = {"hidden": h}
        if capture:
            result.update(self._states)
        if return_logits:
            result["logits"] = self.backbone.lm_head(h)
        if targets is not None:
            require(targets.shape == input_ids.shape, "Targets must already align with hidden positions")
            # Recompute only head+CE chunks during backward, not Transformer blocks.
            # Avoid retaining a [all_tokens,151936] logits/CE graph in memory.
            flat, y = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
            parts = []
            def ce(x, label):
                logits = F.linear(x, self.backbone.lm_head.weight)
                return F.cross_entropy(logits.float(), label, reduction="none", ignore_index=-100)
            for start in range(0, len(y), loss_chunk):
                x, label = flat[start:start+loss_chunk], y[start:start+loss_chunk]
                if torch.is_grad_enabled():
                    parts.append(checkpoint(ce, x, label, use_reentrant=False))
                else:
                    parts.append(ce(x, label))
            losses = torch.cat(parts).reshape(targets.shape)
            result.update(losses=losses, loss_sum=losses.sum(), target_count=(targets != -100).sum())
        return result

    def backbone_contract(self):
        c = self.backbone.config
        fields = ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings",
                  "rms_norm_eps", "layer_types", "hidden_act", "attention_bias", "attention_dropout",
                  "max_position_embeddings", "initializer_range", "pad_token_id", "bos_token_id", "eos_token_id")
        contract = {k: getattr(c, k) for k in fields}
        for k in ("rope_parameters", "rope_theta", "rope_scaling", "sliding_window",
                  "use_sliding_window", "max_window_layers"):
            if hasattr(c, k):
                contract[k] = getattr(c, k)
        return dict(config=contract, hooks=HOOKS)
