"""HF Qwen3 blocks with independent vocabulary interfaces and T1 transitions.

Validated against Transformers 5.9.0. B0 is the unmodified HF causal LM.
No pretrained model weights or unused full-width vocabulary tables are created.
"""
import copy
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


ARMS = ("B0", "A128", "A256", "A512", "C", "D")
EXPECTED_COUNTS = dict(B0=249969152, A128=251017728, A256=251017728,
                       A512=251017728, C=198485504, D=247117568)


def intermediate_size(width):
    return 3 * width


class AllocationConfig(Qwen3Config):
    model_type = "capacity_allocation_qwen3"

    def __init__(self, widths=None, input_rank=128, output_rank=896,
                 reference_width=1024, scale_policy="fanin_scaled_head", **kwargs):
        widths = list(widths or [1024] * 6)
        head_dim = kwargs.get("head_dim", 128)
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer for rotary embeddings")
        if (not widths or any(type(d) is not int or d <= 0 or d % head_dim for d in widths)
                or input_rank <= 0 or output_rank <= 0 or reference_width <= 0):
            raise ValueError("Positive ranks and widths divisible by head_dim are required")
        if scale_policy != "fanin_scaled_head":
            raise ValueError("Only the registered fan-in / scaled-head initialization is supported")
        if kwargs.get("tie_word_embeddings", False):
            raise ValueError("Allocation models require independent vocabulary tables")
        kwargs.update(hidden_size=widths[0], num_hidden_layers=len(widths),
                      intermediate_size=intermediate_size(widths[0]),
                      num_attention_heads=2 * widths[0] // head_dim,
                      num_key_value_heads=widths[0] // head_dim,
                      tie_word_embeddings=False, head_dim=head_dim,
                      layer_types=["full_attention"] * len(widths), use_sliding_window=False)
        super().__init__(**kwargs)
        self.widths = widths
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.reference_width = reference_width
        self.scale_policy = scale_policy


def experiment_config(arm, *, depth=6, tiny=False, attention="sdpa"):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}; choose from {ARMS}")
    if depth not in (6, 12) or (arm in ("C", "D") and depth != 6):
        raise ValueError("Only uniform arms have a specified 12-layer confirmation")
    # Tiny variants preserve proportions and topology, not production FFN rounding ratios.
    divisor = 32 if tiny else 1
    ref = 1024 // divisor
    widths = {"C": [256, 256, 512, 512, 1024, 1024],
              "D": [512, 512, 1024, 1024, 1280, 1280]}.get(arm, [1024] * depth)
    widths = [d // divisor for d in widths]
    common = dict(vocab_size=97 if tiny else 151936, hidden_size=ref,
                  intermediate_size=intermediate_size(ref), num_hidden_layers=depth,
                  num_attention_heads=16, num_key_value_heads=8,
                  head_dim=128 // divisor, hidden_act="silu", max_position_embeddings=40960,
                  rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
                  rms_norm_eps=1e-6, initializer_range=0.02, attention_bias=False,
                  mlp_bias=False, attention_dropout=0.0, tie_word_embeddings=arm == "B0",
                  bos_token_id=None, eos_token_id=96 if tiny else 151645,
                  pad_token_id=None if tiny else 151643, use_cache=False, layer_types=["full_attention"] * depth,
                  use_sliding_window=False, experiment_arm=arm, tiny_test=tiny)
    if arm == "B0":
        config = Qwen3Config(**common)
    else:
        rin = {"A256": 256, "A512": 512}.get(arm, 128) // divisor
        config = AllocationConfig(widths=widths, input_rank=rin, output_rank=ref-rin,
                                  reference_width=ref, **common)
    config._attn_implementation = attention
    return config


class FanInLinear(nn.Linear):
    """Marker for initialization at variance 1 / fan_in (including square adapters)."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)


class ScaledVocabularyHead(nn.Linear):
    def __init__(self, rank, vocab_size, reference_width):
        super().__init__(rank, vocab_size, bias=False)
        self.reference_width = reference_width


class InputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.input_rank)
        self.projection = FanInLinear(config.input_rank, config.widths[0])

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))


class OutputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = FanInLinear(config.widths[-1], config.output_rank)
        self.head = ScaledVocabularyHead(config.output_rank, config.vocab_size,
                                         config.reference_width)

    @property
    def weight(self):
        return self.head.weight

    def forward(self, hidden_states):
        return self.head(self.projection(hidden_states))


class T1DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx, next_width):
        super().__init__(config, layer_idx)
        self.transition = (FanInLinear(config.hidden_size, next_width)
                           if next_width != config.hidden_size else nn.Identity())

    def forward(self, hidden_states, **kwargs):
        # Parent is a GradientCheckpointingLayer: its __call__ checkpoints this
        # entire forward, including the boundary projection, when enabled.
        return self.transition(super().forward(hidden_states, **kwargs))


class ScaleInitialization:
    @torch.no_grad()
    def _init_weights(self, module):
        # HF dispatches initialization to nested PreTrainedModel instances too.
        # Both the body and outer LM must therefore own the same policy.
        if isinstance(module, FanInLinear):
            nn.init.normal_(module.weight, std=module.in_features ** -0.5)
        elif isinstance(module, ScaledVocabularyHead):
            nn.init.normal_(module.weight, std=self.config.initializer_range *
                            math.sqrt(module.reference_width / module.in_features))
        else:
            super()._init_weights(module)


class AllocationBody(ScaleInitialization, Qwen3Model):
    def __init__(self, config):
        # Reuse HF's forward, RoPE, causal masks and per-layer dynamic KV cache.
        # Construct only the required shapes, not a throwaway uniform Qwen3Model.
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = InputInterface(config)
        self.layers = nn.ModuleList()
        for i, width in enumerate(config.widths):
            layer_config = copy.deepcopy(config)
            layer_config.hidden_size = width
            layer_config.intermediate_size = intermediate_size(width)
            layer_config.num_attention_heads = 2 * width // config.head_dim
            layer_config.num_key_value_heads = width // config.head_dim
            next_width = config.widths[min(i + 1, len(config.widths) - 1)]
            self.layers.append(T1DecoderLayer(layer_config, i, next_width))
        self.norm = Qwen3RMSNorm(config.widths[-1], eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = False
        # The owning causal LM performs post_init once, with the custom policy.


class AllocationForCausalLM(ScaleInitialization, Qwen3ForCausalLM):
    config_class = AllocationConfig
    _tied_weights_keys = {}
    _tp_plan = {}  # Tensor parallelism is not validated; use ordinary DDP.

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = AllocationBody(config)
        self.vocab_size = config.vocab_size
        self.lm_head = OutputInterface(config)
        self.post_init()

AutoConfig.register(AllocationConfig.model_type, AllocationConfig)
AutoModelForCausalLM.register(AllocationConfig, AllocationForCausalLM)


def build_model(config):
    return (AllocationForCausalLM(config) if isinstance(config, AllocationConfig)
            else Qwen3ForCausalLM(config))


def parameter_report(model):
    config = model.config
    widths = getattr(config, "widths", [config.hidden_size] * config.num_hidden_layers)
    blocks = sum(6*d*d + 3*d*intermediate_size(d) + 2*d + 2*config.head_dim for d in widths)
    if isinstance(config, AllocationConfig):
        tables = config.vocab_size * (config.input_rank + config.output_rank)
        adapters = config.input_rank * widths[0] + widths[-1] * config.output_rank
        transitions = sum(a*b for a, b in zip(widths, widths[1:]) if a != b)
    else:
        tables = config.vocab_size * config.hidden_size
        adapters = transitions = 0
    expected = tables + adapters + transitions + blocks + widths[-1]
    actual = sum(p.numel() for p in model.parameters())
    if actual != expected:
        raise AssertionError(f"Parameter mismatch: instantiated={actual}, formula={expected}")
    return dict(total=actual, tables=tables, adapters=adapters, transitions=transitions,
                blocks=blocks, final_norm=widths[-1])


@torch.no_grad()
def activation_report(model, input_ids):
    """Fixed-batch scale diagnostics; restores train/eval mode and removes hooks."""
    result, hooks = {}, []
    def hook(name):
        def record(module, inputs, output):
            result[name + "/input_rms"] = inputs[0].float().square().mean().sqrt().item()
            result[name + "/output_rms"] = output.float().square().mean().sqrt().item()
        return record
    for name, module in model.named_modules():
        if isinstance(module, FanInLinear) or name == "model.norm":
            hooks.append(module.register_forward_hook(hook(name)))
    embed = model.get_input_embeddings()
    lookup = embed.embedding if isinstance(embed, InputInterface) else embed
    result["embedding_rms"] = lookup(input_ids).float().square().mean().sqrt().item()
    was_training = model.training
    try:
        model.eval()
        logits = model(input_ids, use_cache=False).logits.float()
        result["logit_std"] = logits.std().item()
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)
    return result
