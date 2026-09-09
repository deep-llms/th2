"""Qwen3-derived capacity-allocation models and vocabulary interfaces.

B0 remains the unmodified HF causal LM. Custom arms construct only their
declared interface/body shapes and retain Qwen3 attention, RoPE, normalization,
GQA and gated-MLP conventions unless an arm explicitly changes a boundary.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


ORIGINAL_ARMS = ("B0", "A128", "A256", "A512", "C", "D")
NEW_ARMS = (
    "T768", "T512", "P512-128-384", "A640", "A768", "A768-Direct",
    "FixedResidual", "WNW", "D-1024", "O1024-I256", "O1024-I232",
    "O1280", "C-Direct", "D-Direct",
)
ARMS = ORIGINAL_ARMS + NEW_ARMS
SHARED_ARMS = ("T768", "T512", "P512-128-384", "FixedResidual")
FULLY_TIED_ARMS = ("B0", "T768", "T512", "FixedResidual")

# Exact unique-parameter counts. Production verification fails closed if a
# supposedly frozen shape changes.
EXPECTED_COUNTS = dict(
    B0=249969152, A128=251017728, A256=251017728,
    A512=251017728, C=198485504, D=247117568,
    T768=212646400, T512=173226496, **{
        "P512-128-384": 251542016,
        "A640": 251017728,
        "A768": 251017728,
        "A768-Direct": 243939328,
        "FixedResidual": 217853440,
        "WNW": 238827008,
        "D-1024": 266729216,
        "O1024-I256": 253865472,
        "O1024-I232": 250206720,
        "O1280": 250627840,
        "C-Direct": 198813184,
        "D-Direct": 246855424,
    })


def intermediate_size(width):
    return 3 * width


def _scaled(value, divisor):
    """Scale production ranks for tiny topology tests without producing zero."""
    return max(1, int(round(value / divisor)))


class AllocationConfig(Qwen3Config):
    model_type = "capacity_allocation_qwen3"

    def __init__(self, widths=None, input_rank=None, output_rank=None,
                 reference_width=1024, scale_policy="fanin_scaled_head",
                 interface_type="independent", shared_rank=0,
                 input_private_rank=0, output_private_rank=0,
                 body_type="t1", compute_widths=None, skip_input_adapter=False,
                 skip_output_adapter=False, **kwargs):
        widths = list(widths or [1024] * 6)
        compute_widths = list(compute_widths or widths)
        head_dim = kwargs.get("head_dim", 128)
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer for rotary embeddings")
        if (not widths or len(compute_widths) != len(widths)
                or any(type(d) is not int or d <= 0 or d % head_dim
                       for d in widths + compute_widths)
                or type(reference_width) is not int or reference_width <= 0):
            raise ValueError("Positive widths divisible by head_dim are required")
        if scale_policy != "fanin_scaled_head":
            raise ValueError("Only the registered fan-in / scaled-head initialization is supported")
        if interface_type not in ("independent", "shared"):
            raise ValueError("Unknown vocabulary interface type")
        if body_type not in ("t1", "direct", "fixed_residual"):
            raise ValueError("Unknown body type")
        if body_type == "fixed_residual":
            if len(set(widths)) != 1 or any(d > widths[0] for d in compute_widths):
                raise ValueError("Fixed-residual arms require one outer width and no wider inner block")
        elif compute_widths != widths:
            raise ValueError("compute_widths differ from widths only for fixed-residual arms")
        if body_type == "direct" and any(b < a for a, b in zip(widths, widths[1:])):
            raise ValueError("Direct boundary implementation supports widening only")
        if type(skip_input_adapter) is not bool or type(skip_output_adapter) is not bool:
            raise ValueError("Adapter skip flags must be boolean")

        fields = (shared_rank, input_private_rank, output_private_rank)
        if any(type(x) is not int or x < 0 for x in fields):
            raise ValueError("Shared/private ranks must be nonnegative integers")
        if interface_type == "independent":
            input_rank = 128 if input_rank is None else input_rank
            output_rank = 896 if output_rank is None else output_rank
            if shared_rank or input_private_rank or output_private_rank:
                raise ValueError("Independent interfaces cannot declare shared/private ranks")
            tied = False
        else:
            if shared_rank <= 0:
                raise ValueError("Shared interfaces require a positive shared rank")
            derived_input = shared_rank + input_private_rank
            derived_output = shared_rank + output_private_rank
            if input_rank is not None and input_rank != derived_input:
                raise ValueError("input_rank disagrees with shared plus input-private rank")
            if output_rank is not None and output_rank != derived_output:
                raise ValueError("output_rank disagrees with shared plus output-private rank")
            input_rank, output_rank, tied = derived_input, derived_output, True
        if (type(input_rank) is not int or input_rank <= 0
                or type(output_rank) is not int or output_rank <= 0):
            raise ValueError("Positive integer input/output ranks are required")
        if ((skip_input_adapter and input_rank != widths[0]) or
                (skip_output_adapter and output_rank != widths[-1])):
            raise ValueError("Skipped vocabulary adapters require matching endpoint dimensions")
        if (skip_output_adapter and interface_type == "shared"
                and (output_private_rank or shared_rank != widths[-1])):
            raise ValueError(
                "A skipped shared-output adapter requires one full-width shared table"
            )
        requested_tying = kwargs.get("tie_word_embeddings", tied)
        if requested_tying != tied:
            raise ValueError("tie_word_embeddings disagrees with the declared interface")

        kwargs.update(hidden_size=widths[0], num_hidden_layers=len(widths),
                      intermediate_size=intermediate_size(widths[0]),
                      num_attention_heads=2 * widths[0] // head_dim,
                      num_key_value_heads=widths[0] // head_dim,
                      tie_word_embeddings=tied, head_dim=head_dim,
                      layer_types=["full_attention"] * len(widths),
                      use_sliding_window=False)
        super().__init__(**kwargs)
        self.widths = widths
        self.compute_widths = compute_widths
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.reference_width = reference_width
        self.scale_policy = scale_policy
        self.interface_type = interface_type
        self.shared_rank = shared_rank
        self.input_private_rank = input_private_rank
        self.output_private_rank = output_private_rank
        self.body_type = body_type
        self.skip_input_adapter = skip_input_adapter
        self.skip_output_adapter = skip_output_adapter


def _arm_spec(arm):
    uniform = [1024] * 6
    specs = {
        "A128": dict(widths=uniform, input_rank=128, output_rank=896),
        "A256": dict(widths=uniform, input_rank=256, output_rank=768),
        "A512": dict(widths=uniform, input_rank=512, output_rank=512),
        "C": dict(widths=[256, 256, 512, 512, 1024, 1024], input_rank=128, output_rank=896),
        "D": dict(widths=[512, 512, 1024, 1024, 1280, 1280], input_rank=128, output_rank=896),
        "T768": dict(widths=uniform, interface_type="shared", shared_rank=768),
        "T512": dict(widths=uniform, interface_type="shared", shared_rank=512),
        "P512-128-384": dict(widths=uniform, interface_type="shared", shared_rank=512,
                              input_private_rank=128, output_private_rank=384),
        "A640": dict(widths=uniform, input_rank=640, output_rank=384),
        "A768": dict(widths=uniform, input_rank=768, output_rank=256),
        "A768-Direct": dict(widths=[768] + [1024] * 5, input_rank=768,
                             output_rank=256, body_type="direct", skip_input_adapter=True),
        "FixedResidual": dict(widths=uniform, compute_widths=[512, 512, 768, 768, 1024, 1024],
                              interface_type="shared", shared_rank=1024,
                              body_type="fixed_residual", skip_input_adapter=True,
                              skip_output_adapter=True),
        "WNW": dict(widths=[1024, 1024, 768, 768, 1024, 1024],
                    input_rank=128, output_rank=896),
        "D-1024": dict(widths=[512, 512, 1024, 1024, 1280, 1280],
                       input_rank=128, output_rank=1024),
        "O1024-I256": dict(widths=[512, 512, 768, 768, 1024, 1024],
                           input_rank=256, output_rank=1024),
        "O1024-I232": dict(widths=[512, 512, 768, 768, 1024, 1024],
                           input_rank=232, output_rank=1024),
        "O1280": dict(widths=[256, 256, 512, 512, 768, 1280],
                      input_rank=64, output_rank=1280),
        "C-Direct": dict(widths=[256, 256, 512, 512, 1024, 1024],
                         input_rank=128, output_rank=896, body_type="direct"),
        "D-Direct": dict(widths=[512, 512, 1024, 1024, 1280, 1280],
                         input_rank=128, output_rank=896, body_type="direct"),
    }
    return copy.deepcopy(specs[arm])


def experiment_config(arm, *, depth=6, tiny=False, attention="sdpa"):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}; choose from {ARMS}")
    if depth not in (6, 12) or (arm not in ("B0", "A128") and depth != 6):
        raise ValueError("Only B0/A128 have a specified 12-layer confirmation")
    divisor = 32 if tiny else 1
    ref = 1024 // divisor
    common = dict(vocab_size=97 if tiny else 151936, hidden_size=ref,
                  intermediate_size=intermediate_size(ref), num_hidden_layers=depth,
                  num_attention_heads=16, num_key_value_heads=8,
                  head_dim=128 // divisor, hidden_act="silu", max_position_embeddings=40960,
                  rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
                  rms_norm_eps=1e-6, initializer_range=0.02, attention_bias=False,
                  mlp_bias=False, attention_dropout=0.0, tie_word_embeddings=arm == "B0",
                  bos_token_id=None, eos_token_id=96 if tiny else 151645,
                  pad_token_id=None if tiny else 151643, use_cache=False,
                  layer_types=["full_attention"] * depth, use_sliding_window=False,
                  experiment_arm=arm, tiny_test=tiny)
    if arm == "B0":
        config = Qwen3Config(**common)
    else:
        spec = _arm_spec(arm)
        common["tie_word_embeddings"] = spec.get("interface_type") == "shared"
        if depth == 12:
            spec["widths"] = [1024] * 12
        for key in ("widths", "compute_widths"):
            if key in spec:
                spec[key] = [_scaled(value, divisor) for value in spec[key]]
        for key in ("input_rank", "output_rank", "shared_rank",
                    "input_private_rank", "output_private_rank"):
            if key in spec:
                spec[key] = _scaled(spec[key], divisor) if spec[key] else 0
        config = AllocationConfig(reference_width=ref, **spec, **common)
    config._attn_implementation = attention
    return config


class FanInLinear(nn.Linear):
    """Marker for initialization at variance 1 / fan_in."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)


class ScaledVocabularyHead(nn.Linear):
    def __init__(self, rank, vocab_size, reference_width):
        super().__init__(rank, vocab_size, bias=False)
        self.reference_width = reference_width


class ScaledFanInLinear(FanInLinear):
    """Output adapter whose initialization preserves reference-width logit scale."""

    def __init__(self, in_features, out_features, gain):
        super().__init__(in_features, out_features)
        self.gain = gain


def _adapter(in_features, out_features, skip=False):
    if skip:
        if in_features != out_features:
            raise ValueError("An adapter can be skipped only when its dimensions match")
        return nn.Identity()
    return FanInLinear(in_features, out_features)


def _transition(in_features, out_features):
    return nn.Identity() if in_features == out_features else FanInLinear(in_features, out_features)


class InputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.input_rank)
        self.projection = _adapter(config.input_rank, config.widths[0],
                                   config.skip_input_adapter)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))


class OutputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = _adapter(config.widths[-1], config.output_rank,
                                   config.skip_output_adapter)
        self.head = ScaledVocabularyHead(config.output_rank, config.vocab_size,
                                         config.reference_width)

    @property
    def weight(self):
        return self.head.weight

    def forward(self, hidden_states):
        return self.head(self.projection(hidden_states))


class SharedInputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.shared_rank)
        self.input_private = (nn.Embedding(config.vocab_size, config.input_private_rank)
                              if config.input_private_rank else None)
        self.projection = _adapter(config.input_rank, config.widths[0],
                                   config.skip_input_adapter)

    @property
    def weight(self):
        return self.shared.weight

    def forward(self, input_ids):
        shared = self.shared(input_ids)
        values = (torch.cat((shared, self.input_private(input_ids)), dim=-1)
                  if self.input_private is not None else shared)
        return self.projection(values)


class SharedOutputInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        gain = math.sqrt(config.reference_width / config.output_rank)
        self.shared_projection = (nn.Identity() if config.skip_output_adapter else
                                  ScaledFanInLinear(config.widths[-1], config.shared_rank, gain))
        self.shared_head = nn.Linear(config.shared_rank, config.vocab_size, bias=False)
        self.output_private_projection = (ScaledFanInLinear(
                                            config.widths[-1], config.output_private_rank, gain)
                                          if config.output_private_rank else None)
        self.output_private_head = (nn.Linear(config.output_private_rank, config.vocab_size, bias=False)
                                    if config.output_private_rank else None)

    @property
    def weight(self):
        return self.shared_head.weight

    def forward(self, hidden_states):
        logits = self.shared_head(self.shared_projection(hidden_states))
        if self.output_private_head is not None:
            logits = logits + self.output_private_head(
                self.output_private_projection(hidden_states))
        return logits


class T1DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx, next_width):
        super().__init__(config, layer_idx)
        self.transition = _transition(config.hidden_size, next_width)

    def forward(self, hidden_states, **kwargs):
        return self.transition(super().forward(hidden_states, **kwargs))


class DirectWideningDecoderLayer(Qwen3DecoderLayer):
    """A Qwen block whose MLP produces the next, wider residual width."""

    def __init__(self, config, layer_idx, next_width):
        super().__init__(config, layer_idx)
        if next_width < config.hidden_size:
            raise ValueError("Direct boundary supports widening only")
        self.next_width = next_width
        if next_width != config.hidden_size:
            self.mlp.down_proj = nn.Linear(config.intermediate_size, next_width, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if self.next_width == self.hidden_size:
            return super().forward(hidden_states, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values,
                                   use_cache=use_cache, position_embeddings=position_embeddings,
                                   **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return F.pad(residual, (0, self.next_width - self.hidden_size)) + hidden_states


class FixedResidualDecoderLayer(Qwen3DecoderLayer):
    """Lift only a narrow Qwen block's residual delta into a fixed outer width."""

    def __init__(self, config, layer_idx, outer_width):
        super().__init__(config, layer_idx)
        self.outer_width = outer_width
        self.down = _transition(outer_width, config.hidden_size)
        self.up = _transition(config.hidden_size, outer_width)

    def forward(self, hidden_states, **kwargs):
        if self.hidden_size == self.outer_width:
            return super().forward(hidden_states, **kwargs)
        narrow = self.down(hidden_states)
        updated = super().forward(narrow, **kwargs)
        return hidden_states + self.up(updated - narrow)


class ScaleInitialization:
    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, ScaledFanInLinear):
            nn.init.normal_(module.weight, std=module.gain * module.in_features ** -0.5)
        elif isinstance(module, FanInLinear):
            nn.init.normal_(module.weight, std=module.in_features ** -0.5)
        elif isinstance(module, ScaledVocabularyHead):
            nn.init.normal_(module.weight, std=self.config.initializer_range *
                            math.sqrt(module.reference_width / module.in_features))
        else:
            super()._init_weights(module)


class AllocationBody(ScaleInitialization, Qwen3Model):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = (SharedInputInterface(config) if config.interface_type == "shared"
                             else InputInterface(config))
        self.layers = nn.ModuleList()
        for i, (width, compute_width) in enumerate(zip(config.widths, config.compute_widths)):
            layer_width = compute_width if config.body_type == "fixed_residual" else width
            layer_config = copy.deepcopy(config)
            layer_config.hidden_size = layer_width
            layer_config.intermediate_size = intermediate_size(layer_width)
            layer_config.num_attention_heads = 2 * layer_width // config.head_dim
            layer_config.num_key_value_heads = layer_width // config.head_dim
            next_width = config.widths[min(i + 1, len(config.widths) - 1)]
            if config.body_type == "fixed_residual":
                layer = FixedResidualDecoderLayer(layer_config, i, width)
            elif config.body_type == "direct":
                layer = DirectWideningDecoderLayer(layer_config, i, next_width)
            else:
                layer = T1DecoderLayer(layer_config, i, next_width)
            self.layers.append(layer)
        self.norm = Qwen3RMSNorm(config.widths[-1], eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = False


class AllocationForCausalLM(ScaleInitialization, Qwen3ForCausalLM):
    config_class = AllocationConfig
    _tied_weights_keys = {"lm_head.shared_head.weight": "model.embed_tokens.shared.weight"}
    _tp_plan = {}

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = AllocationBody(config)
        self.vocab_size = config.vocab_size
        self.lm_head = (SharedOutputInterface(config) if config.interface_type == "shared"
                        else OutputInterface(config))
        self.post_init()


AutoConfig.register(AllocationConfig.model_type, AllocationConfig)
AutoModelForCausalLM.register(AllocationConfig, AllocationForCausalLM)


def build_model(config):
    return (AllocationForCausalLM(config) if isinstance(config, AllocationConfig)
            else Qwen3ForCausalLM(config))


def _block_parameters(width, head_dim, output_width=None):
    output_width = width if output_width is None else output_width
    return 12 * width * width + 3 * width * output_width + 2 * width + 2 * head_dim


def parameter_report(model):
    config = model.config
    widths = getattr(config, "widths", [config.hidden_size] * config.num_hidden_layers)
    if isinstance(config, AllocationConfig):
        table_rank = ((config.shared_rank + config.input_private_rank + config.output_private_rank)
                      if config.interface_type == "shared"
                      else config.input_rank + config.output_rank)
        tables = config.vocab_size * table_rank
        adapters = ((0 if config.skip_input_adapter else config.input_rank * widths[0]) +
                    (0 if config.skip_output_adapter else widths[-1] * config.output_rank))
        transitions = 0
        blocks = 0
        if config.body_type == "t1":
            blocks = sum(_block_parameters(d, config.head_dim) for d in widths)
            transitions = sum(a*b for a, b in zip(widths, widths[1:]) if a != b)
        elif config.body_type == "direct":
            next_widths = widths[1:] + widths[-1:]
            blocks = sum(_block_parameters(d, config.head_dim, out)
                         for d, out in zip(widths, next_widths))
        else:
            for outer, inner in zip(widths, config.compute_widths):
                blocks += _block_parameters(inner, config.head_dim)
                if inner != outer:
                    transitions += 2 * outer * inner
        expected = tables + adapters + transitions + blocks + widths[-1]
    else:
        tables = config.vocab_size * config.hidden_size
        adapters = transitions = 0
        blocks = sum(_block_parameters(d, config.head_dim) for d in widths)
        expected = tables + blocks + widths[-1]
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
    if isinstance(embed, InputInterface):
        raw = embed.embedding(input_ids)
    elif isinstance(embed, SharedInputInterface):
        values = [embed.shared(input_ids)]
        if embed.input_private is not None:
            values.append(embed.input_private(input_ids))
        raw = torch.cat(values, dim=-1)
    else:
        raw = embed(input_ids)
    result["embedding_rms"] = raw.float().square().mean().sqrt().item()
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
