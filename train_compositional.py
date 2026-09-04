"""Train a causal LM from scratch with compositional embeddings.

Adapted from train.py — same data pipeline, same backbone, same Trainer.
Only change: embed_tokens is replaced with a compositional module.

For Original ANT (which needs YOGI optimizer), use train_original_ant.py instead.

Data is loaded from per-language directories saved by prepare_data.py.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from itertools import chain

import torch
import torch.nn as nn

import datasets
from datasets import load_from_disk, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from compositional import (
    ANTEmbed, ResidualANTEmbed, V0Embed, V1Embed, V2Embed,
    IsolationControlEmbed, LowRankEmbed, SharedLocalEmbed, PureLocalEmbed,
    PVQEmbed, SlimEmbed, GroupReduceEmbed, NestedLadderEmbed, ProductCodeEmbed,
    ResidualSubspaceExpertsEmbed, TTEmbedding, RankLiftEmbed,
    TieredRankLiftEmbed, FunnelingEmbed, DeFINEEmbed,
)
from compositional.compressed_baselines import (
    balanced_exact_modes,
    balanced_padded_modes,
)
from compositional.compression_init import (
    allocate_frequency_proportional_ranks,
    file_sha256,
    frequency_group_ids,
    frequency_group_ids_from_populations,
    group_parameter_count,
    load_frequency_counts,
)
from compositional.losses import load_balance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments — same as train.py, plus CompositionalArguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization. Don't set if training from scratch."},
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name_or_path"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from huggingface.co"},
    )
    token: str | None = field(
        default=None,
        metadata={"help": "HF auth token for downloading gated models/tokenizers"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code from the Hub"},
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/sampled",
        metadata={"help": "Directory containing per-language raw text datasets (saved by prepare_data.py)"},
    )
    block_size: int | None = field(
        default=None,
        metadata={"help": "Optional input sequence length after tokenization. Defaults to model max length."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets"},
    )


@dataclass
class CompositionalArguments:
    arm: str = field(
        default="ant",
        metadata={
            "help": "Embedding arm to train.",
            "choices": ["ant", "residual_ant", "v0", "v1", "v2",
                        "isolation_control", "lowrank", "global_lowrank",
                        "shared_local", "pure_local", "pvq", "slim",
                        "groupreduce", "nested_ladder",
                        "residual_subspace_experts", "product_code",
                        "ranklift", "tiered_ranklift", "funneling",
                        "define", "tt"],
        },
    )
    K: int = field(default=4096, metadata={"help": "Codebook size (number of anchors)."})
    d_x: int = field(default=128, metadata={"help": "Base token table dimension."})
    shared_rank: int = field(default=64, metadata={
        "help": "Shared token-subspace rank for the shared_local arm."
    })
    local_embed_rank: int = field(default=64, metadata={
        "help": "Per-group token-subspace rank for the shared_local arm."
    })
    pure_local_rank: int = field(default=128, metadata={
        "help": "Per-group token-subspace rank for the pure_local arm."
    })
    num_groups: int = field(default=4, metadata={
        "help": "Number of contiguous vocabulary groups for shared_local or "
                "pure_local."
    })
    embedding_init_path: str | None = field(default=None, metadata={
        "help": "Optional strict embedding state used to initialize a published "
                "compressed baseline. P-VQ and faithful GroupReduce runs must "
                "be converted from a dense checkpoint instead of silently "
                "using random factors."
    })
    allow_from_scratch_baseline_init: bool = field(default=False, metadata={
        "help": "Explicitly allow random/static initialization for P-VQ or "
                "GroupReduce. Such runs are adaptations, not reproductions of "
                "the papers' post-hoc training procedures."
    })
    pvq_shared_dim: int = field(default=768, metadata={
        "help": "P-VQ shared/codebook window width w."
    })
    pvq_num_codes: int = field(default=128, metadata={
        "help": "P-VQ final codebook size K."
    })
    pvq_assignment_seed: int = field(default=42, metadata={
        "help": "Seed used only for the explicitly allowed random-code P-VQ adaptation."
    })
    slim_num_components: int = field(default=8, metadata={
        "help": "Number K of coordinate subvectors per Slim embedding."
    })
    slim_num_subvectors: int = field(default=0, metadata={
        "help": "Total learned Slim subvectors M; 0 means vocab_size (1/8 "
                "float-parameter ratio when K=8)."
    })
    slim_mapping_seed: int = field(default=42, metadata={
        "help": "Seed for Slim's fixed balanced-random mapping table."
    })
    groupreduce_num_groups: int = field(default=4, metadata={
        "help": "Number of GroupReduce vocabulary blocks."
    })
    groupreduce_ranks: str = field(default="", metadata={
        "help": "Comma-separated rank per GroupReduce block. Empty uses d_x "
                "for every block."
    })
    groupreduce_populations: str = field(default="", metadata={
        "help": "Optional comma-separated frequency-block populations. Their "
                "sum must equal the vocabulary size; group 0 receives the "
                "highest-frequency block. Empty uses equal-sized blocks."
    })
    groupreduce_frequency_path: str | None = field(default=None, metadata={
        "help": "Token-frequency .npz/.pt used to construct the static "
                "frequency bins for a GroupReduce end-to-end adaptation."
    })
    groupreduce_frequency_key: str = field(default="counts")
    groupreduce_frequency_pseudocount: float = field(default=1.0)
    groupreduce_target_params: int = field(default=0, metadata={
        "help": "Resolve GroupReduce's proportional ranks to this trainable "
                "embedding budget when explicit ranks are omitted."
    })
    nested_tier_ranks: str = field(default="64,128,320,512", metadata={
        "help": "Comma-separated additive rank for each Nested Ladder tier."
    })
    nested_tier_populations: str = field(
        default="151936,32768,8192,2048",
        metadata={
            "help": "Comma-separated nested tier populations; the first must "
                    "equal the model vocabulary size."
        },
    )
    nested_frequency_path: str | None = field(
        default="resources/token_freq_sample10.npz",
        metadata={
            "help": "Token-frequency .npz/.pt used only to build fresh static "
                    "Nested Ladder memberships. Checkpoints store membership."
        },
    )
    nested_frequency_key: str = field(default="counts")
    rse_base_rank: int = field(default=120, metadata={
        "help": "Global token-factor rank for residual_subspace_experts."
    })
    rse_expert_rank: int = field(default=80, metadata={
        "help": "Bottleneck rank of each residual subspace expert."
    })
    rse_num_experts: int = field(default=12, metadata={
        "help": "Number of residual subspace experts."
    })
    rse_router_dim: int = field(default=32, metadata={
        "help": "Cosine-router query/key dimension."
    })
    rse_top_k: int = field(default=2, metadata={
        "help": "Number of residual experts selected per token."
    })
    rse_router_temperature: float = field(default=1.0, metadata={
        "help": "Positive softmax temperature within the selected experts."
    })
    product_code_head_size: int = field(default=2048, metadata={
        "help": "Number of highest-importance tokens that keep private dense rows."
    })
    product_code_num_hashes: int = field(default=4, metadata={
        "help": "Number of codebooks (code coordinates) per tail token."
    })
    product_code_num_buckets: int = field(default=4096, metadata={
        "help": "Rows per codebook."
    })
    product_code_assignment: str = field(default="hashed", metadata={
        "help": "Tail code assignment: 'hashed' (deterministic keyed hash, pure "
                "from-scratch) or 'pq' (codes from scripts/make_pq_codes.py; "
                "post-hoc-informed, report as such).",
        "choices": ["hashed", "pq"],
    })
    product_code_codes_path: str | None = field(default=None, metadata={
        "help": "Codes artifact (.pt with 'codes' and 'tail_ids') for assignment=pq."
    })
    product_code_importance_path: str | None = field(
        default="resources/token_importance_langbalanced.npz",
        metadata={"help": "Token importance vector used only to select the dense "
                          "head for a fresh run; checkpoints store the partition."},
    )
    product_code_importance_key: str = field(default="counts")
    product_code_seed: int = field(default=0, metadata={
        "help": "Seed for the keyed hash (hashed assignment only)."
    })
    ranklift_code_dim: int = field(default=124, metadata={
        "help": "Private per-token code width for RankLift."
    })
    ranklift_lift_dim: int = field(default=336, metadata={
        "help": "Nonlinear shared feature width added by RankLift."
    })
    ranklift_rms_eps: float = field(default=1e-6, metadata={
        "help": "Positive epsilon for RankLift's parameter-free RMSNorm."
    })
    tiered_ranklift_code_dims: str = field(
        default="1024,512,192,64",
        metadata={
            "help": "Comma-separated private token-code widths, one per "
                    "Tiered RankLift vocabulary group."
        },
    )
    tiered_ranklift_lift_dims: str = field(
        default="0,0,320,192",
        metadata={
            "help": "Comma-separated nonlinear lift widths, one per Tiered "
                    "RankLift group; zero makes that group ordinary linear "
                    "GroupReduce."
        },
    )
    tiered_ranklift_populations: str = field(
        default="2048,6144,24576,119168",
        metadata={
            "help": "Comma-separated disjoint vocabulary-group populations "
                    "for Tiered RankLift, highest importance first."
        },
    )
    tiered_ranklift_frequency_path: str | None = field(
        default="resources/token_importance_langbalanced.npz",
        metadata={
            "help": "Static token-importance artifact used to build fresh "
                    "Tiered RankLift memberships; checkpoints store membership."
        },
    )
    tiered_ranklift_frequency_key: str = field(default="counts")
    tiered_ranklift_rms_eps: float = field(default=1e-6, metadata={
        "help": "Positive epsilon for Tiered RankLift's parameter-free RMSNorm."
    })
    funneling_rank: int = field(default=128, metadata={
        "help": "Bottleneck width of the from-scratch Funneling control."
    })
    define_code_dim: int = field(default=112, metadata={
        "help": "Low-dimensional map and tied classifier width for DeFINE."
    })
    define_expansion_dims: str = field(default="656,1184,1724", metadata={
        "help": "Comma-separated DeFINE HGT expansion widths."
    })
    define_group_counts: str = field(default="16,8,4", metadata={
        "help": "Comma-separated DeFINE group counts, one per expansion."
    })
    tt_order: int = field(default=3, metadata={
        "help": "TT matrix order used when explicit shapes are omitted."
    })
    tt_vocab_shape: str = field(default="", metadata={
        "help": "Comma-separated TT vocabulary modes; product may pad V."
    })
    tt_embedding_shape: str = field(default="", metadata={
        "help": "Comma-separated TT hidden modes; product must equal hidden size."
    })
    tt_rank: int = field(default=128, metadata={
        "help": "Uniform internal TT rank when tt_ranks is empty."
    })
    tt_ranks: str = field(default="", metadata={
        "help": "Optional comma-separated internal TT ranks."
    })
    tt_target_std: float = field(default=0.0, metadata={
        "help": "Target standard deviation of the effective TT table. 0 uses "
                "the paper's modified Glorot value; set 0.02 only for an "
                "explicit Qwen-initialization ablation."
    })
    tt_implementation: str = field(default="materialize", metadata={
        "help": "TT execution path. 'materialize' matches the released paper "
                "implementation; 'direct' is an optimized tied adaptation.",
        "choices": ["materialize", "direct"],
    })
    tt_materialize_chunk_size: int = field(default=1024, metadata={
        "help": "Rows per activation-checkpointed TT table chunk."
    })
    mos_components: int = field(default=1, metadata={
        "help": "Number of tied Mixture-of-Softmaxes output components. "
                "One disables MoS and preserves the existing tied head."
    })
    mos_context_rank: int = field(default=256, metadata={
        "help": "Bottleneck rank of each non-identity MoS context map."
    })
    mos_chunk_size: int = field(default=2048, metadata={
        "help": "Hidden positions per activation-checkpointed MoS chunk."
    })
    d_k: int = field(default=64, metadata={"help": "Router key dimension."})
    gamma: float = field(default=1.0, metadata={"help": "Score temperature for entmax."})
    num_heads: int = field(default=1, metadata={"help": "Number of selection heads (ANT/V2 only)."})
    max_k: int = field(default=16, metadata={"help": "Max anchors per token (V0/V1 only)."})
    v0_mode: str = field(default="post", metadata={"help": "V0 beta mode.", "choices": ["post", "pre"]})
    v1_query: str = field(default="content", metadata={"help": "V1 query.", "choices": ["content", "cls"]})
    localenc: str = field(default="attn", metadata={"help": "V2 LocalEnc.", "choices": ["attn", "conv", "conv_lite"]})
    lambda_div: float = field(default=0.0, metadata={"help": "Load-balance loss weight."})
    tie_output: bool = field(default=False, metadata={
        "help": "Tie the output lm_head to the input embedding weights. "
                "Removes the free V×d lm_head (~155.6M params) and computes "
                "output logits from the embedding module's own weights. "
                "Supported for lowrank/global_lowrank, shared_local, pure_local, "
                "pvq, slim, groupreduce, nested_ladder, "
                "residual_subspace_experts, product_code, ranklift, "
                "tiered_ranklift, "
                "funneling, define, tt, "
                "original_ant, ant, and residual_ant; not supported "
                "for context-dependent arms (v0, v1, v2, isolation_control).",
    })
    independent_lowrank_output: bool = field(default=False, metadata={
        "help": "Use a separately parameterized rank-d_x output head, initialized "
                "as an exact factor copy of a lowrank input embedding. This is a "
                "diagnostic control for output rank versus hard weight sharing. "
                "Only supported with --arm lowrank and mutually exclusive with "
                "--tie_output.",
    })


# ---------------------------------------------------------------------------
# EmbeddingShim
# ---------------------------------------------------------------------------

class EmbeddingShim(nn.Module):
    """Wraps compositional embedding as model.model.embed_tokens."""

    def __init__(self, embed_module):
        super().__init__()
        self.embed = embed_module
        self._last_theta = None

    def forward(self, input_ids):
        e, theta = self.embed(input_ids)
        self._last_theta = theta
        return e


# ---------------------------------------------------------------------------
# CompositionalTrainer
# ---------------------------------------------------------------------------

def _matches_declared_no_decay(name, declared):
    """True when ``name`` is one of the embedding's declared no-decay params.

    Names are matched by suffix so wrapper prefixes (``model.embed_tokens.embed.``,
    DDP's ``module.``) do not matter.
    """
    return any(name == leaf or name.endswith("." + leaf) for leaf in declared)


class CompositionalTrainer(Trainer):

    def __init__(self, *args, embed_shim=None, comp_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_shim = embed_shim
        self.comp_args = comp_args
        self._comp_sums = {}
        self._comp_counts = {}
        self._device_metric_sums = {}
        self._device_metric_counts = {}

    def get_decay_parameter_names(self, model):
        """Honor an embedding module's ``no_decay_parameters()`` declaration.

        Mirrors the ``pop_step_metrics`` duck-typed protocol: a module that
        owns parameters which must not be decayed (ProductCodeEmbed's
        zero-initialized gate offsets) declares them itself; the trainer stays
        arm-agnostic.
        """
        names = super().get_decay_parameter_names(model)
        declare = getattr(self.embed_shim.embed, "no_decay_parameters", None)
        declared = tuple(declare()) if callable(declare) else ()
        if not declared:
            return names
        return [
            name for name in names
            if not _matches_declared_no_decay(name, declared)
        ]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        # Forward num_items_in_batch so the model normalizes the summed CE across
        # gradient-accumulation steps — without it, logged loss (and gradients)
        # are scaled by gradient_accumulation_steps.
        loss_kwargs = {}
        if kwargs.get("num_items_in_batch") is not None:
            loss_kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"]
        outputs = model(input_ids=input_ids, labels=inputs.get("labels", input_ids),
                        **loss_kwargs)
        lm_loss = outputs.loss
        # num_items_in_batch is the GLOBAL token count (all-reduced) when
        # average_tokens_across_devices is on, while each rank's CE sum covers
        # local tokens only; DDP then averages gradients across ranks. Scale by
        # num_processes to restore sum semantics — mirrors HF's default
        # compute_loss exactly (verified against baseline logging).
        if loss_kwargs and getattr(self.args, "average_tokens_across_devices", False):
            lm_loss = lm_loss * self.accelerator.num_processes

        theta = self.embed_shim._last_theta
        pop_router_aux = getattr(
            self.embed_shim.embed, "pop_router_aux_loss", None
        )
        router_aux_loss = (
            pop_router_aux() if callable(pop_router_aux) else None
        )

        with torch.no_grad():
            if theta is not None:
                active = (theta > 0).float()
                usage = active.mean(dim=(0, 1))
                avg_nnz = active.sum(-1).mean()
                dead_rate = (usage == 0).float().mean()
                p = theta.clamp_min(1e-9)
                entropy = -(p * p.log()).sum(-1).mean()
            else:
                zero = torch.tensor(0.0, device=lm_loss.device)
                avg_nnz = dead_rate = entropy = zero

        total_loss = lm_loss
        div_loss_val = 0.0
        if theta is not None and self.comp_args.lambda_div > 0:
            div_loss = (
                router_aux_loss
                if router_aux_loss is not None
                else load_balance(theta)
            )
            # With num_items_in_batch normalization the micro-batch losses SUM to
            # the true batch loss, so the per-micro-batch div term must be scaled
            # by 1/accum to keep its effective weight at lambda_div.
            div_scale = (1.0 / self.args.gradient_accumulation_steps
                         if loss_kwargs else 1.0)
            total_loss = lm_loss + self.comp_args.lambda_div * div_scale * div_loss
            div_loss_val = div_loss.detach().item()

        for k, v in [("avg_nnz", avg_nnz.item()), ("dead_rate", dead_rate.item()),
                     ("entropy", entropy.item()), ("div_loss", div_loss_val)]:
            self._comp_sums[k] = self._comp_sums.get(k, 0.0) + v
            self._comp_counts[k] = self._comp_counts.get(k, 0) + 1

        metric_sources = [self.embed_shim.embed]
        output_head = getattr(self.model, "lm_head", None)
        if output_head is not None:
            metric_sources.append(output_head)
        for source in metric_sources:
            pop_metrics = getattr(source, "pop_step_metrics", None)
            if not callable(pop_metrics):
                continue
            for key, (metric_sum, metric_count) in (pop_metrics() or {}).items():
                if key in self._device_metric_sums:
                    self._device_metric_sums[key].add_(metric_sum.detach())
                    self._device_metric_counts[key].add_(metric_count.detach())
                else:
                    self._device_metric_sums[key] = metric_sum.detach().clone()
                    self._device_metric_counts[key] = metric_count.detach().clone()

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if self._comp_counts:
            for k, v in self._comp_sums.items():
                logs[k] = v / self._comp_counts[k]
            lm_loss = logs.get("loss", 0.0)
            if lm_loss > 0:
                logs["perplexity"] = math.exp(min(lm_loss, 20))
            self._comp_sums = {}
            self._comp_counts = {}
        if self._device_metric_sums:
            keys = sorted(self._device_metric_sums)
            sums = torch.stack([
                self._device_metric_sums[key] for key in keys
            ]).cpu().tolist()
            counts = torch.stack([
                self._device_metric_counts[key] for key in keys
            ]).cpu().tolist()
            for key, metric_sum, metric_count in zip(keys, sums, counts):
                if metric_count > 0:
                    logs[key] = metric_sum / metric_count
            self._device_metric_sums = {}
            self._device_metric_counts = {}
        super().log(logs, *args, **kwargs)


class SaveEmbeddingCallback(TrainerCallback):
    def __init__(self, embed_shim, independent_output_head=None):
        self.embed_shim = embed_shim
        self.independent_output_head = independent_output_head

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            torch.save(
                self.embed_shim.embed.state_dict(),
                os.path.join(checkpoint_dir, "embedding.pt"),
            )
            if self.independent_output_head is not None:
                from compositional.tied_head import INDEPENDENT_OUTPUT_FILENAME
                torch.save(
                    self.independent_output_head.state_dict(),
                    os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_train_config(save_dir, model_args, data_args, training_args, comp_args):
    config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": {
            k: v for k, v in training_args.to_dict().items()
            if v is not None and v != "" and k not in ("_n_gpu", "local_rank")
        },
        "compositional": asdict(comp_args),
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def _parse_int_list(value, *, name):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    try:
        return tuple(int(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error


def _load_embedding_state(path):
    """Load a strict plain embedding state dictionary."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Embedding initialization not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Embedding initialization is not a non-empty state dict: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError(
            "Embedding initialization must be a plain string-to-tensor state dict"
        )
    return state


def build_arm(comp_args, vocab_size, embed_dim, initial_state=None):
    ca = comp_args
    shared = dict(d_x=ca.d_x, d_k=ca.d_k, gamma=ca.gamma)

    if ca.arm in ("lowrank", "global_lowrank"):
        return LowRankEmbed(vocab_size, embed_dim, rank=ca.d_x)
    if ca.arm == "shared_local":
        return SharedLocalEmbed(
            vocab_size, embed_dim,
            shared_rank=ca.shared_rank,
            local_rank=ca.local_embed_rank,
            num_groups=ca.num_groups,
        )
    if ca.arm == "pure_local":
        return PureLocalEmbed(
            vocab_size,
            embed_dim,
            rank=ca.pure_local_rank,
            num_groups=ca.num_groups,
        )
    if ca.arm == "pvq":
        assignments = None
        if initial_state is not None:
            assignments = initial_state.get("assignments")
            expected = {
                "codebook": (ca.pvq_num_codes, ca.pvq_shared_dim),
                "exclusive": (
                    vocab_size, embed_dim - ca.pvq_shared_dim
                ),
            }
            for key, shape in expected.items():
                if key not in initial_state or tuple(initial_state[key].shape) != shape:
                    actual = None if key not in initial_state else tuple(initial_state[key].shape)
                    raise ValueError(
                        f"P-VQ initialization {key} has shape {actual}; expected {shape}"
                    )
        return PVQEmbed(
            vocab_size,
            embed_dim,
            shared_dim=ca.pvq_shared_dim,
            num_codes=ca.pvq_num_codes,
            assignments=assignments,
            assignment_seed=ca.pvq_assignment_seed,
        )
    if ca.arm == "slim":
        mapping = None if initial_state is None else initial_state.get("mapping")
        return SlimEmbed(
            vocab_size,
            embed_dim,
            num_components=ca.slim_num_components,
            num_subvectors=(ca.slim_num_subvectors or vocab_size),
            mapping=mapping,
            mapping_seed=ca.slim_mapping_seed,
        )
    if ca.arm == "groupreduce":
        if initial_state is not None:
            group_ids = initial_state.get("group_ids")
            ranks = []
            group = 0
            while f"left_factors.{group}" in initial_state:
                ranks.append(initial_state[f"left_factors.{group}"].shape[1])
                group += 1
            if not ranks:
                raise ValueError(
                    "GroupReduce initialization has no left_factors.* tensors"
                )
            declared = _parse_int_list(
                ca.groupreduce_ranks, name="groupreduce_ranks"
            )
            if declared and tuple(ranks) != declared:
                raise ValueError(
                    f"GroupReduce initialization ranks {tuple(ranks)} do not "
                    f"match --groupreduce_ranks {declared}"
                )
            if ca.groupreduce_num_groups != len(ranks):
                raise ValueError(
                    f"GroupReduce initialization has {len(ranks)} groups but "
                    f"--groupreduce_num_groups={ca.groupreduce_num_groups}"
                )
            declared_populations = _parse_int_list(
                ca.groupreduce_populations, name="groupreduce_populations"
            )
            if declared_populations:
                actual_populations = tuple(
                    torch.bincount(
                        group_ids, minlength=len(ranks)
                    ).tolist()
                )
                if actual_populations != declared_populations:
                    raise ValueError(
                        "GroupReduce initialization populations "
                        f"{actual_populations} do not match "
                        f"--groupreduce_populations {declared_populations}"
                    )
        else:
            ranks = _parse_int_list(
                ca.groupreduce_ranks, name="groupreduce_ranks"
            )
            populations = _parse_int_list(
                ca.groupreduce_populations, name="groupreduce_populations"
            )
            if ca.groupreduce_frequency_path:
                # Explicit populations are a structural frequency ranking,
                # not weighted-SVD input. Preserve raw zero/one counts so its
                # (-count, token_id) order exactly matches Nested Ladder.
                grouping_pseudocount = (
                    0.0 if populations
                    else ca.groupreduce_frequency_pseudocount
                )
                counts = load_frequency_counts(
                    ca.groupreduce_frequency_path,
                    vocab_size,
                    key=ca.groupreduce_frequency_key,
                    pseudocount=grouping_pseudocount,
                )
                if populations:
                    if len(populations) != ca.groupreduce_num_groups:
                        raise ValueError(
                            "--groupreduce_populations must have one entry per group"
                        )
                    group_ids = frequency_group_ids_from_populations(
                        counts, populations
                    )
                    logger.info(
                        "GroupReduce explicit-population frequency artifact: "
                        "%s (sha256=%s)",
                        ca.groupreduce_frequency_path,
                        file_sha256(ca.groupreduce_frequency_path),
                    )
                else:
                    group_ids = frequency_group_ids(
                        counts, ca.groupreduce_num_groups
                    )
                if not ranks:
                    if ca.groupreduce_target_params <= 0:
                        raise ValueError(
                            "GroupReduce frequency allocation requires "
                            "--groupreduce_target_params or explicit ranks"
                        )
                    ranks = allocate_frequency_proportional_ranks(
                        counts,
                        group_ids,
                        embed_dim,
                        ca.groupreduce_target_params,
                    )
            else:
                if populations:
                    raise ValueError(
                        "--groupreduce_populations requires "
                        "--groupreduce_frequency_path"
                    )
                group_ids = None
                ranks = ranks or (ca.d_x,) * ca.groupreduce_num_groups
            if len(ranks) != ca.groupreduce_num_groups:
                raise ValueError(
                    "--groupreduce_ranks must have one entry per group"
                )
            if populations:
                logger.info(
                    "GroupReduce explicit profile: populations=%s ranks=%s "
                    "parameters=%d",
                    populations,
                    tuple(ranks),
                    group_parameter_count(group_ids, ranks, embed_dim),
                )
        return GroupReduceEmbed(
            vocab_size, embed_dim, group_ranks=ranks, group_ids=group_ids
        )
    if ca.arm == "nested_ladder":
        ranks = _parse_int_list(
            ca.nested_tier_ranks, name="nested_tier_ranks"
        )
        populations = _parse_int_list(
            ca.nested_tier_populations,
            name="nested_tier_populations",
        )
        member_ids = None
        counts = None
        if initial_state is not None:
            structure = NestedLadderEmbed.structure_from_state(initial_state)
            if (structure["vocab_size"] != vocab_size
                    or structure["embed_dim"] != embed_dim):
                raise ValueError(
                    "Nested Ladder checkpoint model dimensions do not match "
                    f"the requested model (state V={structure['vocab_size']}, "
                    f"d={structure['embed_dim']}; requested V={vocab_size}, "
                    f"d={embed_dim})"
                )
            state_ranks = structure["tier_ranks"]
            state_populations = structure["tier_populations"]
            if state_ranks != ranks or state_populations != populations:
                raise ValueError(
                    "Nested Ladder checkpoint structure "
                    f"(ranks={state_ranks}, populations={state_populations}) "
                    "does not match the requested structure "
                    f"(ranks={ranks}, populations={populations})"
                )
            member_ids = structure["member_ids"]
        else:
            if not ca.nested_frequency_path:
                raise ValueError(
                    "A fresh Nested Ladder run requires "
                    "--nested_frequency_path"
                )
            counts = load_frequency_counts(
                ca.nested_frequency_path,
                vocab_size,
                key=ca.nested_frequency_key,
                pseudocount=0.0,
            )
            logger.info(
                "Nested Ladder frequency artifact: %s (sha256=%s)",
                ca.nested_frequency_path,
                file_sha256(ca.nested_frequency_path),
            )
        return NestedLadderEmbed(
            vocab_size,
            embed_dim,
            tier_ranks=ranks,
            tier_populations=populations,
            counts=counts,
            member_ids=member_ids,
        )
    if ca.arm == "residual_subspace_experts":
        if initial_state is not None:
            structure = ResidualSubspaceExpertsEmbed.structure_from_state(
                initial_state
            )
            requested = {
                "vocab_size": int(vocab_size),
                "embed_dim": int(embed_dim),
                "base_rank": ca.rse_base_rank,
                "expert_rank": ca.rse_expert_rank,
                "num_experts": ca.rse_num_experts,
                "router_dim": ca.rse_router_dim,
                "top_k": ca.rse_top_k,
                "router_temperature": ca.rse_router_temperature,
            }
            if structure != requested:
                raise ValueError(
                    "Residual-expert checkpoint structure does not match the "
                    f"requested structure (state={structure}, "
                    f"requested={requested})"
                )
        return ResidualSubspaceExpertsEmbed(
            vocab_size,
            embed_dim,
            base_rank=ca.rse_base_rank,
            expert_rank=ca.rse_expert_rank,
            num_experts=ca.rse_num_experts,
            router_dim=ca.rse_router_dim,
            top_k=ca.rse_top_k,
            router_temperature=ca.rse_router_temperature,
        )
    if ca.arm == "product_code":
        head_ids = None
        codes = None
        importance = None
        assignment = ca.product_code_assignment
        if initial_state is not None:
            structure = ProductCodeEmbed.structure_from_state(initial_state)
            if structure["vocab_size"] != vocab_size or structure["embed_dim"] != embed_dim:
                raise ValueError(
                    "Product Code checkpoint dimensions (vocab_size="
                    f"{structure['vocab_size']}, embed_dim={structure['embed_dim']}) "
                    f"do not match the model ({vocab_size}, {embed_dim})"
                )
            requested = (
                ca.product_code_head_size,
                ca.product_code_num_hashes,
                ca.product_code_num_buckets,
            )
            found = (
                structure["head_size"],
                structure["num_hashes"],
                structure["num_buckets"],
            )
            if requested != found:
                raise ValueError(
                    "Product Code checkpoint structure (head_size, num_hashes, "
                    f"num_buckets)={found} does not match the requested "
                    f"{requested}"
                )
            head_ids = structure["head_ids"]
            codes = structure["codes"]
            assignment = "checkpoint"
        else:
            if not ca.product_code_importance_path:
                raise ValueError(
                    "A fresh Product Code run requires "
                    "--product_code_importance_path"
                )
            importance = load_frequency_counts(
                ca.product_code_importance_path,
                vocab_size,
                key=ca.product_code_importance_key,
                pseudocount=0.0,
            )
            logger.info(
                "Product Code importance artifact: %s (sha256=%s)",
                ca.product_code_importance_path,
                file_sha256(ca.product_code_importance_path),
            )
            if assignment == "pq":
                if not ca.product_code_codes_path:
                    raise ValueError(
                        "--product_code_assignment pq requires "
                        "--product_code_codes_path"
                    )
                artifact = torch.load(
                    ca.product_code_codes_path, map_location="cpu",
                    weights_only=True,
                )
                codes = artifact["codes"]
                provenance = artifact.get("provenance")
                if not isinstance(provenance, dict):
                    raise ValueError(
                        "PQ codes artifact has no provenance dict; regenerate it "
                        "with scripts/make_pq_codes.py"
                    )
                for key, requested in (
                    ("head_size", ca.product_code_head_size),
                    ("num_hashes", ca.product_code_num_hashes),
                    ("num_buckets", ca.product_code_num_buckets),
                ):
                    if key not in provenance:
                        raise ValueError(f"PQ codes artifact provenance lacks {key!r}")
                    if int(provenance[key]) != int(requested):
                        raise ValueError(
                            f"PQ codes artifact was built with {key}="
                            f"{provenance[key]} but the run requests {requested}"
                        )
                logger.info(
                    "Product Code PQ codes artifact: %s (sha256=%s)",
                    ca.product_code_codes_path,
                    file_sha256(ca.product_code_codes_path),
                )
            elif assignment != "hashed":
                raise ValueError(f"unknown --product_code_assignment {assignment!r}")
        module = ProductCodeEmbed(
            vocab_size,
            embed_dim,
            ca.product_code_head_size,
            ca.product_code_num_hashes,
            ca.product_code_num_buckets,
            importance=importance,
            head_ids=head_ids,
            codes=codes,
            assignment=assignment,
            seed=ca.product_code_seed,
        )
        if initial_state is None and assignment == "pq":
            # The artifact's partition must be the one the module derived from
            # the importance file, otherwise codes would be applied to the
            # wrong tokens.
            if not torch.equal(
                torch.as_tensor(artifact["tail_ids"], dtype=torch.long),
                module.tail_ids,
            ):
                raise ValueError(
                    "PQ codes artifact tail ids do not match the head "
                    "selection implied by the importance file and "
                    "--product_code_head_size"
                )
        return module
    if ca.arm == "ranklift":
        return RankLiftEmbed(
            vocab_size,
            embed_dim,
            code_dim=ca.ranklift_code_dim,
            lift_dim=ca.ranklift_lift_dim,
            rms_eps=ca.ranklift_rms_eps,
        )
    if ca.arm == "tiered_ranklift":
        declared_code_dims = _parse_int_list(
            ca.tiered_ranklift_code_dims,
            name="tiered_ranklift_code_dims",
        )
        declared_lift_dims = _parse_int_list(
            ca.tiered_ranklift_lift_dims,
            name="tiered_ranklift_lift_dims",
        )
        declared_populations = _parse_int_list(
            ca.tiered_ranklift_populations,
            name="tiered_ranklift_populations",
        )
        if initial_state is not None:
            structure = TieredRankLiftEmbed.structure_from_state(initial_state)
            if (structure["vocab_size"] != vocab_size
                    or structure["embed_dim"] != embed_dim):
                raise ValueError(
                    "Tiered RankLift checkpoint model dimensions do not match "
                    f"the requested model (state V={structure['vocab_size']}, "
                    f"d={structure['embed_dim']}; requested V={vocab_size}, "
                    f"d={embed_dim})"
                )
            actual_populations = structure["group_sizes"]
            mismatches = {
                "code_dims": (structure["code_dims"], declared_code_dims),
                "lift_dims": (structure["lift_dims"], declared_lift_dims),
                "populations": (actual_populations, declared_populations),
                "rms_eps": (structure["rms_eps"], ca.tiered_ranklift_rms_eps),
            }
            mismatches = {
                key: value for key, value in mismatches.items()
                if value[0] != value[1]
            }
            if mismatches:
                raise ValueError(
                    "Tiered RankLift checkpoint structure does not match the "
                    f"requested configuration: {mismatches}"
                )
            group_ids = structure["group_ids"]
        else:
            if not declared_code_dims or len(declared_code_dims) != len(
                declared_lift_dims
            ):
                raise ValueError(
                    "Tiered RankLift code/lift dimensions must be non-empty "
                    "lists of equal length"
                )
            if len(declared_populations) != len(declared_code_dims):
                raise ValueError(
                    "Tiered RankLift populations must have one entry per group"
                )
            if not ca.tiered_ranklift_frequency_path:
                raise ValueError(
                    "A fresh Tiered RankLift run requires "
                    "--tiered_ranklift_frequency_path"
                )
            importance = load_frequency_counts(
                ca.tiered_ranklift_frequency_path,
                vocab_size,
                key=ca.tiered_ranklift_frequency_key,
                pseudocount=0.0,
            )
            group_ids = frequency_group_ids_from_populations(
                importance, declared_populations
            )
            logger.info(
                "Tiered RankLift importance artifact: %s (sha256=%s), "
                "populations=%s code_dims=%s lift_dims=%s",
                ca.tiered_ranklift_frequency_path,
                file_sha256(ca.tiered_ranklift_frequency_path),
                declared_populations,
                declared_code_dims,
                declared_lift_dims,
            )
        return TieredRankLiftEmbed(
            vocab_size,
            embed_dim,
            code_dims=declared_code_dims,
            lift_dims=declared_lift_dims,
            group_ids=group_ids,
            rms_eps=ca.tiered_ranklift_rms_eps,
        )
    if ca.arm == "funneling":
        return FunnelingEmbed(
            vocab_size,
            embed_dim,
            rank=ca.funneling_rank,
        )
    if ca.arm == "define":
        expansion_dims = _parse_int_list(
            ca.define_expansion_dims, name="define_expansion_dims"
        )
        group_counts = _parse_int_list(
            ca.define_group_counts, name="define_group_counts"
        )
        return DeFINEEmbed(
            vocab_size,
            embed_dim,
            code_dim=ca.define_code_dim,
            expansion_dims=expansion_dims,
            group_counts=group_counts,
        )
    if ca.arm == "tt":
        vocab_modes = _parse_int_list(ca.tt_vocab_shape, name="tt_vocab_shape")
        embedding_modes = _parse_int_list(
            ca.tt_embedding_shape, name="tt_embedding_shape"
        )
        if not vocab_modes:
            vocab_modes = balanced_padded_modes(vocab_size, ca.tt_order)
        if not embedding_modes:
            embedding_modes = balanced_exact_modes(embed_dim, ca.tt_order)
        if len(vocab_modes) != len(embedding_modes):
            raise ValueError("TT vocabulary and embedding shapes must have equal order")
        ranks = _parse_int_list(ca.tt_ranks, name="tt_ranks") or ca.tt_rank
        return TTEmbedding(
            vocab_size,
            embed_dim,
            vocab_modes=vocab_modes,
            embedding_modes=embedding_modes,
            tt_ranks=ranks,
            target_std=(ca.tt_target_std or None),
            implementation=ca.tt_implementation,
            materialize_chunk_size=ca.tt_materialize_chunk_size,
        )
    if ca.arm == "ant":
        return ANTEmbed(vocab_size, ca.K, embed_dim, **shared, num_heads=ca.num_heads)
    if ca.arm == "residual_ant":
        return ResidualANTEmbed(vocab_size, ca.K, embed_dim, **shared, num_heads=ca.num_heads)
    if ca.arm == "v0":
        return V0Embed(vocab_size, ca.K, embed_dim, **shared, max_k=ca.max_k, mode=ca.v0_mode)
    if ca.arm == "v1":
        return V1Embed(vocab_size, ca.K, embed_dim, **shared, max_k=ca.max_k, query=ca.v1_query)
    if ca.arm == "v2":
        return V2Embed(vocab_size, ca.K, embed_dim, **shared, num_heads=ca.num_heads, localenc=ca.localenc)
    if ca.arm == "isolation_control":
        return IsolationControlEmbed(vocab_size, ca.K, embed_dim, **shared,
                                     num_heads=ca.num_heads, localenc=ca.localenc)
    raise ValueError(f"Unknown arm: {ca.arm}")


def validate_output_configuration(comp_args):
    """Validate output-head flags and return the effective output mode."""
    if comp_args.tie_output and comp_args.independent_lowrank_output:
        raise ValueError(
            "--tie_output and --independent_lowrank_output are mutually exclusive"
        )
    if comp_args.independent_lowrank_output and comp_args.arm != "lowrank":
        raise ValueError(
            "--independent_lowrank_output currently requires --arm lowrank"
        )
    if comp_args.mos_components <= 0:
        raise ValueError("--mos_components must be positive")
    if comp_args.mos_context_rank <= 0:
        raise ValueError("--mos_context_rank must be positive")
    if comp_args.mos_chunk_size <= 0:
        raise ValueError("--mos_chunk_size must be positive")
    if comp_args.mos_components > 1 and not comp_args.tie_output:
        raise ValueError("--mos_components > 1 requires --tie_output")
    if comp_args.arm in {
        "pure_local", "pvq", "slim", "groupreduce", "nested_ladder",
        "residual_subspace_experts", "product_code", "ranklift",
        "tiered_ranklift", "funneling", "define", "tt",
    } and not comp_args.tie_output:
        raise ValueError(
            f"--arm {comp_args.arm} requires --tie_output for the compressed "
            "input/output baseline; retaining Qwen's dense lm_head would test "
            "a different architecture"
        )
    if comp_args.tie_output:
        return "tied"
    if comp_args.independent_lowrank_output:
        return "independent_lowrank"
    return "dense"


def _checkpoint_model_state_keys(checkpoint_dir):
    """Read checkpoint tensor names without materializing the full model."""
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return set(handle.keys())

    safetensors_index = os.path.join(
        checkpoint_dir, "model.safetensors.index.json"
    )
    if os.path.isfile(safetensors_index):
        with open(safetensors_index) as handle:
            return set(json.load(handle)["weight_map"])

    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return set(torch.load(
            pytorch_path, map_location="cpu", weights_only=True
        ))

    pytorch_index = os.path.join(
        checkpoint_dir, "pytorch_model.bin.index.json"
    )
    if os.path.isfile(pytorch_index):
        with open(pytorch_index) as handle:
            return set(json.load(handle)["weight_map"])

    raise FileNotFoundError(
        f"Resume checkpoint has no supported HF model state: {checkpoint_dir}"
    )


def validate_resume_compatibility(checkpoint_dir, comp_args,
                                  training_args=None,
                                  expected_embed_module=None,
                                  expected_output_head=None):
    """Fail before Trainer can silently load a mismatched custom topology.

    Transformers restores checkpoints with ``strict=False``. Without this
    guard, resuming an independent-output run with tied/dense flags (or the
    reverse) can leave a freshly initialized head in place while training
    appears to continue normally.
    """
    if checkpoint_dir is None:
        return
    if not isinstance(checkpoint_dir, str) or not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Invalid resume checkpoint: {checkpoint_dir}")

    # Trainer treats all of these files as optional and silently continues
    # with a fresh optimizer/scheduler/step/RNG state when they are absent.
    # That is unsafe for our automatic-resume workflow: model weights from a
    # partial save must not be mistaken for an exact continuation.
    required_trainer_files = [
        "trainer_state.json", "optimizer.pt", "scheduler.pt",
    ]
    missing_trainer_files = [
        filename for filename in required_trainer_files
        if not os.path.isfile(os.path.join(checkpoint_dir, filename))
        or os.path.getsize(os.path.join(checkpoint_dir, filename)) == 0
    ]
    if missing_trainer_files:
        raise FileNotFoundError(
            "Resume checkpoint is missing required Trainer state: "
            f"{missing_trainer_files}"
        )

    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    try:
        with open(trainer_state_path) as handle:
            saved_global_step = json.load(handle)["global_step"]
        if (not isinstance(saved_global_step, int)
                or isinstance(saved_global_step, bool)):
            raise TypeError(
                f"global_step must be an integer, got {saved_global_step!r}"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"Invalid trainer_state.json in {checkpoint_dir}: {error}"
        ) from error

    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if checkpoint_name.startswith("checkpoint-"):
        try:
            checkpoint_step = int(checkpoint_name.removeprefix("checkpoint-"))
        except ValueError as error:
            raise ValueError(
                f"Invalid checkpoint directory name: {checkpoint_name}"
            ) from error
        if saved_global_step != checkpoint_step:
            raise ValueError(
                "Resume checkpoint step mismatch: trainer_state.json has "
                f"global_step={saved_global_step}, directory is {checkpoint_name}"
            )

    if training_args is not None:
        world_size = int(training_args.world_size)
        if world_size > 1:
            required_rng_files = [
                f"rng_state_{rank}.pth" for rank in range(world_size)
            ]
        else:
            required_rng_files = ["rng_state.pth"]
        actual_rng_files = {
            filename for filename in os.listdir(checkpoint_dir)
            if (filename == "rng_state.pth"
                or (filename.startswith("rng_state_")
                    and filename.endswith(".pth")))
        }
        expected_rng_files = set(required_rng_files)
        if actual_rng_files != expected_rng_files:
            raise ValueError(
                "Resume RNG layout does not match the current world size "
                f"{world_size} (expected {sorted(expected_rng_files)}, found "
                f"{sorted(actual_rng_files)})"
            )
    else:
        # Unit/library callers may not have TrainingArguments. Still require
        # evidence that at least one valid-layout RNG state was saved.
        rng_candidates = [
            filename for filename in os.listdir(checkpoint_dir)
            if (filename == "rng_state.pth"
                or (filename.startswith("rng_state_")
                    and filename.endswith(".pth")))
        ]
        required_rng_files = rng_candidates or ["rng_state.pth"]
    missing_rng_files = [
        filename for filename in required_rng_files
        if not os.path.isfile(os.path.join(checkpoint_dir, filename))
        or os.path.getsize(os.path.join(checkpoint_dir, filename)) == 0
    ]
    if missing_rng_files:
        raise FileNotFoundError(
            "Resume checkpoint is missing required RNG state: "
            f"{missing_rng_files}"
        )

    embedding_path = os.path.join(checkpoint_dir, "embedding.pt")
    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(
            f"Compositional resume checkpoint has no embedding.pt: "
            f"{checkpoint_dir}"
        )

    from compositional.tied_head import INDEPENDENT_OUTPUT_FILENAME
    output_path = os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME)
    has_output_sidecar = os.path.isfile(output_path)
    expects_output_sidecar = comp_args.independent_lowrank_output
    if has_output_sidecar != expects_output_sidecar:
        expected = "present" if expects_output_sidecar else "absent"
        raise ValueError(
            f"Resume topology mismatch: {INDEPENDENT_OUTPUT_FILENAME} must be "
            f"{expected} for the requested output configuration"
        )

    config_candidates = [
        os.path.join(checkpoint_dir, "train_config.json"),
        os.path.join(os.path.dirname(checkpoint_dir), "train_config.json"),
    ]
    config_path = next(
        (path for path in config_candidates if os.path.isfile(path)), None
    )
    if config_path is None:
        raise FileNotFoundError(
            "Cannot safely verify compositional resume topology without "
            f"train_config.json: {checkpoint_dir}"
        )

    with open(config_path) as handle:
        saved = json.load(handle)["compositional"]
    current = asdict(comp_args)
    saved.setdefault("tie_output", False)
    saved.setdefault("independent_lowrank_output", False)
    mismatches = {
        key: (saved[key], current[key])
        for key in saved.keys() & current.keys()
        if saved[key] != current[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(f"Resume compositional config mismatch ({details})")

    model_keys = _checkpoint_model_state_keys(checkpoint_dir)
    embedding_state = torch.load(
        embedding_path, map_location="cpu", weights_only=True
    )
    if expected_embed_module is not None:
        expected_embedding_state = expected_embed_module.state_dict()
        if set(embedding_state) != set(expected_embedding_state):
            raise ValueError(
                "Resume embedding.pt has the wrong parameter schema "
                f"(expected {sorted(expected_embedding_state)}, found "
                f"{sorted(embedding_state)})"
            )
        shape_mismatches = {
            key: (tuple(embedding_state[key].shape),
                  tuple(expected_embedding_state[key].shape))
            for key in expected_embedding_state
            if embedding_state[key].shape != expected_embedding_state[key].shape
        }
        if shape_mismatches:
            raise ValueError(
                "Resume embedding.pt has incompatible parameter shapes: "
                f"{shape_mismatches}"
            )
    else:
        expected_embedding_state = embedding_state
    embed_prefix = "model.embed_tokens.embed."
    expected_embed_keys = {
        embed_prefix + key for key in expected_embedding_state
    }
    actual_embed_keys = {
        key for key in model_keys if key.startswith(embed_prefix)
    }
    if actual_embed_keys != expected_embed_keys:
        raise ValueError(
            "Resume HF state does not contain the complete compositional input "
            f"topology (expected {sorted(expected_embed_keys)}, found "
            f"{sorted(actual_embed_keys)})"
        )

    actual_head_keys = {
        key for key in model_keys if key.startswith("lm_head.")
    }
    if comp_args.independent_lowrank_output:
        output_state = torch.load(
            output_path, map_location="cpu", weights_only=True
        )
        if expected_output_head is not None:
            expected_output_state = expected_output_head.state_dict()
        else:
            expected_output_state = {
                "X": output_state.get("X"),
                "proj_weight": output_state.get("proj_weight"),
            }
        if (set(output_state) != {"X", "proj_weight"}
                or set(output_state) != set(expected_output_state)):
            raise ValueError(
                f"Resume {INDEPENDENT_OUTPUT_FILENAME} has the wrong parameter "
                f"schema (expected ['X', 'proj_weight'], found "
                f"{sorted(output_state)})"
            )
        output_shape_mismatches = {
            key: (tuple(output_state[key].shape),
                  tuple(expected_output_state[key].shape))
            for key in expected_output_state
            if output_state[key].shape != expected_output_state[key].shape
        }
        if output_shape_mismatches:
            raise ValueError(
                f"Resume {INDEPENDENT_OUTPUT_FILENAME} has incompatible "
                f"parameter shapes: {output_shape_mismatches}"
            )
        expected_head_keys = {
            "lm_head." + key for key in expected_output_state
        }
    elif comp_args.tie_output:
        expected_output_state = (
            expected_output_head.state_dict()
            if expected_output_head is not None else {}
        )
        expected_head_keys = {
            "lm_head." + key for key in expected_output_state
        }
        if expected_head_keys and actual_head_keys == expected_head_keys:
            from compositional.loading import _load_checkpoint_tensors
            saved_output_state = _load_checkpoint_tensors(
                checkpoint_dir, expected_head_keys
            )
            output_shape_mismatches = {
                key: (
                    tuple(saved_output_state["lm_head." + key].shape),
                    tuple(expected_output_state[key].shape),
                )
                for key in expected_output_state
                if saved_output_state["lm_head." + key].shape
                != expected_output_state[key].shape
            }
            if output_shape_mismatches:
                raise ValueError(
                    "Resume tied output has incompatible parameter shapes: "
                    f"{output_shape_mismatches}"
                )
    else:
        expected_head_keys = {"lm_head.weight"}
    if actual_head_keys != expected_head_keys:
        raise ValueError(
            "Resume HF state has the wrong output-head topology "
            f"(expected {sorted(expected_head_keys)}, found "
            f"{sorted(actual_head_keys)})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CompositionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, comp_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, comp_args = parser.parse_args_into_dataclasses()

    output_mode = validate_output_configuration(comp_args)
    if (model_args.model_name_or_path
            and os.path.isdir(model_args.model_name_or_path)):
        from compositional.loading import is_compositional
        if is_compositional(model_args.model_name_or_path):
            raise ValueError(
                "A compositional checkpoint cannot be loaded through "
                "--model_name_or_path because that path would discard its "
                "sidecars. Continue training with --resume_from_checkpoint "
                "and the matching architecture flags instead."
            )

    # Setup logging — same as train.py
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Compositional parameters {comp_args}")

    set_seed(training_args.seed)

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected: {last_checkpoint}. Resuming training.")

    checkpoint = (
        training_args.resume_from_checkpoint
        if training_args.resume_from_checkpoint is not None
        else last_checkpoint
    )

    # Published P-VQ and GroupReduce are initialized by compressing a trained
    # dense table.  For a new compact run, the conversion artifact supplies
    # both learned factors and structural integer buffers.  For resume, use the
    # checkpoint sidecar as the structural source so an external init artifact
    # is not trusted over the checkpoint being resumed.
    initial_embedding_state = None
    if checkpoint is not None:
        checkpoint_embedding = os.path.join(checkpoint, "embedding.pt")
        if os.path.isfile(checkpoint_embedding):
            initial_embedding_state = _load_embedding_state(checkpoint_embedding)
    elif comp_args.embedding_init_path is not None:
        initial_embedding_state = _load_embedding_state(
            comp_args.embedding_init_path
        )
    elif (comp_args.arm in {"pvq", "groupreduce"}
          and not comp_args.allow_from_scratch_baseline_init):
        raise ValueError(
            f"--arm {comp_args.arm} is a post-hoc published method and requires "
            "--embedding_init_path from a dense-checkpoint conversion. To run "
            "a clearly labelled from-scratch adaptation instead, explicitly "
            "pass --allow_from_scratch_baseline_init."
        )

    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Must set --model_name_or_path or --config_name")

    config.tie_word_embeddings = False

    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name is None:
        raise ValueError("Must set --tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **config_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, **config_kwargs)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    # Replace embed_tokens with compositional embedding
    embed_module = build_arm(
        comp_args,
        config.vocab_size,
        config.hidden_size,
        initial_state=initial_embedding_state,
    )
    if initial_embedding_state is not None:
        embed_module.load_state_dict(initial_embedding_state, strict=True)
    if training_args.bf16:
        embed_module = embed_module.to(torch.bfloat16)
    embed_shim = EmbeddingShim(embed_module)
    model.model.embed_tokens = embed_shim

    independent_output_head = None
    if comp_args.tie_output:
        from compositional.tied_head import make_tied_head
        tied_head = make_tied_head(
            embed_module,
            comp_args.arm,
            config.vocab_size,
            mos_components=comp_args.mos_components,
            mos_context_rank=comp_args.mos_context_rank,
            mos_chunk_size=comp_args.mos_chunk_size,
        )
        model.lm_head = tied_head
        logger.info(
            "Output tied to input embedding (lm_head replaced; MoS K=%d, "
            "context rank=%d, chunk=%d)",
            comp_args.mos_components,
            comp_args.mos_context_rank,
            comp_args.mos_chunk_size,
        )
    elif comp_args.independent_lowrank_output:
        from compositional.tied_head import IndependentLowRankHead
        independent_output_head = IndependentLowRankHead(embed_module)
        model.lm_head = independent_output_head
        logger.info(
            "Output uses independent rank-%d factors cloned from the input "
            "initialization", independent_output_head.rank
        )

    emb_params = sum(p.numel() for p in embed_shim.parameters())
    output_head_params = sum(p.numel() for p in model.lm_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Embedding [{comp_args.arm}]: {emb_params:,} params (K={comp_args.K})")
    if comp_args.tie_output and comp_args.mos_components > 1:
        context_params = sum(
            getattr(model.lm_head, name).numel()
            for name in ("context_down", "context_up", "context_bias")
        )
        prior_params = sum(p.numel() for p in model.lm_head.prior.parameters())
        logger.info(
            "BT-MoS interface + head: blocks=%s contexts=%s prior=%s total=%s",
            f"{emb_params:,}",
            f"{context_params:,}",
            f"{prior_params:,}",
            f"{emb_params + context_params + prior_params:,}",
        )
    elif output_head_params:
        logger.info("Output-head parameters: %s", f"{output_head_params:,}")
    logger.info(f"Output mode: {output_mode}")
    logger.info(f"Total params: {total_params:,}")

    # Load data — identical to train.py
    datasets_list = []
    for lang_dir in sorted(os.listdir(data_args.data_dir)):
        lang_path = os.path.join(data_args.data_dir, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        shard_dirs = sorted(
            os.path.join(lang_path, d) for d in os.listdir(lang_path)
            if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d))
        )
        if shard_dirs:
            total = 0
            for sd in shard_dirs:
                ds = load_from_disk(sd)
                total += ds.num_rows
                datasets_list.append(ds)
            logger.info(f"[{lang_dir}] {total:,} documents ({len(shard_dirs)} shards)")
        else:
            ds = load_from_disk(lang_path)
            logger.info(f"[{lang_dir}] {ds.num_rows:,} documents")
            datasets_list.append(ds)

    if not datasets_list:
        raise ValueError(f"No datasets found in {data_args.data_dir}")

    raw_dataset = concatenate_datasets(datasets_list)
    logger.info(f"Combined: {raw_dataset.num_rows:,} documents")
    column_names = raw_dataset.column_names

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = min(tokenizer.model_max_length,
                         getattr(config, "max_position_embeddings", 1024))
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_dataset.shuffle(seed=training_args.seed)
    logger.info(f"Training dataset: {train_dataset.num_rows:,} sequences of {block_size} tokens")

    # Initialize Trainer
    trainer = CompositionalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        embed_shim=embed_shim,
        comp_args=comp_args,
        callbacks=[SaveEmbeddingCallback(
            embed_shim, independent_output_head=independent_output_head
        )],
    )

    validate_resume_compatibility(
        checkpoint,
        comp_args,
        training_args,
        expected_embed_module=embed_module,
        expected_output_head=(
            independent_output_head if independent_output_head is not None
            else model.lm_head
        ),
    )

    # Save train config BEFORE training — runs killed at a stop-step never reach
    # the post-training save, and eval needs this file to rebuild the embedding.
    if training_args.should_save:
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_train_config(training_args.output_dir, model_args, data_args, training_args, comp_args)

    # Training
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    if training_args.should_save:
        torch.save(embed_shim.embed.state_dict(),
                   os.path.join(training_args.output_dir, "embedding.pt"))
        if independent_output_head is not None:
            from compositional.tied_head import INDEPENDENT_OUTPUT_FILENAME
            torch.save(
                independent_output_head.state_dict(),
                os.path.join(
                    training_args.output_dir, INDEPENDENT_OUTPUT_FILENAME
                ),
            )

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.should_save:
        save_train_config(training_args.output_dir, model_args, data_args, training_args, comp_args)
    logger.info(f"Training complete. Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
