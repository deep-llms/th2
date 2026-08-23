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
    IsolationControlEmbed, LowRankEmbed, SharedLocalEmbed,
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
                        "isolation_control", "lowrank", "shared_local"],
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
    num_groups: int = field(default=4, metadata={
        "help": "Number of contiguous vocabulary groups for the shared_local arm."
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
                "Supported for lowrank, shared_local, original_ant, ant, and "
                "residual_ant; not supported "
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

class CompositionalTrainer(Trainer):

    def __init__(self, *args, embed_shim=None, comp_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_shim = embed_shim
        self.comp_args = comp_args
        self._comp_sums = {}
        self._comp_count = 0

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
            div_loss = load_balance(theta)
            # With num_items_in_batch normalization the micro-batch losses SUM to
            # the true batch loss, so the per-micro-batch div term must be scaled
            # by 1/accum to keep its effective weight at lambda_div.
            div_scale = (1.0 / self.args.gradient_accumulation_steps
                         if loss_kwargs else 1.0)
            total_loss = lm_loss + self.comp_args.lambda_div * div_scale * div_loss
            div_loss_val = div_loss.detach().item()

        self._comp_count += 1
        for k, v in [("avg_nnz", avg_nnz.item()), ("dead_rate", dead_rate.item()),
                     ("entropy", entropy.item()), ("div_loss", div_loss_val)]:
            self._comp_sums[k] = self._comp_sums.get(k, 0.0) + v

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if self._comp_count > 0:
            for k, v in self._comp_sums.items():
                logs[k] = v / self._comp_count
            lm_loss = logs.get("loss", 0.0)
            if lm_loss > 0:
                logs["perplexity"] = math.exp(min(lm_loss, 20))
            self._comp_sums = {}
            self._comp_count = 0
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


def build_arm(comp_args, vocab_size, embed_dim):
    ca = comp_args
    shared = dict(d_x=ca.d_x, d_k=ca.d_k, gamma=ca.gamma)

    if ca.arm == "lowrank":
        return LowRankEmbed(vocab_size, embed_dim, rank=ca.d_x)
    if ca.arm == "shared_local":
        return SharedLocalEmbed(
            vocab_size, embed_dim,
            shared_rank=ca.shared_rank,
            local_rank=ca.local_embed_rank,
            num_groups=ca.num_groups,
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
        expected_head_keys = set()
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
    embed_module = build_arm(comp_args, config.vocab_size, config.hidden_size)
    if training_args.bf16:
        embed_module = embed_module.to(torch.bfloat16)
    embed_shim = EmbeddingShim(embed_module)
    model.model.embed_tokens = embed_shim

    independent_output_head = None
    if comp_args.tie_output:
        from compositional.tied_head import make_tied_head
        tied_head = make_tied_head(embed_module, comp_args.arm, config.vocab_size)
        model.lm_head = tied_head
        logger.info(f"Output tied to input embedding (lm_head replaced)")
    elif comp_args.independent_lowrank_output:
        from compositional.tied_head import IndependentLowRankHead
        independent_output_head = IndependentLowRankHead(embed_module)
        model.lm_head = independent_output_head
        logger.info(
            "Output uses independent rank-%d factors cloned from the input "
            "initialization", independent_output_head.rank
        )

    emb_params = sum(p.numel() for p in embed_shim.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Embedding [{comp_args.arm}]: {emb_params:,} params (K={comp_args.K})")
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

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    validate_resume_compatibility(
        checkpoint,
        comp_args,
        training_args,
        expected_embed_module=embed_module,
        expected_output_head=independent_output_head,
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
