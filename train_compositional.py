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
    ANTEmbed, V0Embed, V1Embed, V2Embed, IsolationControlEmbed,
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
            "choices": ["ant", "v0", "v1", "v2", "isolation_control"],
        },
    )
    K: int = field(default=4096, metadata={"help": "Codebook size (number of anchors)."})
    d_x: int = field(default=128, metadata={"help": "Base token table dimension."})
    d_k: int = field(default=64, metadata={"help": "Router key dimension."})
    gamma: float = field(default=1.0, metadata={"help": "Score temperature for entmax."})
    num_heads: int = field(default=1, metadata={"help": "Number of selection heads (ANT/V2 only)."})
    max_k: int = field(default=16, metadata={"help": "Max anchors per token (V0/V1 only)."})
    v0_mode: str = field(default="post", metadata={"help": "V0 beta mode.", "choices": ["post", "pre"]})
    v1_query: str = field(default="content", metadata={"help": "V1 query.", "choices": ["content", "cls"]})
    localenc: str = field(default="attn", metadata={"help": "V2 LocalEnc.", "choices": ["attn", "conv", "conv_lite"]})
    lambda_div: float = field(default=0.0, metadata={"help": "Load-balance loss weight."})


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
            total_loss = lm_loss + self.comp_args.lambda_div * div_loss
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
    def __init__(self, embed_shim):
        self.embed_shim = embed_shim

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            torch.save(
                self.embed_shim.embed.state_dict(),
                os.path.join(checkpoint_dir, "embedding.pt"),
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

    if ca.arm == "ant":
        return ANTEmbed(vocab_size, ca.K, embed_dim, **shared, num_heads=ca.num_heads)
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

    emb_params = sum(p.numel() for p in embed_shim.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Embedding [{comp_args.arm}]: {emb_params:,} params (K={comp_args.K})")
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
        callbacks=[SaveEmbeddingCallback(embed_shim)],
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    if training_args.should_save:
        torch.save(embed_shim.embed.state_dict(),
                   os.path.join(training_args.output_dir, "embedding.pt"))

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
