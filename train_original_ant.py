"""Train a causal LM from scratch with Original ANT embedding (Liang et al. 2021).

Separate from train_compositional.py because Original ANT needs YOGI optimizer
with per-coordinate L1 proximal — requires a HybridOptimizer.

Same data pipeline, same backbone, same Trainer as train.py.
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

from compositional import OriginalANT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default=None)
    config_name: str | None = field(default=None)
    tokenizer_name: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    token: str | None = field(default=None)
    trust_remote_code: bool = field(default=False)


@dataclass
class DataArguments:
    data_dir: str = field(default="data/sampled")
    block_size: int | None = field(default=None)
    preprocessing_num_workers: int | None = field(default=None)
    overwrite_cache: bool = field(default=False)


@dataclass
class OriginalANTArguments:
    K: int = field(default=4096, metadata={"help": "Codebook size."})
    emb_lr: float = field(default=1e-2, metadata={"help": "YOGI learning rate for embedding."})
    lam: float = field(default=1e-3, metadata={"help": "L1 proximal penalty target."})


# ---------------------------------------------------------------------------
# EmbeddingShim
# ---------------------------------------------------------------------------

class EmbeddingShim(nn.Module):
    def __init__(self, embed_module):
        super().__init__()
        self.embed = embed_module
        self._last_theta = None

    def forward(self, input_ids):
        e, theta = self.embed(input_ids)
        self._last_theta = theta
        return e


# ---------------------------------------------------------------------------
# HybridOptimizer — AdamW for backbone, YOGI for embedding
# ---------------------------------------------------------------------------

def lam_at(step, lam_target, warmup_steps, total_steps):
    if step < warmup_steps:
        return 0.0
    return lam_target * (step - warmup_steps) / max(1, total_steps - warmup_steps)


class HybridOptimizer(torch.optim.Optimizer):
    """AdamW for groups with use_yogi=False, YOGI+proximal for use_yogi=True."""

    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8,
                        "weight_decay": 0.0, "use_yogi": False, "apply_proximal": False}
        super().__init__(params, defaults)
        self.l1_penalty = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_yogi", False):
                self._yogi_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _adamw_step(self, group):
        lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            m, v = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            p.data.mul_(1 - lr * wd)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / bc1)

    def _yogi_step(self, group):
        lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
        apply_proximal = group.get("apply_proximal", False)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.full_like(p, 1e-6)
            m, v = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            g2 = grad * grad
            v.add_((v - g2).sign_() * g2, alpha=beta2 - 1)
            denom = v.sqrt().add_(eps)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            step_size = lr * math.sqrt(bc2) / bc1
            p.addcdiv_(m, denom, value=-step_size)
            if self.l1_penalty > 0 and apply_proximal:
                thr = self.l1_penalty * (step_size / denom)
                p.data.sub_(thr)
                p.data.clamp_min_(0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OriginalANTTrainer(Trainer):

    def __init__(self, *args, embed_shim=None, ant_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_shim = embed_shim
        self.ant_args = ant_args
        self._comp_sums = {}
        self._comp_count = 0

    def _get_hybrid_opt(self):
        """Unwrap AcceleratedOptimizer to reach HybridOptimizer."""
        opt = self.optimizer
        return getattr(opt, "optimizer", opt)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update l1_penalty on the underlying HybridOptimizer (not the wrapper)
        if self.optimizer is not None:
            self._get_hybrid_opt().l1_penalty = lam_at(
                self.state.global_step, self.ant_args.lam,
                self.args.warmup_steps, self.state.max_steps,
            )

        # Forward num_items_in_batch so the model normalizes the summed CE across
        # gradient-accumulation steps — without it, logged loss (and gradients)
        # are scaled by gradient_accumulation_steps.
        loss_kwargs = {}
        if kwargs.get("num_items_in_batch") is not None:
            loss_kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"]
        outputs = model(input_ids=inputs["input_ids"], labels=inputs.get("labels", inputs["input_ids"]),
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
                avg_nnz = active.sum(-1).mean().item()
                dead_rate = (active.mean(dim=(0, 1)) == 0).float().mean().item()
            else:
                avg_nnz = dead_rate = 0.0

        self._comp_count += 1
        self._comp_sums["avg_nnz"] = self._comp_sums.get("avg_nnz", 0.0) + avg_nnz
        self._comp_sums["dead_rate"] = self._comp_sums.get("dead_rate", 0.0) + dead_rate
        self._comp_sums["l1_penalty"] = self._get_hybrid_opt().l1_penalty if self.optimizer else 0.0

        return (lm_loss, outputs) if return_outputs else lm_loss

    def log(self, logs, *args, **kwargs):
        if self._comp_count > 0:
            logs["avg_nnz"] = self._comp_sums.get("avg_nnz", 0.0) / self._comp_count
            logs["dead_rate"] = self._comp_sums.get("dead_rate", 0.0) / self._comp_count
            logs["l1_penalty"] = self._comp_sums.get("l1_penalty", 0.0)
            if "loss" in logs and logs["loss"] > 0:
                logs["perplexity"] = math.exp(min(logs["loss"], 20))
            self._comp_sums = {}
            self._comp_count = 0
        super().log(logs, *args, **kwargs)

    def create_optimizer(self):
        embed_module = self.embed_shim.embed
        embed_param_ids = set(id(p) for p in self.embed_shim.parameters())
        bb_params = [p for p in self.model.parameters() if id(p) not in embed_param_ids]

        self.optimizer = HybridOptimizer([
            {"params": bb_params, "lr": self.args.learning_rate,
             "betas": (self.args.adam_beta1, self.args.adam_beta2),
             "eps": self.args.adam_epsilon,
             "weight_decay": self.args.weight_decay, "use_yogi": False},
            {"params": embed_module.non_sparse_params(), "lr": self.ant_args.emb_lr,
             "betas": (0.9, 0.999), "eps": 1e-3, "weight_decay": 0.0, "use_yogi": True},
            {"params": embed_module.sparse_params(), "lr": self.ant_args.emb_lr,
             "betas": (0.9, 0.999), "eps": 1e-3, "weight_decay": 0.0,
             "use_yogi": True, "apply_proximal": True},
        ])
        return self.optimizer



class SaveEmbeddingCallback(TrainerCallback):
    def __init__(self, embed_shim):
        self.embed_shim = embed_shim

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            torch.save(self.embed_shim.embed.state_dict(),
                       os.path.join(checkpoint_dir, "embedding.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, OriginalANTArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, ant_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, ant_args = parser.parse_args_into_dataclasses()

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

    set_seed(training_args.seed)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Config + tokenizer + model
    config_kwargs = {"cache_dir": model_args.cache_dir, "token": model_args.token,
                     "trust_remote_code": model_args.trust_remote_code}
    config = AutoConfig.from_pretrained(model_args.config_name or model_args.model_name_or_path, **config_kwargs)
    config.tie_word_embeddings = False

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **config_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    logger.info(f"Training from scratch: {sum(p.numel() for p in model.parameters()):,} params")

    # Replace embed_tokens
    embed_module = OriginalANT(config.vocab_size, ant_args.K, config.hidden_size)
    if training_args.bf16:
        embed_module = embed_module.to(torch.bfloat16)
    embed_shim = EmbeddingShim(embed_module)
    model.model.embed_tokens = embed_shim
    logger.info(f"Original ANT embedding: {sum(p.numel() for p in embed_shim.parameters()):,} params (K={ant_args.K})")

    # Data — same as train.py
    datasets_list = []
    for lang_dir in sorted(os.listdir(data_args.data_dir)):
        lang_path = os.path.join(data_args.data_dir, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        shard_dirs = sorted(
            os.path.join(lang_path, d) for d in os.listdir(lang_path)
            if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d)))
        if shard_dirs:
            for sd in shard_dirs:
                datasets_list.append(load_from_disk(sd))
        else:
            datasets_list.append(load_from_disk(lang_path))

    raw_dataset = concatenate_datasets(datasets_list)
    column_names = raw_dataset.column_names

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    with training_args.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function, batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache)

    block_size = data_args.block_size or min(tokenizer.model_max_length,
                                              getattr(config, "max_position_embeddings", 1024))

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first():
        lm_dataset = tokenized_dataset.map(
            group_texts, batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = lm_dataset.shuffle(seed=training_args.seed)
    logger.info(f"Dataset: {train_dataset.num_rows:,} sequences of {block_size} tokens")

    # Trainer
    trainer = OriginalANTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        embed_shim=embed_shim,
        ant_args=ant_args,
        callbacks=[SaveEmbeddingCallback(embed_shim)],
    )

    def write_train_config():
        with open(os.path.join(training_args.output_dir, "train_config.json"), "w") as f:
            json.dump({"model": asdict(model_args), "data": asdict(data_args),
                       "training": {k: v for k, v in training_args.to_dict().items()
                                    if v is not None and k not in ("_n_gpu", "local_rank")},
                       "compositional": {"arm": "original_ant", **asdict(ant_args)}},
                      f, indent=2, default=str)

    # Save train config BEFORE training — runs killed at a stop-step never reach
    # the post-training save, and eval needs this file to rebuild the embedding.
    if training_args.should_save:
        os.makedirs(training_args.output_dir, exist_ok=True)
        write_train_config()

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    if training_args.should_save:
        torch.save(embed_shim.embed.state_dict(),
                   os.path.join(training_args.output_dir, "embedding.pt"))
        write_train_config()

    logger.info(f"Training complete. Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
