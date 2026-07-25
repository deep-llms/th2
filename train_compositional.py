"""Train a causal LM from scratch with compositional embeddings.

Adapted from train.py — same data pipeline, same backbone, same arg structure.
Only the training loop changes: uses accelerate with two optimizers (backbone
AdamW + embedding-specific optimizer) instead of HF Trainer.

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

import datasets
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader

from accelerate import Accelerator

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from compositional import (
    OriginalANT, ANTEmbed, V0Embed, V1Embed, V2Embed, IsolationControlEmbed,
    Yogi, compute_loss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments — same pattern as train.py
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
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where model checkpoints will be written."},
    )
    seed: int = field(default=42)
    bf16: bool = field(default=False)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=4)
    num_train_epochs: int = field(default=1)
    max_train_steps: int | None = field(
        default=None,
        metadata={"help": "Override total optimizer steps (computed from data if omitted)."},
    )
    learning_rate: float = field(default=3e-4)
    min_lr_rate: float = field(
        default=0.1,
        metadata={"help": "Minimum LR as fraction of max LR for cosine schedule."},
    )
    warmup_steps: int = field(default=500)
    weight_decay: float = field(default=0.1)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=250)
    dataloader_num_workers: int = field(default=8)
    run_name: str = field(default="compositional")
    report_to: str = field(default="wandb")
    resume_from_checkpoint: str | None = field(default=None)

    # Compositional embedding
    arm: str = field(
        default="original_ant",
        metadata={
            "help": "Embedding arm to train.",
            "choices": [
                "original_ant", "ant",
                "v0", "v1",
                "v2", "isolation_control",
            ],
        },
    )
    K: int = field(default=4096, metadata={"help": "Codebook size (number of anchors)."})
    d_x: int = field(default=128, metadata={"help": "Base token table dimension."})
    d_k: int = field(default=64, metadata={"help": "Router key dimension."})
    gamma: float = field(default=1.0, metadata={"help": "Score temperature for entmax."})
    num_heads: int = field(
        default=1,
        metadata={"help": "Number of selection heads (ANT/V2 only; V0/V1 stay at 1)."},
    )
    max_k: int = field(default=16, metadata={"help": "Max anchors per token (V0/V1 only)."})
    v0_mode: str = field(
        default="post",
        metadata={"help": "V0 beta application: post (after SAT) or pre (before SAT).",
                  "choices": ["post", "pre"]},
    )
    v1_query: str = field(
        default="content",
        metadata={"help": "V1 context query: content (mean-pool) or cls (learned).",
                  "choices": ["content", "cls"]},
    )
    localenc: str = field(
        default="attn",
        metadata={"help": "V2/isolation_control LocalEnc variant.",
                  "choices": ["attn", "conv", "conv_lite"]},
    )
    emb_lr: float | None = field(
        default=None,
        metadata={"help": "Embedding optimizer LR (default: learning_rate for ant/v*, 1e-2 for original_ant)."},
    )
    lam: float = field(default=1e-3, metadata={"help": "L1 proximal penalty target (original_ant)."})
    lambda_div: float | None = field(
        default=None,
        metadata={"help": "Load-balance loss weight (default: 0). Set to 1e-2 to enable."},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_train_config(save_dir, model_args, data_args, training_args):
    config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": asdict(training_args),
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def get_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_rate=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def lam_at(step, lam_target, warmup_steps, total_steps):
    if step < warmup_steps:
        return 0.0
    return lam_target * (step - warmup_steps) / max(1, total_steps - warmup_steps)


def build_arm(training_args, vocab_size, embed_dim):
    ta = training_args
    shared = dict(d_x=ta.d_x, d_k=ta.d_k, gamma=ta.gamma)

    if ta.arm == "original_ant":
        return OriginalANT(vocab_size, ta.K, embed_dim)
    if ta.arm == "ant":
        return ANTEmbed(vocab_size, ta.K, embed_dim, **shared,
                        num_heads=ta.num_heads)
    if ta.arm == "v0":
        return V0Embed(vocab_size, ta.K, embed_dim, **shared,
                       max_k=ta.max_k, mode=ta.v0_mode)
    if ta.arm == "v1":
        return V1Embed(vocab_size, ta.K, embed_dim, **shared,
                       max_k=ta.max_k, query=ta.v1_query)
    if ta.arm == "v2":
        return V2Embed(vocab_size, ta.K, embed_dim, **shared,
                       num_heads=ta.num_heads, localenc=ta.localenc)
    if ta.arm == "isolation_control":
        return IsolationControlEmbed(vocab_size, ta.K, embed_dim, **shared,
                                     num_heads=ta.num_heads, localenc=ta.localenc)
    raise ValueError(f"Unknown arm: {ta.arm}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Resolve per-arm defaults
    is_orig_ant = training_args.arm == "original_ant"
    if training_args.emb_lr is None:
        training_args.emb_lr = 1e-2 if is_orig_ant else training_args.learning_rate
    if training_args.lambda_div is None:
        training_args.lambda_div = 0.0

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16" if training_args.bf16 else "no",
    )

    # Setup logging — same pattern as train.py
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARNING)

    logger.warning(
        f"Process rank: {accelerator.process_index}, device: {accelerator.device}, "
        f"n_gpu: {accelerator.num_processes}, "
        f"distributed training: {accelerator.num_processes > 1}, "
        f"mixed precision: {accelerator.mixed_precision}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # Detect last checkpoint for resume
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            d for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(
                os.path.join(training_args.output_dir, d)
            )
        ]
        if checkpoints:
            last_checkpoint = os.path.join(
                training_args.output_dir,
                max(checkpoints, key=lambda x: int(x.split("-")[1])),
            )
            logger.info(f"Checkpoint detected: {last_checkpoint}. Resuming training.")

    # -------------------------------------------------------------------------
    # Load config — same as train.py, plus tie_word_embeddings=False
    # -------------------------------------------------------------------------
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        if model_args.config_name.endswith(".json") and os.path.isfile(model_args.config_name):
            with open(model_args.config_name) as f:
                config_dict = json.load(f)
            config = AutoConfig.for_model(**config_dict)
        else:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Must set --model_name_or_path or --config_name")

    config.tie_word_embeddings = False

    # -------------------------------------------------------------------------
    # Load tokenizer — same as train.py
    # -------------------------------------------------------------------------
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path or model_args.config_name
    if tokenizer_name is None:
        raise ValueError("Must set --tokenizer_name when training from scratch without --model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Load model — same as train.py, plus remove embed_tokens
    # -------------------------------------------------------------------------
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info(f"Loaded pretrained model: {model_args.model_name_or_path}")
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    model.model.embed_tokens = None

    bb_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Backbone (no input embed): {bb_params:,} params")

    # -------------------------------------------------------------------------
    # Build embedding module
    # -------------------------------------------------------------------------
    embed = build_arm(training_args, config.vocab_size, config.hidden_size)
    if training_args.bf16:
        embed = embed.to(torch.bfloat16)
    emb_params = sum(p.numel() for p in embed.parameters())
    logger.info(f"Embedding [{training_args.arm}]: {emb_params:,} params (K={training_args.K})")

    trainable_params = bb_params + emb_params
    logger.info(f"Total params: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # Optimizers — backbone AdamW + embedding optimizer (arm-specific)
    # -------------------------------------------------------------------------
    bb_opt = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
    )

    if is_orig_ant:
        emb_opt = Yogi(
            [
                {"params": embed.non_sparse_params()},
                {"params": embed.sparse_params(), "apply_proximal": True},
            ],
            lr=training_args.emb_lr,
        )
    else:
        emb_opt = torch.optim.AdamW(
            embed.parameters(),
            lr=training_args.emb_lr,
            weight_decay=training_args.weight_decay,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
        )

    # -------------------------------------------------------------------------
    # Load data — same as train.py
    # -------------------------------------------------------------------------
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

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Determine block_size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if hasattr(config, "max_position_embeddings"):
            max_pos = config.max_position_embeddings
        else:
            max_pos = 1024
        if block_size > max_pos:
            logger.warning(
                f"Tokenizer model_max_length ({block_size}) > max_position_embeddings ({max_pos}). "
                f"Using block_size={min(1024, max_pos)}."
            )
            block_size = min(1024, max_pos) if max_pos > 0 else 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"block_size ({data_args.block_size}) > tokenizer model_max_length ({tokenizer.model_max_length}). "
                f"Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Group texts into chunks of block_size
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

    with accelerator.main_process_first():
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_dataset.shuffle(seed=training_args.seed)
    logger.info(f"Training dataset: {train_dataset.num_rows:,} sequences of {block_size} tokens")

    train_dataset.set_format(type="torch", columns=["input_ids"])

    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------------------------------------------
    # Prepare distributed (before schedulers — dataloader length changes)
    # -------------------------------------------------------------------------
    model, embed, bb_opt, emb_opt, dataloader = accelerator.prepare(
        model, embed, bb_opt, emb_opt, dataloader
    )

    # Compute training steps from prepared dataloader
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / training_args.gradient_accumulation_steps
    )
    if training_args.max_train_steps is None:
        max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    else:
        max_train_steps = training_args.max_train_steps
        training_args.num_train_epochs = math.ceil(
            max_train_steps / num_update_steps_per_epoch
        )

    # Schedulers (after prepare so step count is correct)
    bb_sched = get_cosine_schedule_with_min_lr(
        bb_opt, training_args.warmup_steps, max_train_steps, training_args.min_lr_rate
    )
    emb_sched = get_cosine_schedule_with_min_lr(
        emb_opt, training_args.warmup_steps, max_train_steps, training_args.min_lr_rate
    )
    bb_sched, emb_sched = accelerator.prepare(bb_sched, emb_sched)

    # -------------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------------
    completed_steps = 0
    starting_epoch = 0
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        state_path = os.path.join(checkpoint, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            completed_steps = state["completed_steps"]
            starting_epoch = state["epoch"]
        logger.info(f"Resumed from {checkpoint}, step {completed_steps}")

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------
    if accelerator.is_main_process and training_args.report_to == "wandb":
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "sparse_embedding"),
            name=training_args.run_name,
            config={**asdict(model_args), **asdict(data_args), **asdict(training_args)},
            mode=os.environ.get("WANDB_MODE", "online"),
            resume="allow" if checkpoint else None,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    eff_batch = (
        training_args.per_device_train_batch_size
        * accelerator.num_processes
        * training_args.gradient_accumulation_steps
    )
    logger.info("*** Train ***")
    logger.info(f"  Num examples = {len(train_dataset):,}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Per-device batch size = {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps:,}")
    logger.info(f"  Effective batch = {eff_batch} seqs = {eff_batch * block_size:,} tokens/step")
    if is_orig_ant:
        logger.info(f"  YOGI lr={training_args.emb_lr}, lam_target={training_args.lam}")
    else:
        logger.info(f"  AdamW emb_lr={training_args.emb_lr}, lambda_div={training_args.lambda_div}")

    model.train()
    embed.train()

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        active_dataloader = dataloader

        if checkpoint is not None and epoch == starting_epoch and completed_steps > 0:
            skip_batches = completed_steps * training_args.gradient_accumulation_steps
            active_dataloader = accelerator.skip_first_batches(dataloader, skip_batches)
            logger.info(f"Skipped {skip_batches} batches for resume")

        for batch in active_dataloader:
            with accelerator.accumulate(model, embed):
                input_ids = batch["input_ids"]
                e, theta = embed(input_ids)
                outputs = model(inputs_embeds=e)

                loss, logs = compute_loss(
                    outputs.logits, input_ids, theta,
                    lambda_div=training_args.lambda_div,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                    if is_orig_ant:
                        unwrapped_emb_opt = (
                            emb_opt.optimizer if hasattr(emb_opt, "optimizer") else emb_opt
                        )
                        unwrapped_emb_opt.l1_penalty = lam_at(
                            completed_steps, training_args.lam,
                            training_args.warmup_steps, max_train_steps,
                        )
                    else:
                        accelerator.clip_grad_norm_(
                            embed.parameters(), training_args.max_grad_norm
                        )

                bb_opt.step()
                emb_opt.step()
                bb_sched.step()
                emb_sched.step()
                bb_opt.zero_grad()
                emb_opt.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1

                if completed_steps % training_args.logging_steps == 0:
                    lm_loss_val = accelerator.gather(logs["lm_loss"]).mean().item()
                    avg_nnz_val = accelerator.gather(logs["avg_nnz"]).mean().item()
                    dead_rate_val = accelerator.gather(logs["dead_rate"]).mean().item()
                    ent_val = accelerator.gather(logs["entropy"]).mean().item()
                    bb_lr = bb_sched.get_last_lr()[0]
                    emb_lr = emb_sched.get_last_lr()[0]
                    ppl = math.exp(min(lm_loss_val, 20))

                    if is_orig_ant:
                        unwrapped_emb_opt = (
                            emb_opt.optimizer if hasattr(emb_opt, "optimizer") else emb_opt
                        )
                        l1_pen = unwrapped_emb_opt.l1_penalty
                    else:
                        l1_pen = 0.0

                    if accelerator.is_main_process:
                        log_str = (
                            f"step {completed_steps}/{max_train_steps} | "
                            f"loss {lm_loss_val:.4f} | ppl {ppl:.1f} | "
                            f"nnz {avg_nnz_val:.1f} | dead {dead_rate_val:.3f} | "
                            f"bb_lr {bb_lr:.2e} | emb_lr {emb_lr:.2e}"
                        )
                        if l1_pen > 0:
                            log_str += f" | l1 {l1_pen:.2e}"
                        logger.info(log_str)

                        if training_args.report_to == "wandb":
                            import wandb
                            wandb.log({
                                "lm_loss": lm_loss_val,
                                "perplexity": ppl,
                                "avg_nnz": avg_nnz_val,
                                "dead_rate": dead_rate_val,
                                "entropy": ent_val,
                                "bb_lr": bb_lr,
                                "emb_lr": emb_lr,
                                "l1_penalty": l1_pen,
                            }, step=completed_steps)

                if completed_steps % training_args.save_steps == 0:
                    save_dir = os.path.join(
                        training_args.output_dir, f"checkpoint-{completed_steps}"
                    )
                    accelerator.save_state(save_dir)
                    if accelerator.is_main_process:
                        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                            json.dump({"completed_steps": completed_steps, "epoch": epoch}, f)
                    logger.info(f"Saved checkpoint: {save_dir}")

                if completed_steps >= max_train_steps:
                    break

        if completed_steps >= max_train_steps:
            break

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(training_args.output_dir, "backbone"))

        unwrapped_embed = accelerator.unwrap_model(embed)
        torch.save(
            unwrapped_embed.state_dict(),
            os.path.join(training_args.output_dir, "embedding.pt"),
        )

        tokenizer.save_pretrained(training_args.output_dir)
        save_train_config(training_args.output_dir, model_args, data_args, training_args)

    logger.info(f"Training complete. Model saved to: {training_args.output_dir}")

    if accelerator.is_main_process and training_args.report_to == "wandb":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
