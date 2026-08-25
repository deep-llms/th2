"""Train a causal LM from scratch on multilingual data.

Adapted from HuggingFace's run_clm.py example:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Data is loaded from per-language directories saved by prepare_data.py.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from itertools import chain

import torch
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
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


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
class PVQCurriculumArguments:
    pvq_curriculum: bool = field(default=False, metadata={
        "help": "Run P-VQ's dense-table clustering curriculum. Requires a "
                "pretrained dense exactly-tied model; compact conversion is a "
                "separate step after this run."
    })
    pvq_shared_dim: int = field(default=768)
    pvq_k_begin: int = field(default=1024)
    pvq_k_end: int = field(default=128)
    pvq_k_decay: int = field(default=128)
    pvq_cluster_every: int = field(default=1000)
    pvq_curriculum_steps: int = field(default=10000)
    pvq_cluster_iters: int = field(default=100)
    pvq_cluster_restarts: int = field(default=10)
    pvq_cluster_chunk_size: int = field(default=4096)
    pvq_cluster_device: str = field(default="cuda")
    pvq_final_recluster: bool = field(default=True)


def save_train_config(save_dir, model_args, data_args, training_args,
                      pvq_args=None):
    config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": {
            k: v for k, v in training_args.to_dict().items()
            if v is not None and v != "" and k not in ("_n_gpu", "local_rank")
        },
    }
    if pvq_args is not None:
        config["pvq_curriculum"] = asdict(pvq_args)
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def main():
    parser = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments,
        PVQCurriculumArguments,
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, pvq_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, pvq_args = parser.parse_args_into_dataclasses()

    # Setup logging
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
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    if pvq_args.pvq_curriculum:
        logger.info(f"P-VQ curriculum parameters {pvq_args}")

    set_seed(training_args.seed)

    # Detect last checkpoint for resume
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

    if pvq_args.pvq_curriculum:
        if not model_args.model_name_or_path:
            raise ValueError(
                "P-VQ curriculum requires --model_name_or_path pointing to a "
                "pretrained dense tied checkpoint"
            )
        if not getattr(config, "tie_word_embeddings", False):
            raise ValueError(
                "P-VQ curriculum requires tie_word_embeddings=true"
            )

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name is None:
        raise ValueError("Must set --tokenizer_name when training from scratch without --model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    if pvq_args.pvq_curriculum:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise ValueError(
                "Loaded model does not have exact input-output parameter tying"
            )

    # Load per-language datasets (supports both single-dir and sharded layouts)
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

    with training_args.main_process_first(desc="dataset map tokenization"):
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
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    callbacks = []
    if pvq_args.pvq_curriculum:
        from compositional.pvq_curriculum import PVQCurriculumCallback
        callbacks.append(PVQCurriculumCallback(
            shared_dim=pvq_args.pvq_shared_dim,
            k_begin=pvq_args.pvq_k_begin,
            k_end=pvq_args.pvq_k_end,
            k_decay=pvq_args.pvq_k_decay,
            cluster_every=pvq_args.pvq_cluster_every,
            curriculum_steps=pvq_args.pvq_curriculum_steps,
            cluster_iters=pvq_args.pvq_cluster_iters,
            cluster_restarts=pvq_args.pvq_cluster_restarts,
            cluster_chunk_size=pvq_args.pvq_cluster_chunk_size,
            cluster_device=pvq_args.pvq_cluster_device,
            seed=training_args.seed,
            resume_checkpoint=checkpoint,
            final_recluster=pvq_args.pvq_final_recluster,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    if training_args.should_save:
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_train_config(
            training_args.output_dir, model_args, data_args, training_args,
            pvq_args,
        )

    # Training
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.should_save:
        save_train_config(
            training_args.output_dir, model_args, data_args, training_args,
            pvq_args,
        )
    logger.info(f"Training complete. Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
