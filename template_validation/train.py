"""Local causal-LM baseline using ordinary HF Trainer and Accelerate.

Use --config_name plus --tokenizer_name for random initialization, or
--model_name_or_path for pretrained initialization. Assets must be local.
Local Parquet/save_to_disk text is packed with explicit document EOS and HF
caching. Default: within-document SDPA, reset positions, masked boundary targets.
"""
import os

# Set before importing HF integrations. Remote assets are acquired separately.
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  WANDB_MODE="offline")

import json
import logging
import sys
from dataclasses import dataclass, field, asdict, replace
from copy import copy
from pathlib import Path

import datasets
import transformers
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,
    TrainingArguments, TrainerCallback, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from training_data import prepare_training_data, require_resume_data, save_data_recipe, require_held_out
from packed_training import PackedCollator, PackedTrainer
from resume_recipe import build_training_recipe, require_resume_training, save_training_recipe

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None, metadata={"help": "Local pretrained weights; omit for random initialization"})
    config_name: str | None = field(
        default=None, metadata={"help": "Local config directory or JSON file"})
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Local tokenizer directory; defaults to model_name_or_path"})
    cache_dir: str | None = None
    trust_remote_code: bool = False
    attention_backend: str = field(default="sdpa", metadata={"help": "sdpa or fa4; FA4 requires its optional pinned CUDA kernel"})


@dataclass
class DataArguments:
    data_dir: str = field(default="data/sampled", metadata={
        "help": "Local Parquet file/directory or saved HF Dataset; select one split explicitly"})
    block_size: int | None = None
    preprocessing_num_workers: int | None = None
    overwrite_cache: bool = False
    document_end_token: str | None = field(default=None, metadata={
        "help": "Existing single document-end token; defaults to tokenizer EOS (may be a chat-end token)"})
    map_batch_size: int = 1000
    isolate_documents: bool = True
    reset_document_positions: bool = True
    data_cache_dir: str | None = field(default=None, metadata={
        "help": "Optional shared HF cache for local Parquet loading; no separate token export"})
    eval_data_dir: str | None = field(default=None, metadata={
        "help": "Separate local held-out text; enable with --eval_strategy steps --eval_steps N"})
    eval_max_sequences: int | None = field(default=None, metadata={
        "help": "Optional fixed seed-0 prefix of packed validation sequences; default is the full split"})


@dataclass
class BaselineTrainingArguments(TrainingArguments):
    stop_after: int | None = field(default=None, metadata={
        "help": "Absolute optimizer step to stop/save at; max_steps remains the full LR schedule"})

    @property
    def device(self):
        device = super().device
        # Accelerate may report cpu:0 in MULTI_CPU. Trainer uses this as
        # torch.load(map_location=...), which rejects indexed CPU locations.
        if device.type == "cpu":
            import torch
            return torch.device("cpu")
        return device

    def __post_init__(self):
        super().__post_init__()
        if self.stop_after is not None:
            if type(self.stop_after) is not int or not 1 <= self.stop_after <= self.max_steps:
                raise ValueError("stop_after must be an integer in [1, max_steps]; set max_steps to the full schedule")
            if self.save_only_model:
                raise ValueError("stop_after requires save_only_model=False for a resumable checkpoint")
            if self.ignore_data_skip:
                raise ValueError("stop_after requires ignore_data_skip=False to preserve resumed data order")


def check_resume_steps(checkpoint, args):
    """Do not accidentally compress a saved full schedule when resuming a pilot."""
    state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
    config_path = Path(checkpoint) / "train_config.json"
    previous_max_steps = (json.loads(config_path.read_text())["training"].get("max_steps", -1)
                          if config_path.is_file() else -1)
    if args.max_steps > 0 or previous_max_steps > 0:
        if state["max_steps"] != args.max_steps:
            raise ValueError("Resume must keep the checkpoint's full max_steps schedule; change stop_after instead")
        end = args.stop_after if args.stop_after is not None else args.max_steps
        if state["global_step"] >= end:
            raise ValueError("Checkpoint already reached the requested stopping step; choose a later stop_after")


class StopAfterCallback(TrainerCallback):
    """Use HF's normal checkpoint path after a complete optimizer update."""
    def on_train_begin(self, args, state, control, **kwargs):
        if args.stop_after is not None and state.global_step >= args.stop_after:
            raise ValueError("Checkpoint already reached stop_after; refusing an extra optimizer step")

    def on_step_end(self, args, state, control, **kwargs):
        if args.stop_after is not None and state.global_step >= args.stop_after:
            control.should_training_stop = control.should_save = control.should_log = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if args.stop_after is not None and state.global_step != args.stop_after:
            raise RuntimeError(f"Training ended at {state.global_step}, expected stop_after={args.stop_after}")


def build_model(model_args, isolate_documents=True):
    """Model extension point; keep the baseline Trainer/data pipeline unchanged."""
    if model_args.attention_backend not in ("sdpa", "fa4"):
        raise ValueError("attention_backend must be sdpa or fa4")
    if model_args.attention_backend == "fa4" and not isolate_documents:
        raise ValueError("FA4 currently requires document isolation")
    config_path = model_args.config_name or model_args.model_name_or_path
    if config_path is None or not Path(config_path).exists():
        raise ValueError("Supply an existing local --config_name or --model_name_or_path")
    kwargs = dict(cache_dir=model_args.cache_dir, local_files_only=True,
                  trust_remote_code=model_args.trust_remote_code)
    if Path(config_path).is_file():
        with open(config_path) as handle:
            config_dict = json.load(handle)
        model_type = config_dict.pop("model_type")
        config = AutoConfig.for_model(model_type, **config_dict)
    else:
        config = AutoConfig.from_pretrained(config_path, **kwargs)
    if isolate_documents:
        if config.model_type != "qwen3" or any(t != "full_attention" for t in config.layer_types):
            raise ValueError("Document isolation is validated for full-attention Qwen3 only; validate other architectures before enabling it")
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("Document isolation requires validated default RoPE; sequence-dependent RoPE needs separate validation")
    backend = "sdpa"
    if model_args.attention_backend == "fa4":
        if config.attention_dropout != 0.:
            raise ValueError("FA4 requires attention_dropout=0; dropout is never silently dropped")
        from packed_fa4 import register_backend
        backend = register_backend()
    # Native HF layers retain projection, QK norm, RoPE and output projection.
    config.use_cache = False
    if model_args.model_name_or_path:
        if not Path(model_args.model_name_or_path).is_dir():
            raise ValueError("model_name_or_path must be a local pretrained checkpoint directory")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, attn_implementation=backend, **kwargs)
    else:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=model_args.trust_remote_code, attn_implementation=backend)
    return model


def save_train_config(save_dir, model_args, data_args, training_args):
    config = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": {k: v for k, v in training_args.to_dict().items()
                     if v is not None and v != "" and k not in ("_n_gpu", "local_rank", "hub_token")},
    }
    with open(os.path.join(save_dir, "train_config.json"), "w") as handle:
        json.dump(config, handle, indent=2, default=str)


def prepare_validation_data(data_args, training_args, tokenizer, tokenizer_name,
                            block_size, backend, train_recipe):
    from packed_evaluation import preserve_rng_and_mode
    # Do not construct new TrainingArguments: that resets shared Accelerate state.
    eval_args = copy(training_args)
    eval_args.seed = eval_args.data_seed = 0
    eval_data = replace(data_args, data_dir=data_args.eval_data_dir)
    with preserve_rng_and_mode():
        dataset, recipe = prepare_training_data(eval_data, eval_args, tokenizer, tokenizer_name,
                                                block_size, attention_backend=backend)
    require_held_out(train_recipe, recipe)
    full_size = len(dataset)
    if data_args.eval_max_sequences is not None:
        if data_args.eval_max_sequences < 1:
            raise ValueError("eval_max_sequences must be positive")
        dataset = dataset.select(range(min(data_args.eval_max_sequences, full_size)))
    receipt = dict(data_recipe=recipe, selection=dict(full_sequences=full_size,
        selected_sequences=len(dataset), max_sequences=data_args.eval_max_sequences, shuffle_seed=0))
    return dataset, receipt


class SaveConfigCallback(TrainerCallback):
    def __init__(self, model_args, data_args, data_recipe, training_recipe):
        self.model_args, self.data_args = model_args, data_args
        self.data_recipe = data_recipe
        self.training_recipe = training_recipe

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            save_train_config(checkpoint_dir, self.model_args, self.data_args, args)
            save_data_recipe(checkpoint_dir, self.data_recipe)
            save_training_recipe(checkpoint_dir, self.training_recipe)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, BaselineTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Only offline reporting is supported by this runner-facing entry point.
    if set(training_args.report_to or []) - {"wandb", "tensorboard"}:
        raise ValueError("Use report_to=none, wandb (offline), or tensorboard")
    wants_eval = training_args.do_eval or training_args.eval_strategy != "no" or training_args.eval_on_start
    if bool(data_args.eval_data_dir) != bool(wants_eval):
        raise ValueError("Provide eval_data_dir together with do_eval, eval_on_start or eval_strategy")
    if data_args.eval_max_sequences is not None and (not wants_eval or data_args.eval_max_sequences < 1):
        raise ValueError("eval_max_sequences requires enabled validation and a positive limit")
    if not Path(data_args.data_dir).exists():
        raise ValueError("data_dir must be an existing local dataset directory or Parquet file")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.warning("rank=%s device=%s world_size=%s bf16=%s fp16=%s",
                   training_args.process_index, training_args.device,
                   training_args.world_size, training_args.bf16, training_args.fp16)
    set_seed(training_args.seed)
    if model_args.attention_backend == "fa4" and (
            training_args.device.type != "cuda" or not (training_args.bf16 or training_args.fp16)):
        raise ValueError("FA4 training requires a CUDA device and --bf16 true or --fp16 true")

    # Preserve ordinary Trainer resume; reject incompatible/incomplete checkpoints first.
    last_checkpoint = None
    output = Path(training_args.output_dir)
    if output.is_dir():
        last_checkpoint = get_last_checkpoint(str(output))
        if any(output.iterdir()) and not (last_checkpoint or training_args.resume_from_checkpoint):
            raise ValueError("Output directory is nonempty without a checkpoint; choose a fresh directory")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if checkpoint is not None and not Path(checkpoint).is_dir():
        raise ValueError("Resume checkpoint must be an existing local directory")
    if checkpoint is not None:
        check_resume_steps(checkpoint, training_args)

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name is None or not Path(tokenizer_name).is_dir():
        raise ValueError("Supply a local tokenizer directory (required for random initialization)")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, cache_dir=model_args.cache_dir, local_files_only=True,
        trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(model_args, data_args.isolate_documents)
    config = model.config
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
    logger.info("Parameters: %s total, %s trainable",
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad))

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

    if block_size < 2:
        raise ValueError("block_size must be at least 2 for next-token training")

    train_dataset, data_recipe = prepare_training_data(
        data_args, training_args, tokenizer, tokenizer_name, block_size,
        attention_backend=model_args.attention_backend)
    if checkpoint is not None:
        require_resume_data(checkpoint, data_recipe)
    logger.info("Training: %s sequences, document-end token %s; data fingerprint %s",
                len(train_dataset), data_recipe["document_end"], data_recipe["shuffled_fingerprint"])

    eval_dataset = validation_recipe = None
    if wants_eval:
        eval_dataset, validation_recipe = prepare_validation_data(data_args, training_args,
            tokenizer, tokenizer_name, block_size, model_args.attention_backend, data_recipe)

    # Keep segment IDs until the collator. HF retains native model loss/optimizer.
    training_args.remove_unused_columns = False
    trainer = PackedTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer, data_collator=PackedCollator(
            data_args.isolate_documents, data_args.reset_document_positions, model_args.attention_backend),
        callbacks=[StopAfterCallback()],
    )
    training_recipe = build_training_recipe(model, training_args, model_args.attention_backend, trainer.accelerator)
    training_recipe["validation"] = validation_recipe
    if wants_eval:
        trainer._validate_packed_evaluation()
    if checkpoint is not None:
        require_resume_training(checkpoint, training_recipe, training_args)
    trainer.add_callback(SaveConfigCallback(model_args, data_args, data_recipe, training_recipe))
    if training_args.should_save:
        save_data_recipe(training_args.output_dir, data_recipe)
        save_training_recipe(training_args.output_dir, training_recipe)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # Periodic/start evaluations are driven by Trainer. do_eval alone requests
    # one final evaluation, as in the ordinary HF training examples.
    if training_args.do_eval and training_args.eval_strategy == "no":
        trainer.save_metrics("eval", trainer.evaluate())
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["global_step"] = trainer.state.global_step
    metrics["schedule_steps"] = trainer.state.max_steps
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if training_args.should_save:
        save_train_config(training_args.output_dir, model_args, data_args, training_args)
    logger.info("Training complete. Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    main()
