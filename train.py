"""Four-arm Qwen pretraining using the baseline HF Trainer/Accelerate pipeline.

Accepts standard TrainingArguments plus model/data/arm arguments, or one JSON
config. All assets are local. Packing is the unchanged two-map EOS CLM pipeline.
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline")
# Match the original Qwen shell launch even when the queue invokes Python
# directly. Set this before importing HF/Accelerate or initializing NCCL.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import json
import logging
import copy
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

import datasets
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerState

from deep_kv.model import DeepKV
from deep_kv.packing import preprocess_dataset
from deep_kv.training import DeepKVTrainer, PilotCallback, compute_metrics, offline_wandb_run

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    config_name: str = field(metadata={"help": "Local Qwen3 config file or directory; random initialization"})
    tokenizer_name: str = field(metadata={"help": "Local tokenizer directory"})


@dataclass
class DataArguments:
    data_dir: str = field(metadata={"help": "Saved text: split root, language directory, or Arrow dataset"})
    eval_data_dir: str = field(metadata={"help": "Separate held-out text directory"})
    block_size: int = 2048
    preprocessing_num_workers: int = 160
    overwrite_cache: bool = False
    eval_rows: int = 4882
    monitor_rows: int = 128


@dataclass
class PilotArguments:
    arm: str = "A"
    consumer: int = 5
    deep_target: int = 21
    lm_chunk: int = 128
    checkpoint_layers: bool = True
    checkpoint_lm: bool = True
    checkpoint_aux: bool = True
    causal_attention: bool = False
    allow_performance_change_on_resume: bool = False
    stop_after: int | None = None


def resume_performance_changes(previous, requested, allow=False):
    """Keep the scientific recipe strict; explicitly permit execution-only changes."""
    before, after = copy.deepcopy(previous), copy.deepcopy(requested)
    defaults = dict(checkpoint_layers=True, checkpoint_lm=True, checkpoint_aux=True,
                    causal_attention=False, lm_chunk=128)
    changes = {}
    for key, default in defaults.items():
        old = before['pilot'].get(key, default)
        new = after['pilot'].get(key, default)
        before['pilot'][key] = after['pilot'][key] = default
        if old != new:
            changes[key] = dict(before=old, after=new)
    if before != after:
        raise ValueError("Resume configuration/data differs from the saved run")
    if changes and not allow:
        raise ValueError("Resume configuration changes require allow_performance_change_on_resume")
    return changes


def load_text(directory):
    """The baseline's sorted language/shard concatenation; accepts one language too."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Missing sampled text: {root}")
    def shards(path):
        if (path / "state.json").is_file():
            return [path]
        return sorted(p for p in path.glob("shard_*") if p.is_dir())
    paths = shards(root)
    if not paths:
        for language in sorted(p for p in root.iterdir() if p.is_dir()):
            found = shards(language)
            if not found:
                raise ValueError(f"Missing Arrow dataset: {language}")
            paths.extend(found)
    if not paths:
        raise ValueError(f"No datasets found in {root}")
    parts = [datasets.load_from_disk(str(p)) for p in paths]
    return datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, PilotArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pilot, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pilot, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], level=training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    if transformers.__version__ != "5.9.0":
        raise ValueError("Use the pinned Transformers 5.9.0 environment")
    if training_args.max_steps <= 0:
        raise ValueError("Set max_steps to the full training schedule; stop_after is the optional cutoff")
    end = training_args.max_steps if pilot.stop_after is None else pilot.stop_after
    if type(end) is not int or not 1 <= end <= training_args.max_steps:
        raise ValueError("stop_after must be an integer within the full training schedule")
    if not 0 < data_args.monitor_rows <= data_args.eval_rows or data_args.block_size < 2:
        raise ValueError("Invalid context/evaluation sizes")
    if Path(data_args.data_dir).resolve() == Path(data_args.eval_data_dir).resolve():
        raise ValueError("Training and validation directories must differ")
    if training_args.gradient_checkpointing:
        raise ValueError("Use checkpoint_layers for the custom model's checkpointing")
    # The wrapper consumes Context, so Trainer must retain the CLM input columns.
    training_args.remove_unused_columns = False
    training_args.ddp_find_unused_parameters = False
    set_seed(training_args.seed)

    if Path(model_args.config_name).is_file():
        config = AutoConfig.for_model(**json.loads(Path(model_args.config_name).read_text()))
    else:
        config = AutoConfig.from_pretrained(model_args.config_name, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("This pilot implements Qwen3")
    config._attn_implementation = "sdpa"
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if data_args.block_size > min(tokenizer.model_max_length, config.max_position_embeddings):
        raise ValueError("block_size exceeds the model/tokenizer context limit")
    model = DeepKV.from_scratch(config, pilot.arm, seed=training_args.seed, consumer=pilot.consumer,
                               deep_target=pilot.deep_target, lm_chunk=pilot.lm_chunk,
                               checkpoint_layers=pilot.checkpoint_layers,
                               checkpoint_lm=pilot.checkpoint_lm, checkpoint_aux=pilot.checkpoint_aux,
                               causal_attention=pilot.causal_attention)
    raw_train, raw_eval = load_text(data_args.data_dir), load_text(data_args.eval_data_dir)
    train_dataset = preprocess_dataset(raw_train, tokenizer, data_args.block_size, training_args,
                                      num_proc=data_args.preprocessing_num_workers,
                                      overwrite_cache=data_args.overwrite_cache).shuffle(seed=training_args.seed)
    # Keep the existing single-worker validation packing and fixed prefix.
    eval_dataset = preprocess_dataset(raw_eval, tokenizer, data_args.block_size, training_args,
                                     num_proc=1, overwrite_cache=data_args.overwrite_cache)
    rows_per_update = (training_args.per_device_train_batch_size * training_args.world_size
                       * training_args.gradient_accumulation_steps)
    if len(train_dataset) < training_args.max_steps * rows_per_update or len(eval_dataset) < data_args.eval_rows:
        raise ValueError("Insufficient packed data for the full training/evaluation budget")
    eval_dataset = eval_dataset.select(range(data_args.eval_rows))
    tokens_per_update = rows_per_update * data_args.block_size

    # Ordinary config recording also guards matched-arm comparisons and resume.
    settings = training_args.to_dict()
    for key in ("output_dir", "logging_dir", "run_name", "resume_from_checkpoint", "local_rank"):
        settings.pop(key, None)
    experiment = json.loads(json.dumps({"model": asdict(model_args), "model_config": config.to_dict(),
                  "data": asdict(data_args), "pilot": {k: v for k, v in asdict(pilot).items()
                      if k not in ("stop_after", "allow_performance_change_on_resume")},
                  "training": settings, "world_size": training_args.world_size,
                  "tokens_per_update": tokens_per_update,
                  "train_fingerprint": train_dataset._fingerprint, "eval_fingerprint": eval_dataset._fingerprint}))
    output = Path(training_args.output_dir)
    checkpoint = training_args.resume_from_checkpoint or (get_last_checkpoint(str(output)) if output.is_dir() else None)
    if checkpoint:
        if Path(checkpoint).resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
        previous = json.loads((output / "train_config.json").read_text())
        changes = resume_performance_changes(previous, experiment, pilot.allow_performance_change_on_resume)
        state = TrainerState.load_from_json(str(Path(checkpoint) / "trainer_state.json"))
        if state.max_steps != training_args.max_steps:
            raise ValueError("Checkpoint training schedule differs from the saved run")
        if state.global_step > end:
            raise ValueError("Cutoff precedes checkpoint")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a resumable checkpoint; use a fresh directory")
    # Finish every rank's fresh/resume checks before rank zero creates outputs.
    training_args.distributed_state.wait_for_everyone()
    if training_args.should_save:
        output.mkdir(parents=True, exist_ok=True)
        if checkpoint and changes:
            # Preserve the exact prior recipe before replacing the current one.
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            record = dict(checkpoint=str(Path(checkpoint).resolve()), global_step=state.global_step,
                          changes=changes, previous=previous, requested=experiment)
            with (output / f'resume-transition-{state.global_step}-{stamp}.json').open('x') as handle:
                json.dump(record, handle, indent=2, allow_nan=False)
        temporary_config = output / "train_config.json.tmp"
        temporary_config.write_text(json.dumps(experiment, indent=2) + "\n")
        temporary_config.replace(output / "train_config.json")
        (output / "result.json").unlink(missing_ok=True)
    training_args.distributed_state.wait_for_everyone()

    callback = PilotCallback(end, tokens_per_update)
    with offline_wandb_run(training_args):
        trainer = DeepKVTrainer(model=model, args=training_args, train_dataset=train_dataset,
                               eval_dataset=eval_dataset.select(range(data_args.monitor_rows)),
                               processing_class=tokenizer, data_collator=default_data_collator,
                               compute_metrics=compute_metrics, callbacks=[callback])
        callback.trainer = trainer
        if checkpoint and state.global_step == end:
            # HF train() can take a step even at an already reached cutoff.
            trainer._load_from_checkpoint(checkpoint)
            trainer.state = state
        else:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
        if trainer.state.global_step != end:
            raise ValueError("Training stopped outside the requested iteration")
        trainer.save_model()
        trainer.save_state()
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.accelerator.wait_for_everyone()
        if training_args.should_save:
            result = {"arm": pilot.arm, "global_step": end, "schedule_steps": training_args.max_steps,
                      "input_tokens": end * tokens_per_update,
                      "status": "complete" if end == training_args.max_steps else "stopped", "evaluation": metrics}
            temporary = output / "result.json.tmp"
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "result.json")


if __name__ == "__main__":
    main()
