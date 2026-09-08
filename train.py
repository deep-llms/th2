"""Train a six-layer Qwen3 capacity-allocation arm from sampled HF text datasets.

CLI/data flow follows sparse_embedding/train.py: HfArgumentParser,
TrainingArguments, cached multiprocess dataset.map, shuffle, Trainer. Model
construction lives in capacity_allocation/modeling.py. No network/GPU control.
"""
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import (AutoTokenizer, HfArgumentParser, Trainer, TrainerCallback,
                          TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from capacity_allocation.data import load_text_data, preprocess_text, sha256, write_json
from capacity_allocation.modeling import ARMS, activation_report, build_model, experiment_config, parameter_report


@dataclass
class ModelArguments:
    tokenizer_name: str = field(metadata={'help': 'Local Qwen3 tokenizer directory; never downloaded'})
    arm: str = 'B0'
    num_hidden_layers: int = 6
    attn_implementation: str = 'sdpa'
    tiny_test: bool = False


@dataclass
class DataArguments:
    data_dir: str = field(metadata={'help': 'Old sampled train/ directory containing en/ etc.'})
    eval_data_dir: str | None = None
    languages: str = 'en'
    block_size: int = 2048
    preprocessing_num_workers: int = 16
    preprocessing_batch_size: int = 1000
    preprocessing_cache_dir: str | None = None
    overwrite_cache: bool = False


@dataclass
class RunArguments:
    stop_at_step: int | None = field(default=None, metadata={
        'aliases': ['--stop-at-step'], 'help': 'Graceful cutoff; does not change epoch/LR schedule'})


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step is not None and state.global_step >= self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


class CausalTrainer(Trainer):
    def _get_num_items_in_batch(self, batch_samples, device):
        # HF 5.9 includes the first label in its default denominator; causal LM
        # loss predicts only labels[:, 1:]. Preserve HF accumulation/DDP scaling.
        shifted = [{'labels': batch['labels'][:, 1:]} for batch in batch_samples]
        return super()._get_num_items_in_batch(shifted, device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')
        with torch.no_grad(), self.compute_loss_context_manager():
            logits = model(**inputs, use_cache=False).logits
            targets = labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().transpose(1, 2), targets,
                ignore_index=-100, reduction='none')
            # Per-example sums/counts allow Accelerate to trim DDP padding.
            scores = torch.stack((losses.sum(dim=1, dtype=torch.float64),
                                  targets.ne(-100).sum(dim=1).double()), dim=1)
        return None, scores, torch.zeros_like(scores[:, :1])


def nll_metrics(prediction):
    scores = np.asarray(prediction.predictions, dtype=np.float64)
    count = scores[:, 1].sum()
    if count <= 0:
        raise ValueError('No scored validation targets')
    return {'loss': float(scores[:, 0].sum()/count), 'scored_targets': int(count)}


def validate_resume_checkpoint(checkpoint, world_size, *, needs_scaler=False):
    """Fail closed on partial ordinary Trainer/DDP checkpoints before resuming."""
    path = Path(checkpoint)
    required = ['trainer_state.json', 'config.json', 'optimizer.pt', 'scheduler.pt']
    required += (['rng_state.pth'] if world_size == 1 else
                 [f'rng_state_{rank}.pth' for rank in range(world_size)])
    if needs_scaler:
        required.append('scaler.pt')
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text())['weight_map']
        if not mapping or any(Path(name).name != name for name in mapping.values()):
            raise ValueError('Invalid checkpoint weight index')
        required += sorted(set(mapping.values()))
    else:
        required.append('model.safetensors')
    missing = [name for name in required
               if not (path/name).is_file() or (path/name).stat().st_size == 0]
    if missing:
        raise ValueError(f'Incomplete resume checkpoint; missing/nonempty state required: {missing}')
    state = json.loads((path/'trainer_state.json').read_text())
    if path.name != f"checkpoint-{state['global_step']}":
        raise ValueError('Checkpoint directory and saved global_step disagree')
    return state


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, RunArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args, run_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, run_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if training_args.push_to_hub:
        raise ValueError('Runner training is offline; exporting results uses the run system, not Hub uploads')
    if training_args.label_smoothing_factor != 0:
        raise ValueError('label_smoothing_factor must be zero: all arms use ordinary next-token cross entropy')
    if training_args.save_only_model:
        raise ValueError('save_only_model must be false: checkpoints must retain optimizer/scheduler/RNG state')
    if training_args.deepspeed or training_args.fsdp:
        raise ValueError('This training/resume implementation supports ordinary Trainer/DDP, not DeepSpeed/FSDP')
    if model_args.arm not in ARMS or model_args.attn_implementation not in ('eager', 'sdpa'):
        raise ValueError('Unknown arm or unsupported attention implementation')
    if training_args.max_steps > 0:
        raise ValueError('Use num_train_epochs for the LR horizon and stop_at_step for the cutoff, not max_steps')
    if training_args.num_train_epochs <= 0 or (run_args.stop_at_step is not None and run_args.stop_at_step <= 0):
        raise ValueError('Epochs and optional stop_at_step must be positive')
    if training_args.n_gpu > 1:
        raise ValueError('Use accelerate/torchrun with one process per GPU, not DataParallel')
    if training_args.prediction_loss_only:
        raise ValueError('prediction_loss_only must be false for exact validation aggregation')
    if training_args.eval_strategy != 'no' and not data_args.eval_data_dir:
        raise ValueError('eval_data_dir is required for periodic evaluation')
    if data_args.eval_data_dir and Path(data_args.eval_data_dir).resolve() == Path(data_args.data_dir).resolve():
        raise ValueError('Training and evaluation directories must be separate')
    root = Path(training_args.output_dir).resolve()
    checkpoint = training_args.resume_from_checkpoint or (get_last_checkpoint(str(root)) if root.is_dir() else None)
    if (root/'result.json').exists():
        raise ValueError('Completed run exists; choose a fresh output directory')
    if root.exists() and any(root.iterdir()) and not checkpoint:
        raise ValueError('Nonempty output without a resumable checkpoint; refusing overwrite')
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
        if Path(checkpoint).parent != root or not (Path(checkpoint)/'trainer_state.json').is_file():
            raise ValueError('Resume checkpoint must be inside this run')
        state = validate_resume_checkpoint(checkpoint, training_args.world_size,
                                           needs_scaler=training_args.fp16 and not training_args.use_cpu)
        step = state['global_step']
        if run_args.stop_at_step is not None and run_args.stop_at_step <= step:
            raise ValueError('Stop must be after the resumed checkpoint')

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, local_files_only=True)
    tokenizer.model_max_length = 10**30  # Never truncate documents to the tokenizer context limit.
    config = experiment_config(model_args.arm, depth=model_args.num_hidden_layers,
                               tiny=model_args.tiny_test, attention=model_args.attn_implementation)
    if not model_args.tiny_test and (tokenizer.eos_token_id != 151645 or len(tokenizer) > 151936
                                    or data_args.block_size != 2048):
        raise ValueError('Production requires the Qwen3 tokenizer and 2048-token blocks')
    if len(tokenizer) > config.vocab_size:
        raise ValueError('Tokenizer exceeds model vocabulary')
    if not tokenizer.is_fast:
        raise ValueError('Use the fast Qwen tokenizer for batched preprocessing')
    languages = tuple(data_args.languages.split(','))
    options = dict(block_size=data_args.block_size, num_proc=data_args.preprocessing_num_workers,
                   batch_size=data_args.preprocessing_batch_size, cache_dir=data_args.preprocessing_cache_dir,
                   overwrite_cache=data_args.overwrite_cache)
    with training_args.main_process_first(desc='dataset map tokenization and packing'):
        train_dataset = preprocess_text(load_text_data(data_args.data_dir, languages), tokenizer, **options)
        eval_dataset = (preprocess_text(load_text_data(data_args.eval_data_dir, languages), tokenizer, **options)
                        if data_args.eval_data_dir else None)
    train_dataset = train_dataset.shuffle(seed=training_args.data_seed if training_args.data_seed is not None
                                          else training_args.seed)
    logging.info('Training: %s blocks; evaluation: %s blocks', len(train_dataset),
                 len(eval_dataset) if eval_dataset is not None else 0)
    specification = dict(model=asdict(model_args), data=asdict(data_args),
                         execution=dict(world_size=training_args.world_size,
                             effective_batch_size=training_args.world_size *
                             training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps),
                         training={k:v for k,v in training_args.to_dict().items()
                                   if k not in ('resume_from_checkpoint', 'local_rank', 'hub_token')},
                         train_fingerprint=train_dataset._fingerprint,
                         eval_fingerprint=eval_dataset._fingerprint if eval_dataset is not None else None,
                         code={p.name:sha256(p) for p in [Path(__file__)] +
                               sorted((Path(__file__).parent/'capacity_allocation').glob('*.py'))})
    with training_args.main_process_first(desc='record training configuration'):
        if training_args.process_index == 0:
            if checkpoint:
                if json.loads((root/'train_config.json').read_text()) != specification:
                    raise ValueError('Resume data/config/code differs from original run')
            else:
                root.mkdir(parents=True, exist_ok=True)
                write_json(root/'train_config.json', specification)

    set_seed(training_args.seed)
    model = build_model(config)
    counts = parameter_report(model)
    trainer = CausalTrainer(model=model, args=training_args, train_dataset=train_dataset,
                            eval_dataset=eval_dataset, processing_class=tokenizer,
                            data_collator=default_data_collator, compute_metrics=nll_metrics,
                            callbacks=[StopAtStep(run_args.stop_at_step)])
    if trainer.is_world_process_zero() and not checkpoint:
        write_json(root/'parameters.json', counts)
        batch = train_dataset[0]['input_ids'][:128].unsqueeze(0).to(training_args.device)
        write_json(root/'initial_scales.json', activation_report(model, batch))
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    expected = min(trainer.state.max_steps, run_args.stop_at_step or trainer.state.max_steps)
    if trainer.state.global_step != expected:
        raise RuntimeError(f'Training stopped at {trainer.state.global_step}; expected {expected}')
    if not math.isfinite(train_result.training_loss):
        raise RuntimeError('Non-finite training loss')
    evaluation = trainer.evaluate() if eval_dataset is not None else {}
    if eval_dataset is not None:
        if not math.isfinite(evaluation['eval_loss']):
            raise RuntimeError('Non-finite evaluation loss')
        if evaluation['eval_scored_targets'] != len(eval_dataset)*(data_args.block_size-1):
            raise RuntimeError('Evaluation target count mismatch')
    trainer.save_model(str(root/'final'))
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        write_json(root/'result.json', dict(success=True, arm=model_args.arm,
            status='stopped_at_step' if expected < trainer.state.max_steps else 'horizon_complete',
            global_step=trainer.state.global_step, schedule_steps=trainer.state.max_steps,
            parameters=counts, train_metrics=train_result.metrics, eval_metrics=evaluation))


if __name__ == '__main__':
    main()
