"""Short real-Trainer batch/accumulation benchmark; never resume/save research weights."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'offline'

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, default_data_collator, set_seed
from capacity_allocation.data import load_text_data, preprocess_text, write_json
from capacity_allocation.modeling import ARMS, build_model, experiment_config, parameter_report
from train import CausalTrainer


class MeasureUpdates(TrainerCallback):
    def __init__(self, warmup, measured):
        if warmup < 1 or measured < 2:
            raise ValueError('Require warmup >= 1 and at least two measured optimizer steps')
        self.warmup, self.measured = warmup, measured
        self.durations = []

    @staticmethod
    def sync(args):
        if args.device.type == 'cuda':
            torch.cuda.synchronize(args.device)

    def on_train_begin(self, args, state, control, **kwargs):
        if args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)

    def on_step_begin(self, args, state, control, **kwargs):
        self.sync(args)
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.sync(args)
        elapsed = time.perf_counter()-self.started
        if state.global_step > self.warmup:
            self.durations.append(elapsed)
        if state.global_step >= self.warmup+self.measured:
            control.should_training_stop = True
            control.should_save = False
        return control


def summarize(ranks, effective_batch, block_size):
    lengths = {len(row['durations']) for row in ranks}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError('Incomplete per-rank timing coverage')
    # Slowest rank determines each synchronous optimizer update.
    durations = [max(values) for values in zip(*(row['durations'] for row in ranks))]
    if not all(math.isfinite(value) and value > 0 for value in durations):
        raise ValueError('Invalid timings')
    median = statistics.median(durations)
    return dict(median_optimizer_step_s=median, mean_optimizer_step_s=statistics.mean(durations),
        input_tokens_per_second=effective_batch*block_size/median,
        measured_step_seconds=durations,
        peak_allocated_gib=max(row['peak_allocated_gib'] for row in ranks),
        peak_reserved_gib=max(row['peak_reserved_gib'] for row in ranks), ranks=ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--accumulation', type=int, required=True)
    parser.add_argument('--warmup-updates', type=int, default=5)
    parser.add_argument('--measure-updates', type=int, default=20)
    parser.add_argument('--data-root')
    parser.add_argument('--tokenizer')
    parser.add_argument('--cache-dir')
    parser.add_argument('--cache-report')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tiny-test', action='store_true', help='CPU-only synthetic unit test')
    args = parser.parse_args()
    if args.batch_size < 1 or args.accumulation < 1:
        parser.error('Positive batch/accumulation required')
    root = Path(args.output_dir)
    block_size = 16 if args.tiny_test else 2048
    settings = TrainingArguments(output_dir=str(root), use_cpu=args.tiny_test,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        num_train_epochs=1, bf16=not args.tiny_test,
        learning_rate=3e-4, lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': 0.1}, warmup_steps=500,
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.,
        seed=42, data_seed=42, ddp_find_unused_parameters=False, ddp_timeout=1800,
        save_strategy='no', eval_strategy='no', logging_steps=1, logging_nan_inf_filter=False,
        dataloader_num_workers=0 if args.tiny_test else 8, report_to='none', disable_tqdm=True)
    effective = settings.world_size*args.batch_size*args.accumulation
    if not args.tiny_test and (settings.world_size != 8 or effective != 512):
        raise ValueError('Production benchmark requires eight GPUs and effective batch512')
    with settings.main_process_first(desc='fresh benchmark output'):
        if settings.process_index == 0:
            root.mkdir(parents=True, exist_ok=False)
    measure = MeasureUpdates(args.warmup_updates, args.measure_updates)
    set_seed(42)
    tokenizer = None
    if args.tiny_test:
        ids = torch.randint(0, 97, ((args.warmup_updates+args.measure_updates+3)*effective, block_size)).tolist()
        data = Dataset.from_dict(dict(input_ids=ids, labels=ids)).with_format('torch')
    else:
        if not all((args.data_root, args.tokenizer, args.cache_dir, args.cache_report)):
            raise ValueError('Production requires local data/tokenizer/prepared-cache report')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        tokenizer.model_max_length = 10**30
        if tokenizer.eos_token_id != 151645 or len(tokenizer) > 151936:
            raise ValueError('Expected original Qwen3 tokenizer')
        with settings.main_process_first(desc='reuse verified training cache'):
            data = preprocess_text(load_text_data(Path(args.data_root)/'train', ('en',)), tokenizer,
                block_size=2048, num_proc=160, batch_size=1000, cache_dir=args.cache_dir).shuffle(seed=42)
        prepared = json.loads(Path(args.cache_report).read_text())
        if not prepared['success'] or data._fingerprint != prepared['train']['fingerprint']:
            raise ValueError('Benchmark data differs from current research training')
    if len(data) < effective*(args.warmup_updates+args.measure_updates+1):
        raise ValueError('Insufficient data for full benchmark updates')
    set_seed(42)
    model = build_model(experiment_config(args.arm, tiny=args.tiny_test, attention='sdpa'))
    model.config.use_cache = False
    trainer = CausalTrainer(model=model, args=settings, train_dataset=data,
        processing_class=tokenizer, data_collator=default_data_collator, callbacks=[measure])
    started = time.monotonic()
    trained = trainer.train()
    elapsed = time.monotonic()-started
    if trainer.state.global_step != args.warmup_updates+args.measure_updates or len(measure.durations) != args.measure_updates:
        raise RuntimeError('Benchmark step count mismatch')
    if not math.isfinite(trained.training_loss) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError('Nonfinite benchmark weights/loss')
    device = settings.device
    row = dict(rank=settings.process_index, durations=measure.durations, training_wall_s=elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else 0.)
    rows = [None]*settings.world_size
    if dist.is_initialized():
        dist.all_gather_object(rows, row)
    else:
        rows = [row]
    if settings.process_index == 0:
        result = dict(success=True, arm=args.arm, batch_size=args.batch_size, accumulation=args.accumulation,
            effective_batch=effective, block_size=block_size, world_size=settings.world_size,
            tiny=args.tiny_test, warmup_updates=args.warmup_updates, measured_updates=args.measure_updates,
            train_fingerprint=data._fingerprint, parameters=parameter_report(model),
            metrics=summarize(rows, effective, block_size))
        write_json(root/'result.json', result)
        print('BATCH_BENCHMARK_COMPLETE', json.dumps(result), flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
