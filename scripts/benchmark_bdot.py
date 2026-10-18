"""Synthetic complete-update timing through the project's actual HF Trainer.

Use an explicitly allocated CUDA_VISIBLE_DEVICES, optionally torchrun for DDP.
No dataset download, experiment launch, GPU allocation or process management.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', required=True, choices=['cpu', 'cuda', 'cuda:0'])
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--document-length', type=int, default=128)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--attention-backend', choices=['sdpa', 'fa4'], default='sdpa')
    p.add_argument('--progress', type=float, default=1.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--router-prior', type=float, help='Diagnostic override; record separately from production config')
    p.add_argument('--expert-backend', choices=['reference', 'grouped'], default='reference')
    a = p.parse_args()
    if min(a.batch_size, a.accumulation, a.steps, a.warmup) < 1 or min(a.length, a.document_length) < 2:
        p.error('Invalid benchmark sizes')
    if not 0 <= a.progress <= 1:
        p.error('progress must be in [0, 1]')
    out = Path(a.output).resolve()
    run_dir = out.with_name(out.stem + '_trainer')
    if out.exists() or run_dir.exists():
        p.error('Choose fresh receipt and Trainer output paths')
    if a.device != 'cpu' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Set CUDA_VISIBLE_DEVICES to the explicitly allocated physical GPU(s)')
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from train import ModelArguments, BaselineTrainingArguments, build_model
    from packed_training import PackedCollator, PackedTrainer
    from bdot.trainer import BDOTTrainer

    args = BaselineTrainingArguments(
        output_dir=str(run_dir), use_cpu=a.device == 'cpu', bf16=a.bf16,
        max_steps=a.warmup + a.steps, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.accumulation, gradient_checkpointing=a.checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        learning_rate=3e-4, adam_beta1=.9, adam_beta2=.95, weight_decay=.1,
        max_grad_norm=1., lr_scheduler_type='constant', optim='adamw_torch',
        save_strategy='no', logging_strategy='no', report_to='none',
        remove_unused_columns=False, disable_tqdm=True, dataloader_num_workers=0,
        dataloader_drop_last=True, train_sampling_strategy='sequential',
        seed=a.seed, data_seed=0, ddp_broadcast_buffers=False,
    )
    set_seed(a.seed)
    model = build_model(ModelArguments(config_name=a.config, attention_backend=a.attention_backend))
    if hasattr(model, 'options'):
        from dataclasses import replace
        model.options = replace(model.options, expert_backend=a.expert_backend)
        model.config.bdot = model.options.to_dict()
        for step in model.routed_steps:
            step.options = model.options
    elif a.expert_backend != 'reference':
        p.error('Grouped expert execution applies to B-DOT models only')
    if a.router_prior is not None:
        if not hasattr(model, 'routed_steps') or model.options.routing != 'learned':
            p.error('--router-prior requires learned routing')
        with torch.no_grad():
            for step in model.routed_steps:
                step.router.bias.zero_()
                step.router.bias[step.step] = a.router_prior
    gen = torch.Generator().manual_seed(0)
    count = args.world_size * a.batch_size * a.accumulation * (a.warmup + a.steps)
    ids = torch.randint(model.config.vocab_size, (count, a.length), generator=gen)
    segments = (torch.arange(a.length) // a.document_length).tolist()
    rows = [dict(input_ids=row.tolist(), labels=row.tolist(), attention_mask=[1]*a.length,
                 segments=segments) for row in ids]

    class Timing(TrainerCallback):
        def __init__(self):
            self.durations = []

        def sync(self):
            if args.device.type == 'cuda':
                torch.cuda.synchronize(args.device)

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            # Freeze the tested phase after the model's ordinary routing callback.
            if hasattr(model, 'progress'):
                model.progress = a.progress
            if state.global_step == a.warmup:
                self.sync()
                if args.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(args.device)
                self.total_start = time.perf_counter()
            if state.global_step >= a.warmup:
                self.sync()
                self.start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            # Registered after RoutingUpdateCallback: includes finish_update.
            if state.global_step > a.warmup:
                self.sync()
                self.durations.append(time.perf_counter() - self.start)
                self.total_elapsed = time.perf_counter() - self.total_start

    cls = BDOTTrainer if hasattr(model, 'options') else PackedTrainer
    trainer = cls(model=model, args=args, train_dataset=Dataset.from_list(rows),
                  data_collator=PackedCollator(attention_backend=a.attention_backend))
    timer = Timing()
    trainer.add_callback(timer)
    trainer.train()
    if len(timer.durations) != a.steps:
        raise AssertionError('Incomplete measured update count')
    elapsed = torch.tensor(timer.total_elapsed, device=args.device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
    elapsed = float(elapsed)
    if trainer.is_world_process_zero():
        from importlib.metadata import version, PackageNotFoundError
        versions = {}
        for package in ('torch', 'transformers', 'accelerate', 'datasets', 'flash-attn-4'):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        root = Path(__file__).resolve().parents[1]
        result = dict(
            status='completed', settings=vars(a), versions=versions, torch=torch.__version__,
            cuda=torch.version.cuda, world_size=args.world_size,
            gpu=torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else None,
            parameters=sum(p.numel() for p in model.parameters()), config=model.config.to_dict(),
            seconds=elapsed, seconds_per_update=elapsed/a.steps,
            rank0_update_seconds=timer.durations, rank0_median_update_seconds=statistics.median(timer.durations),
            tokens_per_second=args.world_size*a.batch_size*a.accumulation*a.length*a.steps/elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.type == 'cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if args.device.type == 'cuda' else None,
            routing_metrics=getattr(model, 'last_update_metrics', None),
            scope='Synthetic HF Trainer updates, forward/backward/clip/AdamW/scheduler/routing. '
                  'Excludes preprocessing, evaluation and checkpoint saving; not a convergence result.',
            source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                           ['bdot/model.py', 'bdot/grouped.py', 'bdot/trainer.py', 'packed_fa4.py',
                            'packed_training.py', 'train.py', 'scripts/benchmark_bdot.py']},
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({k: result[k] for k in ('status', 'seconds_per_update', 'tokens_per_second',
                                                'peak_allocated_bytes', 'world_size')}, indent=2))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
