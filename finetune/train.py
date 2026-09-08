"""Fine-tune a fresh checkpoint copy on one task/language, then evaluate."""
import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed
from eval.runtime import checkpoint_identity, languages, load_checkpoint
from eval.benchmarks import task_plan, load_tasks, evaluate
from finetune.tasks import TASK_CONFIGS, GenerativeDataset
from capacity_allocation.data import write_json


def fit(model, dataset, config, *, device, precision, seed, num_workers=0):
    set_seed(seed)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    steps = len(loader)*config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*steps), steps)
    model.train()
    started = time.monotonic()
    history, completed = [], 0
    for epoch in range(config['epochs']):
        total_nll, targets = 0., 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = batch['labels'][:, 1:].ne(-100).sum().item()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                loss = model(**batch, use_cache=False).loss
            if not count or not torch.isfinite(loss).item():
                raise RuntimeError('Empty/nonfinite fine-tuning loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total_nll += loss.item()*count
            targets += count
            completed += 1
        history.append(dict(epoch=epoch+1, nll=total_nll/targets, scored_targets=targets))
        print(f'FINETUNE_EPOCH {history[-1]} steps={completed}/{steps}', flush=True)
    if completed != steps or not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError('Incomplete fine-tuning or nonfinite parameters')
    return dict(steps=completed, epochs=history, seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer-name')
    parser.add_argument('--task', choices=TASK_CONFIGS, required=True)
    parser.add_argument('--train-language', default='en')
    parser.add_argument('--languages', default='en', help='Evaluation languages, independent of training language')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--benchmark-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    selected = languages(args.languages)
    train_languages = languages(args.train_language)
    if len(train_languages) != 1 or args.num_workers < 0:
        parser.error('One training language per independent fine-tuning job; nonnegative workers')
    plan, unavailable = task_plan(selected, [args.task])
    train_plan, _ = task_plan(train_languages, [args.task])
    # Coverage must exist for every requested eval language for this fine-tuning task.
    tasks = load_tasks(list({item['task']: item for item in plan+train_plan}.values()), args.dataset_root)
    training_task = tasks[train_plan[0]['task']]
    root.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model, tokenizer = load_checkpoint(args.checkpoint, args.tokenizer_name, args.device, args.precision)
    config = TASK_CONFIGS[args.task].copy()
    dataset = GenerativeDataset(training_task, tokenizer, config['max_length'])
    specification = dict(checkpoint=checkpoint_identity(args.checkpoint), task=args.task,
        train_language=args.train_language, languages=selected, seed=args.seed, config=config,
        precision=args.precision, master_weights='fp32', weight_decay=0.01,
        warmup_ratio=0.1, schedule='linear', clip_norm=1.,
        source_examples=dataset.source_count, used_examples=len(dataset), skipped_examples=dataset.skipped,
        train_data_fingerprint=dataset.data_fingerprint, lm_eval_version='0.4.10')
    write_json(root/'run_config.json', specification)
    training = fit(model, dataset, config, device=args.device, precision=args.precision,
                   seed=args.seed, num_workers=args.num_workers)
    # Retain the trained model even if subsequent benchmark evaluation fails.
    model.save_pretrained(root/'model', safe_serialization=True)
    tokenizer.save_pretrained(root/'model')
    write_json(root/'training_complete.json', dict(success=True, **specification, training=training))
    results = evaluate(model, tokenizer, {item['task']: tasks[item['task']] for item in plan},
        device=args.device, precision=args.precision, batch_size=args.benchmark_batch_size, seed=args.seed)
    write_json(root/'result.json', dict(success=True, **specification, training=training,
        benchmarks=results, unavailable_benchmarks=unavailable))
    print(f'FINETUNE_AND_EVAL_COMPLETE {root}', flush=True)


if __name__ == '__main__':
    main()
