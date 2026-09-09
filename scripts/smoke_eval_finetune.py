"""Bounded eval/finetune smoke on a local checkpoint; NOT research results.

Uses small real benchmark subsets and fresh checkpoint reloads for each task.
All writes are confined to a fresh smoke directory. No downloads/GPU management.
"""
import argparse
import gc
from pathlib import Path

from eval.runtime import offline
offline()
import torch
from datasets import Dataset
from capacity_allocation.data import write_json
from eval.runtime import checkpoint_identity, load_checkpoint
from eval.benchmarks import evaluate, load_tasks, task_plan
from eval.ppl import evaluate as evaluate_ppl
from finetune.tasks import GenerativeDataset, TASK_CONFIGS
from finetune.train import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASK_CONFIGS,
                        default=list(TASK_CONFIGS))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if not args.tasks or len(set(args.tasks)) != len(args.tasks):
        parser.error('Select unique tasks')
    source = Path(args.checkpoint).resolve(strict=True)
    root = Path(args.output_dir).resolve()
    if root == source or source in root.parents:
        parser.error('Smoke outputs must not be inside the input checkpoint')
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    before = checkpoint_identity(source)
    plan, _ = task_plan('en', args.tasks)
    tasks = load_tasks(plan, args.dataset_root)
    reports = {}
    ppl = None
    for item in plan:
        name, task = item['task'], tasks[item['task']]
        config = TASK_CONFIGS[item['group']].copy()
        train_split = task.config.training_split
        eval_split = task.config.test_split or task.config.validation_split
        if not train_split or train_split == eval_split or not task.has_training_docs():
            raise ValueError('Need distinct real training and evaluation splits')
        full_train, full_eval = len(task.dataset[train_split]), len(task.eval_docs)
        task.dataset[train_split] = task.dataset[train_split].select(range(min(128, full_train)))
        task.dataset[eval_split] = task.dataset[eval_split].select(range(min(2, full_eval)))
        task.task_docs = task.eval_docs
        model, tokenizer = load_checkpoint(source, device=args.device, precision='bf16')
        if item == plan[0]:
            # Synthetic long context tests the actual PPL entry point/shape only.
            # It is never a corpus-quality score or the training validation set.
            Dataset.from_dict({'text': ['This is a local smoke test. ' * 1024]}).save_to_disk(
                str(root/'ppl_input/en'))
            ppl = evaluate_ppl(model, tokenizer, root/'ppl_input', 'en',
                device=args.device, precision='bf16', block_size=2048,
                workers=1, map_batch_size=1000, cache_dir=root/'ppl_cache')
            assert ppl['token_weighted']['scored_targets'] >= 2047
        observed = []
        hook = model.model.layers[0].self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: observed.append(str(output.dtype)))
        benchmark = evaluate(model, tokenizer, {name: task}, device=args.device,
                             precision='bf16', batch_size=2)
        benchmark_dtypes = sorted(set(observed))
        observed.clear()
        dataset = GenerativeDataset(task, tokenizer, config['max_length'])
        # Two real optimizer updates at the production task's batch/sequence size.
        count = 2 * config['batch_size']
        if len(dataset) < count:
            raise ValueError(f'Not enough scored training rows for {name}')
        parameter = next(model.parameters())
        original = parameter.detach().clone()
        config['epochs'] = 1
        training = fit(model, dataset.items[:count], config, device=args.device,
                       precision='bf16', seed=42)
        training_dtypes = sorted(set(observed))
        hook.remove()
        changed = not torch.equal(original, parameter.detach())
        assert training['steps'] == 2 and changed
        assert training_dtypes == ['torch.bfloat16']
        assert all(p.dtype == torch.float32 for p in model.parameters())
        del original
        # Exercise save/reload on the first task only, retaining one model per arm.
        reload_ok = None
        if item == plan[0]:
            model.eval()
            probe = dataset[0]['input_ids'][:16].unsqueeze(0).to(args.device)
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                expected = model(probe, use_cache=False).logits.detach().cpu()
            model.save_pretrained(root/'model', safe_serialization=True)
            tokenizer.save_pretrained(root/'model')
            del model, parameter
            gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
            model, _ = load_checkpoint(root/'model', device=args.device, precision='bf16')
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                actual = model(probe, use_cache=False).logits.detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reload_ok = True
        observed.clear()
        hook = model.model.layers[0].self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: observed.append(str(output.dtype)))
        try:
            after_benchmark = evaluate(model, tokenizer, {name: task}, device=args.device,
                                       precision='bf16', batch_size=2)
        finally:
            hook.remove()
        after_benchmark_dtypes = sorted(set(observed))
        reports[name] = dict(full_train=full_train, full_eval=full_eval,
            smoke_train_examples=count, smoke_eval_examples=2,
            benchmark=benchmark, after_benchmark=after_benchmark,
            benchmark_dtypes=benchmark_dtypes,
            after_benchmark_dtypes=after_benchmark_dtypes,
            training=training, training_dtypes=training_dtypes,
            weights_changed=changed, reload_ok=reload_ok)
        write_json(root/f'{name}.json', dict(smoke_only=True, **reports[name]))
        print('SMOKE_TASK_FINISHED', name, 'benchmark_dtypes', benchmark_dtypes, flush=True)
        del model, dataset
        if 'parameter' in locals(): del parameter
        gc.collect()
        if args.device == 'cuda': torch.cuda.empty_cache()
    assert checkpoint_identity(source) == before, 'Source checkpoint changed'
    success = all(r['benchmark_dtypes'] == ['torch.bfloat16'] and
                  r['after_benchmark_dtypes'] == ['torch.bfloat16'] for r in reports.values())
    write_json(root/'report.json', dict(success=success, smoke_only=True,
        checkpoint=before, tasks=reports, synthetic_long_context_ppl=ppl,
        source_checkpoint_unchanged=True))
    if not success:
        raise SystemExit('SMOKE FAILED: benchmark BF16 precision mismatch; see report.json')
    print('EVAL_FINETUNE_SMOKE_PASS_NOT_RESEARCH_RESULTS', flush=True)


if __name__ == '__main__':
    main()
