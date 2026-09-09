"""Offline CPU smoke of real English benchmark snapshots; NOT research evaluation.

Loads complete datasets to check coverage, then scores only a few rows per task
using tiny random models. Never downloads, touches GPUs, or reads training runs.
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from capacity_allocation.data import write_json
from capacity_allocation.modeling import ARMS, build_model, experiment_config
from eval.benchmarks import ENGLISH_CORE_GROUPS, evaluate, load_tasks, summarize_benchmarks, task_plan
from eval.blimp_tasks import BLIMP_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--examples-per-task', type=int, default=2)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['B0', 'C'])
    args = parser.parse_args()
    if args.examples_per_task < 1 or len(set(args.arms)) != len(args.arms):
        parser.error('Positive examples-per-task and unique arms required')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    plan, missing = task_plan('en', ENGLISH_CORE_GROUPS)
    assert not missing and len(plan) == 74
    tasks = load_tasks(plan, args.dataset_root)
    expected_counts = {name: 1000 for name in BLIMP_TASKS}
    expected_counts.update(lambada_openai=5153, hellaswag=10042, piqa=1838,
                           winogrande=1267, arc_easy=2376, arc_challenge=1172, boolq=3270)
    coverage = {}
    for name, task in tasks.items():
        count = len(task.eval_docs)
        if count != expected_counts[name]:
            raise ValueError(f'Unexpected full dataset coverage for {name}: {count}')
        split = task.config.test_split or task.config.validation_split
        coverage[name] = dict(full_examples=count, split=split, fingerprint=task.eval_docs._fingerprint)
        task.dataset[split] = task.dataset[split].select(range(min(args.examples_per_task, count)))
        task.task_docs = task.eval_docs
    write_json(root/'dataset_coverage.json', dict(success=True, tasks=coverage))
    print('ALL_74_TASKS_FULL_DATASET_COVERAGE_VERIFIED', flush=True)

    # Byte-level tokenizer covers arbitrary real text without requiring a
    # downloaded model/tokenizer. Only a tiny random model is being tested.
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(['This is a tiny tokenizer for smoke testing only.'],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=['<unk>', '<eos>']))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                        eos_token='<eos>', pad_token='<eos>')
    tokenizer.model_max_length = 10**30
    reports = {}
    for arm in args.arms:
        config = experiment_config(arm, tiny=True)
        config.vocab_size = len(tokenizer)
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        model = build_model(config).eval()
        result = evaluate(model, tokenizer, tasks, device='cpu', precision='fp32', batch_size=4)
        summary = summarize_benchmarks(result)
        reports[arm] = dict(smoke_only=True, random_tiny_model=True, benchmarks=result, benchmark_summaries=summary)
        write_json(root/f'{arm}_smoke.json', reports[arm])
        print('REAL_DATA_TINY_MODEL_SMOKE_PASSED', arm, len(result), flush=True)
    write_json(root/'complete.json', dict(success=True, smoke_only=True, arms=args.arms,
        tasks=len(tasks), examples_per_task=args.examples_per_task, dataset_coverage=coverage))
    print('ENGLISH_BENCHMARK_SMOKE_COMPLETE_NOT_RESEARCH_RESULTS', flush=True)


if __name__ == '__main__':
    main()
