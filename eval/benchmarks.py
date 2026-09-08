"""Use official lm-eval 0.4.10 task definitions, with local dataset paths.

No shared package YAML mutations and no remote fallback. Only selected
task/language snapshots are required. Missing language coverage is recorded.
"""
from copy import deepcopy
from importlib.metadata import version
import math
from pathlib import Path

from eval.runtime import languages, offline

TASKS = {
    'xnli': {lang: f'xnli_{lang}' for lang in ('en', 'ar', 'de', 'ru', 'vi', 'zh')},
    'belebele': dict(zip(('en', 'ar', 'de', 'ru', 'vi', 'zh'),
        ('belebele_eng_Latn', 'belebele_arb_Arab', 'belebele_deu_Latn',
         'belebele_rus_Cyrl', 'belebele_vie_Latn', 'belebele_zho_Hans'))),
    'xcopa': {'vi': 'xcopa_vi', 'zh': 'xcopa_zh'},
    'xstorycloze': {lang: f'xstorycloze_{lang}' for lang in ('en', 'ar', 'ru', 'zh')},
    'paws-x': {lang: f'paws_{lang}' for lang in ('en', 'de', 'zh')},
    'hellaswag': {'en': 'hellaswag', **{lang: f'hellaswag_{lang}' for lang in ('ar', 'de', 'ru', 'vi')}},
    'arc_easy': {'en': 'arc_easy', **{lang: f'arc_{lang}' for lang in ('ar', 'de', 'ru', 'vi', 'zh')}},
}
# Preserve the old five English zero-shot tasks; ARC-Easy is additionally
# evaluated so every fine-tuned task also has a matching zero-shot result.
DEFAULT_GROUPS = ('xnli', 'belebele', 'xstorycloze', 'paws-x', 'hellaswag', 'arc_easy')
REPOSITORIES = {
    'xnli': 'facebook/xnli', 'belebele': 'facebook/belebele',
    'xcopa': 'cambridgeltl/xcopa', 'xstorycloze': 'juletxara/xstory_cloze',
    'paws-x': 'google-research-datasets/paws-x',
    'hellaswag': 'Rowan/hellaswag', 'arc_easy': 'allenai/ai2_arc',
}


def task_plan(selected_languages='en', groups=DEFAULT_GROUPS):
    selected_languages = languages(selected_languages)
    groups = list(groups)
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(TASKS):
        raise ValueError('Select unique known benchmark groups')
    plan, unavailable = [], []
    for group in groups:
        for lang in selected_languages:
            if lang not in TASKS[group]:
                unavailable.append(dict(group=group, language=lang))
                continue
            repo = REPOSITORIES[group]
            if lang != 'en' and group in ('hellaswag', 'arc_easy'):
                repo = 'alexandrainst/' + ('m_hellaswag' if group == 'hellaswag' else 'm_arc')
            plan.append(dict(task=TASKS[group][lang], group=group, language=lang, repository=repo))
    if not plan or set(selected_languages) - {item['language'] for item in plan}:
        raise ValueError('No benchmark coverage for one or more selected languages')
    return plan, unavailable


def local_task_config(config, item, dataset_root):
    """Validate our known task mapping before giving HF an absolute local path."""
    config = deepcopy(config)
    if config.get('task') != item['task'] or config.get('output_type') != 'multiple_choice':
        raise ValueError('Unexpected lm-eval task definition')
    aliases = {'xnli': 'facebook/xnli', 'xcopa': 'cambridgeltl/xcopa',
               'paws-x': 'google-research-datasets/paws-x'}
    repository = aliases.get(config.get('dataset_path'), config.get('dataset_path'))
    if repository != item['repository']:
        raise ValueError(f'Unexpected repository for {item["task"]}: {repository}')
    path = Path(dataset_root).resolve(strict=True)/item['repository']
    if not path.is_dir():
        raise FileNotFoundError(f'Missing selected benchmark snapshot: {path}')
    # These selected official configs do not require URL data_files or custom
    # builders. Reject changes instead of forwarding potential network inputs.
    kwargs = config.get('dataset_kwargs') or {}
    if set(kwargs) - {'trust_remote_code'} or config.get('custom_dataset'):
        raise ValueError('Review unexpected dataset loading options before use')
    config['dataset_path'] = str(path)
    config['dataset_kwargs'] = {}
    return config


def load_tasks(plan, dataset_root):
    offline()
    if version('lm_eval') != '0.4.10':
        raise RuntimeError('Use reviewed lm_eval==0.4.10; review task/API changes before upgrading')
    from lm_eval.tasks import TaskManager
    from lm_eval.api.task import ConfigurableTask
    manager = TaskManager()
    configs = [local_task_config(manager._get_config(item['task']), item, dataset_root) for item in plan]
    return {config['task']: ConfigurableTask(config=config) for config in configs}


def evaluate(model, tokenizer, tasks, *, device, precision, batch_size=8, seed=42):
    offline()
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    if not tasks or batch_size <= 0:
        raise ValueError('Nonempty task set and positive batch required')
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device,
              batch_size=batch_size, max_length=2048, add_bos_token=False,
              prefix_token_id=tokenizer.eos_token_id)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        raw = lm_eval.simple_evaluate(model=lm, tasks=list(tasks.values()), num_fewshot=0,
            log_samples=False, bootstrap_iters=0, random_seed=seed, numpy_random_seed=seed,
            torch_random_seed=seed, fewshot_random_seed=seed)
    if set(raw['results']) != set(tasks):
        raise RuntimeError('Benchmark result coverage mismatch')
    results = {}
    for name, metrics in raw['results'].items():
        values = {key: float(metrics[key]) for key in ('acc,none', 'acc_norm,none') if key in metrics}
        if 'acc,none' not in values or not all(math.isfinite(x) and 0 <= x <= 1 for x in values.values()):
            raise RuntimeError(f'Missing/nonfinite benchmark accuracy: {name}')
        counts = raw['n-samples'][name]
        expected = len(tasks[name].eval_docs)
        if counts['effective'] != expected or expected <= 0:
            raise RuntimeError(f'Incomplete benchmark: {name}')
        results[name] = dict(metrics=values, samples=counts,
            split=tasks[name].config.test_split or tasks[name].config.validation_split,
            data_fingerprint=tasks[name].eval_docs._fingerprint)
    return results
