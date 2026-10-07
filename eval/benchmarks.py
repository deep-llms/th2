"""Core benchmark evaluation using lm-evaluation-harness."""

TASK_CONFIGS = {
    "xnli": [
        "xnli_en", "xnli_vi", "xnli_zh", "xnli_ru", "xnli_de", "xnli_ar",
    ],
    "belebele": [
        "belebele_eng_Latn", "belebele_vie_Latn", "belebele_zho_Hans",
        "belebele_rus_Cyrl", "belebele_deu_Latn", "belebele_arb_Arab",
    ],
    "xcopa": [
        "xcopa_vi", "xcopa_zh",
    ],
    "xstorycloze": [
        "xstorycloze_en", "xstorycloze_ar", "xstorycloze_ru", "xstorycloze_zh",
    ],
    "paws-x": [
        "paws_en", "paws_de", "paws_zh",
    ],
    "hellaswag": [
        "hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi",
    ],
}

ENGLISH_TASKS = {
    'xnli': ['xnli_en'],
    'belebele': ['belebele_eng_Latn'],
    'xcopa': [],  # XCOPA has no English subset; do not silently substitute COPA.
    'xstorycloze': ['xstorycloze_en'],
    'paws-x': ['paws_en'],
    'hellaswag': ['hellaswag'],
    'piqa': ['piqa'],
    'arc_easy': ['arc_easy'],
    'arc_challenge': ['arc_challenge'],
    'winogrande': ['winogrande'],
}


def resolve_tasks(task_groups=None, english_only=False):
    default = task_groups is None
    groups = list(TASK_CONFIGS) if default else task_groups
    known = {task for group in TASK_CONFIGS.values() for task in group}
    english = {task for group in ENGLISH_TASKS.values() for task in group}
    selected = []
    for group in groups:
        if english_only and group == 'xcopa':
            if not default:
                raise ValueError('XCOPA has no English subset; select other tasks')
            print('Skipping XCOPA: no English subset')
            continue
        if group in TASK_CONFIGS or group in ENGLISH_TASKS:
            tasks = (ENGLISH_TASKS[group] if english_only else
                     TASK_CONFIGS.get(group, ENGLISH_TASKS.get(group)))
        elif group in known | english and (not english_only or group in english):
            tasks = [group]
        else:
            raise ValueError(f'Unknown task or non-English task in English-only mode: {group}')
        selected.extend(task for task in tasks if task not in selected)
    if not selected:
        raise ValueError('No evaluation tasks selected')
    return selected


def local_dataset_paths(root, manifest):
    """Verify pinned raw snapshots; task templates still control splits/prompts."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(root).resolve()
    spec = json.loads(Path(manifest).read_text())
    if spec.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark dataset manifest')
    paths = {}
    for repo in spec['repositories']:
        directory = (root / repo['path']).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError('Missing or invalid local dataset directory')
        for entry in repo['files']:
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError(f'Missing or invalid dataset file: {entry["path"]}')
            if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {entry["path"]}')
        for name in repo['tasks']:
            if name in paths:
                raise ValueError(f'Duplicate dataset mapping: {name}')
            config = repo['load_configs'][name]
            if config['loader'] not in ('parquet', 'json', 'csv'):
                raise ValueError('Unsupported local dataset loader')
            verified = {entry['path'] for entry in repo['files']}
            if any(file not in verified for files in config['data_files'].values() for file in files):
                raise ValueError('Dataset configuration references an unverified file')
            paths[name] = dict(dataset_path=config['loader'], dataset_name=None,
                dataset_kwargs={**config.get('kwargs', {}), 'data_files': {
                    split: [str(root / file) for file in files]
                    for split, files in config['data_files'].items()}})
    return paths


def task_configs(names, dataset_paths=None):
    # Override renamed dataset repositories in memory, never edit site-packages.
    overrides = {'xnli_': 'facebook/xnli', 'xcopa_': 'cambridgeltl/xcopa',
                 'paws_': 'google-research-datasets/paws-x'}
    if dataset_paths is not None:
        missing = set(names) - dataset_paths.keys()
        if missing:
            raise ValueError(f'Missing local datasets for tasks: {sorted(missing)}')
        return [dict(task=name, **dataset_paths[name]) for name in names]
    return [dict(task=name, **next(({'dataset_path': repo} for prefix, repo in overrides.items()
                                  if name.startswith(prefix)), {})) for name in names]


def eval_benchmarks(model, tokenizer, task_groups=None, num_fewshot=0, batch_size=16,
                    device='cuda', english_only=False, seed=42, limit=None, dataset_paths=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    task_list = resolve_tasks(task_groups, english_only)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        backend='causal',
        max_length=min(2048, model.config.max_position_embeddings),
        softmax_dtype='float32',
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_configs(task_list, dataset_paths),
        num_fewshot=num_fewshot,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
        apply_chat_template=False, log_samples=True, limit=limit,
    )

    return results


def print_benchmark_results(results):
    from lm_eval.utils import make_table
    print(make_table(results))
