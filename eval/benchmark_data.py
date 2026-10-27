"""Verified local benchmark inputs; reused from deep2shallow eval/benchmarks.py."""
from functools import lru_cache


@lru_cache(maxsize=1)
def benchmark_task_manager():
    from pathlib import Path
    import lm_eval
    from lm_eval.tasks import TaskManager
    root=Path(lm_eval.__file__).parent/'tasks'
    # Index only this screen's canonical folders, not every installed benchmark.
    folders=('hellaswag','arc','piqa','winogrande','super_glue','openbookqa','sciq','lambada')
    return TaskManager(include_defaults=False,include_path=[str(root/name) for name in folders])
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


def task_configs(names, paths):
    """Load complete canonical templates before changing only data locations.

    In harness 0.4.13 an inline dict is a complete task, not a named-task
    override as in 0.4.10. Omitting the template would lose prompts/splits.
    """
    from importlib.metadata import version
    from lm_eval.tasks._yaml_loader import load_yaml
    if version('lm-eval') != '0.4.13':raise ValueError('Require lm-eval==0.4.13')
    manager = benchmark_task_manager()
    configs = []
    for name in names:
        config = load_yaml(manager.task_index[name].yaml_path, resolve_func=True)
        if config['output_type'] not in ('loglikelihood','multiple_choice'):
            raise ValueError('This screen supports likelihood tasks only')
        configs.append({**config, **paths[name]})
    return configs
