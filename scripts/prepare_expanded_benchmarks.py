"""Freeze benchmark datasets into portable parquet snapshots with checksums.

Run on a networked dev machine. This never submits runner commands. A source
lock records HF commit IDs before downloads; reruns reuse that lock. Dataset
scripts are not executed. The resulting manifest uses the existing offline
loader on both dev and GPU machines.
"""
import argparse
import hashlib
import json
from pathlib import Path

from eval.benchmarks import resolve_tasks, task_configs


def specifications(tasks, supervised=(), profile='standard'):
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config
    if profile not in ('standard','paired'): raise ValueError('Unknown benchmark profile')
    if profile=='paired':
        from eval.paired.protocol import configs
        result={}
        for task in tasks:
            for config in configs(task):
                splits=list(dict.fromkeys(config[k] for k in ('training_split','validation_split','test_split','fewshot_split') if config.get(k)))
                result[config['task']]=(config['dataset_path'],config.get('dataset_name'),splits)
        if supervised: raise ValueError('Use the standard profile for supervised data')
        return result
    manager = TaskManager()
    result = {}
    for name in resolve_tasks(tasks, english_only=True) if tasks else []:
        if name in ('xcsqa_en', 'xcodah_en'):
            result[name] = ('INK-USC/xcsr', 'X-CSQA-en' if name == 'xcsqa_en' else 'X-CODAH-en', ['validation'])
        else:
            config = load_yaml_config(manager.task_index[name]['yaml_path'])
            config.update(task_configs([name])[0])
            repo = {'blimp':'nyu-mll/blimp'}.get(config['dataset_path'], config['dataset_path'])
            splits = list(dict.fromkeys(config[k] for k in ('training_split', 'validation_split', 'test_split', 'fewshot_split') if config.get(k)))
            result[name] = (repo, config.get('dataset_name'), splits)
    for task in supervised:
        if task == 'squad_v2': result[task] = ('rajpurkar/squad_v2', None, ['train', 'validation'])
        elif task.startswith('glue_'):
            name = task.removeprefix('glue_')
            if name.startswith('mnli_'): name = 'mnli'
            result[task] = ('nyu-mll/glue', name, ['train', 'validation_matched', 'validation_mismatched'] if name == 'mnli' else ['train', 'validation'])
        else: raise ValueError('Unknown supervised task: '+task)
    return result


def normalize_xcsr(row):
    if row['lang'] != 'en': raise ValueError('Expected English XCSR')
    choices = row['question']['choices']
    if not row['answerKey'] or row['answerKey'] not in choices['label']:
        raise ValueError('XCSR test labels are hidden; use labeled validation only')
    return dict(id=row['id'], question=row['question']['stem'].strip(), choices=choices['text'],
                label=choices['label'].index(row['answerKey']))


def freeze(output, lock_path, tasks, supervised=(), profile='standard'):
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi
    from importlib.metadata import version
    if version('lm_eval') != '0.4.10': raise ValueError('Require lm_eval==0.4.10')
    specs = specifications(tasks, supervised, profile)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock['specifications'] != {k:list(v) for k,v in specs.items()}:
            raise ValueError('Source lock task definitions differ')
    else:
        repos = sorted({v[0] for v in specs.values()})
        lock = dict(specifications=specs, revisions={repo:HfApi().dataset_info(repo).sha for repo in repos},
                    lm_eval=version('lm_eval'))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('x') as f: json.dump(lock, f, indent=2)
    repositories = []
    for name, (repo, config, splits) in specs.items():
        revision = lock['revisions'][repo]
        files, data_files = [], {}
        for split in splits:
            data = load_dataset(repo, config, split=split, revision=revision)
            if name in ('xcsqa_en', 'xcodah_en'):
                data = Dataset.from_list([normalize_xcsr(row) for row in data])
            path = root/name/(split+'.parquet'); path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize into a temporary file; never overwrite an existing different snapshot.
            temporary = path.with_suffix('.pending.parquet')
            data.to_parquet(temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                temporary.unlink(); raise ValueError('Existing snapshot differs: '+str(path))
            temporary.replace(path)
            relative = str(path.relative_to(root))
            files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest, rows=len(data)))
            data_files[split] = [relative]
        repositories.append(dict(repo_id=repo, revision=revision, dataset_name=config, path=name,
            tasks=[name], files=files, load_configs={name:dict(loader='parquet', data_files=data_files)}))
    manifest = dict(schema_version=1, repositories=repositories, source_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                    lm_eval=version('lm_eval'))
    path = root/'manifest.json'
    content = json.dumps(manifest, indent=2)+'\n'
    if path.exists() and path.read_text() != content: raise ValueError('Existing manifest differs')
    path.write_text(content)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--source-lock', required=True)
    p.add_argument('--tasks', nargs='*', default=[])
    p.add_argument('--supervised', nargs='*', default=[])
    p.add_argument('--profile', choices=['standard','paired'], default='standard')
    args = p.parse_args()
    if not args.tasks and not args.supervised: p.error('Select benchmark or supervised tasks')
    print(freeze(args.output, args.source_lock, args.tasks, args.supervised, args.profile))


if __name__ == '__main__': main()
