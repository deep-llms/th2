"""Frozen paired protocol, distinct from the original benchmark reporting rules."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from functools import lru_cache

VERSION = 'paired-v3.1'
RULES = {
    'arc_easy': 'char', 'arc_challenge': 'char', 'hellaswag': 'char', 'piqa': 'char',
    'winogrande': 'raw', 'xnli_en': 'raw', 'xstorycloze_en': 'raw',
    'lambada_openai': 'll', 'sciq': 'char', 'openbookqa': 'char',
    'social_iqa_norm': 'char', 'commonsense_qa_cf': 'char',
    'mmlu_continuation': 'char', 'blimp': 'raw',
}
Z = list(RULES)
F = {'arc_challenge': 25, 'arc_easy': 5, 'hellaswag': 10, 'piqa': 5,
     'paws_en': 5, 'winogrande': 5, 'xnli_en': 5, 'xstorycloze_en': 5}
PROBES = ['babi_cf_qa1', 'babi_cf_qa2', 'babi_cf_qa3']
RULES.update(paws_en='raw', **{name: 'raw' for name in PROBES})
CUSTOM = ['social_iqa_norm', 'commonsense_qa_cf', *PROBES]
PRECISION = ['hellaswag', 'arc_easy', 'piqa', 'lambada_openai', 'mmlu_continuation']


def serial(value):
    if callable(value):
        return value.__module__ + '.' + value.__qualname__
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inventory(task):
    from eval.benchmarks import resolve_tasks
    if task in CUSTOM:
        return [task]
    return resolve_tasks([task], english_only=True)


@lru_cache(maxsize=1)
def manager():
    from lm_eval.tasks import TaskManager
    return TaskManager(include_path=str(Path(__file__).parent / 'tasks'))


def configs(task, dataset_paths=None):
    """Keep canonical prompts and metrics, including acc_norm where provided."""
    from lm_eval.utils import load_yaml_config
    from eval.benchmarks import task_configs
    tm = manager()
    result = []
    for name in inventory(task):
        config = load_yaml_config(tm.task_index[name]['yaml_path'])
        if name not in CUSTOM:
            config.update(task_configs([name])[0])
        if dataset_paths is not None:
            if name not in dataset_paths:
                raise ValueError('Missing paired dataset: ' + name)
            config.update(dataset_paths[name])
        result.append(config)
    return result


def tasks(suite):
    if suite == 'Z': return {t: 0 for t in Z}
    if suite == 'F': return dict(F)
    if suite == 'probes': return {t: 2 for t in PROBES}
    if suite == 'paloma': return {t: 0 for t in inventory('paloma')}
    raise ValueError('Unknown paired suite: ' + suite)


def definition(position_documents=2000):
    if type(position_documents) is not int or not 2<=position_documents<=2000:
        raise ValueError('Position document count must be an integer between 2 and 2000')
    if version('lm_eval') != '0.4.10':
        raise ValueError('Paired evaluation requires lm_eval==0.4.10')
    directory=Path(__file__).parent
    project=directory.parent.parent
    sources=[*directory.glob('*.py'),*directory.glob('tasks/*'),*(project/'deep_kv').glob('*.py'),
             *(directory.parent/name for name in ('models.py','generation.py','benchmarks.py','eval_checkpoint.py'))]
    implementation={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources) if p.is_file()}
    return dict(version=VERSION, implementation=implementation, harness='0.4.10', seed=42, resamples=10000,
        analysis_seed=20261012, suites={s: tasks(s) for s in ('Z','F','probes','paloma')},
        rules=RULES, task_configs={t: serial(configs(t)) for t in [*RULES, *inventory('paloma')]},
        confirmatory=dict(tasks=Z, sequence=['M1','M2'], alpha=.05,
            null='joint invariance under independent pair swaps at the declared sampling unit',
            interpretation='checkpoint comparison; calibration supports assumptions, not a universal error bound'),
        corrections=dict(admission='descriptive only; full fixed suite is primary',
            effect_ratio='descriptive training-progress reference, not a PPL prediction',
            probe_comparison='joint-bootstrap difference of contrasts sharing A20',
            repeatability='maximum absolute M1/M2 difference <= 1% of paired reference SD',
            precision_sample='seeded without replacement; stratified by MMLU subject',
            precision_scope='five-task diagnostic; not a guarantee on every task',
            resample_degeneracy='infinite standardized resample statistics retained for ordering; nonfinite CI unavailable'),
        precision=dict(tasks=PRECISION, per_task=1000, per_mmlu_subject=18,
                       variance_tolerance=.10, bias_se_tolerance=.10),
        position=dict(documents=position_documents, length=2048, buckets=[[1,64],[64,128],[128,256],[256,512],[512,1024],[1024,2048]]))


def freeze(path, position_documents=2000):
    value = definition(position_documents)
    value['protocol_sha256'] = digest(value)
    write_json(path, value)
    return value


def read(path):
    value = json.loads(Path(path).read_text())
    body = {k:v for k,v in value.items() if k != 'protocol_sha256'}
    if value.get('protocol_sha256') != digest(body) or body != definition(body['position']['documents']):
        raise ValueError('Frozen protocol differs from this implementation')
    return value
