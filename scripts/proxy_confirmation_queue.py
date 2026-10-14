"""Three fresh 12-layer runs: matched seed replication and combined P7 routing."""
import argparse
import json
from pathlib import Path

from scripts.proxy_overnight_queue import make as make_training_jobs, read
from run_experiments import load_jobs

CASES = (
    ('a-seed1042', 'A', .1, 1042),
    ('targetgrad-seed1042', 'P7-simple-targetgrad', .1, 1042),
    ('targetgrad-lm-seed42', 'P7-simple-targetgrad-lm', .1, 42),
)
EVAL_PYTHON = '/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11'


def checkpoint_path(root, case):
    label, arm, _, seed = case
    return Path(root) / label / f'seed-{seed}' / arm / 'checkpoint-2500'


def make(recipe_path, output, rows, predecessor, *, cases=CASES, module=__name__):
    # Reuse the established numerical gate, per-case 8-GPU smoke/validator,
    # train.py launch and final validator. Every stage uses a fresh directory.
    output=Path(output)
    summary=dict(name='summarize',argv=['{python}','-m',module,
        'summarize','--root','{run_dir}','--reference',str(predecessor)],
        required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})])
    items=make_training_jobs(recipe_path,output.with_name('training-jobs.json'),rows,
        cases=cases,summary_job=summary)
    for stage in ('numerics','smoke','full'):
        items.append(dict(name='eval-'+stage,gpus=list(range(8)),timeout_seconds=43200,
            argv=[EVAL_PYTHON,'-u','-m',module,'evaluate',
                  '--root','{run_dir}','--reference',str(predecessor),'--stage',stage],
            required_outputs=[dict(path=f'downstream/{stage}/summary.json',json_equals={'status':'passed'})]))
    with output.open('x') as f:json.dump(dict(jobs=items),f,indent=2)
    load_jobs(output)
    return items


def validate_training(root, cases=CASES):
    rows=[]; fingerprints={}
    for case in cases:
        label,arm,weight,seed=case;ck=checkpoint_path(root,case);folder=ck.parent
        r=read(folder/'result.json');c=read(folder/'train_config.json')
        if (read(Path(root)/label/'validated.json')['status']!='passed'
                or r['global_step']!=2500 or r['status']!='complete' or r['arm']!=arm
                or read(ck/'trainer_state.json')['global_step']!=2500
                or c['training']['seed']!=seed or c['training']['data_seed']!=seed
                or c['pilot']['arm']!=arm or c['pilot']['proxy_lambda_max']!=weight
                or c['pilot']['attention_backend']!='fa4' or not c['training']['bf16']
                or c['model_config']['num_hidden_layers']!=12
                or (arm!='A' and c['pilot']['proxy_module_seed']!=seed+1)):
            raise ValueError('Invalid confirmation checkpoint: '+label)
        pair=(c['train_fingerprint'],c['eval_fingerprint'])
        fingerprints.setdefault(seed,pair)
        if pair!=fingerprints[seed]:raise ValueError('Same-seed data fingerprints differ')
        rows.append(dict(label=label,arm=arm,seed=seed,lambda_max=weight,
            checkpoint=str(ck),evaluation=r['evaluation'],training_cost=r['training_cost'],
            fingerprints=pair,tokenizer=c['model']['tokenizer_name']))
    return rows


def summarize(root, reference, *, cases=CASES):
    from scripts.check_downstream_eval import save
    rows=validate_training(root,cases)
    # The reference is the completed remaining-arm evaluation scope. Its
    # source screen contains the original seed42 A/targetgrad checkpoints.
    scope=read(Path(reference)/'scope.json')
    original=Path(scope['source_screen'])/'supervised/run/seed-42'
    baseline=read(original/'A/result.json');config=read(original/'A/train_config.json')
    if baseline['arm']!='A' or baseline['global_step']!=2500 or baseline['status']!='complete':
        raise ValueError('Invalid original baseline')
    references={42:dict(evaluation=baseline['evaluation'],
        fingerprints=(config['train_fingerprint'],config['eval_fingerprint']),
        tokenizer=config['model']['tokenizer_name'])}
    references.update({row['seed']:row for row in rows if row['arm']=='A'})
    if config['training']['seed']!=42 or config['training']['data_seed']!=42:
        raise ValueError('Original baseline seed differs')
    for row in rows:
        a=references[row['seed']]
        if (tuple(row['fingerprints'])!=tuple(a['fingerprints']) or row['tokenizer']!=a['tokenizer']
                or row['evaluation']['eval_target_tokens']!=a['evaluation']['eval_target_tokens']):
            raise ValueError('Baseline data/tokenizer mismatch')
        row['lm_loss_delta_vs_same_seed_A']=row['evaluation']['eval_lm_loss']-a['evaluation']['eval_lm_loss']
    save(Path(root)/'summary.json',dict(status='passed',results=rows,reference=str(reference)))


def evaluation_reference(reference):
    from scripts.proxy_screen_evaluation import PROTOCOLS, validate_scope
    from scripts.train_then_burn import digest
    import importlib.metadata as md
    previous=read(Path(reference)/'scope.json');validate_scope(previous)
    completed=read(Path(reference)/'supervised/run/full/summary.json')
    if (completed['status']!='passed' or completed['diagnostic'] is not False
            or set(completed['checkpoints'])!=set(previous['checkpoints']) or len(completed['checkpoints'])!=27
            or previous['protocols']!={k:[n,t] for k,(n,t) in PROTOCOLS.items()}
            or digest(previous['manifest'])!=previous['manifest_sha256']):
        raise ValueError('Reference evaluation did not validate')
    for package,version in previous['versions'].items():
        if md.version(package)!=version:raise ValueError('Evaluation environment changed: '+package)
    return previous


def preflight(root, reference):
    from eval.benchmarks import local_dataset_paths, resolve_tasks
    from scripts.proxy_screen_evaluation import PROTOCOLS
    from scripts.check_downstream_eval import save
    previous=evaluation_reference(reference)
    mapping=local_dataset_paths(previous['data_root'],previous['manifest'])
    if set(mapping)!=set(resolve_tasks([t for _,ts in PROTOCOLS.values() for t in ts],True)):
        raise ValueError('Benchmark coverage differs')
    original=Path(previous['source_screen'])/'supervised/run/seed-42/A'
    result=read(original/'result.json');config=read(original/'train_config.json')
    if (result['arm']!='A' or result['global_step']!=2500 or result['status']!='complete'
            or config['training']['seed']!=42 or config['training']['data_seed']!=42):
        raise ValueError('Original baseline is not the completed seed42 screen')
    save(Path(root)/'evaluation-preflight.json',dict(status='passed',versions=previous['versions'],
        tasks=len(mapping),reference=str(reference),manifest_sha256=previous['manifest_sha256']))


def evaluate(root, reference, stage, *, cases=CASES, scope_kind="confirmation3"):
    from scripts.proxy_screen_evaluation import run, validate_scope
    from scripts.check_downstream_eval import save
    from scripts.train_then_burn import digest
    import subprocess
    import sys
    root=Path(root);scope_path=root/'downstream-scope.json'
    if stage=='numerics':
        previous=evaluation_reference(reference)
        from accelerate.commands.config.config_args import default_yaml_config_file
        if Path(default_yaml_config_file).read_bytes()!=Path('resources/accelerate_config.yaml').read_bytes():
            raise ValueError('Evaluation Accelerate cache differs from resource')
        subprocess.run([str(Path(sys.executable).parent/'accelerate'),'env'],check=True)
        if read(root/'summary.json')['status']!='passed':raise ValueError('Training summary did not pass')
        rows=validate_training(root,cases);specs={}
        for row in rows:
            ck=row['checkpoint']
            specs[ck]={k:row[k] for k in ('arm','label','seed','lambda_max')}
            specs[ck].update(step=2500,checkpoint_sha256=digest(Path(ck)/'model.safetensors'))
        scope=dict(kind=scope_kind,training_root=str(root),checkpoints=list(specs),specs=specs,
            protocols=previous['protocols'],data_root=previous['data_root'],manifest=previous['manifest'],
            manifest_sha256=previous['manifest_sha256'],versions=previous['versions'],
            evaluation_seed=42,reference=str(reference),finetuning=False,paired_loss=False)
        validate_scope(scope);save(scope_path,scope)
    run(stage,root/'downstream'/stage,scope_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('summarize','evaluate','preflight'))
    p.add_argument('--root',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--stage',choices=('numerics','smoke','full'))
    a=p.parse_args()
    if a.mode=='summarize':summarize(a.root,a.reference)
    elif a.mode=='preflight':preflight(a.root,a.reference)
    elif a.stage:evaluate(a.root,a.reference,a.stage)
    else:p.error('Evaluation requires a stage')
