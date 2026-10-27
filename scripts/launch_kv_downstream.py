"""Freeze an evaluation-only queue; reuse the existing GPU/burn supervisor."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from eval.benchmark_data import local_dataset_paths
from efficient_llm.kv_long import verify_handoff
from scripts.gpu_process import inspect
from scripts.train_then_burn import BURN_HASH, GUARD_HASH, digest

OVERLAYS = ('eval/benchmark_data.py','eval/kv_harness.py','eval/downstream_suite.py',
            'scripts/check_kv_downstream_adapter.py','scripts/launch_kv_downstream.py',
            'resources/downstream_english_20261023.json')


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def verify_tokenizer(path, files):
    for name, expected in files.items():
        p=Path(path)/name
        if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:
            raise ValueError('Training tokenizer changed: '+name)


def jobs_for(checkpoints, data, manifest, tokenizer):
    jobs = [dict(name='adapter-check', gpus=[0], timeout_seconds=600,
        argv=['{python}','-m','scripts.check_kv_downstream_adapter','--checkpoints',str(checkpoints),
              '--tokenizer',str(tokenizer),
              '--output','{run_dir}/adapter-check.json'],
        required_outputs=[dict(path='adapter-check.json',json_equals=dict(status='ok'))])]
    for stage, limit in [('smoke',8),('full',None)]:
        jobs.append(dict(name=stage, gpus=list(range(8)), timeout_seconds=7200,
            argv=['{python}','-m','eval.downstream_suite','--checkpoints',str(checkpoints),
                  '--output','{run_dir}/'+stage,'--dataset-root',str(data),'--manifest',str(manifest)]
                  + ['--tokenizer',str(tokenizer)]
                  + (['--limit',str(limit)] if limit is not None else []),
            required_outputs=[dict(path=stage+'/summary.json',json_equals=dict(status='ok',diagnostic=limit is not None))]))
    return dict(jobs=jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--host',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args();source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!=a.host:raise ValueError('Wrong host')
    if a.output.exists():raise ValueError('Fresh output required')
    for name in [a.session,a.session+'_burn']:
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
            raise ValueError('Existing session '+name)
    for package, expected in [('lm-eval','0.4.13'),('transformers','5.9.0'),('datasets','4.8.5')]:
        if version(package)!=expected:raise ValueError('Unvalidated runtime: '+package)
    burn=a.parent/'source/resources/llm_pretrain_burn.py'
    verify_handoff(a.output.parent,burn,digest)
    if (digest(burn)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
            or Path('/mnt/local/_gpu_guard/DISABLED').exists()):
        raise ValueError('Unknown burn/guard source')
    parent_manifest=json.loads((a.parent/'source-manifest.json').read_text())
    # Keep the trained model and established ownership/cleanup machinery unchanged.
    for rel in ['efficient_llm/kv_mlp.py','scripts/train_then_burn.py','scripts/gpu_process.py',
                'scripts/gpu_status.py','run_experiments.py']:
        if digest(source/rel)!=parent_manifest[rel]:raise ValueError('Core source changed: '+rel)
    complete=json.loads((a.parent/'supervised/run/complete.json').read_text())
    for job in complete['jobs']:
        if job['status']!='ok' or job['returncode']!=0:raise ValueError('Parent failed')
        for artifact in job['artifacts']:
            if digest(a.parent/'supervised/run'/artifact['path'])!=artifact['sha256']:
                raise ValueError('Parent result changed')
    manifest=source/'resources/downstream_english_20261023.json'
    local_dataset_paths(a.data_root,manifest)
    tokenizer_files=json.loads((a.parent/'source/resources/cx_english_20261020.json').read_text())['tokenizer_files']
    verify_tokenizer(a.tokenizer,tokenizer_files)
    checkpoints={arm:str(a.parent/'supervised/run'/arm/'final') for arm in ['A','B']}
    weights={}
    for arm, cp in checkpoints.items():
        cp=Path(cp);r=json.loads((cp.parent/'result.json').read_text());c=json.loads((cp/'config.json').read_text())
        mapping=[i-1 if arm=='B' and i%4==3 else i for i in range(28)]
        if (r['steps']!=20752 or r['smoke'] or r['arm']!=arm or r['kv_sources']!=mapping or
                c['kv_sources']!=mapping or c['mlp_widths']!=[3072]*28):
            raise ValueError('Wrong checkpoint or architecture: '+arm)
        with (cp/'model.safetensors').open('rb') as f:weights[arm]=hashlib.file_digest(f,'sha256').hexdigest()
    a.output.mkdir();frozen=a.output/'source'
    files=dict(parent_manifest)
    for rel in files:
        old=a.parent/'source'/rel
        if digest(old)!=files[rel]:raise ValueError('Trained source snapshot changed: '+rel)
        target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,target)
    for rel in OVERLAYS:
        target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,target)
        files[rel]=digest(target)
    write(a.output/'source-manifest.json',files)
    write(a.output/'checkpoints.json',checkpoints)
    write(a.output/'input-identity.json',dict(checkpoints=checkpoints,weights_sha256=weights,
        tokenizer=str(a.tokenizer),tokenizer_files=tokenizer_files,
        parent_complete_sha256=digest(a.parent/'supervised/run/complete.json'),
        benchmark_manifest_sha256=digest(manifest)))
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1',
        'TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','NCCL_NVLS_ENABLE':'0',
        'CUDA_VISIBLE_DEVICES':''}
    from accelerate.commands.config.config_args import default_config_file, load_config_from_file
    cached=Path(default_config_file);cached.parent.mkdir(parents=True,exist_ok=True)
    if cached.exists():shutil.copyfile(cached,a.output/'previous-accelerate.yaml')
    config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cached)
    actual=load_config_from_file(str(cached))
    if cached.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:
        raise ValueError('Accelerate config mismatch')
    result=subprocess.run([sys.executable,'-m','accelerate.commands.env'],cwd=frozen,
        env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'},capture_output=True,text=True,check=True)
    (a.output/'accelerate-env.log').write_text(result.stdout+result.stderr);print(result.stdout,flush=True)
    live=inspect()
    if len(live['gpus'])!=8 or not all('B200' in g['name'] for g in live['gpus']):
        raise ValueError('Expected eight B200s')
    write(a.output/'inspection.json',live)
    jobs=a.output/'jobs.json'
    write(jobs,jobs_for(a.output/'checkpoints.json',a.data_root,frozen/'resources/downstream_english_20261023.json',a.tokenizer))
    from run_experiments import load_jobs
    load_jobs(jobs)
    argv=['env',*[k+'='+env[k] for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE',
          'TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS',
          'TOKENIZERS_PARALLELISM','PYTHONDONTWRITEBYTECODE','NCCL_NVLS_ENABLE','CUDA_VISIBLE_DEVICES']],
          sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobs),
          '--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
          '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(burn)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('KV_DOWNSTREAM_SUBMITTED',str(a.output),flush=True)


if __name__=='__main__':main()
