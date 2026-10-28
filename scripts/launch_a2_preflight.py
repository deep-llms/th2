"""Initial actual-parent cost gate; no scientific healing or automatic pilot.

Reuse the existing burn supervisor. Run only the real 12-update T8 and one
representative survey before deciding whether the entire preflight fits budget.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
from importlib.metadata import version

from efficient_llm.kv_long import verify_handoff
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest,BURN_HASH,GUARD_HASH
from efficient_llm.a2.data import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--predecessor-burn-script',type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--session',required=True)
    p.add_argument('--stage',choices=('initial','early-grid'),default='initial')
    p.add_argument('--prior',type=Path)
    a=p.parse_args();source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing output')
    for name in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
            raise ValueError('Existing session')
    for name,expected in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
        if version(name).split('+')[0]!=expected:raise ValueError('Wrong review runtime: '+name)
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):
        raise ValueError('Reviewed source commit required')
    from scripts.a2_preflight_inputs import verify_prepared, initial_asset_plan
    from scripts.a2_prepare_inputs import verify_inputs
    plan=json.loads((source/'resources/a2_inputs_20261028.json').read_text())
    inputs_receipt=verify_prepared(a.inputs,source/'docs/A2_pilot_spec.md',plan)
    weights_receipt=verify_inputs(initial_asset_plan(plan))
    if a.stage=='early-grid':
        if a.prior is None:raise ValueError('Prior completed preflight required')
        from efficient_llm.a2.early_grid import verify_reuse
        verify_reuse(source,a.inputs,a.prior,json.loads((source/'resources/a2_early_grid_reuse.json').read_text()))
    verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
    if (digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
        or Path('/mnt/local/_gpu_guard/DISABLED').exists()):raise ValueError('Unknown burn/guard owner')
    a.output.mkdir();frozen=a.output/'source'
    files=[*source.glob('*.py')]
    for directory in ('efficient_llm','scripts','tests/a2'):
        files.extend((source/directory).rglob('*.py'))
    files.extend(source/name for name in ('resources/accelerate_b200.yaml','resources/llm_pretrain_burn.py',
        'resources/a2_inputs_20261028.json','docs/A2_pilot_spec.md'))
    if a.stage=='early-grid':
        files.extend(source/name for name in ('resources/a2_early_grid_reuse.json',
            'docs/results/a2_review_20261028/validation.json','docs/results/a2_review_20261028/identity_metadata.json'))
    identities={}
    for file in files:
        relative=file.relative_to(source);target=frozen/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        identities[str(relative)]=digest(target)
    write_json(a.output/'source-manifest.json',dict(code_commit=a.code_commit,files=identities))
    write_json(a.output/'inputs-verified.json',dict(data=inputs_receipt,weights=weights_receipt))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
         'OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false',
         'PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
    config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache)
    actual=load_config_from_file(str(cache))
    if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:
        raise ValueError('Accelerate mismatch')
    verification=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,
        env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    (a.output/'accelerate-env.log').write_text(verification.stdout+verification.stderr)
    print(verification.stdout,flush=True)
    live=inspect();write_json(a.output/'inspection.json',live)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    from scripts.a2_preflight_inputs import initial_jobs,early_grid_jobs
    jobs=(initial_jobs(frozen,a.inputs.resolve(),port) if a.stage=='initial'
        else early_grid_jobs(a.inputs.resolve(),a.prior.resolve(),port))
    path=a.output/'jobs.json';write_json(path,dict(jobs=jobs))
    from run_experiments import load_jobs
    load_jobs(path)
    argv=['env',*[key+'='+env[key] for key in env if key in (
        'CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE',
        'HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE',
        'TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')],sys.executable,'-u','-m','scripts.train_then_burn',
        '--config',str(path),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_INITIAL_PREFLIGHT_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
