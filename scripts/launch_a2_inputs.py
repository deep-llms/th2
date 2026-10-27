"""Launch one offline CPU preparation job in an immutable source snapshot.

No GPU reclamation, guard/burn changes, package install or model training.
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

from efficient_llm.a2.data import write_json
from efficient_llm.diagnostic_data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--code-commit',required=True)
    a=p.parse_args();source=Path(__file__).resolve().parents[1]
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing output')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):
        raise ValueError('Reviewed source commit required')
    if subprocess.run(['tmux','has-session','-t',a.session],capture_output=True).returncode==0:
        raise ValueError('Existing session')
    if version('transformers')!='4.57.1':raise ValueError('Use the isolated A2 runtime')
    a.output.mkdir();frozen=a.output/'source';identities={}
    files=[*source.glob('*.py')]
    for folder in ('efficient_llm','scripts'):
        files.extend((source/folder).rglob('*.py'))
    files.extend(source/name for name in ('docs/A2_pilot_spec.md','resources/a2_inputs_20261028.json',
                                         'resources/a2_secondary_20261028.json'))
    for file in files:
        relative=file.relative_to(source);target=frozen/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        identities[str(relative)]=sha256(target)
    write_json(a.output/'source-manifest.json',dict(code_commit=a.code_commit,files=identities))
    job=dict(name='prepare',timeout_seconds=14400,argv=[sys.executable,'-u','-m','scripts.a2_prepare_inputs',
        '--plan',str(frozen/'resources/a2_inputs_20261028.json'),
        '--secondary-plan',str(frozen/'resources/a2_secondary_20261028.json'),
        '--output','{run_dir}/inputs'],required_outputs=[dict(path='inputs/complete.json',json_equals={'status':'passed'})])
    config=a.output/'jobs.json';write_json(config,dict(jobs=[job]))
    from run_experiments import load_jobs
    load_jobs(config)
    env=dict(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='offline',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='8',PYTHONPATH=str(frozen))
    argv=['env',*[k+'='+v for k,v in env.items()],sys.executable,'-u',str(frozen/'run_experiments.py'),
          '--config',str(config),'--project-dir',str(frozen),'--run-dir',str(a.output/'run')]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'runner.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_CPU_INPUTS_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()
