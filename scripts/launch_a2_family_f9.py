"""CPU preparation then authorized native burn reclaim for F9 only."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from eval.a2_family_f9 import SPEC_SHA,read,write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.kv_long import verify_handoff
from scripts.a2_family_f9_run import locked
from scripts.a2_quant_io import lock
from scripts.gpu_process import inspect,process
from scripts.launch_a2_downstream import freeze_source
from scripts.train_then_burn import BURN_HASH,GUARD_HASH,approved_launcher
from run_experiments import load_jobs


def predecessor(cfg):
    script=Path(cfg['predecessor_burn_script'])
    verify_handoff(script.parents[3],script,sha256)
    if sha256(script)!=BURN_HASH or sha256('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:
        raise ValueError('Unrecognized native burn/guard')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Native GPU guard already owned')
    fresh=inspect()
    if fresh['host']!=cfg['host'] or len(fresh['gpus'])!=8:raise ValueError('Wrong physical host/GPUs')
    for gpu in fresh['gpus']:
        if not gpu['pids']:raise ValueError('Expected owned native burn on every GPU')
        for pid in gpu['pids']:
            if not (approved_launcher(pid,{str(script):BURN_HASH}) or approved_launcher(process(pid)['ppid'],{str(script):BURN_HASH})):
                raise ValueError('Unknown GPU process; no signals sent')
    return fresh


def jobs(config,root):
    return dict(jobs=[dict(name='family-F9-evaluation',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f9_run','--config',str(config),'--receipt','{run_dir}/f9-complete.json'],
        required_outputs=[dict(path='f9-complete.json',json_equals=dict(execution_complete=True))]),
        dict(name='family-F9-scientific-backup',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f9_backup','--root',str(root),'--receipt','{run_dir}/f9-backup.json'],
        required_outputs=[dict(path='f9-backup.json',json_equals=dict(verified=True))])])


def launch(plan,output,session,commit):
    source=Path(__file__).resolve().parents[1];cfg=read(plan);out=Path(output)
    if cfg['spec_sha256']!=SPEC_SHA or cfg['host']!=socket.gethostname() or out.exists():
        raise ValueError('Wrong spec/host or output exists; no duplicate launch')
    if version('transformers')!='4.57.1' or version('torch').split('+')[0]!='2.14.1':raise ValueError('Wrong native runtime')
    proof=read(source/cfg['cpu_proof'])
    if not proof['launch_ready'] or proof['status']!='PASS':raise ValueError('CPU admission not ready')
    for p,h in proof['files'].items():
        if sha256(source/p)!=h:raise ValueError('Changed tested source: '+p)
    for name in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    predecessor(cfg)
    out.mkdir();frozen=freeze_source(source,out,commit,extra_files=cfg['extra_source_files'])
    result=out/'results/a2_family_20261107/downstream';result.mkdir(parents=True)
    cfg.update(code_commit=commit,downstream_root=str(result),source_manifest=lock(out/'source-manifest.json'))
    for common in cfg['benchmarks'].values():
        for key in ('input_spec','manifest','tokenizer_manifest'):
            if key in common and not Path(common[key]['path']).is_absolute():common[key]=lock(frozen/common[key]['path'])
    cfg['previous_report']=lock(frozen/cfg['previous_report']['path']) if not Path(cfg['previous_report']['path']).is_absolute() else cfg['previous_report']
    mapped=write(out/'prepare-plan.json',cfg)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':commit,
        'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0'}
    with (out/'prepare.log').open('x') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.a2_family_f9_prepare','--plan',mapped['path'],
            '--output',str(result/'preparation')],cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    ready=read(result/'preparation/ready.json');cfg=locked(ready['config'])
    if not ready['cpu_ready'] or not ready['models']:raise ValueError('No prepared models')
    cfg['downstream_root']=str(result);config=write(result/'config.json',cfg)
    job=write(out/'jobs.json',jobs(config['path'],result));load_jobs(Path(job['path']))
    fresh=predecessor(cfg);inspection=write(out/'inspection.json',fresh)
    command=[sys.executable,'-u','-m','scripts.train_then_burn','--config',job['path'],
        '--output',str(out/'supervised'),'--inspection',inspection['path'],'--host',cfg['host'],
        '--burn-session',session+'_burn','--predecessor-burn-script',cfg['predecessor_burn_script']]
    shell='exec '+shlex.join(command)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    write(out/'launch.json',dict(config=config,jobs=job,source_manifest=cfg['source_manifest'],session=session,
        scope='F9 evaluation only; F0-F8/Depth immutable; no training',native_handoff=True))
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    print('F9_NATIVE_QUEUE_SUBMITTED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('plan','output','session','code-commit'):p.add_argument('--'+key,required=True)
    a=p.parse_args();launch(a.plan,a.output,a.session,a.code_commit)
