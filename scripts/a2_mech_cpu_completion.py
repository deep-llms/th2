"""CPU branch and dependent readout for the existing A2 M run; no GPU actions.

The original GPU coordinator and frozen source are never edited or restarted.
This finite registered workflow computes missing weights, then consumes the
native terminal receipt and saved probe arrays. It creates no notification timer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from scripts.a2_quant_io import lock,write
from scripts.launch_a2_downstream import freeze_source


def active_weight_workers(root):
    matches=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:args=[v.decode() for v in (entry/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeError):continue
        if 'scripts.a2_mech_worker' in args and '--phase' in args and args[args.index('--phase')+1]=='weights' and any(str(root) in v for v in args):matches.append(int(entry.name))
    return matches


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
    if socket.gethostname()!='thiennh-p6-6y3o-worker-0':raise ValueError('Wrong machine')
    root=a.root;starter=root/'cpu_completion';starter.mkdir(exist_ok=False)
    base=root/'results/a2_mech_20261031';config=lock(base/'config/plan.json');cfg=json.loads(Path(config['path']).read_text())
    frozen=freeze_source(Path(__file__).resolve().parents[1],starter,a.code_commit,extra_files=('docs/A2_mech_spec.md','resources/a2_mech_plan_20261031.json'))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    events=[];started=time.monotonic()
    def worker(ds,parent):
        target=base/'workers/weights'/ds/parent
        if (target/'result.json').exists():return dict(dataset=ds,parent=parent,reused=lock(target/'result.json'))
        if target.exists():
            preserved=starter/'partial_before_completion'/ds/parent
            preserved.parent.mkdir(parents=True,exist_ok=True);target.rename(preserved)
        log=starter/(ds+'-'+parent+'.log')
        argv=[sys.executable,'-u','-m','scripts.a2_mech_worker','--config',config['path'],'--config-sha256',config['sha256'],'--dataset',ds,'--parent',parent,'--phase','weights','--output',str(target)]
        with log.open('x') as stream:
            code=subprocess.run(argv,cwd=root/'source',env={**env,'PYTHONPATH':str(root/'source')},stdout=stream,stderr=subprocess.STDOUT).returncode
        return dict(dataset=ds,parent=parent,returncode=code,result=lock(target/'result.json') if (target/'result.json').exists() else None)
    # If original weights are running, consume their outputs instead of duplicating them.
    active=active_weight_workers(root)
    write(starter/'admission.json',dict(spec_sha256=cfg['spec_sha256'],active_weight_workers=active,code_commit=a.code_commit,config=config,no_gpu_actions=True))
    while active:
        time.sleep(15);active=active_weight_workers(root)
    items=[(d,p) for d in ('dclm','fwe','nem') for p in ('S72','P300','S188','P1T')]
    def safe_worker(item):
        try:return worker(*item)
        except Exception as error:return dict(dataset=item[0],parent=item[1],error=repr(error),action='continue registered peers and readout')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(safe_worker,items):
            events.append(result);write(starter/'progress.json',dict(spec_sha256=cfg['spec_sha256'],phase='weights',events=events));print('CPU_WEIGHTS',json.dumps(result),flush=True)
    write(starter/'progress.json',dict(spec_sha256=cfg['spec_sha256'],phase='dependent_readout_waiting_for_native_terminal',events=events))
    while True:
        try:native=json.loads((root/'supervised/supervisor.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):native={}
        if native.get('finished_at'):break
        time.sleep(15)
    with (starter/'readout.log').open('x') as stream:
        rc=subprocess.run([sys.executable,'-u','-m','scripts.a2_mech_finalize','--root',str(root),'--code-commit',a.code_commit],cwd=frozen,env={**env,'PYTHONPATH':str(frozen)},stdout=stream,stderr=subprocess.STDOUT).returncode
    write(starter/'complete.json',dict(spec_sha256=cfg['spec_sha256'],execution_complete=True,readout_returncode=rc,events=events,elapsed_seconds=time.monotonic()-started,no_gpu_actions=True))
    print('CPU_COMPLETION',rc,flush=True)


if __name__=='__main__':main()
