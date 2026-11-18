"""Admit the authorized OHMV causal token-order follow-up, freeze source, start owned queue."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.train import write_json
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH

BASE=Path('/mnt/local/_outputs/deep-llms_th2/strength_refinement_20261119')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--prepared-sha256',required=True);a=ap.parse_args();root=Path.cwd();execution=BASE/'execution';session='strength_refinement_20261119_a01'
    if len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Explicit development-supplied commit required; runner sync tree has no Git metadata')
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    if execution.exists() or (BASE/'runs').exists():raise ValueError('Prior admission/work found; retrieve rather than relaunch')
    if torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Runtime changed')
    guard=Path('/mnt/local/_gpu_guard')
    if sha(guard/'gpu_guard.sh')!=GUARD_HASH or (guard/'DISABLED').exists():raise ValueError('Guard protocol/owner changed')
    if (guard/'burn.pids').exists() and (guard/'burn.pids').read_text().strip():raise ValueError('Unexpected guard-owned workers')
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Session already exists')
    pin=json.loads((root/'resources/strength_refinement_source_20261119.json').read_text())
    for rel,digest in pin['files'].items():
        if sha(root/rel)!=digest:raise ValueError('Source mismatch '+rel)
    if sha(BASE/'inputs/complete.json')!=a.prepared_sha256:raise ValueError('Prepared manifest mismatch')
    prepared=json.loads((BASE/'inputs/complete.json').read_text())
    for rel,digest in prepared['files'].items():
        if sha(BASE/'inputs'/rel)!=digest:raise ValueError('Prepared input/config mismatch')
    for path,digest in prepared['hashes'].items():
        if sha(path)!=digest:raise ValueError('Data changed '+path)
    prior=json.loads((BASE/'cpu-ownership.json').read_text());current=inspect()
    if ownership(current['gpus'])!=ownership(prior['gpus']):raise ValueError('GPU mapping changed')
    for pid,record in prior['processes'].items():
        if process(int(pid))!=record:raise ValueError('Prior native identity changed')
    native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
    if len(current['workers'])!=8 or any(not approved_launcher(process(p)['ppid'],{native:digest}) for p in current['workers']):raise ValueError('Unverified native workers')
    execution.mkdir();frozen=execution/'source'
    for rel in pin['files']:
        target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
        if sha(target)!=pin['files'][rel]:raise ValueError('Freeze mismatch')
    write_json(execution/'source.json',pin['files']);write_json(execution/'admission.json',dict(commit=a.commit,ownership=current,utc=time.time(),input_proof=a.prepared_sha256))
    env=['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','OMP_NUM_THREADS=4','PYTHONPATH='+str(frozen)]
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.strength_refinement_queue','--base',str(BASE)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(session=session,commit=a.commit,utc=time.time(),source_sha256=sha(execution/'source.json')))
    print('STRENGTH_REFINEMENT_SUBMITTED',json.dumps(dict(session=session,source_sha256=sha(execution/'source.json'))),flush=True)
if __name__=='__main__':main()
