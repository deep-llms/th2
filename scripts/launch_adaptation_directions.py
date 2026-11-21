"""Admit only the bounded frozen pilot after previous owned studies hand off."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
import torch,numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH

BASE=Path('/mnt/local/_outputs/deep-llms_th2/adaptation_directions_20261121_a01')
PREVIOUS=BASE.with_name('adaptation_factors_20261121_a02')

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--prepared-sha',required=True);a=p.parse_args()
    if len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Commit required')
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0' or torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Node/runtime mismatch')
    if (BASE/'execution').exists():raise ValueError('Existing execution: reconcile, no relaunch')
    p=Path('resources/adaptation_directions_source_20261121.json')
    if sha(p)!=a.source_sha:raise ValueError('Source manifest changed')
    pins=json.loads(p.read_text())['files']
    for name,h in pins.items():
        if sha(name)!=h:raise ValueError('Source mismatch '+name)
    prepared=BASE/'prepared/prepared.json'
    if sha(prepared)!=a.prepared_sha:raise ValueError('Actual CPU prepared identity mismatch')
    r=json.loads(prepared.read_text())
    if r['actual_panels_loaded']!=2 or not r['cpu_native_parity'] or r['train_updates']!=0:raise ValueError('CPU qualification missing')
    oldpins=json.loads((BASE/'source.json').read_text())
    for name,h in oldpins.items():
        if pins.get(name)!=h or sha(BASE/'source'/name)!=h:raise ValueError('CPU-qualified source changed')
    for seed in (0,1):
        root=BASE/'prepared'/f's{seed}'
        if sha(root/'config.json')!=r['seeds'][str(seed)]['config_sha256'] or sha(root/'cpu-reference.pt')!=r['seeds'][str(seed)]['cpu_reference_sha256']:raise ValueError('Prepared seed changed')
    terminal=json.loads((PREVIOUS/'terminal.json').read_text())
    if terminal.get('success') is not True or terminal.get('native_verified') is not True:raise ValueError('Prior study not handed off; do not stop it')
    guard=Path('/mnt/local/_gpu_guard')
    if (guard/'DISABLED').exists() or sha(guard/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard currently owned/changed')
    previous=json.loads((PREVIOUS/'native-verified.json').read_text())['samples'][-1];current=inspect()
    if ownership(previous['gpus'])!=ownership(current['gpus']) or any(process(int(pid))!=v for pid,v in previous['processes'].items()):raise ValueError('Ownership changed')
    native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
    if sha(native)!=digest or len(current['workers'])!=8 or any(not approved_launcher(process(pid)['ppid'],{native:digest}) for pid in current['workers']):raise ValueError('Unverified native workers')
    session='adaptation_directions_20261121_a01'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
    execution=BASE/'execution';execution.mkdir();frozen=execution/'source'
    for name,h in pins.items():
        dest=frozen/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
        if sha(dest)!=h:raise ValueError('Frozen bytes changed')
    write_json(execution/'source.json',pins);write_json(execution/'admission.json',dict(commit=a.commit,ownership=current,prepared_sha256=a.prepared_sha,utc=time.time(),mode='frozen_inference_only'))
    env=['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1','OMP_NUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(frozen)]
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.adaptation_directions_queue','--base',str(BASE)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True);subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(session=session,commit=a.commit,source_sha256=sha(execution/'source.json'),utc=time.time()))
    print('ADAPTATION_DIRECTIONS_SUBMITTED',json.dumps(dict(session=session,source_sha256=sha(execution/'source.json'))),flush=True)
if __name__=='__main__':main()
