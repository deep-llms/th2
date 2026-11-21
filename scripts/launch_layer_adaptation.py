"""Freeze the registered paired continuation after successful GPU qualification."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.adaptation_reservation import verify
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH

BASE=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_continuation_a01')
GATE=BASE.with_name('layer_adaptation_20261121_gpu_gate_a03')
PREPARED=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_a01/prepared')

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--prepared-sha256',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--gate-sha256',required=True);a=p.parse_args()
    if len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Explicit commit required')
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0':raise ValueError('Wrong node')
    if torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Runtime mismatch')
    if BASE.exists():raise ValueError('Existing gate output; retrieve, never relaunch')
    source=Path('resources/layer_adaptation_run_source_20261121.json')
    if sha(source)!=a.source_sha256:raise ValueError('Source manifest identity')
    pins=json.loads(source.read_text())['files']
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Source mismatch '+path)
    if sha(PREPARED/'manifest.json')!=a.prepared_sha256:raise ValueError('Prepared hash mismatch')
    prepared=json.loads((PREPARED/'manifest.json').read_text())
    for path,h in prepared['files'].items():
        if sha(PREPARED/path)!=h:raise ValueError('Prepared file changed '+path)
    # Reservation plan refers to the immutable CPU preparation source tree.
    old_cwd=Path.cwd()
    try:
        os.chdir(PREPARED.parent/'source');verify(PREPARED/'reservation')
    finally:os.chdir(old_cwd)
    guard=Path('/mnt/local/_gpu_guard')
    if sha(guard/'gpu_guard.sh')!=GUARD_HASH or (guard/'DISABLED').exists():raise ValueError('Guard changed/owned')
    if (guard/'burn.pids').exists() and (guard/'burn.pids').read_text().strip():raise ValueError('Unexpected guard workers')
    previous=GATE
    terminal=json.loads((previous/'terminal.json').read_text())
    if terminal.get('native_verified') is not True or terminal.get('success') is not True:raise ValueError('GPU gate/hand off incomplete')
    if sha(GATE/'gpu-admission.json')!=a.gate_sha256:raise ValueError('Actual GPU gate hash changed')
    from efficient_llm.warm_order.adaptation_run import bound_config
    for name in ('A-s0','B-s0','A-s1','B-s1'):
        bound_config(PREPARED,a.prepared_sha256,name,GATE/'gpu-admission.json',a.gate_sha256)
    prior=json.loads((previous/'native-verified.json').read_text())['samples'][-1];current=inspect()
    if ownership(prior['gpus'])!=ownership(current['gpus']):raise ValueError('Reinspect changed ownership')
    if any(process(int(pid))!=r for pid,r in prior['processes'].items()):raise ValueError('Native PID identity changed')
    native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
    if sha(native)!=digest or len(current['workers'])!=8 or any(not approved_launcher(process(pid)['ppid'],{native:digest}) for pid in current['workers']):raise ValueError('Unverified native workers')
    session='layer_adaptation_20261121_a01'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Session exists')
    if shutil.disk_usage(BASE.parent).free<100*1024**3:raise ValueError('Require100GiB free for snapshots, moments and node backups')
    execution=BASE/'execution';execution.mkdir(parents=True);frozen=execution/'source'
    for path,h in pins.items():
        dest=frozen/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if sha(dest)!=h:raise ValueError('Freeze mismatch')
    write_json(execution/'source.json',pins)
    write_json(execution/'admission.json',dict(commit=a.commit,ownership=current,utc=time.time(),prepared=str(PREPARED/'manifest.json'),prepared_sha256=a.prepared_sha256,gate=str(GATE/'gpu-admission.json'),gate_sha256=a.gate_sha256))
    env=['CUBLAS_WORKSPACE_CONFIG=:4096:8','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','OMP_NUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(frozen)]
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.layer_adaptation_queue','--base',str(BASE)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(session=session,commit=a.commit,source_sha256=sha(execution/'source.json'),utc=time.time()))
    print('ADAPTATION_CONTINUATION_SUBMITTED',json.dumps(dict(session=session,source_sha256=sha(execution/'source.json'))),flush=True)

if __name__=='__main__':main()
