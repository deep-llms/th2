"""Launch bounded anchor numerical/resume qualification after owned handoff."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time,hashlib
from pathlib import Path
import torch,numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.embedding_anchor_runtime import verify_forward_bridge
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
from scripts.warm_order_queue import NATIVE,NATIVE_SHA

BASE=Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_gpu_20261122_a01')
CPU=BASE.with_name('embedding_anchor_20261122_a02')
PREVIOUS=BASE.with_name('attention_shift_20261121_a02')


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--source-sha',required=True)
    p.add_argument('--plan-sha',required=True);a=p.parse_args()
    if len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Source revision required')
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0' or torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Node/runtime mismatch')
    if BASE.exists():raise ValueError('Existing qualification output; reconcile before writes')
    path=Path('resources/embedding_anchor_gpu_qualification_20261122.json')
    if sha(path)!=a.plan_sha:raise ValueError('Qualification registration changed')
    plan=json.loads(path.read_text())
    if (plan['mode']!='qualification_only' or plan['production_updates']!=0 or plan['full_fit_admitted']
            or plan['steps']!=40 or plan['qualification_updates_per_seed']!=440):raise ValueError('Unexpected budget')
    path=Path('resources/embedding_anchor_gpu_source_20261122.json')
    if sha(path)!=a.source_sha:raise ValueError('Source manifest changed')
    pins=json.loads(path.read_text())['files']
    for name,h in pins.items():
        if sha(name)!=h:raise ValueError('Source mismatch '+name)
    oldpins=json.loads((CPU/'source.json').read_text())
    if hashlib.sha256((json.dumps({'files':oldpins},indent=2)+'\n').encode()).hexdigest()!=plan['cpu_source_sha256']:raise ValueError('CPU source package mismatch')
    for name,h in oldpins.items():
        if sha(CPU/'source'/name)!=h:raise ValueError('CPU frozen source changed')
        if name!='efficient_llm/warm_order/embedding_anchor.py' and pins.get(name)!=h:raise ValueError('Unexpected CPU-source migration')
    verify_forward_bridge(CPU/'source/efficient_llm/warm_order/embedding_anchor.py')
    if plan['prepared_root']!=str(CPU/'prepared') or sha(CPU/'prepared/prepared.json')!=plan['prepared_sha256']:raise ValueError('CPU preparation mismatch')
    r=json.loads((CPU/'prepared/prepared.json').read_text())
    if r['fixture_only'] or r['gpu_admitted'] or not r['cpu_references_complete'] or r['train_updates']!=0:raise ValueError('Actual CPU qualification required')
    for seed in (0,1):
        root=CPU/'prepared'/f's{seed}';m=json.loads((root/'config.json').read_text())
        if sha(root/'config.json')!=plan['seed_configs'][str(seed)] or plan['seed_configs'][str(seed)]!=r['seeds'][str(seed)]['config_sha256']:raise ValueError('Seed config changed')
        for name,key in [('initial-fullstate.pt','initial_sha256'),('cpu-reference.pt','reference_sha256')]:
            if sha(root/name)!=m[key]:raise ValueError('CPU artifact changed')
    terminal=json.loads((PREVIOUS/'terminal.json').read_text())
    if terminal.get('success') is not True or terminal.get('native_verified') is not True:raise ValueError('Prior workload not handed off')
    guard=Path('/mnt/local/_gpu_guard')
    if (guard/'DISABLED').exists() or sha(guard/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard owned/changed')
    previous=json.loads((CPU/'cpu-terminal.json').read_text())['current_ownership'];current=inspect()
    if ownership(previous['gpus'])!=ownership(current['gpus']) or any(process(int(pid))!=v for pid,v in previous['processes'].items()):raise ValueError('Ownership changed; inspect before actions')
    if sha(NATIVE)!=NATIVE_SHA or len(current['workers'])!=8 or any(not approved_launcher(process(pid)['ppid'],{NATIVE:NATIVE_SHA}) for pid in current['workers']):raise ValueError('Unknown GPU work')
    if shutil.disk_usage('/mnt/local').free<80*1024**3:raise ValueError('Insufficient space for node-only qualification snapshots/backups')
    session='embedding_anchor_gpu_20261122_a01'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Session already exists')
    execution=BASE/'execution';execution.mkdir(parents=True);frozen=execution/'source'
    for name,h in pins.items():
        dest=frozen/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
        if sha(dest)!=h:raise ValueError('Frozen bytes changed')
    write_json(execution/'source.json',pins);write_json(execution/'plan.json',plan)
    write_json(execution/'admission.json',dict(commit=a.commit,ownership=current,prepared_sha256=plan['prepared_sha256'],utc=time.time(),mode='qualification_only_no_fullfit'))
    env=['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_DATASETS_OFFLINE=1','OMP_NUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH='+str(frozen)]
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.embedding_anchor_qualification_queue','--base',str(BASE)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(session=session,commit=a.commit,source_sha256=sha(execution/'source.json'),utc=time.time()))
    print('ANCHOR_QUALIFICATION_SUBMITTED',json.dumps(dict(session=session,source_sha256=sha(execution/'source.json'))),flush=True)

if __name__=='__main__':main()
