"""Admit exactly the registered anchor pilot using already qualified kernels."""
import argparse,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
import torch,numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.embedding_anchor_prepare import validate_plan
from efficient_llm.warm_order.embedding_anchor_device import device_sources
from efficient_llm.warm_order.embedding_anchor_runtime import verify_forward_bridge
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
from scripts.warm_order_queue import NATIVE,NATIVE_SHA

BASE=Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_fit_20261122_a01')


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    if len(a.commit)!=40 or any(c not in '0123456789abcdef' for c in a.commit):raise ValueError('Exact source revision required')
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0' or torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':raise ValueError('Node/runtime mismatch')
    if BASE.exists():raise ValueError('Existing pilot; reconcile before writes')
    plan_path=Path('resources/embedding_anchor_fit_20261122.json')
    if sha(plan_path)!=a.plan_sha:raise ValueError('Fit registration changed')
    plan=json.loads(plan_path.read_text())
    if (plan['mode']!='one_registered_anchor_pilot' or plan['seeds']!=[0,1]
            or plan['arms']!=['A','C'] or plan['physical_fit_updates_per_seed']!=3072
            or plan['total_physical_fit_updates']!=6144 or plan['measurement_rows']!=512):
        raise ValueError('Unexpected pilot budget')
    if sha(plan['pilot_plan'])!=plan['pilot_plan_sha256']:raise ValueError('Pilot math/recipe changed')
    validate_plan(json.loads(Path(plan['pilot_plan']).read_text()),fixture_only=False)
    source=Path('resources/embedding_anchor_fit_source_20261122.json')
    if sha(source)!=a.source_sha:raise ValueError('Source manifest changed')
    pins=json.loads(source.read_text())['files']
    for name,h in pins.items():
        if sha(name)!=h:raise ValueError('Source mismatch '+name)
    q=Path(plan['qualification_root']);old=q/'execution/source.json'
    if sha(old)!=plan['qualification_source_sha256']:raise ValueError('Qualified source identity changed')
    for name,h in json.loads(old.read_text()).items():
        if sha(q/'execution/source'/name)!=h or pins.get(name)!=h:
            raise ValueError('Qualified code changed '+name)
    if sha(q/'backup.json')!=plan['qualification_backup_sha256']:raise ValueError('Qualification backups changed')
    terminal=json.loads((q/'terminal.json').read_text())
    if not terminal.get('success') or not terminal.get('native_verified'):raise ValueError('Qualification not handed off')
    if sha(q/'native-verified.json')!=plan['qualification_native_sha256']:raise ValueError('Handoff proof changed')
    prepared=Path(plan['prepared_root'])
    if sha(prepared/'prepared.json')!=plan['prepared_sha256']:raise ValueError('CPU preparation changed')
    prepared_manifest=json.loads((prepared/'prepared.json').read_text())
    verify_forward_bridge(prepared.parent/'source/efficient_llm/warm_order/embedding_anchor.py')
    for seed in ('0','1'):
        root=prepared/('s'+seed);meta=json.loads((root/'config.json').read_text())
        if sha(root/'config.json')!=plan['seed_configs'][seed] or plan['seed_configs'][seed]!=prepared_manifest['seeds'][seed]['config_sha256']:
            raise ValueError('Prepared seed binding changed')
        for name,key in [('initial-fullstate.pt','initial_sha256'),('cpu-reference.pt','reference_sha256')]:
            if sha(root/name)!=meta[key]:raise ValueError('CPU artifact changed')
        gate=plan['gates'][seed]
        if sha(gate['path'])!=gate['sha256']:raise ValueError('Qualification receipt changed')
        g=json.loads(Path(gate['path']).read_text())
        if (g['config_sha256']!=plan['seed_configs'][seed] or g['source']!=device_sources()
                or g['fixture_only'] or g['device']!='cuda' or g['actual_updates_executed']!=440
                or g['production_updates_executed']!=0 or not g['lineage_matches_single_segment']):
            raise ValueError('Actual qualification receipt does not bind pilot')
        # Worker repeats full gate/state/policy verification with one visible GPU.
        # Launcher performs metadata-only checks before ownership handoff.
    guard=Path('/mnt/local/_gpu_guard')
    if (guard/'DISABLED').exists() or sha(guard/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Guard owned/changed')
    previous=json.loads((q/'native-verified.json').read_text())['samples'][-1];current=inspect()
    if ownership(previous['gpus'])!=ownership(current['gpus']) or any(process(int(pid))!=v for pid,v in previous['processes'].items()):
        raise ValueError('Ownership changed; inspect before actions')
    if sha(NATIVE)!=NATIVE_SHA or len(current['workers'])!=8 or any(not approved_launcher(process(pid)['ppid'],{NATIVE:NATIVE_SHA}) for pid in current['workers']):raise ValueError('Unknown GPU workload')
    if shutil.disk_usage('/mnt/local').free<80*1024**3:raise ValueError('Insufficient checkpoint/backup space')
    session='embedding_anchor_fit_20261122_a01'
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Session exists')
    execution=BASE/'execution';execution.mkdir(parents=True);frozen=execution/'source'
    for name,h in pins.items():
        dest=frozen/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
        if sha(dest)!=h:raise ValueError('Frozen source mismatch')
    write_json(execution/'source.json',pins);write_json(execution/'plan.json',plan)
    write_json(execution/'admission.json',dict(commit=a.commit,ownership=current,utc=time.time(),
        mode='registered_pilot_fullfit',qualified_kernel_preserved=True,prepared_sha256=plan['prepared_sha256']))
    env=['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
         'HF_DATASETS_OFFLINE=1','OMP_NUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1',
         'CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH='+str(frozen)]
    command='exec '+shlex.join(['env',*env,sys.executable,'-u','-m','scripts.embedding_anchor_fit_queue','--base',str(BASE)])+' >'+shlex.quote(str(execution/'queue.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write_json(execution/'submitted.json',dict(session=session,commit=a.commit,source_sha256=sha(execution/'source.json'),utc=time.time()))
    print('ANCHOR_FIT_SUBMITTED',json.dumps(dict(session=session,source_sha256=sha(execution/'source.json'))),flush=True)

if __name__=='__main__':main()
