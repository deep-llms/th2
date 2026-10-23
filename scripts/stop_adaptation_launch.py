"""Launch only after numerical/DDP/resume gates and verified burn restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_stop_adaptation import prepare,read,write
from scripts.bdot_stop_launch import snapshot
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect

OVERLAYS=('bdot/model.py','scripts/prepare_stop_adaptation.py','scripts/stop_adaptation_launch.py',
          'eval/stop_study.py','scripts/stop_eval_suite.py','scripts/check_stop_head_gpu.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','verification','parent'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--session',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.host:raise ValueError('Wrong node')
    burn_gate(a.verification)
    run=a.verification/'supervised/run'
    if read(a.verification/'supervised/supervisor.json')['training_status']!='ok':raise ValueError('Verification failed')
    if read(run/'resume/summary.json')['status']!='passed':raise ValueError('Eight-GPU FA4 resume failed')
    if any(read(run/f'ddp/rank{r}.json')['status']!='passed' for r in range(2)):raise ValueError('DDP failed')
    numerics=read(run/'numerics.json')
    if numerics['status']!='ok':raise ValueError('Numerics failed')
    for name,r in numerics['models'].items():
        times={x['fixed_depth']:x['decode_ms'] for x in r['timing'] if x['batch']==1}
        if times[2]>.9*times[8]:raise ValueError('Even optimistic fixed exit lacks 10% speed headroom: '+name)
    verified=a.verification/'source'
    for rel,checksum in read(verified/'SOURCE_MANIFEST.json')['files'].items():
        if hashlib.sha256((verified/rel).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Verified source changed: '+rel)
    # Only the post-load freeze may change model.py after the GPU gate; never
    # accidentally launch a differently optimized scientific implementation.
    old=(verified/'bdot/model.py').read_text();new=(ROOT/'bdot/model.py').read_text()
    begin=new.index('    @classmethod\n    def from_pretrained',new.index('class BDOTForCausalLM'))
    end=new.index('    def __init__(self, config):',begin)
    if new[:begin]+new[end:]!=old:raise ValueError('Unexpected unverified model change')
    a.root.mkdir(parents=True,exist_ok=False)
    source=a.root/'source';snapshot(verified,source,OVERLAYS)
    jobs=prepare(a.parent,a.root,source)
    jobs.insert(0,dict(name='head_smoke',gpus=list(range(8)),timeout_seconds=1800,
        argv=['{python}',str(source/'scripts/check_stop_head_gpu.py'),
              '--checkpoints',str(a.root/'branches.json'),'--output','{run_dir}/head-smoke.json'],
        required_outputs=[dict(path='head-smoke.json',json_equals=dict(status='ok'))]))
    jobs.append(dict(name='STOP-evaluation',gpus=list(range(8)),timeout_seconds=10800,
        argv=['{python}',str(source/'scripts/stop_eval_suite.py'),'--root',str(a.root),
              '--parent',str(a.parent),'--output','{run_dir}/evaluation'],
        required_outputs=[dict(path='evaluation/summary.json',json_equals=dict(status='ok'))]))
    write(a.root/'jobs.json',dict(jobs=jobs))
    burn_gate(a.verification);write(a.root/'preflight.json',inspect())
    subprocess.run([sys.executable,str(source/'scripts/bdot_b200_launch.py'),
        '--root',str(a.root),'--host',a.host,'--session',a.session,'--jobs-config',str(a.root/'jobs.json'),
        '--predecessor-burn-script',str(verified/'resources/llm_pretrain_burn.py')],cwd=source,check=True)


if __name__=='__main__':main()
