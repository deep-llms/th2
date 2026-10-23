"""Prepare gated STOP verification using the established all-eight-GPU supervisor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bdot_long_pair import verify_source, read, write
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect

OVERLAYS = ('bdot/model.py', 'bdot/partial_mlp.py', 'bdot/stopping.py',
    'train.py', 'packed_evaluation.py', 'resume_recipe.py',
    'tests/bdot/test_stopping.py', 'tests/bdot/ddp_probe.py', 'tests/bdot/resume_gpu_probe.py',
    'scripts/check_stop_gpu.py', 'scripts/bdot_stop_launch.py')


def snapshot(original, target, overlays):
    target.mkdir()
    for rel in read(original/'SOURCE_MANIFEST.json')['files']:
        dest=target/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original/rel, dest)
    for rel in overlays:
        dest=target/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in target.rglob('*') if p.is_file()}
    write(target/'SOURCE_MANIFEST.json', dict(files=files))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','predecessor'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--host',required=True);p.add_argument('--session',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.host: raise ValueError('Wrong node')
    original=Path(read(a.parent/'long-pair.json')['source'])
    verify_source(original)
    burn_gate(a.predecessor)
    if read(a.parent/'supervised/run/long-pair-report.json')['status']!='ok':
        raise ValueError('Parent pair incomplete')
    checkpoints={arm:str(a.parent/'prepared/runs'/(arm+'-s0')/'checkpoint-10000')
                 for arm in ('D','Partial-25-cosine')}
    for cp in checkpoints.values():
        if read(Path(cp)/'trainer_state.json')['global_step']!=10000:
            raise ValueError('Parent checkpoint step mismatch')
    a.root.mkdir(parents=True,exist_ok=False)
    source=a.root/'source';snapshot(original,source,OVERLAYS)
    write(a.root/'checkpoints.json',checkpoints)
    config=read(source/'configs/bdot/Partial-25-cosine.json')
    config['bdot']['curriculum']='none'
    config['bdot_stop']=dict(train_random_prob=.5,min_steps=2)
    write(a.root/'resume-model.json',config)
    jobs=[]
    def job(name,argv,artifact,status='ok',gpus=None):
        jobs.append(dict(name=name,argv=argv,gpus=list(range(8)) if gpus is None else gpus,
                         timeout_seconds=1800,required_outputs=[dict(path=artifact,json_equals=dict(status=status))]))
    job('numerics',['{python}',str(source/'scripts/check_stop_gpu.py'),
        '--checkpoints',str(a.root/'checkpoints.json'),'--output','{run_dir}/numerics.json'],'numerics.json')
    job('ddp',['{python}','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(source/'tests/bdot/ddp_probe.py'),'{run_dir}/ddp','--stop','--device','cuda','--bf16'],
        'ddp/rank0.json',status='passed')
    job('resume',['{python}',str(source/'tests/bdot/resume_gpu_probe.py'),'suite',
        '--output','{run_dir}/resume','--accelerate-config',str(source/'resources/accelerate_b200.yaml'),
        '--steps','8','--split','4','--soft-split','2','--curriculum','none',
        '--variant-config',str(a.root/'resume-model.json')], 'resume/summary.json', status='passed')
    write(a.root/'jobs.json',dict(jobs=jobs))
    burn_gate(a.predecessor)
    write(a.root/'preflight.json',inspect())
    subprocess.run([sys.executable,str(source/'scripts/bdot_b200_launch.py'),
        '--root',str(a.root),'--host',a.host,'--session',a.session,'--jobs-config',str(a.root/'jobs.json'),
        '--predecessor-burn-script',str(a.predecessor/'source/resources/llm_pretrain_burn.py')],
        cwd=source,check=True)


if __name__=='__main__':main()
