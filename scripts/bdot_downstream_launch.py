"""Prepare a fresh downstream queue from the unchanged, trained source snapshot."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.benchmark_data import local_dataset_paths
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect

OVERLAYS=('eval/bdot_harness.py','eval/benchmark_data.py','eval/downstream_suite.py',
          'scripts/check_downstream_adapter.py','scripts/bdot_downstream_launch.py',
          'tests/bdot/test_harness.py','tests/test_downstream_suite.py',
          'resources/downstream_english_20261023.json')


def read(path):return json.loads(path.read_text())


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--host',required=True)
    p.add_argument('--session',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.host:raise ValueError('Wrong host')
    if version('lm-eval')!='0.4.13':raise ValueError('Unexpected harness version')
    previous=read(a.parent/'long-pair.json');source=Path(previous['source'])
    manifest_file=source/'SOURCE_MANIFEST.json'
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest()!=previous['source_manifest_sha256']:
        raise ValueError('Parent scientific source manifest changed')
    manifest=read(manifest_file)
    for rel,checksum in manifest['files'].items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Parent source changed: '+rel)
    if read(a.parent/'supervised/run/long-pair-report.json')['status']!='ok':
        raise ValueError('Pretraining pair did not pass final validation')
    local_dataset_paths(a.data_root,ROOT/'resources/downstream_english_20261023.json')
    burn_gate(a.parent)
    checkpoints={arm:str(a.parent/'prepared/runs'/(arm+'-s0')/'checkpoint-10000')
                 for arm in ('D','Partial-25-cosine')}
    for cp in checkpoints.values():
        path=Path(cp)
        if read(path/'trainer_state.json')['global_step']!=10000 or not (path/'tokenizer.json').is_file():
            raise ValueError('Incomplete checkpoint/tokenizer')
    a.root.mkdir(parents=True,exist_ok=False)
    snapshot=a.root/'source';snapshot.mkdir()
    for rel in manifest['files']:
        target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/rel,target)
    shutil.copy2(manifest_file,snapshot/'PARENT_SOURCE_MANIFEST.json')
    for rel in OVERLAYS:
        target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target)
    files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in snapshot.rglob('*') if p.is_file()}
    write(snapshot/'SOURCE_MANIFEST.json',dict(parent_manifest_sha256=previous['source_manifest_sha256'],files=files))
    write(a.root/'checkpoints.json',checkpoints)
    jobs=[]
    def job(name,argv,artifact,**expected):
        jobs.append(dict(name=name,argv=argv,gpus=list(range(8)),timeout_seconds=7200,
                         required_outputs=[dict(path=artifact,json_equals=dict(status='ok',**expected))]))
    cp=str(a.root/'checkpoints.json');data=str(a.data_root);dm=str(snapshot/'resources/downstream_english_20261023.json')
    job('adapter_check',['{python}',str(snapshot/'scripts/check_downstream_adapter.py'),
        '--checkpoints',cp,'--output','{run_dir}/adapter-check.json'],'adapter-check.json')
    for stage,limit in [('smoke',8),('full',None)]:
        argv=['{python}','-m','eval.downstream_suite','--checkpoints',cp,
              '--output','{run_dir}/'+stage,'--dataset-root',data,'--manifest',dm]
        if limit is not None:argv+=['--limit',str(limit)]
        job(stage,argv,stage+'/summary.json',diagnostic=limit is not None)
    write(a.root/'jobs.json',dict(jobs=jobs))
    write(a.root/'preparation.json',dict(status='ok',parent=str(a.parent),source=str(snapshot),
          scientific_source=str(source),overlays=OVERLAYS,checkpoints=checkpoints,
          manifest_sha256=hashlib.sha256(Path(dm).read_bytes()).hexdigest()))
    burn_gate(a.parent)
    write(a.root/'preflight.json',inspect())
    subprocess.run([sys.executable,str(snapshot/'scripts/bdot_b200_launch.py'),
        '--root',str(a.root),'--host',a.host,'--session',a.session,
        '--jobs-config',str(a.root/'jobs.json'),
        '--predecessor-burn-script',str(source/'resources/llm_pretrain_burn.py')],cwd=snapshot,check=True)


if __name__=='__main__':main()
