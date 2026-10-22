"""Read-only runtime, GPU ownership and asset inventory for attention comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_process import inspect

PROBE = r'''
import importlib.metadata as md, json, sys
import torch, transformers, accelerate, datasets, tokenizers, safetensors
packages = dict(sorted((d.metadata['Name'].lower().replace('_','-'), d.version)
                      for d in md.distributions() if d.metadata['Name']))
out = dict(python=sys.version, executable=sys.executable, packages=packages,
           cuda=torch.version.cuda, abi=torch._C._GLIBCXX_USE_CXX11_ABI,
           compiled_archs=torch._C._cuda_getArchFlags(), imports='ok')
if 'flash-attn' in packages:
    from flash_attn import flash_attn_varlen_func
    import flash_attn_2_cuda
    out['fa2_import']='ok'
if 'flash-attn-4' in packages:
    from flash_attn.cute.interface import flash_attn_varlen_func
    out['fa4_import']='ok'
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--require-absent')
    a = p.parse_args()
    if socket.gethostname() != a.host or a.output.exists():
        raise ValueError('Wrong host or existing receipt')
    prefix = Path('/mnt/local/conda-py311/envs')
    if a.require_absent and (prefix/a.require_absent).exists():
        raise ValueError('Requested new environment already exists')
    report = inspect()
    report['environments'] = {}
    for name in ('train', 'eval', 'train_fa2', 'train_fa2_t28'):
        python = prefix/name/'bin/python3.11'
        if not python.exists():
            continue
        run = subprocess.run([str(python), '-c', PROBE], capture_output=True, text=True,
            timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'',
                             'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1'})
        if run.returncode:
            raise RuntimeError(f'{name} import failed: {run.stderr[-3000:]}')
        report['environments'][name] = json.loads(run.stdout.strip().splitlines()[-1])
    report['safety_files'] = {}
    for file in ('/mnt/local/_gpu_guard/gpu_guard.sh', '/tmp/llm_pretrain_burn.py',
                 str(ROOT/'resources/llm_pretrain_burn.py')):
        path=Path(file)
        report['safety_files'][file]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    report['assets'] = {}
    for kind in ('_data', '_models', '_outputs'):
        path=Path('/mnt/local')/kind/'deep-llms_th2'
        report['assets'][str(path)]=sorted(str(x.relative_to(path)) for x in path.glob('*/*')) if path.exists() else []
    report['gpu_health']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)
    report['disk']=subprocess.check_output(['df','-h','/mnt/local'],text=True)
    if a.compare:
        old=json.loads(a.compare.read_text())
        for name in ('train','eval','train_fa2'):
            if report['environments'][name] != old['environments'][name]:
                raise ValueError(f'Protected environment changed: {name}')
        if report['safety_files'] != old['safety_files']:
            raise ValueError('Safety/driver workload files changed')
        report['protected_environments_unchanged']=True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: json.dump(report,f,indent=2)
    print('ATTENTION_INVENTORY '+json.dumps(report),flush=True)
    print('ATTENTION_INVENTORY_OK',flush=True)


if __name__ == '__main__': main()
