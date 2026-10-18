"""Read-only q359 runtime/GPU inspection; writes only a fresh local receipt."""
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
from scripts.train_then_burn import BURN_HASH, GUARD_HASH, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Receipt already exists')
    if socket.gethostname() != 'thiennh-p6-q359-worker-0':
        raise RuntimeError('Wrong node')
    report = inspect()
    report['guard_sha256'] = digest('/mnt/local/_gpu_guard/gpu_guard.sh')
    report['burn_sha256'] = digest(ROOT/'resources/llm_pretrain_burn.py')
    assert report['guard_sha256'] == GUARD_HASH
    assert report['burn_sha256'] == BURN_HASH
    assert not report['guard_disabled'], 'Another workload owns the guard'
    assert len(report['gpus']) == 8 and all('B200' in g['name'] for g in report['gpus'])
    code = '''
import importlib.metadata as md, json, sys, torch, transformers, accelerate, datasets
names=['torch','transformers','accelerate','datasets','tokenizers','safetensors',
       'flash-attn-4','nvidia-cutlass-dsl','quack-kernels','lm-eval']
versions={}
for name in names:
    try: versions[name]=md.version(name)
    except md.PackageNotFoundError: versions[name]=None
print(json.dumps(dict(python=sys.version,executable=sys.executable,versions=versions,
                     cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))
'''
    report['environments'] = {}
    for name in ('train_env', 'eval_env', 'eval_fa4', 'train'):
        python = Path('/mnt/local/conda-py311/envs')/name/'bin/python'
        if not python.is_file():
            continue
        run = subprocess.run([str(python), '-c', code], text=True, capture_output=True,
            timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        if run.returncode:
            report['environments'][name] = {'import_returncode': run.returncode,
                                           'error_tail': run.stderr[-2000:]}
        else:
            report['environments'][name] = json.loads(run.stdout.strip().splitlines()[-1])
    report['asset_directories'] = {}
    for prefix in ('_data', '_models', '_outputs'):
        path = Path('/mnt/local')/prefix/'deep-llms_th2'
        report['asset_directories'][str(path)] = sorted(p.name for p in path.iterdir()) if path.is_dir() else None
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'train.py', ROOT/'bdot/model.py', ROOT/'packed_fa4.py']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print('BDOT_PREFLIGHT ' + json.dumps(report), flush=True)
    print('BDOT_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    main()
