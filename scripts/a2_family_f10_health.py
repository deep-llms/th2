"""Compact read-only F10 status; never returns raw evaluation arrays."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
from eval.a2_family_f9 import read
from efficient_llm.diagnostic_data import sha256
from scripts.gpu_process import inspect


def health(root):
    root=Path(root);results=root/'results/a2_family_20261107/f10'
    print('UTC',datetime.now(timezone.utc).isoformat(),flush=True)
    print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
    errors=[]
    manifest=read(root/'source-manifest.json')
    for name,h in manifest['files'].items():
        if sha256(root/'source'/name)!=h:errors.append('source:'+name)
    cfg=read(root/'config.json')
    for key in ('prepared','FT16','input_manifest','f9_manifest','f10a'):
        item=cfg[key]
        if sha256(item['path'])!=item['sha256']:errors.append('input:'+key)
    print('SOURCE_DATA_LOCKS',len(manifest['files']),errors,flush=True)
    for path in [results/'progress.json',results/'flags.json',results/'FT15.json',results/'report_complete.json',results/'complete.json',root/'supervised/supervisor.json',root/'supervised/burn-verified.json']:
        if path.exists():print(path.name,path.read_text()[-8000:],flush=True)
    checks=[]
    for path in sorted((results/'FT14').glob('*/FT14.json')):
        v=read(path);checks.append(dict(model=path.parent.name,passed=v['passed'],difference=v['difference']))
    print('FT14',json.dumps(checks),flush=True)
    print('COUNTS',json.dumps({name:len(list(results.glob(pattern))) for name,pattern in [('nine','nine/*/per_item.json'),('new','new/*/likelihood_items.json'),('generation','new/*/generation_items.json')]}),flush=True)
    logs=sorted(results.glob('*/*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]
    for p in logs:
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-700));text=f.read().decode(errors='replace')
        print('LOG',str(p.relative_to(results)),text,flush=True)
    for p in [root/'supervisor.log',*sorted((root/'supervised').rglob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]]:
        if p.is_file():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1500));text=f.read().decode(errors='replace')
            print('NATIVE_LOG',str(p.relative_to(root)),text,flush=True)
    p=results/'backup/verified.json'
    if p.exists():print('BACKUP',json.dumps({k:v for k,v in read(p).items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();health(a.root)
