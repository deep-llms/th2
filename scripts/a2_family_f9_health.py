"""Read-only F9 status, frozen-source verification and physical PID ancestry."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
from efficient_llm.diagnostic_data import sha256
from eval.a2_family_f9 import SPEC_SHA


def health(root):
    root=Path(root);result=root/'results/a2_family_20261107/downstream';gpu=inspect()
    out=dict(spec_sha256=SPEC_SHA,utc=datetime.now(timezone.utc).isoformat(),root=str(root),inspection=gpu,artifacts={},logs={},ancestry={},sources={},errors=[])
    out['preparation_files']=[dict(name=p.name,bytes=p.stat().st_size) for p in (result/'preparation').glob('*') if p.is_file()]
    out['cpu_processes']=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if any(x in args for x in (b'scripts.launch_a2_family_f9',b'scripts.a2_family_f9_prepare',b'eval.a2_requests')):
                out['cpu_processes'].append(dict(identity=process(int(p.name)),io=(p/'io').read_text(),stat=(p/'stat').read_text()))
        except (OSError,ValueError):pass
    for p in (result/'preparation').glob('*_prepare.log'):
        out['logs'][str(p.relative_to(root))]=p.read_text(errors='replace').splitlines()[-6:]
    manifest=root/'source-manifest.json'
    if manifest.is_file():
        m=json.loads(manifest.read_text());bad=[p for p,h in m['files'].items() if sha256(root/'source'/p)!=h]
        out['sources']=dict(files=len(m['files']),changed=bad,manifest_sha256=sha256(manifest))
    names=['prepare.log','supervisor.log','supervised/supervisor.json','supervised/run/complete.json',
        'supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json',
        'results/a2_family_20261107/downstream/preparation/ready.json',
        'results/a2_family_20261107/downstream/FT12_loads.json',
        'results/a2_family_20261107/downstream/flags.json',
        'results/a2_family_20261107/downstream/progress.json',
        'results/a2_family_20261107/downstream/complete.json',
        'results/a2_family_20261107/downstream/FT13.json']
    names += [str(p.relative_to(root)) for p in result.glob('FT11_*.json')]
    for name in names:
        p=root/name
        if not p.is_file():continue
        if p.suffix=='.log':out['logs'][name]=p.read_text(errors='replace').splitlines()[-20:]
        else:
            v=json.loads(p.read_text());out['artifacts'][name]=dict(sha256=sha256(p),value=v)
    for label in ('FT12_loads','FT11_batch1','FT11_batch2','scores'):
        folder=result/label
        out[label]=dict(complete=len(list(folder.glob('*/complete.json'))),replayed=len(list(folder.glob('*/replay.json'))),logs=len(list(folder.glob('*.log'))))
        for p in sorted(folder.glob('*.log'),key=lambda x:x.stat().st_mtime)[-8:]:
            out['logs'][str(p.relative_to(root))]=p.read_text(errors='replace').splitlines()[-4:]
    out['FT12_spots']=[]
    for p in sorted((result/'FT12_loads').glob('*/FT12_spot.json')):
        v=json.loads(p.read_text())
        out['FT12_spots'].append(dict(model=p.parent.name,sha256=sha256(p),
            passed=v.get('passed'),expected=v.get('expected'),
            loss=v.get('result',{}).get('loss'),error=v.get('error')))
    for p in (result/'INTERIM.md',result/'REPORT_FULL.md',result/'backup/verified.json'):
        if p.is_file():out['artifacts'][str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha256(p))
    for row in gpu['gpus']:
        for pid in row['pids']:
            chain=[];seen=set()
            while pid>1 and pid not in seen and len(chain)<10:
                seen.add(pid)
                try:value=process(pid)
                except Exception as e:chain.append(dict(pid=pid,error=repr(e)));break
                chain.append(value);pid=value['ppid']
            out['ancestry'][str(row['index'])]=chain
    print('F9_HEALTH '+json.dumps(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();health(a.root)
