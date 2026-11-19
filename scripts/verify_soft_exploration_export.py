"""Local bounded result-only extraction and independent metric/count replay."""
import argparse,json,tarfile
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.soft_exploration_report import calculate,check_arrays,NAMES

def verify(archive,proof,out):
    archive=Path(archive);out=Path(out);proof=json.loads(Path(proof).read_text())
    if archive.stat().st_size!=proof['bytes'] or sha(archive)!=proof['archive_sha256']:raise ValueError('Archive identity failed')
    if proof['bytes']>=25_000_000 or out.exists():raise ValueError('Oversized or nonfresh export')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=set();total=0
        for m in members:
            p=Path(m.name);total+=m.size
            if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name in names or total>100_000_000:raise ValueError('Unsafe archive')
            if p.suffix not in ('.json','.jsonl','.npz','.csv','.md'):raise ValueError('Unexpected payload')
            names.add(m.name)
        if names!=set(proof['files'])|{'export-members.json'}:raise ValueError('Member set mismatch')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
    if json.loads((out/'export-members.json').read_text())!=proof['files']:raise ValueError('Manifest mismatch')
    for p,h in proof['files'].items():
        if sha(out/p)!=h:raise ValueError('Member hash mismatch')
    arrays={}
    for name in NAMES:
        cfg=json.loads((out/'runs'/name/'config.json').read_text())
        for d in ('dclm','fwe'):
            a=dict(np.load(out/'runs'/name/d/'summary.npz'));check_arrays(a,cfg['train']['model']['length']);arrays[name,d]=a
    calculated=calculate(arrays);report=json.loads((out/'REPORT.json').read_text())
    for k,v in calculated.items():
        if report[k]!=v:raise ValueError('Report replay mismatch '+k)
    result=dict(status='PASS',members=len(proof['files']),checks=len(calculated['rows'])+len(calculated['comparisons']),archive_sha256=sha(archive),scope='Lossless per-sequence metric/CI and gate-count replay; forward/reference checks ran on node')
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--proof',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(verify(a.archive,a.proof,a.out)),flush=True)
