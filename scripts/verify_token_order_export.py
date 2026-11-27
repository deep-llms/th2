"""Local, bounded Tier1 extraction and per-sequence metric/CI replay. No remote actions."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.evaluate import paired_ci
from efficient_llm.warm_order.token_report import NAMES
from efficient_llm.warm_order.data import sha


def verify(archive,proof,out,target_count=480):
    archive=Path(archive);out=Path(out);proof=json.loads(Path(proof).read_text())
    if archive.stat().st_size!=proof['bytes'] or sha(archive)!=proof['archive_sha256']:raise ValueError('Archive integrity failure')
    if archive.stat().st_size>=25_000_000 or out.exists():raise ValueError('Oversized or nonfresh export')
    out.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=set();total=0
        for m in members:
            p=Path(m.name);total+=m.size
            if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name in names or total>100_000_000:raise ValueError('Unsafe archive')
            if p.suffix in ('.bin','.starts') or p.name in ('weights.pt','fullstate.pt','costs.npy','features.npy'):raise ValueError('Forbidden large/raw member')
            names.add(m.name)
        expected=set(proof['files'])|{'export-members.json'}
        if names!=expected:raise ValueError('Unexpected archive member set')
        for m in members:
            dst=out/m.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(tar.extractfile(m).read())
    if json.loads((out/'export-members.json').read_text())!=proof['files']:raise ValueError('Manifest disagreement')
    for p,want in proof['files'].items():
        if sha(out/p)!=want:raise ValueError('Member hash failure '+p)
    report=json.loads((out/'REPORT.json').read_text());arrays={};checks=0
    for name in NAMES:
        cfg=json.loads((out/'runs'/name/'config.json').read_text());done=json.loads((out/'runs'/name/'complete.json').read_text())
        if done['frozen_weights_sha256']!=cfg['hashes'][cfg['weights']]:raise ValueError('Weight provenance mismatch')
        for domain in ('dclm','fwe'):
            a=np.load(out/'runs'/name/domain/'summary.npz');arrays[name,domain]=a
            rows=[r for r in report['rows'] if r['name']==name and r['domain']==domain]
            for r in rows:
                if not np.isfinite(a[r['metric']]).all() or abs(float(a[r['metric']].mean())-r['nll'])>1e-12:raise ValueError('Metric replay failed')
                if len(a['ids'])!=r['sequences'] or r['tokens']!=target_count*r['sequences']:raise ValueError('Denominator mismatch')
                checks+=1
            for key in ['learned','prefix','token_id']:
                if key+'_counts' in a:
                    if not np.all(a[key+'_counts'].sum(1)==target_count):raise ValueError('Routing coverage mismatch')
    for key,want in report['comparisons'].items():
        name,domain,comparison=key.split('/');a=arrays[name,domain];ref=comparison.removeprefix('learned-minus-')
        b=arrays['D-s'+name[-1],domain] if ref=='dense' else a
        np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['clusters'],b['clusters'])
        got=paired_ci(a['learned']-b['canonical' if ref=='dense' else ref],a['clusters'])
        for k,v in got.items():
            if abs(v-want[k])>1e-12:raise ValueError('CI replay failed '+key)
        checks+=1
    for arm in ('R','W'):
        value=all(report['comparisons'][f'{arm}-s{s}/{d}/learned-minus-{r}']['mean']<=-.01 for s in (0,1) for d in ('dclm','fwe') for r in ('dense','constant'))
        if value!=report['promising_screen'][arm]:raise ValueError('Decision replay failed')
    for key,want in report['means'].items():
        arm,domain,metric=key.split('/')
        got=np.mean([r['nll'] for r in report['rows'] if r['name'].startswith(arm) and r['domain']==domain and r['metric']==metric])
        if abs(float(got)-want)>1e-12:raise ValueError('Seed mean mismatch')
    result=dict(status='PASS',members=len(proof['files']),checks=checks,archive_sha256=sha(archive),
        scope='Local per-sequence replay; raw token-cost replay completed separately on node, raw caches not downloaded')
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--proof',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(verify(a.archive,a.proof,a.out)),flush=True)
if __name__=='__main__':main()
