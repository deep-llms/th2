"""Offline Tier1 verification/replay; no Torch unpickle and no remote actions."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify(archive,manifest,destination):
    archive=Path(archive);proof=json.loads(Path(manifest).read_text());out=Path(destination)
    if out.exists():raise ValueError('Fresh output required')
    if archive.stat().st_size!=proof['bytes'] or sha(archive)!=proof['archive_sha256']:raise ValueError('Archive hash/size mismatch')
    expected=proof['files'];payloads={}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected):raise ValueError('Unexpected member count')
        if sum(m.size for m in members)>100_000_000:raise ValueError('Unexpected expanded size')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name in payloads or m.name not in expected:raise ValueError('Unsafe/unexpected member')
            b=tar.extractfile(m).read()
            if hashlib.sha256(b).hexdigest()!=expected[m.name]:raise ValueError('Member hash mismatch: '+m.name)
            payloads[m.name]=b
    if set(payloads)!=set(expected):raise ValueError('Missing members')
    out.mkdir(parents=True)
    for name,b in payloads.items():p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    summary=json.loads((out/'REPORT.json').read_text());rows=[]
    for name in ('D-s0','D-s1','R-s0','R-s1','W-s0','W-s1'):
        run=json.loads((out/f'runs/{name}/complete.json').read_text());r=json.loads((out/f'eval/{name}/report.json').read_text())
        if run['step']!=12000 or run['input_tokens']!=196608000 or run['scored_tokens']!=184320000:raise ValueError('Budget differs')
        if run['weights_sha256']!=r['weights_sha256']:raise ValueError('Wrong evaluated checkpoint')
        data=np.load(out/f'eval/{name}/test_costs.npz',allow_pickle=False);cost=data['costs'];choices=data['choices']
        if not np.isfinite(cost).all() or len(cost)!=8192 or len(data['ids'])!=8192:raise ValueError('Invalid test coverage/costs')
        if name.startswith('D'):
            if cost.shape!=(8192,1):raise ValueError('Dense shape')
            metrics={'canonical':cost[:,0]}
        else:
            fit=np.load(out/f'eval/{name}/fit_dev_costs.npz',allow_pickle=False)
            constant=int(np.argmin(fit['fit'].mean(axis=0)))
            if r['constant']!=constant or cost.shape!=(8192,8):raise ValueError('Constant selection/shape mismatch')
            if np.any((choices<0)|(choices>=8)):raise ValueError('Invalid learned choice')
            metrics=dict(canonical=cost[:,0],uniform=cost.mean(1),constant=cost[:,constant],learned=cost[np.arange(len(cost)),choices],oracle=cost.min(1))
        for k,v in metrics.items():
            np.testing.assert_allclose(data[k],v,rtol=0,atol=1e-6)
            if abs(float(v.mean())-r['metrics'][k])>1e-6:raise ValueError('NLL replay differs')
        combined=next(x for x in summary['rows'] if x['arm']==r['arm'] and x['seed']==r['seed'])
        if combined!=r:raise ValueError('Combined report differs from row')
        rows.append(dict(name=name,**{k:float(v.mean()) for k,v in metrics.items()}))
    for seed in (0,1):
        d=np.load(out/f'eval/D-s{seed}/test_costs.npz',allow_pickle=False)
        for arm in ('R','W'):
            x=np.load(out/f'eval/{arm}-s{seed}/test_costs.npz',allow_pickle=False)
            np.testing.assert_array_equal(d['ids'],x['ids']);np.testing.assert_array_equal(d['clusters'],x['clusters'])
        w=np.load(out/f'eval/W-s{seed}/test_costs.npz',allow_pickle=False)
        value=float((w['learned']-d['canonical']).mean())
        if abs(value-summary['comparisons'][f'W_learned_minus_D_s{seed}']['mean'])>1e-6:raise ValueError('Paired comparison differs')
    with (out/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['name','canonical','uniform','constant','learned','oracle']);w.writeheader();w.writerows(rows)
    result=dict(status='verified',archive_sha256=proof['archive_sha256'],members=len(payloads),run_count=6,test_sequences_each=8192,rows=rows,note='Router weights not unpickled locally; actual choices/costs replayed. Native handoff and large checkpoint backups require node proof separately.')
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(verify(a.archive,a.manifest,a.out)))
if __name__=='__main__':main()
