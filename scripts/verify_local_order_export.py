"""Local result-only SHA and NLL replay; never load model/optimizer weights."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify(archive,proof_path,out):
    archive=Path(archive);out=Path(out);proof=json.loads(Path(proof_path).read_text())
    if out.exists():raise ValueError('Fresh local output required')
    if archive.stat().st_size!=proof['bytes'] or sha(archive)!=proof['archive_sha256']:raise ValueError('Archive identity mismatch')
    expected=proof['files'];payload={}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or sum(m.size for m in members)>100_000_000:raise ValueError('Member/expanded size cap')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or m.name not in expected or m.name in payload or p.suffix=='.pt':raise ValueError('Unsafe/unexpected/weight member')
            b=tar.extractfile(m).read()
            if hashlib.sha256(b).hexdigest()!=expected[m.name]:raise ValueError('Member hash mismatch')
            payload[m.name]=b
    if set(payload)!=set(expected):raise ValueError('Missing members')
    out.mkdir(parents=True)
    for name,b in payload.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    report=json.loads((out/'REPORT.json').read_text());prepared=json.loads((out/'inputs/complete.json').read_text())
    rows=[];arrays={}
    for arm in ('C','U','CF','PG'):
        for seed in (0,1):
            name=f'{arm}-s{seed}';ev=out/'eval'/name;run=json.loads((out/f'runs/{name}/complete.json').read_text());r=json.loads((ev/'report.json').read_text())
            if run['step']!=4000 or run['input_tokens']!=65536000 or run['scored_tokens']!=61440000:raise ValueError('Training budget mismatch')
            if sum(run['counts'])!=128000 or run['weights_sha256']!=r['weights_sha256']:raise ValueError('Training/eval identity mismatch')
            if run['compute']['ce_block_tokens']!=65536000*6:raise ValueError('CE compute mismatch')
            if run['compute']['extra_forward_block_tokens']!=65536000*6*{'C':0,'U':0,'CF':4,'PG':1}[arm]:raise ValueError('Extra compute mismatch')
            if next(x for x in report['rows'] if x['arm']==arm and x['seed']==seed)!=r:raise ValueError('Combined row mismatch')
            constant=0
            if arm!='C':
                fit=np.load(ev/'fit_costs.npz',allow_pickle=False)['costs'];constant=int(fit.mean(0).argmin())
                if constant!=r['constant']:raise ValueError('Training-only route selection mismatch')
            for domain in ('dclm','fwe'):
                a=np.load(ev/f'{domain}_costs.npz',allow_pickle=False);arrays[arm,seed,domain]=a
                costs=a['costs'];count=prepared['heldout_counts'][domain];choices=a['choices']
                if costs.shape!=(count,1 if arm=='C' else 4) or not np.isfinite(costs).all():raise ValueError('Evaluation shape/nonfinite')
                values={'canonical':costs[:,0]}
                if arm!='C':values.update(constant=costs[:,constant],uniform=costs.mean(1),oracle=costs.min(1),mixture=costs@np.array([.79,.07,.07,.07],dtype=np.float32))
                if arm in ('CF','PG'):values['learned']=costs[np.arange(count),choices]
                for k,v in values.items():
                    np.testing.assert_allclose(v,a[k],atol=1e-7,rtol=0)
                    if abs(float(v.mean())-r['domains'][domain]['metrics'][k])>1e-7:raise ValueError('NLL replay mismatch')
                rows.append(dict(arm=arm,seed=seed,domain=domain,**{k:float(v.mean()) for k,v in values.items()}))
    for seed in (0,1):
        for domain in ('dclm','fwe'):
            dense=arrays['C',seed,domain]
            for arm in ('U','CF','PG'):
                a=arrays[arm,seed,domain];np.testing.assert_array_equal(a['ids'],dense['ids']);np.testing.assert_array_equal(a['clusters'],dense['clusters'])
                metric='mixture' if arm=='U' else 'learned'
                value=float((a[metric]-dense['canonical']).mean())
                if abs(value-report['comparisons'][f'{domain}_{arm}_minus_C_s{seed}']['mean'])>1e-7:raise ValueError('Paired comparison mismatch')
    with (out/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','seed','domain','canonical','constant','uniform','mixture','learned','oracle'],lineterminator='\n');w.writeheader();w.writerows(rows)
    result=dict(status='verified',members=len(payload),runs=8,domains=prepared['heldout_counts'],archive_sha256=proof['archive_sha256'],weights_loaded=False)
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--proof',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(verify(a.archive,a.proof,a.out)))
