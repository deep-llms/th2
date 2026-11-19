"""Local, bounded Tier1 extraction and per-sequence metric/CI replay. No remote actions."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.evaluate import paired_ci
from efficient_llm.warm_order.auxiliary_report import NAMES
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
    from efficient_llm.warm_order.auxiliary_report import calculate
    report=json.loads((out/'REPORT.json').read_text());arrays={}
    for name in NAMES:
        policy=json.loads((out/'runs'/name/'policy.json').read_text())
        for domain in ('dclm','fwe'):
            a=dict(np.load(out/'runs'/name/domain/'summary.npz'));arrays[name,domain]=a
            ix=np.arange(len(a['ids']));np.testing.assert_array_equal(a['choices'],a['scores'].argmin(1))
            for metric,expected in [('identity',a['costs'][:,0]),('uniform',a['costs'].mean(1)),('constant',a['costs'][:,policy['constant']]),('learned',a['costs'][ix,a['choices']]),('random',a['costs'][ix,a['random_choices']]),('oracle',a['costs'].min(1))]:np.testing.assert_array_equal(a[metric],expected)
            np.testing.assert_allclose(a['actual'],a['learned'],atol=2e-4,rtol=0)
            for k,v in a.items():
                if not np.isfinite(v).all():raise ValueError('Nonfinite summary')
                if k.startswith('count-') and (np.any(v<0) or np.any(v>target_count)):raise ValueError('Route count mismatch')
    got=calculate(arrays)
    for key in ('rows','comparisons','continuation_all_seeds_domains','flags'):
        if got[key]!=report[key]:raise ValueError('Scientific replay mismatch '+key)
    checks=len(report['rows'])+len(report['comparisons'])
    result=dict(status='PASS',members=len(proof['files']),checks=checks,archive_sha256=sha(archive),
        scope='Local per-sequence replay; raw token-cost replay completed separately on node, raw caches not downloaded')
    (out/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--proof',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(verify(a.archive,a.proof,a.out)),flush=True)
if __name__=='__main__':main()
