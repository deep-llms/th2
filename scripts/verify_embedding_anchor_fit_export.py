"""Replay small anchor-pilot scores and validate lineage/qualification joins."""
import argparse,json,tarfile
from dataclasses import asdict
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor import exposure
from efficient_llm.warm_order.embedding_anchor_eval import replay


def verify(archive,backup,expected_sha,out,qualification):
    archive,out,qualification=Path(archive),Path(out),Path(qualification)
    b=json.loads(Path(backup).read_text())
    if out.exists() or sha(archive)!=expected_sha or b['archive_sha256']!=expected_sha:
        raise ValueError('Trusted archive and fresh output required')
    if b['physical_fit_updates']!=6144 or len(b['node_only_copies'])!=20:
        raise ValueError('Pilot budget/backup inventory mismatch')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        if len(members)!=17 or len(set(m.name for m in members))!=17 or sum(m.size for m in members)>25_000_000:
            raise ValueError('Unexpected small archive inventory')
        if set(m.name for m in members)!=set(b['files']):raise ValueError('Member inventory mismatch')
        for m in members:
            p=Path(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt':raise ValueError('Unsafe archive member')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
            if sha(p)!=b['files'][m.name]:raise ValueError('Member hash mismatch')
    read=lambda p:json.loads(Path(p).read_text())
    plan=read(out/'execution/plan.json');sources=read(out/'execution/source.json')
    old=read(qualification/'execution/source.json')
    if sha(qualification/'execution/source.json')!=plan['qualification_source_sha256'] or any(sources.get(n)!=h for n,h in old.items()):
        raise ValueError('Qualified source changed')
    if len(sources)!=362 or len(old)!=356:raise ValueError('Unexpected source package')
    checks={}
    for seed in ('0','1'):
        root=out/('s'+seed);done=read(root/'complete.json');r=read(root/'fit-recipe.json')
        gate=qualification/('s'+seed)/'complete.json'
        if (sha(gate)!=plan['gates'][seed]['sha256'] or done['registration']!=r
                or r['gate_sha256']!=sha(gate) or r['seed']!=int(seed) or r['fixture_only']
                or r['physical_fit_updates']!=3072 or r['prepared_sha256']!=plan['prepared_sha256']
                or r['plan_sha256']!=plan['pilot_plan_sha256']
                or r['kernel_source']!=read(gate)['source']
                or r['fit_source_sha256']!=sources['efficient_llm/warm_order/embedding_anchor_fit.py']
                or sha(root/'fit-recipe.json')!=done['fit_recipe_sha256']):
            raise ValueError('Qualification/fit registration mismatch')
        checks[seed]=replay(root/'evaluation');m=read(root/'evaluation/manifest.json');proof=m['checkpoint_proofs']
        if (sha(root/'evaluation/REPORT.json')!=done['report_sha256'] or proof['warmup']!=done['warmup']
                or proof['step']!=1920 or proof['source']!=r['kernel_source']
                or proof['prepared']['sha256']!=plan['seed_configs'][seed]):raise ValueError('Checkpoint/report mismatch')
        traces={arm:read(root/arm/'trace.json') for arm in ('A','C')}
        for arm,t in traces.items():
            if len(t)!=1920:raise ValueError('Incomplete training trace')
            for i,row in enumerate(t):
                expected=json.loads(json.dumps(asdict(exposure(i,arm))))
                if row['step']!=i+1 or row['exposure']!=expected:raise ValueError('Schedule mismatch')
            if proof['arms'][arm]['checkpoint']!=done['runs'][arm]:raise ValueError('Wrong endpoint')
        if traces['A'][:768]!=traces['C'][:768] or any(a['row_ids_sha256']!=c['row_ids_sha256'] for a,c in zip(traces['A'],traces['C'])):
            raise ValueError('Paired ancestry/data mismatch')
        for pin in [done['warmup'],*done['runs'].values()]:
            if not any(v['source']==pin['path'] and v['sha256']==pin['sha256'] for v in b['node_only_copies'].values()):
                raise ValueError('Endpoint backup absent')
    return dict(status='PASS',members=17,seeds=checks,source_files=362,node_backup_fullstates=20,
                weights_replayed_locally=False,scope='Two-seed reused exploratory pilot; no original-document freshness claim')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('archive','backup','sha256','out','qualification'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(verify(a.archive,a.backup,a.sha256,a.out,a.qualification)))
