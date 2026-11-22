"""Verify paired terminal reports; preserve fullstates locally on the node."""
import json
import shutil
import tarfile
from pathlib import Path

from .data import sha
from .embedding_anchor_device import device_sources
from .embedding_anchor_eval import replay


def validate_seed(root, plan, seed):
    root=Path(root);r=json.loads((root/'complete.json').read_text());reg=r['registration']
    if (reg['seed']!=seed or reg['physical_fit_updates']!=3072
            or reg['kernel_source']!=device_sources()
            or reg['prepared_sha256']!=plan['prepared_sha256']
            or reg['plan_sha256']!=plan['pilot_plan_sha256']
            or reg['gate_sha256']!=plan['gates'][str(seed)]['sha256']
            or reg['fit_source_sha256']!=sha(Path(__file__).with_name('embedding_anchor_fit.py'))):
        raise ValueError('Terminal fit registration mismatch')
    if sha(root/'fit-recipe.json')!=r['fit_recipe_sha256']:raise ValueError('Recipe changed')
    if json.loads((root/'fit-recipe.json').read_text())!=reg:raise ValueError('Recipe differs')
    if sha(root/'evaluation/REPORT.json')!=r['report_sha256']:raise ValueError('Report changed')
    report=replay(root/'evaluation');m=json.loads((root/'evaluation/manifest.json').read_text())
    proof=m['checkpoint_proofs']
    if (report['rows']!=512 or proof['step']!=1920 or proof['warmup']!=r['warmup']
            or proof['source']!=device_sources() or proof['seed']!=seed
            or proof['prepared']['sha256']!=plan['seed_configs'][str(seed)]):
        raise ValueError('Terminal evaluation binding mismatch')
    for arm,pin in r['runs'].items():
        if arm not in ('A','C') or proof['arms'][arm]['checkpoint']!=pin or sha(pin['path'])!=pin['sha256']:
            raise ValueError('Terminal state changed')
    if set(r['runs'])!={'A','C'} or sha(r['warmup']['path'])!=r['warmup']['sha256']:
        raise ValueError('Missing paired endpoints')
    traces={arm:json.loads((root/arm/'trace.json').read_text()) for arm in ('A','C')}
    if any(len(t)!=1920 for t in traces.values()) or traces['A'][:768]!=traces['C'][:768]:
        raise ValueError('Common warmup trace mismatch')
    for a,c in zip(traces['A'],traces['C']):
        if a['step']!=c['step'] or a['row_ids_sha256']!=c['row_ids_sha256']:
            raise ValueError('Paired row mismatch')
        for k in ('path','donor','lr','row_start','row_stop'):
            if a['exposure'][k]!=c['exposure'][k]:raise ValueError('Paired schedule mismatch')
    expected={root/'shared/000000.pt',root/'shared/000768.pt'}
    expected.update(root/arm/f'{step:06d}.pt' for arm in ('A','C') for step in (1152,1344,1536,1920))
    if set(root.rglob('*.pt'))!=expected:raise ValueError('Missing or unexpected fullstates')
    return r, sorted(expected)


def finish(base):
    base=Path(base);plan=json.loads((base/'execution/plan.json').read_text());large=[];files=[]
    for seed in (0,1):
        root=base/f's{seed}';r,states=validate_seed(root,plan,seed)
        if r['registration']['fixture_only']:raise ValueError('Actual fit required')
        large.extend(states)
        files.extend(root/n for n in ('complete.json','fit-recipe.json','A/trace.json','C/trace.json',
                                      'evaluation/REPORT.json','evaluation/scores.npz','evaluation/manifest.json'))
    backup=Path('/mnt/local/_backup/deep-llms_th2')/base.name;backup.mkdir(parents=True,exist_ok=False)
    copies={}
    for src in large:
        dest=backup/src.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest);h=sha(src)
        if sha(dest)!=h:raise ValueError('Node backup mismatch')
        copies[str(dest)]=dict(source=str(src),sha256=h)
    files.extend(base/('execution/'+n) for n in ('source.json','plan.json','admission.json'))
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>25_000_000:raise ValueError('Small archive limit exceeded')
    proof=dict(archive_sha256=sha(base/'tier1.tar.gz'),archive_bytes=(base/'tier1.tar.gz').stat().st_size,
               files={str(p.relative_to(base)):sha(p) for p in files},node_only_copies=copies,
               physical_fit_updates=6144,flags=['two_seeds','reused_exploratory','historical_doc_overlap_unknown'])
    (base/'backup.json').write_text(json.dumps(proof,indent=2)+'\n')
