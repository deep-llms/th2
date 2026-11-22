"""Keep all qualification states on-node; archive only small receipts."""
import json,shutil,tarfile
from pathlib import Path
from .data import sha
from .embedding_anchor_device import device_sources


def finish(base):
    base=Path(base);plan=json.loads((base/'execution/plan.json').read_text());files=[];large=[]
    for seed in (0,1):
        root=base/f's{seed}';report=json.loads((root/'complete.json').read_text())
        if (report['fixture_only'] or report['device']!='cuda' or report['source']!=device_sources()
                or report['actual_updates_executed']!=440 or report['production_updates_executed']!=0
                or report['production_fit_admitted'] or not report['lineage_matches_single_segment']
                or set(report['segments'])!=set(plan['segments'])):raise ValueError('Incomplete qualification')
        if sha(root/'numerical.json')!=report['numerical_sha256'] or sha(root/'warmup-lineage.pt')!=report['warmup_lineage_sha256']:raise ValueError('Qualification artifacts changed')
        for v in report['segments'].values():
            if not v['exact_resume'] or v['updates_per_lineage']!=40 or v['actual_updates_executed']!=80:raise ValueError('Segment incomplete')
            for k in ('whole','resumed'):
                if sha(v[k]['path'])!=v[k]['sha256']:raise ValueError('Segment checkpoint changed')
        files.extend([root/'complete.json',root/'numerical.json']);large.extend(root.glob('*.pt'))
        large.extend((Path(plan['prepared_root'])/f's{seed}').glob('*.pt'))
    if len(large)!=36:raise ValueError('Expected32 qualification states+4 CPU states/references')
    backup=Path('/mnt/local/_backup/deep-llms_th2')/base.name;backup.mkdir(parents=True,exist_ok=False)
    copies={}
    for i,src in enumerate(sorted(large)):
        dest=backup/f'{i:02d}-{src.parent.name}-{src.name}';shutil.copyfile(src,dest);h=sha(src)
        if sha(dest)!=h:raise ValueError('Node backup mismatch')
        copies[str(dest)]=dict(source=str(src),sha256=h)
    files.extend([base/'execution/source.json',base/'execution/plan.json'])
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>25_000_000:raise ValueError('Archive exceeds small-results limit')
    proof=dict(archive_sha256=sha(base/'tier1.tar.gz'),archive_bytes=(base/'tier1.tar.gz').stat().st_size,
               files={str(p.relative_to(base)):sha(p) for p in files},node_only_copies=copies,
               production_fit_admitted=False,production_updates=0)
    (base/'backup.json').write_text(json.dumps(proof,indent=2)+'\n')
