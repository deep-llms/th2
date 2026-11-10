"""Additive, CPU-only preservation of all finished v1.9 scoring evidence."""
import argparse
import json
from pathlib import Path
import shutil
from eval.a2_family_f9 import read
from eval.a2_family_f9_v110 import write
from efficient_llm.diagnostic_data import sha256
from scripts.a2_family_f9_backup import backup
from scripts.a2_quant_io import lock


def snapshot(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Snapshot already exists; no duplicate')
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.npy','.csv','.md','.log'))
    if not files:raise ValueError('No scientific evidence found')
    identities={}
    for p in files:
        if p.is_symlink() or 'backup' in p.relative_to(source).parts:raise ValueError('Unexpected symlink/old archive')
        identities[str(p.relative_to(source))]=dict(sha256=sha256(p),bytes=p.stat().st_size)
    output.mkdir(parents=True)
    for name,v in identities.items():
        dest=output/'original'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,dest)
        if sha256(dest)!=v['sha256'] or sha256(source/name)!=v['sha256']:raise ValueError('Snapshot input changed')
    # Index only existing score/replay pairs; all raw scores are still archived,
    # including scores whose replay had not run at the intentional drain.
    pairs={};scores=[];missing_replay=[]
    for label in ('FT11_batch1','FT11_batch2','scores'):
        for p in sorted((output/'original'/label).glob('*/complete.json')):
            done=read(p)
            if not done.get('execution_complete') or 'scores' not in done:continue
            s=p.parent/'scores.json';r=p.parent/'replay.json'
            if not s.is_file() or sha256(s)!=done['scores']['sha256']:raise ValueError('Original score receipt differs')
            key=label+'__'+p.parent.name;scores.append(key)
            if r.is_file():pairs[key]=dict(scores=lock(s),replay=lock(r))
            else:missing_replay.append(key)
    write(output/'scored_index.json',dict(models=pairs,preservation_only=True,
        note='Index of pre-existing score/replay pairs only, not F9 scope or completed-run claim'))
    write(output/'snapshot.json',dict(source=str(source),files=identities,raw_scores=scores,
        missing_original_replay=missing_replay,models_with_scores=sorted({s.split('__',1)[1] for s in scores}),
        scope='Preserve all v1.9 raw scores, both FT11 batches, differences, admission/source receipts; no model run'))
    return backup(output,write)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.prior_root)
    if not read(root/'scope_v110_drain/native_handoff_requested.json')['drained']:raise ValueError('Old scoring not drained')
    sv=read(root/'supervised/supervisor.json')
    if not sv.get('finished_at') or sv.get('handoff_error'):raise ValueError('Old supervisor not finished safely')
    proof=snapshot(root/'results/a2_family_20261107/downstream',a.output)
    print('F9_V19_PRESERVATION_VERIFIED '+json.dumps(read(proof['path'])),flush=True)
