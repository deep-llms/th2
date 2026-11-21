"""Verify completed paired studies; back up node state and package small evidence."""
import json,shutil,tarfile
from pathlib import Path
from .adaptation_study import replay
from .adaptation_train import check_runtime
from .data import sha
from .horizon_train import write_json

NAMES=('A-s0','B-s0','A-s1','B-s1')


def small_study_files(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    names={'manifest.json','config.json','REPORT.json'}
    for step in manifest['snapshots']:
        names.update(step+'/loss/'+n for n in ('manifest.json','REPORT.json','scores.npz'))
        for comparison in ('immediate','ordinary_drift','swapped_drift','remaining'):
            names.update(step+'/geometry/'+comparison+'/'+n for n in ('manifest.json','REPORT.json','row_ids.npy'))
    return [directory/n for n in sorted(names)]


def finish(base,backup):
    base=Path(base);backup=Path(backup)
    if backup.exists() or (base/'backup.json').exists():raise ValueError('Fresh backup destination required')
    reports={};files=[base/'execution/source.json',base/'execution/admission.json']
    for seed in (0,1):
        root=base/'study'/f's{seed}';replay(root)
        reports[str(seed)]=json.loads((root/'REPORT.json').read_text());files.extend(small_study_files(root))
    copies=[]
    for name in NAMES:
        root=base/'runs'/name;c=json.loads((root/'recipe.json').read_text())['config'];check_runtime(c,'cuda',None,False)
        if c['branch']!=name[0] or c['row_reservation']['seed']!=int(name[-1]):raise ValueError('Branch/seed identity mismatch')
        done=json.loads((root/'complete.json').read_text())
        if done['step']!=2000 or sha(root/'fullstate.pt')!=done['fullstate_sha256']:raise ValueError('Incomplete/changed continuation')
        for step in c['snapshots']:
            p=root/'snapshots'/f'{step:06d}'/'fullstate.pt';cp=json.loads(p.with_name('checkpoint.json').read_text())
            if cp['step']!=step or sha(p)!=cp['sha256']:raise ValueError('Snapshot mismatch')
            copies.append(p);files.append(p.with_name('checkpoint.json'))
        copies.append(root/'fullstate.pt')
        files.extend(root/n for n in ('recipe.json','complete.json','checkpoint.json','trace.jsonl'))
    write_json(base/'REPORT.json',dict(seeds=reports,
        flags=['dclm_only','two_seeds','paired_rows_adam_rng_lr','parent_not_proven_converged',
               'historical_overlap_unknown','no_document_freshness_claim','geometry_not_causal_attribution'],
        interpretation='Read per-seed recovery relative to normal continuation; component and weight interventions are conditional effects, not causal percentages.'))
    files.append(base/'REPORT.json')
    for p in files:
        if not p.is_file() or p.is_symlink():raise ValueError('Missing/nonregular small artifact')
        if p.suffix=='.pt' or p.name.startswith('moments_'):raise ValueError('Large artifact cannot be exported')
    files=sorted(set(files));inventory={str(p.relative_to(base)):sha(p) for p in files}
    write_json(base/'small-files.json',inventory);files.append(base/'small-files.json')
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>25_000_000:raise ValueError('Small archive exceeded bound; do not export')
    backup.mkdir(parents=True);large={}
    for p in copies:
        dest=backup/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        h=sha(p)
        if sha(dest)!=h:raise ValueError('Backup state mismatch')
        large[str(dest)]=h
    shutil.copyfile(base/'tier1.tar.gz',backup/'tier1.tar.gz')
    digest=sha(base/'tier1.tar.gz')
    if sha(backup/'tier1.tar.gz')!=digest:raise ValueError('Archive backup mismatch')
    proof=dict(node_fullstate_copies=large,archive_sha256=digest,archive_bytes=(base/'tier1.tar.gz').stat().st_size,
               files=inventory,weights_exported=False,raw_moments_exported=False,
               node_replay=True,local_replay_pending=True)
    write_json(base/'backup.json',proof)
    return proof
