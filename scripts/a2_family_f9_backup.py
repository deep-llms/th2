"""Self-contained F9 scientific archive, verified byte-for-byte before handoff."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
from eval.a2_family_f9 import read,write
from efficient_llm.diagnostic_data import sha256


def backup(root, writer=write):
    root=Path(root);out=root/'backup';out.mkdir(exist_ok=False)
    # Materialize reused scores and preparations so archives never depend on
    # temporary node paths. Existing original artifacts remain untouched.
    index=read(root/'scored_index.json');copies={}
    for model,row in index['models'].items():
        copies[model]={}
        for kind in ('scores','replay'):
            item=row[kind];p=Path(item['path'])
            if sha256(p)!=item['sha256']:raise ValueError('Changed scored artifact')
            if root in p.parents:dest=p
            else:
                dest=root/'reused'/model/(kind+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,dest)
            copies[model][kind]=dict(path=str(dest.relative_to(root)),sha256=item['sha256'])
    writer(root/'portable_index.json',dict(models=copies))
    files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if rel.parts[0]=='backup' or not p.is_file():continue
        if p.is_symlink():raise ValueError('Unexpected result symlink')
        if p.suffix in ('.json','.jsonl','.md','.log','.npy','.csv'):
            files[str(rel)]=dict(sha256=sha256(p),bytes=p.stat().st_size)
    archive=out/'family_f9_scientific.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in files:tar.add(root/name,arcname=name,recursive=False)
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            if not m.isfile() or m.name not in files or m.name in seen:raise ValueError('Archive coverage mismatch')
            seen.add(m.name);h=hashlib.sha256()
            with tar.extractfile(m) as stream:
                for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
            if m.size!=files[m.name]['bytes'] or h.hexdigest()!=files[m.name]['sha256']:raise ValueError('Archive hash mismatch')
    if seen!=set(files):raise ValueError('Incomplete archive')
    parts=[]
    with archive.open('rb') as stream:
        while chunk:=stream.read(24<<20):
            p=out/f'family_f9_scientific.part{len(parts):03d}';p.write_bytes(chunk)
            parts.append(dict(path=str(p),sha256=sha256(p),bytes=len(chunk)))
    proof=writer(out/'verified.json',dict(verified=True,files=files,parts=parts,archive_sha256=sha256(archive),
        members=len(files),scope='F9 model hashes, requests, per-item LL/correctness, reports and logs; excludes model weights'))
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--receipt',required=True);p.add_argument('--spec-sha256');a=p.parse_args()
    writer=write
    if a.spec_sha256:
        from eval.a2_family_f9_v110 import SPEC_SHA,write as writer
        if a.spec_sha256!=SPEC_SHA:raise ValueError('Wrong backup spec')
    proof=backup(a.root,writer);writer(a.receipt,dict(verified=True,proof=proof))
