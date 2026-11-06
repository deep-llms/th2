"""Verify a self-contained scientific result archive; never copy model weights."""
import hashlib
import tarfile
from pathlib import Path
from efficient_llm.a2_family.common import write,sha


def backup(root):
    root=Path(root);output=root/'backup';output.mkdir(exist_ok=False)
    files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if rel.parts[0]=='backup' or (len(rel.parts)>2 and rel.parts[0]=='heals' and rel.parts[2] in ('trainer','final')) or not p.is_file():continue
        if p.is_symlink():raise ValueError('Unexpected symlink in scientific outputs')
        if p.suffix in ('.json','.jsonl','.md','.log','.yaml','.csv','.npy'):
            files[str(rel)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    if 'REPORT.md' not in files or 'decision_family.json' not in files:raise ValueError('Missing final scientific report')
    manifest=write(output/'manifest.json',dict(files=files,scope='Family scientific arrays, raw rank traces, configs, logs and report; native terminal/Depth results separate'))
    archive=output/'family_scientific.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for rel in files:tar.add(root/rel,arcname=rel,recursive=False)
    observed=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in files or member.name in observed:raise ValueError('Unexpected archive member')
            observed.add(member.name);h=hashlib.sha256();f=tar.extractfile(member)
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            if h.hexdigest()!=files[member.name]['sha256'] or member.size!=files[member.name]['bytes']:raise ValueError('Archive verification failed')
    if observed!=set(files):raise ValueError('Archive coverage differs')
    parts=[]
    with archive.open('rb') as f:
        index=0
        while chunk:=f.read(24<<20):
            p=output/f'family_scientific.part{index:03d}';p.write_bytes(chunk);parts.append(dict(path=str(p),sha256=sha(p),bytes=len(chunk)));index+=1
    return write(output/'verified.json',dict(verified=True,files=[dict(path=str(archive),sha256=sha(archive)),manifest,*parts],members=len(files),parts=parts,raw_rank_traces=sum('sample-order-rank' in n for n in files)))
