"""Keep a verified scientific archive on node; never split/export raw data."""
import hashlib
import csv
from pathlib import Path
import shutil
import tarfile
from eval.a2_family_f11 import write,SPEC_SHA
from eval.a2_family_f9 import read
from efficient_llm.diagnostic_data import sha256
from scripts.a2_family_f9_run import locked


def backup(root,cfg):
    root=Path(root);out=root/'backup';out.mkdir(exist_ok=False)
    evidence=root/'provenance';evidence.mkdir()
    for key in ('source_manifest','cpu_proof','f10_config_lock','preparation_proof','data_manifest','assets'):
        item=cfg[key];locked(item);shutil.copyfile(item['path'],evidence/(key+'.json'))
    shutil.copyfile(Path(cfg['output_root'])/'config.json',evidence/'config.json')
    source=Path(cfg['source_manifest']['path']).parent/'source'
    for name,h in read(cfg['source_manifest']['path'])['files'].items():
        if sha256(source/name)!=h:raise ValueError('Source changed')
        target=evidence/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    prior=Path(cfg['f10']['output'])/'backup/verified.json';old=read(prior)
    archive=prior.parent/'family_f10_scientific.tar.gz'
    if not old['verified'] or sha256(archive)!=old['archive_sha256']:raise ValueError('F10 archive changed')
    write(evidence/'F10_archive_reference.json',dict(path=str(archive),sha256=old['archive_sha256'],raw_stays_on_node=True))
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(root.rglob('*')) if p.is_file() and out not in p.parents and not ('heals' in p.relative_to(root).parts and ('final' in p.relative_to(root).parts or 'trainer' in p.relative_to(root).parts))}
    archive=out/'family_f11_scientific.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in files:tar.add(root/name,arcname=name,recursive=False)
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in files or member.name in seen:raise ValueError('Archive coverage mismatch')
            seen.add(member.name);h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
            if member.size!=files[member.name]['bytes'] or h.hexdigest()!=files[member.name]['sha256']:raise ValueError('Archive byte mismatch')
    if seen!=set(files):raise ValueError('Incomplete archive')
    return write(out/'verified.json',dict(verified=True,files=files,members=len(files),archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size,
        raw_stays_on_node=True,export_only=['REPORT_FULL_v1.13.md','F11_MODELS.csv','verification.json']))
