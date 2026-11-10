"""Keep a verified scientific archive on node; never split/export raw data."""
import hashlib
from pathlib import Path
import shutil
import tarfile
from eval.a2_family_f10 import write
from eval.a2_family_f9 import read
from efficient_llm.diagnostic_data import sha256
from scripts.a2_family_f9_run import locked


def backup(root,cfg):
    root=Path(root);out=root/'backup';out.mkdir(exist_ok=False)
    evidence=root/'provenance';evidence.mkdir()
    inputs=dict(config=Path(cfg['source_manifest']['path']).parent/'config.json',
                source_manifest=cfg['source_manifest']['path'],prepared=cfg['prepared']['path'],FT16=cfg['FT16']['path'],
                f10a_tests=Path(cfg['f10a_root'])/'tests.json',f10a_complete=Path(cfg['f10a_root'])/'complete.json',
                input_manifest=cfg['input_manifest']['path'],preparation_ready=cfg['preparation_ready']['path'],cpu_proof=cfg['cpu_proof']['path'])
    ft16=locked(cfg['FT16']);locked(ft16['fewshot_examples']);inputs['fewshot_examples']=ft16['fewshot_examples']['path']
    for key in ('source_manifest','prepared','FT16','input_manifest','f10a','preparation_ready','cpu_proof','f9_manifest'):locked(cfg[key])
    for name,path in inputs.items():shutil.copyfile(path,evidence/(name+'.json'))
    shutil.copyfile(Path(cfg['f10a_root'])/'tests.csv',root/'F10a_TESTS.csv')
    source=Path(cfg['source_manifest']['path']).parent/'source'
    for name,h in read(cfg['source_manifest']['path'])['files'].items():
        if sha256(source/name)!=h:raise ValueError('Source changed before backup')
        target=evidence/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    # Prior F9 raw data already has its verified native archive. Bind that
    # archive explicitly instead of copying 863 MB into every later package.
    f9=read(cfg['f9_manifest']['path']);prior_archive=Path(cfg['f9_manifest']['path']).parent/'family_f9_scientific.tar.gz'
    if sha256(prior_archive)!=f9['archive_sha256']:raise ValueError('Prior F9 archive changed')
    write(evidence/'F9_archive_reference.json',dict(path=str(prior_archive),sha256=f9['archive_sha256'],scope='immutable saved F9 likelihoods used by F10b'))
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(root.rglob('*')) if p.is_file() and out not in p.parents}
    archive=out/'family_f10_scientific.tar.gz'
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
        raw_stays_on_node=True,export_only=['REPORT_FULL_v1.11.md','F10_MODELS.csv','F10a_TESTS.csv','verification.json']))
