"""Small evidence archive and node-only basis backups for the frozen pilot."""
import json,shutil,tarfile
from pathlib import Path
from .data import sha
from .horizon_train import write_json
from .path_background_study import replay


def finish(base):
    base=Path(base);files=[];reports={};backups={}
    backup=base/'basis-backup'
    if backup.exists():raise ValueError('Existing backup; reconcile')
    for seed in (0,1):
        root=base/'prepared'/f's{seed}';study=root/'study';replay(study)
        plan=json.loads((root/'plan.json').read_text());manifest=json.loads((study/'manifest.json').read_text())
        if (sha(root/'plan.json')!=manifest['plan_sha256'] or manifest['model_pins']!=plan['parent_hashes'] or
                manifest['input_pins']!=plan['expected_inputs'] or sha(study/'basis.pt')!=manifest['basis_sha256']):raise ValueError('Pilot provenance mismatch')
        dest=backup/f's{seed}-basis.pt';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(study/'basis.pt',dest)
        if sha(dest)!=manifest['basis_sha256']:raise ValueError('Basis backup mismatch')
        backups[str(seed)]=dict(path=str(dest),sha256=sha(dest))
        reports[str(seed)]=json.loads((study/'REPORT.json').read_text())
        files.extend([root/'plan.json',*(study/f for f in ('REPORT.json','scores.npz','manifest.json','parity.json','complete.json'))])
    write_json(base/'REPORT.json',dict(seeds=reports,interpretation='Exploratory frozen two-path intervention; no automatic success/causal fraction claim'))
    files.extend([base/'REPORT.json',base/'prepared/prepared.json',base/'execution/source.json',base/'execution/admission.json'])
    inventory={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'small-files.json',inventory);files.append(base/'small-files.json')
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in sorted(files):
            if p.suffix=='.pt' or p.is_symlink():raise ValueError('No weights/basis export')
            tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>5_000_000:raise ValueError('Oversized archive')
    report=dict(archive_sha256=sha(base/'tier1.tar.gz'),archive_bytes=(base/'tier1.tar.gz').stat().st_size,
                basis_backups=backups,node_replay=True,weights_exported=False,basis_exported=False,files=inventory)
    write_json(base/'backup.json',report);return report
