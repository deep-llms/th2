#1 +20+a
#th2-wc69-a2-why-y2-package-20261105-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PACKAGE'
from pathlib import Path
import json,tarfile,hashlib
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-rest-20261105-a01/results/a2_why_20261104')
out=root.parents[1]/'exports/why_y2_20261105_a01'
assert not out.exists();out.mkdir(parents=True)
spec='87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea'
names=[f'Y2-{ds}-{arm}-{variant}-0.2B' for ds in ('dclm','fwe') for variant in ('rdsc','avg') for arm in ('P300','P1T')]
files=[root/'config/plan.json',root/'config/spec.md',root/'Y1_noise.json',root/'Y2_NOTE.md',root/'Y2_readings.json',root/'preparation_checks.json']
for name in names:
    heal=root/'heals'/name;r=json.loads((heal/'result.json').read_text())
    assert r['status']=='complete' and r['spec_sha256']==spec and len(r['lr_updates'])==762
    assert (heal/'order.json').is_file()
    workers=sorted(heal.glob('tests-rank*.json'));assert len(workers)==8
    files.extend(sorted(heal.glob('*.json')))
    arrays=sorted((root/'perseq'/name).glob('*.json'));assert len(arrays)==3;files.extend(arrays)
files=sorted(set(files));entries=[]
for p in files:
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
    entries.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(spec_sha256=spec,created_at=datetime.now(timezone.utc).isoformat(),run_ids=names,files=entries,scope='Eight completed Y2 base runs only; Y1 noise included for registered comparison; retain scientific flags; no weights or training data')
manifest_path=out/'MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=out/'why_y2_20261105_a01.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    tar.add(manifest_path,arcname='MANIFEST.json',recursive=False)
assert archive.stat().st_size<25*1024*1024
receipt=dict(spec_sha256=spec,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),files=len(entries),runs=len(names))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('WHY_Y2_PACKAGE',json.dumps(receipt),flush=True)
PACKAGE
