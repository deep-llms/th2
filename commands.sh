#1 +60+a
#th2-tjx3-p7-final-results-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import hashlib,io,json,tarfile,time
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH
root=Path('/mnt/local/_outputs/deep-llms_th2/p7-checks-20261006-a01');supervised=root/'supervised';run=supervised/'run'
s=json.loads((supervised/'supervisor.json').read_text())
assert s.get('finished_at') and s['training_status']=='ok' and s['training_returncode']==0
assert s['burn']['collective_progress_verified'] and (run/'complete.json').is_file()
for backend in ('fa4','sdpa'):
 assert json.loads((run/backend/'validation.json').read_text())['status']=='passed'
 assert json.loads((run/backend/'throughput.json').read_text())['status']=='passed'
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
inspection=inspect()
assert not inspection['guard_disabled'] and len(inspection['workers'])==8
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
burn=supervised/'burn.log'
first=[l for l in burn.read_text().splitlines() if 'gpu_burn_progress rank=0' in l]
time.sleep(15)
second=[l for l in burn.read_text().splitlines() if 'gpu_burn_progress rank=0' in l]
assert first and second and first[-1]!=second[-1]
inspection['collective_progress']=[first[-1],second[-1]]
(root/'final-inspection.json').write_text(json.dumps(inspection,indent=2))
paths=list(root.glob('*.json'))+list(supervised.glob('*.json'))+list(run.glob('*.json'))
paths += [root/'launch.log',supervised/'training.log']
paths += list(run.glob('*.log'))
for backend in ('fa4','sdpa'):
 paths+=list((run/backend).glob('*.json'))+list((run/backend/'seed-42').glob('*.json'))
 for arm in ('A','P7','P7-kq','P7-ems','P7-mlp'):
  paths+=list((run/backend/'seed-42'/arm).glob('*.json'))
archive=root/'p7-results.tar.gz';assert not archive.exists()
manifest={}
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(set(paths)):
  if not p.is_file():continue
  assert not p.is_symlink() and p.stat().st_size<50*2**20
  name=str(p.relative_to(root));data=p.read_bytes()
  manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  t.add(p,arcname=name,recursive=False)
 data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('MANIFEST.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
print('FINAL_HANDOFF',json.dumps(s),flush=True)
print('CURRENT_BURN',json.dumps(inspection),flush=True)
print('RESULT_ARCHIVE',str(archive),archive.stat().st_size,sha,'files',len(manifest),flush=True)
for backend in ('fa4','sdpa'):
 print('VALIDATION',backend,(run/backend/'validation.json').read_text(),flush=True)
 print('THROUGHPUT',backend,(run/backend/'throughput.json').read_text(),flush=True)
PY
