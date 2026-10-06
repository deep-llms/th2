#1 +60+a
#th2-tjx3-p7-review-preflight-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import hashlib, importlib.metadata as md, io, json, tarfile
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, burn_progress
project=Path.cwd()
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-fa4-four-head-2500-20261006-a01')
baseline=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A')
destination=Path('/mnt/local/_outputs/deep-llms_th2/p7-review-preflight-20261006-a01')
destination.mkdir(exist_ok=False)
inspection=inspect()
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
inspection['only_approved_burn_workers']=all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
print('INSPECTION',json.dumps(inspection),flush=True)
(destination/'inspection.json').write_text(json.dumps(inspection,indent=2))
print('VERSIONS',json.dumps({name:md.version(name) for name in ['torch','transformers','accelerate','flash-attn-4','datasets','nvidia-cutlass-dsl']}),flush=True)
files={'inspection.json':destination/'inspection.json'}
for name in ['preflight.json','recipe.json','supervised/supervisor.json','supervised/run/complete.json','supervised/run/run.json','supervised/burn-verified.json']:
    path=root/name
    if path.is_file():
        files['p4/'+name]=path
        print('STATUS',name,path.read_text()[-12000:],flush=True)
for path in (root/'supervised/run').glob('*.json'):files['p4/supervised/run/'+path.name]=path
training=root/'supervised/run/training'
for path in training.glob('*.json'):files['p4/training/'+path.name]=path
for path in (training/'seed-42').glob('*.json'):files['p4/training/seed-42/'+path.name]=path
for arm,folder in [('A',baseline),('P4-iso-4h',training/'seed-42/P4-iso-4h'),('P4-4h',training/'seed-42/P4-4h')]:
    for path in folder.glob('*.json'):files[arm+'/'+path.name]=path
    if (folder/'result.json').is_file():
        result=json.loads((folder/'result.json').read_text())
        print('RESULT',arm,json.dumps({k:result.get(k) for k in ['status','global_step','schedule_steps','training_cost','evaluation']}),flush=True)
    else:print('NOT_COMPLETE',arm,flush=True)
    print('CHECKPOINTS',arm,[p.name for p in sorted(folder.glob('checkpoint-*'))],flush=True)
burn=root/'supervised/burn.log'
if burn.is_file():print('BURN_COLLECTIVE_PROGRESS',burn_progress(burn.read_text()),burn.read_text()[-4000:],flush=True)
manifest={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()}
assert all(item['bytes']<20*2**20 for item in manifest.values())
archive=destination/'p4-results.tar.gz'
with tarfile.open(archive,'w:gz') as output:
    for name,path in files.items():output.add(path,arcname=name,recursive=False)
    data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('MANIFEST.json');info.size=len(data)
    output.addfile(info,io.BytesIO(data))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
print('RESULT_ARCHIVE',str(archive),archive.stat().st_size,digest,flush=True)
(destination/'p4-results.sha256').write_text(digest+'  p4-results.tar.gz\n')
PY
