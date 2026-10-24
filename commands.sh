#1 +30+a
#th2-6y3o-dot-stop-v2-preflight-20261024-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,json,shutil
from pathlib import Path
from importlib.metadata import version
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
previous=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-adapt-20261023-a01')
parent=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-long-pair-20261023-a01')
burn_gate(previous)
read=lambda p:json.loads(p.read_text())
proof={'live':inspect(),'old_study_status':read(previous/'supervised/run/evaluation/summary.json')['status'],
 'versions':{k:version(k) for k in ['torch','transformers','accelerate','datasets','flash-attn-4']},
 'disk_free_bytes':shutil.disk_usage(parent).free,'parents':{}}
expected={'D':'a0b189ba8257686d148714bcf4b2b4c2c8514440a12f7e2518e357cbfef51330',
 'Partial-25-cosine':'d30977a61cd4b549e16dd24616d3cdd96d33b2cfd766cba4dd713dca2d3d9840'}
for arm,h in expected.items():
 cp=parent/'prepared/runs'/(arm+'-s0')/'checkpoint-10000'
 actual=hashlib.file_digest((cp/'model.safetensors').open('rb'),'sha256').hexdigest()
 assert actual==h and read(cp/'trainer_state.json')['global_step']==10000
 proof['parents'][arm]={'sha256':actual,'step':10000}
proof['source_matches']=all(hashlib.sha256((previous/'source'/k).read_bytes()).hexdigest()==v
 for k,v in read(previous/'source/SOURCE_MANIFEST.json')['files'].items())
assert proof['source_matches'] and proof['disk_free_bytes']>250*1024**3
print('STOP_V2_PREFLIGHT '+json.dumps(proof),flush=True)
PY
