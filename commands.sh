#1 +20+a
#th2-wc69-a2-depth-x72-results-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,hashlib,gzip,base64
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01')
base=root/'results/a2_depth_20261031';name='ST1-dclm-X72-S1-12.8B-lr1e-3';heal=base/'heals'/name
result=json.loads((heal/'result.json').read_text())
assert result['status']=='complete' and len(result['lr_updates'])==48828
assert hashlib.sha256((heal/'result.json').read_bytes()).hexdigest()=='5b3f8895d6fb5d5751856634870df352f474d92f52ca14525453d061149a53dc'
files=[heal/'result.json',heal/'recipe.json',*[heal/f'tests-rank{i}.json' for i in range(8)],*[base/'perseq'/name/(label+'.json') for label in ('parent','pre-S1','post-S1')],base/'notes/X72.json',base/'REPORT_PART_C.md',base/'decision_depth_partc.json',base/'tests_partc.json',base/'runs_partc.jsonl',base/'logs/C-X72-attempt001.log',root/'source-manifest.json',base/'config/plan.json',base/'DT6.json',base/'DT6_checks.json']
print('RESULTS_AT',datetime.now(timezone.utc).isoformat(),flush=True)
manifest=[]
for p in files:
 assert p.is_file() and not p.is_symlink(),str(p)
 raw=p.read_bytes();assert len(raw)<12*1024*1024
 row=dict(path=str(p.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 packed=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
 print('RESULT_FILE',json.dumps(dict(row,gzip_b64=packed)),flush=True);manifest.append(row)
print('RESULTS_MANIFEST',json.dumps(dict(spec_sha256='2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe',files=manifest)),flush=True)
print('RESULTS_COMPLETE',flush=True)
PY
