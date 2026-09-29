#1 +60+a
#th2-78gg-FG-completion-check-20260929-0210
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python - <<'CHECK'
import hashlib, json, re, socket, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.train_then_burn import approved_launcher, BURN_HASH, burn_progress
from scripts.verified_gpu_reclaim import process
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-FG-2500-20260928-a01')
print('FG_FINAL_CHECK',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
names=['recipe.json','production/supervisor.json','production/run/run.json','production/run/complete.json','production/run/comparison.json','production/burn-verified.json']
names += [f'production/run/{arm}/{name}' for arm in 'FG' for name in ('result.json','train_config.json','trainer_state.json')]
for name in names:
 p=root/name
 if p.exists():
  data=p.read_bytes()
  print('ARCHIVE_FILE',json.dumps(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),content=data.decode())),flush=True)
 else: print('MISSING',name,flush=True)
p=root/'production/run/arm-G.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-5000));print('G_LOG_TAIL',f.read().decode(errors='replace'),flush=True)
previous=None
for sample in range(2):
 gpus=snapshot(list(range(8)))
 print('GPU_SNAPSHOT',json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),gpus=gpus)),flush=True)
 p=root/'production/burn.log'
 if not p.exists(): break
 text=p.read_text()
 progress=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',text)
 last=tuple(map(float,progress[-1])) if progress else None
 approved={str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
 workers={pid for g in gpus for pid in g['pids']}
 identities=all(approved_launcher(process(pid)['ppid'],approved) for pid in workers)
 fresh=bool(last and previous and last[0]>previous[0] and last[1]>previous[1])
 print('BURN_LIVE',json.dumps(dict(sample=sample,ready_and_progress=burn_progress(text),approved_workers=identities,
          eight_workers=len(workers)==8 and all(len(g['pids'])==1 for g in gpus),latest=last,advanced_since_previous=fresh)),flush=True)
 previous=last
 if sample==0: time.sleep(12)
print('FG_FINAL_CHECK_COMPLETE',datetime.now(timezone.utc).isoformat(),flush=True)
CHECK
