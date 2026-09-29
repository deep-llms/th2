#1 +60+a
#th2-78gg-capacity-final-20260929-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'CHECK'
import hashlib,json,time,re
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,BURN_HASH,burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/capacity-20260929-a01')
run=root/'production/run'
print('PERFORMANCE_FINAL_CHECK',datetime.now(timezone.utc).isoformat(),flush=True)
print('SOURCE_HASHES',json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
 ['scripts/benchmark_capacity.py','scripts/benchmark_training.py','train.py','deep_kv/model.py','deep_kv/training.py','resources/accelerate_config.yaml']}),flush=True)
report=json.loads((run/'run.json').read_text())
print('QUEUE',json.dumps(report),flush=True)
supervisor=json.loads((root/'production/supervisor.json').read_text())
print('SUPERVISOR',json.dumps(supervisor),flush=True)
assert report['status']=='ok', 'Queue not successfully complete'
assert json.loads((run/'complete.json').read_text())==report
assert supervisor['training_status']=='ok' and supervisor['training_returncode']==0
assert 'burn' in supervisor and 'handoff_error' not in supervisor
assert json.loads((run/'probes/capacity-summary.json').read_text())['status']=='ok'
for job in report['jobs']:
 assert job['status']=='ok' and job['returncode']==0
 for artifact in job.get('artifacts',[]):
  path=run/artifact['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
paths=[root/'preflight.json',root/'jobs.json',root/'gpu_inspection.json',
 root/'production/reclaim.json',root/'production/gpus-free-before-training.json',root/'production/supervisor.json',
 root/'production/burn-verified.json',run/'run.json',run/'complete.json',run/'probes/capacity-summary.json']
summary=json.loads((run/'probes/capacity-summary.json').read_text())
for name,result in summary['results'].items():
 path=run/'probes'/name
 paths.append(run/'probes'/f'{name}.json')
 if result['status']=='ok':
  paths.extend([path/'result.json',path/'train_config.json',*[path/f'benchmark-rank{i}.json' for i in range(8)]])
 else:
  paths.extend(path.glob('capacity-oom-rank*.json'))
for path in paths:
 raw=path.read_bytes()
 print('ARTIFACT_JSON',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
source=str(Path('resources/llm_pretrain_burn.py').resolve())
before=inspect()
assert not before['guard_disabled'] and len(before['workers'])==8
assert all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in before['workers'])
burn=root/'production/burn.log'
a=burn.read_text();assert burn_progress(a)
time.sleep(12)
b=burn.read_text();after=inspect()
assert before['workers']==after['workers']
progress=lambda text: re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',text)[-1]
x,y=progress(a),progress(b)
assert int(y[0])>int(x[0]) and float(y[1])>float(x[1])
print('LIVE_BURN',json.dumps(dict(before=before,after=after,progress_before=x,progress_after=y)),flush=True)
print('PERFORMANCE_FINAL_VERIFIED',flush=True)
CHECK
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
