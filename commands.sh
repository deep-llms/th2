#1 +30+a
#th2-swt-readonly-machine-identity-and-training-status-20260909-b01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
python3 - <<'PY'
import json, os
from pathlib import Path
for name in ('/proc/sys/kernel/random/boot_id', '/proc/uptime'):
    print(name, Path(name).read_text().strip())
stat = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()
print('PID1_START_TICKS', stat[19])
for name in ('/mnt/local/_outputs/deep-llms_th2/swt', '/mnt/local/_data/deep-llms_th2',
             '/mnt/local/_models/deep-llms_th2', '/mnt/local/conda-py311/envs/swt'):
    path = Path(name)
    print('DIRECTORY', name, 'EXISTS', path.is_dir())
    if path.is_dir(): print('ENTRIES', sorted(p.name for p in path.iterdir()))
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01')
print('RUN_ROOT_EXISTS', root.is_dir())
for arm in ('B0','A128','A256','A512','C','D'):
    path = root/arm
    checkpoints = sorted(int(p.name.split('-')[-1]) for p in path.glob('checkpoint-*')
                         if p.is_dir() and p.name.split('-')[-1].isdigit())
    print('CHECKPOINTS', arm, checkpoints)
    result_path = path/'result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        print('RESULT', arm, json.dumps({k:result.get(k) for k in
              ('success','status','global_step','schedule_steps','train_metrics','eval_metrics')}))
    else: print('RESULT_MISSING', arm)
for name in ('training_complete.json','burn_verified.json'):
    path = root/name
    print('MARKER', name, 'EXISTS', path.is_file())
    if path.is_file(): print(path.read_text())
for name in ('pipeline.log','burn.log'):
    path = root/name
    print('LOG_TAIL', name, 'EXISTS', path.is_file())
    if path.is_file():
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-8000))
            print(handle.read().decode(errors='replace'))
print('SWT_READONLY_MACHINE_AND_TRAINING_CHECK_COMPLETE')
PY
