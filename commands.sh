#1 +60+a
#th2-q359-10k-startup-status-20261009-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,subprocess
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01')
for path in (root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',root/'supervised/run/run.json'):
    if path.is_file():print(path.name,path.read_text(),flush=True)
print('SUPERVISOR_TAIL','\n'.join((root/'supervisor.log').read_text(errors='replace').splitlines()[-15:]),flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_TAIL',path.name,'\n'.join(path.read_text(errors='replace').splitlines()[-15:]),flush=True)
print('TMUX',subprocess.run(['tmux','display-message','-p','-t',root.name,'#{pane_dead} #{pane_pid}'],capture_output=True,text=True,check=True).stdout.strip(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
for arm in ('A','P6-iso'):
    path=root/'supervised/run/seed-1042'/arm/'train_config.json'
    if path.is_file():
        c=json.loads(path.read_text())
        print('TRAIN_CONFIG',arm,json.dumps({k:c[k] for k in ('world_size','tokens_per_update','train_fingerprint','eval_fingerprint')}),flush=True)
        print('CHECKPOINTS',arm,[p.name for p in path.parent.glob('checkpoint-*')],flush=True)
PYREMOTE
