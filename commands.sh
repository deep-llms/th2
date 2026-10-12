#1 +60+a
#th2-q359-expanded-paired-health-20261012-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 -u - <<'PY'
import datetime,json,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a02')
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
for rel in ('supervisor.log','preflight.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/supervisor.json'):
 p=root/rel
 if p.exists():print(rel,p.read_text()[-24000:],flush=True)
p=root/'position-sample.json'
if p.exists():
 s=json.loads(p.read_text());print('POSITION_SAMPLE',len(s['rows']),s['sample_sha256'],s.get('selection_sha256'))
for p in sorted((root/'supervised/run').glob('*.log')):
 print('JOB_LOG',p.name,p.read_text()[-2500:],flush=True)
subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_pid} #{pane_dead} #{pane_current_command}'],check=True)
subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],check=True)
PY
