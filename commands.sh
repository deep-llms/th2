#1 +60+a
#th2-q359-position-extension-inspect-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 -u - <<'PY'
import datetime,json,subprocess
from pathlib import Path
r=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a01');data=Path('/mnt/local/_data/deep-llms_th2/expanded-20261012')
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ('supervisor.log','preflight.json','supervised/supervisor.json','supervised/run/run.json'):
 p=r/name
 if p.exists():print(name,p.read_text()[-20000:])
m=json.loads(Path('resources/expanded_inputs_20261012.json').read_text());fs={f['path']:f for repo in m['repositories'] for f in repo['files']}
print('MISSING',json.dumps([x for x,f in fs.items() if not (data/x).is_file() or (data/x).stat().st_size!=f['bytes']]))
subprocess.run(['tmux','list-panes','-t',r.name,'-F','#{pane_pid} #{pane_dead} #{pane_current_command}'],check=True)
subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],check=True)
PY
