#1 +60+a
#th2-q359-expanded-paired-health-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 -u - <<'PY'
import datetime,json,subprocess,time
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a01')
data=Path('/mnt/local/_data/deep-llms_th2/expanded-20261012')
manifest=json.loads(Path('resources/expanded_inputs_20261012.json').read_text());files={f['path']:f['bytes'] for r in manifest['repositories'] for f in r['files']}
for i in range(30):
    ready=sum((data/p).is_file() and (data/p).stat().st_size==size for p,size in files.items())
    print('STATUS',datetime.datetime.now(datetime.timezone.utc).isoformat(),'INPUT_FILES',ready,'/',len(files),flush=True)
    if (root/'supervisor.log').exists():print('\n'.join((root/'supervisor.log').read_text().splitlines()[-12:]),flush=True)
    for rel in ('preflight.json','supervised/run/run.json','supervised/supervisor.json'):
        p=root/rel
        if p.exists():print(rel,p.read_text()[-14000:],flush=True)
    dead=subprocess.check_output(['tmux','display-message','-p','-t',root.name,'#{pane_dead}'],text=True).strip()
    if dead=='1' or (root/'supervised/run/run.json').exists():break
    time.sleep(20)
subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],check=True)
PY
