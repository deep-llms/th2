#1 +15+a
#th2-q359-followup-review-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from datetime import datetime, timezone
from pathlib import Path
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ('q359-proxy-12layer-screen22-20261013-a01','q359-proxy-followup11-20261014-a01'):
    root=Path('/mnt/local/_outputs/deep-llms_th2')/name
    print('ROOT',name,flush=True)
    for relative in ('supervised/supervisor.json','supervised/run/run.json'):
        p=root/relative
        if not p.exists():
            print('MISSING',relative);continue
        x=json.loads(p.read_text())
        if 'jobs' not in x: print(relative,json.dumps(x),flush=True);continue
        print('STATUS',x.get('status'),flush=True)
        print('COUNTS',{s:sum(j['status']==s for j in x['jobs']) for s in set(j['status'] for j in x['jobs'])},flush=True)
        for j in x['jobs']:
            if j['status'] not in ('ok','pending','queued'):
                print('JOB',json.dumps(j),flush=True)
                p=root/'supervised/run'/(j['name']+'.log')
                if p.exists():
                    with p.open('rb') as f:
                        f.seek(max(0,p.stat().st_size-16000));lines=f.read().decode(errors='replace').splitlines()
                    print('LOG','\n'.join(lines[-5:]),flush=True)
    p=root/'supervisor.log'
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-12000));lines=f.read().decode(errors='replace').splitlines()
        print('SUPERVISOR_TAIL','\n'.join(lines[-8:]),flush=True)
PYREMOTE
