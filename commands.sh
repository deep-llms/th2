#1 +60+a
#th2-ohmv-dot-synth-s07-status-20261124-a01
set -euo pipefail
date -u
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s07')
for n in ('status.json', 'failure.json', 'terminal.json', 'export/manifest.json'):
    p = base/n
    t = p.read_text() if p.is_file() else 'absent'
    print('SYNTH', n, t if n != 'export/manifest.json' or t == 'absent' else {k: v for k, v in json.loads(t).items() if k != 'members'}, flush=True)
compact = {}
for s in sorted(base.glob('s*/*/*/summary.json')):
    v = json.loads(s.read_text())
    compact[f"s{v['seed']}/{v['arm']}/{v['task']}"] = dict(params=v['params'], apps=v['applications'], secs=round(v['train_seconds']),
        res={st: {k: [round(r['exact'], 3), round(r['token'], 3), round(r.get('answer_depth', -1), 2)] for k, r in res.items()}
             for st, res in v['results'].items() if st != 'adaptive'},
        adaptive={tau: {k: [round(r['exact'], 3), round(r['depth'], 2), round(r['oracle'], 3)] for k, r in res.items()}
                  for tau, res in v['results'].get('adaptive', {}).items()},
        log=[(l['step'], round(l['loss'], 3)) for l in v['log']][::3])
print('SYNTH_RESULTS', json.dumps(compact, separators=(',', ':')), flush=True)
for p in sorted(base.glob('s*/*/*/progress.json')):
    if not (p.parent/'summary.json').is_file():
        print('SYNTH_PROGRESS', p.parent.relative_to(base), p.read_text(), flush=True)
for log in sorted(base.glob('train-*.log')):
    t = log.read_text()
    if 'Error' in t or 'Traceback' in t:
        print('SYNTH_ERR', log.name, t[-1500:], flush=True)
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
