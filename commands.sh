#1 +60+a
#th2-readonly-ranklift-matched-early-ppl-20260906-a01
set -euo pipefail
date -u
hostname
python3 - <<'PY'
from pathlib import Path
import json
base=Path('/mnt/local/_outputs/@PROJECT@')
for arm in ('tiered_ranklift_raw_t4_c512','unified_ranklift_raw_t4_m460'):
    checkpoints=sorted((base/arm).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
    for checkpoint in checkpoints:
        path=checkpoint/'trainer_state.json'
        if not path.is_file(): continue
        try:
            state=json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows=[row for row in state.get('log_history',[]) if 'loss' in row]
        print('EARLY_HISTORY '+json.dumps(dict(arm=arm,checkpoint=checkpoint.name,global_step=state.get('global_step'),rows=rows)))
        break
    else:
        print('NO_READABLE_STATE',arm)
PY
echo 'TH2 READONLY MATCHED EARLY PPL COMPLETE'
