#1 +30+a
#th2-ccm-inspect-preparation-handoff-20260913-a01
set -euo pipefail
date -u
hostname
/usr/bin/python3 - <<'PY'
from pathlib import Path
import os, json
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)<=1: continue
    try:
        argv=(p/'cmdline').read_bytes().split(b'\0')
        if b'scripts/prepare_pilot_data.sh' not in argv: continue
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        print(json.dumps(dict(pid=int(p.name),parent=int(stat[1]),start_ticks=stat[19],
                             argv=[a.decode(errors='replace') for a in argv if a],
                             stdout=os.readlink(p/'fd/1'),stderr=os.readlink(p/'fd/2'))))
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
PY
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py verify-burn
echo CCM_PREPARATION_HANDOFF_INSPECTION_COMPLETE
