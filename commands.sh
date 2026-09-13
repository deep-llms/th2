#1 +30+a
#th2-ccm-inspect-preparation-log-path-20260913-a01
set -euo pipefail
date -u
hostname
/usr/bin/python3 - <<'PY'
from pathlib import Path
import os, json
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)<=1: continue
    try:
        targets=[]
        for fd in (p/'fd').iterdir():
            try: targets.append(os.readlink(fd))
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
        if 'pipe:[2444481181]' in targets:
            print(json.dumps(dict(pid=int(p.name),files=[x for x in targets if x.endswith('.log') or x == 'pipe:[2444481181]'])))
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
PY
find /mnt/local -maxdepth 4 -type f -name '*prepare-full-pilot*.log' -print
echo CCM_PREPARATION_LOG_PATH_INSPECTION_COMPLETE
