#1 +30+a
#th2-6y3o-dropbox-probe-20261031-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'CHECK'
from datetime import datetime, timezone
import json, socket, subprocess
print('DROPBOX_PROBE', json.dumps({'marker':'6Y3O-QUANT-DROPBOX-20261031-A01','utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}),flush=True)
print('DROPBOX_PROBE_COMPLETE',flush=True)
CHECK
