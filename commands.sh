#1 +20+a
#th2-wc69-family-f9-v110-recovery-status-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY_NODE'
from pathlib import Path
import json,hashlib
r=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-report-recovery-20261110-a01')
names=['REPORT_FULL_v1.10.md','F9_MODELS.csv','verification.json']
files={n:dict(bytes=(r/n).stat().st_size,sha256=hashlib.sha256((r/n).read_bytes()).hexdigest()) for n in names if (r/n).is_file()}
print('RECOVERY_FILES '+json.dumps(files))
assert sum(v['bytes'] for v in files.values())<1048576,'Compact export unexpectedly large'
if 'verification.json' in files:print('RECOVERY_PROOF '+(r/'verification.json').read_text())
PY_NODE
tail -n 12 /mnt/local/deep-llms_th2/_run_log_/_run-2026-11-10_14-41-10-th2-wc69-family-f9-v110-report-recovery-20261110-a01.log
ps -eo pid,ppid,etime,args | grep '[s]cripts.a2_family_f9_report_recover' || true
