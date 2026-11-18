#1 +10+a
#th2-ohmv-warm-order-guard-inspect-20261118-a01
set -euo pipefail
python3 - <<'PYCODE'
from pathlib import Path
import hashlib,json,subprocess
p=Path('/mnt/local/_gpu_guard')
print('GUARD_FILES',json.dumps([dict(name=f.name,bytes=f.stat().st_size) for f in p.iterdir() if f.is_file()]))
for name in ['gpu_guard.sh','polite_burn.py']:
 f=p/name
 if f.exists():
  b=f.read_bytes();print('GUARD_SOURCE',name,hashlib.sha256(b).hexdigest());print(b.decode())
print('DISABLED',(p/'DISABLED').exists())
print('TMUX',subprocess.run(['tmux','list-sessions','-F','#{session_name}:#{session_windows}'],capture_output=True,text=True).stdout)
print('GUARD_INSPECTION_DONE')
PYCODE
