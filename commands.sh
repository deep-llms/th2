#1 +30+a
#th2-6y3o-stock-wheel-download-inspect-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'CHECK'
from pathlib import Path
for root in (Path('/mnt/local/_data/deep-llms_th2/fa2_stock_20261020'),Path('/mnt/local/_download_log_'),Path('/mnt/local/_download_status'),Path('/mnt/local/deep-llms_th2/_download_log_')):
    print('DOWNLOAD_PATH',str(root),root.exists(),flush=True)
    if root.is_dir():
        for p in sorted(root.rglob('*')):
            if p.is_file():
                print(str(p),p.stat().st_size,flush=True)
                if p.suffix in ('.log','.json') and p.stat().st_size<100000:print(p.read_text(errors='replace')[-12000:],flush=True)
for p in sorted(Path('/mnt/local').glob('*download*')):print('DOWNLOAD_ROOT',str(p),flush=True)
CHECK
