#1 +30+a
#th2-6y3o-fa2-installer-files-20261019-a01
set -euo pipefail
python3 - <<'CHECK'
import os, json, datetime, socket
from pathlib import Path
print('INSTALLER_DISCOVERY_HOST '+json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname())),flush=True)
skip={'envs','pkgs','site-packages','.git','node_modules','_data','_models','_outputs','huggingface','pip','uv','torch','triton','__pycache__'}
count=0
for root in (Path('/mnt/local'),Path('/tmp'),Path('/var/tmp')):
    if not root.is_dir(): continue
    for directory, dirs, files in os.walk(root,followlinks=False):
        here=Path(directory); depth=len(here.relative_to(root).parts)
        dirs[:]=sorted(d for d in dirs if d not in skip and not (here/d).is_symlink()) if depth<3 else []
        if depth==0:
            print('INSTALLER_TOP '+json.dumps(dict(root=str(root),dirs=dirs)),flush=True)
        for name in sorted(files):
            path=here/name
            candidate=('install' in str(path).lower() and path.suffix in ('.log','.txt','.sh','.py')) or 'train_fa2' in name or (here.name in ('_run_log_','_install_log_','_install_logs_','_logs_','logs') and path.suffix=='.log')
            if not candidate or path.is_symlink(): continue
            try:
                stat=path.stat()
                print('INSTALLER_CANDIDATE '+json.dumps(dict(path=str(path),bytes=stat.st_size,mtime_utc=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat())),flush=True)
                count+=1
                if count>=100: raise RuntimeError('Discovery candidate limit reached')
            except FileNotFoundError: pass
print('INSTALLER_DISCOVERY_COMPLETE',flush=True)
CHECK
