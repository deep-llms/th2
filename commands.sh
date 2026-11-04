#1 +20+a
#th2-wc69-a2-why-fwe-empty-reset-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYREMOTE'
import os,json,socket
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-wc69-worker-0'
raw=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu')
assert not raw.is_symlink() and raw.resolve()==raw and raw.is_dir()
expected={raw/f'{i:05d}.parquet' for i in range(44)}
assert set(raw.iterdir())==expected
for p in expected:assert p.is_dir() and not p.is_symlink() and not list(p.iterdir()),str(p)
open_fds=[];downloads=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');words=[Path(a).name for a in args[:3]]
  if any(x in words for x in ['hf','huggingface-cli','wget','curl','aria2c','rsync','scp','rclone','aws','s5cmd']):downloads.append(int(p.name))
  for fd in (p/'fd').iterdir():
   try:target=os.readlink(fd)
   except FileNotFoundError:continue
   if target==str(raw) or target.startswith(str(raw)+'/'):open_fds.append(dict(pid=int(p.name),path=target))
 except (FileNotFoundError,ProcessLookupError):continue
assert not downloads,downloads
assert not open_fds,open_fds
# rmdir refuses any unexpected new content; never recursively remove files.
for p in sorted(expected):p.rmdir()
raw.rmdir();raw.mkdir()
assert not list(raw.iterdir())
print('FWE_EMPTY_RESET_JSON',json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),path=str(raw),removed_empty_directories=45,deleted_files=0,recreated_empty_root=True,signals_sent=0,authority='user authorized stop old downloader, delete raw download folder and retry; controller none, no downloader found')),flush=True)
PYREMOTE
