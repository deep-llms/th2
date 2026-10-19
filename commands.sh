#1 +30+a
#th2-6y3o-fa2-failure-detail-20261019-a01
set -euo pipefail
python3 - <<'CHECK'
import datetime, json, re, socket, hashlib, os
from pathlib import Path
print('FA2_DIAG_HOST '+json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname())),flush=True)
def clean(text):
    text=re.sub(r'(?:https?|s3)://[^\s\x22\x27<>]+','[URL_REDACTED]',text)
    text=re.sub(r'\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+','[TOKEN_REDACTED]',text)
    text=re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b','[AWS_KEY_REDACTED]',text)
    text=re.sub(r'(?i)([\w]*(?:token|secret|password|credential|authorization)[\w]*\s*[:=]\s*)[^\n]+',r'\1[REDACTED]',text)
    return text
path=Path('/mnt/local/_env_status/train_fa2.status.json')
raw=path.read_bytes()
print('FA2_INSTALL_STATUS '+json.dumps(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),sanitized_text=clean(raw.decode()))),flush=True)
script=Path('/mnt/local/_conda_install/apply_env.sh')
lines=script.read_text().splitlines()
matched=set()
for i,line in enumerate(lines):
    if re.search(r'\bexit\s+3\b|\.whl|wheel.*(?:missing|found)|missing.*wheel',line,re.I):
        matched.update(range(max(0,i-2),min(len(lines),i+3)))
for i in sorted(matched):
    print('FA2_INSTALLER_SOURCE '+json.dumps(dict(line=i+1,text=clean(lines[i]))),flush=True)
cache=Path('/mnt/local/_wheelcache')
count=0
if cache.is_dir():
    for directory, dirs, files in os.walk(cache,followlinks=False):
        here=Path(directory); depth=len(here.relative_to(cache).parts)
        dirs[:]=[d for d in dirs if not (here/d).is_symlink()] if depth<3 else []
        for name in sorted(files):
            if name.endswith('.whl') and 'flash' in name.lower():
                print('FA2_CACHED_WHEEL '+json.dumps(dict(path=str(here/name),bytes=(here/name).stat().st_size)),flush=True)
                count+=1
                if count>100: raise RuntimeError('Wheel listing limit reached')
print('FA2_DIAG_COMPLETE',flush=True)
CHECK
