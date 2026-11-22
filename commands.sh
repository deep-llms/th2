#1 +10+a
#th2-ohmv-b200-current-status-20261122-a02
set -eu
/usr/bin/python3 - <<'PY'
import json,datetime,subprocess
from pathlib import Path
def query(kind,fields):
    output=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True)
    return [[v.strip() for v in line.split(',')] for line in output.splitlines() if line.strip()]
gpus=query('gpu','index,uuid,name')
workers=[]
for uuid,pid in query('compute-apps','gpu_uuid,pid'):
    root=Path('/proc')/str(int(pid))
    parent=int((root/'stat').read_text().rsplit(')',1)[1].split()[1])
    argv=(Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0')
    workers.append([uuid,int(pid),parent,argv[1:2]==[b'/tmp/llm_pretrain_burn.py']])
print('B200_CURRENT_STATUS',json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpus=gpus,workers=workers)))
PY
