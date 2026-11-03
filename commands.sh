#1 +30+a
#th2-wc69-a2-data-rebuild-health-20261103-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,socket
from scripts.a2_prepare_corpora import sha
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-data-rebuild-20261103-a02')
print('SNAPSHOT',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
for name in ('source-manifest.json','restore.json','corpus_identity.json','failure.json','complete.json'):
 p=root/name
 if p.exists():print('ARTIFACT',name,p.read_text(),flush=True)
source=json.loads((root/'source-manifest.json').read_text())
print('FROZEN_SOURCE_UNCHANGED',all(sha(root/'source'/n)==h for n,h in source['files'].items()),flush=True)
plan=json.loads(Path('resources/a2_corpora_30b_20261029.json').read_text())
ready=[];partial=[]
for rec in plan['corpora']['dclm']['files'][:216]:
 p=Path(rec['local']);p=p/Path(rec['path']).name if p.is_dir() else p
 if p.is_file():
  (ready if p.stat().st_size==rec['bytes'] else partial).append(rec['index'])
print('RAW_SIZE_READY',json.dumps(dict(indices=ready,partial_indices=partial,needed=216)),flush=True)
for i in ready[:4]:
 rec=plan['corpora']['dclm']['files'][i];p=Path(rec['local']);p=p/Path(rec['path']).name if p.is_dir() else p
 print('RAW_HASH_CHECK',i,sha(p)==rec['sha256'],flush=True)
data=Path('/mnt/local/_data/deep-llms_th2/a2_dclm_rebuild_20261103')
for name in ('status.json','failure.json','complete.json'):
 p=data/name
 if p.exists():print('DATA_ARTIFACT',name,p.read_text(),flush=True)
p=root/'prepare.log'
if p.exists():print('LOG_TAIL',p.read_text()[-7000:],flush=True)
print('TMUX',subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout,flush=True)
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'scripts.a2_rebuild_wc69_data --work' in cmd or 'scripts.a2_prepare_corpora --plan' in cmd:
   print('CPU_PROCESS',p.parent.name,cmd,flush=True)
 except (OSError,ProcessLookupError):pass
for query in [('gpu','index,uuid,utilization.gpu,memory.used'),('compute-apps','pid,gpu_uuid,process_name')]:
 r=subprocess.run(['nvidia-smi','--query-'+query[0]+'='+query[1],'--format=csv,noheader,nounits'],capture_output=True,text=True)
 print('GPU_READONLY',query[0],r.returncode,r.stdout,flush=True)
 if query[0]=='compute-apps' and r.returncode==0:
  seen=set()
  for row in r.stdout.splitlines():
   pid=int(row.split(',')[0])
   for _ in range(8):
    if pid in seen or pid<1:break
    seen.add(pid);base=Path('/proc')/str(pid)
    try:
     stat=(base/'stat').read_text().rsplit(')',1)[1].split();parent=int(stat[1])
     cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
     print('OWNERSHIP',pid,parent,stat[19],cmd,flush=True);pid=parent
    except OSError:break
PY
