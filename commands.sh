#1 +30+a
#th2-ohmv-data-inventory-20261118-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYINV'
import json,os,socket,shutil
from pathlib import Path
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
roots=['/mnt/local/_data/deep-llms_th2','/mnt/local/_models/deep-llms_th2','/mnt/local/_backup/deep-llms_th2','/mnt/local/_outputs/deep-llms_th2']
result={'host':socket.gethostname(),'free_bytes':shutil.disk_usage('/mnt/local').free,'cpus':len(os.sched_getaffinity(0)),'roots':{},'raw':{}}
for name in roots:
 p=Path(name);result['roots'][name]={'exists':p.exists(),'children':[{'name':x.name,'dir':x.is_dir(),'symlink':x.is_symlink()} for x in sorted(p.iterdir())][:100] if p.is_dir() else []}
plan=json.loads(Path('resources/a2_corpora_30b_20261029.json').read_text())
for name,limit in [('dclm',216),('fineweb_edu',44)]:
 files=plan['corpora'][name]['files'][:limit];found=[]
 for r in files:
  p=Path(r['local']);p=p/Path(r['path']).name if p.is_dir() else p
  if p.exists():found.append({'index':r['index'],'bytes':p.stat().st_size,'size_matches':p.stat().st_size==r['bytes']})
 result['raw'][name]=found
for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
 p=Path(plan['tokenizer_path'])/name;result[name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
print('A2_DOT_DATA_INVENTORY '+json.dumps(result),flush=True)
PYINV
printf 'DATA_INVENTORY_COMPLETE\n'
