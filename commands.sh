#1 +20+a
#th2-wc69-a2-why-health-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
from pathlib import Path
import json,hashlib,subprocess
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import burn_progress
from scripts.a2_quant_io import lock
from datetime import datetime,timezone
base=Path('/mnt/local/_outputs/deep-llms_th2');depth=base/'wc69-a2-depth-partc-20261103-a01';pause=base/'wc69-a2-why-pause-20261104-a01';data=base/'wc69-a2-why-data-20261104-a01'
result=dict(utc=datetime.now(timezone.utc).isoformat(),inspection=inspect(),files={},locks={},logs={},sessions={})
for root,names in ((pause,['pause/request.json','pause/saved_state.json','pause/signal.json','pause/terminal.json']), (depth,['supervised/supervisor.json','supervised/burn-verified.json','source-manifest.json']), (data,['source-manifest.json','restore.json','tests.json','dclm/tests.json','dclm/complete.json','dclm_ready.json','complete.json','failure.json','historical_fwe_identity.json'])):
 for name in names:
  p=root/name
  if p.is_file():result['files'][str(p)]=json.loads(p.read_text());result['locks'][str(p)]=lock(p)
for root,name in ((pause,'pause.log'),(depth,'supervisor.log'),(depth,'supervised/burn.log'),(data,'prepare.log')):
 p=root/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));result['logs'][str(p)]=f.read().decode(errors='replace')
if (depth/'supervised/burn.log').exists():result['burn_progress']=burn_progress((depth/'supervised/burn.log').read_text())
for session in ('a2_why_pause_wc69_20261104_a01','a2_why_data_wc69_20261104_a01'):
 r=subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_dead} #{pane_dead_status} #{pane_pid}'],capture_output=True,text=True)
 result['sessions'][session]=dict(returncode=r.returncode,output=r.stdout)
records=dict(result['inspection']['processes'])
for k,v in list(records.items()):
 if v['ppid']>1:records[str(v['ppid'])]=process(v['ppid'])
result['live_identities']=records
source=json.loads((depth/'source-manifest.json').read_text())
result['depth_source_unchanged']=all(hashlib.sha256((depth/'source'/n).read_bytes()).hexdigest()==h for n,h in source['files'].items())
print('WHY_STATUS_JSON',json.dumps(result),flush=True)
PY
