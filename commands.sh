#1 +20+a
#th2-wc69-a2-why-health-20261104-a03
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
yt1=base/'wc69-a2-why-yt1-20261104-a01';yr=yt1/'results/a2_why_20261104'
for root,names in ((yt1,['source-manifest.json','supervised/supervisor.json','supervised/burn-verified.json']), (yr,['preparation_checks.json','progress.json','runs.jsonl','ledger.json']), (Path('/mnt/local/_data/deep-llms_th2/a2_why_fwe_rebuild_20261104'),['status.json','complete.json'])):
 for name in names:
  p=root/name
  if p.is_file():
   result['files'][str(p)]=[json.loads(x) for x in p.read_text().splitlines()] if name.endswith('.jsonl') else json.loads(p.read_text())
   result['locks'][str(p)]=lock(p)
if (yt1/'supervised/burn.log').exists():result['yt1_burn_progress']=burn_progress((yt1/'supervised/burn.log').read_text())
for p in [yt1/'supervisor.log',yt1/'supervised/burn.log']+sorted((yr/'logs').glob('*.log')):
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));result['logs'][str(p)]=f.read().decode(errors='replace')
for p in sorted((yr/'heals').glob('*/result.json'))+sorted((yr/'heals').glob('*/tests-rank*.json')):
 result['files'][str(p)]=json.loads(p.read_text());result['locks'][str(p)]=lock(p)
if (yt1/'source-manifest.json').is_file():
 source=json.loads((yt1/'source-manifest.json').read_text());result['why_source_unchanged']=all(hashlib.sha256((yt1/'source'/n).read_bytes()).hexdigest()==h for n,h in source['files'].items())
if (data/'corpus_plan.json').is_file():
 plan=json.loads((data/'corpus_plan.json').read_text());result['fwe_raw_sizes']=[]
 for item in plan['corpora']['fineweb_edu']['files']:
  p=Path(item['local']);p=p/Path(item['path']).name if p.is_dir() else p
  result['fwe_raw_sizes'].append(dict(path=str(p),expected=item['bytes'],actual=p.stat().st_size if p.is_file() else None))
for _ in range(3):
 for v in list(records.values()):
  if v['ppid']>1 and str(v['ppid']) not in records:records[str(v['ppid'])]=process(v['ppid'])

rawbase=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/raw/fineweb_edu')
result['fwe_actual_layout']=[]
if rawbase.exists():
 for p in sorted(rawbase.rglob('*')):
  if len(result['fwe_actual_layout'])>=400:break
  if p.is_file():result['fwe_actual_layout'].append(dict(path=str(p),bytes=p.stat().st_size))
result['data_processes']=subprocess.run(['ps','-eo','pid,ppid,etime,pcpu,stat,args'],capture_output=True,text=True).stdout.splitlines()
result['data_processes']=[x for x in result['data_processes'] if 'a2_why_prepare' in x or 'a2_prepare_corpora' in x]
print('WHY_STATUS_JSON',json.dumps(result),flush=True)
PY
