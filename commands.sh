#1 +120+a
#th2-wc69-a2-why-rest-health-20261105-a03
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
for session in ('a2_why_pause_wc69_20261104_a01','a2_why_data_wc69_20261104_a01','a2_why_rest_wc69_20261105_a01'):
 r=subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_dead} #{pane_dead_status} #{pane_pid}'],capture_output=True,text=True)
 result['sessions'][session]=dict(returncode=r.returncode,output=r.stdout)
records=dict(result['inspection']['processes'])
for k,v in list(records.items()):
 if v['ppid']>1:records[str(v['ppid'])]=process(v['ppid'])
result['live_identities']=records
source=json.loads((depth/'source-manifest.json').read_text())
result['depth_source_unchanged']=all(hashlib.sha256((depth/'source'/n).read_bytes()).hexdigest()==h for n,h in source['files'].items())
yt1=base/'wc69-a2-why-rest-20261105-a01';yr=yt1/'results/a2_why_20261104'
for root,names in ((yt1,['source-manifest.json','supervised/supervisor.json','supervised/burn-verified.json']), (yr,['preparation_checks.json','progress.json','runs.jsonl','ledger.json','Y1_noise.json','Y2_readings.json','Y3_readings.json','decision_why.json','coordinator_checks.json']), (Path('/mnt/local/_data/deep-llms_th2/a2_why_fwe_rebuild_20261104'),['status.json','complete.json'])):
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

reuse_locks=[{'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/result.json', 'sha256': '49e924c6e444930870cbb320938864305ef8cd34b0e32651d8ed199124b7df56'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank0.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank1.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank2.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank3.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank4.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank5.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank6.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01/results/a2_why_20261104/heals/YT1-ST1-dclm-P300-S1-0.2B-lr1e-3/tests-rank7.json', 'sha256': '37dde046caee8a5cdbe721d5a3574e61b2da6df2e13b3ff2f50621f9193e4e55'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/result.json', 'sha256': '4a6797adf233e543f591e2dfb34cbb1eb1dbc90fc4d04703751bc525e24f722e'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank0.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank1.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank2.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank3.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank4.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank5.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank6.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2031/tests-rank7.json', 'sha256': 'dcf47d2ff2cd855fd0f9b3c0d65228869e953881b61ed977e1ef812f77a29bc4'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/result.json', 'sha256': '9d18ced7172d5017ea29d561e665f29594b9b815238ccb899f300ff23469f8b7'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank0.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank1.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank2.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank3.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank4.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank5.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank6.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P1T-S1-0.2B-seed2032/tests-rank7.json', 'sha256': '8374b0456ae395079324c687f852904dfd429e9872e1494d2329ac1be5c90f44'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/result.json', 'sha256': '0b3b42f304040d2e5b619fac32eba2983e0090d630397e38648389c9d90a7ffe'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank0.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank1.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank2.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank3.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank4.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank5.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank6.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2031/tests-rank7.json', 'sha256': '41dec6b7844e469bd20318dc3488f609f24f0d313c7d517d9c930e770d51a2a9'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/result.json', 'sha256': '5639ff2f43311c1dd55474f4f2a57e347ab408e9b72b252eba08038bc63ccca0'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank0.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank1.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank2.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank3.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank4.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank5.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank6.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}, {'path': '/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-y1dclm-20261104-a01/results/a2_why_20261104/heals/Y1-dclm-P300-S1-0.2B-seed2032/tests-rank7.json', 'sha256': 'fa763230478127d99eb7b9bf6f6da540b3d83f62111789a3f1fb0930671bcf77'}]
result['reuse_locks_verified']=[]
for entry in reuse_locks:
 p=Path(entry['path']);actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 result['reuse_locks_verified'].append(dict(path=entry['path'],expected=entry['sha256'],actual=actual,passed=actual==entry['sha256']))

for relative in ['fwe/tests.json','fwe/complete.json','fwe/2031/tests.json','fwe/2032/tests.json']:
 p=data/relative
 if p.is_file():result['files'][str(p)]=json.loads(p.read_text());result['locks'][str(p)]=lock(p)
manifest=json.loads((data/'source-manifest.json').read_text())
result['data_source_unchanged']=all(hashlib.sha256((data/'source'/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())

p=base/'wc69-a2-why-rest-20261105-a01.preflight.json'
if p.is_file():result['files'][str(p)]=json.loads(p.read_text());result['locks'][str(p)]=lock(p)
result['run_log_audits']=[]
for p in sorted((yr/'logs').glob('*.log')):
 text=p.read_text(errors='replace');result['run_log_audits'].append(dict(path=str(p),nonfinite_flags=text.count('WHY_NONFINITE_KD_FLAG'),bytes=p.stat().st_size))
result['startup_processes']=[x for x in subprocess.run(['ps','-eo','pid,ppid,etime,pcpu,stat,args'],capture_output=True,text=True).stdout.splitlines() if 'scripts.launch_a2_why' in x or 'scripts.a2_why_run' in x or '/a2_env/bin/python3.11 -u -' in x]
print('WHY_STATUS_JSON',json.dumps(result),flush=True)
PY
