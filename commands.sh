#1 +30+a
#th2-wc69-a2-data-rebuild-health-20261103-a08
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
for name in ('source-manifest.json','corpus_identity.json','failure.json','complete.json'):
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
for i in ready[-4:]:
 rec=plan['corpora']['dclm']['files'][i];p=Path(rec['local']);p=p/Path(rec['path']).name if p.is_dir() else p
 print('RAW_HASH_CHECK',i,sha(p)==rec['sha256'],flush=True)
data=Path('/mnt/local/_data/deep-llms_th2/a2_dclm_rebuild_20261103')
for name in ('status.json','failure.json','complete.json'):
 p=data/name
 if p.exists():print('DATA_ARTIFACT',name,p.read_text(),flush=True)
parts=[]
for receipt in sorted((data/'parts/dclm').glob('*/complete.json')):
 value=json.loads(receipt.read_text());i=value['source']['index']
 if value['source']!=plan['corpora']['dclm']['files'][i]:raise ValueError('Shard source identity differs')
 parts.append(dict(index=i,streams=value['streams'],seconds=value['seconds'],receipt_sha256=sha(receipt)))
print('COMPLETED_SHARDS',json.dumps(parts),flush=True)
restore=root/'restore.json'
if restore.exists():
 value=json.loads(restore.read_text());print('RESTORE_SUMMARY',json.dumps(dict(status=value['status'],files=len(value['files']),receipt_sha256=sha(restore))),flush=True)
shuffle=Path('/mnt/local/_data/deep-llms_th2/a2_dclm_shuffle2030_rebuild_20261103')
for name in ('status.json','complete.json'):
 p=shuffle/name
 if p.exists():print('SHUFFLE_ARTIFACT',name,p.read_text(),flush=True)
print('SHUFFLE_FILE_SIZES',json.dumps({p.name:p.stat().st_size for p in shuffle.glob('*') if p.is_file()}),flush=True)
p=root/'prepare.log'
if p.exists():print('LOG_TAIL',p.read_text()[-7000:],flush=True)
print('TMUX',subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout,flush=True)
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'scripts.a2_rebuild_wc69_data --work' in cmd or 'scripts.a2_prepare_corpora --plan' in cmd or 'scripts.a2_stage1_data --plan' in cmd:
   print('CPU_PROCESS',p.parent.name,cmd,flush=True)
 except (OSError,ProcessLookupError):pass
for path in ('/tmp/llm_pretrain_burn.py','/mnt/local/_gpu_guard/gpu_guard.sh','/mnt/local/_gpu_guard/polite_burn.py','/mnt/local/_gpu_guard/DISABLED'):
 p=Path(path);print('GUARD_FILE',json.dumps(dict(path=path,exists=p.exists(),sha256=sha(p) if p.is_file() else None)),flush=True)
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
     print('OWNERSHIP',pid,parent,stat[19],cmd[:400],flush=True);pid=parent
    except OSError:break
from scripts.gpu_process import inspect,process
live=inspect();identities={}
for gpu in live['gpus']:
 for pid in gpu['pids']:
  worker=process(pid);parent=process(worker['ppid'])
  identities[str(pid)]=worker;identities[str(parent['pid'])]=parent
print('STRUCTURED_OWNERSHIP',json.dumps(dict(inspection=live,known_burn_identities=identities)),flush=True)
cfg=json.loads(Path('resources/a2_depth_partc_plan_20261101.json').read_text())
backup=Path('/mnt/local/_backup/deep-llms_th2')
locks=dict(cfg['reference_nll'],prepared=cfg['prepared'],historical_inputs=cfg['stream_history']['historical_inputs'],part_ab_report=cfg['part_ab_report'],B_terminal=cfg['predecessor']['complete'])
for key,item in locks.items():
 dest=Path(item['path']);src=backup/dest.relative_to('/mnt/local/_outputs/deep-llms_th2')
 print('RESTORE_CANDIDATE',json.dumps(dict(key=key,source=str(src),destination=str(dest),expected_sha256=item['sha256'],exists=src.is_file(),bytes=src.stat().st_size if src.is_file() else None,sha256=sha(src) if src.is_file() else None)),flush=True)
 if key in ('prepared','dclm/P300/parent','dclm/P1T/parent') and src.is_file():
  value=json.loads(src.read_text());summary={k:value[k] for k in ('protocol','protocol_sha256','request_sha256','loss','tokens','sequences') if k in value}
  print('REFERENCE_SUMMARY',key,json.dumps(summary),flush=True)
for path in ('/mnt/local/conda-py311/envs/eval/bin/python3.11','/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023'):
 p=Path(path);print('EVAL_ASSET',json.dumps(dict(path=path,exists=p.exists())),flush=True)

asset_receipt=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-assets-20261103-a02/complete.json')
if asset_receipt.exists():
 value=json.loads(asset_receipt.read_text())
 print('DEPTH_ASSETS_RESTORED',json.dumps(dict(status=value['status'],files=len(value['files']),sha256=sha(asset_receipt),bytes=sum(x['bytes'] for x in value['files']))),flush=True)
benchmark=Path('/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023')
manifest=json.loads(Path('resources/downstream_english_20261023.json').read_text())
checked=[]
for repo in manifest['repositories']:
 for item in repo['files']:
  path=benchmark/item['path']
  checked.append(dict(path=item['path'],exists=path.is_file(),sha_matches=path.is_file() and sha(path)==item['sha256']))
print('BENCHMARK_FILES',json.dumps(checked),flush=True)
eval_python=Path('/mnt/local/conda-py311/envs/eval/bin/python3.11')
if eval_python.exists():
 code="from importlib.metadata import version; import json; from lm_eval import evaluator; import torch, transformers, tokenizers, datasets; print(json.dumps(dict(packages={k:version(k) for k in ('lm-eval','torch','transformers','tokenizers','datasets')},torch_cuda=torch.version.cuda)))"
 result=subprocess.run([str(eval_python),'-c',code],capture_output=True,text=True)
 print('EVAL_RUNTIME',result.returncode,result.stdout[-3000:],result.stderr[-2000:],flush=True)
 result=subprocess.run([str(eval_python),'-m','pip','check'],capture_output=True,text=True)
 print('EVAL_PIP_CHECK',result.returncode,result.stdout[-2000:],result.stderr[-1000:],flush=True)

PY
