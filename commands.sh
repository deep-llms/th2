#1 +20+a
#th2-wc69-depth-final-backup-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'BACKUP'
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01/results/a2_depth_20261031')
run=root.parent.parent
family=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02')
out=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-final-backup-20261110-a01')
spec='2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
terminal=read(family/'supervised/run/depth-complete.json');assert all(terminal[k] for k in ['execution_complete','training_coverage_complete','downstream_complete','readout_complete'])
native=read(family/'supervised/run/complete.json');assert native['status']=='ok' and all(v['returncode']==0 for v in native['jobs'])
inspection=inspect();burn=read(family/'supervised/burn-verified.json');assert burn['collective_progress_verified']
assert [(g['index'],g['uuid'],g['pids']) for g in inspection['gpus']]==[(g['index'],g['uuid'],g['pids']) for g in burn['gpus']]
chains={}
for g in inspection['gpus']:
 assert len(g['pids'])==1
 p=process(g['pids'][0]);parent=process(p['ppid']);assert parent['pid']==123817 and parent['start_ticks']==415891689 and parent['launcher_kind']=='gpu_burn'
 assert parent['command_prefix'][1]==str(family/'source/resources/llm_pretrain_burn.py')
 chains[str(g['index'])]=[p,parent]
source_checks=[]
for base in [run,run/'continuations/attempt001',family]:
 m=read(base/('execution-source-manifest.json' if base.name=='attempt001' else 'source-manifest.json'));bad=[p for p,h in m['files'].items() if sha(base/'source'/p)!=h];assert not bad
 source_checks.append(dict(root=str(base),files=len(m['files']),changed=bad))
locks={'X72':'5b3f8895d6fb5d5751856634870df352f474d92f52ca14525453d061149a53dc','P300':'8cc3ed270ff861292681f04c65cb78ad0190d529db3117bc6e7736fefdce7a9a','P1T':'b8f02aa472a6a1863dcc3df8a341a3b5db50060d321f252fecf60f71db0ef28e'}
for arm,h in locks.items():
 p=root/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'/'result.json';assert sha(p)==h
 result=read(p);assert result['status']=='complete' and result['spec_sha256']==spec and len(result['lr_updates'])==48828
 marker=read(p.parent/'arm-complete.json');assert marker['identity']==result['full_state_identity']
 for item in marker['files']:
  q=Path(item['path']);rel=q.relative_to(p.parent)
  if 'final' not in rel.parts:assert sha(q)==item['sha256']
cfg=read(root/'config/plan.json');assert sha(root/'config/plan.json')=='798208e92b5e31535eb1859c28c4ecbc83bdd2874438f7a8aa2148e8d744b7b4'
down=read(root/'downstream/complete.json');assert down['coverage_complete'] and down['execution_complete'] and down['production_batch_size']==2 and len(down['scores'])==6
sources={};files={}
def add(p,name,expected=None):
 p=Path(p);assert p.is_file() and not p.is_symlink();assert not Path(name).is_absolute() and '..' not in Path(name).parts
 h=sha(p)
 if expected:assert h==expected,(str(p),'hash')
 if name in files:assert files[name]['sha256']==h;return
 sources[name]=p;files[name]=dict(bytes=p.stat().st_size,sha256=h,source_path=str(p))
# Result-only files: no weights, optimizer states, tokenizer/data caches or credentials.
for p in root.rglob('*'):
 rel=p.relative_to(root)
 if any(x in ('trainer','final','backup','clean-state') or x.startswith('checkpoint-') for x in rel.parts):continue
 if p.is_file() and p.suffix in ('.json','.jsonl','.npy','.csv','.md','.log','.yaml','.txt'):
  add(p,str(rel))
for group in ('scores','replays','independent_replays'):
 for name,lock in down[group].items():add(lock['path'],f'dependencies/{group}/{name}.json',lock['sha256'])
for name,lock in [('prepared',cfg['prepared']),('input_spec',cfg['downstream_common']['input_spec']),('manifest',cfg['downstream_common']['manifest'])]:add(lock['path'],f'dependencies/{name}.json',lock['sha256'])
for rel in ['supervised/run/complete.json','supervised/run/depth-complete.json','supervised/run/depth-resume-admission.json','supervised/run/jobs.snapshot.json','supervised/burn-verified.json','supervised/supervisor.json','source-manifest.json']:
 add(family/rel,'handoff/'+rel)
for label,base in [('original',run),('continuation',run/'continuations/attempt001')]:
 for name in ['source-manifest.json','execution-source-manifest.json','jobs.json','admission.json']:
  if (base/name).is_file():add(base/name,f'provenance/{label}/{name}')
# Keep actual pause/resume evidence and verify pinned copy metadata against its sealed state.
pause=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01/pause');state=read(pause/'resume_state.json');members={}
for name,h in state['hashes'].items():
 p=Path(state['checkpoint'])/name;assert p.is_file();members[name]=dict(bytes=p.stat().st_size)
 if name.endswith('.json'):assert sha(p)==h;members[name]['sha256']=h
for p in pause.glob('*.json'):add(p,'pause/'+p.name)
out.mkdir(exist_ok=False)
health=dict(utc=datetime.now(timezone.utc).isoformat(),inspection=inspection,chains=chains,source_checks=source_checks,pinned_pause_members=members,terminal=terminal,native=native,downstream_coverage_complete=down['coverage_complete'],downstream_flags=down['flags'],scope='Science members and source bytes hashed; pinned large checkpoint/weights checked for existence and size only, JSON hashes checked. No new model run.')
(out/'health.json').write_text(json.dumps(health,indent=2)+'\n');add(out/'health.json','final_health.json')
archive=out/'depth_final_scientific.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for rel in sorted(files):tar.add(sources[rel],arcname=rel,recursive=False)
seen=set()
with tarfile.open(archive,'r:gz') as tar:
 for x in tar:
  assert x.isfile() and x.name in files and x.name not in seen;seen.add(x.name);data=tar.extractfile(x).read();assert len(data)==files[x.name]['bytes'] and hashlib.sha256(data).hexdigest()==files[x.name]['sha256']
assert seen==set(files)
parts=[]
with archive.open('rb') as f:
 while True:
  b=f.read(24*1024*1024)
  if not b:break
  p=out/f'depth_final_scientific.part{len(parts):03}';p.write_bytes(b);parts.append(dict(path=str(p),bytes=len(b),sha256=sha(p)))
manifest=dict(spec_sha256=spec,utc=datetime.now(timezone.utc).isoformat(),files=files,members=len(files),parts=parts,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),raw_rank_traces=sum('sample-order-rank' in x for x in files),result_locks=locks,verified=True,scope='All available final Depth science, scores, parent dependencies, both CPU replays, prepared requests, provenance and native handoff; excludes weights/optimizer/data. Independent local replay still required.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DEPTH_FINAL_BACKUP',json.dumps(dict(manifest_path=str(out/'manifest.json'),manifest_sha256=sha(out/'manifest.json'),members=len(files),parts=parts,archive=manifest['archive'],raw_rank_traces=manifest['raw_rank_traces'],health=health)),flush=True)
BACKUP
