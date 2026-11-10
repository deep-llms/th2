"""CPU-only F9 report recovery on the node; export compact results only."""
import argparse,json,math,csv
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from eval.a2_family_f9 import items
from eval.a2_family_f9_v110 import select,export_items,sampler,continuous_report,SPEC_SHA
from eval.a2_family_f9_report import analyze
from efficient_llm.diagnostic_data import sha256
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
r=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def compare(x,y,path='root'):
 if isinstance(x,dict):
  assert isinstance(y,dict) and set(x)<=set(y),(path,'keys')
  for k,v in x.items():compare(v,y[k],path+'/'+k)
 elif isinstance(x,list):
  assert isinstance(y,list) and len(x)==len(y),(path,'length')
  for i,(v,w) in enumerate(zip(x,y)):compare(v,w,path+'/'+str(i))
 elif isinstance(x,(int,float)) and not isinstance(x,bool):
  assert isinstance(y,(int,float)) and math.isclose(x,y,abs_tol=1e-10,rel_tol=1e-10),(path,x,y)
 else:assert x==y,(path,x,y)
# Verify every saved scientific artifact against the completed backup manifest.
assert sha256(r/'backup/verified.json')==a.manifest_sha256
manifest=read(r/'backup/verified.json');assert manifest['verified']
for name,v in manifest['files'].items():
 target=r/name
 assert target.is_file() and target.stat().st_size==v['bytes'] and sha256(target)==v['sha256'],name
cfg=read(r/'config.json')
for item,source in [(cfg['source_manifest'],r.parents[2]/'source'),(cfg['evaluator_source_manifest'],Path(cfg['evaluator_source']))]:
 assert sha256(item['path'])==item['sha256']
 for name,h in read(item['path'])['files'].items():assert sha256(source/name)==h,name
for name in ('eval/a2_family_f9.py','eval/a2_family_f9_v110.py','eval/a2_family_f9_report.py','eval/a2_requests.py'):
 assert sha256(Path(name))==sha256(r.parents[2]/'source'/name),name
from scripts.gpu_process import inspect,process
native=read(r.parents[2]/'supervised/supervisor.json');assert native['training_returncode']==0 and native['finished_at']
burn=read(r.parents[2]/'supervised/burn-verified.json');assert burn['collective_progress_verified']
current=inspect()
for expected,actual in zip(burn['gpus'],current['gpus'],strict=True):
 assert expected['uuid']==actual['uuid'] and expected['pids']==actual['pids']
 for pid in actual['pids']:assert process(pid)['ppid']==140037
print('INPUT_HASHES_VERIFIED '+str(len(manifest['files'])),flush=True)
idx=read(r/'portable_index.json')['models'];assert len(idx)==55
recorded=read(r/'contrasts.json');entries=recorded['model_metadata'];groups,excluded=select(entries)
assert not excluded and [len(g) for g in groups]==[13,12,9,12,9]
prepared={f:read(r/'preparation'/(f+'_prepared.json')) for f in ('gpt_neox','smollm3')}
replays={};continuous={};checks=[]
for model,v in idx.items():
 loaded={}
 for kind in ('scores','replay'):
  target=r/v[kind]['path'];assert sha256(target)==v[kind]['sha256'];loaded[kind]=read(target)
 saved=read(r/'per_item'/(model+'.json'));family='smollm3' if saved['model']['family']=='smollm3' else 'gpt_neox'
 rebuilt,accuracy=export_items(loaded['replay'],prepared[family],loaded['scores'])
 compare(rebuilt,saved)
 assert saved['model']==loaded['scores']['checkpoint_identity']
 replays[model]=loaded['replay'];continuous[model]=rebuilt['continuous_records'];checks.append(model)
 print('ITEMS_VERIFIED '+model,flush=True)
records={m:items(v) for m,v in replays.items()}
accuracy=analyze(entries,replays,draw_sampler=sampler(records));compare(accuracy,recorded)
# Recomputed arrays stay in memory; only compact final outputs will be exported.
secondary=continuous_report(entries,replays,continuous);compare(secondary,read(r/'continuous.json'))

proof=dict(verified=True,models=len(checks),models_checked=checks,joint_bootstrap_resamples=10000,seed=0,spec_sha256=SPEC_SHA,scope='Raw LL to per-item accuracy/continuous reconstruction and complete report numerical replay; no model run')
from eval.a2_family_f9_report import render
from eval.a2_family_f9_v110 import render_secondary
from scripts.a2_family_f9_v110_run import locked_report
# All recorded numerical results have passed independent CPU reconstruction.
section=render(recorded).replace('Family v1.9 — F9 downstream','Family v1.10 — F9 downstream')
section+='\n'+render_secondary(recorded,secondary)+'\nSpec SHA256: '+SPEC_SHA+'\n'
section+='\n### Report recovery and verification\n\nAll 55 models completed scoring and item replay. The original final_readout failed because the prior Markdown report was parsed as JSON. This CPU-only recovery verifies the locked prior report as text, reconstructs all per-item accuracy and continuous metrics from saved raw likelihoods, and reproduces the complete joint 10,000-resample seed-0 readout within 1e-10. No model was rerun. The original failure and all historical scientific flags are retained. Raw artifacts and the full backup remain on the node.\n'
full=locked_report(cfg['previous_report'])+'\n'+section
(out/'REPORT_FULL_v1.10.md').write_text(full)
byid={e['model_id']:e for e in entries}
columns=['model','family','dataset','role','arm','budget_B','final_NLL','MC8_accuracy_pct','MC7_accuracy_pct','LAMBADA_accuracy_pct','MC8_p_correct','MC7_p_correct','LAMBADA_target_NLL']
with (out/'F9_MODELS.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(columns)
 for model,v in accuracy['models'].items():
  e=byid[model];c=secondary['models'][model]
  w.writerow([model,e['family'],e['dataset'],e['role'],e['arm'],e.get('budget'),e.get('final_nll'),100*v['MC8'],100*v['MC7'],100*v['lambada_openai'],c['MC8'],c['MC7'],c['LAMBADA_target_NLL']])
proof.update(native_handoff_verified=True,gpu_ownership_verified=True,utc=datetime.now(timezone.utc).isoformat(),original_manifest_members=len(manifest['files']),original_archive_sha256=manifest['archive_sha256'],retained_flags=read(r/'flags.json')['flags'],report_error_resolved=True,previous_report=cfg['previous_report'],files={n:dict(bytes=(out/n).stat().st_size,sha256=sha256(out/n)) for n in ['REPORT_FULL_v1.10.md','F9_MODELS.csv']})
(out/'verification.json').write_text(json.dumps(proof,indent=2))
print('F9_REPORT_RECOVERY_VERIFIED '+json.dumps({k:v for k,v in proof.items() if k not in ['models_checked','retained_flags']}),flush=True)
