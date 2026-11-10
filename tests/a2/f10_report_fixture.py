"""Synthetic full readout integration; no model evaluation or actual result claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import tempfile,json
from eval.a2_family_f9_v110 import select
from eval.a2_family_f10 import select_models,write
from eval.a2_family_f10_report import report
from eval.a2_family_f10_tasks import likelihood_metrics,answer_metrics
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256
entries=sum(select(json.load(open('resources/a2_family_f9_plan_20261110.json'))['models'])[0],[])
ones,groups=select_models(entries,json.load(open('resources/a2_family_f10_plan_20261111.json')))
with tempfile.TemporaryDirectory() as d:
 root=Path(d)/'run';root.mkdir();old=Path(d)/'f9';old.mkdir();a=Path(d)/'f10a';a.mkdir()
 def saved(e):
  tasks={};cont={};samples={}
  for t in COUNTS:
   tasks[t]=[dict(doc_id=i,raw_loglikelihoods=[-2.,-3.],choice_character_lengths=[2,2],gold_choice=0,p_correct=.6,correctness=1,target_NLL=2.) for i in range(3)]
   cont[t]=dict(identity=t,values=[.6]*3)
   samples[t]=[dict(doc_id=i,target=' ok',arguments=[['Q',' ok'],['Q',' no']]) for i in range(3)]
  return dict(tasks=tasks,continuous_records=cont),dict(samples=samples)
 index={}
 for e in entries:
  m=e['model_id'];ex,r=saved(e);write(old/'per_item'/(m+'.json'),ex);write(old/'raw'/m/'replay.json',r)
  index[m]={'replay':{'path':'raw/'+m+'/replay.json'}}
 for e in ones:
  ex,r=saved(e);write(root/'nine'/e['model_id']/'per_item.json',ex);write(root/'nine'/e['model_id']/'replay.json',r)
 write(old/'contrasts.json',dict(model_metadata=entries));write(old/'portable_index.json',dict(models=index))
 files={str(p.relative_to(old)):{'sha256':sha256(p)} for p in old.rglob('*') if p.is_file()};write(old/'backup/verified.json',dict(files=files))
 rows={'mmlu':[dict(doc_id='m'+str(i),subject='s',requests=[0,1],character_lengths=[2,3],gold=0) for i in range(3)],
       'triviaqa':[dict(doc_id='t'+str(i),requests=[2],character_lengths=[2],gold=0,aliases=['ok']) for i in range(3)]}
 prepared={'tasks':rows,'request_sha256':'r','task_identity':{'mmlu':'m','triviaqa':'t'}};prep=Path(d)/'prepared.json';write(prep,prepared);prepared=json.load(open(prep))
 for e in sum(groups,[]):
  f=root/'new'/e['model_id'];scores={'scores':[[-2.,True],[-4.,False],[-2.,True]],'prepared_sha256':digest(prepared),'request_sha256':'r'}
  write(f/'scores.json',scores);write(f/'likelihood_items.json',dict(tasks={t:likelihood_metrics(rs,scores['scores'],t) for t,rs in rows.items()},task_identity=prepared['task_identity']))
  if e['role']!='oneshot':write(f/'generation_items.json',dict(items=[dict(doc_id=r['doc_id'],text='ok',**answer_metrics('ok',r['aliases'])) for r in rows['triviaqa']],task_identity='t',exact_match=1.,token_f1=1.))
 write(a/'tests.json',{'families':{}});prior=Path(d)/'prior.md';prior.write_text('PRIOR REPORT\n')
 cfg={'f9_data_root':str(old),'oneshot_entries':ones,'groups':groups,'prepared':lock(prep),'f10a_root':str(a),'previous_report':lock(prior),'retained_flags':[]}
 report(cfg,root)
 done=json.load(open(root/'report_complete.json'));assert done['complete'],done
 assert (root/'REPORT_FULL_v1.11.md').read_text().startswith('PRIOR REPORT\n')
 assert len(json.load(open(root/'F10c.json'))['model_scores'])==42
 print('FULL_REPORT_FIXTURE_PASS',done,'report_bytes',(root/'REPORT_FULL_v1.11.md').stat().st_size)
