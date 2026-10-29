"""Registered Stage1 orchestration. No implicit GPU launch outside its CLI."""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
from efficient_llm.a2.data import write_json
from efficient_llm.a2.protocol import lr_for_update,warmup_steps
from efficient_llm.a2_stage1.protocol import GRID,ARMS,Run,remaining,grid_choice,decision,SPEC_SHA
from efficient_llm.diagnostic_data import sha256


def read(p):return json.loads(Path(p).read_text())

def checked(path,expected):
 if sha256(path)!=expected:raise ValueError('Changed artifact: '+str(path))
 return read(path)


def baseline(parent,teacher,provenance):
 if parent['provenance']!=provenance or teacher['provenance']!=provenance:raise ValueError('Baseline provenance changed')
 return dict(parent['pre'],parent_final=parent['baseline']['final']['loss'],teacher_final=teacher['baseline']['final']['loss'],
  parent_secondary=parent['baseline']['secondary']['loss'],teacher_secondary=teacher['baseline']['secondary']['loss'],provenance=provenance)


def finalize_preflight(root,inputs):
 inputs=read(inputs);prov=inputs['provenance'];parents={p:read(root/(p+'.json')) for p in ('P300','P1T','S72')}
 numerical=read(root/'numerical.json');t8=read(root/'throughput.json');t17=read(root/'t17/T17.json')
 records=[*parents.values(),numerical,t8,t17]
 if any(r['provenance']!=prov for r in records):raise ValueError('Mixed preflight provenance')
 if not numerical['passed'] or not t8['passed'] or not t17['passed']:raise ValueError('Required numerical/throughput/T17 failed')
 if not all(parents[p]['T1']['passed'] and parents[p]['sanity']['passed'] for p in ('P300','P1T')):raise ValueError('Annealed parent T1 failed')
 eligible=parents['S72']['T1']['passed'] and parents['S72']['sanity']['passed'] and inputs['T3']['passed']
 step=t8['step_seconds'];count=51108 if eligible else 35088
 # Include loading/full validation/export, possible one extra edge pair,
 # downstream and cleanup explicitly; measured fit timings refine later ETAs.
 seconds=(count+1524)*step+(15 if eligible else 12)*180+3600
 tests=dict(T1={p:v['T1'] for p,v in parents.items()},T3=inputs['T3'],T5=numerical['T5_loss'],T12=numerical,
  T7=inputs['data_manifest']['T7'],T8=t8,T17=t17,T16=dict(passed=read(root/'t17/result.json')['roundtrip_difference']<=1e-6),
  T6={b:{str(i):lr_for_update(i,n,3e-4) for i in (1,warmup_steps(n),math.ceil(n/2),n-1,n)} for b,n in [('0.2B',762),('0.8B',3051),('3.2B',12207)]},
  T18=dict(inputs['data_manifest']['T18'],parent_loss={p:dict(dclm=v['baseline']['secondary']['loss'],culturax=v['baseline']['final']['loss'],flag=v['baseline']['secondary']['loss']>=v['baseline']['final']['loss']) for p,v in parents.items() if v['baseline'] is not None}))
 report=dict(status='passed',provenance=prov,eligible_x72=eligible,parents=parents,tests=tests,step_seconds=step,
  annealing_gain=None if parents['S72']['baseline'] is None else parents['S72']['baseline']['final']['loss']-parents['P300']['baseline']['final']['loss'],
  estimated_remaining_seconds=seconds,report_before_launch=seconds+inputs.get('preflight_cap_seconds',7200)>30*3600,
  preflight_artifacts={str(p.relative_to(root)):sha256(p) for p in [*(root/(k+'.json') for k in parents),root/'numerical.json',root/'throughput.json',root/'t17/T17.json',root/'t17/result.json']})
 write_json(root/'tests.json',report);write_json(root/'parents.json',parents)
 print('STAGE1_PREFLIGHT_PASSED',json.dumps(dict(eligible_x72=eligible,step_seconds=step,estimated_remaining_hours=seconds/3600)),flush=True)
 return report


def validate_fit(row,run,prov):
 if any(row.get(k)!=v for k,v in run.record().items()) or row['provenance']!=prov:raise ValueError('Fit identity mismatch')
 if row.get('status')=='complete':
  if row['coverage']!=dict(examples=run.steps*64,microbatches=run.steps*2,world_size=8,exact_order=True):raise ValueError('Wrong training coverage')
  expected=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)]
  if any(abs(a-b)>1e-12 for a,b in zip(row['lr_updates'],expected,strict=True)):raise ValueError('Wrong schedule')
  if row['roundtrip_difference']>1e-6:raise ValueError('Checkpoint roundtrip failed')
 elif not row.get('diverged'):raise ValueError('Non-divergence failure')


def report_runs(rows,chosen,parents):
 records={(r['arm'],r['budget']):r for r in rows if r['lr']==chosen and r.get('status')=='complete' and not r.get('diverged')}
 budgets={}
 for b in ('0.2B','0.8B','3.2B'):
  p,q,x=(records.get((a,b)) for a in ARMS)
  budgets[b]=dict(E=None if not p or not q else q['delta_post']-p['delta_post'],A_init=None if not p or not x else x['val_post']-p['val_post'],
   E_dclm=None if not p or not q else q['delta_secondary_post']-p['delta_secondary_post'],A_init_dclm=None if not p or not x else x['val_secondary_post']-p['val_secondary_post'])
 main=budgets['3.2B'];ratios={k:main[k]/budgets['0.8B'][k] if main[k] is not None and budgets['0.8B'][k] is not None and abs(budgets['0.8B'][k])>=.003 else None for k in ('E','A_init')}
 return dict(budgets=budgets,length=decision(main['E'],'length'),state=decision(main['A_init'],'state'),ratios=ratios,
  E_pre=(parents['P1T']['pre']['final']['loss']-parents['P1T']['baseline']['final']['loss'])-(parents['P300']['pre']['final']['loss']-parents['P300']['baseline']['final']['loss']),
  parent_difference=parents['P1T']['baseline']['final']['loss']-parents['P300']['baseline']['final']['loss'],
  exposure_caveat='Own-corpus heldout may have been seen during parent pretraining, more by P1T; no decisions use it.')


def execute(root,inputs_path,preflight,port,spent_seconds,runner=subprocess.run):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);inputs=read(inputs_path);prov=inputs['provenance'];gate=read(preflight/'tests.json')
 if gate['status']!='passed' or gate['provenance']!=prov:raise ValueError('Missing same-source preflight')
 for rel,digest in gate['preflight_artifacts'].items():checked(preflight/rel,digest)
 if gate['report_before_launch']:raise ValueError('Over30h estimate: user report required before launch')
 if spent_seconds+gate['estimated_remaining_seconds']*1.5+600>36*3600:raise ValueError('Insufficient36h budget')
 started=time.monotonic();rows=[];grid=[];parents=gate['parents'];step=gate['step_seconds']
 write_json(root/'config/inputs.json',inputs);write_json(root/'tests.json',gate)
 write_json(root/'manifests/heal.json',inputs['data_manifest'])
 (root/'config/stage1_spec.md').write_bytes(Path(inputs['spec']).read_bytes())
 with (root/'parents.jsonl').open('x') as f:
  for name,record in parents.items():f.write(json.dumps(dict(record,parent=name))+'\n')
 launcher=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(Path(__file__).resolve().parents[1]/'resources/accelerate_b200.yaml'),'--main_process_port',str(port),'-m','efficient_llm.a2_stage1.model_stage','heal','--inputs',str(inputs_path)]
 def fit(run,is_grid):
  estimate=run.steps*step+180;elapsed=spent_seconds+time.monotonic()-started
  if elapsed+1.5*estimate+600>36*3600:raise TimeoutError('Stage1 cap before fit')
  rp=root/(run.run_id+'.json');pp=root/(run.run_id+'-pre.json');out=root/run.run_id
  write_json(rp,run.record());write_json(pp,baseline(parents[run.parent],parents[run.teacher],prov))
  print('STAGE1_FIT_START',run.run_id,'estimate_seconds',estimate,flush=True)
  with (root/(run.run_id+'.log')).open('x') as log:
   proc=runner(launcher+['--run',str(rp),'--pre',str(pp),'--output',str(out),'--estimate',str(estimate)],stdout=log,stderr=subprocess.STDOUT,timeout=1.5*estimate)
  row=read(out/'result.json');validate_fit(row,run,prov)
  if proc.returncode and not (is_grid and row.get('diverged')):raise RuntimeError('Fit process failed')
  rows.append(row);write_json(root/'runs.json',rows)
  with (root/'runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  write_json(root/'progress.json',dict(status='running',completed=len(rows),latest=run.run_id,spent_seconds=spent_seconds+time.monotonic()-started))
  if row.get('diverged') and not is_grid:raise ValueError('Non-grid divergence; no LR retry authorized')
  print('STAGE1_FIT_END',run.run_id,row.get('val_post'),flush=True);return row
 for lr in GRID:
  for arm in ARMS[:2]:grid.append(fit(Run(arm,'0.2B',lr),True))
 choice=grid_choice(grid);lrs=GRID
 if choice['edge_lr'] is not None:
  edge=choice['edge_lr'];lrs=(*GRID,edge)
  for arm in ARMS[:2]:grid.append(fit(Run(arm,'0.2B',edge),True))
  choice=grid_choice(grid,lrs)
 choice['own_best']={a:grid_choice([r for r in grid if r['arm']==a],lrs,(a,))['chosen_lr'] for a in ARMS[:2]}
 write_json(root/'lr_choice.json',choice);chosen=choice['chosen_lr']
 for run in remaining(chosen,gate['eligible_x72']):fit(run,False)
 result=report_runs(rows,chosen,parents);result.update(chosen_lr=chosen,own_best_lrs=choice['own_best'],stage1_spec_sha256=SPEC_SHA,spent_seconds=spent_seconds+time.monotonic()-started,downstream_status='pending')
 own={a:next(r for r in grid if r['arm']==a and r['lr']==choice['own_best'][a]) for a in ARMS[:2]}
 result['E_short_own_best']=own['P1T']['delta_post']-own['P300']['delta_post']
 write_json(root/'decision_stage1.json',result)
 (root/'REPORT.md').write_text('Stage1 training complete; downstream evaluation pending.\n\n'
  +'Common LR: '+str(chosen)+'\n\n'+json.dumps(result,indent=2)+'\n')
 write_json(root/'complete.json',dict(status='complete',training_complete=True,full_stage_complete=False,downstream_pending=True,runs=len(rows),decision_sha256=sha256(root/'decision_stage1.json')))


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('finalize','heals'))
 for k in ('output','inputs'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--preflight',type=Path);p.add_argument('--port',type=int);p.add_argument('--spent-seconds',type=float,default=0);a=p.parse_args()
 if a.action=='finalize':finalize_preflight(a.output,a.inputs)
 else:execute(a.output,a.inputs,a.preflight,a.port,a.spent_seconds)
if __name__=='__main__':main()
