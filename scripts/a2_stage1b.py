"""Registered FWE parts, reusing Stage1 schedule, fit validation and labels."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from efficient_llm.a2.data import write_json
from efficient_llm.diagnostic_data import sha256
from efficient_llm.a2_stage1b.protocol import Run,GRID,ARMS,SPEC_SHA,REPLICATION_SPEC_SHA,phase_runs,grid_choice,cross_dataset_label
from scripts.a2_stage1 import read,checked,baseline,validate_fit,report_runs as _report_runs
from scripts.a2_remaining import read_locked

def report_runs(rows,chosen,parents):
 result=_report_runs(rows,chosen,parents)
 for v in result['budgets'].values():
  v['E_fwe']=v.pop('E_dclm');v['A_init_fwe']=v.pop('A_init_dclm')
 return result

def execute(root,inputs_path,preflight,port,spent_seconds,runner=subprocess.run,*,phase="part1",prior_lock=None):
 root=Path(root);inputs=read(inputs_path);prov=inputs['provenance'];gate=read(preflight/'tests.json')
 if inputs.get('dataset')!='fwe' or prov.get('stage1b_spec_sha256')!=REPLICATION_SPEC_SHA or sha256(inputs['replication_spec'])!=REPLICATION_SPEC_SHA:raise ValueError('Stage1b spec mismatch')
 root.mkdir(parents=True,exist_ok=False)
 if gate['status']!='passed' or gate['provenance']!=prov:raise ValueError('Missing same-source preflight')
 for rel,digest in gate['preflight_artifacts'].items():checked(preflight/rel,digest)
 if gate['report_before_launch']:raise ValueError('Over20h estimate: user report required before launch')
 estimate_total=gate['estimated_remaining_seconds'] if phase=='part1' else sum(r.steps*gate['step_seconds']+180 for r in phase_runs('part2',.001,gate['eligible_x72']))+3600
 if spent_seconds+estimate_total+600>24*3600:raise ValueError('Insufficient24h budget')
 started=time.monotonic();rows=[];grid=[];parents=gate['parents'];step=gate['step_seconds']
 write_json(root/'config/inputs.json',inputs);write_json(root/'tests.json',gate)
 write_json(root/'manifests/heal.json',inputs['data_manifest'])
 if inputs.get('source_plan'):(root/'manifests/source_plan.json').write_bytes(Path(inputs['source_plan']).read_bytes())
 (root/'config/stage1_spec.md').write_bytes(Path(inputs['spec']).read_bytes())
 (root/'config/stage1b_spec.md').write_bytes(Path(inputs['replication_spec']).read_bytes())
 with (root/'parents.jsonl').open('x') as f:
  for name,record in parents.items():f.write(json.dumps(dict(record,parent=name))+'\n')
 launcher=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(Path(__file__).resolve().parents[1]/'resources/accelerate_b200.yaml'),'--main_process_port',str(port),'-m','efficient_llm.a2_stage1b.model_stage','heal','--inputs',str(inputs_path)]
 def fit(run,is_grid):
  estimate=run.steps*step+180;elapsed=spent_seconds+time.monotonic()-started
  if elapsed+1.5*estimate+600>24*3600:raise TimeoutError('Stage1 cap before fit')
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
 if phase=='part1':
  if prior_lock is not None:raise ValueError('Part1 cannot consume prior fits')
  for lr in GRID:
   for arm in ARMS[:2]:grid.append(fit(Run(arm,'0.2B',lr),True))
  choice=grid_choice(grid);lrs=GRID
  if choice['edge_lr'] is not None:
   edge=choice['edge_lr'];lrs=(*GRID,edge)
   for arm in ARMS[:2]:grid.append(fit(Run(arm,'0.2B',edge),True))
   choice=grid_choice(grid,lrs)
  choice['own_best']={a:grid_choice([r for r in grid if r['arm']==a],lrs,(a,))['chosen_lr'] for a in ARMS[:2]}
  write_json(root/'lr_choice.json',choice)
  grid_report=report_runs(rows,choice['chosen_lr'],parents)
  (root/'GRID_NOTE.md').write_text('FineWeb-Edu grid complete; common LR '+str(choice['chosen_lr'])+'\n\n'+json.dumps(dict(grid=[{k:r[k] for k in ('arm','lr','val_tune_post','delta_post')} for r in grid],choice=choice,budget_0p2=grid_report['budgets']['0.2B']),indent=2)+'\n')
  print('STAGE1B_GRID_NOTE',json.dumps(dict(choice=choice,budget_0p2=grid_report['budgets']['0.2B'])),flush=True)
 elif phase=='part2':
  if prior_lock is None:raise ValueError('Verified part1+interim lock required')
  prior=read_locked(prior_lock)
  if prior.get('stage1b_spec_sha256')!=REPLICATION_SPEC_SHA:raise ValueError('Wrong prior preregistration')
  gate1=read_locked(prior['complete']);interim=read_locked(prior['interim'])
  if gate1.get('phase')!='part1' or gate1.get('status')!='complete' or interim.get('status')!='verified' or not interim.get('report_sent'):raise ValueError('Part1/interim handoff missing')
  if gate1.get('runs_sha256')!=prior['runs']['sha256'] or gate1.get('choice_sha256')!=prior['choice']['sha256']:raise ValueError('Prior terminal hashes differ')
  rows=read_locked(prior['runs']);choice=read_locked(prior['choice'])
  for row in rows:validate_fit(row,Run(row['arm'],row['budget'],row['lr']),prov)
  if len({r['run_id'] for r in rows})!=len(rows):raise ValueError('Duplicate prior fits')
  grid=[r for r in rows if r['budget']=='0.2B']
  if any(r['status']!='complete' or r.get('diverged') for r in rows if r['budget']!='0.2B'):raise ValueError('Incomplete part1 mains')
  if {(r['arm'],r['budget']) for r in rows if r['budget']!='0.2B'}!={('P300','3.2B'),('P1T','3.2B')}:raise ValueError('Wrong part1 fits')
  lrs=tuple(sorted({r['lr'] for r in grid}))
  if lrs not in (GRID,tuple(sorted((*GRID,3e-5))),tuple(sorted((*GRID,3e-3)))):raise ValueError('Unregistered prior grid')
  edge=grid_choice([r for r in grid if r['lr'] in GRID])['edge_lr']
  expected_lrs=tuple(sorted((*GRID,edge))) if edge is not None else GRID
  if lrs!=expected_lrs:raise ValueError('Prior grid edge rule changed')
  if grid_choice(grid,lrs)['chosen_lr']!=choice['chosen_lr']:raise ValueError('Prior LR mismatch')
  if choice['own_best']!={a:grid_choice([r for r in grid if r['arm']==a],lrs,(a,))['chosen_lr'] for a in ARMS[:2]}:raise ValueError('Prior own-best LR mismatch')
  if any(r['lr']!=choice['chosen_lr'] for r in rows if r['budget']!='0.2B'):raise ValueError('Part1 mains did not use common LR')
  write_json(root/'prior.json',prior);write_json(root/'lr_choice.json',choice)
  with (root/'runs.jsonl').open('x') as f:
   for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
 else:raise ValueError('Unknown part')
 chosen=choice['chosen_lr']
 for run in phase_runs(phase,chosen,gate['eligible_x72']):fit(run,False)
 result=report_runs(rows,chosen,parents);result.update(phase=phase,stage1b_spec_sha256=REPLICATION_SPEC_SHA,chosen_lr=chosen,own_best_lrs=choice['own_best'],stage1_spec_sha256=SPEC_SHA,spent_seconds=spent_seconds+time.monotonic()-started,downstream_status='pending')
 own={a:next(r for r in grid if r['arm']==a and r['lr']==choice['own_best'][a]) for a in ARMS[:2]}
 result['E_short_own_best']=own['P1T']['delta_post']-own['P300']['delta_post']
 result['cross_dataset_length']=cross_dataset_label(result['length']['label'])
 write_json(root/'decision_stage1_fwe.json',result)
 (root/'REPORT.md').write_text('Stage1b '+phase+' training complete; downstream evaluation pending.\n\n'
  +'Common LR: '+str(chosen)+'\n\n'+json.dumps(result,indent=2)+'\n')
 write_json(root/'complete.json',dict(status='complete',phase=phase,training_complete=phase=='part2',full_stage_complete=False,downstream_pending=True,runs=len(rows),runs_sha256=sha256(root/'runs.json'),choice_sha256=sha256(root/'lr_choice.json'),decision_sha256=sha256(root/'decision_stage1_fwe.json')))


def main():
 p=argparse.ArgumentParser()
 for n in ('output','inputs','preflight'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--phase',choices=('part1','part2'),required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--spent-seconds',type=float,required=True)
 p.add_argument('--prior');p.add_argument('--prior-sha256');a=p.parse_args()
 prior=dict(path=a.prior,sha256=a.prior_sha256) if a.prior else None
 execute(a.output,a.inputs,a.preflight,a.port,a.spent_seconds,phase=a.phase,prior_lock=prior)
if __name__=='__main__':main()
