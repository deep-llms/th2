"""Independent CPU replay of original downloaded C arrays/recipes/order/report.
No production scoring, readout, training or decision helpers are imported.
"""
import argparse,json,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
spec='2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe'
errors=[];checks=0;diffs=[];inputs={};values={};diagnostics=[]
def read(path):
 raw=path.read_bytes();inputs[str(path.relative_to(a.root))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def check(ok,name,detail=None):
 global checks
 checks+=1
 if not ok:errors.append(dict(check=name,detail=detail))
def near(x,y,name):
 d=abs(x-y);diffs.append(d);check(math.isfinite(d) and d<=1e-12,name,dict(actual=x,reported=y,difference=d))
def array(v,name,n=None):
 xs=v['per_sequence_nll'];check(bool(xs) and all(math.isfinite(x) for x in xs),name+':finite')
 check(v['sequences']==len(xs) and v['tokens']==4095*len(xs),name+':counts')
 if n is not None:check(len(xs)==n,name+':coverage')
 mean=math.fsum(xs)/len(xs);near(mean,v['loss'],name+':mean');return mean
cfg=read(a.root/'config/plan.json');saved=read(a.root/'decision_depth_partc.json')
check(cfg['spec_sha256']==saved['spec_sha256']==spec,'spec')
runs=[('dclm','S1',arm,f'ST1-dclm-{arm}-S1-12.8B-lr1e-3') for arm in ('P300','P1T','X72')]
for ds,block,arm,name in runs:
 try:
  root=a.root/'heals'/name;r=read(root/'result.json');recipe=read(root/'recipe.json')
  check(r['spec_sha256']==spec and r['run_id']==name,'identity:'+name)
  check(recipe['steps']==48828 and recipe['lr']==.001 and recipe['tokens']==48828*64*4096,'schedule:'+name)
  check(recipe['removed']==list(range({'W1':2,'W9':18,'S1':16}[block],{'W1':8,'W9':24,'S1':22}[block])),'layers:'+name)
  check(recipe['initial_parent']==('S72' if arm=='X72' else arm) and recipe['teacher']==('P300' if arm=='X72' else arm),'parents:'+name)
  expected=[.001*(i/977 if i<=977 else (48828-i)/47851) for i in range(1,48829)]
  check(len(r['lr_updates'])==48828 and all(abs(x-y)<=abs(y)*1e-12 for x,y in zip(r['lr_updates'],expected)),'actualLR:'+name)
  order=read(root/'order.json')['rank_microbatches'];check(len(order)==8 and all(len(rank)==97656 for rank in order),'rankcount:'+name)
  joined=[sample for batch in zip(*order) for rank in batch for sample in rank]
  check(joined==list(range(48828*64)) and all(len(b)==4 for rank in order for b in rank),'actualOrder:'+name)
  parent=array(r['baseline'],name+'/parent',2461);final=array(r['scores']['final'],name+'/final',2461)
  for k,v in r['scores'].items():array(v,name+'/'+k,2461 if k=='final' else None)
  for k,v in r['pre'].items():array(v,name+'/pre/'+k,2461 if k=='final' else None)
  near(final-parent,r['delta_post'],name+'/delta')
  for label,value in [('parent',r['baseline']),('pre-'+block,r['pre']['final']),('post-'+block,r['scores']['final'])]:
   file=read(a.root/'perseq'/name/(label+'.json'));check(file['per_sequence_nll']==value['per_sequence_nll'],name+'/'+label+':savedArray')
  rankflags=[]
  for rank in range(8):
   cs=read(root/f'tests-rank{rank}.json')['checks']
   rankflags.extend(dict(c,rank=rank) for c in cs if not c['passed'])
  check(rankflags==r['flags'] and r['unverified']==bool(rankflags),name+':flagPropagation')
  diagnostics.extend(rankflags);values[ds,block,arm]=(final,parent,final-parent)

 except Exception as exc:check(False,name,repr(exc))
try:
 p=values['dclm','S1','P300'];t=values['dclm','S1','P1T'];x=values['dclm','S1','X72']
 e=t[2]-p[2];init=x[0]-p[0]
 for arm in ('P300','P1T','X72'):
  val,par,delta=values['dclm','S1',arm]
  for k,v in dict(val=val,parent_val=par,delta_post=delta).items():near(v,saved['parents'][arm][k],arm+'/'+k)
 for k,v in dict(E=e,A_init=init,E_ratio_3_2B=e/cfg['stage1_dclm_reference']['3.2B']['E'],A_init_ratio_3_2B=init/cfg['stage1_dclm_reference']['3.2B']['A_init']).items():near(v,saved[k],k)
 for key,val,pos,neg,zero,threshold in [('E_label',e,'persists','reversed','erased',.008),('A_init_label',init,'anneal first','prune before you anneal','no effect',.01)]:
  clauses={'INCOMPLETE':False,pos:val>=threshold,neg:val<=-threshold,zero:abs(val)<.003}
  label=pos if clauses[pos] else neg if clauses[neg] else zero if clauses[zero] else 'grey'
  check(clauses==saved[key]['clauses'],key+':clauses');check(label==saved[key]['label'],key+':label')
 check(saved['predictions']==dict(C1=e>=.008,C2=init<.01),'predictions')
 dt6=read(a.root/'DT6.json')
 check(dt6['passed'] and dt6['selected_sequences']==3124992 and dt6['prefix_sequences']==781248,'DT6 counts')
 check(dt6['reader_prefix_sha256']==dt6['tokens']['prefix_sha256'],'DT6 prefix')
 check(dt6['tokens']['full_sha256']==cfg['stream_history']['data_manifest']['streams']['heal']['token_sha256'],'DT6 historical full stream')
except Exception as exc:check(False,'C readout',repr(exc))
result=dict(spec_sha256=spec,verification_passed=not errors,checks=checks,errors=errors,max_abs_difference=max(diffs,default=0),inputs=inputs,scientific_diagnostics=diagnostics,scope='Independent saved-array means, actual LR/order/parent recipe, per-rank flag propagation, E/A/relative values/labels/clauses; does not independently load model weights or verify native terminal handoff')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','scientific_diagnostics')}))
