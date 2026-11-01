from pathlib import Path
import json,sys
import numpy as np
base=Path(sys.argv[1]);out=Path(sys.argv[2])
saved=json.loads((base/'decision_depth.json').read_text());parents=['S72','P300','S188','P1T']
sets=[*[f'W{i}' for i in range(1,10)],'S2','S3','S4'];spec='2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe'
errors=[];nchecks=0

def equal(a,b,name):
 global nchecks
 nchecks+=1
 if a is None or b is None:assert a is b,name;return
 if isinstance(a,(bool,str)) or isinstance(b,(bool,str)):assert a==b,(name,a,b);return
 delta=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors.append(delta)
 assert delta<1e-10,(name,delta)

def ranks(v):
 v=np.asarray(v);return np.array([sum(v<x)+(sum(v==x)+1)/2 for x in v])

def rho(v):
 a=ranks(range(9));b=ranks(v)
 return float(np.corrcoef(a,b)[0,1]) if b.std() else None
all_tables={};rows=0
for ds in ['dclm','fwe','nem']:
 arrays={}
 for st in ['parent']+sets:
  cols=[]
  for parent in parents:
   row=json.loads((base/'part_a'/ds/parent/(st+'.json')).read_text());arr=np.asarray(row['per_sequence_nll'],dtype=np.float64)
   assert row['spec_sha256']==spec and arr.shape==(2461,) and np.isfinite(arr).all()
   assert row['tokens']==10077795 and row['sequences']==2461
   equal(arr.mean(),row['loss'],f'{ds}/{parent}/{st}/mean');rows+=1;cols.append(arr)
  arrays[st]=np.stack(cols,axis=1)
 # Draw batches from one seeded generator, independently of production readout.
 rng=np.random.default_rng(0);samples=[]
 for i in range(100):samples.append(rng.multinomial(2461,np.ones(2461)/2461,size=100))
 tables={}
 for st in sets:
  d=arrays[st]-arrays['parent'];means=d.mean(0);table=saved['datasets'][ds][st];tables[st]={}
  for j,parent in enumerate(parents):
   for key,value in [('parent_nll',arrays['parent'][:,j].mean()),('compressed_nll',arrays[st][:,j].mean()),('damage',means[j]),('relative_damage',means[j]/arrays['parent'][:,j].mean())]:equal(value,table['damage'][parent][key],ds+st+parent+key)
  draw=np.concatenate([count.astype(np.float64)@d/2461 for count in samples])
  pairs={'L_ann':(3,1,1),'L_stab':(2,0,0),'R_short':(1,0,0),'R_long':(3,2,2)}
  for c,(a,b,r) in pairs.items():
   v=means[a]-means[b];num=draw[:,a]-draw[:,b];den=draw[:,r];mask=np.isfinite(den)&(np.abs(den)>1e-15);ci=np.quantile(num,[.025,.975]);rel=v/means[r] if means[r] else None
   row=table['contrasts'][c]
   for key,value in [('estimate',v),('ci95',ci),('reference_damage',means[r]),('relative_to_reference_damage',rel),('relative_ci95',np.quantile(num[mask]/den[mask],[.025,.975])),('reference_damage_ci95',np.quantile(den,[.025,.975])),('relative_undefined_draws',int((~mask).sum()))]:equal(value,row[key],ds+st+c+key)
   if c!='R_long':
    T=max(.003,.1*means[r]);t=max(.003,.03*means[r]);excludes=ci[0]>0 or ci[1]<0
    label='more damage' if v>=T and excludes else 'less damage' if v<=-T and excludes else 'no effect' if abs(v)<t else 'grey'
    equal(label,row['label'],ds+st+c+'label')
   tables[st][c]={'estimate':float(v),'relative_to_reference_damage':float(rel)}
  equal(means[0]-means[1]-means[2]+means[3],table['contrasts']['I']['estimate'],ds+st+'I')
  equal(np.quantile(draw[:,0]-draw[:,1]-draw[:,2]+draw[:,3],[.025,.975]),table['contrasts']['I']['ci95'],ds+st+'Ici')
 for scale in ['estimate','relative_to_reference_damage']:
  for c in ['L_ann','L_stab','R_short']:equal(rho([tables[f'W{i}'][c][scale] for i in range(1,10)]),saved['correlations'][ds][scale][c],ds+scale+c+'rho')
 all_tables[ds]=tables
 print('REPLAY_DATASET_PASS',ds,flush=True)
for scale,key in [('estimate','predictions_nats'),('relative_to_reference_damage','predictions_relative')]:
 clauses={}
 for ds,t in all_tables.items():
  def c(w,n):return t[w][n][scale]
  clauses[ds]=dict(A1_W9_gt_W8=c('W9','R_short')>c('W8','R_short'),A1_W8_gt_W1=c('W8','R_short')>c('W1','R_short'),A2_L_ann=c('W1','L_ann')>c('W8','L_ann'),A2_L_stab=c('W1','L_stab')>c('W8','L_stab'),A3_negative=c('W1','R_short')<0,A4=c('W9','R_short')>=c('W9','L_stab'))
  assert clauses[ds]==saved[key]['per_dataset'][ds]
 preds=dict(A1=all(c['A1_W9_gt_W8'] and c['A1_W8_gt_W1'] for c in clauses.values()),A2=all(c['A2_L_ann'] and c['A2_L_stab'] for c in clauses.values()),A3=sum(c['A3_negative'] for c in clauses.values())>=2,A4=all(c['A4'] for c in clauses.values()))
 assert preds==saved[key]['predictions']
result=dict(spec_sha256=spec,status='PASS',saved_arrays=rows,comparisons=nchecks,max_absolute_difference=max(errors),resamples=10000,seed=0,method='Independent NumPy batch-count replay of raw arrays, explicit contrast pairs/labels/clauses and tied rank correlations; no production statistics/readout imports',predictions_nats=saved['predictions_nats'],predictions_relative=saved['predictions_relative'])
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
