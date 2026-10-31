"""Independent CPU audit of exported Breadth arrays; does not call production readout."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
root=a.root;load=lambda p:json.loads(p.read_text());decision=load(root/'decision_breadth.json');cfg=load(root/'config/plan.json')
checks=[];errors=[]
def check(name,ok,detail=None):checks.append(dict(test=name,passed=bool(ok),detail=detail))
def close(name,actual,expected,tol=1e-12):
 err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))));errors.append(err);check(name,err<=tol,dict(max_abs_error=err,tolerance=tol))
def mean(row):
 x=np.asarray(row['per_sequence_nll'],dtype=np.float64);w=np.asarray(row.get('per_sequence_tokens',np.ones(len(x))),dtype=np.float64)
 return float(np.sum(x*w)/np.sum(w))
curves={};rows={};files=sorted((root/'perseq').glob('*/*.json'));check('56_arrays',len(files)==56,len(files))
for f in files:
 v=load(f);x=np.asarray(v['per_sequence_nll']);check('finite:'+str(f.relative_to(root)),np.isfinite(x).all());check('shape:'+str(f.relative_to(root)),len(x)==v['sequences'])
 close('mean:'+str(f.relative_to(root)),mean(v),v['loss'])
 rows[(f.parent.name,f.stem)]=v
entries=cfg['downloads']['checkpoints'];blocks=['B_early','B_S1','B_end']
docs=load(root/'documents/documents.json')['documents'];counts=np.array([d['predicted_tokens'] for d in docs]);check('11717_documents',len(docs)==11717);check('document_order',[d['index'] for d in docs]==list(range(len(docs))))
check('truncation',all(1<d['tokens']<=4096 and d['predicted_tokens']==d['tokens']-1 for d in docs))
for ent in entries:
 name=ent['revision'];parent=rows[(name,'parent')];check('document_counts:'+name,np.array_equal(parent['per_sequence_tokens'],counts))
 check('step_tokens:'+name,ent['tokens']==ent['step']*2359296)
 curves[name]={};close('parent:'+name,mean(parent),decision['curves'][name]['parent'])
 for block in blocks:
  v=rows[(name,block)];check('paired_counts:'+name+'/'+block,v['per_sequence_tokens']==parent['per_sequence_tokens']);check('checkpoint:'+name+'/'+block,v['checkpoint']==ent['commit'])
  delta=np.asarray(v['per_sequence_nll'])-parent['per_sequence_nll'];damage=float(np.sum(delta*counts)/counts.sum());curves[name][block]=damage
  close('damage:'+name+'/'+block,damage,decision['curves'][name]['damage'][block])
  expected={'B_early':list(range(3,12)),'B_S1':list(range(24,33)),'B_end':list(range(27,36))}[block];check('removed:'+name+'/'+block,v['removed']==expected)
matched={};rhos={};names=[e['revision'] for e in entries[:9]]
for block in blocks:
 early,late=names[2],names[4]
 x=np.asarray(rows[(early,block)]['per_sequence_nll'])-rows[(early,'parent')]['per_sequence_nll']
 y=np.asarray(rows[(late,block)]['per_sequence_nll'])-rows[(late,'parent')]['per_sequence_nll']
 rng=np.random.default_rng(0);draws=[]
 # Identical paired document draws; independent vectorized estimator.
 for start in range(0,10000,100):
  ix=rng.integers(0,len(x),(100,len(x)));den=np.sum(x[ix]*counts[ix],axis=1);num=np.sum((y-x)[ix]*counts[ix],axis=1);draws.extend((num/den).tolist())
 ci=np.quantile(draws,[.025,.975]);value=float(np.sum((y-x)*counts)/np.sum(x*counts));matched[block]={'value':value,'ci95':ci.tolist()}
 close('bootstrap:'+block,ci,decision['matched_pair'][block]['ci95']);close('matched:'+block,value,decision['matched_pair'][block]['value'])
 damage=np.asarray([curves[n][block] for n in names]);ranks=np.array([1+np.sum(damage<z)+(np.sum(damage==z)-1)/2 for z in damage]);rho=float(np.corrcoef(np.arange(1,10),ranks)[0,1]);rhos[block]=rho;close('spearman:'+block,rho,decision['spearman'][block])
 for name in names:close('relative:'+name+'/'+block,curves[name][block]/curves[names[0]][block]-1,decision['curves'][name]['relative_to_first'][block])
 ea,ez=[e['revision'] for e in entries[-2:]];diff=curves[ez][block]-curves[ea][block]
 close('end_difference:'+block,diff,decision['descriptive_end_pair'][block]['difference']);close('end_relative:'+block,diff/curves[ea][block],decision['descriptive_end_pair'][block]['relative'])
stable={}
for ds in ['dclm','fwe']:
 damage={}
 for arm in ['S72','S188']:
  name=f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3';r=load(root/'heals'/name/'result.json');damage[arm]=mean(rows[(name,'post-S1')])-mean(rows[(name,'parent')])
  close('healed_delta:'+name,damage[arm],r['delta_post']);check('coverage:'+name,r['coverage']=={'examples':48768,'microbatches':1524,'world_size':8,'exact_order':True})
  expected=[.001*(j/16 if j<=16 else (762-j)/(762-16)) for j in range(1,763)];close('LR:'+name,r['lr_updates'],expected,1e-15);check('roundtrip:'+name,r['roundtrip_difference']==0)
  for rank in range(8):
   t=load(root/'heals'/name/f'tests-rank{rank}.json')['checks'];check('rank_checks:'+name+':'+str(rank),all(z['passed'] for z in t))
  order=load(root/'heals'/name/'order.json')['rank_microbatches'];flat=np.asarray(order);check('order_shape:'+name,flat.shape==(8,1524,4));check('unique_rows:'+name,len(np.unique(flat))==48768)
 value=damage['S188']-damage['S72'];stable[ds]={'delta_post':damage,'E_stab':value,'relative':value/damage['S72']};v=decision['stable_healing'][ds]
 for key in ['E_stab','relative']:close(ds+':'+key,stable[ds][key],v[key])
 ref=cfg['annealed_reference'][ds];close(ds+':E_ann',ref['E_ann'],v['E_ann']);close(ds+':E_ann_relative',ref['E_ann']/ref['P300_delta'],v['E_ann_relative']);close(ds+':annealed_vs_stable',ref['P300_delta']-damage['S72'],v['annealed_vs_stable'])
clauses={'SM1':matched['B_S1']['value']>0 and matched['B_S1']['ci95'][0]>0,'SM2':rhos['B_S1']>=.7,'SM3':curves[names[-1]]['B_early']/curves[names[0]]['B_early']>curves[names[-1]]['B_S1']/curves[names[0]]['B_S1'],'SP1':all(v['E_stab']>=.008 for v in stable.values())}
check('clauses',clauses==decision['clauses']);tests=load(root/'tests.json')['checks'];check('scientific_flags_preserved',[t for t in tests if not t['passed']]==decision['flags'])
result={'method':'Independent numpy CPU formulas; no production readout imports; paired document bootstrap10000seed0','checks':checks,'passed':all(c['passed'] for c in checks),'max_abs_error':max(errors),'matched_pair':matched,'spearman':rhos,'stable_healing':stable,'clauses':clauses,'production_check_count':len(tests),'production_flags':decision['flags'],'perseq_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'spec_sha256':decision['spec_sha256']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','max_abs_error','clauses','production_check_count','production_flags']}));print('FAILED',[c for c in checks if not c['passed']])
