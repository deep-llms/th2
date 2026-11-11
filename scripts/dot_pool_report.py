"""Recompute compact pilot tables/readings from actual results, retain every flag."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.data import write,sha
from efficient_llm.dot_pool import SPEC_SHA

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
 flags=[];tables={'E1':[],'E2':[]};details=[];downstream=[];diagnostics={k:[] for k in ['shuffle','reentry','usage','taylor_validity']}
 for study,arms in [('E1',['D','L','SO','DOT','SR']),('E2',['D','PL','SO','DOT','SR'])]:
  for arm in arms:
   for seed in [0,1]:
    path=root/f'{study}-{arm}-s{seed}';final=path/'final_eval.json'
    if not final.exists():flags.append(f'Missing result: {path.name}');continue
    cfg=json.loads((path/'config.json').read_text());f=json.loads(final.read_text());train=json.loads((path/'train_complete.json').read_text());metrics=f['metrics'];row=dict(arm=arm,seed=seed)
    if study=='E1':
     row.update({name:metrics[name]['accuracy'] for name in ['id_test','ood_test','train','atoms','ood_atoms']});row['epochs']=cfg['steps']/77;cross=[]
     for ep in path.glob('eval-*.json'):
      z=json.loads(ep.read_text())
      if z['metrics'].get('ood_dev',{}).get('accuracy',0)>=.5:cross.append(z['step']/77)
     row['epoch_OOD50']=min(cross) if cross else 'never'
    else:
     row['final_nll']=metrics['val']['nll'];pre=path/f"eval-{int(cfg['steps']*.8)}.json";row['pre_decay_nll']=json.loads(pre.read_text())['metrics']['val']['nll'] if pre.exists() else None;row['tokens']=cfg['steps']*262144;row['tokens_per_second']=row['tokens']/train['elapsed']
    row.update(seconds=train['elapsed'],**cfg['counts']);tables[study].append(row);flags.extend([f'{path.name}: {x}' for x in f['flags']])
    analysis=path/'analysis_complete.json'
    if not analysis.exists():flags.append(path.name+': analysis not complete')
    else:flags.extend([f'{path.name}: {x}' for x in json.loads(analysis.read_text())['flags']])
    if (path/'downstream.json').exists():downstream.append(dict(arm=arm,seed=seed,**json.loads((path/'downstream.json').read_text())['metrics']))
    for name in diagnostics:
     if (path/(name+'.json')).exists():diagnostics[name].append(dict(study=study,arm=arm,seed=seed,**json.loads((path/(name+'.json')).read_text())))
 for key,rows in {**tables,'downstream':downstream}.items():
  if rows:
   with (root/(key.lower()+'_summary.csv' if key in tables else key+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for name,rows in diagnostics.items():
  write(root/(name+'_summary.json'),dict(rows=rows))
  if rows:
   keys=sorted(set().union(*(r.keys() for r in rows)))
   with (root/(name+'.csv')).open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
 readings=[]
 def means(study,metric):
  out={}
  for arm in {x['arm'] for x in tables[study]}:
   rows=sorted([x for x in tables[study] if x['arm']==arm],key=lambda x:x['seed'])
   if len(rows)==2:values=[r[metric] for r in rows];out[arm]=(float(np.mean(values)),abs(values[0]-values[1]))
  return out
 e1=means('E1','ood_test');id1=means('E1','id_test');e2=means('E2','final_nll')
 def compare(name,values,a,b,threshold,label_pos,label_neg,neutral):
  if a not in values or b not in values:readings.append(dict(name=name,label='incomplete'));return
  diff=values[a][0]-values[b][0];label=label_pos if diff>=threshold else label_neg if diff<=-threshold else neutral
  readings.append(dict(name=name,difference=diff,label=label,within_seed_noise=abs(diff)>=threshold and abs(diff)<2*max(values[a][1],values[b][1])))
 if all(k in e1 for k in ['D','L']):
  good=id1['D'][0]>=.9 and e1['D'][0]<.2 and e1['L'][0]>=e1['D'][0]+.2;readings.append(dict(name='E1-R0',label='setup consistent' if good else 'setup flag'))
  if e1['L'][0]<e1['D'][0]+.2:flags.append('E1 inconclusive: composition did not emerge within budget')
 compare('E1-R1',e1,'SO','L',.1,'better','worse','matches')
 compare('E1-R2',e1,'DOT','SO',.1,'helps','hurts','no clear difference')
 for arm in ['SO','DOT']:compare('E1-R3-'+arm,e1,arm,'SR',.1,'learned routing matters','random better','no clear difference')
 if 'D' in e2 and 'PL' in e2:readings.append(dict(name='E2-R0',label='consistent' if e2['PL'][0]<=e2['D'][0]-.02 else 'MoE implementation or recipe suspect'))
 compare('E2-R1',e2,'PL','SO',.01,'sharing helps','sharing hurts','no clear difference')
 if 'SO' in e2 and 'DOT' in e2:
  d=e2['SO'][0]-e2['DOT'][0];label='router helps' if d>=.01 else 'small' if d>=.003 else 'no effect' if abs(d)<.003 else 'small, worse' if d>-.01 else 'router hurts';readings.append(dict(name='E2-R2',difference=d,label=label,within_seed_noise=abs(d)>=.003 and abs(d)<2*max(e2['SO'][1],e2['DOT'][1])))
 for arm in ['SO','DOT']:compare('E2-R3-'+arm,e2,'SR',arm,.01,'learned routing helps','random better','no clear difference')
 for r in diagnostics['shuffle']:readings.append(dict(name=f"E2-R4-{r['arm']}-s{r['seed']}",cost=r['mean_cost'],label='order matters' if r['mean_cost']>=.01 else 'order matters little'))
 for r in diagnostics['taylor_validity']:readings.append(dict(name=f"E2-R5-{r['arm']}-s{r['seed']}",median=r['median_spearman'],label='informative' if r['median_spearman'] is not None and r['median_spearman']>=.3 else 'weak'))
 for r in diagnostics['reentry']:
  if r['study']=='E1':
   ci=r['groups']['all']['ci'];readings.append(dict(name=f"E1-R4-{r['arm']}-s{r['seed']}",label='re-entry observed' if ci and ci[0]>0 else 'not established',ci=ci))
 for name in ['tests.json','kernel_tests.json']:
  p=root/name
  if p.exists():flags.extend([f"{name}: {x['test']} failed: {x['detail']}" for x in json.loads(p.read_text())['checks'] if not x['passed']])
 if (root/'status.json').exists():flags.extend(str(x) for x in json.loads((root/'status.json').read_text()).get('flags',[]))
 flags.append('Tier2/3 checkpoints not yet off-node; only owner-approved large-transfer mechanism may clear this flag.')

 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  for study in tables:
   fig,axes=plt.subplots(1,2,figsize=(12,4))
   for r in tables[study]:
    path=root/f"{study}-{r['arm']}-s{r['seed']}";data=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()];axes[0].plot([z['step'] for z in data],[z['nll'] for z in data],label=path.name)
    es=sorted([json.loads(p.read_text()) for p in path.glob('eval-*.json')],key=lambda x:x['step']);axes[1].plot([z['step'] for z in es],[z['metrics']['ood_dev']['accuracy'] if study=='E1' else z['metrics']['val']['nll'] for z in es],label=path.name)
   axes[0].set_title('Training NLL');axes[1].set_title('OOD-dev accuracy' if study=='E1' else 'Validation NLL');axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(root/(study.lower()+'_curves.png'));plt.close(fig)
 except Exception as e:flags.append('Plot failure: '+repr(e))
 text=['# DOT-pool pilot v1.2',f'Spec SHA-256: `{SPEC_SHA}`','',f'Coverage: E1 {len(tables["E1"])}/10; E2 {len(tables["E2"])}/10.','', '## Flags','']+['- '+x for x in flags]+['','## Budget and runtime','']
 if (root/'budget.json').exists():text+=['```json',(root/'budget.json').read_text(),'```']
 for study,rows in tables.items():
  text+=['',f'## {study}','']
  if rows:
   keys=list(rows[0]);text+=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(str(row[k]) for k in keys)+' |' for row in rows]
 text+=['','## Registered readings','']+['- '+json.dumps(r,ensure_ascii=False) for r in readings]
 text+=['','## Artifacts','','Per-run artifact_manifest.json lists paths, sizes, hashes, locations and verification times. manifest.json indexes compact Tier1 exports. Native checkpoint states remain on node until an approved large transfer.','', 'Downstream reference: 13686ed is the A2 project downstream commit, not a lm-eval upstream revision; reused canonical lm-eval0.4.13 request/replay bridge, context512.','Taylor gradient comes from the main backward with summed-task scaling; the accepted auxiliary-balance contribution is retained.']
 extended=[]
 for path in sorted(root.glob('E1-*-extended')):
  if (path/'final_eval.json').exists():
   cfg=json.loads((path/'config.json').read_text());final=json.loads((path/'final_eval.json').read_text());extended.append(dict(arm=cfg['arm'],seed=cfg['seed'],epochs=cfg['steps']/77,**{k:v['accuracy'] for k,v in final['metrics'].items()}))
 if extended:
  write(root/'e1_extended.json',dict(rows=extended));text+=['','## Conditional Phase4 endpoint','','Original E1 tables/readings above are preserved. Extended endpoints:','```json',json.dumps(extended,indent=2),'```']
 text+=['','## Environment and timings','','Hardware: eight NVIDIA B200. Training uses FP32 parameters and BF16 CUDA autocast. E2 DDP world size8; E1 one rank per run. torch.compile disabled: dynamic routing shapes/hooks retain eager execution; measured throughput governs budgets.']
 for path in [root/'phases.jsonl',root/'phase4.json']:
  if path.exists():text+=['',path.name,'```',path.read_text(),'```']
 runtime=[]
 for path in sorted(root.glob('E[12]-*/config.json')):
  cfg=json.loads(path.read_text());runtime.append(dict(run=path.parent.name,runtime=cfg.get('runtime'),attention=cfg['model'].get('backend','sdpa'),experts=cfg['model'].get('expert_kernel','loop')))
 text+=['','```json',json.dumps(runtime,indent=2),'```','E2 dense forward FLOPs/token at sequence512:215M; router overhead at most3.3M/token; theoretical Taylor forward overhead approximately7%, measured speed is in budget.json.']
 write(root/'readout.json',dict(tables=tables,extended=extended,downstream=downstream,readings=readings,flags=flags))
 (root/'REPORT.md').write_text('\n'.join(text)+'\n')
 write(root/'report_complete.json',dict(coverage=dict(E1=len(tables['E1']),E2=len(tables['E2'])),report_sha256=sha(root/'REPORT.md'),flags=flags))
if __name__=='__main__':main()
