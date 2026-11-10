"""Offline independent replay from saved native likelihoods and canonical samples.
Does not import production metric/collector/bootstrap helpers or load a model.
"""
import json,hashlib,math,argparse
from pathlib import Path
from collections import Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();b=a.root;checks=[]
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def ck(n,v):
 checks.append(dict(test=n,passed=bool(v)))
 if not v:raise ValueError(n)
models=['P300-parent','P1T-parent','S72-parent','P300-student','P1T-student','X72-student']
counts=dict(hellaswag=10042,arc_easy=2376,arc_challenge=1172,piqa=1838,winogrande=1267,boolq=3270,openbookqa=500,sciq=1000,lambada_openai=5153)
norm={'hellaswag','arc_easy','arc_challenge','piqa','openbookqa'}
weights=np.array([[1,0,0,-1,0,0],[0,1,0,0,-1,0],[0,0,1,0,0,-1],[-1,1,0,1,-1,0],[0,0,0,-1,0,1],[-1,0,1,1,0,-1]],float).T
contrasts=['P300_drop','P1T_drop','X72_drop_from_S72','P1T_minus_P300_drop','X72_minus_P300_accuracy','X72_minus_P300_drop']
d=read(b/'downstream/complete.json');prep=read(b/'dependencies/prepared.json');ck('prepared_request_digest',prep['request_sha256']==digest(prep['requests']));protocol=prep['protocol'];ck('full_protocol',set(protocol['tasks'])==set(counts) and protocol['max_length']==4096 and protocol['num_fewshot']==0 and protocol['seed']==42 and protocol['limit'] is None)
arrays={t:{} for t in counts};prompts={};summary={}
for model in models:
 scorefile=b/'dependencies/scores'/(model+'.json');score=read(scorefile);r=read(b/'dependencies/replays'/(model+'.json'));fresh=read(b/'dependencies/independent_replays'/(model+'.json'))
 ck(model+':score_lock',hashlib.sha256(scorefile.read_bytes()).hexdigest()==d['scores'][model]['sha256'])
 ck(model+':identity',score['prepared_sha256']==digest(prep) and score['request_sha256']==prep['request_sha256'] and score['batch_size']==2 and len(score['scores'])==len(prep['requests']))
 ck(model+':replay_binding',r['a2_metadata']['scored_file_sha256']==d['scores'][model]['sha256'] and r['a2_metadata']['protocol']==protocol and r['a2_metadata']['prepared_sha256']==digest(prep))
 # Match every saved sample response to the native likelihood request, including multiplicities.
 native=Counter((tuple(req['text']),float(value[0]),bool(value[1])) for req,value in zip(prep['requests'],score['scores']))
 actual=Counter();summary[model]={}
 for task,n in counts.items():
  rows=sorted(r['samples'][task],key=lambda x:x['doc_id']);frows=sorted(fresh['samples'][task],key=lambda x:x['doc_id']);ck(model+task+':coverage',len(rows)==n and len({x['doc_id'] for x in rows})==n)
  ident=digest([(x['doc_id'],x['doc'],x['arguments'],x['target']) for x in rows]);ck(model+task+':paired_prompts',prompts.setdefault(task,ident)==ident)
  computed=[]
  for x in rows:
   resp=x['filtered_resps'];args=x['arguments'];ckey='acc_norm' if task in norm else 'acc'
   for arg,v in zip(args,resp):actual[(tuple(arg),float(v[0]),bool(v[1]))]+=1
   if task=='lambada_openai':value=int(resp[0][1])
   else:
    gold=int(x['doc']['answer'])-1 if task=='winogrande' else int(x['target']);ll=np.array([v[0] for v in resp]);
    if task in norm:
     if not all(v[1].startswith(' ') for v in args):raise ValueError('Unexpected choice delimiter')
     ll=ll/np.array([len(v[1])-1 for v in args])
    value=int(int(np.argmax(ll))==gold)
   if value!=x[ckey]:raise ValueError(f'{model}/{task}/{x["doc_id"]}: primary accuracy mismatch')
   computed.append(value)
  ck(model+task+':fresh_sample_accuracy',computed==[x[ckey] for x in frows]);values=np.array(computed);arrays[task][model]=values
  acc=float(values.mean());ck(model+task+':aggregate',abs(acc-r['results'][task][ckey+',none'])<1e-12);summary[model][task]=acc
 ck(model+':native_response_coverage',native==actual)
 summary[model]['MC_mean']=sum(summary[model][t] for t in counts if t!='lambada_openai')/8
rng=np.random.default_rng(0);mcpoint=np.zeros(6);mcdraws=np.zeros((10000,6));results={}
def report(point,draws):
 ci=np.quantile(draws,[.025,.975],axis=0);return {n:dict(estimate=float(point[i]),ci95=ci[:,i].tolist()) for i,n in enumerate(contrasts)}
for t in sorted(counts):
 mat=np.stack([arrays[t][m] for m in models],axis=1);patterns,frequency=np.unique(mat,axis=0,return_counts=True);n=len(mat)
 draws=rng.multinomial(n,frequency/n,size=10000)@(patterns@weights)/n;point=mat.mean(0)@weights
 results[t]=report(point,draws)
 for name,v in results[t].items():ck(t+name+':pairedCI',np.allclose([v['estimate'],*v['ci95']],[d['paired']['tasks'][t]['contrasts'][name]['estimate'],*d['paired']['tasks'][t]['contrasts'][name]['ci95']],rtol=0,atol=1e-12))
 if t!='lambada_openai':mcpoint+=point/8;mcdraws+=draws/8
mc=report(mcpoint,mcdraws)
for n,v in mc.items():ck(n+':MC_CI',np.allclose([v['estimate'],*v['ci95']],[d['paired']['mc_mean'][n]['estimate'],*d['paired']['mc_mean'][n]['ci95']],rtol=0,atol=1e-12))
ck('native_fresh_paired_exact',d['paired']==d['independent_paired'])
out=dict(verification_passed=True,checks=checks,passed=len(checks),failed=0,summary=summary,MC_contrasts=mc,task_contrasts=results,flags=d['flags'],scope='Independent native-score-to-response coverage, primary per-item accuracy, aggregate, pairing and paired bootstrap 10000 seed0. No new model inference or retokenization. Benchmark-item uncertainty only.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','summary','MC_contrasts','flags']},indent=2))
