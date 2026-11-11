"""Native DOT diagnostics and canonical A2-request likelihood scoring, offline."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool.model import Model,Config
from efficient_llm.dot_pool.data import Stream,e1_batch,write,sha
from efficient_llm.dot_pool.train import autocast,seed


def load(run):
 run=Path(run);cfg=json.loads((run/'config.json').read_text());m=Model(Config(**cfg['model'])).cuda().eval()
 state=torch.load(run/'final_state/state.pt',map_location='cpu',weights_only=False);m.load_state_dict(state['model']);del state
 return run,cfg,m
@torch.no_grad()
def routes(run,cfg,m):
 seed(9182);data=Path(cfg['data']);m.capture=True;out={};study=cfg['study']
 names=['id_dev','ood_dev','id_test','ood_test','ood_bridge','ood_dev_bridge'] if study=='E1' else ['val']
 for name in names:
  rows=np.load(data/(name+'.npy'),mmap_mode='r') if study=='E1' else Stream(data,name)
  n=len(rows) if study=='E1' else min(64,len(rows));ids=[];gates=[];correct=[]
  for start in range(0,n,32):
   ix=np.arange(start,min(start+32,n));inp=e1_batch(rows[ix],'cuda') if study=='E1' else rows.batch(ix,'cuda')
   with autocast(torch.device('cuda')):logits=m(**{k:v for k,v in inp.items() if k!='labels'})
   if m.routed:
    ii=torch.stack([r['ids'] for r in m.routes],dim=2);gg=torch.stack([r['gates'] for r in m.routes],dim=2)
    if study=='E1':
     pos=(inp['labels']!=-100).long().argmax(1)-1;b=torch.arange(len(ix),device='cuda');ii=ii[b,pos];gg=gg[b,pos];correct.extend((logits[b,pos].argmax(-1)==inp['labels'][b,pos+1]).cpu().tolist())
    ids.append(ii.cpu().numpy().astype(np.int16));gates.append(gg.cpu().numpy().astype(np.float32))
   del logits
  if ids:out[name+'_ids']=np.concatenate(ids);out[name+'_gates']=np.concatenate(gates)
  if correct:out[name+'_correct']=np.array(correct)
 if out:np.savez_compressed(run/'routes_sample.npz',**out)
 m.capture=False;m.routes=[]

def overlap(a,b):
 aa=set(a.flatten());bb=set(b.flatten());return len(aa&bb)/len(aa|bb)
def e1_analysis(run):
 path=run/'routes_sample.npz'
 if not path.exists():return
 r=np.load(path);a=r['ood_test_ids'][:,4:8];b=r['ood_bridge_ids'][:,:4];correct=r['ood_test_correct'];rng=np.random.default_rng(0)
 actual=np.array([overlap(x,y) for x,y in zip(a,b)]);chance=np.zeros(len(a))
 for _ in range(100):chance+=np.array([overlap(x,y) for x,y in zip(a,b[rng.permutation(len(b))])])/100
 rows={}
 for name,mask in [('all',np.ones(len(a),bool)),('correct',correct),('wrong',~correct)]:
  v=(actual-chance)[mask]
  if not len(v):rows[name]=dict(n=0,overlap=None,excess=None,ci=None);continue
  draws=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(10000)]);rows[name]=dict(n=len(v),overlap=float(actual[mask].mean()),chance=float(chance[mask].mean()),excess=float(v.mean()),ci=np.quantile(draws,[.025,.975]).tolist())
 write(run/'reentry.json',dict(groups=rows,bootstrap_seed=0,bootstrap_draws=10000,random_repairings=100))

@torch.no_grad()
def e2_shuffle(run,cfg,m):
 data=Stream(cfg['data'],'val');n=min(len(data),1000000//512);sums=np.zeros(4);targets=0;rng=np.random.default_rng(0);m.capture=True
 for start in range(0,n,16):
  ix=np.arange(start,min(n,start+16));inp=data.batch(ix,'cuda')
  with autocast(torch.device('cuda')):out=m(**inp)
  sums[0]+=float(out['nll_sum']);targets+=int(out['count']);original=[(r['ids'],r['gates']) for r in m.routes];m.capture=False
  for j in range(3):
   permutations=np.stack([rng.permutation(m.c.middle) for _ in ix]);forced=[]
   for s in range(m.c.middle):forced.append(tuple(torch.stack([original[int(permutations[b,s])][kind][b] for b in range(len(ix))]) for kind in [0,1]))
   m.forced_routes=forced
   with autocast(torch.device('cuda')):out=m(**inp)
   sums[j+1]+=float(out['nll_sum'])
  m.forced_routes=None;m.capture=True
 m.capture=False;m.routes=[]
 write(run/'shuffle.json',dict(targets=targets,sequences=n,normal_nll=sums[0]/targets,shuffled_nll=(sums[1:]/targets).tolist(),costs=((sums[1:]-sums[0])/targets).tolist(),mean_cost=float((sums[1:].mean()-sums[0])/targets),seed=0))

def e2_reentry(run):
 p=run/'routes_sample.npz'
 if not p.exists():return
 ids=np.load(p)['val_ids'].reshape(-1,20,2);distinct=[];reused=0;counts=np.zeros((160,20),int)
 for row in ids:
  distinct.append(len(set(row.flatten())))
  for s,pair in enumerate(row):
   previous=set(row[:max(0,s-1)].flatten());reused+=sum(int(v in previous) for v in pair)
   for v in pair:counts[v,s]+=1
 totals=counts.sum(1);prob=counts/np.maximum(totals[:,None],1);entropy=-(prob*np.log(np.maximum(prob,1e-30))).sum(1)
 write(run/'reentry.json',dict(sample='first64 validation sequences',tokens=len(ids),mean_distinct=float(np.mean(distinct)),reuse_rate=reused/(len(ids)*40),expert_step_entropy=entropy.tolist()))

@torch.autocast(device_type='cuda',dtype=torch.bfloat16)
def taylor_validity(run,cfg,m):
 from scipy.stats import spearmanr
 data=Stream(cfg['data'],'val');rng=np.random.default_rng(0);rows=[];m.capture=True;seed(9182)
 for seq in range(256):
  inp=data.batch([seq],'cuda');valid=(inp['labels'][0,1:]!=-100).nonzero().flatten()
  if not len(valid):continue
  t=int(valid[int(rng.integers(len(valid)))]);mask=inp['labels'].clone();mask[:,:t+1]=-100;inp['labels']=mask
  with torch.no_grad():base=m(**inp);base_loss=float(base['nll_sum']);trace=list(m.routes)
  # Differentiate only w.r.t. FFN outputs, without parameter updates or Taylor replay.
  for s in [1,9,17]:
   values=trace[s]['x'][0,t:t+1];previous=None if s==0 else (trace[s-1]['ids'][0,t:t+1],trace[s-1]['gates'][0,t:t+1])
   with torch.no_grad():
    logits=m._router_logits(s,values,previous);cand=(logits.sigmoid()+m.beta[0]).topk(16,-1).indices;u=m.pools[0](values,cand)[0]*m.c.k;y=trace[s]['y'][0,t]
   intervention=y.detach().clone().requires_grad_(True);m.intervention=dict(s=s,b=0,t=t,value=intervention)
   out=m(**inp);g=torch.autograd.grad(out['nll_sum'],intervention)[0];scores=-(g.float()[None]*(u.float()-y.float())).sum(-1).detach().cpu().numpy();delta=[]
   with torch.no_grad():
    for v in u:
     m.intervention=dict(s=s,b=0,t=t,value=v);delta.append(float(m(**inp)['nll_sum'])-base_loss)
   m.intervention=None;rho=float(spearmanr(scores,-np.array(delta)).statistic)
   top=int((logits.sigmoid()+m.beta[0]).gather(-1,cand).argmax());rows.append(dict(sequence=seq,token=t,step=s+1,spearman=rho if np.isfinite(rho) else None,top_best=top==int(np.argmin(delta)),delta_L=delta,c=scores.tolist()))
 m.capture=False;m.routes=[]
 valid=[r['spearman'] for r in rows if r['spearman'] is not None]
 write(run/'taylor_validity.json',dict(rows=rows,median_spearman=float(np.median(valid)) if valid else None,top_best_fraction=float(np.mean([r['top_best'] for r in rows])),sign='delta_L = counterfactual minus normal summed task NLL at positions >= t'))

@torch.no_grad()
def downstream(run,cfg,m):
 from eval.a2_request_bridge import digest
 request=run.parent/'requests.json';prepared=json.loads(request.read_text());requests=prepared['requests'];seed(42);scores=[None]*len(requests);order=[];truncated=0
 for i,r in enumerate(requests):
  context=r['context'] or [0];target=r['continuation'];tokens=context+target
  if len(target)>512:raise ValueError('Continuation exceeds512')
  truncated+=len(tokens)>513;order.append((i,tokens[-513:][:-1],target))
 order.sort(key=lambda x:-len(x[1]))
 for off in range(0,len(order),8):
  batch=order[off:off+8];ids=torch.zeros((len(batch),len(batch[0][1])),device='cuda',dtype=torch.long)
  for b,(_,tokens,_) in enumerate(batch):ids[b,:len(tokens)]=torch.tensor(tokens,device='cuda')
  with autocast(torch.device('cuda')):logits=m(ids)
  for b,(i,tokens,target) in enumerate(batch):
   l=logits[b,len(tokens)-len(target):len(tokens)].float();wanted=torch.tensor(target,device='cuda');logp=l.log_softmax(-1).gather(-1,wanted[:,None]).sum();scores[i]=[float(logp),bool(torch.equal(l.argmax(-1),wanted))]
 write(run/'downstream_scores.json',dict(scores=scores,likelihood_requests=len(scores),truncated_requests=truncated,precision='fp32_parameters_bf16_cuda_autocast',max_length=512,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256']))
 argv=['/mnt/local/conda-py311/envs/eval/bin/python3.11','-m','eval.a2_requests','replay','--tokenizer','/mnt/local/_models/deep-llms_th2/a2/dclm/P300','--dataset-root','/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023','--manifest','resources/downstream_english_20261023.json','--tasks','hellaswag','piqa','arc_easy','lambada_openai','--max-length','512','--prepared',str(request),'--scored',str(run/'downstream_scores.json'),'--output',str(run/'downstream_raw.json')]
 subprocess.run(argv,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
 raw=json.loads((run/'downstream_raw.json').read_text());primary={t:raw['results'][t][('acc_norm,none' if t in ['hellaswag','piqa'] else 'acc,none')] for t in ['hellaswag','piqa','arc_easy','lambada_openai']}
 write(run/'downstream.json',dict(metrics=primary,protocol='canonical A2 13686ed requests/replay; lm-eval0.4.13; context512; zero-shot',requests_sha256=sha(request)))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['routes','analysis']);p.add_argument('--run',required=True);a=p.parse_args();run,cfg,m=load(a.run);torch.set_num_threads(2)
 if a.mode=='routes':routes(run,cfg,m);return
 flags=[]
 funcs=[lambda:e1_analysis(run)] if cfg['study']=='E1' else [lambda:downstream(run,cfg,m)]
 if cfg['study']=='E2' and cfg['arm'] in ['SO','DOT','SR']:funcs+=[lambda:e2_reentry(run)]
 if cfg['study']=='E2' and cfg['arm'] in ['SO','DOT']:funcs+=[lambda:e2_shuffle(run,cfg,m)]
 if cfg['study']=='E2' and cfg['arm']=='DOT':funcs+=[lambda:taylor_validity(run,cfg,m)]
 for fn in funcs:
  try:fn()
  except Exception as e:
   import traceback;flags.append(dict(error=repr(e),trace=traceback.format_exc()));m.intervention=None;m.forced_routes=None;m.capture=False
 write(run/'analysis_complete.json',dict(flags=flags,complete=not flags))
if __name__=='__main__':main()
