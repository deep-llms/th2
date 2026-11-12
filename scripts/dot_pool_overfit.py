"""T7 production architectures and specified fixed batches, offline GPU only."""
import argparse,dataclasses,json,time
from pathlib import Path
import numpy as np,torch
from efficient_llm.dot_pool.model import Config,Model
from efficient_llm.dot_pool.data import e1_batch,Stream,write
from efficient_llm.dot_pool.train import seed,autocast

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--fallback',action='store_true');p.add_argument('--no-stem-coda',action='store_true');a=p.parse_args();torch.set_num_threads(2);device=torch.device('cuda');records=[]
 data=Path(a.data)/a.study.lower();inp=e1_batch(np.load(data/'train.npy')[:256],device) if a.study=='E1' else Stream(data,'train').batch(np.arange(8),device)
 for lr in [.001,.003]:
  seed(71);c=Config.production(a.study,a.arm,backend='sdpa' if a.fallback else 'fa4',expert_kernel='loop' if a.fallback else 'grouped',**({'stem':0,'coda':0} if a.no_stem_coda else {}));m=Model(c).to(device);opt=torch.optim.AdamW(m.parameter_groups(),lr=lr,betas=(.9,.95),eps=1e-8,fused=True);initial=None;targets=int((inp['labels'][:,1:]!=-100).sum());t0=time.monotonic();history=[]
  for step in range(300):
   opt.zero_grad(set_to_none=True);m.gradient_factor=targets
   with autocast(device):out=m(**inp);loss=out['nll_sum']/targets+out['balance_loss']+out['taylor_loss']
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.finish_update();m.promote_labels();value=float(out['nll_sum'].detach()/targets);initial=value if initial is None else initial
   if step%10==0:history.append(dict(step=step+1,nll=value))
   if value<(.5*initial if a.arm=='SR' else .1):break
  passed=value<(.5*initial if a.arm=='SR' else .1);records.append(dict(lr=lr,initial=initial,final=value,steps=step+1,passed=passed,seconds=time.monotonic()-t0,history=history))
  write(a.output,dict(test='T7-'+a.study+'-'+a.arm,passed=passed,attempts=records,batch=list(inp['tokens'].shape),config=dataclasses.asdict(c)))
  del opt,m,out,loss;torch.cuda.empty_cache()
  if passed:break
if __name__=='__main__':main()
