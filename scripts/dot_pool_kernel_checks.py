"""Actual B200 forward/backward checks for grouped experts and FA4."""
import argparse,copy,dataclasses,json,traceback
import torch
from efficient_llm.dot_pool.model import Config,Model,Experts
from efficient_llm.dot_pool.data import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();checks=[];torch.manual_seed(71)
 def attempt(name,fn):
  try:detail=fn();ok=detail.pop('passed')
  except Exception as e:ok=False;detail=dict(error=repr(e),trace=traceback.format_exc())
  checks.append(dict(test=name,passed=ok,detail=detail));write(a.output,dict(passed=all(x['passed'] for x in checks),checks=checks));print(checks[-1],flush=True)
 def grouped():
  c=Config.production('E1','SO',expert_kernel='grouped');pool=Experts(c.pool,c).cuda();ref=copy.deepcopy(pool);ref.c=dataclasses.replace(c,expert_kernel='loop');x=torch.randn(128,c.d,device='cuda');ids=torch.randint(c.pool,(128,2),device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):y=pool(x,ids);z=ref(x,ids)
  y.float().square().mean().backward();z.float().square().mean().backward();err=float((y-z).float().norm()/z.float().norm());grad=max(float((u.grad-v.grad).norm()/v.grad.norm().clamp_min(1e-10)) for u,v in zip(pool.parameters(),ref.parameters()));return dict(passed=err<.02 and grad<.03,relative_output=err,relative_grad=grad)
 def attention():
  c=Config.production('E2','D',backend='fa4');c=dataclasses.replace(c,middle=2,stem=1,coda=1,vocab=128);m=Model(c).cuda();ref=copy.deepcopy(m)
  for module in ref.modules():
   if hasattr(module,'c'):module.c=dataclasses.replace(module.c,backend='sdpa')
  x=torch.randint(1,128,(2,512),device='cuda');seg=torch.arange(512,device='cuda')[None].expand_as(x)//64;pos=torch.arange(512,device='cuda')[None].expand_as(x)%64
  with torch.autocast('cuda',dtype=torch.bfloat16):y=m(x,segments=seg,positions=pos);z=ref(x,segments=seg,positions=pos)
  y.float().square().mean().backward();z.float().square().mean().backward();error=float((y-z).float().norm()/z.float().norm());grad=float((m.embed_tokens.weight.grad-ref.embed_tokens.weight.grad).norm()/ref.embed_tokens.weight.grad.norm());return dict(passed=error<.02 and grad<.03,relative_output=error,relative_grad=grad)
 attempt('grouped_forward_backward',grouped);torch.cuda.empty_cache();attempt('fa4_document_forward_backward',attention)
if __name__=='__main__':main()
