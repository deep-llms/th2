"""Production numerical gate and first40 actual updates, continued via fullstate."""
import argparse,json
from pathlib import Path
import torch
from .model import Config,Model
from .local_policy import LocalRouter,routed_nll,policy_gradient_loss,full_information_loss,LOCAL_ORDERS
from .train import amp,write_json
from .local_train import train


def numerical():
    torch.manual_seed(719)
    c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,prefix=4,length=32)
    model=Model(c);router=LocalRouter(c.d)
    gm=Model(c).cuda();gr=LocalRouter(c.d).cuda()
    gm.load_state_dict(model.state_dict());gr.load_state_dict(router.state_dict())
    x=torch.randint(128,(8,32));choices=torch.arange(8)%4
    def forward(m,r,x,actions):
        logits=r(m.features(x))
        with torch.no_grad():costs=torch.stack([m.nll(x,p) for p in LOCAL_ORDERS],-1)
        values=routed_nll(m,x,actions)
        loss=values.mean()
        head=policy_gradient_loss(logits,actions,values,costs[:,0],.1)+full_information_loss(logits,costs)
        loss.backward();head.backward()
        return float(loss.detach()),torch.cat([p.grad.flatten() for p in m.parameters()]),torch.cat([p.grad.flatten() for p in r.parameters()])
    loss,body,head=forward(model,router,x,choices)
    with amp('cuda'):gl,gb,gh=forward(gm,gr,x.cuda(),choices.cuda())
    be=float((gb.float().cpu()-body).norm()/body.norm());he=float((gh.float().cpu()-head).norm()/head.norm().clamp_min(1e-6))
    if abs(gl-loss)>.03 or be>.06 or he>.15:raise ValueError('BF16 grouped/router gradient gate failed')
    changed=x.clone();changed[:,c.prefix:]=(changed[:,c.prefix:]+1)%c.vocab
    with amp('cuda'):
        torch.testing.assert_close(gr(gm.features(x.cuda())),gr(gm.features(changed.cuda())),atol=0,rtol=0)
    return dict(cpu_loss=loss,gpu_loss=gl,body_relative_gradient_error=be,head_relative_gradient_error=he,causality=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--name',required=True,choices=['CF-s0','PG-s0'])
    a=p.parse_args();base=Path(a.base)
    if torch.cuda.device_count()!=1:raise ValueError('One visible GPU')
    proof=numerical();torch.cuda.empty_cache()
    cfg=json.loads((base/f'inputs/{a.name}.json').read_text());train(cfg,base/f'runs/{a.name}','cuda',stop_after=40)
    proof['first40']=json.loads((base/f'runs/{a.name}/trace.jsonl').read_text().splitlines()[-1])
    proof['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    write_json(base/f'gpu-probe-{a.name}.json',proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
