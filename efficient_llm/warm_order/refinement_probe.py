"""Real first40 production updates after CPU/FP32 versus BF16 gradient gate."""
import argparse,json
from pathlib import Path
import torch
from .model import Config,Model
from .refinement import Refinement
from .refinement_train import train,update
from .refinement_eval import probe
from .token_router import precision
from .train import amp,write_json


def numerical(device='cuda'):
    precision();torch.manual_seed(719);torch.set_num_threads(1)
    c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,prefix=4,length=32);proof={}
    for arm in ('F7','J7'):
        m=Refinement(Model(c),frozen=arm=='F7');g=Refinement(Model(c),frozen=arm=='F7').to(device);g.load_state_dict(m.state_dict());x=torch.randint(128,(4,32))
        # Nonzero output projections exercise gradients beyond identity initialization.
        with torch.no_grad():
            m.extra.self_attn.o_proj.weight.normal_(0,.02);m.extra.mlp.down_proj.weight.normal_(0,.02)
        g.load_state_dict(m.state_dict())
        o=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=0);go=torch.optim.SGD([p for p in g.parameters() if p.requires_grad],lr=0)
        a=update(m,x,arm,o)
        with amp(device):b=update(g,x.to(device),arm,go)
        grad=torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]);gg=torch.cat([p.grad.flatten().float().cpu() for p in g.parameters() if p.grad is not None]);err=float((gg-grad).norm()/grad.norm())
        if abs(a['loss']-b['loss'])>.03 or err>.08:raise ValueError('BF16 gradient gate failed')
        proof[arm]=dict(cpu_loss=a['loss'],gpu_loss=b['loss'],relative_gradient_error=err,fp32_probe=probe(g.eval(),x.to(device)))
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--name',choices=['F7-s0','J7-s0'],required=True);a=p.parse_args();base=Path(a.base)
    if torch.cuda.device_count()!=1:raise ValueError('One GPU required')
    proof=numerical();torch.cuda.empty_cache();c=json.loads((base/f'inputs/{a.name}.json').read_text());train(c,base/f'runs/{a.name}',stop_after=40)
    proof['first40']=json.loads((base/f'runs/{a.name}/progress.json').read_text());proof['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    write_json(base/f'gpu-probe-{a.name}.json',proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
