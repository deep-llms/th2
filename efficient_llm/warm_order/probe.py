"""Real GPU numerical gate, then first40 steps of the actual R-s0 schedule."""
import argparse,json
from pathlib import Path
import torch
from .model import Config,Model,ORDERS
from .train import amp,train,write_json

def numerical():
    if torch.cuda.device_count()!=1:raise ValueError('One visible probe GPU required')
    torch.manual_seed(591);c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,prefix=4,length=32)
    ref=Model(c);gpu=Model(c).cuda();gpu.load_state_dict(ref.state_dict());x=torch.randint(128,(3,32))
    loss=ref.nll(x,ORDERS[5]).mean();loss.backward();rg=torch.cat([p.grad.flatten() for p in ref.parameters()])
    with amp('cuda'):gl=gpu.nll(x.cuda(),ORDERS[5]).mean()
    gl.backward();gg=torch.cat([p.grad.float().cpu().flatten() for p in gpu.parameters()]);error=float((gg-rg).norm()/rg.norm())
    if abs(float(gl)-float(loss))>.03 or error>.05:raise ValueError('GPU BF16/reference mismatch')
    gpu.eval();z=x.clone();z[:,4:]=(z[:,4:]+1)%128
    with amp('cuda'):
        torch.testing.assert_close(gpu.features(x.cuda()),gpu.features(z.cuda()),rtol=0,atol=0)
        a=gpu.hidden(x.cuda(),ORDERS[0])[:,:4];b=gpu.hidden(z.cuda(),ORDERS[0])[:,:4]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    return dict(cpu_loss=float(loss),gpu_bf16_loss=float(gl),relative_gradient_error=error,prefix_causality=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    proof=numerical();torch.cuda.empty_cache()
    cfg=json.loads((base/'inputs/R-s0.json').read_text())
    train(cfg,base/'runs/R-s0','cuda',stop_after=40)
    trace=json.loads((base/'runs/R-s0/trace.jsonl').read_text().splitlines()[-1]);proof['first40']=trace
    proof['rough_full_fit_seconds']=trace['elapsed_seconds']/40*cfg['steps']
    proof['peak_allocated_bytes']=torch.cuda.max_memory_allocated();write_json(base/'gpu-probe.json',proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
