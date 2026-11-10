"""Two CPU/Gloo ranks: chunked CE DDP accumulation equals global CE update."""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from test_family_f11 import native
from efficient_llm.a2_family_f11_loss import LMModel
from efficient_llm.a2.model import prune

dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==2
m=native();prune(m,[1,2]);ref=native();prune(ref,[1,2]);ddp=DDP(LMModel(m,2))
opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.1)
reference=torch.optim.AdamW(ref.parameters(),lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.1)
ids=torch.tensor([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])
for micro in range(2):
    with ddp.no_sync() if micro==0 else nullcontext():
        loss=ddp(ids[micro*2+rank:micro*2+rank+1])['loss']/2;loss.backward()
logits=ref(input_ids=ids,use_cache=False).logits[:,:-1]
loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].reshape(-1));loss.backward()
for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,r.grad,rtol=2e-5,atol=3e-6)
opt.step();reference.step()
for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,r,rtol=2e-5,atol=3e-6)
dist.barrier()
if rank==0:print('F11_TWO_RANK_CPU_DDP_CE_GLOBAL_UPDATE_PASS',flush=True)
dist.destroy_process_group()
