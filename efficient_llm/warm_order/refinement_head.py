"""Two-action specialization of the tested token-head exact-resume trainer."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .refinement import ExitRouter
from .token_router import expected_loss
from .data import sha
from .train import write_json

def fit_head(features, costs, valid_features, valid_costs, seed, out, device,
             epochs=8, batch=4096, stop_after=None, resume=False):
    """Fullstate epoch-boundary resume; private permutation generator and fixed schedule."""
    out=Path(out)
    def digest(array):
        raw=memoryview(np.ascontiguousarray(array)).cast('B');h=hashlib.sha256()
        for i in range(0,len(raw),8*1024*1024):h.update(raw[i:i+8*1024*1024])
        return h.hexdigest()
    recipe=dict(input_sha256=[digest(a) for a in (features,costs,valid_features,valid_costs)],seed=seed,epochs=epochs,batch=batch,lr=.001,weight_decay=.01,
        features=list(features.shape),costs=list(costs.shape),valid=list(valid_features.shape))
    if resume and (out/'complete.json').exists():raise ValueError('Head already complete')
    if resume:
        state=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if state['recipe']!=recipe:raise ValueError('Head recipe changed')
    elif out.exists():raise ValueError('Fresh head output required')
    else:out.mkdir(parents=True)
    torch.manual_seed(seed);head=ExitRouter(features.shape[-1]).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    gen=torch.Generator().manual_seed(seed+1);epoch0=0;trace=[];best=float('inf');best_state=None
    x=torch.as_tensor(np.asarray(features).reshape(-1,features.shape[-1]).copy(),device=device)
    c=torch.as_tensor(np.asarray(costs).reshape(-1,2).copy(),device=device)
    vx=torch.as_tensor(np.asarray(valid_features).reshape(-1,features.shape[-1]).copy(),device=device)
    vc=torch.as_tensor(np.asarray(valid_costs).reshape(-1,2).copy(),device=device)
    if resume:
        head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);gen.set_state(state['generator'])
        epoch0=state['epoch'];trace=state['trace'];best=state['best'];best_state=state['best_head']
    for epoch in range(epoch0,epochs):
        head.train();order=torch.randperm(len(x),generator=gen)
        for ix in order.split(batch):
            ix=ix.to(device);opt.zero_grad(set_to_none=True)
            loss=expected_loss(head(x[ix]),c[ix]);loss.backward();opt.step()
        head.eval()
        with torch.no_grad():
            total=0.
            for i in range(0,len(vx),batch):
                ch=head(vx[i:i+batch]).argmax(-1)
                total+=float(vc[i:i+batch].gather(1,ch[:,None]).double().sum())
        value=total/len(vx);trace.append(dict(epoch=epoch+1,validation_hard_nll=value))
        if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        full=dict(recipe=recipe,epoch=epoch+1,head=head.state_dict(),optimizer=opt.state_dict(),
                  generator=gen.get_state(),trace=trace,best=best,best_head=best_state)
        torch.save(full,out/'fullstate.tmp');(out/'fullstate.tmp').replace(out/'fullstate.pt')
        write_json(out/'progress.json',dict(epoch=epoch+1,best=best,trace=trace))
        if stop_after is not None and epoch+1>=stop_after:return None
    head.load_state_dict(best_state);head.eval();torch.save(best_state,out/'router.pt')
    write_json(out/'complete.json',dict(recipe=recipe,trace=trace,best=best,
        best_epoch=min(trace,key=lambda r:r['validation_hard_nll'])['epoch'],
        router_sha256=sha(out/'router.pt'),fullstate_sha256=sha(out/'fullstate.pt')))
    return head
