"""Third-stage P7 candidates: head-batched FA4 and fused target preparation."""
from contextlib import contextmanager

import torch


def pack_heads(q,k,v,kp,vp):
    # One FA4 call, same independent GQA groups. Native heads followed by the
    # four proxy query heads; no new keys or cross-head mixing.
    def pack(a,b):
        return torch.cat((a.transpose(1,2),b.transpose(1,2)),dim=2).transpose(1,2)
    return pack(q,q[:,-4:]),pack(k,kp),pack(v,vp)


def target_reference(values,mu,inverse,clip,stats,clips):
    raw=((values[0]+values[1])+values[2])+values[3]
    centered=raw-mu
    scaled=centered*inverse
    with torch.no_grad():
        moment=centered.sum((0,1)) if stats else None
        square=centered.square().sum((0,1)) if stats else None
        count=(scaled.abs()>clip).sum() if clips else None
    return scaled.clamp(-clip,clip),moment,square,count


@contextmanager
def candidate(mode,backend='inductor'):
    from deep_kv.proxy import ProxyModel
    import scripts.p7_speed_trial as trial
    original_attention=ProxyModel.memory_attention
    owned_window=ProxyModel.__dict__.get('_target_window')
    packed=torch.compile(pack_heads,backend=backend,fullgraph=True,dynamic=False)
    try:
        if mode in ('head_batch','kernel_both'):
            def attention(self,layer,u,head,prediction,mask,rotary,plan):
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                from deep_kv.proxy_memory import proxy_mass
                if self.arm!='P7-simple-targetgrad' or self.attention_backend!='fa4':
                    return original_attention(self,layer,u,head,prediction,mask,rotary,plan)
                if not self.proxy_enabled:return self.attention(layer,u,None,mask,rotary)
                a=layer.self_attn;shape=(*u.shape[:-1],-1,a.head_dim)
                q=a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
                k=a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
                v=a.v_proj(u).view(shape).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*rotary)
                kp,vp=head.entries(prediction,rotary,v[:,-2:])
                if self._proxy_mass is not None:
                    self._proxy_mass[str(a.layer_idx+1)]=proxy_mass(q[:,-4:],k[:,-2:],kp,plan.documents,a.scaling)
                qj,kj,vj=packed(q,k,v,kp,vp)
                output,lse=trial.attention_lse(qj,kj,vj,mask,a.scaling,self.fa4_kernel)
                if self._fa4_observer is not None:self._fa4_observer(qj,kj,vj,mask)
                h=q.shape[1]
                return trial.merge_attention(output[:,:,:h],output[:,:,h:],lse[:,:,:h],lse[:,:,h:])
            ProxyModel.memory_attention=attention
        if mode in ('target_kernel','kernel_both'):
            if backend=='aot_eager':function=target_reference
            else:
                from deep_kv.target_kernel import target_window
                function=target_window
            ProxyModel._target_window=staticmethod(function)
        yield
    finally:
        ProxyModel.memory_attention=original_attention
        if owned_window is None:
            if '_target_window' in ProxyModel.__dict__:del ProxyModel._target_window
        else:ProxyModel._target_window=owned_window


def primitive_gate(device):
    """Independent CUDA oracle: irregular tiles, clipping, overlap accumulation."""
    from deep_kv.target_kernel import target_window
    reports=[]
    for n,h in ((1,1),(17,9),(65,131),(128,1024)):
        torch.manual_seed(712+n+h)
        initial=[torch.randn(1,n,h,device=device) for _ in range(5)]
        mu=torch.randn(h,device=device);inverse=torch.rand(h,device=device)+.5
        # Exact clamp boundaries, zero, and outside-clamp entries.
        initial[0][0,0,0]=3.;mu[0]=0.;inverse[0]=1.
        for v in initial[1:]:v[0,0,0]=0.
        for stats,clips in ((False,False),(True,False),(True,True)):
            captures=[]
            for fn in (target_reference,target_window):
                values=[v.clone().requires_grad_() for v in initial]
                left=fn(values[:4],mu,inverse,3.,stats,clips)
                right=fn(values[1:],mu,inverse,3.,stats,clips)
                # Overlapping windows share FP32 input gradients before any
                # upstream BF16 cast, exactly as the backbone implementation.
                (left[0].square().sum()+.7*right[0].sum()).backward()
                captures.append((left,right,[v.grad for v in values]))
            for a,b in zip(*captures):
                for x,y in zip(a,b):
                    if x is None:assert y is None
                    else:torch.testing.assert_close(x,y,rtol=2e-5,atol=3e-5)
            reports.append(dict(rows=n,channels=h,statistics=stats,clipping=clips,status='passed'))
    return reports
