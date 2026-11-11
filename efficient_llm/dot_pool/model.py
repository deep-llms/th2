"""One causal model for all registered DOT-pool arms. No network access."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class Config:
    study:str='E1'
    arm:str='D'
    d:int=512
    heads:int=8
    kv_heads:int=4
    head_dim:int=64
    ff:int=1536
    expert_ff:int=768
    middle:int=8
    pool:int=64
    vocab:int=2240
    stem:int=2
    coda:int=2
    k:int=2
    candidates:int=16
    backend:str='sdpa'
    expert_kernel:str='loop'
    label_rate:float=0.10
    label_alpha:float=0.1
    balance_alpha:float=1e-4
    balance_rate:float=1e-3
    @classmethod
    def production(cls,study,arm,**kw):
        return cls(study=study,arm=arm,**({'middle':20,'pool':160,'vocab':50304,'label_rate':.02} if study=='E2' else {}),**kw)

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        return self.weight*(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.q_proj=nn.Linear(c.d,c.heads*c.head_dim,bias=False)
        self.k_proj=nn.Linear(c.d,c.kv_heads*c.head_dim,bias=False)
        self.v_proj=nn.Linear(c.d,c.kv_heads*c.head_dim,bias=False)
        self.o_proj=nn.Linear(c.heads*c.head_dim,c.d,bias=False)
        self.q_norm=RMSNorm(c.head_dim);self.k_norm=RMSNorm(c.head_dim)
        self.register_buffer('inv',1/(10000**(torch.arange(0,c.head_dim,2).float()/c.head_dim)),persistent=False)
    def forward(self,x,pos,segments):
        b,t,_=x.shape;c=self.c
        q=self.q_norm(self.q_proj(x).view(b,t,c.heads,c.head_dim));k=self.k_norm(self.k_proj(x).view(b,t,c.kv_heads,c.head_dim));v=self.v_proj(x).view(b,t,c.kv_heads,c.head_dim)
        angles=pos.float()[...,None]*self.inv;angles=torch.cat([angles,angles],-1)[:,:,None,:]
        cos,sin=angles.cos().to(q.dtype),angles.sin().to(q.dtype)
        def rotary(z):return z*cos+torch.cat([-z[...,c.head_dim//2:],z[...,:c.head_dim//2]],-1)*sin
        q,k=rotary(q),rotary(k)
        if c.backend=='fa4':
            from flash_attn.cute.interface import flash_attn_varlen_func
            # Segment identity resets at every packed row, including fragments.
            start=torch.ones_like(segments,dtype=torch.bool);start[:,1:]=segments[:,1:]!=segments[:,:-1]
            starts=start.flatten().nonzero().flatten().to(torch.int32)
            cu=torch.cat([starts,starts.new_tensor([b*t])]);maxlen=int((cu[1:]-cu[:-1]).max())
            out=flash_attn_varlen_func(q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),cu,cu,maxlen,maxlen,causal=True)
            if isinstance(out,tuple):out=out[0]
            out=out.reshape(b,t,-1)
        else:
            mask=(segments[:,:,None]==segments[:,None,:]) & torch.ones(t,t,dtype=torch.bool,device=x.device).tril()
            out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2).repeat_interleave(c.heads//c.kv_heads,1),v.transpose(1,2).repeat_interleave(c.heads//c.kv_heads,1),attn_mask=mask[:,None]).transpose(1,2).reshape(b,t,-1)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self,d,f):
        super().__init__();self.gate_proj=nn.Linear(d,f,bias=False);self.up_proj=nn.Linear(d,f,bias=False);self.down_proj=nn.Linear(f,d,bias=False)
    def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class Experts(nn.Module):
    def __init__(self,n,c):
        super().__init__();self.n=n;self.c=c
        self.gate_up=nn.Parameter(torch.empty(n,c.d,2*c.expert_ff));self.down=nn.Parameter(torch.empty(n,c.expert_ff,c.d))
        nn.init.normal_(self.gate_up,std=.02);nn.init.normal_(self.down,std=.02)
        self._cast=None
    def reset_cast(self):self._cast=None
    def forward(self,x,ids):
        shape=ids.shape;flat=ids.reshape(-1);k=shape[-1]
        order=torch.argsort(flat,stable=True);which=torch.div(order,k,rounding_mode='floor');rows=x.reshape(-1,x.shape[-1])[which]
        counts=torch.bincount(flat,minlength=self.n);offs=counts.cumsum(0,dtype=torch.int32)
        if self.c.expert_kernel=='grouped' and x.is_cuda:
            if self._cast is None:self._cast=(self.gate_up.to(torch.bfloat16),self.down.to(torch.bfloat16))
            w1,w2=self._cast
            with torch.autocast('cuda',enabled=False):
                z=F.grouped_mm(rows.to(torch.bfloat16),w1,offs=offs)
                a,b=z.chunk(2,-1);vals=F.grouped_mm(F.silu(a)*b,w2,offs=offs)
        else:
            parts=[];offset=0
            for m,n in enumerate(counts.tolist()):
                if n:
                    z=rows[offset:offset+n]@self.gate_up[m];a,b=z.chunk(2,-1);parts.append((F.silu(a)*b)@self.down[m]);offset+=n
            vals=torch.cat(parts) if parts else rows.new_empty((0,self.c.d))
        inverse=torch.argsort(order)
        return vals[inverse].reshape(*shape,self.c.d)

class Block(nn.Module):
    def __init__(self,c,dense):
        super().__init__();self.input_layernorm=RMSNorm(c.d);self.self_attn=Attention(c);self.post_attention_layernorm=RMSNorm(c.d)
        self.mlp=MLP(c.d,c.ff) if dense else None
    def attention(self,h,pos,segments):return h+self.self_attn(self.input_layernorm(h),pos,segments)
    def forward(self,h,pos,segments):
        h=self.attention(h,pos,segments);return h+self.mlp(self.post_attention_layernorm(h))

class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.embed_tokens=nn.Embedding(c.vocab,c.d)
        self.stem=nn.ModuleList([Block(c,True) for _ in range(c.stem)])
        self.middle=nn.ModuleList([Block(c,c.arm in ('D','L')) for _ in range(c.middle//2 if c.arm=='L' else c.middle)])
        self.coda=nn.ModuleList([Block(c,True) for _ in range(c.coda)]);self.norm=RMSNorm(c.d)
        self.shared=c.arm in ('SO','DOT','SR');self.routed=c.arm in ('PL','SO','DOT','SR')
        if self.routed:
            n=c.pool if self.shared else 8
            self.pools=nn.ModuleList([Experts(n,c) for _ in range(1 if self.shared else c.middle)])
            if self.shared:self.depth=nn.Parameter(torch.zeros(c.middle,c.d))
            if c.arm!='SR':
                self.router=nn.Parameter(torch.empty(c.middle,n,c.d));nn.init.normal_(self.router,std=.02)
                self.register_buffer('beta',torch.zeros(1 if self.shared else c.middle,n))
                self.register_buffer('loads',torch.zeros_like(self.beta),persistent=False)
            if c.arm=='DOT':self.path=nn.Parameter(torch.zeros(n,c.d))
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
        self.pending=[];self.labels=[];self.gradient_factor=1.;self.capture=False;self.routes=[]
        self.forced_routes=None;self.intervention=None;self.label_stats=[]
    def _router_logits(self,s,x,previous):
        if self.c.arm=='DOT' and previous is not None:
            ids,gates=previous;x=x+(self.path[ids]*gates[...,None]).sum(-2)
        return F.linear(x.float(),self.router[s].float())
    def replay_loss(self):
        if not self.pending:return self.embed_tokens.weight.sum()*0
        losses=[]
        for r in self.pending:
            z=self._router_logits(r['s'],r['x'],r['previous']);logq=F.log_softmax(z.gather(-1,r['ids']),-1)
            losses.append(-(r['q']*logq).sum(-1))
        return self.c.label_alpha*torch.cat(losses).mean()
    def promote_labels(self):
        self.pending=self.labels;self.labels=[]
    def finish_update(self):
        if self.routed and self.c.arm!='SR':
            if torch.distributed.is_initialized():torch.distributed.all_reduce(self.loads)
            with torch.no_grad():self.beta.add_(self.c.balance_rate*torch.sign(self.loads.mean(-1,keepdim=True)-self.loads));self.loads.zero_()
    def _label(self,s,x,y,score,previous,pool,eligible,label_mask):
        mask=label_mask if label_mask is not None else (torch.rand(eligible.shape,device=x.device)<self.c.label_rate)&eligible
        if not mask.any():return
        xx=x[mask].detach();ids=score[mask].topk(min(self.c.candidates,pool.n),-1).indices
        with torch.no_grad():u=pool(xx,ids).float();yy=y[mask].detach().float()
        prev=None if previous is None else (previous[0][mask].detach(),previous[1][mask].detach())
        def hook(grad):
            with torch.no_grad():
                g=grad[mask].float()*self.gradient_factor
                c=-(g[:,None,:]*(self.c.k*u-yy[:,None,:])).sum(-1);std=c.std(-1,unbiased=False,keepdim=True)
                q=((c-c.mean(-1,keepdim=True))/(std+1e-6)).softmax(-1)
                self.labels.append(dict(s=s,x=xx,ids=ids,previous=prev,q=q,c=c,std=std.flatten()))
            return grad
        y.register_hook(hook)
    def forward(self,tokens,segments=None,positions=None,labels=None,label_masks=None):
        c=self.c;b,t=tokens.shape
        segments=torch.zeros_like(tokens) if segments is None else segments
        positions=torch.arange(t,device=tokens.device).expand(b,t) if positions is None else positions
        valid=segments>=0;eligible=valid.clone()
        if c.study=='E1' and labels is not None:
            eligible.zero_();eligible[:,:-1]=labels[:,1:]!=-100
        for pool in getattr(self,'pools',[]):pool.reset_cast()
        h=self.embed_tokens(tokens);self.routes=[];previous=None;balances=[];shared_counts=None;shared_probs=None;shared_d=None
        for layer in self.stem:h=layer(h,positions,segments)
        for s in range(c.middle):
            layer=self.middle[s%len(self.middle)];h=layer.attention(h,positions,segments);x=layer.post_attention_layernorm(h)
            if not self.routed:h=h+layer.mlp(x);continue
            if self.shared:x=x+self.depth[s]
            pool=self.pools[0 if self.shared else s];n=pool.n
            if c.arm=='SR':
                ids=torch.rand(b,t,n,device=h.device).topk(c.k,-1).indices;gates=torch.full((b,t,c.k),1/c.k,device=h.device)
            else:
                logits=self._router_logits(s,x,previous);sig=logits.sigmoid();score=sig+self.beta[0 if self.shared else s]
                ids=score.topk(c.k,-1).indices;gates=sig.gather(-1,ids);gates=gates/gates.sum(-1,keepdim=True)
                if self.training:
                    counts=torch.zeros(b,n,device=h.device).scatter_add_(1,ids.reshape(b,-1),valid[:,:,None].expand(-1,-1,c.k).reshape(b,-1).float())
                    probs=((sig/sig.sum(-1,keepdim=True))*valid[:,:,None]).sum(1);d=valid.sum(1).clamp_min(1).float()
                    with torch.no_grad():self.loads[0 if self.shared else s].add_(counts.sum(0))
                    if self.shared:
                        shared_counts=counts if shared_counts is None else shared_counts+counts;shared_probs=probs if shared_probs is None else shared_probs+probs;shared_d=d if shared_d is None else shared_d+d
                    else:balances.append((n*counts/(c.k*d[:,None])*(probs/d[:,None])).sum(-1).mean())
            if self.forced_routes is not None:ids,gates=self.forced_routes[s]
            y=(pool(x,ids)*gates[...,None]).sum(-2)*c.k
            if self.intervention is not None and self.intervention['s']==s:
                y=y.clone();y[self.intervention['b'],self.intervention['t']]=self.intervention['value']
            if self.training and c.arm=='DOT' and torch.is_grad_enabled():self._label(s,x,y,score,previous,pool,eligible,None if label_masks is None else label_masks[s])
            if self.capture:self.routes.append(dict(ids=ids.detach(),gates=gates.detach(),x=x.detach(),y=y.detach()))
            previous=(ids,gates);h=h+y
        for layer in self.coda:h=layer(h,positions,segments)
        h=self.norm(h)
        aux=h.sum()*0
        if shared_counts is not None:balances.append((n*shared_counts/(c.k*shared_d[:,None])*(shared_probs/shared_d[:,None])).sum(-1).mean())
        if balances:aux=c.balance_alpha*torch.stack(balances).mean()
        if labels is None:return F.linear(h,self.embed_tokens.weight)
        hh=h[:,:-1].reshape(-1,c.d);target=labels[:,1:].reshape(-1);count=(target!=-100).sum()
        total=h.sum()*0
        def ce(v,w,y):return F.cross_entropy(F.linear(v,w).float(),y,ignore_index=-100,reduction='sum')
        for i in range(0,len(target),1024):
            v=hh[i:i+1024];y=target[i:i+1024]
            total=total+(checkpoint(ce,v,self.embed_tokens.weight,y,use_reentrant=False) if self.training and h.requires_grad else ce(v,self.embed_tokens.weight,y))
        return dict(nll_sum=total,count=count,balance_loss=aux,taylor_loss=self.replay_loss() if self.training else aux*0)
    def parameter_groups(self):
        decay=[];plain=[]
        for name,p in self.named_parameters():
            (plain if p.ndim<2 or name in ('depth','path') else decay).append(p)
        return [dict(params=decay,weight_decay=.1),dict(params=plain,weight_decay=0.)]
    def counts(self):
        total=sum(p.numel() for p in self.parameters())-self.embed_tokens.weight.numel();c=self.c
        if c.arm=='L':active=total+sum(p.numel() for p in self.middle.parameters())
        elif self.routed:
            expert_total=sum(p.numel() for p in self.pools.parameters());active=total-expert_total+c.middle*c.k*3*c.d*c.expert_ff
        else:active=total
        return dict(nonembedding_total=total,nonembedding_active=active,embedding=self.embed_tokens.weight.numel())
