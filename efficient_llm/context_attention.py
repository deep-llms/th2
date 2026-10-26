"""Full-history versus sink+recent attention on contiguous, unpadded Qwen3 streams.

Static policies execute only their selected attention branches, including decode.
Teacher/oracle/router references compute both branches and make no speed claim.
All native KV is retained; static cache group order is private to this context.
"""
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from efficient_llm.group_selection import contributions, subset_masks
from efficient_llm.routers import subset_costs


def local_attention(module, q, k, v, *, start, window, sinks, chunk=256):
    """Exact causal sink+recent attention; slice history BEFORE attention.

    Absolute RoPE positions are untouched. Overlapping sink/recent keys occur once.
    Chunking bounds prefill masks/work; single-token decode reads <= sinks+window keys.
    q/k/v layout B,H,T,D; return native B,T,H,D layout.
    """
    pieces=[]
    for a in range(0,q.shape[2],chunk):
        b=min(a+chunk,q.shape[2]); end=start+b
        left=min(end,max(sinks,start+a-window+1))
        sink_end=min(sinks,end)
        if sinks==0 and b-a==1:
            # Pure sliding-window decode needs only a view, no concatenation or
            # positions tensor. Keep the existing sink reference unchanged.
            y,_=sdpa_attention_forward(module,q[:,:,a:b],k[:,:,left:end],v[:,:,left:end],None,
                dropout=0.,scaling=module.scaling,is_causal=False)
            pieces.append(y)
            continue
        if left<=sink_end:
            keys=k[:,:,:end];values=v[:,:,:end];positions=torch.arange(end,device=q.device)
        else:
            keys=torch.cat((k[:,:,:sink_end],k[:,:,left:end]),dim=2)
            values=torch.cat((v[:,:,:sink_end],v[:,:,left:end]),dim=2)
            positions=torch.cat((torch.arange(sink_end,device=q.device),torch.arange(left,end,device=q.device)))
        if b-a==1:
            mask=None  # all sliced keys are allowed for this one current query
        else:
            queries=torch.arange(start+a,end,device=q.device)
            mask=(positions[None,:]<=queries[:,None]) & ((positions[None,:]<sinks) | (positions[None,:]>=queries[:,None]-window+1))
            mask=mask[None,None]
        y,_=sdpa_attention_forward(module,q[:,:,a:b],keys,values,mask,
            dropout=0.,scaling=module.scaling,is_causal=False)
        pieces.append(y)
    return torch.cat(pieces,dim=1) if len(pieces)>1 else pieces[0]


def mix_outputs(full, local, selected):
    g=selected.shape[-1]
    shape=(*full.shape[:2],g,-1)
    return torch.where(selected[...,None],full.reshape(shape),local.reshape(shape)).reshape_as(full)


class ContextPolicy:
    MODES={'teacher','local','static','oracle','router','reference_static'}

    def __init__(self,model,*,mode,keep=4,window=128,sinks=4,stride=16,static_masks=None,routers=None,bank=0,decode_backend='sdpa'):
        c=model.config
        if c.model_type!='qwen3' or c._attn_implementation!='sdpa' or min(window,stride)<1 or sinks<0 or mode not in self.MODES:
            raise ValueError('Require native SDPA Qwen3 and valid context policy')
        if any(layer.self_attn.sliding_window is not None for layer in model.model.layers):
            raise ValueError('Native sliding layers are outside this contract')
        self.model,self.mode,self.window,self.sinks,self.stride=model,mode,window,sinks,stride
        if decode_backend not in ('sdpa','triton','triton_compact') or (decode_backend!='sdpa' and mode not in ('router','reference_static','local')):
            raise ValueError('Selective decode supports router, reference_static or local only')
        self.decode_backend=decode_backend
        self.keep=keep
        if decode_backend=='triton_compact' and window+sinks>256:
            raise ValueError('Compact backend supports <=256 local keys')
        self.groups,self.repeat=c.num_key_value_heads,c.num_attention_heads//c.num_key_value_heads
        self.masks=subset_masks(self.groups,keep,device=next(model.parameters()).device)
        self.routers,self.bank=routers,bank
        if mode=='router' and (routers is None or bank not in (0,1) or routers.spec['keep']!=keep or routers.spec['groups']!=self.groups or routers.spec['layers']!=c.num_hidden_layers):
            raise ValueError('Router recipe mismatch')
        self.static=static_masks
        self.orders=[]
        if mode in ('static','reference_static'):
            if static_masks is None or static_masks.dtype!=torch.bool or tuple(static_masks.shape)!=(c.num_hidden_layers,self.groups):
                raise ValueError('One boolean static group assignment per layer is required')
            self.static=static_masks.to(self.masks.device)
            for row in static_masks.cpu():
                selected=row.nonzero().flatten();others=(~row).nonzero().flatten()
                order=torch.cat((selected,others)).to(self.masks.device)
                heads=(order[:,None]*self.repeat+torch.arange(self.repeat,device=order.device)).flatten()
                self.orders.append((len(selected),order,heads,heads.argsort()))
        self.handles=[];self.original=[];self.inputs={};self.costs={};self.stats={}
        self.layer_totals=torch.zeros(c.num_hidden_layers,2,dtype=torch.float64,device=self.masks.device)
        self.start=0
        # A cache retains this token after the policy exits; unlike id(self),
        # its identity cannot be recycled while the cache still owns it.
        self.cache_owner=object()

    def _begin(self,module,args,kwargs):
        if kwargs.get('attention_mask') is not None or kwargs.get('position_ids') is not None:
            raise ValueError('Context screen requires unpadded contiguous streams with automatic positions')
        cache=kwargs.get('past_key_values')
        self.start=0
        if cache is not None:
            owner=getattr(cache,'_context_policy_owner',None)
            if owner is not None and owner is not self.cache_owner:
                raise ValueError('Cannot switch policies on an existing KV cache')
            if owner is None:
                if int(cache.get_seq_length())!=0: raise ValueError('Require a fresh cache for context policy')
            else:
                if not cache._context_policy_valid:raise ValueError('Previous cache update was incomplete')
                self.start=cache._context_policy_length
            cache._context_policy_valid=False

    def __enter__(self):
        if self.original:raise RuntimeError('Context policy already installed')
        self.handles.append(self.model.model.register_forward_pre_hook(self._begin,with_kwargs=True))
        for i,layer in enumerate(self.model.model.layers):
            module=layer.self_attn;self.original.append(module.forward)
            def forward(module,hidden_states,position_embeddings,attention_mask,past_key_values=None,index=i,**kwargs):
                cache=past_key_values
                if cache is not None:cache._context_policy_owner=self.cache_owner
                result=self._forward(index,module,hidden_states,position_embeddings,attention_mask,cache,kwargs)
                if cache is not None and index==len(self.original)-1:
                    cache._context_policy_length=self.start+hidden_states.shape[1]
                    cache._context_policy_valid=True
                return result
            module.forward=MethodType(forward,module)
        return self

    def __exit__(self,*args):
        for layer,original in zip(self.model.model.layers,self.original):layer.self_attn.forward=original
        for handle in self.handles:handle.remove()
        self.original.clear();self.handles.clear();self.inputs.clear();self.costs.clear()

    def take(self):
        layers=len(self.model.model.layers)
        if set(self.inputs)!=set(range(layers)) or set(self.costs)!=set(range(layers)):
            raise RuntimeError('Incomplete context teacher features')
        x=torch.stack([self.inputs[i] for i in range(layers)],1)
        costs=torch.stack([self.costs[i] for i in range(layers)],1)
        self.inputs.clear();self.costs.clear()
        return x,costs

    def _forward(self,index,module,x,position_embeddings,attention_mask,cache,kwargs):
        mode=self.mode
        if mode in ('static','reference_static') and self.orders[index][0]==self.groups and (self.decode_backend=='sdpa' or x.shape[1]>1):
            return self.original[index](hidden_states=x,position_embeddings=position_embeddings,
                attention_mask=attention_mask,past_key_values=cache,**kwargs)
        # Router selection precedes QKV/attention; later outputs cannot change it.
        selected=self.masks[self.routers.layer_choice(x,index,self.bank)] if mode=='router' else None
        shape=(*x.shape[:-1],-1,module.head_dim)
        q=module.q_norm(module.q_proj(x).view(shape)).transpose(1,2)
        k=module.k_norm(module.k_proj(x).view(shape)).transpose(1,2)
        v=module.v_proj(x).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        if mode=='static' and 0<self.orders[index][0]<self.groups:
            n,order,heads,inverse=self.orders[index]
            q=q.index_select(1,heads);k=k.index_select(1,order);v=v.index_select(1,order)
        if cache is not None:k,v=cache.update(k,v,module.layer_idx)
        if self.decode_backend in ('triton','triton_compact') and x.shape[1]==1:
            from efficient_llm.selective_decode import decode,decode_compact
            if mode=='local':selected=self.masks.new_zeros(x.shape[0],1,self.groups)
            if mode=='reference_static':selected=self.static[index].expand(x.shape[0],1,-1)
            options=dict(length=self.start+1,window=self.window,sinks=self.sinks,scale=module.scaling)
            if self.decode_backend=='triton_compact':
                count=self.keep if mode=='router' else (0 if mode=='local' else self.orders[index][0])
                output=decode_compact(q,k,v,selected[:,0],global_groups=count,**options)
            else:output=decode(q,k,v,selected[:,0],**options)
        elif mode=='local' or (mode=='static' and self.orders[index][0]==0):
            output=local_attention(module,q,k,v,start=self.start,window=self.window,sinks=self.sinks)
        elif mode=='static':
            n,order,heads,inverse=self.orders[index];h=n*self.repeat
            global_out,_=sdpa_attention_forward(module,q[:,:h],k[:,:n],v[:,:n],attention_mask,dropout=0.,scaling=module.scaling)
            local_out=local_attention(module,q[:,h:],k[:,n:],v[:,n:],start=self.start,window=self.window,sinks=self.sinks)
            output=torch.cat((global_out,local_out),dim=2).index_select(2,inverse)
        else:
            full,_=sdpa_attention_forward(module,q,k,v,attention_mask,dropout=0.,scaling=module.scaling)
            local=local_attention(module,q,k,v,start=self.start,window=self.window,sinks=self.sinks)
            if mode=='teacher':
                sample=slice(self.stride-1,None,self.stride)
                self.inputs[index]=x[:,sample].detach().reshape(-1,x.shape[-1]).clone()
                f=full[:,sample].flatten(2).reshape(-1,module.o_proj.in_features)
                d=f.float()-local[:,sample].flatten(2).reshape_as(f).float()
                with torch.autocast(x.device.type,enabled=False):
                    c=contributions(d,module.o_proj.weight,self.groups)
                    self.costs[index]=subset_costs(c,self.masks).detach()
                    energy=contributions(f,module.o_proj.weight,self.groups).sum(-2).square().sum(-1)
                    self.layer_totals[index]+=torch.stack((c.sum(-2).square().sum(-1).double().sum(),energy.double().sum()))
                output=full
            else:
                if mode=='oracle':
                    d=full.flatten(2).float()-local.flatten(2).float()
                    parts=[]
                    for offset in range(0,d.shape[1],128):
                        c=contributions(d[:,offset:offset+128],module.o_proj.weight,self.groups)
                        parts.append(self.masks[subset_costs(c,self.masks)[...,0,:].argmin(-1)])
                    selected=torch.cat(parts,dim=1)
                elif mode=='reference_static':selected=self.static[index].expand(*x.shape[:2],-1)
                output=mix_outputs(full,local,selected)
        return module.o_proj(output.reshape(*x.shape[:-1],-1).contiguous()),None
