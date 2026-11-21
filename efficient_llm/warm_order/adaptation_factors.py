"""Fixed-context attention N/P/V and FFN N/G/U/D weight-factor panels."""
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from .adaptation_analysis import _frozen_cpu, SWAPPED
from .adaptation_runtime import measurement_policy
from .adaptation_components import _attention_parts,_attention_output
from .layer_dependence import _validate,paired_scores

GROUPS={'attn':{
    'N':('input_layernorm.','self_attn.q_norm.','self_attn.k_norm.'),
    'P':('self_attn.q_proj.','self_attn.k_proj.'),
    'V':('self_attn.v_proj.','self_attn.o_proj.')},
    'ffn':{'N':('post_attention_layernorm.',),'G':('mlp.gate_proj.',),
           'U':('mlp.up_proj.',),'D':('mlp.down_proj.',)}}
CELLS=['attn/'+''.join(map(str,b)) for b in product(range(2),repeat=3)]
CELLS+=['ffn/'+''.join(map(str,b)) for b in product(range(2),repeat=4)]+['S']


def partition(block):
    names=set(block.state_dict());out={}
    for family,groups in GROUPS.items():
        for key,prefix in groups.items():out[family+'/'+key]=sorted(n for n in names if n.startswith(prefix))
    flattened=[n for group in out.values() for n in group]
    if not all(out.values()) or len(flattened)!=len(set(flattened)) or set(flattened)!=names:
        raise ValueError('Factor parameter partition mismatch')
    return out


@torch.no_grad()
def factors_crossovers(parent,adapted,tokens,target=2,path=SWAPPED):
    for model in (parent,adapted):_frozen_cpu(model);_validate(model,tokens)
    if asdict(parent.c)!=asdict(adapted.c):raise ValueError('Matching configurations required')
    if type(target) is not int or target!=2 or path!=SWAPPED:raise ValueError('Registered middle.2/swap required')
    blocks=(parent.middle[2],adapted.middle[2]);groups=partition(blocks[0])
    if partition(blocks[1])!=groups:raise ValueError('Mismatched partition')
    pos,seg=_validate(parent,tokens)
    h=parent.middle[0](parent.stem(parent.embed(tokens),pos,seg),pos,seg)
    z0=blocks[0].attention(h,pos,seg)
    def finish(y):
        for index in (1,3):y=parent.middle[index](y,pos,seg)
        return parent.norm(parent.coda(y,pos,seg))
    cells={}
    for n,p,v in product(range(2),repeat=3):
        norm,pattern,value=blocks[n],blocks[p].self_attn,blocks[v].self_attn
        mixed=SimpleNamespace(c=pattern.c,inv=pattern.inv,q_proj=pattern.q_proj,k_proj=pattern.k_proj,
            v_proj=value.v_proj,q_norm=norm.self_attn.q_norm,k_norm=norm.self_attn.k_norm)
        parts=_attention_parts(mixed,norm.input_layernorm(h))
        z=h+_attention_output(value,parts['pattern'],parts['value'])
        y=z+blocks[0].mlp(blocks[0].post_attention_layernorm(z))
        key=f'attn/{n}{p}{v}';cells[key]=finish(y)
    for n,g,u,d in product(range(2),repeat=4):
        x=blocks[n].post_attention_layernorm(z0)
        update=blocks[d].mlp.down_proj(F.silu(blocks[g].mlp.gate_proj(x))*blocks[u].mlp.up_proj(x))
        key=f'ffn/{n}{g}{u}{d}';cells[key]=finish(z0+update)
    cells['S']=finish(h)
    za=blocks[1].attention(h,pos,seg)
    refs={'attn/000':parent.hidden(tokens,path),'ffn/0000':parent.hidden(tokens,path),
          'attn/111':finish(za+blocks[0].mlp(blocks[0].post_attention_layernorm(za))),
          'ffn/1111':finish(z0+blocks[1].mlp(blocks[1].post_attention_layernorm(z0)))}
    policy=measurement_policy();parity={}
    for key,ref in refs.items():
        torch.testing.assert_close(cells[key],ref,atol=policy['component_atol'],rtol=policy['component_rtol'])
        parity[key]=float((cells[key]-ref).abs().max())
    if any(not torch.isfinite(v).all() for v in cells.values()):raise ValueError('Nonfinite factor activation')
    return dict(cells=cells,parity_max_abs=parity,partition=groups,input_source='parent',reader_source='parent')


def factors_effects(loss):
    if set(loss)!=set(CELLS):raise ValueError('All registered factor cells required')
    if any(not isinstance(v,torch.Tensor) or v.shape!=loss[CELLS[0]].shape or not torch.isfinite(v).all() for v in loss.values()):
        raise ValueError('Finite matched tensor losses required')
    out={}
    for family,groups in GROUPS.items():
        size=len(groups);zero='0'*size;base=loss[family+'/'+zero];singles=[]
        for j,group in enumerate(groups):
            for bits in product(range(2),repeat=size):
                if bits[j]:continue
                a=''.join(map(str,bits));b=a[:j]+'1'+a[j+1:]
                out[f'{family}/{group}/other{a[:j]+a[j+1:]}']=loss[family+'/'+a]-loss[family+'/'+b]
            one=zero[:j]+'1'+zero[j+1:];singles.append(base-loss[family+'/'+one])
        joint=base-loss[family+'/'+'1'*size];out[family+'/joint']=joint
        out[family+'/nonadditivity']=joint-sum(singles)
    return out


@torch.no_grad()
def score_factors(parent,adapted,tokens,result):
    scores={k:paired_scores(parent,tokens,h,result['cells']['attn/000']) for k,h in result['cells'].items()}
    return dict(cells=scores,effects_nll_sum=factors_effects({k:v['nll_sum'] for k,v in scores.items()}))
