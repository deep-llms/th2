"""Attention/FFN weights crossed at fixed parent input and downstream reader."""
from dataclasses import asdict
from itertools import product
import torch
from .adaptation_analysis import _frozen_cpu, SWAPPED
from .adaptation_runtime import measurement_policy
from .layer_dependence import _validate, paired_scores


def partition(block):
    groups = {'attention': ('input_layernorm.', 'self_attn.'),
              'ffn': ('post_attention_layernorm.', 'mlp.')}
    names = set(block.state_dict())
    result = {k: sorted(n for n in names if n.startswith(prefix)) for k,prefix in groups.items()}
    if not all(result.values()) or set(result['attention']) & set(result['ffn']) or set().union(*map(set,result.values())) != names:
        raise ValueError('Target block parameter partition mismatch')
    return result


@torch.no_grad()
def sublayer_crossovers(parent, adapted, tokens, target=2, path=SWAPPED):
    for model in (parent, adapted):
        _frozen_cpu(model);_validate(model,tokens)
    if asdict(parent.c)!=asdict(adapted.c):raise ValueError('Matching configurations required')
    if type(target) is not int or target!=2 or path!=SWAPPED:
        raise ValueError('Registered middle.2 in fixed swapped path required')
    blocks=(parent.middle[target],adapted.middle[target])
    groups=partition(blocks[0])
    if partition(blocks[1])!=groups:raise ValueError('Mismatched partition')
    pos,seg=_validate(parent,tokens)
    h=parent.middle[0](parent.stem(parent.embed(tokens),pos,seg),pos,seg)
    z=[b.attention(h,pos,seg) for b in blocks]
    outputs={f'{a}{f}':z[a]+blocks[f].mlp(blocks[f].post_attention_layernorm(z[a]))
             for a,f in product(range(2),repeat=2)}
    def finish(y):
        for index in (1,3):y=parent.middle[index](y,pos,seg)
        return parent.norm(parent.coda(y,pos,seg))
    cells={k:finish(v) for k,v in outputs.items()};cells['S']=finish(h)
    policy=measurement_policy();parity={}
    for bit,b in enumerate(blocks):
        direct=finish(b(h,pos,seg));end=cells[str(bit)*2]
        torch.testing.assert_close(end,direct,atol=policy['component_atol'],rtol=policy['component_rtol'])
        parity[str(bit)]=float((end-direct).abs().max())
    torch.testing.assert_close(cells['00'],parent.hidden(tokens,path),atol=policy['component_atol'],rtol=policy['component_rtol'])
    if any(not torch.isfinite(v).all() for v in cells.values()):raise ValueError('Nonfinite activation')
    return dict(cells=cells,parity_max_abs=parity,partition=groups,
                input_source='parent',reader_source='parent',bit_order=['attention','ffn'])


def sublayer_effects(loss):
    if set(loss)!={'00','01','10','11','S'}:raise ValueError('Four cells and skip required')
    if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in loss.values()):
        raise ValueError('Finite tensor losses required')
    if any(v.shape!=loss['00'].shape for v in loss.values()):raise ValueError('Matched losses required')
    out={'attention/ffn0':loss['00']-loss['10'],
         'attention/ffn1':loss['01']-loss['11'],
         'ffn/attention0':loss['00']-loss['01'],
         'ffn/attention1':loss['10']-loss['11'],
         'joint':loss['00']-loss['11']}
    out['interaction']=out['joint']-out['attention/ffn0']-out['ffn/attention0']
    out.update({'usefulness/'+k:loss['S']-loss[k] for k in ('00','01','10','11')})
    return out


@torch.no_grad()
def score_sublayer(parent, adapted, tokens, result):
    scores={k:paired_scores(parent,tokens,h,result['cells']['00']) for k,h in result['cells'].items()}
    return dict(cells=scores,effects_nll_sum=sublayer_effects({k:v['nll_sum'] for k,v in scores.items()}))
