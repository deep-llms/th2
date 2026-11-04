"""Apply specified merges before pruning; keep all receiving normalization state."""
import torch
from efficient_llm.a2.model import prune


def formula(values, variant):
    if variant=='rdsc':
        # Exactly theta_receiver + sum(theta_removed - theta_receiver).
        return values[0] + torch.stack([x-values[0] for x in values[1:]]).sum(dim=0)
    if variant=='avg':return torch.stack(values).mean(dim=0)
    raise ValueError('Unregistered merge')


@torch.no_grad()
def merge_and_prune(model, variant, checks):
    layers=model.model.layers
    manifest=[]
    for receiver,others in ((15,(16,17,18)),(22,(19,20,21))):
        before={k:v.detach().clone() for k,v in layers[receiver].named_parameters()}
        sources=[dict(layers[i].named_parameters()) for i in (receiver,*others)]
        changed=[]
        for name,target in layers[receiver].named_parameters():
            # q_norm/k_norm live inside attention and must also remain unchanged.
            is_weight_or_bias=name.rsplit('.',1)[-1] in ('weight','bias')
            eligible=name.startswith(('self_attn.','mlp.')) and 'norm' not in name.lower() and is_weight_or_bias
            if not eligible:continue
            values=[before[name]]+[s[name].detach() for s in sources[1:]]
            value=formula(values[1:]+values[:1] if variant=='avg' and receiver==22 else values,variant)
            target.copy_(value)
            # Recompute independently, in the same specified arithmetic order.
            if variant=='rdsc':
                terms=torch.stack([s[name].detach()-before[name] for s in sources[1:]])
                expected=before[name]+terms.sum(dim=0)
            else:
                ordered=[before[name]]+[s[name].detach() for s in sources[1:]]
                if receiver==22:ordered=ordered[1:]+ordered[:1]
                expected=torch.stack(ordered).mean(dim=0)
            checks.record(f'YT3_formula:{receiver}:{name}',torch.equal(target,expected))
            changed.append(name)
        checks.record(f'YT3_nonempty_merge:{receiver}',bool(changed))
        checks.record(f'YT3_receiving_norms:{receiver}',all(torch.equal(p,before[n]) for n,p in layers[receiver].named_parameters() if n not in changed))
        manifest.append(dict(receiver=receiver,removed=list(others),parameters=changed,variant=variant))
    kept=prune(model,list(range(16,22)))
    checks.record('YT3_kept',kept==list(range(16))+[22,23],dict(kept=kept,formulas=manifest))
    return kept,manifest
