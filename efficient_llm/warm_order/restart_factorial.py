"""Strict name-based Adam transfer for the dense restart factorial."""
import copy
from pathlib import Path
import torch
from .model import Config
from .soft_route import Model as SoftModel


def transfer_body_adam(model, optimizer, parent, *, phase_steps, terminal_lr):
    """Parent model/files must already be authenticated by pair_order.load_initial.

    Reconstruct the saved optimizer's parameter order from its exact dense schema;
    transfer by name, never by target position. Validate before mutating the target.
    """
    recipe = parent['recipe']; c = recipe['config']; earlier = recipe['parent_recipe']['config']
    if (c['arm'] != 'C' or c['schedule'] != 'inherit-terminal-lr-constant-no-floor'
            or c['steps'] != phase_steps or parent['step'] != phase_steps
            or c['parent_steps'] != earlier['steps'] or earlier['arm'] != 'C'):
        raise ValueError('Unsupported recovery optimizer lineage')
    with torch.random.fork_rng(devices=[]):
        old = SoftModel(Config(**c['model']), 'dense', earlier['canonical_probability'])
    old.load_state_dict(parent['model'], strict=True)
    source_names = [n for n,p in old.named_parameters() if p.requires_grad]
    target = dict(model.named_parameters())
    target_names = [n for n,p in model.named_parameters() if p.requires_grad]
    if set(source_names) != set(target_names) or any(model.is_router(n) for n in target_names):
        raise ValueError('Dense trainable-name mismatch')
    saved = parent['optimizer']; groups = saved['param_groups']
    if len(groups) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('Expected one dense optimizer group')
    g = groups[0]; ids = g['params']; steps = phase_steps + c['parent_steps']
    if (len(ids) != len(source_names) or len(set(ids)) != len(ids) or set(ids) != set(saved['state'])
            or tuple(g['betas']) != (.9,.95) or g['weight_decay'] != .1
            or g['eps'] != optimizer.param_groups[0]['eps'] or g.get('amsgrad',False)
            or g.get('maximize',False) or g.get('capturable',False) or g.get('differentiable',False)
            or abs(g['lr']-terminal_lr)>1e-12 or g['peak'] != earlier['lr']
            or abs(g['lr']-.1*g['peak'])>1e-12):
        raise ValueError('Recovery Adam group mismatch')
    if any(not torch.equal(target[n].detach().cpu(),parent['model'][n]) for n in target_names):
        raise ValueError('Adam transfer requires identical body weights')
    named = {}
    for name, sid in zip(source_names, ids):
        values = saved['state'][sid]; p = target[name]
        if set(values) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Unexpected Adam state fields')
        step = values['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or float(step)!=steps:
            raise ValueError('Wrong cumulative Adam counter')
        for key in ('exp_avg','exp_avg_sq'):
            v=values[key]
            if v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all():
                raise ValueError('Invalid named Adam moment: '+name)
        if (values['exp_avg_sq']<0).any():raise ValueError('Negative Adam variance')
        named[name]=copy.deepcopy(values)
    dest=optimizer.state_dict(); target_ids=dest['param_groups'][0]['params']
    # Verify the actual target optimizer ordering against its parameter objects.
    reverse={id(p):n for n,p in model.named_parameters()}
    order=[reverse[id(p)] for p in optimizer.param_groups[0]['params']]
    if len(order)!=len(set(order)) or set(order)!=set(target_names):
        raise ValueError('Target optimizer coverage mismatch')
    dest['state']={sid:named[name] for sid,name in zip(target_ids,order)}
    optimizer.load_state_dict(dest)
    return dict(parameters=len(order),cumulative_adam_steps=steps,terminal_lr=terminal_lr,
                names=order,transfer='authenticated-source-order-to-name-to-target-order')


def initialize_optimizer(model, c, device):
    if c['arm']!='C' or c['factor'] not in ('FL','IH','IL','FH'):
        raise ValueError('Unregistered dense factorial cell')
    if c['terminal_lr']<=0 or c['lr']<=0:raise ValueError('Invalid LR policy')
    body=[p for n,p in model.named_parameters() if p.requires_grad and not model.is_router(n)]
    opt=torch.optim.AdamW([dict(params=body,lr=c['lr'],peak_lr=c['lr'],weight_decay=.1,group_name='body')],betas=(.9,.95))
    # Validate lineage and full Adam state even for reset cells; construction is RNG-neutral.
    parent=torch.load(Path(c['parent'])/'fullstate.pt',map_location='cpu',weights_only=False)
    proof=transfer_body_adam(model,opt,parent,phase_steps=c['parent_steps'],terminal_lr=c['terminal_lr'])
    if c['factor'].startswith('F'):opt.state.clear()
    proof.update(factor=c['factor'],inherited=c['factor'].startswith('I'),initial_state_entries=len(opt.state))
    return opt,proof
