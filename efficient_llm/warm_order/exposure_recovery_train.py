"""A new, fixed-LR recovery phase with authenticated parent Adam/RNG state."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Rows, sha
from .exposure_recovery import load_parent, restore_rng
from .soft_exploration_train import amp, save_state, write_json


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['efficient_llm/warm_order/' + n + '.py' for n in (
        'model', 'data', 'soft_route', 'soft_exploration', 'soft_exploration_train',
        'exposure_recovery', 'exposure_recovery_train')]
    paths.append('efficient_llm/dot_pool/model.py')
    return {p: sha(root/p) for p in paths}


def train(config, out, device='cuda', stop_after=None, resume=False):
    c = dict(config)
    if c['schedule'] != 'inherit-terminal-lr-constant-no-floor':
        raise ValueError('Unregistered recovery schedule')
    if str(device).startswith('cuda') and torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    out = Path(out)
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('Fresh phase output required')
    if not isinstance(c['steps'], int) or c['steps'] <= 0:
        raise ValueError('Positive phase steps required')
    torch.set_num_threads(c.get('threads', 4))
    ids = np.load(c['indices'], allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Expected integer row order')
    needed = c['batch'] * c['steps']
    if len(ids) != needed or len(np.unique(ids)) != needed:
        raise ValueError('Exact unique phase row count required')
    inputs = {p: sha(p) for p in (c['data']+'.bin', c['data']+'.starts', c['indices'])}
    if inputs != c['expected_inputs']:
        raise ValueError('Phase data/order identity mismatch')
    model, opt, parent = load_parent(c['parent'], c['parent_hashes'], c['name'],
        expected_steps=c.get('parent_steps', 2000), device=device)
    parent_cfg = parent['recipe']['config']
    rows = Rows(c['data'], model.c.length)
    if ids.min() < 0 or ids.max() >= len(rows.ids):
        raise ValueError('Phase row out of range')
    if c['batch'] != parent_cfg['batch']:
        raise ValueError('Changed parent batch')
    recipe = dict(config=c, inputs=inputs, source=source_hashes(),
                  parent_recipe=parent['recipe'], torch=torch.__version__, numpy=np.__version__,
                  precision='bf16' if str(device).startswith('cuda') else 'fp32')
    inherited_lrs = [g['lr'] for g in opt.param_groups]
    step, elapsed, block_tokens, counts = 0, 0., 0, {}
    if resume:
        checkpoint = json.loads((out/'checkpoint.json').read_text())
        if sha(out/'fullstate.pt') != checkpoint['sha256']:
            raise ValueError('Phase checkpoint hash mismatch')
        state = torch.load(out/'fullstate.pt', map_location='cpu', weights_only=False)
        if state['recipe'] != recipe or state['step'] != checkpoint['step']:
            raise ValueError('Phase resume recipe mismatch')
        if (out/'complete.json').exists():
            raise ValueError('Phase already complete')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        step, elapsed = state['step'], state['elapsed']
        block_tokens, counts = state['block_tokens'], state['counts']
        trace = [json.loads(x) for x in (out/'trace.jsonl').read_text().splitlines()]
        if trace and trace[-1]['step'] > step:
            raise ValueError('Trace ahead of checkpoint')
        write_json(out/'resume.json', dict(resumed_step=step,
            loaded_fullstate_sha256=checkpoint['sha256'], source=recipe['source']))
    else:
        state = parent
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'recipe.json', recipe)
    if [g['lr'] for g in opt.param_groups] != inherited_lrs:
        raise ValueError('Phase LR changed from inherited terminal value')
    restore_rng(state, device)
    end = c['steps'] if stop_after is None else min(stop_after, c['steps'])
    if end <= step:
        raise ValueError('Stop must follow current phase step')
    model.train()
    # ExposureModel loads with the floor explicitly disabled. Never call its
    # set_training_update(): that would restart the predecessor's curriculum.
    if hasattr(model, 'disable_exploration'):
        model.disable_exploration()
    start = time.monotonic()
    for s in range(step, end):
        x = rows.batch(ids[s*c['batch']:(s+1)*c['batch']], device)
        opt.zero_grad(set_to_none=True)
        with amp(device):
            loss = model.nll(x).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite recovery loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        step = s+1
        block_tokens += (6 if parent_cfg['arm']=='C' else 18)*x.numel()
        for depth, choices in enumerate(model.last_routes):
            for k, n in enumerate(torch.bincount(choices.flatten(), minlength=4).tolist()):
                key = f'{depth}/{k}'
                counts[key] = counts.get(key, 0)+n
        if step % c.get('log_every', 50) == 0 or step == end:
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            record = dict(step=step, parent_step=parent['step'], loss=float(loss.detach()),
                lr=[g['lr'] for g in opt.param_groups], grad_norm=float(norm),
                first_row=int(ids[s*c['batch']]), elapsed_seconds=elapsed+time.monotonic()-start,
                block_tokens=block_tokens, exploration_epsilon=0.)
            with (out/'trace.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json', record)
            print(json.dumps(record), flush=True)
        if step % c.get('save_every', 1000) == 0 or step == end:
            save_state(out, model, opt, recipe, step, elapsed+time.monotonic()-start, block_tokens, counts)
    if step == c['steps']:
        torch.save(dict(config=parent_cfg['model'], arm=parent_cfg['arm'], seed=parent_cfg['seed'],
            canonical_probability=parent_cfg['canonical_probability'], model=model.state_dict()), out/'weights.pt')
        write_json(out/'complete.json', dict(step=step, parent_step=parent['step'], name=c['name'],
            arm=parent_cfg['arm'], seed=parent_cfg['seed'], input_tokens=needed*model.c.length,
            scored_tokens=needed*(model.c.length-model.c.prefix), block_tokens=block_tokens,
            parent_block_tokens=parent['block_tokens'], counts=counts,
            elapsed_seconds=elapsed+time.monotonic()-start, inherited_lrs=inherited_lrs,
            weights_sha256=sha(out/'weights.pt'), fullstate_sha256=sha(out/'fullstate.pt')))
    return model, opt


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True);p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda');p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true');a=p.parse_args()
    train(json.loads(Path(a.config).read_text()), a.out, a.device, a.stop_after, a.resume)
