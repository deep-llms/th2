"""Paired continuation of authenticated dense Horizon parents.

Separate experiment, constant inherited terminal LR, original parameter order.
CPU qualified; bounded CUDA qualification and pinned production admission path.
Ownership and actual GPU numerical/fullstate gates remain external requirements.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .adaptation_analysis import CANONICAL, SWAPPED
from .data import Rows, sha
from .exposure_recovery import restore_rng
from .horizon_train import (amp, lr_at, save_state, write_json,
                            source_hashes as horizon_source_hashes)
from .model import Model, Config
from .adaptation_reservation import validate_binding

PARENT_FILES = ('fullstate.pt', 'weights.pt', 'recipe.json', 'complete.json', 'checkpoint.json')
POLICY = 'inherit-terminal-lr-constant'


def validate_adam(model, optimizer_state, steps, lr):
    """Legacy Horizon ID order derives from its pinned model.parameters() call.

    Require that exact order, then persist the resulting name map for this phase.
    Do not guess a mapping for arbitrary checkpoint formats.
    """
    named = list(model.named_parameters())
    groups = optimizer_state['param_groups']
    if len(groups) != 1 or groups[0]['params'] != list(range(len(named))):
        raise ValueError('Unrecognized parent parameter-ID order')
    g = groups[0]
    if (tuple(g['betas']) != (.9, .95) or g['weight_decay'] != .1 or
            not math.isclose(g['lr'], lr, rel_tol=0, abs_tol=1e-12) or
            g.get('amsgrad', False) or g.get('maximize', False)):
        raise ValueError('Adam policy mismatch')
    if set(optimizer_state['state']) != set(g['params']):
        raise ValueError('Missing or extraneous Adam state')
    for i, (name, parameter) in enumerate(named):
        state = optimizer_state['state'][i]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != steps:
            raise ValueError('Adam counter/schema mismatch: '+name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            if (value.shape != parameter.shape or value.dtype != parameter.dtype or
                    not torch.isfinite(value).all()):
                raise ValueError('Adam moment mismatch: '+name)
        if (state['exp_avg_sq'] < 0).any():
            raise ValueError('Negative Adam second moment')
    return [name for name, _ in named]


def load_parent(directory, pins):
    """Read trusted hash-pinned terminal files; never write the parent directory."""
    directory = Path(directory)
    if set(pins) != set(PARENT_FILES):
        raise ValueError('Five exact parent file pins required')
    for name in PARENT_FILES:
        if sha(directory/name) != pins[name]:
            raise ValueError('Parent hash mismatch: '+name)
    recipe = json.loads((directory/'recipe.json').read_text())
    done = json.loads((directory/'complete.json').read_text())
    checkpoint = json.loads((directory/'checkpoint.json').read_text())
    c = recipe['config']
    if c['arm'] != 'D' or recipe['source'] != horizon_source_hashes():
        raise ValueError('Only pinned original dense Horizon schema supported')
    if (done['step'] != c['steps'] or done['seed'] != c['seed'] or done['arm'] != 'D' or
            done['weights_sha256'] != pins['weights.pt'] or
            done['fullstate_sha256'] != pins['fullstate.pt'] or
            checkpoint['sha256'] != pins['fullstate.pt'] or checkpoint['step'] != c['steps']):
        raise ValueError('Parent terminal metadata mismatch')
    state = torch.load(directory/'fullstate.pt', map_location='cpu', weights_only=False)
    weights = torch.load(directory/'weights.pt', map_location='cpu', weights_only=False)
    if state['recipe'] != recipe or state['step'] != c['steps'] or weights['config'] != c['model']:
        raise ValueError('Parent fullstate metadata mismatch')
    if state['model'].keys() != weights['model'].keys() or any(
            not torch.equal(v, weights['model'][k]) or not torch.isfinite(v).all()
            for k, v in state['model'].items()):
        raise ValueError('Parent weights/fullstate mismatch')
    model = Model(Config(**c['model']))
    model.load_state_dict(state['model'], strict=True)
    lr = lr_at(c['steps']-1, c['steps'], c['lr'], c['warmup'])
    names = validate_adam(model, state['optimizer'], c['steps'], lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.1)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    return model, optimizer, state, names


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ['model', 'data', 'adaptation_analysis', 'adaptation_train',
             'horizon_train', 'exposure_recovery', 'adaptation_reservation',
             'adaptation_runtime','adaptation_components','adaptation_eval',
             'adaptation_pipeline','adaptation_geometry','adaptation_geometry_pipeline',
             'adaptation_study','adaptation_timecourse','adaptation_gpu_probe','layer_dependence']
    return {p: sha(root/p) for p in ['efficient_llm/warm_order/'+n+'.py' for n in names]
            + ['efficient_llm/dot_pool/model.py']}


def check_runtime(config, device, stop_after, gpu_qualification):
    """GPU qualification is bounded; full training requires a pinned gate receipt."""
    if device == 'cpu':
        if gpu_qualification: raise ValueError('GPU qualification requires CUDA')
        return 'cpu-qualification'
    if device != 'cuda': raise ValueError('Explicit cpu or single visible cuda required')
    if gpu_qualification:
        if type(stop_after) is not int or not 0 < stop_after <= 40:
            raise ValueError('GPU qualification is limited to phase step40')
        if 'row_reservation' not in config:
            raise ValueError('Bound actual reservation required for GPU qualification')
        validate_binding(config)
        return 'gpu-qualification'
    pin = config.get('gpu_admission')
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('Production GPU admission pending')
    if sha(pin['path']) != pin['sha256']: raise ValueError('GPU admission hash mismatch')
    gate = json.loads(Path(pin['path']).read_text())
    if (gate.get('source') != source_hashes() or gate.get('torch') != torch.__version__ or
            gate.get('precision') != 'bf16' or gate.get('tf32') is not False or
            gate.get('reservation_sha256') != config['row_reservation']['manifest_sha256'] or
            gate.get('native_component_parity_pass') is not True):
        raise ValueError('GPU admission source/runtime/reservation/parity mismatch')
    checks=gate.get('branches', {})
    if set(checks) != {'A-s0','B-s0','A-s1','B-s1'}:
        raise ValueError('All four GPU branch gates required')
    for record in checks.values():
        if (record.get('step') != 40 or record.get('exact_resume') is not True or
                record.get('finite') is not True or record.get('parent_step0_exact') is not True):
            raise ValueError('Incomplete GPU fullstate qualification')
    key=f"{config['branch']}-s{config['row_reservation']['seed']}"
    if checks[key].get('parent_hashes') != config['parent_hashes']:
        raise ValueError('GPU admission parent mismatch')
    return 'gpu-admitted'


def train(config, out, *, stop_after=None, resume=False, device='cpu', gpu_qualification=False):
    scope = check_runtime(config, device, stop_after, gpu_qualification)
    c = copy.deepcopy(config)
    if c['branch'] not in ('A', 'B') or c['lr_policy'] != POLICY or c['domain'] != 'dclm':
        raise ValueError('Unregistered branch/LR/domain')
    for key in ('steps', 'batch', 'threads'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError('Positive integer required: '+key)
    snapshots = c['snapshots']
    if (not isinstance(snapshots, list) or any(type(v) is not int for v in snapshots) or
            snapshots != sorted(set(snapshots)) or not snapshots or
            snapshots[0] != 0 or snapshots[-1] != c['steps']):
        raise ValueError('Sorted snapshot list must span zero to terminal step')
    out = Path(out)
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('Fresh output required')
    reservation = validate_binding(c) if 'row_reservation' in c else None
    row_paths = [c['indices'], c['confirmation_indices']]
    if 'calibration_indices' in c: row_paths.append(c['calibration_indices'])
    inputs = {p: sha(p) for p in [c['data']+'.bin', c['data']+'.starts']+row_paths}
    if inputs != c['expected_inputs']:
        raise ValueError('Data/order/confirmation identity mismatch')
    ids = np.load(c['indices'], allow_pickle=False)
    confirmation = np.load(c['confirmation_indices'], allow_pickle=False)
    selections = [ids, confirmation]
    if 'calibration_indices' in c:
        selections.append(np.load(c['calibration_indices'], allow_pickle=False))
    for a in selections:
        if a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a) or len(np.unique(a)) != len(a):
            raise ValueError('Nonempty unique integer rows required')
    if len(ids) != c['batch']*c['steps'] or np.intersect1d(ids, confirmation).size:
        raise ValueError('Phase count or confirmation overlap')
    if any(np.intersect1d(a, b).size for i, a in enumerate(selections) for b in selections[i+1:]):
        raise ValueError('Training/calibration/confirmation overlap')
    torch.set_num_threads(c['threads'])
    model, optimizer, parent, names = load_parent(c['parent'], c['parent_hashes'])
    if c['batch'] != parent['recipe']['config']['batch']:
        raise ValueError('Changed parent batch')
    if reservation is not None and (reservation['length'] != model.c.length or
            c['row_reservation']['seed'] != parent['recipe']['config']['seed']):
        raise ValueError('Reservation parent seed/length mismatch')
    rows = Rows(c['data'], model.c.length)
    if any(a.min() < 0 or a.max() >= len(rows.ids) for a in selections):
        raise ValueError('Row out of range')
    # Reject boundary-invalid rows before creating outputs or updating weights.
    for a in selections:
        for start in range(0, len(a), 4096):
            if np.any(rows.starts[a[start:start+4096], 1:]):
                raise ValueError('Cross-document row')
    path = CANONICAL if c['branch']=='A' else SWAPPED
    common = {k: v for k, v in c.items() if k != 'branch'}
    recipe = dict(config=c, paired_common=common, source=source_hashes(), inputs=inputs,
                  parent_recipe=parent['recipe'], parameter_names=names,
                  torch=torch.__version__, numpy=np.__version__,
                  precision='bf16' if device=='cuda' else 'fp32', runtime_scope=scope,
                  tf32=False if device=='cuda' else None)
    lr = optimizer.param_groups[0]['lr']
    step, counts, block_tokens = 0, {}, 0
    state = parent
    if resume:
        cp = json.loads((out/'checkpoint.json').read_text())
        if sha(out/'fullstate.pt') != cp['sha256']:
            raise ValueError('Phase checkpoint hash mismatch')
        state = torch.load(out/'fullstate.pt', map_location='cpu', weights_only=False)
        if state['recipe'] != recipe or state['step'] != cp['step'] or (out/'complete.json').exists():
            raise ValueError('Phase recipe/terminal mismatch')
        step = state['step']
        validate_adam(model, state['optimizer'], parent['step']+step, lr)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        counts, block_tokens = state['counts'], state['block_tokens']
        trace = [json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()] if (out/'trace.jsonl').exists() else []
        if [r['step'] for r in trace] != list(range(1, step+1)):
            raise ValueError('Ambiguous trace/checkpoint position')
    end = c['steps'] if stop_after is None else stop_after
    if type(end) is not int or not step < end <= c['steps']:
        raise ValueError('Stop must follow current step within phase')
    if device == 'cuda':
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible GPU per paired fit')
        if len(state['cuda_rng']) != 1:
            raise ValueError('Exactly one inherited CUDA RNG stream required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        saved_optimizer = optimizer.state_dict()
        model.to(device)
        # Construct after movement; preserve parameter identity/name order and Adam moments.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.95), weight_decay=.1)
        optimizer.load_state_dict(saved_optimizer)
    if not resume:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'recipe.json', recipe)
    # Allocation completed: resume exact parent/phase streams immediately before work.
    restore_rng(state, device)
    model.train()

    def save(snapshot):
        save_state(out, model, optimizer, recipe, step, 0., block_tokens, counts)
        if snapshot:
            destination = out/'snapshots'/f'{step:06d}'
            if destination.exists():
                raise ValueError('Refuse overwriting snapshot')
            destination.mkdir(parents=True)
            # Separate bytes, not a hardlink to mutable latest checkpoint.
            import shutil
            shutil.copyfile(out/'fullstate.pt', destination/'fullstate.pt')
            shutil.copyfile(out/'checkpoint.json', destination/'checkpoint.json')

    if not resume:
        save(True)
    for s in range(step, end):
        ix = ids[s*c['batch']:(s+1)*c['batch']]
        x = rows.batch(ix, device)
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            loss = model.nll(x, path).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite continuation loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step = s+1; block_tokens += 6*x.numel()
        key = ','.join(map(str,path)); counts[key] = counts.get(key, 0)+1
        with (out/'trace.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step, parent_step=parent['step'], rows=ix.tolist(),
                path=path, loss=float(loss.detach()), grad_norm=float(norm), lr=lr))+'\n')
        if step in snapshots or step == end:
            save(step in snapshots)
    if step == c['steps']:
        write_json(out/'complete.json', dict(step=step, parent_step=parent['step'],
            branch=c['branch'], fullstate_sha256=sha(out/'fullstate.pt'),
            input_tokens=len(ids)*model.c.length, scored_tokens=len(ids)*(model.c.length-model.c.prefix),
            block_tokens=block_tokens, path=path, inherited_lr=lr,
            qualification=scope))
    return model, optimizer
