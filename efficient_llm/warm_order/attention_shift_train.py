"""Bounded donor-free bias fitting with immutable parent and exact small-state resume.

CUDA requires the separate actual-checkpoint parity and exact-resume admission.
No donor model, activation target, router or parent optimizer is used in fitting.
"""
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .attention_shift import PATHS, freeze, fingerprint, shifted
from .adaptation_train import load_parent
from .adaptation_reservation import validate_binding
from .adaptation_runtime import training_runtime, training_policy, measurement_runtime
from .data import Rows, sha
from .exposure_recovery import restore_rng
from .horizon_train import amp, write_json


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ['attention_shift', 'attention_shift_train', 'attention_shift_eval', 'attention_shift_node', 'adaptation_train',
             'adaptation_reservation', 'adaptation_runtime', 'adaptation_gpu_probe', 'model', 'data',
             'horizon_train', 'exposure_recovery']
    paths = ['efficient_llm/warm_order/'+n+'.py' for n in names]+['efficient_llm/dot_pool/model.py']
    return {p: sha(root/p) for p in paths}


def validate(c):
    if (c.get('schema') != 1 or c.get('domain') != 'dclm' or c.get('target') != 2 or
            c.get('branch') not in PATHS or c.get('seed') not in (0, 1) or
            c.get('lr') != .001 or c.get('lr_policy') != 'constant' or
            c.get('betas') != [.9, .95] or c.get('weight_decay') != 0 or
            c.get('clip_grad_norm') != 1 or type(c.get('fixture_only')) is not bool):
        raise ValueError('Unregistered correction configuration')
    for k in ('steps', 'batch', 'threads', 'monitor_every', 'monitor_rows'):
        if type(c.get(k)) is not int or c[k] <= 0:
            raise ValueError('Positive integer required: '+k)
    if (not c.get('snapshots') or c['snapshots'] != sorted(set(c['snapshots'])) or c['snapshots'][0] != 0 or
            c['snapshots'][-1] != c['steps'] or any(type(s) is not int for s in c['snapshots'])):
        raise ValueError('Snapshots must span zero to terminal step')
    if not c['fixture_only'] and (c['steps'], c['batch'], c['snapshots'], c['monitor_every'], c['monitor_rows']) != (500, 32, [0, 100, 500], 100, 32):
        raise ValueError('Production budget differs from registered trial')


def selections(c, length):
    validate(c)
    if sha(c['data_recipe']['path']) != c['data_recipe']['sha256']:
        raise ValueError('Data recipe hash mismatch')
    previous = read(c['data_recipe']['path'])['config']
    keys = ['parent', 'parent_hashes', 'data', 'indices', 'calibration_indices', 'confirmation_indices', 'expected_inputs']
    if any(previous.get(k) != c.get(k) for k in keys):
        raise ValueError('Prior data recipe binding mismatch')
    if not c['fixture_only']:
        validate_binding(previous)
        if previous['row_reservation'] != c['row_reservation'] or c['row_reservation']['seed'] != c['seed']:
            raise ValueError('Original reservation seed mismatch')
    paths = [c['data']+'.bin', c['data']+'.starts']+[c[k] for k in ('indices','calibration_indices','confirmation_indices')]
    if {p: sha(p) for p in paths} != c['expected_inputs']:
        raise ValueError('Actual input identity mismatch')
    rows = Rows(c['data'], length)
    all_train, cal, confirm = [np.load(c[k], allow_pickle=False) for k in ('indices','calibration_indices','confirmation_indices')]
    for a in (all_train, cal, confirm):
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a) or
                len(np.unique(a)) != len(a) or a.min() < 0 or a.max() >= len(rows.ids)):
            raise ValueError('Unique in-range row IDs required')
    if len(all_train) != previous['steps']*previous['batch'] or c['steps']*c['batch'] > len(all_train) or len(cal) < c['monitor_rows']:
        raise ValueError('Registered row extent mismatch')
    if any(np.intersect1d(a,b).size for a,b in ((all_train,cal),(all_train,confirm),(cal,confirm))):
        raise ValueError('Training/calibration/confirmation overlap')
    used = all_train[:c['steps']*c['batch']]
    for a in (used, cal, confirm):
        for start in range(0,len(a),4096):
            if np.any(rows.starts[a[start:start+4096],1:]):
                raise ValueError('Cross-boundary row')
    import hashlib
    binding = dict(training_array_sha256=c['expected_inputs'][c['indices']], selected_prefix_rows=len(used),
                   selected_prefix_int64_sha256=hashlib.sha256(used.astype('<i8').tobytes()).hexdigest(),
                   calibration_rows=len(cal), confirmation_rows=len(confirm), document_freshness_verified=False)
    return rows, used, cal, confirm, binding


def validate_optimizer(state, step, width):
    groups = state['param_groups']
    if len(groups) != 1 or groups[0]['params'] != [0]:
        raise ValueError('Expected one correction parameter')
    g = groups[0]
    if (g['lr'] != .001 or tuple(g['betas']) != (.9,.95) or g['eps'] != 1e-8 or
            g['weight_decay'] != 0 or g.get('amsgrad',False) or g.get('maximize',False)):
        raise ValueError('Correction Adam policy mismatch')
    if step == 0:
        if state['state']: raise ValueError('Fresh correction optimizer must have no moments')
        return
    if set(state['state']) != {0}: raise ValueError('Correction optimizer state missing')
    s = state['state'][0]
    if set(s) != {'step','exp_avg','exp_avg_sq'} or float(s['step']) != step:
        raise ValueError('Correction Adam counter mismatch')
    for k in ('exp_avg','exp_avg_sq'):
        if s[k].shape != (width,) or s[k].dtype != torch.float32 or not torch.isfinite(s[k]).all():
            raise ValueError('Correction Adam moments invalid')
    if (s['exp_avg_sq'] < 0).any(): raise ValueError('Negative second moment')


def monitor(model, bias, rows, ids, device):
    mode=model.training;model.eval()
    # No stochastic evaluation; fork restores CPU/CUDA RNG even after exceptions.
    devices=[torch.cuda.current_device()] if device=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad(), measurement_runtime(device):
            sums={b:0. for b in PATHS};count=0
            for start in range(0,len(ids),2):
                x=rows.batch(ids[start:start+2],device)
                for b in PATHS:sums[b]+=float(shifted(model,x,bias,b).double().sum())
                count+=len(x)
            return dict(nll={b:v/count for b,v in sums.items()},rows=count)
    finally:model.train(mode)


def admission_check(c, admission, stop_after, qualification):
    if c['fixture_only'] or not admission:
        raise ValueError('Actual GPU correction admission required; fixtures cannot admit CUDA')
    if sha(admission['path']) != admission['sha256']:
        raise ValueError('Correction admission hash mismatch')
    r=read(admission['path'])
    if (r['source']!=source_hashes() or r['config']!=c or not r['pass_'] or
            r['policy']!=training_policy('cuda') or r['fixture_only']):
        raise ValueError('Correction admission source/config/policy mismatch')
    if qualification:
        if r['kind']!='numerical' or type(stop_after) is not int or not 0<stop_after<=40:
            raise ValueError('Qualification must be bounded to40 updates')
    elif r['kind']!='resume' or r['step']!=40 or not r['exact_resume']:
        raise ValueError('Full correction fit requires exact40-step qualification')


def train(c, out, *, stop_after=None, resume=False, device='cpu', admission=None, qualification=False):
    validate(c)
    if device == 'cuda':admission_check(c,admission,stop_after,qualification)
    elif device!='cpu':raise ValueError('Explicit cpu/cuda device required')
    with training_runtime(device):
        return _train(c,out,stop_after=stop_after,resume=resume,device=device)


def _train(config,out,*,stop_after,resume,device):
    c=copy.deepcopy(config);out=Path(out)
    if not resume and out.exists() and any(out.iterdir()):raise ValueError('Fresh output required')
    torch.set_num_threads(c['threads'])
    model, old_optimizer, parent, _ = load_parent(c['parent'],c['parent_hashes'])
    del old_optimizer
    if parent['recipe']['config']['seed']!=c['seed']:raise ValueError('Parent seed mismatch')
    rows,ids,cal,confirm,row_binding=selections(c,model.c.length)
    freeze(model);original=fingerprint(model);model.to(device)
    bias=torch.nn.Parameter(torch.zeros(model.c.d,device=device))
    opt=torch.optim.AdamW([bias],lr=c['lr'],betas=(.9,.95),eps=1e-8,weight_decay=0.)
    recipe=dict(config=c,source=source_hashes(),parent_fingerprint=original,row_binding=row_binding,
                torch=torch.__version__,numpy=np.__version__,policy=training_policy(device),
                parameter_names=['attention_shift'],parent_step=parent['step'],trainable_parameters=model.c.d,
                donor_loaded=False,parent_optimizer_inherited=False)
    step=0;state=parent
    if resume:
        cp=read(out/'checkpoint.json')
        if sha(out/'fullstate.pt')!=cp['sha256']:raise ValueError('Correction checkpoint hash mismatch')
        state=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if state['recipe']!=recipe or state['step']!=cp['step'] or (out/'complete.json').exists():
            raise ValueError('Correction recipe/terminal mismatch')
        step=state['step'];validate_optimizer(state['optimizer'],step,model.c.d)
        if state['bias'].shape!=bias.shape or state['bias'].dtype!=bias.dtype or not torch.isfinite(state['bias']).all():
            raise ValueError('Invalid correction state')
        with torch.no_grad():bias.copy_(state['bias'])
        opt.load_state_dict(state['optimizer'])
        trace=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
        if [v['step'] for v in trace]!=list(range(1,step+1)):raise ValueError('Trace/checkpoint ambiguity')
        validation=[json.loads(s) for s in (out/'validation.jsonl').read_text().splitlines()]
        expected=[0]+list(range(c['monitor_every'],step+1,c['monitor_every']))
        if [v['step'] for v in validation]!=expected:raise ValueError('Monitor/checkpoint ambiguity')
    end=c['steps'] if stop_after is None else stop_after
    if type(end) is not int or not step<end<=c['steps']:raise ValueError('Invalid stop step')
    if not resume:
        out.mkdir(parents=True,exist_ok=True);write_json(out/'recipe.json',recipe)
    restore_rng(state,device);model.train()

    def save(snapshot):
        validate_optimizer(opt.state_dict(),step,model.c.d)
        saved=dict(recipe=recipe,step=step,bias=bias.detach().cpu().clone(),optimizer=opt.state_dict(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                   numpy_rng=np.random.get_state(),python_rng=random.getstate())
        tmp=out/'fullstate.tmp';torch.save(saved,tmp);os.replace(tmp,out/'fullstate.pt')
        write_json(out/'checkpoint.json',dict(step=step,sha256=sha(out/'fullstate.pt'),bytes=(out/'fullstate.pt').stat().st_size))
        if snapshot:
            import shutil
            d=out/'snapshots'/f'{step:06d}';d.mkdir(parents=True,exist_ok=False)
            for name in ('fullstate.pt','checkpoint.json'):shutil.copyfile(out/name,d/name)

    def record_monitor():
        v=dict(step=step,**monitor(model,bias,rows,cal[:c['monitor_rows']],device))
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')

    if not resume:record_monitor();save(True)
    start=time.monotonic()
    for s in range(step,end):
        ix=ids[s*c['batch']:(s+1)*c['batch']];x=rows.batch(ix,device);opt.zero_grad(set_to_none=True)
        with amp(device):loss=shifted(model,x,bias,c['branch']).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite correction loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([bias],1.,error_if_nonfinite=True);opt.step()
        if not torch.isfinite(bias).all():raise ValueError('Nonfinite correction')
        step=s+1
        with (out/'trace.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,rows=ix.tolist(),branch=c['branch'],lr=c['lr'],loss=float(loss.detach()),
                                   grad_norm=float(norm),bias_norm=float(bias.detach().norm())),allow_nan=False)+'\n')
        if step%c['monitor_every']==0:record_monitor()
        if step in c['snapshots'] or step==end:save(step in c['snapshots'])
    if fingerprint(model)!=original or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen body changed')
    if step==c['steps']:
        write_json(out/'complete.json',dict(step=step,branch=c['branch'],fullstate_sha256=sha(out/'fullstate.pt'),
                   parent_fingerprint=original,frozen_body_unchanged=True,trainable_parameters=model.c.d,
                   input_tokens=len(ids)*model.c.length,scored_tokens=len(ids)*(model.c.length-model.c.prefix),
                   elapsed_this_invocation=time.monotonic()-start,
                   scope='CPU_fixture_only' if c['fixture_only'] else 'Frozen_body_donor_free_correction',
                   device=device))
    return model,bias,opt
