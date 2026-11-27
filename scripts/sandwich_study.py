"""Tied-translator sandwich pilot: warm sandwich, frozen-T middles, composed stack, controls.

Design and decision rules: docs/SANDWICH_20261123.md. One seed (0), three stages, one GPU per job:
  A: sw_warm  E->T->M0->T->H (all train)      | not_warm E->M0->H (all train)
  B: sw_m1/sw_m2 train only M1/M2 inside E->T->M_k->T->H with E,T,H frozen, disjoint rows
     not_m1/not_m2 the same without T
  C: sw_stack / sw_stack_mix compose E->T->M1->T->M2->T->H and train everything
     (mix alternates M1/M2 order); not_stack E->M1->M2->H; joint = same stack from scratch;
     dense5 = five distinct blocks from scratch. joint/dense5 get the sandwich's total
     block-application budget.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import training_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import lr_at
from efficient_llm.warm_order.sandwich import M1, M2, ROUTES, Sandwich
from scripts.anchor_long_queue import stage, supervise
from scripts.anchor_long_study import score
from scripts.loop_screen_study import BATCH, check_parity, inputs

SEED = 0
N1, N2, N3 = 4096, 4096, 8192
NJ = (3 * N1 + 2 * 3 * N2 + 5 * N3) // 5  # joint/dense5 steps at the sandwich's total block applications
CALIBRATION_EVERY = 512
DEVICE = 'cuda'
JOBS = {  # name -> (stage, route used for training, trainable blocks or None, steps, first row)
    'sw_warm': ('A', 'sandwich_warm', None, N1, 0),
    'not_warm': ('A', 'not_warm', None, N1, 0),
    'sw_m1': ('B', 'sandwich_m1', (M1,), N2, N1 * BATCH),
    'sw_m2': ('B', 'sandwich_m2', (M2,), N2, (N1 + N2) * BATCH),
    'not_m1': ('B', 'not_m1', (M1,), N2, N1 * BATCH),
    'not_m2': ('B', 'not_m2', (M2,), N2, (N1 + N2) * BATCH),
    'sw_stack': ('C', 'sandwich_stack', None, N3, (N1 + 2 * N2) * BATCH),
    'sw_stack_mix': ('C', 'sandwich_stack', None, N3, (N1 + 2 * N2) * BATCH),
    'not_stack': ('C', 'not_stack', None, N3, (N1 + 2 * N2) * BATCH),
    'joint': ('C', 'sandwich_stack', None, NJ, 0),
    'dense5': ('C', 'dense5', None, NJ, 0),
}
PARENTS = {'sw_m1': 'sw_warm', 'sw_m2': 'sw_warm', 'not_m1': 'not_warm', 'not_m2': 'not_warm'}
COMPOSE = {'sw_stack': ('sw_warm', 'sw_m1', 'sw_m2'), 'sw_stack_mix': ('sw_warm', 'sw_m1', 'sw_m2'),
           'not_stack': ('not_warm', 'not_m1', 'not_m2')}
EVALS = {  # routes scored on the 512 measurement rows at the end of each job
    'sw_warm': ('sandwich_warm',), 'not_warm': ('not_warm',),
    'sw_m1': ('sandwich_m1',), 'sw_m2': ('sandwich_m2',), 'not_m1': ('not_m1',), 'not_m2': ('not_m2',),
    'sw_stack': ('sandwich_stack', 'sandwich_swap'), 'sw_stack_mix': ('sandwich_stack', 'sandwich_swap'),
    'not_stack': ('not_stack', 'not_swap'), 'joint': ('sandwich_stack', 'sandwich_swap'),
    'dense5': ('dense5', 'dense5_swap'),
}
ZERO_SHOT = {'sw_stack': ('sandwich_stack', 'sandwich_swap', 'sandwich_m1', 'sandwich_m2'),
             'not_stack': ('not_stack', 'not_swap', 'not_m1', 'not_m2')}


def build(config, name):
    torch.manual_seed(SEED)  # identical base init: T=blocks.0, M0..M2=blocks.1..3 everywhere
    return Sandwich(config, with_extra=(name == 'dense5'))


def save_model(path, model):
    torch.save(model.state_dict(), path)
    write_json(path.with_suffix('.json'), dict(path=str(path), sha256=sha(path)))


def load_model(path):
    pin = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != pin['sha256']:
        raise ValueError('Model state changed: '+str(path))
    return torch.load(path, map_location='cpu', weights_only=True)


def evaluate(model, rows, ids, route):
    return score(model, rows, ids, {'v': dict(path=ROUTES[route], skip=None)}, DEVICE)['v']


def initial_model(base, name, config):
    model = build(config, name)
    if name in PARENTS:
        model.load_state_dict(load_model(base/PARENTS[name]/'model.pt'))
    if name in COMPOSE:
        warm, a, b = COMPOSE[name]
        state = load_model(base/warm/'model.pt')
        sa, sb = load_model(base/a/'model.pt'), load_model(base/b/'model.pt')
        for key in state:  # everything a middle job did not own must equal the warm model exactly
            for part, own in ((sa, f'blocks.{M1}.'), (sb, f'blocks.{M2}.')):
                if not key.startswith(own) and not torch.equal(part[key], state[key]):
                    raise ValueError('Frozen parameter drifted in middle job: '+key)
        state.update({k: v for k, v in sa.items() if k.startswith(f'blocks.{M1}.')})
        state.update({k: v for k, v in sb.items() if k.startswith(f'blocks.{M2}.')})
        model.load_state_dict(state)
    return model


def qualify(model, rows, cal, route):
    cpu = Sandwich(model.c, with_extra=len(model.blocks) == 5)
    cpu.load_state_dict(model.state_dict())
    x = rows.batch(cal[:2], 'cpu')
    from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
    with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
        ref = cpu.eval().token_losses(x, ROUTES[route])
    model.to(DEVICE)
    with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        got = model.eval().token_losses(x.to(DEVICE), ROUTES[route]).cpu()
    check_parity(got, ref)


def worker(base, name):
    stage_name, route, trainable, steps, first = JOBS[name]
    out = base/name
    out.mkdir(parents=True, exist_ok=False)
    meta, config, rows, ids, cal, measurement = inputs(SEED)
    if first + steps * BATCH > len(ids):
        raise ValueError('Not enough training rows')
    model = initial_model(base, name, config)
    qualify(model, rows, cal, route)
    values = {}
    if name in ZERO_SHOT:
        for r in ZERO_SHOT[name]:
            values['zeroshot_'+r] = evaluate(model, rows, measurement, r)
    params = model.set_trainable(trainable, embed_head=trainable is None)
    opt = torch.optim.AdamW(params, lr=lr_at(0, steps), betas=(.9, .95), eps=1e-8, weight_decay=0)
    frozen = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}

    def calibration(step, r):
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(dict(step=step, route=r, nll=float(evaluate(model, rows, cal, r).mean()))), file=f)

    calibration(0, route)
    started = time.time()
    with training_runtime(DEVICE), anchor_runtime(DEVICE):
        for step in range(steps):
            for group in opt.param_groups:
                group['lr'] = lr_at(step, steps)
            r = ('sandwich_swap' if step % 2 else 'sandwich_stack') if name == 'sw_stack_mix' else route
            model.train()
            x = rows.batch(ids[first + step*BATCH:first + (step+1)*BATCH], DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model.nll(x, ROUTES[r]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            done = step + 1
            if done % 128 == 0:
                elapsed = time.time() - started
                write_json(out/'progress.json', dict(step=done, total=steps, last_batch=float(loss.detach()),
                                                     elapsed=elapsed, tokens_per_s=done*BATCH*512/elapsed))
            if done % CALIBRATION_EVERY == 0:
                calibration(done, route)
    for k, v in model.named_parameters():
        if k in frozen and not torch.equal(v.detach(), frozen[k]):
            raise ValueError('Frozen parameter changed during training: '+k)
    save_model(out/'model.pt', model)
    for r in EVALS[name]:
        values[r] = evaluate(model, rows, measurement, r)
    np.savez_compressed(out/'scores.npz', row_ids=measurement, **values)
    write_json(out/'summary.json', dict(
        name=name, stage=stage_name, route=route, steps=steps, first_row=first, trainable=trainable,
        block_applications_per_token=len(ROUTES[route]), train_seconds=time.time()-started,
        nll={k: float(v.mean()) for k, v in values.items()}, score_sha256=sha(out/'scores.npz'),
        trainable_parameters=sum(p.numel() for p in params)))


def prepare(base):
    meta, config, rows, ids, cal, measurement = inputs(SEED)
    last = max(first + steps * BATCH for _, _, _, steps, first in JOBS.values())
    if last > len(ids) or np.intersect1d(np.asarray(ids[:last]), np.concatenate([cal, measurement])).size:
        raise ValueError('Training rows insufficient or overlapping evaluation rows')
    write_json(base/'prepared.json', dict(config=config.__dict__, jobs=JOBS, n=dict(N1=N1, N2=N2, N3=N3, NJ=NJ)))


def experiment(base):
    module = 'scripts.sandwich_study'
    for stage_name in 'ABC':
        names = [n for n, j in JOBS.items() if j[0] == stage_name]
        stage(base, 'train', [(SEED, n) for n in names], module)
    report, names = {}, ['prepared.json', 'execution/source.json']
    for n in JOBS:
        s = json.loads((base/n/'summary.json').read_text())
        if sha(base/n/'scores.npz') != s['score_sha256']:
            raise ValueError('Scores changed: '+n)
        report[n] = s
        names += [f'{n}/summary.json', f'{n}/scores.npz', f'{n}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    import tarfile
    archive = base/'export'/'sandwich_results.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))
    backup = base/'backup'
    backup.mkdir()
    for n in ('sw_stack', 'joint', 'dense5'):
        shutil.copyfile(base/n/'model.pt', backup/f'{n}.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'train'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(SEED,))
    p.add_argument('--arm', choices=tuple(JOBS))
    p.add_argument('--handoff', type=Path)
    p.add_argument('--handoff-sha256')
    a = p.parse_args()
    if a.action == 'queue':
        try:
            supervise(a.base, prepare, experiment, a.handoff, a.handoff_sha256)
        except Exception as err:
            write_json(a.base/'failure.json', dict(reason=repr(err)))
            raise
    else:
        worker(a.base, a.arm)
