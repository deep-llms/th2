"""Synthetic DOT benchmark s11 (docs/DOT_SYNTH_20261124.md): completes 4 seeds for the random-depth fixed order and pure
loop (s10 had 2 each), adds the greedy-body router without prior under random depth, and a 5th seed. Background, s10:
random training depth 2..8 made deep inference safe and confidence halting adaptive. Background, s09: s08: with the body on the greedy path the prior router
trained on 4/4 seeds and beat the fixed order, but it converged to one static order per seed, mostly repeating one
middle. s09: (a) control = a sandwich that always uses middle 0 (a pure loop) on chain, seeds 0-3; (b) the greedy-body
prior router on opseq (different operations per input, where input-dependent routing should matter), seeds 0-3,
against the s02/s04 opseq arms.

Eight arms, one GPU each; every worker trains a fresh model per task (chain, opseq) from scratch on on-line generated
data at the training difficulties, then measures answer accuracy at every test difficulty (including harder than
training) and, for sandwich arms, with more steps at inference than in training. Baselines are standard transformers
(dense4: same parameters as every sandwich arm; dense8: twice the parameters).
"""
import argparse
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.dot_synth import TASKS, VOCAB, to_device
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import SPECS, lr_at, make_model, native_path
from efficient_llm.warm_order.model import Config
from scripts.anchor_long_queue import stage, supervise

MODEL = dict(d=256, heads=4, kv_heads=2, head_dim=64, ff=1024, vocab=VOCAB)
STEPS, BATCH, PEAK_LR, WARMUP = 8000, 512, 1e-3, 256
EVAL_ROWS, EVAL_SEED, LOG_EVERY = 512, 777, 500
HALT_THRESHOLDS = (.9, .99, 2.)  # 2. = never reached: pick the most confident depth
CURRICULUM = .5  # fraction of training over which the training levels are opened one by one (0 = all from start)
RUN_TASKS = ('chain',)
TASK_SETTINGS = {'chain': (8000, .5), 'opseq': (5000, 0.)}  # (steps, curriculum): chain as s03-s08, opseq as s02/s04
ARM_TASKS = {'syn_loop1_d4': ('chain',), 'syn_prior_d8g': ('opseq',)}  # others: RUN_TASKS (chain)
DEVICE = 'cuda'  # tests switch this to 'cpu'
JOBS = [(1, 'syn_fixed_d4r'), (3, 'syn_fixed_d4r'), (2, 'syn_loop1_d4r'), (3, 'syn_loop1_d4r'),
        (0, 'syn_route_d4gr'), (1, 'syn_route_d4gr'), (4, 'syn_fixed_d4r'), (4, 'dense4')]


def config_for(task):
    return Config(**MODEL, **TASKS[task]['spec']())


def applications(arm, depth=None):
    sw = SPECS[arm].get('sandwich')
    if sw is None:
        return SPECS[arm]['blocks']
    return 2 * (depth or sw['depth']) + 1


def answer_mask(task, device):
    """bool (L - prefix,) of answer positions inside the scored suffix, and the answer units (index lists)."""
    spec, units = TASKS[task]['spec'](), TASKS[task]['answers']()
    mask = torch.zeros(spec['length'] - spec['prefix'], dtype=torch.bool, device=device)
    mask[[i for u in units for i in u]] = True
    return mask, units


def levels_at(levels, step, steps=None, curriculum=None):
    """With a curriculum, difficulty level i (sorted) joins at step i * curriculum * steps / len(levels)."""
    steps = STEPS if steps is None else steps
    curriculum = CURRICULUM if curriculum is None else curriculum
    if not curriculum:
        return levels
    n = 1 + min(len(levels) - 1, int(step / (curriculum * steps / len(levels))))
    return tuple(sorted(levels)[:n])


def eval_sets(task):
    t = TASKS[task]
    rng = np.random.default_rng(EVAL_SEED)
    return {k: t['batch'](EVAL_ROWS, (k,), rng)[0] for k in t['test']}


def row_keys(task, tokens):
    """Per-row description of the instance for routing diagnostics (opseq: the operation sequence)."""
    if task == 'opseq':
        from efficient_llm.warm_order.dot_synth import M_MAX, NOP, OP0, OPS
        return ['-'.join(OPS[t - OP0] for t in row[:M_MAX] if t != NOP) for row in tokens]
    return None


def route_summary(paths, keys):
    """paths: list of per-row path tuples at the first answer-predicting position."""
    from collections import Counter
    total = Counter(paths)
    out = dict(distinct=len(total), top=[['-'.join(map(str, q)), n / len(paths)] for q, n in total.most_common(5)])
    if keys is not None:
        by = {}
        for q, key in zip(paths, keys):
            by.setdefault(key, Counter())[q] += 1
        # how deterministic is the path given the instance's operation sequence (1 = a fixed path per sequence)
        out['modal_given_key'] = sum(c.most_common(1)[0][1] for c in by.values()) / len(paths)
        out['per_key'] = {key: ['-'.join(map(str, c.most_common(1)[0][0])), c.most_common(1)[0][1] / sum(c.values())]
                          for key, c in sorted(by.items())[:24]}
    return out


@torch.no_grad()
def measure(model, arm, task, sets, depth=None, routes=False, units_out=None):
    """Per difficulty: unit accuracy (a unit = one chain answer, or the whole opseq string), answer-token accuracy,
    answer loss, (STOP arms) middles used at the answer-predicting positions and (routes=True) path statistics."""
    model.eval()
    if hasattr(model, 'test_depth'):
        model.test_depth = depth
    p = model.c.prefix
    mask, units = answer_mask(task, DEVICE)
    out = {}
    trace_ok = routes and hasattr(model, 'trace')
    try:
        with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
            for k, tokens in sets.items():
                exact, tok, loss, used, paths = [], [], [], [], []
                for i in range(0, len(tokens), 256):
                    x = to_device(tokens[i:i + 256], DEVICE)
                    if trace_ok:
                        model.trace = []
                    h = model.readout_hidden(model.hidden(x, native_path(arm)))[:, p - 1:-1]
                    if trace_ok:
                        steps = torch.stack(model.trace, -1)[:, p - 1].cpu().tolist()  # first answer-predicting position
                        paths += [tuple(q) for q in steps]
                        model.trace = None
                    logits = F.linear(h, model.embed.weight).float()
                    y = x[:, p:]
                    hit = logits.argmax(-1) == y
                    unit_hit = torch.stack([hit[:, u].all(1) for u in units], 1)
                    exact.append(unit_hit.float().mean(1))
                    if units_out is not None:  # per unit: correct, confidence (min max-prob over its tokens), prediction
                        conf = logits.softmax(-1).max(-1).values
                        pred = logits.argmax(-1)
                        units_out.setdefault(str(k), []).append(
                            (unit_hit.cpu(), torch.stack([conf[:, u].min(1).values for u in units], 1).cpu(),
                             torch.stack([_unit_code(pred[:, u]) for u in units], 1).cpu()))
                    tok.append(hit[:, mask].float().mean(1))
                    loss.append(F.cross_entropy(logits.transpose(1, 2), y, reduction='none')[:, mask].mean(1))
                    if getattr(model, 'last_used', None) is not None:
                        used.append(model.last_used[:, p - 1:-1][:, mask].float().mean(1))
                row = dict(exact=float(torch.cat(exact).mean()), token=float(torch.cat(tok).mean()),
                           loss=float(torch.cat(loss).mean()))
                if used:
                    row['answer_depth'] = float(torch.cat(used).mean())
                if paths:
                    row['routes'] = route_summary(paths, row_keys(task, tokens))
                out[str(k)] = row
    finally:
        if hasattr(model, 'test_depth'):
            model.test_depth = None
        if trace_ok:
            model.trace = None
    return out


def _unit_code(tokens):
    """Collision-free integer code of a unit's predicted tokens (VOCAB < 2**7, units have at most 8 tokens)."""
    code = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    for i in range(tokens.shape[1]):
        code = code * 128 + tokens[:, i]
    return code


def agreement_halt(per_depth):
    """Halting by convergence: per unit, stop at the first depth whose prediction equals the previous depth's (the
    answer stopped changing), else take the deepest depth. Needs no threshold and no trained STOP."""
    depths = sorted(per_depth)
    res = {}
    for k in per_depth[depths[0]]:
        hit = torch.stack([torch.cat([u[0] for u in per_depth[d][k]]) for d in depths])
        pred = torch.stack([torch.cat([u[2] for u in per_depth[d][k]]) for d in depths])
        same = torch.zeros_like(hit)
        same[1:] = pred[1:] == pred[:-1]
        choice = torch.where(same.any(0), same.float().argmax(0), torch.full_like(same[0], len(depths) - 1, dtype=torch.long))
        chosen = hit.gather(0, choice.unsqueeze(0)).squeeze(0)
        res[k] = dict(exact=float(chosen.float().mean()), depth=float(torch.tensor(depths, dtype=torch.float)[choice].mean()))
    return res


def adaptive_depth(per_depth, thresholds=HALT_THRESHOLDS):
    """Inference-time halting by confidence: per unit, take the first depth (ascending) whose answer confidence reaches
    the threshold, else the most confident depth. per_depth: {depth: {level: [(hit, conf), ...]}}."""
    depths = sorted(per_depth)
    out = {}
    for tau in thresholds:
        res = {}
        for k in per_depth[depths[0]]:
            hit = torch.stack([torch.cat([u[0] for u in per_depth[d][k]]) for d in depths])  # (depths, rows, units)
            conf = torch.stack([torch.cat([u[1] for u in per_depth[d][k]]) for d in depths])
            reach = conf >= tau
            first = torch.where(reach.any(0), reach.float().argmax(0), conf.argmax(0))
            chosen = hit.gather(0, first.unsqueeze(0)).squeeze(0)
            res[k] = dict(exact=float(chosen.float().mean()),
                          depth=float(torch.tensor(depths, dtype=torch.float)[first].mean()),
                          oracle=float(hit.any(0).float().mean()))
        out[str(tau)] = res
    return out


def train_task(out, seed, arm, task):
    steps, curriculum = TASK_SETTINGS.get(task, (STEPS, CURRICULUM))
    out.mkdir(parents=True, exist_ok=False)
    config = config_for(task)
    torch.manual_seed(seed)
    model = make_model(config, arm).to(DEVICE)
    if hasattr(model, 'generator'):
        model.generator = torch.Generator(device=DEVICE).manual_seed(20261124 + seed)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(.9, .95), eps=1e-8, weight_decay=0)
    rng = np.random.default_rng(1000 * seed + list(TASKS).index(task))
    sets = eval_sets(task)
    train_levels = TASKS[task]['train']
    mask, _ = answer_mask(task, DEVICE)
    if hasattr(model, 'answer_mask'):
        model.answer_mask = mask
    log, started = [], time.time()
    with training_runtime(DEVICE), anchor_runtime(DEVICE):
        for step in range(steps):
            for group in opt.param_groups:
                group['lr'] = lr_at(step, steps, peak=PEAK_LR, warmup=WARMUP)
            model.train_progress = step / steps
            rand = (SPECS[arm].get('sandwich') or {}).get('rand_depth')
            if rand is not None:  # train each step at a random depth (inference-time depth robustness)
                model.test_depth = int(rng.integers(rand[0], rand[1] + 1))
            model.train()
            x = to_device(TASKS[task]['batch'](BATCH, levels_at(train_levels, step, steps, curriculum), rng)[0], DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                token_losses = model.token_losses(x, native_path(arm))
                loss = token_losses[:, mask].mean()  # answers only (chain queries are unpredictable)
                if getattr(model, 'extra_loss', None) is not None:
                    extra = model.extra_loss(x, token_losses.detach())
                    if extra is not None:
                        loss = loss + extra
                if model.aux is not None:
                    loss = loss + model.aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            done = step + 1
            if done % LOG_EVERY == 0 or done == steps:
                quick = measure(model, arm, task, {k: v[:128] for k, v in sets.items() if k in train_levels})
                log.append(dict(step=done, loss=float(token_losses.detach()[:, mask].mean()), elapsed=time.time() - started,
                                train_exact={k: v['exact'] for k, v in quick.items()}))
                write_json(out/'progress.json', log[-1] | dict(total=steps))
    sw = SPECS[arm].get('sandwich')
    per_depth = {}
    results = {'native': measure(model, arm, task, sets, routes=True,
                                 units_out=per_depth.setdefault(sw['depth'], {}) if sw else None)}
    if sw is not None:
        for extra_steps in (2, 4, 8, 12):
            d = sw['depth'] + extra_steps
            results[f'depth{d}'] = measure(model, arm, task, sets, d, units_out=per_depth.setdefault(d, {}))
        results['adaptive'] = adaptive_depth(per_depth)
        results['adaptive']['agree'] = agreement_halt(per_depth)
    torch.save(dict(model=model.state_dict(), arm=arm, task=task, seed=seed), out/'terminal.pt')
    summary = dict(seed=seed, arm=arm, task=task, params=params, applications=applications(arm), steps=steps, curriculum=curriculum,
                   batch=BATCH, train_seconds=time.time() - started, results=results, log=log,
                   terminal_sha256=sha(out/'terminal.pt'))
    write_json(out/'summary.json', summary)


def tasks_for(arm):
    return ARM_TASKS.get(arm, RUN_TASKS)


def worker(base, seed, arm):
    for task in tasks_for(arm):
        train_task(base/f's{seed}'/arm/task, seed, arm, task)


def prepare(base):
    write_json(base/'prepared.json', dict(jobs=JOBS, model=MODEL, steps=STEPS, batch=BATCH, peak_lr=PEAK_LR,
                                          eval_rows=EVAL_ROWS, eval_seed=EVAL_SEED, tasks={a: list(tasks_for(a)) for _, a in JOBS}, task_settings=TASK_SETTINGS))


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.dot_synth_study')
    names, report = ['prepared.json', 'execution/source.json'], {}
    for seed, arm in JOBS:
        for task in tasks_for(arm):
            d = f's{seed}/{arm}/{task}'
            s = json.loads((base/d/'summary.json').read_text())
            if sha(base/d/'terminal.pt') != s['terminal_sha256']:
                raise ValueError('terminal changed: ' + d)
            report[d] = {k: v for k, v in s.items() if k != 'log'}
            names.append(f'{d}/summary.json')
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'synth_results.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'train'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2, 3, 4))
    p.add_argument('--arm', choices=sorted({a for _, a in JOBS}))
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
        worker(a.base, a.seed, a.arm)
