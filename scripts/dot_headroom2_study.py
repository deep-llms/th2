"""Learnable routing headroom of trained DOT sandwich models (docs/DOT_SANDWICH_20261123.md, headroom section).

Evaluation only. The per-token oracle of dot_headroom_study is biased low by min selection over noisy evaluations.
Here a small linear path policy is fitted on half of the rows to minimise the EXPECTED per-token loss over the path
table (softmax weights x uniform-path losses), from the token state after the first translator of the model's own
forward (a causal, deployable feature), and is evaluated on the other half (argmax path, no access to held-out
losses). Held-out gain over the model's own routing / the best static path = learnable headroom (approximate:
path losses come from uniform forcing).
"""
import argparse
import itertools
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import SPECS, make_model
from efficient_llm.warm_order.model import Config
from scripts.anchor_long_queue import stage, supervise
import scripts.loop_screen_study as loop

OUT = Path('/mnt/local/_outputs/deep-llms_th2')
MODELS = [('dot_sandwich_20261123_a03', 0, 'sw_fixed'), ('dot_sandwich_20261123_a03', 1, 'sw_fixed'),
          ('dot_sandwich_20261123_a03', 0, 'sw_route'), ('dot_sandwich_20261123_a03', 1, 'sw_route'),
          ('dot_sandwich_20261124_a04', 0, 'sw_prior'), ('dot_sandwich_20261124_a04', 1, 'sw6_route'),
          ('dot_sandwich_20261124_a05', 0, 'sw_mpg'), ('dot_sandwich_20261124_a05', 0, 'sw_branch')]
ROWS, SAMPLED_PATHS = 512, 64
FIT_STEPS, FIT_LR, FIT_WD = 400, 1e-2, 1e-4
DEVICE = 'cuda'
PATH = (0, 1, 2, 3)


def checkpoint(run, seed, arm):
    pin = json.loads((OUT/run/f's{seed}'/arm/'terminal.json').read_text())
    if sha(pin['path']) != pin['sha256']:
        raise ValueError(f'Checkpoint changed: {run}/{seed}/{arm}')
    return pin


def paths_for(k, depth, seed):
    every = list(itertools.product(range(k), repeat=depth))
    if len(every) <= SAMPLED_PATHS:
        return every
    rng = np.random.default_rng(20261124 + seed)
    pick = [every[i] for i in rng.choice(len(every), SAMPLED_PATHS, replace=False)]
    fixed = tuple(s % 3 for s in range(depth))
    return [fixed] + [p for p in pick if p != fixed]


def prepare(base):
    write_json(base/'prepared.json', dict(models=MODELS, rows=ROWS, checkpoints={f'{r}/s{s}/{a}': checkpoint(r, s, a)['sha256']
                                                                                 for r, s, a in MODELS}))


@torch.no_grad()
def worker(base, slot):
    run, seed, arm = MODELS[slot]
    out = base/f'{run}-s{seed}-{arm}'
    out.mkdir(parents=True, exist_ok=False)
    pin = checkpoint(run, seed, arm)
    config = Config(**json.loads((OUT/run/'prepared.json').read_text())['config'])
    model = make_model(config, arm)
    state = torch.load(pin['path'], map_location='cpu', weights_only=False)
    if state['arm'] != arm:
        raise ValueError('Checkpoint identity mismatch')
    model.load_state_dict(state['model'])
    model = model.to(DEVICE).eval()
    meta, _, rows, _, _, measurement = loop.inputs(seed)
    ids = np.asarray(measurement[:ROWS])
    depth, k = model.sw['depth'], len(model.middles)
    paths = paths_for(k, depth, seed)

    def losses(forced_path=None):
        chunks = []
        with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
            for i in range(0, len(ids), 4):
                x = rows.batch(ids[i:i + 4], DEVICE)
                model.forced = None if forced_path is None else {s: torch.full(x.shape, c, dtype=torch.long, device=DEVICE)
                                                                 for s, c in enumerate(forced_path)}
                try:
                    chunks.append(model.token_losses(x, PATH).double().cpu())
                finally:
                    model.forced = None
        return torch.cat(chunks)  # (rows, L - prefix)

    own = losses()
    handle_rows = []
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        T = model.blocks[0]
        for i in range(0, len(ids), 4):
            x = rows.batch(ids[i:i + 4], DEVICE)
            pos = torch.arange(x.shape[1], device=DEVICE).expand_as(x)
            seg = torch.zeros_like(x)
            h = model.embed(x)
            choice, _ = model._choose(h, 0, x.shape)
            h1 = model._t(T, h, model._condition(choice, h.dtype), pos, seg)  # state after the first translator
            handle_rows.append(h1[:, model.c.prefix - 1:-1].float().cpu())
    feat = torch.cat(handle_rows)  # (rows, L - prefix, d)
    table = torch.stack([losses(p) for p in paths])  # (paths, rows, L - prefix)
    n_train = len(ids) // 2
    d = feat.shape[-1]
    X = torch.nn.functional.rms_norm(feat, (d,), eps=1e-6)
    Xtr, Xte = X[:n_train].reshape(-1, d).to(DEVICE), X[n_train:].reshape(-1, d).to(DEVICE)
    Ltr = table[:, :n_train].reshape(len(paths), -1).t().float().to(DEVICE)  # (tokens, paths)
    Lte = table[:, n_train:].reshape(len(paths), -1).t().float().to(DEVICE)
    W = torch.zeros(d, len(paths), device=DEVICE, requires_grad=True)
    b = torch.zeros(len(paths), device=DEVICE, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=FIT_LR, weight_decay=FIT_WD)
    with torch.enable_grad():
        for _ in range(FIT_STEPS):
            opt.zero_grad()
            expected = (torch.softmax(Xtr @ W + b, -1) * Ltr).sum(-1).mean()  # cost-sensitive: expected loss
            expected.backward()
            opt.step()
    with torch.no_grad():
        pick_te = (Xte @ W + b).argmax(-1)
        pick_tr = (Xtr @ W + b).argmax(-1)
    learned_te = float(Lte.gather(1, pick_te[:, None]).mean())
    learned_tr = float(Ltr.gather(1, pick_tr[:, None]).mean())
    fixed_idx = paths.index(tuple(s % 3 for s in range(depth)))
    static_tr = Ltr.mean(0)
    best_static = int(static_tr.argmin())  # chosen on the training half
    own_te = float(own[n_train:].mean())
    summary = dict(run=run, seed=seed, arm=arm, rows=len(ids), paths=len(paths), middles=k, depth=depth,
                   own_heldout=own_te, fixed_heldout=float(Lte[:, fixed_idx].mean()),
                   best_static_heldout=float(Lte[:, best_static].mean()), best_static_path=list(paths[best_static]),
                   learned_heldout=learned_te, learned_train=learned_tr,
                   oracle_heldout=float(Lte.min(1).values.mean()),
                   learned_path_entropy=float(torch.distributions.Categorical(
                       probs=torch.bincount(pick_te, minlength=len(paths)).float() / len(pick_te)).entropy()),
                   learned_top_path_share=float(torch.bincount(pick_te, minlength=len(paths)).max() / len(pick_te)))
    np.savez_compressed(out/'rows.npz', own=own[n_train:].mean(-1).numpy(),
                        learned=Lte.gather(1, pick_te[:, None]).reshape(len(ids) - n_train, -1).mean(-1).cpu().numpy(),
                        best_static=Lte[:, best_static].reshape(len(ids) - n_train, -1).mean(-1).cpu().numpy())
    summary['rows_sha256'] = sha(out/'rows.npz')
    write_json(out/'summary.json', summary)


def experiment(base):
    stage(base, 'eval', [(0, f'm{i}') for i in range(len(MODELS))], 'scripts.dot_headroom2_study')
    names = ['prepared.json', 'execution/source.json']
    for run, seed, arm in MODELS:
        d = f'{run}-s{seed}-{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if sha(base/d/'rows.npz') != s['rows_sha256']:
            raise ValueError('rows changed: '+d)
        names += [f'{d}/summary.json', f'{d}/rows.npz']
    archive = base/'export'/'headroom2.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'eval'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0,))
    p.add_argument('--arm', choices=[f'm{i}' for i in range(len(MODELS))])
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
        worker(a.base, int(a.arm[1:]))
