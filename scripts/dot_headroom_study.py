"""Routing headroom of trained DOT sandwich models (docs/DOT_SANDWICH_20261123.md, headroom section).

Evaluation only. For each terminal, on the first ROWS registered measurement rows, score the model's own routing
and every uniform path (all tokens forced through the same middle at each step; every path for 3 middles, a seeded
sample plus the fixed order for 6 middles). Report: own routing, the fixed order, the best single static path, the
per-sequence oracle and the per-token oracle over these paths. Uniform forcing keeps attention contexts consistent,
so the per-token oracle is an approximation of a token-level counterfactual.
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
MODELS = [('dot_sandwich_20261123_a03', 0, 'sw_fixed'), ('dot_sandwich_20261123_a03', 0, 'sw_route'),
          ('dot_sandwich_20261123_a03', 1, 'sw_route'), ('dot_sandwich_20261124_a04', 0, 'sw_prior'),
          ('dot_sandwich_20261124_a04', 0, 'sw6_route'), ('dot_sandwich_20261124_a04', 1, 'sw6_route'),
          ('dot_sandwich_20261124_a05', 0, 'sw_mpg'), ('dot_sandwich_20261124_a05', 0, 'sw_branch')]
ROWS, SAMPLED_PATHS = 128, 64
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
    table = torch.stack([losses(p) for p in paths])  # (paths, rows, L - prefix)
    fixed_idx = paths.index(tuple(s % 3 for s in range(depth)))
    seq_means = table.mean(-1)  # (paths, rows)
    best_static = int(seq_means.mean(1).argmin())
    token_oracle = table.min(0).values
    seq_oracle = seq_means.min(0).values
    summary = dict(run=run, seed=seed, arm=arm, rows=len(ids), paths=len(paths), middles=k, depth=depth,
                   own=float(own.mean()), fixed=float(table[fixed_idx].mean()),
                   best_static=float(seq_means[best_static].mean()), best_static_path=list(paths[best_static]),
                   sequence_oracle=float(seq_oracle.mean()), token_oracle=float(token_oracle.mean()),
                   token_oracle_best_path_share=float((table.argmin(0) == best_static).double().mean()),
                   path_means={'-'.join(map(str, p)): float(seq_means[i].mean()) for i, p in enumerate(paths)})
    np.savez_compressed(out/'rows.npz', own=own.mean(-1).numpy(), fixed=table[fixed_idx].mean(-1).numpy(),
                        best_static=seq_means[best_static].numpy(), sequence_oracle=seq_oracle.numpy(),
                        token_oracle=token_oracle.mean(-1).numpy())
    summary['rows_sha256'] = sha(out/'rows.npz')
    write_json(out/'summary.json', summary)


def experiment(base):
    stage(base, 'eval', [(0, f'm{i}') for i in range(len(MODELS))], 'scripts.dot_headroom_study')
    names = ['prepared.json', 'execution/source.json']
    for run, seed, arm in MODELS:
        d = f'{run}-s{seed}-{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if sha(base/d/'rows.npz') != s['rows_sha256']:
            raise ValueError('rows changed: '+d)
        names += [f'{d}/summary.json', f'{d}/rows.npz']
    archive = base/'export'/'headroom.tar.gz'
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
