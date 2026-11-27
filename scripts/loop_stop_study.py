"""Round 3: learned per-token STOP over loop passes on frozen loop-screen bodies.

Design and decision rules: docs/LOOP_STOP_20261123.md. For each body (one GPU):
1. Head rows (unused by the 12,288-step body fits): per pass k=1..K, collect each
   token's causal state after pass k and its NLL when the model stops at k.
2. Fit one HaltHead per lambda with the PonderNet-style expected loss.
3. On the 512 registered measurement rows, evaluate in deploy mode (copy-through
   halting, threshold 0.5) against fixed passes 1..K, a random policy matched to
   the learned per-token pass histogram, and the label-oracle (reference only).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import SPECS, STEPS, LoopScreen, native_path
from efficient_llm.warm_order.loop_stop import HaltHead, halted_hidden, ponder_loss
from scripts.anchor_long_queue import stage, supervise
from scripts.loop_screen_study import BATCH, inputs

OUT = Path('/mnt/local/_outputs/deep-llms_th2')
BODY_SETS = {  # job name -> (seed, arm, screen base)
    'round3': {
        'gated_var': (0, 'gated_var', OUT/'loop_screen_20261123_a01'),
        'memory_var': (0, 'memory_var', OUT/'loop_screen2_20261123_a02'),
        'loop_attnres_var': (0, 'loop_attnres_var', OUT/'loop_screen2_20261123_a02'),
        'gated': (0, 'gated', OUT/'loop_screen_20261123_a01'),
        'memory': (0, 'memory', OUT/'loop_screen_20261123_a01'),
        'loop_attnres': (0, 'loop_attnres', OUT/'loop_screen2_20261123_a02'),
    },
    'round5': {  # bodies trained with per-token random depth (docs/LOOP_SCREEN4_20261123.md)
        'loop_attnres_tokvar': (0, 'loop_attnres_tokvar', OUT/'loop_screen4_20261123_a02'),
        'gated_tokvar': (0, 'gated_tokvar', OUT/'loop_screen4_20261123_a02'),
    },
}
BODIES = BODY_SETS['round5']  # round 3 ran with BODY_SETS['round3'] (frozen in loop_stop_20261123_a01)
KMAX = 6
LAMBDAS = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
HEAD_ROWS = 512
DEVICE = 'cuda'


def load_body(name, config):
    seed, arm, base = BODIES[name]
    pin = json.loads((base/f's{seed}'/arm/'terminal.json').read_text())
    if sha(pin['path']) != pin['sha256']:
        raise ValueError('Body checkpoint changed: '+name)
    state = torch.load(pin['path'], map_location='cpu', weights_only=False)
    if state['arm'] != arm or state['step'] != STEPS[arm]:
        raise ValueError('Body identity mismatch: '+name)
    model = LoopScreen(config, arm)
    model.load_state_dict(state['model'])
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, pin['sha256']


def never(k, s):
    return torch.zeros(s.shape[:2], dtype=torch.bool, device=s.device)


def suffix_nll(model, h, x):
    """Per-token NLL (B, length-prefix) from final hidden h, mirroring EmbeddingAnchor.token_losses."""
    p = model.c.prefix
    z = model.readout_hidden(h)[:, p - 1:-1]
    logits = torch.nn.functional.linear(z, model.embed.weight).float()
    return torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, p:], reduction='none')


@torch.no_grad()
def collect(model, rows, ids, path, batch=8):
    """States after each pass (K, N, d) and stop-at-k NLLs (K, N) over scored positions."""
    p = model.c.prefix
    states, nlls = [], []
    for start in range(0, len(ids), batch):
        x = rows.batch(ids[start:start + batch], DEVICE)
        seen = []
        halted_hidden(model, x, path, KMAX, lambda k, s: (seen.append(s[:, p - 1:-1].half()), never(k, s))[1])
        per_k = []
        for k in range(1, KMAX + 1):
            h, _ = halted_hidden(model, x, path, k, never)
            per_k.append(suffix_nll(model, h, x))
        states.append(torch.stack(seen, 0).flatten(1, 2))  # decisions exist after passes 1..KMAX-1
        nlls.append(torch.stack(per_k, 0).flatten(1, 2))
    return torch.cat(states, 1), torch.cat(nlls, 1)


def fit_head(states, nlls, lam, d, steps=400):
    head = HaltHead(d, KMAX).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    n = nlls.shape[1]
    g = torch.Generator(device='cpu').manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(n, 65536),), generator=g).to(DEVICE)
        q = torch.stack([head(states[k - 1, idx], k) for k in range(1, KMAX)], 0)
        q = torch.cat([q, torch.ones_like(q[:1])], 0)
        loss, _ = ponder_loss(q, nlls[:, idx], lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head.eval()


@torch.no_grad()
def deploy(model, rows, ids, path, decide_factory, batch=4):
    """Mean NLL and mean passes in deploy mode; decide_factory(x_batch_index) -> decide fn."""
    total_nll, total_passes, count = 0., 0., 0
    hist = torch.zeros(KMAX, dtype=torch.long)
    per_row = []
    p = model.c.prefix
    for start in range(0, len(ids), batch):
        x = rows.batch(ids[start:start + batch], DEVICE)
        h, used = halted_hidden(model, x, path, KMAX, decide_factory(start, x))
        nll = suffix_nll(model, h, x).double()
        u = used[:, p - 1:-1]
        total_nll += float(nll.sum())
        total_passes += float(u.double().sum())
        count += nll.numel()
        hist += torch.bincount(u.flatten().cpu() - 1, minlength=KMAX)
        per_row.extend(nll.mean(1).cpu().tolist())
    return dict(nll=total_nll / count, passes=total_passes / count,
                block_apps=2 + 2 * total_passes / count, histogram=hist.tolist()), np.array(per_row)


def worker(base, name):
    seed, arm, _ = BODIES[name]
    out = base/name
    out.mkdir(parents=True, exist_ok=False)
    meta, config, rows, ids, cal, measurement = inputs(seed)
    model, body_sha = load_body(name, config)
    path = native_path(arm)
    start_row = STEPS[arm] * BATCH  # rows never seen by this body's fit
    head_ids = np.asarray(ids[start_row:start_row + HEAD_ROWS])
    if np.intersect1d(head_ids, np.concatenate([cal, measurement])).size:
        raise ValueError('Head rows overlap evaluation rows')
    started = time.monotonic()
    result = dict(body=name, arm=arm, seed=seed, body_sha256=body_sha, kmax=KMAX, head_rows=HEAD_ROWS)
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        states, nlls = collect(model, rows, head_ids, path)
        result['head_rows_fixed_nll'] = nlls.double().mean(1).tolist()
        fixed = {}
        arrays = {}
        for k in range(1, KMAX + 1):
            stats, rows_nll = deploy(model, rows, measurement, path,
                                     lambda s, x, k=k: (lambda j, st: torch.full(st.shape[:2], j >= k, dtype=torch.bool,
                                                                                  device=st.device)))
            fixed[k] = stats
            arrays[f'fixed_{k}'] = rows_nll
        result['fixed'] = fixed
        learned, random_matched = {}, {}
        for lam in LAMBDAS:
            with torch.enable_grad():
                head = fit_head(states.float(), nlls.float(), lam, config.d)
            stats, rows_nll = deploy(model, rows, measurement, path,
                                     lambda s, x, head=head: (lambda k, st: head(st, k) >= .5))
            learned[str(lam)] = dict(stats, head_bias=head.b.detach().cpu().tolist(),
                                     head_w_norm=float(head.w.detach().norm()))
            arrays[f'learned_{lam}'] = rows_nll
            probs = torch.tensor(stats['histogram'], dtype=torch.float64)
            probs = probs / probs.sum()
            gen = torch.Generator(device='cpu').manual_seed(1234)

            def random_factory(s, x, probs=probs, gen=gen):
                target = torch.multinomial(probs, x.numel(), replacement=True, generator=gen).reshape(x.shape) + 1
                target = target.to(x.device)
                return lambda k, st: target <= k
            stats_r, rows_r = deploy(model, rows, measurement, path, random_factory)
            random_matched[str(lam)] = stats_r
            arrays[f'random_{lam}'] = rows_r
        result['learned'] = learned
        result['random_matched'] = random_matched
        oracle = {}
        meas_states, meas_nlls = collect(model, rows, measurement, path)
        ks = torch.arange(1, KMAX + 1, device=meas_nlls.device, dtype=meas_nlls.dtype).unsqueeze(-1)
        for lam in LAMBDAS:
            choice = (meas_nlls + lam * ks).argmin(0)
            oracle[str(lam)] = dict(nll=float(meas_nlls.gather(0, choice[None]).double().mean()),
                                    passes=float((choice + 1).double().mean()),
                                    note='uniform-context label oracle; not deployable')
        result['oracle_uniform_context'] = oracle
    result['seconds'] = time.monotonic() - started
    np.savez_compressed(out/'rows.npz', row_ids=measurement, **arrays)
    result['rows_sha256'] = sha(out/'rows.npz')
    write_json(out/'summary.json', result)


def prepare(base):
    for name, (seed, arm, screen) in BODIES.items():
        if not json.loads((screen/'terminal.json').read_text())['success']:
            raise ValueError('Body run not terminal: '+name)
        if not SPECS[arm]['loop']:
            raise ValueError('Body is not a loop arm: '+name)
    write_json(base/'prepared.json', dict(bodies={k: [v[0], v[1], str(v[2])] for k, v in BODIES.items()},
                                          kmax=KMAX, lambdas=LAMBDAS, head_rows=HEAD_ROWS))


def experiment(base):
    jobs = [(0, name) for name in BODIES]
    stage(base, 'stop', jobs, 'scripts.loop_stop_study')
    report = {}
    for _, name in jobs:
        s = json.loads((base/name/'summary.json').read_text())
        if sha(base/name/'rows.npz') != s['rows_sha256']:
            raise ValueError('Rows changed: '+name)
        report[name] = s
    write_json(base/'report.json', report)
    import tarfile
    names = ['prepared.json', 'report.json', 'execution/source.json'] + [f'{n}/rows.npz' for n in BODIES]
    archive = base/'export'/'stop_results.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'stop'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0,))
    p.add_argument('--arm', choices=tuple(BODIES))
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
