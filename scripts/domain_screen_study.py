"""Domain screen (seed 0, four arms, one GPU each). Design: docs/DOMAIN_SCREEN_20261123.md.

DCLM side reuses the verified long-study seed-0 rows and registered measurement rows.
FineWeb-Edu side reads the OHMV-rebuilt packed FWE corpus (GPT-NeoX, same tokenizer)
reshaped to 512-token rows; only single-document rows are used. Selected row ids and
the SHA256 of their token content are recorded (the 120 GB stream is not re-hashed).
A sequence-level router (logistic regression on the mean embedding of the first 32
tokens) is fit on held-out router rows of both domains and evaluated on measurement rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.domain_screen import ARMS, STEPS as BASE_STEPS, DomainScreen, domain_of, eval_routes, train_route
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import lr_at
from scripts.anchor_long_queue import stage, supervise
from scripts.anchor_long_study import score
from scripts.loop_screen_study import BATCH, check_parity, inputs, rows_for

SEED = 0
SCALE = 8  # scale round; the first domain screen (domain_screen_20261123_a01) ran SCALE = 1
STEPS = BASE_STEPS * SCALE
FWE_POOL = 15_625_000  # first 8B tokens of the FWE heal stream (512-token rows)
FWE = Path('/mnt/local/_data/deep-llms_th2/a2_dot_20261118/corpora/fineweb_edu/packed/fineweb_edu')
CALIBRATION_EVERY = 512
DEVICE = 'cuda'
ROUTER_ROWS = 1024
MEASURE_ROWS = 512


def eligible_sample(rows, pool, count, seed, chunk=262_144):
    """Random distinct single-document rows from the first `pool` rows (sorted). Candidates are
    drawn in bounded chunks so boundary checks never read more than `chunk` rows at once."""
    rng = np.random.default_rng(seed)
    pool = min(pool, len(rows.ids))
    picked = np.empty(0, dtype=np.int64)
    for _ in range(4 * (count // chunk + 1) + 20):
        cand = np.setdiff1d(np.unique(rng.integers(0, pool, size=chunk)), picked, assume_unique=False)
        ok = cand[~np.asarray(rows.starts[cand, 1:]).any(1)]
        picked = np.union1d(picked, ok)
        if len(picked) >= count:
            break
    if len(picked) < count:
        raise ValueError('Not enough single-document rows')
    return np.sort(rng.permutation(picked)[:count])


def content_sha(rows, ids):
    h = hashlib.sha256()
    for i in range(0, len(ids), 4096):
        h.update(np.ascontiguousarray(rows.ids[ids[i:i+4096]]).tobytes())
    return h.hexdigest()


def prepare(base):
    meta, config, dclm, ids, cal, measurement = inputs(SEED)
    need = STEPS // 2 * BATCH
    ids = rows_for(SEED, ids, need + ROUTER_ROWS)
    out = base/'domain_rows'
    out.mkdir(parents=True, exist_ok=False)
    heal, final, tuning = Rows(FWE/'heal'), Rows(FWE/'final'), Rows(FWE/'tuning')
    sel = dict(fwe_train=(heal, eligible_sample(heal, FWE_POOL, need, 20261123)),
               fwe_measure=(final, eligible_sample(final, len(final.ids), MEASURE_ROWS, 20261124)),
               fwe_router=(tuning, eligible_sample(tuning, len(tuning.ids), ROUTER_ROWS, 20261125)))
    record = {}
    for name, (rows, picked) in sel.items():
        np.save(out/f'{name}.npy', picked)
        record[name] = dict(count=int(len(picked)), ids_sha256=sha(out/f'{name}.npy'), content_sha256=content_sha(rows, picked))
    dclm_router = np.asarray(ids[need:need + ROUTER_ROWS])  # DCLM rows never used for training
    if np.intersect1d(dclm_router, np.concatenate([cal, measurement])).size or \
            np.intersect1d(np.asarray(ids[:need]), np.concatenate([cal, measurement])).size:
        raise ValueError('DCLM row overlap')
    np.save(out/'dclm_router.npy', dclm_router)
    record['dclm_router'] = dict(count=ROUTER_ROWS, ids_sha256=sha(out/'dclm_router.npy'))
    record['fwe_stems'] = {k: dict(bin_bytes=(FWE/f'{k}.bin').stat().st_size, rows512=len(r.ids))
                           for k, r in (('heal', heal), ('final', final), ('tuning', tuning))}
    write_json(base/'prepared.json', dict(config=config.__dict__, rows=record, steps=STEPS, arms=ARMS))


def load_rows(base):
    meta, config, dclm, ids, cal, measurement = inputs(SEED)
    d = base/'domain_rows'
    prep = json.loads((base/'prepared.json').read_text())['rows']
    for name in ('fwe_train', 'fwe_measure', 'fwe_router', 'dclm_router'):
        if sha(d/f'{name}.npy') != prep[name]['ids_sha256']:
            raise ValueError('Row selection changed: '+name)
    ids = rows_for(SEED, ids, STEPS // 2 * BATCH + ROUTER_ROWS)
    return dict(config=config, dclm=dclm, dclm_ids=ids, cal=cal, dclm_measure=measurement,
                dclm_router=np.load(d/'dclm_router.npy'), fwe_train_rows=Rows(FWE/'heal'),
                fwe_train=np.load(d/'fwe_train.npy'), fwe_measure_rows=Rows(FWE/'final'),
                fwe_measure=np.load(d/'fwe_measure.npy'), fwe_router_rows=Rows(FWE/'tuning'),
                fwe_router=np.load(d/'fwe_router.npy'))


def build(config, arm):
    torch.manual_seed(SEED)
    return DomainScreen(config, arm)


def nll_rows(model, rows, ids, route):
    return score(model, rows, ids, {'v': dict(path=route, skip=None)}, DEVICE)['v']


@torch.no_grad()
def features(model, rows, ids):
    model.eval()
    out = []
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        for i in range(0, len(ids), 64):
            out.append(model.prefix_feature(rows.batch(ids[i:i+64], DEVICE)).cpu())
    return torch.cat(out)


def fit_router(fx, fy, steps=500):
    """Logistic regression on standardized features; returns predictor of P(route B better)."""
    mu, sd = fx.mean(0), fx.std(0) + 1e-6
    w = torch.zeros(fx.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    z = (fx - mu) / sd
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, fy) + 1e-3 * w.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda x: torch.sigmoid(((x - mu) / sd) @ w.detach() + b.detach())


def worker(base, arm):
    out = base/arm
    out.mkdir(parents=True, exist_ok=False)
    D = load_rows(base)
    cpu, model = build(D['config'], arm), build(D['config'], arm).to(DEVICE)
    with torch.no_grad():
        cpu.attnres_query.copy_(torch.randn(cpu.attnres_query.shape, generator=torch.Generator().manual_seed(7)) * .02)
        model.load_state_dict(cpu.state_dict())
        for rows, idx in ((D['dclm'], D['cal'][:2]), (D['fwe_measure_rows'], D['fwe_measure'][:2])):
            x = rows.batch(idx, 'cpu')
            for route in eval_routes(arm).values():
                with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
                    ref = cpu.eval().token_losses(x, route)
                with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
                    got = model.eval().token_losses(x.to(DEVICE), route).cpu()
                check_parity(got, ref)
        model.attnres_query.zero_()
    del cpu
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0, STEPS), betas=(.9, .95), eps=1e-8, weight_decay=0)
    half = BATCH

    def calibration(step):
        rec = dict(step=step)
        for name, route in eval_routes(arm).items():
            rec['dclm_'+name] = float(nll_rows(model, D['dclm'], D['cal'], route).mean())
            rec['fwe_'+name] = float(nll_rows(model, D['fwe_router_rows'], D['fwe_router'][:256], route).mean())
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(rec), file=f)

    calibration(0)
    started = time.time()
    with training_runtime(DEVICE), anchor_runtime(DEVICE):
        for step in range(STEPS):
            for group in opt.param_groups:
                group['lr'] = lr_at(step, STEPS)
            k = step // 2
            if domain_of(step) == 'dclm':
                x = D['dclm'].batch(D['dclm_ids'][k*half:(k+1)*half], DEVICE)
            else:
                x = D['fwe_train_rows'].batch(D['fwe_train'][k*half:(k+1)*half], DEVICE)
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = model.nll(x, train_route(arm, step)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            done = step + 1
            if done % 128 == 0:
                elapsed = time.time() - started
                write_json(out/'progress.json', dict(step=done, total=STEPS, last_batch=float(loss.detach()),
                                                     elapsed=elapsed, tokens_per_s=done*BATCH*512/elapsed))
            if done % CALIBRATION_EVERY == 0:
                calibration(done)
    train_seconds = time.time() - started
    torch.save(model.state_dict(), out/'model.pt')
    write_json(out/'model.json', dict(path=str(out/'model.pt'), sha256=sha(out/'model.pt')))
    sets = dict(dclm_measure=(D['dclm'], D['dclm_measure']), fwe_measure=(D['fwe_measure_rows'], D['fwe_measure']),
                dclm_router=(D['dclm'], D['dclm_router']), fwe_router=(D['fwe_router_rows'], D['fwe_router']))
    values = {f'{s}_{name}': nll_rows(model, rows, idx, route)
              for s, (rows, idx) in sets.items() for name, route in eval_routes(arm).items()}
    summary = dict(arm=arm, steps=STEPS, train_seconds=train_seconds, nll={k: float(v.mean()) for k, v in values.items()},
                   block_applications_per_token=len(next(iter(eval_routes(arm).values()))))
    if arm in ('oracle', 'random'):
        feats = {s: features(model, rows, idx) for s, (rows, idx) in sets.items()}
        fx = torch.cat([feats['dclm_router'], feats['fwe_router']])
        fy = torch.cat([torch.as_tensor(values['dclm_router_B'] < values['dclm_router_A']),
                        torch.as_tensor(values['fwe_router_B'] < values['fwe_router_A'])]).float()
        router = fit_router(fx, fy)
        routing = {}
        for s, label_b in (('dclm_measure', False), ('fwe_measure', True)):
            a, b = values[s+'_A'], values[s+'_B']
            p = router(feats[s]).numpy()
            learned = np.where(p >= .5, b, a)
            values[s+'_learned'] = learned
            routing[s] = dict(route_A=float(a.mean()), route_B=float(b.mean()), domain_label=float((b if label_b else a).mean()),
                              oracle=float(np.minimum(a, b).mean()), learned=float(learned.mean()),
                              learned_frac_B=float((p >= .5).mean()), frac_rows_B_better=float((b < a).mean()))
        summary['routing'] = routing
        summary['router_train_frac_B'] = float(fy.mean())
    np.savez_compressed(out/'scores.npz', **{k: np.asarray(v) for k, v in values.items()})
    summary['score_sha256'] = sha(out/'scores.npz')
    write_json(out/'summary.json', summary)


def experiment(base):
    stage(base, 'train', [(SEED, a) for a in ARMS], 'scripts.domain_screen_study')
    report, names = {}, ['prepared.json', 'execution/source.json']
    for a in ARMS:
        s = json.loads((base/a/'summary.json').read_text())
        if sha(base/a/'scores.npz') != s['score_sha256']:
            raise ValueError('Scores changed: '+a)
        report[a] = s
        names += [f'{a}/summary.json', f'{a}/scores.npz', f'{a}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    import tarfile
    archive = base/'export'/'domain_results.tar.gz'
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
    p.add_argument('--seed', type=int, choices=(SEED,))
    p.add_argument('--arm', choices=ARMS)
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
