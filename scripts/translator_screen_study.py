"""Translator screen (2x2x2, seed 0, one GPU per arm). Design: docs/TRANSLATOR_SCREEN_20261123.md."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import lr_at
from efficient_llm.warm_order.translator_screen import ARMS, ROUTES, STEPS, TranslatorScreen, train_route
from scripts.anchor_long_queue import stage, supervise
from scripts.anchor_long_study import score
from scripts.loop_screen_study import BATCH, check_parity, inputs

SEED = 1  # seed-0 run used SEED = 0 (translator_screen_20261123_a01)
CALIBRATION_EVERY = 512
DEVICE = 'cuda'


def build(config, arm):
    torch.manual_seed(SEED)
    return TranslatorScreen(config, arm)


def evaluate(model, rows, ids, route):
    return score(model, rows, ids, {'v': dict(path=route, skip=None)}, DEVICE)['v']


def worker(base, arm):
    out = base/arm
    out.mkdir(parents=True, exist_ok=False)
    meta, config, rows, ids, cal, measurement = inputs(SEED)
    native, swapped = ROUTES[ARMS[arm]['translator']]
    cpu, model = build(config, arm), build(config, arm).to(DEVICE)
    with torch.no_grad():  # parity at init and with realistic perturbed AttnRes queries
        for perturb in (False, True):
            if perturb and hasattr(cpu, 'attnres_query'):
                cpu.attnres_query.copy_(torch.randn(cpu.attnres_query.shape, generator=torch.Generator().manual_seed(7)) * .02)
                model.load_state_dict(cpu.state_dict())
            x = rows.batch(cal[:2], 'cpu')
            for route in (native, swapped):
                with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
                    ref = cpu.eval().token_losses(x, route)
                with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
                    got = model.eval().token_losses(x.to(DEVICE), route).cpu()
                check_parity(got, ref)
    if hasattr(model, 'attnres_query'):
        with torch.no_grad():
            model.attnres_query.zero_()
    del cpu
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0, STEPS), betas=(.9, .95), eps=1e-8, weight_decay=0)

    def calibration(step):
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(dict(step=step, native=float(evaluate(model, rows, cal, native).mean()),
                                  swapped=float(evaluate(model, rows, cal, swapped).mean()))), file=f)

    calibration(0)
    started = time.time()
    with training_runtime(DEVICE), anchor_runtime(DEVICE):
        for step in range(STEPS):
            for group in opt.param_groups:
                group['lr'] = lr_at(step, STEPS)
            model.train()
            x = rows.batch(ids[step*BATCH:(step+1)*BATCH], DEVICE)
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
    values = {'native': evaluate(model, rows, measurement, native), 'swapped': evaluate(model, rows, measurement, swapped)}
    np.savez_compressed(out/'scores.npz', row_ids=measurement, **values)
    write_json(out/'summary.json', dict(arm=arm, spec=ARMS[arm], steps=STEPS, train_seconds=train_seconds,
                                        nll={k: float(v.mean()) for k, v in values.items()},
                                        score_sha256=sha(out/'scores.npz'), block_applications_per_token=len(native),
                                        parameters=sum(p.numel() for p in model.parameters())))


def prepare(base):
    meta, config, rows, ids, cal, measurement = inputs(SEED)
    need = STEPS * BATCH
    if len(ids) < need or np.intersect1d(np.asarray(ids[:need]), np.concatenate([cal, measurement])).size:
        raise ValueError('Training rows insufficient or overlapping evaluation rows')
    write_json(base/'prepared.json', dict(config=config.__dict__, arms=ARMS, steps=STEPS))


def experiment(base):
    jobs = [(SEED, arm) for arm in ARMS]
    stage(base, 'train', jobs, 'scripts.translator_screen_study')
    report, names = {}, ['prepared.json', 'execution/source.json']
    for _, arm in jobs:
        s = json.loads((base/arm/'summary.json').read_text())
        if sha(base/arm/'scores.npz') != s['score_sha256']:
            raise ValueError('Scores changed: '+arm)
        report[arm] = s
        names += [f'{arm}/summary.json', f'{arm}/scores.npz', f'{arm}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    import tarfile
    archive = base/'export'/'translator_results.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))
    backup = base/'backup'
    backup.mkdir()
    for _, arm in jobs:
        shutil.copyfile(base/arm/'model.pt', backup/f'{arm}.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'train'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(SEED,))
    p.add_argument('--arm', choices=tuple(ARMS))
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
