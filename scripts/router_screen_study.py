"""Router screen: routers trained jointly with the body, one GPU per arm (docs/ROUTER_SCREEN_20261123.md).

Data are the exact 1x domain-screen rows (copied from domain_screen_20261123_a01 after hash checks), so the
verified 1x single/oracle/random/both models are same-data references; `single` is retrained here as an
exact-reproduction check. 12,288 updates, WSD, AdamW, alternating DCLM/FWE batches, seed 0.
"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.domain_screen import STEPS as BASE_STEPS, domain_of
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import lr_at
from efficient_llm.warm_order.router_screen import ARMS, ROUTED, RouterScreen
from scripts.anchor_long_queue import stage, supervise
import scripts.domain_screen_study as domain
from scripts.loop_screen_study import BATCH, check_parity

STEPS = BASE_STEPS  # 1x budget; the domain module default is the 8x scale round (see use_1x)
OUT = Path('/mnt/local/_outputs/deep-llms_th2')
SOURCE = OUT/'domain_screen_20261123_a01'  # verified 1x domain screen (rows and reference models)
JOBS = [(0, 'single')] + [(0, a) for a in ARMS]
DEVICE = 'cuda'
CALIBRATION_EVERY = 512


def use_1x():
    domain.STEPS = STEPS  # set in the router process only (domain.worker/load_rows read it)


def prepare(base):
    use_1x()
    prep = json.loads((SOURCE/'prepared.json').read_text())
    if prep['steps'] != STEPS:
        raise ValueError('Source domain screen is not the 1x run')
    out = base/'domain_rows'
    out.mkdir(parents=True)
    for name in ('fwe_train', 'fwe_measure', 'fwe_router', 'dclm_router'):
        src = SOURCE/'domain_rows'/f'{name}.npy'
        if sha(src) != prep['rows'][name]['ids_sha256']:
            raise ValueError('Source rows changed: '+name)
        shutil.copyfile(src, out/f'{name}.npy')
        if sha(out/f'{name}.npy') != prep['rows'][name]['ids_sha256']:
            raise ValueError('Row copy mismatch: '+name)
    write_json(base/'prepared.json', dict(prep, source=str(SOURCE), source_prepared_sha256=sha(SOURCE/'prepared.json'),
                                          jobs=JOBS, router_arms=ARMS))


def build(config, arm):
    torch.manual_seed(domain.SEED)
    return RouterScreen(config, arm)


def nll_rows(model, rows, ids, path):
    return domain.nll_rows(model, rows, ids, path)


@torch.no_grad()
def choices(model, rows, ids):
    """Argmax routing decision per sequence (sequence arms) or per-token expert fractions (token arms)."""
    model.eval()
    out = []
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        for i in range(0, len(ids), 4):
            x = rows.batch(ids[i:i+4], DEVICE)
            model.token_losses(x, ROUTED)
            c = model.last_choice
            if model.spec['kind'] in ('token', 'moe'):
                c = torch.stack([(c[:, model.c.prefix - 1:-1] == k).float().mean(1) for k in range(model.spec['k'])], 1)
            out.append(c.cpu())
    return torch.cat(out).numpy()


def qualify(model, cpu, D):
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        cpu.attnres_query.copy_(torch.randn(cpu.attnres_query.shape, generator=g) * .02)
        # sharp router for top-k MoE parity: avoids CPU/GPU near-tie flips among 4-8 experts
        cpu.router.copy_(torch.randn(cpu.router.shape, generator=g) * (.3 if cpu.spec['kind'] == 'moe' else .02))
        model.load_state_dict(cpu.state_dict())
        for rows, idx in ((D['dclm'], D['cal'][:2]), (D['fwe_measure_rows'], D['fwe_measure'][:2])):
            x = rows.batch(idx, 'cpu')
            for path in [ROUTED, *model.forced_paths().values()]:
                with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
                    ref = cpu.eval().token_losses(x, path)
                with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
                    got = model.eval().token_losses(x.to(DEVICE), path).cpu()
                check_parity(got, ref)
        model.attnres_query.zero_()
        model.router.zero_()
        model.usage.fill_(1. / model.spec['k'])


def worker(base, arm):
    use_1x()
    if arm == 'single':
        return domain.worker(base, 'single')
    out = base/arm
    out.mkdir(parents=True, exist_ok=False)
    D = domain.load_rows(base)
    cpu, model = build(D['config'], arm), build(D['config'], arm).to(DEVICE)
    qualify(model, cpu, D)
    del cpu
    model.generator = torch.Generator(device=DEVICE).manual_seed(20261123)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0, STEPS), betas=(.9, .95), eps=1e-8, weight_decay=0)
    paths = {'routed': ROUTED, **model.forced_paths()}

    def calibration(step):
        rec = dict(step=step, usage=model.usage.tolist())
        for name, path in paths.items():
            rec['dclm_'+name] = float(nll_rows(model, D['dclm'], D['cal'], path).mean())
            rec['fwe_'+name] = float(nll_rows(model, D['fwe_router_rows'], D['fwe_router'][:256], path).mean())
        rec['dclm_choice'] = choices(model, D['dclm'], D['cal'][:64]).mean(0).tolist() if model.spec['kind'] in ('token', 'moe') else \
            np.bincount(choices(model, D['dclm'], D['cal'][:64]), minlength=model.spec['k']).tolist()
        rec['fwe_choice'] = choices(model, D['fwe_router_rows'], D['fwe_router'][:64]).mean(0).tolist() if model.spec['kind'] in ('token', 'moe') else \
            np.bincount(choices(model, D['fwe_router_rows'], D['fwe_router'][:64]), minlength=model.spec['k']).tolist()
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(rec), file=f)

    calibration(0)
    started, half = time.time(), BATCH
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
                lm, aux, _ = model.train_loss(x)
                loss = lm if aux is None else lm + aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            done = step + 1
            if done % 128 == 0:
                elapsed = time.time() - started
                write_json(out/'progress.json', dict(step=done, total=STEPS, last_batch=float(lm.detach()),
                                                     aux=None if aux is None else float(aux.detach()),
                                                     usage=model.usage.tolist(), elapsed=elapsed,
                                                     tokens_per_s=done*BATCH*512/elapsed))
            if done % CALIBRATION_EVERY == 0:
                calibration(done)
    train_seconds = time.time() - started
    torch.save(model.state_dict(), out/'model.pt')
    write_json(out/'model.json', dict(path=str(out/'model.pt'), sha256=sha(out/'model.pt')))
    sets = dict(dclm_measure=(D['dclm'], D['dclm_measure']), fwe_measure=(D['fwe_measure_rows'], D['fwe_measure']),
                dclm_router=(D['dclm'], D['dclm_router']), fwe_router=(D['fwe_router_rows'], D['fwe_router']))
    values = {f'{s}_{name}': nll_rows(model, rows, idx, path) for s, (rows, idx) in sets.items() for name, path in paths.items()}
    values.update({f'{s}_choice': choices(model, rows, idx) for s, (rows, idx) in sets.items()})
    np.savez_compressed(out/'scores.npz', **values)
    summary = dict(arm=arm, spec=model.spec, steps=STEPS, train_seconds=train_seconds,
                   nll={k: float(v.mean()) for k, v in values.items() if not k.endswith('_choice')},
                   choice={k: (v.mean(0).tolist() if v.ndim == 2 else np.bincount(v, minlength=model.spec['k']).tolist())
                           for k, v in values.items() if k.endswith('_choice')},
                   usage=model.usage.tolist(), score_sha256=sha(out/'scores.npz'))
    write_json(out/'summary.json', summary)


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.router_screen_study')
    report, names = {}, ['prepared.json', 'execution/source.json']
    for _, arm in JOBS:
        s = json.loads((base/arm/'summary.json').read_text())
        if sha(base/arm/'scores.npz') != s['score_sha256']:
            raise ValueError('Invalid terminal: '+arm)
        report[arm] = s
        names += [f'{arm}/summary.json', f'{arm}/scores.npz', f'{arm}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'router_results.tar.gz'
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
    p.add_argument('--seed', type=int, choices=(0,))
    p.add_argument('--arm', choices=[a for _, a in JOBS])
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
