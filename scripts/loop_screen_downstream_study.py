"""Downstream zero-shot scoring of loop-screen terminals (jobs from the screen's prepared.json), one GPU each.

Inputs pinned by hash: the lm-eval requests used by the 18 Nov warm-order screen,
one harness raw.json as the doc/argument/gold template, and each terminal.pt.
Writes per-arm items.npz + summary.json and a compact export archive. No training.
"""
import argparse
import json
from pathlib import Path
import tarfile
import time

import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import ALL_ARMS, SPECS, LoopScreen, native_path
from efficient_llm.warm_order.loop_screen_downstream import digest, items, save_items, score_requests
from efficient_llm.warm_order.model import Config
from scripts.anchor_long_queue import stage, supervise

OUT = Path('/mnt/local/_outputs/deep-llms_th2')
# round 1: loop_screen_20261123_a01, round 2: loop_screen2_20261123_a02, round 4: loop_screen4_20261123_a02,
# scale round: scale_round_20261123_a01/loop
SCREEN = OUT/'moe_loop_20261123_a01'  # token-level MoE in the loop line (docs/MOE_LOOP_ROUND_20261123.md)
ROOT = SCREEN.parent if SCREEN.name == 'loop' else SCREEN  # holds report.json and terminal.json
REQUESTS = (OUT/'warm_downstream_20261118_a02/requests.json',
            '6282100c39545c6d14b219a4a6538e4ecf13b94a12b97c19c68c18c61a450091')
RAW = (OUT/'warm_downstream_replay_20261118_a01/runs/pilot-D-s0/raw.json',
       'a5c658b566af0be4c26c270bf8a948da3ec027ac21998088a700784b23e6c9ba')
DEVICE = 'cuda'


def pinned(path, digest_):
    if sha(path) != digest_:
        raise ValueError('Pinned input changed: '+str(path))
    return json.loads(Path(path).read_text())


def prepare(base):
    prepared, raw = pinned(*REQUESTS), pinned(*RAW)
    if raw['a2_metadata']['prepared_sha256'] != digest(prepared):
        raise ValueError('Raw template does not match requests')
    report = json.loads((ROOT/'report.json').read_text())
    keys = {a for _, a in screen_jobs()} if ROOT != SCREEN else {f's{s}/{a}' for s, a in screen_jobs()}
    if not keys <= set(report) or not json.loads((ROOT/'terminal.json').read_text())['success']:
        raise ValueError('Screen not terminal')
    write_json(base/'prepared.json', dict(requests=len(prepared['requests']), screen=str(SCREEN), jobs=screen_jobs()))


def screen_jobs():
    return [tuple(j) for j in json.loads((SCREEN/'prepared.json').read_text())['jobs']]


def worker(base, seed, arm):
    out = base/f's{seed}-{arm}'
    out.mkdir(parents=True, exist_ok=False)
    prepared, raw = pinned(*REQUESTS), pinned(*RAW)
    pin = json.loads((SCREEN/f's{seed}'/arm/'terminal.json').read_text())
    state = torch.load(pinned_path(pin), map_location='cpu', weights_only=False)
    if state['arm'] != arm or state['step'] != json.loads((SCREEN/f's{seed}'/arm/'summary.json').read_text())['step']:
        raise ValueError('Checkpoint identity mismatch')
    config = Config(**json.loads((SCREEN/'prepared.json').read_text())['config'])
    from efficient_llm.warm_order.loop_screen import make_model
    model = make_model(config, arm)
    model.load_state_dict(state['model'])
    model = model.to(DEVICE).eval()
    from efficient_llm.warm_order.loop_screen import SPECS
    passes = SPECS[arm].get('passes', 3)  # f18b bug: a 4-pass loop arm was scored at 3 passes (LAMBADA 5%)
    model.passes = passes
    for p in model.parameters():
        p.requires_grad_(False)
    path = native_path(arm)
    torch.set_num_threads(4)
    started = time.monotonic()
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        scores, meta = score_requests(model, path, prepared['requests'], DEVICE)
    seconds = time.monotonic() - started
    summary = save_items(items(raw, prepared, scores), out/'items.npz')
    write_json(out/'summary.json', dict(arm=arm, seed=seed, tasks=summary, seconds=seconds, checkpoint_sha256=pin['sha256'],
                                        items_sha256=sha(out/'items.npz'), precision='fp32_no_autocast_no_tf32',
                                        passes=passes, **meta))


def pinned_path(pin):
    if sha(pin['path']) != pin['sha256']:
        raise ValueError('Checkpoint changed')
    return pin['path']


def experiment(base):
    jobs = screen_jobs()
    stage(base, 'score', jobs, 'scripts.loop_screen_downstream_study')
    names = ['prepared.json', 'execution/source.json']
    report = {}
    for seed, arm in jobs:
        name = f's{seed}-{arm}'
        s = json.loads((base/name/'summary.json').read_text())
        if sha(base/name/'items.npz') != s['items_sha256']:
            raise ValueError('Items changed: '+name)
        report[name] = s['tasks']
        names += [f'{name}/summary.json', f'{name}/items.npz']
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'downstream_results.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    if archive.stat().st_size > 20_000_000:
        raise ValueError('Archive above export budget')
    write_json(base/'export'/'manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                    members={n: sha(base/n) for n in names}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'score'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1))
    p.add_argument('--arm', choices=ALL_ARMS)
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
