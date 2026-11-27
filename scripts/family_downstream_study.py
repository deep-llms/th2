"""Downstream zero-shot scoring of the sandwich, translator (two seeds) and domain (1x and 8x) models.

Same pinned four-task requests, harness template, FP32 scorer and runtime scopes as the loop-screen
downstream study (scripts/loop_screen_downstream_study.py). No training. Every checkpoint is
hash-checked against its own model.json; every study must be terminal. Units are spread over the
eight GPUs round-robin; each GPU scores its units in sequence. Design: docs/FAMILY_DOWNSTREAM_20261123.md.
"""
import argparse
import json
from pathlib import Path
import tarfile
import time

import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.domain_screen import ROUTE_A, ROUTE_B, ROUTE_BOTH, DomainScreen
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen_downstream import digest, items, save_items, score_requests
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.sandwich import ROUTES as SW, Sandwich
from efficient_llm.warm_order.translator_screen import ARMS as TR_ARMS, ROUTES as TR, TranslatorScreen
from scripts.anchor_long_queue import stage, supervise
from scripts.loop_screen_downstream_study import RAW, REQUESTS, pinned

OUT = Path('/mnt/local/_outputs/deep-llms_th2')
FAMILIES = {  # family -> (study root holding prepared.json and model dirs, root holding terminal.json)
    'sandwich': (OUT/'sandwich_20261123_a01', OUT/'sandwich_20261123_a01'),
    'translator_s0': (OUT/'translator_screen_20261123_a01', OUT/'translator_screen_20261123_a01'),
    'translator_s1': (OUT/'translator_screen_s1_20261123_a01', OUT/'translator_screen_s1_20261123_a01'),
    'domain_1x': (OUT/'domain_screen_20261123_a01', OUT/'domain_screen_20261123_a01'),
    'domain_8x': (OUT/'scale_round_20261123_a01'/'domain', OUT/'scale_round_20261123_a01'),
}
SANDWICH = {
    'sw_warm': {'native': SW['sandwich_warm']},
    'sw_stack': {'native': SW['sandwich_stack'], 'swap': SW['sandwich_swap']},
    'sw_stack_mix': {'native': SW['sandwich_stack'], 'swap': SW['sandwich_swap']},
    'not_stack': {'native': SW['not_stack']},
    'joint': {'native': SW['sandwich_stack'], 'swap': SW['sandwich_swap']},
    'dense5': {'native': SW['dense5']},
}
DOMAIN = {'single': {'A': ROUTE_A}, 'oracle': {'A': ROUTE_A, 'B': ROUTE_B},
          'random': {'A': ROUTE_A, 'B': ROUTE_B}, 'both': {'both': ROUTE_BOTH}}
UNITS = ([('sandwich', a, r) for a, r in SANDWICH.items()]
         + [(f, a, {'native': TR[TR_ARMS[a]['translator']][0]}) for f in ('translator_s0', 'translator_s1') for a in TR_ARMS]
         + [('domain_1x', a, r) for a, r in DOMAIN.items()]
         + [('domain_8x', a, DOMAIN[a]) for a in ('single', 'oracle')])
GPUS = 8
DEVICE = 'cuda'


def assignment(k):
    return [u for i, u in enumerate(UNITS) if i % GPUS == k]


def build(family, arm, config):
    if family == 'sandwich':
        return Sandwich(config, with_extra=(arm == 'dense5'))
    if family.startswith('translator'):
        return TranslatorScreen(config, arm)
    return DomainScreen(config, arm)


def checkpoint(family, arm):
    root = FAMILIES[family][0]
    pin = json.loads((root/arm/'model.json').read_text())
    if Path(pin['path']) != root/arm/'model.pt' or sha(pin['path']) != pin['sha256']:
        raise ValueError(f'Checkpoint changed: {family}/{arm}')
    return pin


def prepare(base):
    prepared, raw = pinned(*REQUESTS), pinned(*RAW)
    if raw['a2_metadata']['prepared_sha256'] != digest(prepared):
        raise ValueError('Raw template does not match requests')
    for family, (root, terminal) in FAMILIES.items():
        if not json.loads((terminal/'terminal.json').read_text())['success']:
            raise ValueError('Study not terminal: '+family)
    pins = {f'{f}/{a}': checkpoint(f, a)['sha256'] for f, a, _ in UNITS}
    write_json(base/'prepared.json', dict(requests=len(prepared['requests']), units=[[f, a, sorted(r)] for f, a, r in UNITS],
                                          checkpoints=pins, gpus=GPUS))


def score_unit(base, family, arm, routes, prepared, raw):
    out = base/f'{family}-{arm}'
    out.mkdir(parents=True, exist_ok=False)
    pin = checkpoint(family, arm)
    config = Config(**json.loads((FAMILIES[family][0]/'prepared.json').read_text())['config'])
    model = build(family, arm, config)
    model.load_state_dict(torch.load(pin['path'], map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tasks, hashes, meta, started = {}, {}, {}, time.monotonic()
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        for name, route in routes.items():
            scores, meta[name] = score_requests(model, route, prepared['requests'], DEVICE)
            tasks[name] = save_items(items(raw, prepared, scores), out/f'items_{name}.npz')
            hashes[name] = sha(out/f'items_{name}.npz')
    write_json(out/'summary.json', dict(family=family, arm=arm, routes={k: list(v) for k, v in routes.items()},
                                        tasks=tasks, items_sha256=hashes, meta=meta, checkpoint_sha256=pin['sha256'],
                                        seconds=time.monotonic() - started, precision='fp32_no_autocast_no_tf32'))


def worker(base, slot):
    prepared, raw = pinned(*REQUESTS), pinned(*RAW)
    torch.set_num_threads(4)
    for family, arm, routes in assignment(slot):
        score_unit(base, family, arm, routes, prepared, raw)


def experiment(base):
    stage(base, 'score', [(0, f'g{k}') for k in range(GPUS)], 'scripts.family_downstream_study')
    names, report = ['prepared.json', 'execution/source.json'], {}
    for family, arm, routes in UNITS:
        d = f'{family}-{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        for r in routes:
            if sha(base/d/f'items_{r}.npz') != s['items_sha256'][r]:
                raise ValueError('Items changed: '+d)
            names.append(f'{d}/items_{r}.npz')
        report[d] = s['tasks']
        names.append(f'{d}/summary.json')
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'family_downstream.tar.gz'
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
    p.add_argument('--seed', type=int, choices=(0,))
    p.add_argument('--arm', choices=[f'g{k}' for k in range(GPUS)])
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
