"""Token-level DOT screen with downstream in the same handoff (docs/DOT_TOKEN_20261123.md).

Stage 1 trains eight DOT arms (one GPU each, DCLM long-study rows, 1x budget) with the loop-study
qualify/fit code; stage 2 scores every terminal zero-shot on the pinned four-task requests.
Baselines (standard transformer: dense4, dense8, seeds 0/1) are verified earlier runs on the same rows.
"""
import argparse
import json
from pathlib import Path
import tarfile
from unittest import mock

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.anchor_long_queue import stage, supervise
import scripts.loop_screen_downstream_study as downstream
import scripts.loop_screen_study as loop

JOBS = [(0, 'dot_route_t8'), (1, 'dot_route_t8'), (0, 'dot_fixed_t8'), (1, 'dot_fixed_t8'),
        (0, 'dot_random_t8'), (0, 'dot_route_res_t8'), (0, 'dot_route_t4'), (1, 'dot_route_t4')]


def prepare(base):
    loop.make_prepare(JOBS)(base)
    prepared, raw = downstream.pinned(*downstream.REQUESTS), downstream.pinned(*downstream.RAW)
    if raw['a2_metadata']['prepared_sha256'] != downstream.digest(prepared):
        raise ValueError('Raw template does not match requests')


def worker(base, seed, arm, action):
    if action == 'train':
        loop.worker(base, seed, arm, 'qualify')
        loop.worker(base, seed, arm, 'fit')
    else:
        (base/'downstream').mkdir(exist_ok=True)
        with mock.patch.object(downstream, 'SCREEN', base):
            downstream.worker(base/'downstream', seed, arm)


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.dot_token_study')
    report, names = {}, ['prepared.json', 'execution/source.json']
    for seed, arm in JOBS:
        d = f's{seed}/{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if s['step'] != loop.STEPS[arm] or sha(base/d/'scores.npz') != s['score_sha256']:
            raise ValueError('Invalid terminal: '+d)
        report[d] = s
        names += [f'{d}/summary.json', f'{d}/scores.npz', f'{d}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    stage(base, 'score', JOBS, 'scripts.dot_token_study')
    for seed, arm in JOBS:
        d = f'downstream/s{seed}-{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if sha(base/d/'items.npz') != s['items_sha256']:
            raise ValueError('Items changed: '+d)
        names += [f'{d}/summary.json', f'{d}/items.npz']
    archive = base/'export'/'dot_token_results.tar.gz'
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
    p.add_argument('action', choices=('queue', 'train', 'score'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1))
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
        worker(a.base, a.seed, a.arm, a.action)
