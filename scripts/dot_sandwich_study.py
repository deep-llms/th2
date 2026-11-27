"""DOT sandwich (tied T, token-level router before T, recipient ids), no warm-up, with downstream (docs/DOT_SANDWICH_20261123.md).

Stage 1 trains eight sandwich arms (one GPU each, DCLM long-study rows, 1x budget) with the loop-study
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

# f22 (26 Nov): Huginn component analysis (one component removed at a time from huginn_r3 at lr x2.0) and the last
# lr points (docs/FULL_LOOPED_METHODS_20261126.md).
JOBS = [(1, 'huginn_r3_lr28'), (0, 'huginn_r2_lr28'), (0, 'huginn_r3_lr20_prenorm'), (0, 'huginn_r3_lr20_adamw'),
        (0, 'huginn_r3_lr20_zero_state'), (0, 'huginn_r3_lr20_fixed_r'), (0, 'huginn_r3_lr20_add_inject'),
        (0, 'huginn_r3_lr20_no_emb_scale')]
RESCORE_BASE = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a03')
RESCORE = []  # the a03 rescore runs in a04


def prepare(base):
    loop.make_prepare(JOBS)(base)
    prepared, raw = downstream.pinned(*downstream.REQUESTS), downstream.pinned(*downstream.RAW)
    if raw['a2_metadata']['prepared_sha256'] != downstream.digest(prepared):
        raise ValueError('Raw template does not match requests')


def worker(base, seed, arm, action):
    if action == 'train':
        loop.worker(base, seed, arm, 'qualify')
        loop.worker(base, seed, arm, 'fit')
    elif action == 'score':
        (base/'downstream').mkdir(exist_ok=True)
        with mock.patch.object(downstream, 'SCREEN', base):
            downstream.worker(base/'downstream', seed, arm)
    else:  # rescore an earlier run's terminals (read-only on that run) into this run's directory
        (base/'rescore').mkdir(exist_ok=True)
        with mock.patch.object(downstream, 'SCREEN', RESCORE_BASE):
            downstream.worker(base/'rescore', seed, arm)


def export(base, names, label):
    archive = base/'export'/f'{label}.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive, 'w:gz') as t:
        for n in names:
            t.add(base/n, arcname=n)
    if archive.stat().st_size > 20_000_000:
        raise ValueError('Archive above export budget')
    write_json(base/'export'/f'{label}_manifest.json', dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                             members={n: sha(base/n) for n in names}))


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.dot_sandwich_study')
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
    export(base, names, 'train_results')  # training results survive any downstream failure
    try:
        stage(base, 'score', JOBS, 'scripts.dot_sandwich_study')
    except Exception as err:  # keep whatever finished; the supervisor stops remaining children afterwards
        write_json(base/'downstream_failure.json', dict(reason=repr(err)))
    done = []
    for seed, arm in JOBS:
        d = f'downstream/s{seed}-{arm}'
        if (base/d/'summary.json').is_file():
            s = json.loads((base/d/'summary.json').read_text())
            if sha(base/d/'items.npz') == s['items_sha256']:
                done += [f'{d}/summary.json', f'{d}/items.npz']
    if (base/'downstream_failure.json').is_file():
        done.append('downstream_failure.json')
    if RESCORE:
        try:
            stage(base, 'rescore', RESCORE, 'scripts.dot_sandwich_study')
        except Exception as err:
            write_json(base/'rescore_failure.json', dict(reason=repr(err)))
        for seed, arm in RESCORE:
            d = f'rescore/s{seed}-{arm}'
            if (base/d/'summary.json').is_file():
                s = json.loads((base/d/'summary.json').read_text())
                if sha(base/d/'items.npz') == s['items_sha256']:
                    done += [f'{d}/summary.json', f'{d}/items.npz']
        if (base/'rescore_failure.json').is_file():
            done.append('rescore_failure.json')
    if done:
        export(base, done, 'downstream_results')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'train', 'score', 'rescore'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1))
    p.add_argument('--arm', choices=sorted({a for _, a in JOBS + RESCORE}))
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
