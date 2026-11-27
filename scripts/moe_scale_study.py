"""MoE at 4x data with seed-1 static references (docs/MOE_SCALE_20261123.md).

One owned handoff, eight GPUs, DCLM long-study rows (extended into the other seed array), 49,152 updates.
"""
import argparse
import json
from pathlib import Path
import tarfile

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.anchor_long_queue import stage, supervise
import scripts.loop_screen_study as loop

JOBS = [(0, 'attnres_moe_4x'), (0, 'loop_attnres_moe_4x'), (0, 'attnres8_moe_4x'), (1, 'attnres_moe_4x'),
        (1, 'loop_attnres_moe_4x'), (1, 'attnres_4x'), (1, 'loop_attnres_4x'), (1, 'attnres8_4x')]


def prepare(base):
    loop.make_prepare(JOBS)(base)


def worker(base, seed, arm):
    loop.worker(base, seed, arm, 'qualify')
    loop.worker(base, seed, arm, 'fit')


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.moe_scale_study')
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
    archive = base/'export'/'moe_scale_results.tar.gz'
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
        worker(a.base, a.seed, a.arm)
