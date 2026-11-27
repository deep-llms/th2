"""MoE-control + dynamic-AttnRes round, one owned handoff, eight GPUs (docs/MOE_DYN_ROUND_20261123.md).

a01 failed s1-loop_dynattn qualification (over-sharp dynamic perturbation); a02 uses realistic scales.

  moe line  (base/moe, exact 1x domain rows): standard top-k MoE controls moe2_top1, moe4_top1, moe8_top2
  loop line (base/loop, DCLM long-study rows): dynattn, dynattn8, loop_dynattn on seed 0; dynattn, loop_dynattn on seed 1
"""
import argparse
import json
from pathlib import Path
import tarfile

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.anchor_long_queue import stage, supervise
import scripts.loop_screen_study as loop
import scripts.router_screen_study as router

MOE_JOBS = [(0, 'moe2_top1'), (0, 'moe4_top1'), (0, 'moe8_top2')]
LOOP_JOBS = [(0, 'dynattn'), (0, 'dynattn8'), (0, 'loop_dynattn'), (1, 'dynattn'), (1, 'loop_dynattn')]
JOBS = MOE_JOBS + LOOP_JOBS


def prepare(base):
    (base/'moe').mkdir()
    router.prepare(base/'moe')
    (base/'loop').mkdir()
    loop.make_prepare(LOOP_JOBS)(base/'loop')
    write_json(base/'prepared.json', dict(jobs=JOBS))


def worker(base, seed, arm):
    if (seed, arm) in MOE_JOBS:
        router.worker(base/'moe', arm)
    else:
        loop.worker(base/'loop', seed, arm, 'qualify')
        loop.worker(base/'loop', seed, arm, 'fit')


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.moe_dyn_study')
    report, names = {}, ['prepared.json', 'moe/prepared.json', 'loop/prepared.json', 'execution/source.json']
    for seed, arm in JOBS:
        d = f'moe/{arm}' if (seed, arm) in MOE_JOBS else f'loop/s{seed}/{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if sha(base/d/'scores.npz') != s['score_sha256']:
            raise ValueError('Invalid terminal: '+d)
        if (seed, arm) in LOOP_JOBS and s['step'] != loop.STEPS[arm]:
            raise ValueError('Incomplete loop terminal: '+d)
        report[f's{seed}/{arm}'] = s
        names += [f'{d}/summary.json', f'{d}/scores.npz', f'{d}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'moe_dyn_results.tar.gz'
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
