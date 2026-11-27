"""Scale round: is the data-limited regime why routing/specialization lost? (docs/SCALE_ROUND_20261123.md)

One owned handoff, seven GPUs in parallel:
  loop line at 4x data (49,152 updates): loop_attnres, attnres (4 blocks), attnres8, dense8, dense4
  domain line at 8x data (98,304 updates, alternating DCLM/FWE): single vs oracle domain routing
Sub-roots base/loop and base/domain keep each study's own layout and receipts.
"""
import argparse
import json
from pathlib import Path
import tarfile

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.anchor_long_queue import stage, supervise
import scripts.domain_screen_study as domain
import scripts.loop_screen_study as loop

LOOP_JOBS = [(0, 'loop_attnres_4x'), (0, 'attnres_4x'), (0, 'attnres8_4x'), (0, 'dense8_4x'), (0, 'dense4_4x')]
DOMAIN_JOBS = [(0, 'domain_single'), (0, 'domain_oracle')]
JOBS = LOOP_JOBS + DOMAIN_JOBS


def prepare(base):
    (base/'loop').mkdir()
    loop.make_prepare(LOOP_JOBS)(base/'loop')
    (base/'domain').mkdir()
    domain.prepare(base/'domain')
    write_json(base/'prepared.json', dict(jobs=JOBS, domain_scale=domain.SCALE, domain_steps=domain.STEPS))


def worker(base, seed, arm):
    if arm.startswith('domain_'):
        domain.worker(base/'domain', arm[len('domain_'):])
    else:
        loop.worker(base/'loop', seed, arm, 'qualify')
        loop.worker(base/'loop', seed, arm, 'fit')


def experiment(base):
    stage(base, 'train', JOBS, 'scripts.scale_round_study')
    report, names = {}, ['prepared.json', 'loop/prepared.json', 'domain/prepared.json', 'execution/source.json']
    for seed, arm in LOOP_JOBS:
        d = f'loop/s{seed}/{arm}'
        s = json.loads((base/d/'summary.json').read_text())
        if s['step'] != loop.STEPS[arm] or sha(base/d/'scores.npz') != s['score_sha256']:
            raise ValueError('Invalid loop terminal: '+arm)
        report[arm] = s
        names += [f'{d}/summary.json', f'{d}/scores.npz', f'{d}/calibration.jsonl']
    for _, arm in DOMAIN_JOBS:
        d = f'domain/{arm[len("domain_"):]}'
        s = json.loads((base/d/'summary.json').read_text())
        if sha(base/d/'scores.npz') != s['score_sha256']:
            raise ValueError('Invalid domain terminal: '+arm)
        report[arm] = s
        names += [f'{d}/summary.json', f'{d}/scores.npz', f'{d}/calibration.jsonl']
    write_json(base/'report.json', report)
    names.append('report.json')
    archive = base/'export'/'scale_results.tar.gz'
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
        worker(a.base, a.seed, a.arm)
