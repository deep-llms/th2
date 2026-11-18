"""Reserve immutable follow-up windows without looking at evaluation losses."""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .data import Rows, sha
from .model import Config
from .train import write_json
from .evaluate import cluster_ids


def fresh_test_ids(rows, old_ids, maximum=4096, minimum=1024):
    eligible = rows.eligible()
    old_docs = cluster_ids(rows, old_ids)
    candidates = eligible[~np.isin(cluster_ids(rows, eligible), old_docs)]
    if len(candidates) < minimum:
        raise ValueError('Insufficient previously untouched final-set documents')
    return np.random.default_rng(44191).permutation(candidates)[:maximum]


def prepare(plan, out):
    out = Path(out); old = Path(plan['pilot'])
    if out.exists():
        raise ValueError('Fresh preparation output required')
    for path, want in plan['files'].items():
        if sha(path) != want:
            raise ValueError('Pinned input changed: '+path)
    prepared = json.loads((old/'inputs/complete.json').read_text())
    if sha(old/'inputs/complete.json') != plan['pilot_prepared_sha256']:
        raise ValueError('Pilot prepared manifest changed')
    for rel, want in prepared['files'].items():
        if sha(old/'inputs'/rel) != want:
            raise ValueError('Pilot selection changed')
    forbidden = []
    for seed in (0, 1):
        forbidden += [np.load(old/f'inputs/train-s{seed}.npy', allow_pickle=False),
                      np.load(old/f'inputs/eval-s{seed}.npz', allow_pickle=False)['fit']]
    forbidden = np.unique(np.concatenate(forbidden))
    fit_pool = np.setdiff1d(Rows(plan['stems']['fit']).eligible(), forbidden)
    needed = 4000*32
    if len(fit_pool) < needed+2048:
        raise ValueError('Insufficient fresh training windows')
    # Same set across seeds, different order. Selection independent of losses.
    order = np.random.default_rng(44190).permutation(fit_pool)
    train_set, route_fit = order[:needed], order[needed:needed+2048]
    old_test = np.load(old/'inputs/eval-s0.npz', allow_pickle=False)['test']
    tests = {'dclm': fresh_test_ids(Rows(plan['stems']['dclm']), old_test)}
    fwe = Rows(plan['stems']['fwe']).eligible()
    if len(fwe) < 4096:
        raise ValueError('Insufficient cross-domain test windows')
    tests['fwe'] = np.random.default_rng(44192).permutation(fwe)[:4096]
    out.mkdir(parents=True)
    for seed in (0, 1):
        ids = np.random.default_rng(44193+seed).permutation(train_set)
        np.save(out/f'train-s{seed}.npy', ids)
        np.savez(out/f'eval-s{seed}.npz', fit=route_fit, **tests)
        for arm in ('C', 'U', 'CF', 'PG'):
            indices = str(out/f'train-s{seed}.npy'); data = plan['stems']['fit']
            initial = str(old/f'runs/D-s{seed}/weights.pt')
            expected = {p: plan['files'][p] for p in (data+'.bin', data+'.starts', initial)}
            expected[indices] = sha(indices)
            config = dict(data=data, indices=indices, initial_weights=initial,
                          expected_inputs=expected, batch=32, steps=4000, arm=arm, seed=seed,
                          lr=.0001, router_lr=.001, exploration=.1, warmup=100, threads=4,
                          save_every=1000, log_every=50, model=asdict(Config()))
            write_json(out/f'{arm}-s{seed}.json', config)
    write_json(out/'complete.json', dict(status='ready', input_hashes=plan['files'],
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        training_fresh_rows=len(train_set), forbidden_pilot_rows=len(forbidden),
        routing_fit_rows=len(route_fit), training_row_overlap=0,
        heldout_counts={k:len(v) for k,v in tests.items()},
        dclm_test_document_overlap_with_pilot=0,
        flags=['conditional_suffix_LM','two_seeds','unequal_extra_forward_compute',
               'training_rows_disjoint_but_documents_may_recur','DCLM_hypothesis_selection_exploratory',
               'FWE_cross_domain_check','weights_node_only']))
    print('LOCAL_ORDER_READY', (out/'complete.json').read_text(), flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)


if __name__=='__main__':main()
