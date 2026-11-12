"""CPU-only same-relation re-entry; immutable source routes, additive outputs."""
import argparse
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.data import sha, write


def same_relation(a, b, relations, bridges, correct, draws=10000):
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 2):
        raise ValueError('Expected paired four-step top2 routes')
    n = len(a)
    if any(len(x) != n for x in (relations, bridges, correct)) or n == 0:
        raise ValueError('Unaligned item metadata')
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= 64:
        raise ValueError('Invalid expert IDs')
    aa = [set(x.ravel().tolist()) for x in a]
    bb = [set(x.ravel().tolist()) for x in b]
    actual = np.array([len(x & y) / len(x | y) for x, y in zip(aa, bb)])
    chance = np.full(n, np.nan); partners = np.zeros(n, int)
    for i in range(n):
        ix = np.flatnonzero((relations == relations[i]) & (bridges != bridges[i]))
        partners[i] = len(ix)
        if len(ix):
            chance[i] = np.mean([len(aa[i] & bb[j]) / len(aa[i] | bb[j]) for j in ix])
    valid = partners > 0; groups = {}; rng = np.random.default_rng(0)
    for name, subset in [('all', np.ones(n, bool)), ('correct', correct), ('wrong', ~correct)]:
        mask = valid & subset; delta = (actual - chance)[mask]
        ci = np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)], [.025, .975]).tolist() if len(delta) else None
        groups[name] = dict(n=int(mask.sum()), excluded_no_partner=int((subset & ~valid).sum()),
                            overlap=float(actual[mask].mean()) if len(delta) else None,
                            chance_same=float(chance[mask].mean()) if len(delta) else None,
                            excess_same=float(delta.mean()) if len(delta) else None, ci=ci,
                            label='bridge-specific re-entry' if ci and ci[0] > 0 else 'not bridge-specific')
    return dict(groups=groups, bootstrap_seed=0, bootstrap_draws=draws,
                baseline='all other OOD-test items with same r2 and different bridge; equal item weighting',
                partner_counts=partners.tolist(), actual=actual.tolist(),
                chance_same=[float(x) if np.isfinite(x) else None for x in chance])


def analyze(routes, data, output):
    routes, data, output = Path(routes), Path(data), Path(output)
    if output.exists():raise ValueError('Refuse overwrite')
    q = np.load(data / 'ood_test.npy'); bridge = np.load(data / 'ood_bridge.npy')
    if not np.array_equal(q[:, 2], bridge[:, 1]) or not np.array_equal(q[:, 3], bridge[:, 2]):
        raise ValueError('Bridge alignment mismatch')
    with np.load(routes) as r:
        out = same_relation(r['ood_test_ids'][:, 4:8], r['ood_bridge_ids'][:, :4], q[:, 2], bridge[:, 0], r['ood_test_correct'].astype(bool))
    out['inputs'] = {str(p):sha(p) for p in [routes, data/'ood_test.npy', data/'ood_bridge.npy']}
    write(output, out)
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--routes',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    analyze(a.routes,a.data,a.output)
