"""CPU-only paired reanalysis of already exported frozen-model interventions.

No checkpoint loading, inference, optimization, network or remote commands.
Intervals resample documents conditional on each already-trained seed.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_interval(values, clusters, rng, draws=2000):
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    samples = []
    for start in range(0, draws, 50):
        ix = rng.integers(0, len(sums), size=(min(50, draws-start), len(sums)))
        samples.extend((sums[ix].sum(1) / counts[ix].sum(1)).tolist())
    return np.quantile(samples, [.025, .975]).tolist()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'results/order_diagnostics_20261118/export_a01/verified'
    out = root/'results/layer_dependency_20261120'
    out.mkdir(exist_ok=True)
    manifest = json.loads((source/'export-members.json').read_text())
    verified = {}

    def checked(relative):
        p = source/relative
        actual = digest(p)
        if actual != manifest[relative]:
            raise ValueError('Changed prior export: '+relative)
        verified[relative] = actual
        return p

    with checked('RESULTS.csv').open() as f:
        old_means = {(r['name'], r['domain'], r['metric']): float(r['nll'])
                     for r in csv.DictReader(f)}
    result = []
    rng = np.random.default_rng(20261120)
    by_swap = []
    swap_rng = np.random.default_rng(20261121)
    for seed in (0, 1):
        for domain in ('dclm', 'fwe'):
            arrays = {}
            for kind in ('attention', 'ffn', 'block'):
                p = checked(f'runs/{kind}-s{seed}/{domain}/summary.npz')
                with np.load(p, allow_pickle=False) as d:
                    arrays[kind] = {k: d[k].copy() for k in d.files}
                a = arrays[kind]
                for k in range(4):
                    v = a[f'path{k}']
                    if v.shape != a['ids'].shape or not np.isfinite(v).all():
                        raise ValueError('Invalid per-sequence costs')
                    np.testing.assert_allclose(v.mean(), old_means[(f'{kind}-s{seed}', domain, f'path{k}')], rtol=0, atol=1e-10)
            reference = arrays['block']
            for a in arrays.values():
                for key in ('ids', 'clusters'):
                    np.testing.assert_array_equal(a[key], reference[key])
                np.testing.assert_allclose(a['path0'], reference['path0'], rtol=0, atol=2e-4)
            deltas = {kind: np.mean([a[f'path{k}']-a['path0'] for k in (1, 2, 3)], axis=0)
                      for kind, a in arrays.items()}
            # Same swap applied to attention and FFN equals the whole-block
            # intervention, including each component's own trained norm.
            deltas['interaction'] = deltas['block']-deltas['attention']-deltas['ffn']
            for k in (1, 2, 3):
                separate = {kind: a[f'path{k}']-a['path0'] for kind, a in arrays.items()}
                separate['interaction'] = separate['block']-separate['attention']-separate['ffn']
                for kind, values in separate.items():
                    lo, hi = document_interval(values, reference['clusters'], swap_rng)
                    by_swap.append(dict(seed=seed, domain=domain, contrast=kind,
                                        middle_slots_1based=f'{k}-{k+1}', delta_nats=float(values.mean()),
                                        ci_low=lo, ci_high=hi, sequences=len(values),
                                        documents=len(np.unique(reference['clusters'])), scored_tokens_per_sequence=480))
            for kind, values in deltas.items():
                np.testing.assert_allclose(values.mean(), np.mean([r['delta_nats'] for r in by_swap
                    if r['seed']==seed and r['domain']==domain and r['contrast']==kind]), rtol=0, atol=1e-14)
                lo, hi = document_interval(values, reference['clusters'], rng)
                result.append(dict(seed=seed, domain=domain, contrast=kind,
                                   delta_nats=float(values.mean()), ci_low=lo, ci_high=hi,
                                   sequences=len(values), documents=len(np.unique(reference['clusters'])),
                                   scored_tokens_per_sequence=480))
    csv_path = out/'existing_component_effects.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(result)
    swap_path = out/'per_swap_component_effects.csv'
    with swap_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(by_swap[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(by_swap)
    means = {domain: {kind: float(np.mean([r['delta_nats'] for r in result
                                          if r['domain']==domain and r['contrast']==kind]))
                      for kind in ('attention', 'ffn', 'block', 'interaction')}
             for domain in ('dclm', 'fwe')}
    proof = dict(status='PASS', source_study='order_diagnostics_20261118',
                 scope='Reanalysis of existing per-sequence exports; no new model execution',
                 checkpoints='Original warm-order pilot dense D seed0/1, not Horizon',
                 intervention='Mean of three adjacent middle-pair swaps; component travels with its norm',
                 weighting='Equal 480 scored tokens per sequence; paired sequence means are token weighted',
                 bootstrap='2000 document-cluster draws per seed/domain/contrast; descriptive pointwise 95% intervals, no multiplicity adjustment',
                 rng_seed=20261120, means_across_two_seeds=means, rows=len(result),
                 verified_inputs=verified, script_sha256=digest(Path(__file__)),
                 output_sha256=digest(csv_path),
                 per_swap_rows=len(by_swap), per_swap_output_sha256=digest(swap_path), per_swap_rng_seed=20261121,
                 flags=['Historical reused final sets: exploratory', 'Two trained seeds',
                        'Intervals condition on fitted seed; not training-seed uncertainty',
                        'Interaction is not a percentage attribution or isolated mechanism',
                        'Full token-cost arrays and checkpoint weights remain on-node'])
    (out/'existing_component_replay.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', rows=len(result), means=means)))


if __name__ == '__main__':
    main()
