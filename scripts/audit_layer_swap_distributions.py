"""Descriptive paired-sequence audit of authenticated historical exports only."""
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_layer_dependence_design import audit


def main():
    # Reuse archive/source/input authentication; this does not run a model.
    audit()
    root = Path(__file__).resolve().parents[1]
    prior = root/'results/layer_dependency_20261120/existing_component_replay.json'
    verified = json.loads(prior.read_text())['verified_inputs']
    base = root/'results/order_diagnostics_20261118/export_a01/verified'
    rows = []
    inventories = []
    for seed in (0, 1):
        for domain in ('dclm', 'fwe'):
            for kind in ('attention', 'ffn', 'block'):
                relative = f'runs/{kind}-s{seed}/{domain}/summary.npz'
                p = base/relative
                assert hashlib.sha256(p.read_bytes()).hexdigest() == verified[relative]
                with np.load(p, allow_pickle=False) as a:
                    assert set(a.files) == {'path0', 'path1', 'path2', 'path3', 'ids', 'clusters'}
                    inventories.append(dict(file=relative, keys=sorted(a.files),
                                            contains_skip_or_replacement_costs=False))
                    for pair in (1, 2, 3):
                        delta = a[f'path{pair}'] - a['path0']
                        assert delta.shape == a['ids'].shape and np.isfinite(delta).all()
                        positive = np.maximum(delta, 0).mean()
                        negative = np.maximum(-delta, 0).mean()
                        assert np.isclose(positive-negative, delta.mean(), rtol=0, atol=1e-12)
                        rows.append(dict(seed=seed, domain=domain, component=kind,
                                         pair_1based=[pair, pair+1], sequences=len(delta),
                                         mean_delta_nats=float(delta.mean()),
                                         positive_fraction=float((delta > 0).mean()),
                                         negative_fraction=float((delta < 0).mean()),
                                         zero_fraction=float((delta == 0).mean()),
                                         quantiles_10_50_90=np.quantile(delta, [.1, .5, .9]).tolist(),
                                         positive_part_mean=float(positive),
                                         negative_part_magnitude_mean=float(negative)))
    print(json.dumps(dict(scope='Historical pilot D seed0/1 swap sensitivity; not Horizon or usefulness versus skip',
                          new_model_runs=0, rows=rows, inspected_summaries=inventories,
                          input_sha256=verified,
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          flags=['Historical reused final sets; descriptive only.',
                                 'Fractions count 512-token sequences with 480 scored targets, not independent documents.',
                                 'A positive sign is not a practical-significance threshold.',
                                 'No confidence interval or population/generalization claim.',
                                 'No skip costs in these 12 summaries; do not infer block usefulness from swaps.',
                                 'This absence claim concerns only the inspected exports, not all historical experiments.']),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
