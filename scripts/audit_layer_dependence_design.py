"""Local design/evidence audit: no model execution, fitting or remote access."""
from collections import Counter
from fractions import Fraction
from itertools import permutations
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def exact_rank(rows):
    a = [[Fraction(v) for v in row] for row in rows]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        divisor = a[r][c]
        a[r] = [v/divisor for v in a[r]]
        for i in range(r+1, len(a)):
            factor = a[i][c]
            a[i] = [x-factor*y for x, y in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def audit():
    base = ROOT/'results/horizon_20261119'
    handoff = json.loads((base/'node_handoff_verified.json').read_text())
    archive = base/'export_a01/results.tar.gz'
    assert sha(archive) == handoff['archive_sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        members = [m for m in tar if m.name.removeprefix('./') == 'export-members.json']
        assert len(members) == 1 and members[0].isfile()
        manifest = json.load(tar.extractfile(members[0]))
    source = base/'export_a01/verified/execution/source.json'
    assert sha(source) == manifest['execution/source.json']
    pins = json.loads(source.read_text())
    sources = {name: sha(ROOT/name) for name in (
        'efficient_llm/warm_order/model.py',
        'efficient_llm/warm_order/progressive.py',
        'efficient_llm/warm_order/horizon_train.py')}
    assert all(pins[name] == digest for name, digest in sources.items())
    # Import definitions only; no Model construction or training entry point.
    from efficient_llm.warm_order.model import ORDERS
    expected = tuple(tuple((start+direction*j) % 4 for j in range(4))
                     for direction in (1, -1) for start in range(4))
    assert ORDERS == expected
    paths = list(permutations(range(4)))
    edges = list(permutations(range(4), 2))
    seen = {e for p in ORDERS for e in zip(p, p[1:])}
    slot_counts = [[sum(p[s] == b for p in ORDERS) for s in range(4)] for b in range(4)]
    assert slot_counts == [[2]*4 for _ in range(4)]
    novel = [p for p in paths if p not in ORDERS]
    novel_hist = Counter(sum(e not in seen for e in zip(p, p[1:])) for p in novel)
    assert len(seen) == 8 and novel_hist == {1: 8, 2: 8}
    # Exact alias, not an inference about physical causes from rank alone.
    for p in paths:
        for b in range(4):
            assert sum(v == b for _, v in zip(p, p[1:])) == 1-int(p[0] == b)
            assert sum(u == b for u, _ in zip(p, p[1:])) == 1-int(p[-1] == b)
    design = [[1]+[int(p[s] == b) for b in range(4) for s in range(4)]
              +[int(e in list(zip(p, p[1:]))) for e in edges] for p in paths]
    rank = exact_rank(design)
    assert rank == 15
    old = ROOT/'results/layer_dependency_20261120/existing_component_replay.json'
    proof = json.loads(old.read_text())
    export = ROOT/'results/order_diagnostics_20261118/export_a01/verified'
    for name, digest in proof['verified_inputs'].items():
        assert sha(export/name) == digest
    contrast = []
    for seed in (0, 1):
        for domain in ('dclm', 'fwe'):
            arrays = {}
            for kind in ('attention', 'ffn', 'block'):
                with np.load(export/f'runs/{kind}-s{seed}/{domain}/summary.npz', allow_pickle=False) as f:
                    arrays[kind] = {k: f[k].copy() for k in f.files}
            a, f, b = (arrays[k] for k in ('attention', 'ffn', 'block'))
            for x in (a, f):
                for key in ('ids', 'clusters', 'path0'):
                    np.testing.assert_array_equal(x[key], b[key])
            for k in (1, 2, 3):
                value = float(np.mean((a[f'path{k}']-a['path0'])-(f[f'path{k}']-f['path0'])))
                contrast.append(dict(seed=seed, domain=domain, pair_1based=[k, k+1],
                                     attention_minus_ffn_nats=value))
    return dict(scope='Combinatorial/source audit and historical aggregate replay only; no model run',
                production_ready=False, audit_source_sha256=sha(Path(__file__)),
                horizon_archive_sha256=sha(archive), verified_sources=sources,
                slot_counts=slot_counts, trained_full_depth_orders=ORDERS,
                seen_directed_pairs_0based=sorted(seen),
                missing_directed_pairs_0based=sorted(set(edges)-seen),
                unseen_path_new_adjacency_histogram=dict(novel_hist),
                additive_design=dict(rows=24, columns=29, exact_rank=rank,
                                     warning='Includes coding redundancies and slot/neighbor aliases; no unique causal factor decomposition'),
                historical_proof_sha256=sha(old), historical_inputs_verified=len(proof['verified_inputs']),
                baseline_arrays_exactly_equal=True, attention_minus_ffn=contrast,
                limitations=['Route-family enumeration does not measure effect sizes.',
                             'Historical reused final sets and two fitted seeds; exploratory.',
                             'Original-document identity not newly verified.'])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
