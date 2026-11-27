"""Enumerate frozen-context contrasts; no model execution, fitting or remote IO."""
import hashlib
import itertools
import json
from pathlib import Path


def audit():
    # Hold stem/coda fixed and use each of four middle blocks exactly once.
    rows = []
    for cut in range(4):
        prefix, suffix = tuple(range(cut)), tuple(range(cut + 1, 4))
        p = list(itertools.permutations(prefix))
        s = list(itertools.permutations(suffix))
        rows.append(dict(slot_0based=cut, prefix_orders=len(p), suffix_orders=len(s),
                         nontrivial_two_sided_cross=len(p) > 1 and len(s) > 1))
    assert not any(r['nontrivial_two_sided_cross'] for r in rows)
    assert [(r['prefix_orders'], r['suffix_orders']) for r in rows] == [(1, 6), (1, 2), (2, 1), (6, 1)]
    panels = [
        dict(name='prefix_order_at_native_C', target=2,
             native=((0, 1, 2, 3), (1, 0, 2, 3)), skip=((0, 1, 3), (1, 0, 3)),
             held_fixed='C at third middle slot; suffix D + coda'),
        dict(name='suffix_order_at_native_B', target=1,
             native=((0, 1, 2, 3), (0, 1, 3, 2)), skip=((0, 2, 3), (0, 3, 2)),
             held_fixed='B at second middle slot; prefix stem + A'),
    ]
    all_paths = set()
    for panel in panels:
        for native, skip in zip(panel['native'], panel['skip']):
            assert sorted(native) == list(range(4))
            assert tuple(b for b in native if b != panel['target']) == skip
            all_paths.update((native, skip))
    existing = set(itertools.permutations(range(4)))
    existing.update(tuple(b for b in range(4) if b != t) for t in range(4))
    extra = sorted(all_paths - existing)
    assert len(all_paths) == 7 and extra == [(0, 3, 2), (1, 0, 3)]
    root = Path(__file__).resolve().parents[1]
    files = ['efficient_llm/warm_order/model.py',
             'efficient_llm/warm_order/layer_dependence.py',
             'efficient_llm/warm_order/layer_dependence_pipeline.py']
    return dict(scope='Combinatorial design audit only; no measured model effects',
                numbering='0-based middle blocks A=0, B=1, C=2, D=3; stem/coda fixed',
                fixed_partition_order_counts=rows, proposed_panels=panels,
                unique_paths=sorted(all_paths), extra_skip_paths=extra,
                minimum_middle_blocks_for_nontrivial_fixed_partition_cross=5,
                implemented_or_admitted=False,
                limitations=['Panels concern two different native blocks/slots, not one block at one cut.',
                             'A prefix/suffix order effect includes changed semantic computation, not pure interface mismatch.',
                             'Skip changes executed depth; it is a marginal contribution comparator, not equal-compute control.',
                             'Mean zero contribution can hide cancellation; it does not establish bypass.',
                             'No inference about actual weights, data provenance, significance or mechanism shares.'],
                source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
