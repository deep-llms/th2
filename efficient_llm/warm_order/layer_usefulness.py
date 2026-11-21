"""Matched-context contrasts from a pinned frozen CPU panel; no forward or fit.

Does not alter the original panel/report or certify production data provenance.
Positive usefulness means lower NLL than skip within this specific context.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .layer_dependence_pipeline import cases, file_sha, replay


def contrasts(data, protocol):
    """Pair raw loss SUMS before token normalization; never divide by native gain."""
    panel = protocol['cases']
    if panel != json.loads(json.dumps(cases())):
        raise ValueError('Unsupported intervention panel')
    losses = np.asarray(data['nll_sum'])
    counts = np.asarray(data['token_count'])
    if (losses.ndim != 2 or losses.shape[0] != 50 or losses.shape[1] == 0
            or counts.shape != losses.shape or counts.dtype.kind not in 'iu'
            or not np.isfinite(losses).all() or (losses < 0).any()
            or (counts <= 0).any()
            or not np.array_equal(counts, np.broadcast_to(counts[0], counts.shape))):
        raise ValueError('Invalid or unmatched losses/token denominators')
    index = {(c['slot'], c['candidate']): i for i, c in enumerate(panel) if c['kind'] == 'common'}
    denominator = int(counts[0].sum())
    rows = []
    for slot in range(4):
        skip, native = losses[index[slot, None]], losses[index[slot, slot]]
        for candidate in range(4):
            value = losses[index[slot, candidate]]
            rows.append(dict(slot_0based=slot, candidate_0based=candidate,
                             middle_path_0based=[candidate if i == slot else i for i in range(4)],
                             intervention='native' if candidate == slot else 'substitution_with_repetition',
                             omitted_block_0based=None if candidate == slot else slot,
                             other_candidate_occurrence=('none' if candidate == slot else
                                                         'prefix' if candidate < slot else 'suffix'),
                             skip_case=panel[index[slot, None]]['name'],
                             native_case=panel[index[slot, slot]]['name'],
                             candidate_case=panel[index[slot, candidate]]['name'],
                             scored_tokens=denominator, sequences=losses.shape[1],
                             candidate_usefulness_nats=float((skip-value).sum()/denominator),
                             native_usefulness_nats=float((skip-native).sum()/denominator),
                             candidate_minus_native_nats=float((value-native).sum()/denominator),
                             native_mean_improves_over_skip=bool((skip-native).sum() > 0)))
    return rows


def analyze(parent, expected_manifest_sha256):
    """Read only; require externally pinned parent and replay before deriving."""
    parent = Path(parent)
    replay(parent, expected_manifest_sha256)
    protocol = json.loads((parent/'protocol.json').read_text())
    with np.load(parent/'metrics.npz', allow_pickle=False) as f:
        rows = contrasts({k: f[k] for k in ('nll_sum', 'token_count')}, protocol)
    return dict(schema=1, scope='Derived CPU-panel contrasts; no new forward, fitting or production evidence',
                production_ready=False, parent_manifest_sha256=expected_manifest_sha256,
                parent_model_sha256=protocol['model_sha256'],
                analysis_source_sha256=file_sha(Path(__file__)),
                sign_convention='usefulness = NLL(skip) - NLL(candidate); deficit = NLL(candidate) - NLL(native)',
                context='Each slot has one canonical prefix/input and frozen suffix. Only candidates within that slot share context.',
                limitations=['No automatic useful/bypass verdict or significance threshold.',
                             'Every off-diagonal substitution repeats the candidate and omits the native block; it is not inventory-preserving relocation.',
                             'No native-gain ratio: native gain can be zero or negative.',
                             'Row/context comparisons do not isolate input versus output mechanisms.',
                             'Source-document independence and checkpoint production admission not certified.'],
                rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=Path)
    p.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.parent, args.manifest_sha256), indent=2, allow_nan=False))
