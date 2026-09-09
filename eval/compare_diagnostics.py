"""Compare frozen-bundle frequency reports at matched training checkpoints."""
import argparse
import json
from pathlib import Path
from capacity_allocation.data import write_json
from eval.diagnostic_data import BUCKETS


def compare(reference, candidate):
    if reference.get('success') is not True or candidate.get('success') is not True:
        raise ValueError('Only complete diagnostic reports can be compared')
    for key in ('diagnostic_manifest_sha256', 'languages'):
        if reference[key] != candidate[key]:
            raise ValueError(f'Mismatched diagnostic {key}')
    for key in ('global_step', 'consumed_input_tokens', 'consumed_scored_targets'):
        if reference['training'].get(key) != candidate['training'].get(key):
            raise ValueError(f'Unmatched training checkpoint: {key}')
    if reference['settings']['precision'] != candidate['settings']['precision']:
        raise ValueError('Cannot compare different scoring precision')
    result = {}
    for lang in reference['languages']:
        a = reference['diagnostics']['frequency']['by_language'][lang]
        b = candidate['diagnostics']['frequency']['by_language'][lang]
        result[lang] = {}
        for bucket in BUCKETS:
            aa, bb = a['buckets'][bucket], b['buckets'][bucket]
            if aa['scored_targets'] != bb['scored_targets']:
                raise ValueError('Bucket target coverage mismatch')
            result[lang][bucket] = dict(scored_targets=aa['scored_targets'],
                delta_nll=bb['nll']-aa['nll'] if aa['scored_targets'] else None)
    return dict(reference=reference['checkpoint'], candidate=candidate['checkpoint'],
        delta_convention='candidate_minus_reference_negative_is_better', by_language=result,
        uncertainty='not_estimated_do_not_make_central_tail_claims_without_paired_uncertainty')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    reports = [compare(reference, json.loads(Path(p).read_text())) for p in args.candidates]
    write_json(args.output, dict(comparisons=reports))


if __name__ == '__main__':
    main()
