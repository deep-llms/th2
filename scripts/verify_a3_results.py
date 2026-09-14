"""Dev-only: verify/extract the A3 result bundle and check against original evals."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np

ARMS = ('contextual', 'isolated', 'shuffled', 'grad', 'base')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def extract(bundle, dest):
    m = json.loads((bundle/'manifest.json').read_text())
    check(m['success'] is True, 'Incomplete export')
    expected_names = {'definition/bins.json', 'definition/bins.npz', 'full/summary.json'}
    expected_names |= {f'full/{a}/{f}' for a in ARMS for f in ('diagnostics.json', 'statistics.npz')}
    expected = {r['path']: r for r in m['files']}
    check(len(m['files']) == len(expected) == 13 and set(expected) == expected_names, 'Wrong archive contract')
    parts = []
    for i, item in enumerate(m['parts']):
        check(item['path'] == f'results.part{i:03d}', 'Invalid part name/order')
        content = (bundle/item['path']).read_bytes()
        check(len(content) == item['bytes'] and sha(content) == item['sha256'], 'Bad part')
        parts.append(content)
    archive = b''.join(parts)
    check(sha(archive) == m['archive_sha256'], 'Bad archive')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        members = tar.getmembers()
        check(len(members) == 13 and {x.name for x in members} == expected_names, 'Wrong archive membership')
        payloads = {}
        for item in members:
            check(item.isfile() and 0 <= item.size <= 256*1024**2, 'Unsafe/oversize member')
            content = tar.extractfile(item).read()
            record = expected[item.name]
            check(len(content) == record['bytes'] and sha(content) == record['sha256'], 'Bad member')
            payloads[item.name] = content
    dest.mkdir(exist_ok=False)
    for rel, content in payloads.items():
        path = dest/rel
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('xb') as f:
            f.write(content)
    return m


def validate(dest, original, manifest):
    summary = json.loads((dest/'full/summary.json').read_text())
    bm = json.loads((dest/'definition/bins.json').read_text())
    check(summary['bin_definition'] == bm and bm['slots'] == 262144, 'Wrong bin metadata')
    check(sha((dest/'definition/bins.npz').read_bytes()) == bm['npz_sha256'], 'Bad bins')
    with np.load(dest/'definition/bins.npz', allow_pickle=False) as f:
        bins = {k: f[k] for k in f.files}
    for feature, field in (('compile_counts', 'frequency_bin'), ('variance', 'variance_bin')):
        edges = np.quantile(bins[feature], np.arange(1,5)/5)
        check(np.array_equal(np.searchsorted(edges,bins[feature],side='right'),bins[field]), 'Bin assignment changed')
    check(np.array_equal(bins['cell_ids'], bins['frequency_bin']*5+bins['variance_bin']), 'Wrong cells')
    reference_arrays = None
    for arm in ARMS:
        root = dest/'full'/arm
        r = json.loads((root/'diagnostics.json').read_text())
        old = json.loads((original/arm/'metrics.json').read_text())
        check(r == summary['arms'][arm] and not r['smoke'] and r['input_tokens'] == 20_000_000, 'Incomplete report')
        check(r['checkpoint_hash'] == old['checkpoint_hash'] and r['corpus_hash'] == old['corpus_hash']
              and r['vocabulary_hash'] == old['vocabulary_hash'], 'Source identity mismatch')
        check(r['source_metrics_sha256'] == sha((original/arm/'metrics.json').read_bytes()), 'Original metadata changed')
        check(sha((original/arm/'segments.jsonl').read_bytes()) == old['segments_sha256'], 'Original records changed')
        check(r['statistics_sha256'] == sha((root/'statistics.npz').read_bytes()), 'Bad statistics')
        with np.load(root/'statistics.npz', allow_pickle=False) as f:
            x = {k: f[k] for k in f.files}
        for key in ('key_loss_sum','segment_cell_loss_sum'):
            check(x[key].dtype == np.float64 and np.isfinite(x[key]).all(), 'Bad FP64 sums')
        for key in ('key_target_count','segment_cell_target_count'):
            check(x[key].dtype == np.int64 and (x[key]>=0).all(), 'Bad INT64 counts')
        if reference_arrays is not None:
            for key in ('segment_id','doc_id','content_hash','key_target_count','segment_cell_target_count'):
                check(np.array_equal(x[key], reference_arrays[key]), 'Unpaired arms')
        else:
            reference_arrays = x
        lines = (original/arm/'segments.jsonl').read_text().splitlines()
        check(len(lines) == r['segments'] == len(x['segment_id']) == 27926, 'Wrong segment count')
        for i,line in enumerate(lines):
            row = json.loads(line)
            for k in ('segment_id','doc_id','content_hash'):
                check(x[k][i] == row[k], 'Unpaired source segment')
            check(x['segment_cell_target_count'][i].sum() == row['hit'][1], 'Wrong per-segment hits')
            check(np.isclose(x['segment_cell_loss_sum'][i].sum(),row['hit'][0],rtol=1e-12,atol=1e-8), 'Wrong per-segment loss')
        counts = np.bincount(bins['cell_ids'],weights=x['key_target_count'],minlength=25).astype(np.int64)
        sums = np.bincount(bins['cell_ids'],weights=x['key_loss_sum'],minlength=25)
        check(np.array_equal(counts,r['cell_target_count']) and counts.sum()==12942338, 'Wrong cell counts')
        check(np.allclose(sums,r['cell_loss_sum'],rtol=1e-12,atol=1e-7), 'Wrong cell loss sums')
        for cell in range(25):
            cell_report = summary['cells'][cell]
            check(cell_report['frequency_bin']==cell//5+1 and cell_report['variance_bin']==cell%5+1, 'Wrong cell labels')
            check(cell_report['target_count']==counts[cell], 'Wrong summary count')
            value = sums[cell]/counts[cell] if counts[cell] else None
            actual = cell_report['nll'][arm]
            check(actual is None if value is None else np.isclose(actual,value,rtol=1e-12,atol=1e-12), 'Wrong cell NLL')
        print(arm+' A3_HASHES_COUNTS_PAIRING_AND_ORIGINAL_REPLAY_VERIFIED',flush=True)
    for cell in summary['cells']:
        for arm in ARMS[1:]:
            expected = cell['nll']['contextual']-cell['nll'][arm] if cell['target_count'] else None
            check(cell['contextual_minus'][arm]==expected, 'Wrong contrast')
    result = dict(success=True, verified_files=13, arms=list(ARMS), input_tokens_per_arm=20000000,
                  hit_targets_per_arm=12942338, cells=25, original_segment_alignment_verified=True,
                  archive_sha256=manifest['archive_sha256'])
    with (dest/'VERIFIED.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    a=p.parse_args()
    m=extract(a.bundle,a.output)
    validate(a.output,a.original,m)
