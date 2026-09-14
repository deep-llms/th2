"""A3: shared frequency x Contextual-variance grid on seed-17 Stage-2 D_dev.

Descriptive, post-hoc hit-target analysis. No training/core changes and no D_val.
Keep per-key and per-segment cell sufficient statistics for later analyses.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch
from ccm.artifacts import load_table
from ccm.cli import code_hash
from ccm.compiler import batches
from ccm.contracts import require, fresh_dir, read_json, write_json, file_hash
from ccm.data import collate
from ccm.runtime import checkpoint_meta, load_model, model_inputs, to_device
from memory_diagnostics import checked_inputs, reference, masks_for, check_pair, POPS

ARMS = ('contextual', 'isolated', 'shuffled', 'grad', 'base')
CORE = '055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9'


def quintiles(values):
    values = np.asarray(values)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), 'Invalid bin features')
    edges = np.quantile(values, np.arange(1, 5)/5)
    return np.searchsorted(edges, values, side='right').astype(np.int8), edges


def add_row(key_sums, key_counts, cell_ids, slots, loss, hit):
    slots, loss = slots[hit], loss[hit]
    require(np.isfinite(loss).all() and (slots >= 0).all() and (slots < len(key_sums)).all(), 'Invalid hit input')
    # add.at correctly counts repeated occurrences of the same key.
    np.add.at(key_sums, slots, loss)
    np.add.at(key_counts, slots, 1)
    return (np.bincount(cell_ids[slots], weights=loss, minlength=25),
            np.bincount(cell_ids[slots], minlength=25).astype(np.int64))


def means(sums, counts):
    return [float(s/n) if n else None for s, n in zip(sums, counts)]


def preflight(a):
    require(code_hash() == CORE, 'Frozen research core changed')
    for arm in ARMS:
        corpus, vocab, old = checked_inputs(a.data, a.stage2, arm)
        ck = checkpoint_meta(a.stage2/'train'/arm/'checkpoint-3815')
        require(ck['model_sha256'] == old['checkpoint_hash'] and ck['arm'] == arm
                and ck['phase'] == 'stage2' and ck['seed'] == 17 and ck['step'] == 3815,
                'Invalid source checkpoint')
    tensors, meta = load_table(a.table, dict(constructor='contextual', vocabulary_hash=vocab.hash,
        corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=old['source_checkpoint_hash']))
    counts, variance = tensors['counts'].numpy(), tensors['variance'].numpy()
    require(np.array_equal(counts, vocab.counts), 'Compile counts differ from vocabulary')
    frequency_bin, fe = quintiles(counts)
    variance_bin, ve = quintiles(variance)
    out = fresh_dir(a.definition)
    with (out/'bins.npz').open('xb') as f:
        np.savez_compressed(f, keys=vocab.keys, compile_counts=counts, variance=variance,
                           frequency_bin=frequency_bin, variance_bin=variance_bin,
                           cell_ids=frequency_bin*5+variance_bin)
    write_json(out/'bins.json', dict(schema=1, posthoc=True, role='dev', seed=17,
        corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash,
        source_checkpoint_hash=old['source_checkpoint_hash'], diagnostic_table_hash=meta['artifact_hash'],
        slots=len(vocab.keys), npz_sha256=file_hash(out/'bins.npz'),
        frequency_edges=fe.tolist(), variance_edges=ve.tolist(),
        definition='Unweighted quintiles over all selected keys; low to high; searchsorted right; ties not split',
        population='Original hit targets only; identical Contextual-defined bins for every arm'))
    print('A3_FROZEN_INPUTS_AND_SHARED_BIN_DEFINITION_VERIFIED', flush=True)


def definition(path):
    meta = read_json(path/'bins.json')
    require(file_hash(path/'bins.npz') == meta['npz_sha256'], 'Bin definition checksum mismatch')
    with np.load(path/'bins.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    require(np.array_equal(arrays['cell_ids'], arrays['frequency_bin']*5+arrays['variance_bin']), 'Wrong cell mapping')
    require(((arrays['cell_ids'] >= 0) & (arrays['cell_ids'] < 25)).all(), 'Invalid cells')
    return arrays, meta


@torch.no_grad()
def evaluate(a):
    require(a.max_batches >= 0 and code_hash() == CORE, 'Invalid smoke length or changed research core')
    corpus, vocab, old = checked_inputs(a.data, a.stage2, a.arm)
    bins, bmeta = definition(a.definition)
    require(bmeta['vocabulary_hash'] == vocab.hash and bmeta['corpus_hash'] == old['corpus_hash']
            and bmeta['source_checkpoint_hash'] == old['source_checkpoint_hash']
            and np.array_equal(bins['keys'], vocab.keys), 'Bins/reference identity mismatch')
    model, ck = load_model(a.stage2/'train'/a.arm/'checkpoint-3815', a.device)
    require(ck['model_sha256'] == old['checkpoint_hash'] and ck['arm'] == a.arm, 'Wrong model')
    model.set_phase('eval')
    out = fresh_dir(a.output)
    key_sums, key_counts = np.zeros(len(vocab.keys), np.float64), np.zeros(len(vocab.keys), np.int64)
    cell_sums, cell_counts, ids, doc_ids, content_ids = [], [], [], [], []
    totals = {p: [0., 0] for p in POPS}
    inputs = 0
    max_error = abs_error = 0.
    start = time.monotonic()
    with (a.stage2/'eval'/a.arm/'segments.jsonl').open() as originals:
        for j, rows in enumerate(batches(corpus.segments('dev'), 8)):
            if a.max_batches and j >= a.max_batches:
                break
            b = to_device(collate(rows, corpus.meta['special_ids'], vocab), a.device)
            loss = model(**model_inputs(b), loss_chunk=1024)['losses'].double().cpu().numpy()
            masks = {k: v.cpu().numpy() for k, v in masks_for(b).items()}
            slots = b['slots'].cpu().numpy()
            inputs += int(b['attention_mask'].sum())
            for i, row in enumerate(rows):
                record = {k: row[k] for k in ('doc_id', 'content_hash', 'segment_id')}
                source = json.loads(next(originals))
                for p, mask in masks.items():
                    s, n = float(loss[i][mask[i]].sum()), int(mask[i].sum())
                    record[p] = [s, n]
                    totals[p][0] += s
                    totals[p][1] += n
                max_error = max(max_error, check_pair(record, source, 2e-5))
                abs_error += abs(record['overall'][0]-source['overall'][0])
                cs, cn = add_row(key_sums, key_counts, bins['cell_ids'], slots[i], loss[i], masks['hit'][i])
                require(cn.sum() == record['hit'][1] and np.isclose(cs.sum(), record['hit'][0], atol=1e-8, rtol=1e-12),
                        'Cell totals disagree with hit targets')
                cell_sums.append(cs); cell_counts.append(cn)
                ids.append(row['segment_id']); doc_ids.append(row['doc_id']); content_ids.append(row['content_hash'])
            if j % 200 == 0:
                print(json.dumps(dict(arm=a.arm, batches=j+1, input_tokens=inputs, seconds=time.monotonic()-start)), flush=True)
            del b, loss
        if not a.max_batches:
            require(next(originals, None) is None and inputs == 20_000_000, 'Incomplete D_dev')
            require(all(totals[p][1] == old['metrics'][p]['count'] for p in POPS), 'Wrong target totals')
    require(abs_error/totals['overall'][1] <= 2e-6, 'Normal replay discrepancy too large')
    cs, cn = np.array(cell_sums), np.array(cell_counts)
    require(key_counts.sum() == cn.sum() == totals['hit'][1], 'Per-key coverage mismatch')
    require(np.isclose(key_sums.sum(), cs.sum(), rtol=1e-12, atol=1e-7), 'Per-key loss mismatch')
    with (out/'statistics.npz').open('xb') as f:
        np.savez_compressed(f, key_loss_sum=key_sums, key_target_count=key_counts,
            segment_cell_loss_sum=cs, segment_cell_target_count=cn,
            segment_id=np.asarray(ids), doc_id=np.asarray(doc_ids), content_hash=np.asarray(content_ids))
    report = dict(arm=a.arm, seed=17, step=3815, role='dev', posthoc=True, smoke=bool(a.max_batches),
        input_tokens=inputs, segments=len(ids), metrics={p: dict(loss_sum=s, count=n, nll=s/n if n else None) for p,(s,n) in totals.items()},
        checkpoint_hash=ck['model_sha256'], corpus_hash=old['corpus_hash'], vocabulary_hash=vocab.hash,
        bin_definition_sha256=file_hash(a.definition/'bins.json'), source_metrics_sha256=file_hash(a.stage2/'eval'/a.arm/'metrics.json'),
        script_sha256=file_hash(__file__), replay_max_segment_population_nll_difference=max_error,
        replay_weighted_absolute_segment_difference=abs_error/totals['overall'][1],
        statistics_sha256=file_hash(out/'statistics.npz'), wall_seconds=time.monotonic()-start,
        cell_target_count=cn.sum(0).tolist(), cell_loss_sum=cs.sum(0).tolist(), cell_nll=means(cs.sum(0),cn.sum(0)))
    write_json(out/'diagnostics.json', report)
    print('A3_EVALUATION_VERIFIED '+a.arm, flush=True)


def load_statistics(root):
    meta = read_json(root/'diagnostics.json')
    require(file_hash(root/'statistics.npz') == meta['statistics_sha256'], 'Result checksum mismatch')
    with np.load(root/'statistics.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, meta


def summarize(a):
    bins, bm = definition(a.definition)
    reports, stats = {}, {}
    for arm in ARMS:
        x, r = load_statistics(a.output/arm)
        require(not r['smoke'] and r['arm'] == arm and r['input_tokens'] == 20_000_000, 'Incomplete/full panel mismatch')
        require(r['bin_definition_sha256'] == file_hash(a.definition/'bins.json'), 'Different bin definitions')
        old = reference(a.stage2/'eval'/arm)
        require(r['source_metrics_sha256'] == file_hash(a.stage2/'eval'/arm/'metrics.json')
                and r['checkpoint_hash'] == old['checkpoint_hash'], 'Wrong source evaluation')
        require(old['diagnostic_table_hash'] == bm['diagnostic_table_hash'], 'Original marginal bins used a different table')
        require(np.isfinite(x['key_loss_sum']).all() and np.isfinite(x['segment_cell_loss_sum']).all(), 'Nonfinite result')
        require(np.array_equal(np.bincount(bins['cell_ids'], weights=x['key_target_count'], minlength=25).astype(np.int64),
                               x['segment_cell_target_count'].sum(0)), 'Key/cell counts disagree')
        require(np.allclose(np.bincount(bins['cell_ids'], weights=x['key_loss_sum'], minlength=25),
                            x['segment_cell_loss_sum'].sum(0), rtol=1e-12, atol=1e-7), 'Key/cell sums disagree')
        require(np.array_equal(x['segment_cell_target_count'].sum(0), r['cell_target_count'])
                and np.allclose(x['segment_cell_loss_sum'].sum(0), r['cell_loss_sum'], rtol=1e-12, atol=1e-7), 'Report/cell mismatch')
        require(x['key_target_count'].sum() == old['metrics']['hit']['count']
                and x['segment_cell_target_count'].shape == (r['segments'], 25), 'Full hit coverage/shape mismatch')
        require(means(np.asarray(r['cell_loss_sum']), np.asarray(r['cell_target_count'])) == r['cell_nll'], 'Cell NLL mismatch')
        for axis, name in ((1, 'frequency'), (0, 'variance')):
            marginal_s = np.asarray(r['cell_loss_sum']).reshape(5, 5).sum(axis=axis)
            marginal_n = np.asarray(r['cell_target_count']).reshape(5, 5).sum(axis=axis)
            original = np.asarray(old[name]).reshape(5, 2, 2).sum(axis=1)
            require(np.array_equal(marginal_n, original[:, 1]) and
                    np.allclose(marginal_s, original[:, 0], rtol=1e-12, atol=1e-7),
                    'Joint grid fails original marginal check: '+name)
        if stats:
            for k in ('segment_id', 'doc_id', 'content_hash', 'segment_cell_target_count', 'key_target_count'):
                require(np.array_equal(x[k], stats['contextual'][k]), 'Unpaired cross-arm statistic: '+k)
        stats[arm], reports[arm] = x, r
    cells = []
    for cell in range(25):
        count = reports['contextual']['cell_target_count'][cell]
        nll = {arm: reports[arm]['cell_nll'][cell] for arm in ARMS}
        cells.append(dict(frequency_bin=cell//5+1, variance_bin=cell%5+1, target_count=count,
            selected_keys=int((bins['cell_ids']==cell).sum()), observed_keys=int(((bins['cell_ids']==cell)&(stats['contextual']['key_target_count']>0)).sum()),
            nll=nll, contextual_minus={arm: nll['contextual']-nll[arm] if count else None for arm in ARMS[1:]}))
    write_json(a.output/'summary.json', dict(posthoc=True, role='dev', seed=17, step=3815,
        bin_definition=bm, cells=cells, arms=reports,
        inference='Descriptive cell point estimates; no per-cell significance claims or regression fitted; per-key/doc records retained'))
    print('A3_ALL_FIVE_ARMS_PAIRED_AND_GRID_VERIFIED', flush=True)


def pack(a):
    """Compact immutable statistics only; never source models/data/live logs."""
    base = a.output
    require((base/'full/summary.json').is_file(), 'Missing verified summary')
    selected = [base/'definition/bins.json', base/'definition/bins.npz', base/'full/summary.json']
    for arm in ARMS:
        selected += [base/'full'/arm/'diagnostics.json', base/'full'/arm/'statistics.npz']
    out = fresh_dir(base/'export')
    entries = []
    with tarfile.open(out/'results.tar.gz', 'w:gz') as tar:
        for p in selected:
            require(p.is_file() and not p.is_symlink(), 'Missing/linked result')
            content = p.read_bytes(); rel = str(p.relative_to(base))
            info = tarfile.TarInfo(rel); info.size = len(content); info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
            entries.append(dict(path=rel, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    parts = []
    with (out/'results.tar.gz').open('rb') as f:
        while content := f.read(20*1024**2):
            p = out/f'results.part{len(parts):03d}'
            with p.open('xb') as dest:
                dest.write(content)
            parts.append(dict(path=p.name, bytes=len(content), sha256=file_hash(p)))
    write_json(out/'manifest.json', dict(success=True, files=entries, parts=parts, archive_sha256=file_hash(out/'results.tar.gz')))
    print(json.dumps(dict(event='A3_RESULT_BUNDLE_READY', files=len(entries), parts=parts)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight', 'evaluate', 'summarize', 'pack'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--definition', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.action == 'evaluate':
        require(a.arm is not None, 'Missing arm')
    globals()[a.action](a)


if __name__ == '__main__':
    main()
