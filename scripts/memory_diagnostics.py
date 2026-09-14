"""Post-hoc seed-17 Stage-2 A1/A2 diagnostics, D_dev only; no training changes.

The external reader hook leaves slots/targets unchanged. Disabled mode zeros
the actual contribution; it is not a timing benchmark (the reader still runs).
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import torch

from ccm.compiler import batches
from ccm.contracts import require, read_json, write_json, fresh_dir, file_hash
from ccm.data import Corpus, collate
from ccm.keys import Vocabulary
from ccm.runtime import load_model, model_inputs, to_device, checkpoint_meta
from ccm.statistics import bootstrap, records

ARMS = ('contextual', 'isolated', 'grad')
POPS = ('overall', 'hit', 'miss', 'eligible_miss')
EPS = 1e-12


class ContributionProbe:
    def __init__(self, reader):
        self.off = False
        self.mask = None
        self.values = None
        self.handle = reader.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        contribution, gate = output
        if self.off:
            return torch.zeros_like(contribution), gate
        require(self.mask is not None, 'Probe requires original hit/target mask')
        # Convert only hit positions, not full padded hidden-state arrays.
        r = torch.linalg.vector_norm(args[0][self.mask].float(), dim=-1)
        c = torch.linalg.vector_norm(contribution[self.mask].float(), dim=-1)
        self.values = torch.stack((c, r, c/r.clamp_min(EPS)), dim=-1).detach().cpu().numpy()
        # Returning None leaves the normal reader output exactly unchanged.

    def close(self):
        self.handle.remove()


def distribution(values):
    require(len(values) > 0 and np.isfinite(values).all(), 'Missing/nonfinite norm samples')
    return dict(count=len(values), mean=float(values.mean(dtype=np.float64)),
                p10=float(np.quantile(values, .1)), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)))


def masks_for(batch):
    target = batch['targets'] != -100
    hit = (batch['slots'] >= 0) & target
    return dict(overall=target, hit=hit, miss=target & ~hit,
                eligible_miss=batch['eligible'] & target & ~hit)


def reference(path):
    meta = read_json(path/'metrics.json')
    require(meta['role'] == 'dev' and meta['phase'] == 'stage2' and meta['seed'] == 17
            and meta['step'] == meta['total_steps'] == 3815 and not meta['engineering']
            and not meta['final_evaluation'], 'Requires completed seed-17 Stage-2 D_dev')
    require(file_hash(path/'segments.jsonl') == meta['segments_sha256'], 'Reference records changed')
    return meta


def checked_inputs(data, stage2, arm):
    # Verify exactly the files read by these diagnostics, not every 5B-token split.
    corpus = Corpus(data/'corpus', verify=False)
    for name in ('dev.tokens', 'dev.segments.jsonl'):
        require(file_hash(corpus.path/name) == corpus.meta['files'][name], 'D_dev checksum mismatch')
    vocab = Vocabulary.load(data/'vocabulary.npz')
    normal, base = reference(stage2/'eval'/arm), reference(stage2/'eval/base')
    require(normal['arm'] == arm and base['arm'] == 'base', 'Wrong reference arm')
    for key in ('corpus_hash', 'vocabulary_hash', 'source_checkpoint_hash', 'input_tokens'):
        require(normal[key] == base[key], 'Unpaired Base reference: '+key)
    require(normal['corpus_hash'] == corpus.meta['manifest_hash'] == vocab.metadata['corpus_hash']
            and normal['vocabulary_hash'] == vocab.hash, 'Data/vocabulary mismatch')
    require(normal['input_tokens'] == corpus.meta['quotas']['dev'] == 20_000_000, 'Wrong full D_dev quota')
    return corpus, vocab, normal


def check_pair(record, old, normal_tolerance=None):
    for key in ('doc_id', 'content_hash', 'segment_id'):
        require(record[key] == old[key], 'Segment order/identity changed: '+key)
    maximum = 0.
    for pop in POPS:
        s, n = record[pop]
        require(np.isfinite(s) and n == old[pop][1], 'Nonfinite loss or changed target labels')
        if normal_tolerance is not None and n:
            delta = abs(s-old[pop][0])/n
            maximum = max(maximum, delta)
            require(delta <= normal_tolerance, 'Normal replay differs from original evaluation')
    return maximum


@torch.no_grad()
def evaluate(a):
    require(a.max_batches >= 0, 'Invalid smoke length')
    corpus, vocab, old = checked_inputs(a.data, a.stage2, a.arm)
    checkpoint = a.stage2/'train'/a.arm/'checkpoint-3815'
    model, meta = load_model(checkpoint, a.device)
    require(meta['model_sha256'] == old['checkpoint_hash'] and meta['arm'] == a.arm
            and meta['metadata_hash'] == read_json(checkpoint/'checkpoint.json')['metadata_hash'],
            'Checkpoint differs from validated evaluation')
    model.set_phase('eval')
    out = fresh_dir(a.output)
    normal_dir, off_dir = fresh_dir(out/'normal'), fresh_dir(out/'memory_off')
    totals = [{k: [0., 0] for k in POPS} for _ in range(2)]
    norms = []
    inputs = segments = 0
    max_difference = abs_difference_sum = 0.
    start = time.monotonic()
    probe = ContributionProbe(model.reader)
    try:
        with ExitStack() as stack:
            originals = stack.enter_context((a.stage2/'eval'/a.arm/'segments.jsonl').open())
            base = stack.enter_context((a.stage2/'eval/base/segments.jsonl').open())
            files = [stack.enter_context((d/'segments.jsonl').open('x')) for d in (normal_dir, off_dir)]
            for batch_index, rows in enumerate(batches(corpus.segments('dev'), 8)):
                if a.max_batches and batch_index >= a.max_batches:
                    break
                b = to_device(collate(rows, corpus.meta['special_ids'], vocab), a.device)
                masks = masks_for(b)
                probe.mask = masks['hit']
                probe.off = False
                normal = model(**model_inputs(b), loss_chunk=1024)['losses'].double().cpu().numpy()
                require(probe.values is not None and np.isfinite(probe.values).all(), 'Invalid probe output')
                norms.append(probe.values)
                probe.values = None
                probe.off = True
                off = model(**model_inputs(b), loss_chunk=1024)['losses'].double().cpu().numpy()
                masks = {k: v.cpu().numpy() for k, v in masks.items()}
                inputs += int(b['attention_mask'].sum())
                for i, row in enumerate(rows):
                    source = json.loads(next(originals))
                    source_base = json.loads(next(base))
                    for mode, loss in enumerate((normal, off)):
                        record = {k: row[k] for k in ('doc_id', 'content_hash', 'segment_id')}
                        for pop, mask in masks.items():
                            s, n = float(loss[i][mask[i]].sum()), int(mask[i].sum())
                            record[pop] = [s, n]
                            totals[mode][pop][0] += s
                            totals[mode][pop][1] += n
                        diff = check_pair(record, source, 2e-5 if mode == 0 else None)
                        check_pair(record, source_base)
                        if mode == 0:
                            max_difference = max(max_difference, diff)
                            abs_difference_sum += abs(record['overall'][0]-source['overall'][0])
                        files[mode].write(json.dumps(record, allow_nan=False)+'\n')
                    segments += 1
                if batch_index % 100 == 0:
                    print(json.dumps(dict(arm=a.arm, batches=batch_index+1, segments=segments,
                                          input_tokens=inputs, seconds=time.monotonic()-start)), flush=True)
                del b, normal, off
            if not a.max_batches:
                require(next(originals, None) is None and next(base, None) is None, 'Reference has extra segments')
    finally:
        probe.close()
    average_difference = abs_difference_sum/totals[0]['overall'][1]
    require(average_difference <= 2e-6, 'Normal replay mean absolute discrepancy too large')
    if not a.max_batches:
        require(inputs == old['input_tokens'], 'Incomplete D_dev')
        for pop in POPS:
            require(totals[0][pop][1] == totals[1][pop][1] == old['metrics'][pop]['count'], 'Wrong full counts')
    values = np.concatenate(norms)
    require(len(values) == totals[0]['hit'][1], 'Norm coverage must include every original hit target')
    report = dict(arm=a.arm, posthoc=True, role='dev', seed=17, step=3815,
                  checkpoint_hash=meta['model_sha256'], corpus_hash=vocab.metadata['corpus_hash'],
                  vocabulary_hash=vocab.hash, script_sha256=file_hash(__file__),
                  smoke=bool(a.max_batches), input_tokens=inputs, segments=segments,
                  microbatch_segments=8, loss_chunk=1024, wall_seconds=time.monotonic()-start,
                  original_metrics_sha256=file_hash(a.stage2/'eval'/a.arm/'metrics.json'),
                  normal_replay_max_segment_population_nll_difference=max_difference,
                  normal_replay_token_weighted_absolute_segment_difference=average_difference,
                  denominator_epsilon=EPS, denominator_below_epsilon=int((values[:, 1] < EPS).sum()),
                  contribution_l2=distribution(values[:, 0]), pre_memory_residual_l2=distribution(values[:, 1]),
                  contribution_to_residual_ratio=distribution(values[:, 2]),
                  interpretation='Inference-time dependency only, not causal attribution of learning to the backbone')
    for mode, directory in enumerate((normal_dir, off_dir)):
        metrics = dict(old, metrics={k: dict(loss_sum=s, count=n, nll=s/n if n else None)
                                    for k, (s, n) in totals[mode].items()},
                       segments_sha256=file_hash(directory/'segments.jsonl'), posthoc=True,
                       memory_disabled=bool(mode), smoke=bool(a.max_batches), input_tokens=inputs)
        # Do not copy stale normal-evaluation bins, gates or performance to a new mode.
        for k in ('frequency', 'variance', 'gate', 'wall_seconds', 'tokens_per_second', 'peak_accelerator_bytes',
                  'eligible_hit_rate', 'overall_memory_active_rate'):
            metrics.pop(k, None)
        write_json(directory/'metrics.json', metrics)
    report['files'] = {str(p.relative_to(out)): file_hash(p) for p in
                       (normal_dir/'metrics.json', normal_dir/'segments.jsonl', off_dir/'metrics.json', off_dir/'segments.jsonl')}
    write_json(out/'diagnostics.json', report)
    print(json.dumps(dict(event='DIAGNOSTIC_EVALUATION_VERIFIED', arm=a.arm, smoke=report['smoke'])), flush=True)


def summarize(a):
    result = dict(posthoc=True, role='dev', seed=17, step=3815, cluster='doc_id',
                  bootstrap_replicates=10000, bootstrap_seed=20260913, arms={})
    for arm in ARMS:
        root = a.output/arm
        diag = read_json(root/'diagnostics.json')
        require(not diag['smoke'] and diag['arm'] == arm and diag['input_tokens'] == 20_000_000,
                'Partial/wrong diagnostic cannot enter full report')
        for rel, sha in diag['files'].items():
            require(file_hash(root/rel) == sha, 'Diagnostic file changed')
        ref = reference(a.stage2/'eval'/arm)
        require(diag['checkpoint_hash'] == ref['checkpoint_hash'], 'Wrong source checkpoint')
        reports = {}
        for pop in POPS:
            paths = [root/'memory_off', root/'normal', a.stage2/'eval/base']
            maps = [records(p, pop, 'doc_id') for p in paths]
            ids = sorted(maps[0])
            require(all(sorted(m) == ids for m in maps), 'Unpaired documents')
            arrays = [np.array([m[i] for i in ids]) for m in maps]
            require(all(np.array_equal(arrays[0][:, 1], x[:, 1]) for x in arrays), 'Unpaired target counts')
            reports[pop] = dict(normal_nll=float(arrays[1][:, 0].sum()/arrays[1][:, 1].sum()),
                memory_off_nll=float(arrays[0][:, 0].sum()/arrays[0][:, 1].sum()),
                base_nll=float(arrays[2][:, 0].sum()/arrays[2][:, 1].sum()), count=int(arrays[0][:, 1].sum()),
                off_minus_normal=bootstrap([(ids, arrays[0], arrays[1])]),
                off_minus_base=bootstrap([(ids, arrays[0], arrays[2])]))
        result['arms'][arm] = dict(contribution=diag, losses=reports)
        print('BOOTSTRAP_COMPLETE '+arm, flush=True)
    write_json(a.output/'summary.json', result)


def preflight(a):
    from ccm.cli import code_hash
    require(code_hash() == '055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9',
            'Frozen research core changed')
    for arm in ARMS:
        _, _, old = checked_inputs(a.data, a.stage2, arm)
        ck = checkpoint_meta(a.stage2/'train'/arm/'checkpoint-3815')
        require(ck['model_sha256'] == old['checkpoint_hash'] and ck['arm'] == arm
                and ck['phase'] == 'stage2' and ck['seed'] == 17 and ck['step'] == 3815,
                'Invalid source checkpoint')
    print('DIAGNOSTIC_INPUTS_AND_FROZEN_CORE_VERIFIED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('evaluate', 'summarize', 'preflight'))
    p.add_argument('--data', type=Path)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-batches', type=int, default=0, help='Positive = engineering prefix only, never a full result')
    a = p.parse_args()
    if a.action == 'evaluate':
        require(a.data is not None and a.arm is not None, 'Evaluation needs data and arm')
        evaluate(a)
    elif a.action == 'summarize':
        summarize(a)
    else:
        require(a.data is not None, 'Preflight needs data')
        preflight(a)


if __name__ == '__main__':
    main()
