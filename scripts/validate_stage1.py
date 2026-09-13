"""Fail-closed gates for the seed-17 Stage-1 sequence, using local files only."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np
import torch
from ccm.artifacts import load_table, state_hash
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require, read_json, write_json, file_hash, schedule, seed_bundle
from ccm.data import Corpus
from ccm.keys import Vocabulary
from ccm.model import Reader
from ccm.runtime import checkpoint_meta, require_coverage
from pilot_stage1 import COMMON, DATA, PREP, ASSETS, CORE_SHA, ARMS


def stage1_log(path):
    count = 0
    with Path(path).open() as f:
        for count, line in enumerate(f, 1):
            r = json.loads(line)
            require(r['step'] == count and r['input_tokens'] == count*PILOT.batch_tokens, 'Wrong Stage-1 counter')
            require(all(math.isfinite(r[k]) for k in ('nll', 'lr', 'grad_norm')), 'Nonfinite training log')
            require(math.isclose(r['lr'], schedule(count, PILOT.adapt_steps, 5e-4, .05), rel_tol=1e-10),
                    'Wrong Stage-1 LR schedule')
    require(count == PILOT.adapt_steps, 'Incomplete Stage-1 log')


def fresh_reader_hash():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_bundle(17)['reader'])
        return state_hash(Reader(1024).bfloat16().state_dict())


def validate_train(root, arm, corpus, vocab, common):
    path = root/'train'/arm
    done = read_json(path/'complete.json')
    require(done == dict(success=True, phase='stage1', arm=arm, step=977,
                        input_tokens=256114688, checkpoint='checkpoint-977'), 'Wrong arm completion')
    ckpath = path/'checkpoint-977'
    ck = checkpoint_meta(ckpath)
    require(ck['phase'] == 'stage1' and ck['arm'] == arm and ck['seed'] == 17 and
            ck['seeds'] == seed_bundle(17) and ck['step'] == ck['total_steps'] == 977 and
            ck['world_size'] == 8 and not ck['engineering'], 'Wrong Stage-1 checkpoint contract')
    require(ck['source_checkpoint_hash'] == common['model_sha256'] and
            ck['corpus_hash'] == corpus.meta['manifest_hash'] and ck['vocabulary_hash'] == vocab.hash and
            ck['tokenizer'] == common['tokenizer'] and ck['backbone_contract'] == common['backbone_contract'] and
            ck['model_dtype'] == 'torch.bfloat16' and ck['config']['source_code_hash'] == CORE_SHA,
            'Stage-1 provenance/precision mismatch')
    require(ck['paired_initial_reader_hash'] == fresh_reader_hash(), 'Unpaired reader initialization')
    require(ck['config']['microbatch_segments'] == 8 and ck['config']['loss_chunk'] == 1024 and
            ck['config']['activation_checkpointing'] and not ck['config']['online'], 'Wrong runtime settings')
    stage1_log(path/'train.jsonl')
    state = torch.load(ckpath/'model.pt', map_location='cpu', weights_only=True)
    base = torch.load(COMMON/'common/checkpoint-15259/model.pt', map_location='cpu', weights_only=True)
    require({k for k in state if k.startswith('backbone.')} == set(base), 'Wrong backbone state keys')
    require(all(torch.equal(state[k], v) for k, v in base.items()), 'Frozen backbone was modified')
    del base
    require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite model state')
    require(state_hash({k[7:]: v for k, v in state.items() if k.startswith('reader.')}) == ck['reader_hash'],
            'Reader hash mismatch')
    require(ck['reader_hash'] != ck['paired_initial_reader_hash'] and bool(state['reader.wv.weight'].any()),
            'Reader did not learn')
    require(state_hash({'table': state['table']}) == ck['table_hash'], 'Saved table hash mismatch')
    if arm != 'grad':
        table, meta = load_table(COMMON/'tables'/arm, dict(constructor=arm, vocabulary_hash=vocab.hash,
                    corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=common['model_sha256']))
        require(ck['table_artifact_hash'] == meta['artifact_hash'] and torch.equal(state['table'], table['lookup']),
                'Frozen table was modified or wrong table loaded')
        del table
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_bundle(17)['grad_table'])
            initial = torch.empty(PILOT.slots, 1024).normal_(0, .02).bfloat16()
        require(ck['initial_grad_table_hash'] == state_hash({'table': initial}), 'Wrong Grad initialization')
        require(not torch.equal(state['table'], initial), 'Grad table did not learn')
        del initial
    del state
    require(file_hash(ckpath/'optimizer.pt') == ck['optimizer_sha256'], 'Optimizer checksum mismatch')
    opt = torch.load(ckpath/'optimizer.pt', map_location='cpu', weights_only=True)
    reader_count = sum(p.numel() for p in Reader(1024).parameters())
    expected = reader_count+(PILOT.slots*1024 if arm == 'grad' else 0)
    require(sum(v.numel() for v in opt['masters']) == expected, 'Optimizer contains wrong trainable parameters')
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in opt['masters']), 'Invalid fp32 masters')
    groups = opt['optimizer']['param_groups']
    require({(g['multiplier'], g['weight_decay']) for g in groups} ==
            ({(1., .01), (1., 0.), (5., 0.)} if arm == 'grad' else {(1., .01), (1., 0.)}), 'Wrong optimizer groups')
    for g in groups:
        require(math.isclose(g['lr'], 5e-5*g['multiplier'], rel_tol=1e-10), 'Wrong final optimizer LR')
    for v in opt['optimizer']['state'].values():
        require(int(v['step']) == 977, 'Wrong optimizer step/reset policy')
        for name in ('exp_avg', 'exp_avg_sq'):
            require(v[name].dtype == torch.float32 and bool(torch.isfinite(v[name]).all()), 'Invalid Adam state')
    return dict(checkpoint_hash=ck['model_sha256'], paired_initial_reader_hash=ck['paired_initial_reader_hash'],
                frozen_backbone_exact=True, frozen_table_exact=arm != 'grad', step=977, input_tokens=256114688)


def validate_eval(root, arm, corpus, vocab, common, checkpoint=None):
    path = root/'eval'/arm
    m = read_json(path/'metrics.json')
    ck = checkpoint if checkpoint is not None else (common if arm == 'base' else checkpoint_meta(root/'train'/arm/'checkpoint-977'))
    require(m['arm'] == arm and m['role'] == 'dev' and not m['final_evaluation'] and not m['engineering'] and
            m['seed'] == 17 and m['phase'] == ck['phase'] and m['step'] == ck['step'] and
            m['total_steps'] == ck['total_steps'] and m['checkpoint_hash'] == ck['model_sha256'], 'Wrong evaluation contract')
    require(m['corpus_hash'] == corpus.meta['manifest_hash'] and m['vocabulary_hash'] == vocab.hash and
            m['input_tokens'] == PILOT.dev_tokens and m['diagnostic_table_hash'] ==
            read_json(COMMON/'tables/contextual/artifact.json')['artifact_hash'], 'Wrong dev data/diagnostic bins')
    require(file_hash(path/'segments.jsonl') == m['segments_sha256'], 'Evaluation record checksum mismatch')
    totals = {k: [0., 0] for k in ('overall', 'hit', 'miss', 'eligible_miss')}
    binned = {k: np.zeros((10, 2)) for k in ('frequency', 'variance')}
    with (path/'segments.jsonl').open() as f:
        for line, row in zip_longest(f, corpus.segments('dev')):
            require(line is not None and row is not None, 'Missing/extra dev segment')
            r = json.loads(line)
            require(all(r[k] == row[k] for k in ('segment_id', 'doc_id', 'content_hash')), 'Dev alignment mismatch')
            require(r['overall'][1] == len(row['tokens'])-1 and
                    r['hit'][1]+r['miss'][1] == r['overall'][1], 'Wrong target counts')
            for k in totals:
                s, n = r[k]
                require(math.isfinite(s) and s >= 0 and type(n) is int and n >= 0, 'Invalid per-segment metric')
                totals[k][0] += s
                totals[k][1] += n
            for k in binned:
                values = np.asarray(r[k])
                require(values.shape == (10, 2) and bool(np.isfinite(values).all()), 'Invalid diagnostic bins')
                binned[k] += values
    for k, (s, n) in totals.items():
        v = m['metrics'][k]
        require(v['count'] == n and n > 0 and math.isclose(v['loss_sum'], s, rel_tol=1e-10) and
                math.isclose(v['nll'], s/n, rel_tol=1e-10), 'Evaluation summary mismatch')
    for k, v in binned.items():
        require(np.allclose(v, m[k], rtol=1e-10, atol=1e-7) and v[:, 1].sum() == totals['hit'][1] and
                math.isclose(v[:, 0].sum(), totals['hit'][0], rel_tol=1e-10), 'Binned summary mismatch')
    if arm != 'base':
        require(m['gate'] and all(math.isfinite(v) and 0 <= v <= 1 for v in m['gate'].values()), 'Invalid gate stats')
    return dict(metrics_sha256=file_hash(path/'metrics.json'), segments_sha256=m['segments_sha256'],
                checkpoint_hash=m['checkpoint_hash'], nll=m['metrics']['overall']['nll'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('inputs', 'train', 'eval', 'panel'))
    p.add_argument('--workflow', type=Path, required=True)
    p.add_argument('--arm', choices=('base',)+ARMS)
    a = p.parse_args()
    torch.set_num_threads(8)
    require(code_hash() == CORE_SHA, 'Core changed from completed production workflow')
    corpus = Corpus(DATA/'corpus')
    vocab = Vocabulary.load(DATA/'vocabulary.npz')
    common = checkpoint_meta(COMMON/'common/checkpoint-15259')
    require(not corpus.meta['engineering'] and corpus.budget == PILOT and len(vocab.keys) == PILOT.slots, 'Wrong pilot corpus')
    require(common['model_sha256'] == '844c0b0320e88c446f19d2ebd028857ba5b8a4a4e61d2ced740c017eaf8249cf', 'Wrong common checkpoint')
    require(common['phase'] == 'common' and common['arm'] == 'base' and common['step'] == 15259 and
            common['corpus_hash'] == corpus.meta['manifest_hash'], 'Common contract mismatch')
    require_coverage(PREP/'coverage.json', corpus, vocab)
    report = dict(success=True, stage=a.stage, arm=a.arm, corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash)
    if a.stage == 'inputs':
        require(asset_identity(ASSETS, 'resources/qwen3_base_assets.json') == common['tokenizer'], 'Wrong Base assets')
        require(read_json(COMMON/'complete.json').get('success') is True, 'Previous workflow incomplete')
        prior = read_json(COMMON/'validated_tables.json')
        require(prior['success'] and prior['checkpoint_hash'] == common['model_sha256'] and
                prior['corpus_hash'] == corpus.meta['manifest_hash'] and prior['vocabulary_hash'] == vocab.hash, 'Wrong prior gate')
        for arm in ARMS[:-1]:
            t, meta = load_table(COMMON/'tables'/arm, dict(constructor=arm, vocabulary_hash=vocab.hash,
                    corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=common['model_sha256'],
                    tokenizer=common['tokenizer'], backbone_contract=common['backbone_contract']))
            require(t['lookup'].shape == (262144, 1024) and meta['artifact_hash'] == prior['table_artifact_hashes'][arm], 'Wrong table')
            del t
    elif a.stage == 'train':
        require(a.arm in ARMS, 'Choose a trained arm')
        report.update(validate_train(a.workflow, a.arm, corpus, vocab, common))
    elif a.stage == 'eval':
        require(a.arm is not None, 'Choose an evaluated arm')
        report.update(validate_eval(a.workflow, a.arm, corpus, vocab, common))
    else:
        for arm in ARMS:
            require(read_json(a.workflow/f'validated_train_{arm}.json')['success'], 'Missing trained arm gate')
        for arm in ('base',)+ARMS:
            gate = read_json(a.workflow/f'validated_eval_{arm}.json')
            require(gate['success'] and file_hash(a.workflow/'eval'/arm/'metrics.json') == gate['metrics_sha256'], 'Missing/changed eval')
        for right in ('isolated', 'shuffled'):
            r = read_json(a.workflow/'reports'/f'contextual_vs_{right}.json')
            require(r['replicates'] == 10000 and r['cluster'] == 'doc_id' and r['cross_seed_coupling'] == 'independent' and
                    r['backbone_seeds'] == [17] and not r['final_report'] and
                    all(math.isfinite(r[k]) for k in ('mean_difference', 'lower95', 'upper95')), 'Invalid primary report')
    dest = a.workflow/('validated_'+a.stage+('_'+a.arm if a.arm else '')+'.json')
    write_json(dest, report)
    print('STAGE1_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
