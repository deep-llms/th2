"""Offline artifact gates for the authorized full-budget common/compile handoff."""
import argparse
import json
import math
from pathlib import Path
import torch
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require, read_json, write_json, file_hash, schedule
from ccm.data import Corpus
from ccm.keys import Vocabulary
from ccm.runtime import checkpoint_meta, require_coverage
from ccm.artifacts import load_table, tensor_hash


def validate_log(path, steps, batch_tokens):
    count = 0
    with Path(path).open() as f:
        for count, line in enumerate(f, 1):
            r = json.loads(line)
            require(r['step'] == count and r['input_tokens'] == count*batch_tokens, 'Wrong training counter')
            require(all(math.isfinite(r[k]) for k in ['nll', 'lr', 'grad_norm']), 'Nonfinite training log')
            require(math.isclose(r['lr'], schedule(count, steps, 3e-4, .02), rel_tol=1e-10), 'Wrong common schedule')
    require(count == steps, 'Training log incomplete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['data', 'common', 'tables'])
    for name in ['data', 'prep-reports', 'workflow', 'assets']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    data, out = a.data, a.workflow
    corpus = Corpus(data/'corpus')
    require(not corpus.meta['engineering'] and corpus.budget == PILOT, 'Not the locked scientific corpus')
    assets = asset_identity(a.assets, 'resources/qwen3_base_assets.json')
    require(corpus.meta['provenance']['tokenizer'] == assets, 'Wrong pinned Base assets')
    require(corpus.meta['provenance']['source_code_hash'] == code_hash(), 'Core source changed during workflow')
    vocab = Vocabulary.load(data/'vocabulary.npz')
    require(len(vocab.keys) == PILOT.slots, 'Wrong vocabulary capacity')
    require_coverage(a.prep_reports/'coverage.json', corpus, vocab)
    prep = read_json(a.prep_reports/'complete.json')
    require(prep['success'] and prep['corpus_hash'] == corpus.meta['manifest_hash'] and
            prep['vocabulary_hash'] == vocab.hash and prep['quotas'] == PILOT.quotas(), 'Wrong preparation completion')
    require(prep['optimizer_batches'] == dict(common=15259, stage1=977, stage2=3815), 'Wrong immutable batch counts')
    for path, sha in prep['files'].items():
        path = Path(path)
        require(path.resolve().is_relative_to(data.resolve()) or path.resolve().is_relative_to(a.prep_reports.resolve()),
                'Unexpected preparation artifact path')
        require(file_hash(path) == sha, 'Preparation artifact checksum mismatch')
    report = dict(success=True, stage=a.stage, corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash)
    if a.stage != 'data':
        ckpath = out/'common'/f'checkpoint-{PILOT.common_steps}'
        ck = checkpoint_meta(ckpath)
        done = read_json(out/'common/complete.json')
        require(done == dict(success=True, phase='common', arm='base', step=15259,
                            input_tokens=4000055296, checkpoint='checkpoint-15259'), 'Wrong training completion')
        require(ck['phase'] == 'common' and ck['arm'] == 'base' and ck['seed'] == 17 and
                not ck['engineering'] and ck['step'] == ck['total_steps'] == 15259 and ck['world_size'] == 8,
                'Wrong common run contract')
        require(ck['corpus_hash'] == corpus.meta['manifest_hash'] and ck['tokenizer'] == assets and
                ck['config']['source_code_hash'] == code_hash() and ck['model_dtype'] == 'torch.bfloat16',
                'Common provenance/precision mismatch')
        require(file_hash(ckpath/'optimizer.pt') == ck['optimizer_sha256'], 'Optimizer snapshot checksum mismatch')
        config = read_json(ckpath/'config.json')
        require(config['num_hidden_layers'] == 12 and config['hidden_size'] == 1024 and
                config['tie_word_embeddings'], 'Wrong model configuration')
        state = torch.load(ckpath/'model.pt', map_location='cpu', weights_only=True)
        require(all(bool(torch.isfinite(t).all()) for t in state.values()), 'Nonfinite checkpoint weights')
        del state
        validate_log(out/'common/train.jsonl', 15259, 262144)
        report['checkpoint_hash'] = ck['model_sha256']
        if a.stage == 'tables':
            done = read_json(out/'tables/complete.json')
            require(done['success'] and done['tokens'] == PILOT.compile_tokens, 'Incomplete compilation')
            hashes = {}
            for arm in ['shallow', 'contextual', 'delta', 'isolated', 'shuffled']:
                t, meta = load_table(out/'tables'/arm, dict(constructor=arm,
                    vocabulary_hash=vocab.hash, corpus_hash=corpus.meta['manifest_hash'],
                    source_checkpoint_hash=ck['model_sha256'], tokenizer=assets,
                    backbone_contract=ck['backbone_contract']))
                require(t['lookup'].shape == (262144, 1024), 'Wrong table shape')
                if 'counts' in t:
                    require(torch.equal(t['counts'], torch.from_numpy(vocab.counts)), 'Compiler count mismatch')
                if arm == 'contextual':
                    context = t['lookup'].clone()
                if arm == 'shuffled':
                    require(torch.equal(t['lookup'], context[t['permutation']]), 'Shuffle does not match contextual table')
                hashes[arm] = meta['artifact_hash']
                del t
            report['table_artifact_hashes'] = hashes
    write_json(out/f'validated_{a.stage}.json', report)
    print('PILOT_ARTIFACT_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
