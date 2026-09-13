"""Local-only acceptance gates for the five seed-17 matched continuation arms."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch
from transformers import Qwen3Config
from ccm.artifacts import load_table, state_hash
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require, read_json, write_json, file_hash, schedule, seed_bundle
from ccm.data import Corpus
from ccm.keys import Vocabulary
from ccm.model import MemoryLM
from ccm.runtime import checkpoint_meta, optimizer_groups, require_coverage
from pilot_stage2 import COMMON, PREVIOUS, DATA, PREP, ASSETS, CORE_SHA, ARMS
from validate_stage1 import fresh_reader_hash, validate_eval


def stage2_log(path):
    count = 0
    with Path(path).open() as f:
        for count, line in enumerate(f, 1):
            r = json.loads(line)
            require(r['step'] == count and r['input_tokens'] == count*PILOT.batch_tokens, 'Wrong Stage-2 counter')
            require(all(math.isfinite(r[k]) for k in ('nll', 'lr', 'grad_norm')), 'Nonfinite training log')
            require(math.isclose(r['lr'], schedule(count, PILOT.continue_steps, 1.5e-4, .02), rel_tol=1e-10),
                    'Wrong Stage-2 LR schedule')
    require(count == PILOT.continue_steps, 'Incomplete Stage-2 log')


def optimizer_layout(config, arm):
    # Meta-device shapes/names only: no full additional model allocation.
    with torch.device('meta'):
        fixture = MemoryLM(config, arm, slots=PILOT.slots,
                           table=None if arm in ('base', 'grad') else torch.empty(PILOT.slots, config.hidden_size))
    fixture.set_phase('stage2')
    names = {id(p): name for name, p in fixture.named_parameters()}
    groups = optimizer_groups(fixture, 'stage2')
    return [(g['multiplier'], g['weight_decay'], [names[id(p)] for p in g['params']]) for g in groups]


def validate_optimizer(opt, layout, state):
    groups = opt['optimizer']['param_groups']
    require(len(groups) == len(layout), 'Wrong optimizer group count')
    ordered_names = []
    for g, (mult, decay, names) in zip(groups, layout):
        require(g['params'] == list(range(len(ordered_names), len(ordered_names)+len(names))),
                'Optimizer parameter order mismatch')
        require(g['multiplier'] == mult and g['weight_decay'] == decay and
                tuple(g['betas']) == (.9, .95) and g['eps'] == 1e-8 and
                math.isclose(g['lr'], 1.5e-5*mult, rel_tol=1e-10), 'Wrong Stage-2 optimizer settings')
        ordered_names.extend(names)
    require(len(opt['masters']) == len(ordered_names) and
            set(opt['optimizer']['state']) == set(range(len(ordered_names))), 'Missing/extra optimizer state')
    for i, name in enumerate(ordered_names):
        m, v = opt['masters'][i], opt['optimizer']['state'][i]
        require(m.shape == state[name].shape and m.dtype == torch.float32 and bool(torch.isfinite(m).all()) and
                torch.equal(m.to(state[name].dtype), state[name]), 'Invalid/misaligned fp32 master')
        require(int(v['step']) == PILOT.continue_steps, 'Wrong optimizer step/reset policy')
        for k in ('exp_avg', 'exp_avg_sq'):
            require(v[k].shape == m.shape and v[k].dtype == torch.float32 and bool(torch.isfinite(v[k]).all()),
                    'Invalid Adam moment')
    return sum(v.numel() for v in opt['masters'])


def validate_train(root, arm, corpus, vocab, common):
    path = root/'train'/arm
    require(read_json(path/'complete.json') == dict(success=True, phase='stage2', arm=arm, step=3815,
            input_tokens=1000079360, checkpoint='checkpoint-3815'), 'Wrong Stage-2 completion')
    ckpath = path/'checkpoint-3815'
    ck = checkpoint_meta(ckpath)
    require(ck['phase'] == 'stage2' and ck['arm'] == arm and ck['seed'] == 17 and
            ck['seeds'] == seed_bundle(17) and ck['step'] == ck['total_steps'] == 3815 and
            ck['world_size'] == 8 and not ck['engineering'], 'Wrong Stage-2 contract')
    require(ck['source_checkpoint_hash'] == common['model_sha256'] and
            ck['corpus_hash'] == corpus.meta['manifest_hash'] and ck['vocabulary_hash'] == vocab.hash and
            ck['tokenizer'] == common['tokenizer'] and ck['backbone_contract'] == common['backbone_contract'] and
            ck['model_dtype'] == 'torch.bfloat16' and ck['config']['source_code_hash'] == CORE_SHA and
            ck['peak_lr'] == 1.5e-4, 'Wrong provenance/precision/schedule')
    cfg = ck['config']
    require(cfg['checkpoint'] == str(COMMON/'common/checkpoint-15259') and
            cfg['data'] == str(DATA/'corpus') and cfg['vocabulary'] == str(DATA/'vocabulary.npz') and
            cfg['microbatch_segments'] == 8 and cfg['loss_chunk'] == 1024 and cfg['activation_checkpointing'] and
            not cfg['online'] and cfg['delta_decision'] is None, 'Wrong launch configuration')
    stage2_log(path/'train.jsonl')
    state = torch.load(ckpath/'model.pt', map_location='cpu', weights_only=True)
    base = torch.load(COMMON/'common/checkpoint-15259/model.pt', map_location='cpu', weights_only=True)
    require({k for k in state if k.startswith('backbone.')} == set(base), 'Wrong backbone keys')
    changed = [k for k, v in base.items() if not torch.equal(state[k], v)]
    require(changed and all(any(k.startswith(f'backbone.model.layers.{i}.') for k in changed) for i in range(12)),
            'Backbone did not train in every block')
    del base
    require(all(v.dtype == torch.bfloat16 and bool(torch.isfinite(v).all()) for v in state.values()), 'Invalid model tensors')
    require(torch.equal(state['backbone.model.embed_tokens.weight'], state['backbone.lm_head.weight']), 'Tying was broken')
    if arm == 'base':
        require(set(state) == {k for k in state if k.startswith('backbone.')} and
                all(ck[k] is None for k in ('reader_hash', 'table_hash', 'paired_initial_reader_hash',
                                          'table_artifact_hash', 'initial_grad_table_hash')), 'Base has memory state')
    else:
        prior = checkpoint_meta(PREVIOUS/'train'/arm/'checkpoint-977')
        require(ck['paired_initial_reader_hash'] == fresh_reader_hash() and
                ck['paired_initial_reader_hash'] != prior['reader_hash'], 'Reader not freshly paired')
        require(state_hash({k[7:]: v for k, v in state.items() if k.startswith('reader.')}) == ck['reader_hash'] and
                ck['reader_hash'] != ck['paired_initial_reader_hash'] and bool(state['reader.wv.weight'].any()), 'Reader did not learn')
        require(state_hash({'table': state['table']}) == ck['table_hash'], 'Wrong table hash')
        if arm == 'grad':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_bundle(17)['grad_table'])
                initial = torch.empty(PILOT.slots, 1024).normal_(0, .02).bfloat16()
            require(ck['initial_grad_table_hash'] == state_hash({'table': initial}) and
                    ck['initial_grad_table_hash'] != prior['table_hash'] and
                    not torch.equal(state['table'], initial), 'Grad table not fresh/trainable')
            del initial
        else:
            table, meta = load_table(COMMON/'tables'/arm, dict(constructor=arm, vocabulary_hash=vocab.hash,
                        corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=common['model_sha256']))
            require(ck['table_artifact_hash'] == meta['artifact_hash'] and torch.equal(state['table'], table['lookup']),
                    'Frozen table was modified')
            del table
    require(file_hash(ckpath/'optimizer.pt') == ck['optimizer_sha256'], 'Optimizer checksum mismatch')
    opt = torch.load(ckpath/'optimizer.pt', map_location='cpu', weights_only=True)
    layout = optimizer_layout(Qwen3Config.from_pretrained(ckpath, local_files_only=True), arm)
    parameters = validate_optimizer(opt, layout, state)
    return dict(checkpoint_hash=ck['model_sha256'], paired_initial_reader_hash=ck['paired_initial_reader_hash'],
                backbone_changed_tensors=len(changed), optimizer_parameters=parameters,
                frozen_table_exact=arm not in ('base', 'grad'), step=3815, input_tokens=1000079360)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('inputs', 'train', 'eval', 'panel'))
    p.add_argument('--workflow', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    torch.set_num_threads(8)
    require(code_hash() == CORE_SHA, 'Research core changed')
    corpus, vocab = Corpus(DATA/'corpus'), Vocabulary.load(DATA/'vocabulary.npz')
    common = checkpoint_meta(COMMON/'common/checkpoint-15259')
    require(not corpus.meta['engineering'] and corpus.budget == PILOT and len(vocab.keys) == PILOT.slots, 'Wrong data')
    require(common['model_sha256'] == '844c0b0320e88c446f19d2ebd028857ba5b8a4a4e61d2ced740c017eaf8249cf' and
            common['phase'] == 'common' and common['arm'] == 'base' and common['step'] == 15259 and
            common['corpus_hash'] == corpus.meta['manifest_hash'], 'Wrong common checkpoint')
    require_coverage(PREP/'coverage.json', corpus, vocab)
    report = dict(success=True, stage=a.stage, arm=a.arm, corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash)
    if a.stage == 'inputs':
        require(shutil.disk_usage(a.workflow).free >= 150*1024**3, 'Need at least 150 GiB free for fresh checkpoints')
        require(asset_identity(ASSETS, 'resources/qwen3_base_assets.json') == common['tokenizer'], 'Wrong Base assets')
        require(read_json(PREVIOUS/'complete.json').get('success') is True and
                read_json(PREVIOUS/'validated_panel.json')['success'], 'Stage-1 not validated')
        for arm in ('base', 'contextual', 'isolated', 'shuffled', 'shallow', 'delta', 'grad'):
            gate = read_json(PREVIOUS/f'validated_eval_{arm}.json')
            require(gate['success'] and file_hash(PREVIOUS/'eval'/arm/'metrics.json') == gate['metrics_sha256'] and
                    file_hash(PREVIOUS/'eval'/arm/'segments.jsonl') == gate['segments_sha256'], 'Changed Stage-1 evaluation')
        context_nll = read_json(PREVIOUS/'eval/contextual/metrics.json')['metrics']['overall']['nll']
        delta_nll = read_json(PREVIOUS/'eval/delta/metrics.json')['metrics']['overall']['nll']
        require(math.isfinite(context_nll) and math.isfinite(delta_nll) and delta_nll > context_nll,
                'Delta exclusion requires review: overall safeguard no longer fails')
        report.update(include_delta=False, delta_reason='Fails preregistered overall NLL safeguard',
                      contextual_nll=context_nll, delta_nll=delta_nll, arms=list(ARMS))
        prior = read_json(COMMON/'validated_tables.json')
        for arm in ('contextual', 'isolated', 'shuffled'):
            t, meta = load_table(COMMON/'tables'/arm, dict(constructor=arm, vocabulary_hash=vocab.hash,
                    corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=common['model_sha256'],
                    tokenizer=common['tokenizer'], backbone_contract=common['backbone_contract']))
            require(t['lookup'].shape == (262144, 1024) and meta['artifact_hash'] == prior['table_artifact_hashes'][arm], 'Wrong table')
            del t
    elif a.stage == 'train':
        require(a.arm in ARMS, 'Choose a trained arm')
        report.update(validate_train(a.workflow, a.arm, corpus, vocab, common))
    elif a.stage == 'eval':
        require(a.arm in ARMS, 'Choose an evaluated arm')
        ck = checkpoint_meta(a.workflow/'train'/a.arm/'checkpoint-3815')
        require(ck['phase'] == 'stage2' and ck['step'] == ck['total_steps'] == 3815, 'Wrong evaluated phase')
        report.update(validate_eval(a.workflow, a.arm, corpus, vocab, common, checkpoint=ck))
    else:
        for arm in ARMS:
            require(read_json(a.workflow/f'validated_train_{arm}.json')['success'], 'Missing trained arm gate')
            gate = read_json(a.workflow/f'validated_eval_{arm}.json')
            require(gate['success'] and file_hash(a.workflow/'eval'/arm/'metrics.json') == gate['metrics_sha256'] and
                    file_hash(a.workflow/'eval'/arm/'segments.jsonl') == gate['segments_sha256'], 'Missing/changed eval')
        for right in ('isolated', 'shuffled', 'base', 'grad'):
            r = read_json(a.workflow/'reports'/f'contextual_vs_{right}.json')
            require(r['replicates'] == 10000 and r['cluster'] == 'doc_id' and r['cross_seed_coupling'] == 'independent' and
                    r['backbone_seeds'] == [17] and not r['final_report'] and
                    all(math.isfinite(r[k]) for k in ('mean_difference', 'lower95', 'upper95')), 'Invalid comparison report')
    write_json(a.workflow/('validated_'+a.stage+('_'+a.arm if a.arm else '')+'.json'), report)
    print('STAGE2_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
