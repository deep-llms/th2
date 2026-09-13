"""Verify the finished predecessor and the independent seed-29 common model."""
import argparse
import math
from pathlib import Path
import shutil
import torch
from transformers import Qwen3Config

from ccm.cli import code_hash, asset_identity
from ccm.contracts import PILOT, require, read_json, write_json, file_hash, seed_bundle
from ccm.data import Corpus
from ccm.runtime import checkpoint_meta
from pilot_overnight import CORE_SHA, DATA, ASSETS
from queue_seed29 import PREVIOUS, COMPLETE_EVENT
from validate_pilot_handoff import validate_log
from validate_stage2 import optimizer_layout


def validate_previous():
    done = read_json(PREVIOUS/'complete.json')
    require(done.get('success') is True and done.get('stage') == 'complete' and
            done.get('event') == COMPLETE_EVENT, 'Full Stage-2 workflow is not complete')
    panel = read_json(PREVIOUS/'validated_panel.json')
    require(panel['success'] and panel['stage'] == 'panel', 'Previous panel failed')
    for arm in ('base', 'contextual', 'isolated', 'shuffled', 'grad'):
        gate = read_json(PREVIOUS/f'validated_train_{arm}.json')
        ck = checkpoint_meta(PREVIOUS/'train'/arm/'checkpoint-3815')
        require(gate['success'] and gate['checkpoint_hash'] == ck['model_sha256'] and
                ck['phase'] == 'stage2' and ck['arm'] == arm and ck['seed'] == 17 and
                ck['step'] == ck['total_steps'] == 3815 and ck['corpus_hash'] == panel['corpus_hash'],
                'Prior training gate/checkpoint mismatch')
        eg = read_json(PREVIOUS/f'validated_eval_{arm}.json')
        require(eg['success'] and eg['checkpoint_hash'] == ck['model_sha256'] and
                file_hash(PREVIOUS/'eval'/arm/'metrics.json') == eg['metrics_sha256'] and
                file_hash(PREVIOUS/'eval'/arm/'segments.jsonl') == eg['segments_sha256'], 'Prior eval changed')
    for right in ('base', 'grad', 'isolated', 'shuffled'):
        r = read_json(PREVIOUS/'reports'/f'contextual_vs_{right}.json')
        require(r['replicates'] == 10000 and r['cluster'] == 'doc_id' and
                r['backbone_seeds'] == [17] and not r['final_report'] and
                all(math.isfinite(r[k]) for k in ('mean_difference', 'lower95', 'upper95')), 'Missing prior comparison')
    # Intentionally no condition on which method won: the user authorized
    # common pretraining irrespective of the seed-17 scientific outcome.
    return dict(previous_complete_sha256=file_hash(PREVIOUS/'complete.json'),
                previous_panel_sha256=file_hash(PREVIOUS/'validated_panel.json'))


def validate_common(root):
    corpus = Corpus(DATA/'corpus')
    require(not corpus.meta['engineering'] and corpus.budget == PILOT, 'Wrong scientific corpus')
    assets = asset_identity(ASSETS, 'resources/qwen3_base_assets.json')
    ckpath = root/'common/checkpoint-15259'
    ck = checkpoint_meta(ckpath)
    require(read_json(root/'common/complete.json') == dict(success=True, phase='common', arm='base',
            step=15259, input_tokens=4000055296, checkpoint='checkpoint-15259'), 'Incomplete seed-29 training')
    require(ck['seed'] == 29 and ck['seeds'] == seed_bundle(29) and ck['phase'] == 'common' and ck['arm'] == 'base' and
            ck['step'] == ck['total_steps'] == 15259 and ck['world_size'] == 8 and not ck['engineering'] and
            ck['source_checkpoint_hash'] is None and ck['paired_initial_reader_hash'] is None and
            ck['table_hash'] is None and ck['model_dtype'] == 'torch.bfloat16' and ck['peak_lr'] == 3e-4,
            'Wrong independent common-training contract')
    cfg = ck['config']
    require(ck['corpus_hash'] == corpus.meta['manifest_hash'] and ck['tokenizer'] == assets and
            cfg['source_code_hash'] == CORE_SHA and cfg['checkpoint'] is None and cfg['table'] is None and
            cfg['data'] == str(DATA/'corpus') and cfg['model_config'] == str(ASSETS) and
            cfg['microbatch_segments'] == 8 and cfg['loss_chunk'] == 1024 and cfg['activation_checkpointing'] and
            not cfg['online'], 'Wrong data/configuration or inherited model state')
    config = Qwen3Config.from_pretrained(ckpath, local_files_only=True)
    require(config.num_hidden_layers == 12 and config.hidden_size == 1024 and config.tie_word_embeddings,
            'Wrong pilot model')
    validate_log(root/'common/train.jsonl', 15259, 262144)
    state = torch.load(ckpath/'model.pt', map_location='cpu', weights_only=True)
    require(all(k.startswith('backbone.') and v.dtype == torch.bfloat16 and bool(torch.isfinite(v).all())
                for k, v in state.items()), 'Invalid common model state')
    require(torch.equal(state['backbone.model.embed_tokens.weight'], state['backbone.lm_head.weight']), 'Broken tying')
    require(file_hash(ckpath/'optimizer.pt') == ck['optimizer_sha256'], 'Optimizer checksum mismatch')
    opt = torch.load(ckpath/'optimizer.pt', map_location='cpu', weights_only=True)
    layout = optimizer_layout(config, 'base')
    groups = opt['optimizer']['param_groups']
    require(len(groups) == len(layout), 'Wrong optimizer groups')
    names = []
    for g, (mult, decay, ns) in zip(groups, layout):
        require(g['params'] == list(range(len(names), len(names)+len(ns))) and
                g['multiplier'] == mult == 1. and g['weight_decay'] == decay and
                math.isclose(g['lr'], 3e-5, rel_tol=1e-10) and tuple(g['betas']) == (.9, .95) and
                g['eps'] == 1e-8, 'Wrong optimizer mapping/settings')
        names.extend(ns)
    require(len(names) == len(opt['masters']) and set(opt['optimizer']['state']) == set(range(len(names))),
            'Incomplete optimizer state')
    require(sum(m.numel() for m in opt['masters']) == 344354816, 'Wrong parameter count')
    for i, name in enumerate(names):
        m, s = opt['masters'][i], opt['optimizer']['state'][i]
        require(m.dtype == torch.float32 and bool(torch.isfinite(m).all()) and
                torch.equal(m.bfloat16(), state[name]) and int(s['step']) == 15259, 'Invalid masters or counter')
        for k in ('exp_avg', 'exp_avg_sq'):
            require(s[k].dtype == torch.float32 and s[k].shape == m.shape and bool(torch.isfinite(s[k]).all()),
                    'Invalid Adam moments')
    return dict(seed=29, step=15259, input_tokens=4000055296, checkpoint_hash=ck['model_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('previous', 'common'))
    p.add_argument('--workflow', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    require(code_hash() == CORE_SHA, 'Research core changed')
    if a.stage == 'previous':
        require(shutil.disk_usage(a.workflow).free >= 100*1024**3, 'Need 100 GiB free for seed-29 checkpoints')
        result = validate_previous()
    else:
        result = validate_common(a.workflow)
    report = dict(success=True, stage=a.stage, **result)
    write_json(a.workflow/f'validated_{a.stage}.json', report)
    print('SEED29_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
