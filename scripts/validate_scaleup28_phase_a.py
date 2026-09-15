"""Fail-closed gates for the 28L/12B Phase-A queue; local files only, no GPU calls."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch
import transformers
from ccm.artifacts import load_table
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require, read_json, write_json, file_hash, schedule
from ccm.data import Corpus
from ccm.keys import Vocabulary
from ccm.runtime import checkpoint_meta, require_coverage
from ccm.scaleup_protocol import checked_record
from ccm.studies import SCALEUP_BUDGET, open_corpus, require_vocabulary
from ccm.scaleup_data import validate_scaleup
from scaleup28_config import (ASSETS, COMMON_HASH, COMMON_ROOT, COMMON_SAVE_EVERY,
    CONTEXT_EVAL, CONTEXT_EVAL_CKPT, CONTEXT_EVAL_GATE, CORE, CORPUS_HASH, DATA, DATA28,
    FOLLOWUP, OLD_EVENT, OUT, PREP, PREVIOUS, RAW, REVISION, SEED28, SHALLOW_SEEDS,
    STAGE1_ROOT, STUDY, TABLES, TABLE_HASH, VOCAB_HASH)
from validate_stage1 import validate_eval
from validate_stage2 import validate_train

SMOKE_STEPS = dict(pipeline=32, stability=1024)


def context_eval_integrity(seed):
    m = read_json(CONTEXT_EVAL[seed]/'metrics.json')
    require(m['arm'] == 'contextual' and m['phase'] == 'stage2' and m['seed'] == seed
            and m['role'] == 'dev' and not m['final_evaluation'] and not m['engineering']
            and m['step'] == m['total_steps'] == 3815
            and m['checkpoint_hash'] == CONTEXT_EVAL_CKPT[seed]
            and m['source_checkpoint_hash'] == COMMON_HASH[seed]
            and m['diagnostic_table_hash'] == TABLE_HASH[(seed, 'contextual')],
            'Historical contextual evaluation contract changed')
    require(file_hash(CONTEXT_EVAL[seed]/'segments.jsonl') == m['segments_sha256'],
            'Historical contextual records changed')
    gate = read_json(CONTEXT_EVAL_GATE[seed])
    require(gate['success'] and gate['metrics_sha256'] == file_hash(CONTEXT_EVAL[seed]/'metrics.json'),
            'Historical contextual evaluation gate mismatch')
    return m


def inputs(corpus, vocab):
    require(shutil.disk_usage('/mnt/local').free >= 250*1024**3,
            'Need at least 250 GiB free for extension data plus Phase-A checkpoints')
    require((RAW/'en').is_dir(), 'Missing verified raw English shards')
    require(file_hash('resources/culturax_raw_manifest.tsv') == corpus.meta['provenance']['raw_manifest_hash']
            and corpus.meta['provenance']['revision'] == REVISION, 'Raw manifest/revision drifted')
    require(not DATA28.exists() and not DATA28.is_symlink(), 'Extension data root must be fresh')
    done = read_json(PREVIOUS/'complete.json')
    require(done.get('success') is True and done.get('event') == OLD_EVENT,
            'Previous seed29 queue is not verified complete')
    commons = {}
    for seed in SHALLOW_SEEDS:
        common = checkpoint_meta(COMMON_ROOT[seed]/'common/checkpoint-15259')
        require(common['model_sha256'] == COMMON_HASH[seed] and common['phase'] == 'common'
                and common['arm'] == 'base' and common['seed'] == seed and common['step'] == 15259
                and common['corpus_hash'] == corpus.meta['manifest_hash'], f'Wrong seed{seed} common')
        require(asset_identity(str(ASSETS), 'resources/qwen3_base_assets.json') == common['tokenizer'],
                'Wrong pinned Base assets')
        for kind in ('shallow', 'contextual'):
            t, meta = load_table(TABLES[seed]/kind, dict(constructor=kind, vocabulary_hash=vocab.hash,
                    corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=COMMON_HASH[seed],
                    tokenizer=common['tokenizer'], backbone_contract=common['backbone_contract']))
            require(meta['artifact_hash'] == TABLE_HASH[(seed, kind)]
                    and t['lookup'].shape == (262144, 1024), f'Wrong seed{seed} {kind} table')
            del t
        context_eval_integrity(seed)
        checkpoint_meta(STAGE1_ROOT[seed]/'train/shallow/checkpoint-977')
        commons[str(seed)] = common['model_sha256']
    return dict(commons=commons, previous_queue_event=done['event'])


def extension(vocab):
    ext = open_corpus(DATA28, verify=True)
    m = ext.meta
    require(m['historical_manifest_hash'] == CORPUS_HASH and m['historical_vocabulary_hash'] == VOCAB_HASH
            and m['ordered_mapping_hash'] == vocab.mapping_hash and not m['engineering']
            and ext.budget == SCALEUP_BUDGET, 'Extension binding mismatch')
    require(m['quotas'] == dict(compile=1000000000, adapt=256114688, dev=20000000, val=20000000,
            val28=20000000, continue_extension=999817216, common_extension=5999951872),
            'Wrong extension quotas')
    require_vocabulary(ext, vocab)
    result = validate_scaleup(ext)
    require(result['success'] and result['historical_manifest_hash'] == CORPUS_HASH,
            'Extension identity validation failed')
    return dict(extension_manifest_hash=m['manifest_hash'], ordered_mapping_hash=m['ordered_mapping_hash'])


def smoke(name, vocab):
    ext = open_corpus(DATA28, verify=False)
    steps = SMOKE_STEPS[name]
    root = OUT/f'common-{name if name == "pipeline" else "stability"}'
    r = checked_record(root/'stability.json')
    require(r['success'] is True and r['steps'] == steps and r['total_schedule_steps'] == 38147
            and r['study'] == STUDY and r['seed'] == SEED28 and not r['engineering']
            and r['corpus_hash'] == ext.meta['manifest_hash'] and r['common_lr'] == 3e-4
            and r['world_size'] == 8 and r['microbatch_segments'] == 8 and r['loss_chunk'] == 1024
            and r['activation_checkpointing'] and r['source_code_hash'] == CORE
            and r['config_sha256'] == file_hash(ASSETS/'config.json')
            and r['torch'] == torch.__version__ and r['transformers'] == transformers.__version__
            and r['objective_numerical_failure'] is False, f'Wrong {name} smoke record')
    require(not (root/'complete.json').exists(), 'Smoke must not publish full-training completion')
    ck = checkpoint_meta(root/f'checkpoint-{steps}')
    require(ck['phase'] == 'common' and ck['arm'] == 'base' and ck['step'] == steps
            and ck['total_steps'] == 38147 and ck.get('study') == STUDY and ck['seed'] == SEED28
            and ck['world_size'] == 8 and ck['corpus_hash'] == ext.meta['manifest_hash']
            and ck['backbone_contract']['config']['num_hidden_layers'] == 28,
            f'Wrong {name} smoke checkpoint')
    return dict(smoke=name, steps=steps, record_hash=r['record_hash'])


def common(vocab):
    ext = open_corpus(DATA28, verify=False)
    for name in SMOKE_STEPS:
        require(read_json(OUT/f'validated_smoke_{name}.json')['success'], 'Missing smoke gate')
    verified = read_json(OUT/'verified-common-base.json')
    ck = checkpoint_meta(OUT/'common-base/checkpoint-38147')
    require(verified['success'] and verified['metadata_hash'] == ck['metadata_hash']
            and verified['corpus_hash'] == ext.meta['manifest_hash'], 'validate-run gate mismatch')
    cfg = ck['config']
    require(ck['phase'] == 'common' and ck['arm'] == 'base' and ck['seed'] == SEED28
            and ck.get('study') == STUDY and ck['step'] == ck['total_steps'] == 38147
            and ck['world_size'] == 8 and not ck['engineering'] and ck['peak_lr'] == 3e-4
            and ck['model_dtype'] == 'torch.bfloat16'
            and ck['corpus_hash'] == ext.meta['manifest_hash']
            and ck['backbone_contract']['config']['num_hidden_layers'] == 28
            and cfg['source_code_hash'] == CORE and cfg['common_lr'] == 3e-4
            and cfg['stability_steps'] is None and cfg['lr_failure_report'] is None
            and cfg['stability_report'] == str(OUT/'common-stability/stability.json')
            and cfg['save_every'] == COMMON_SAVE_EVERY and cfg['data'] == str(DATA28)
            and cfg['microbatch_segments'] == 8 and cfg['loss_chunk'] == 1024
            and cfg['activation_checkpointing'], 'Wrong theta_10B_28L contract')
    count = 0
    with (OUT/'common-base/train.jsonl').open() as f:
        for count, line in enumerate(f, 1):
            r = json.loads(line)
            require(r['step'] == count, 'Wrong common step counter')
            require(math.isclose(r['lr'], schedule(count, 38147, 3e-4, .02), rel_tol=1e-10),
                    'Wrong 28L common LR schedule')
    require(count == 38147, 'Incomplete 28L common log')
    return dict(theta_10b_28l_hash=ck['model_sha256'], metadata_hash=ck['metadata_hash'],
                input_tokens=38147*262144)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inputs', 'extension', 'shallow-train', 'shallow-eval',
                                      'followup', 'smoke', 'common'))
    p.add_argument('--seed', type=int, choices=SHALLOW_SEEDS)
    p.add_argument('--name', choices=tuple(SMOKE_STEPS))
    a = p.parse_args()
    torch.set_num_threads(8)
    require(code_hash() == CORE, 'Unreviewed current core')
    corpus, vocab = Corpus(DATA/'corpus'), Vocabulary.load(DATA/'vocabulary.npz')
    require(corpus.meta['manifest_hash'] == CORPUS_HASH and vocab.hash == VOCAB_HASH
            and not corpus.meta['engineering'] and corpus.budget == PILOT
            and len(vocab.keys) == PILOT.slots, 'Wrong historical inputs')
    require_coverage(PREP/'coverage.json', corpus, vocab)
    report = dict(success=True, action=a.action, seed=a.seed, name=a.name, core_hash=CORE,
                  corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash)
    qualifier = ''
    if a.action in ('shallow-train', 'shallow-eval', 'followup'):
        require(a.seed in SHALLOW_SEEDS and a.name is None, 'Follow-up gates need --seed only')
        qualifier = f'_{a.seed}'
        root = OUT/f'shallow12_seed{a.seed}'
        common_meta = checkpoint_meta(COMMON_ROOT[a.seed]/'common/checkpoint-15259')
        require(common_meta['model_sha256'] == COMMON_HASH[a.seed], 'Wrong follow-up common writer')
        if a.action == 'shallow-train':
            report.update(validate_train(root, 'shallow', corpus, vocab, common_meta, seed=a.seed,
                                         common_root=COMMON_ROOT[a.seed], table_root=TABLES[a.seed],
                                         stage1_root=STAGE1_ROOT[a.seed], core_sha=CORE))
            ck = checkpoint_meta(root/'train/shallow/checkpoint-3815')
            require(ck.get('study') == FOLLOWUP, 'Follow-up must record its explicit study version')
        elif a.action == 'shallow-eval':
            ck = checkpoint_meta(root/'train/shallow/checkpoint-3815')
            require(ck.get('study') == FOLLOWUP and ck['step'] == ck['total_steps'] == 3815,
                    'Wrong evaluated follow-up checkpoint')
            report.update(validate_eval(root, 'shallow', corpus, vocab, common_meta,
                                        checkpoint=ck, seed=a.seed, table_root=TABLES[a.seed]))
        else:
            m = context_eval_integrity(a.seed)
            r = read_json(root/'reports/contextual_vs_shallow.json')
            require(r['replicates'] == 10000 and r['bootstrap_seed'] == 20260913
                    and r['cluster'] == 'doc_id' and r['cross_seed_coupling'] == 'independent'
                    and r['backbone_seeds'] == [a.seed] and not r['final_report']
                    and r['population'] == 'overall' and r['left'] == [str(CONTEXT_EVAL[a.seed])]
                    and r['right'] == [str(root/'eval/shallow')]
                    and all(math.isfinite(r[k]) for k in ('mean_difference', 'lower95', 'upper95')),
                    'Invalid Deep-vs-Shallow comparison')
            shallow_m = read_json(root/'eval/shallow/metrics.json')
            require(shallow_m['diagnostic_table_hash'] == m['diagnostic_table_hash'],
                    'Deep/Shallow variance bins use different tables')
            report.update(deep_vs_shallow=dict(mean_difference=r['mean_difference'],
                          lower95=r['lower95'], upper95=r['upper95']))
    elif a.action == 'inputs':
        require(a.seed is None and a.name is None, 'Unexpected arguments')
        report.update(inputs(corpus, vocab))
    elif a.action == 'extension':
        require(a.seed is None and a.name is None, 'Unexpected arguments')
        report.update(extension(vocab))
    elif a.action == 'smoke':
        require(a.name in SMOKE_STEPS and a.seed is None, 'Smoke gate needs --name')
        qualifier = f'_{a.name}'
        report.update(smoke(a.name, vocab))
    else:
        require(a.seed is None and a.name is None, 'Unexpected arguments')
        report.update(common(vocab))
    write_json(OUT/f'validated_{a.action.replace("-", "_")}{qualifier}.json', report)
    print('SCALEUP28_PHASE_A_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
