"""Fresh seed-29 replication gates; old inputs retain their original code identity."""
import argparse
import math
from pathlib import Path
import shutil

import torch
from ccm.artifacts import load_table
from ccm.cli import code_hash
from ccm.contracts import PILOT, read_json, write_json, file_hash, require
from ccm.data import Corpus
from ccm.keys import Vocabulary
from ccm.runtime import checkpoint_meta, require_coverage, delta_policy_matches
from seed29_replication_config import (SEED, CORE, OLD_CORE, COMMON_HASH, COMMON, OUT,
                                      DATA, PREP, STAGE1_ARMS, DECISION, stage2_arms)
from validate_seed29 import validate_common
from validate_stage1 import validate_train as train1, validate_eval
from validate_stage2 import validate_train as train2


def inputs(corpus, vocab, common):
    require(shutil.disk_usage(OUT).free >= 200*1024**3, 'Need 200 GiB free for new replication outputs')
    require(corpus.meta['provenance']['source_code_hash'] == OLD_CORE, 'Wrong frozen data provenance')
    prep = read_json(PREP/'complete.json')
    require(prep['success'] and prep['corpus_hash'] == corpus.meta['manifest_hash']
            and prep['vocabulary_hash'] == vocab.hash and prep['quotas'] == PILOT.quotas()
            and prep['optimizer_batches'] == dict(common=15259, stage1=977, stage2=3815),
            'Invalid immutable preparation contract')
    for path, sha in prep['files'].items():
        path = Path(path)
        require(path.resolve().is_relative_to(DATA.resolve()) or path.resolve().is_relative_to(PREP.resolve()),
                'Unexpected input path')
        require(file_hash(path) == sha, 'Changed prepared input')
    done = read_json(COMMON/'complete.json')
    require(done['success'] and done['event'] == 'seed29_common_verified_and_burns_active',
            'Independent common workflow incomplete')
    prior = read_json(COMMON/'validated_common.json')
    require(prior['success'] and prior['checkpoint_hash'] == COMMON_HASH, 'Wrong original common gate')
    result = validate_common(COMMON)  # Full old-contract checkpoint/optimizer validation; no writes.
    require(result['checkpoint_hash'] == COMMON_HASH and common['seed'] == SEED, 'Wrong seed29 common')
    return dict(common=result, original_core_hash=OLD_CORE, new_core_hash=CORE,
                policy_change='User-approved per-seed Delta inclusion; thresholds/model/training unchanged')


def tables(corpus, vocab, common):
    done = read_json(OUT/'tables/complete.json')
    require(done['success'] and done['tokens'] == PILOT.compile_tokens, 'Incomplete compiler pass')
    hashes = {}
    context = None
    for arm in ('shallow', 'contextual', 'delta', 'isolated', 'shuffled'):
        t, m = load_table(OUT/'tables'/arm, dict(constructor=arm, vocabulary_hash=vocab.hash,
            corpus_hash=corpus.meta['manifest_hash'], source_checkpoint_hash=COMMON_HASH,
            tokenizer=common['tokenizer'], backbone_contract=common['backbone_contract'],
            source_seed=SEED, source_code_hash=CORE))
        require(t['lookup'].shape == (262144, 1024) and t['lookup'].dtype == torch.bfloat16
                and bool(torch.isfinite(t['lookup']).all()), 'Wrong/nonfinite table')
        if arm in ('shallow', 'contextual', 'delta'):
            require(torch.equal(t['counts'], torch.from_numpy(vocab.counts)), 'Wrong compiler counts')
            require(bool(torch.isfinite(t['variance']).all()) and bool((t['variance'] >= 0).all()),
                    'Invalid compile variance')
        if arm == 'contextual':
            context = t['lookup'].clone()
        if arm == 'shuffled':
            expected = torch.randperm(262144, generator=torch.Generator().manual_seed(200029))
            require(m['permutation_seed'] == 200029 and torch.equal(t['permutation'], expected)
                    and torch.equal(t['lookup'], context[expected]), 'Wrong seed29 shuffled table')
        hashes[arm] = m['artifact_hash']
    return dict(checkpoint_hash=COMMON_HASH, table_artifact_hashes=hashes, compiler=done)


def report_check(path, left, right, population='overall'):
    r = read_json(path)
    require(r['replicates'] == 10000 and r['bootstrap_seed'] == 20260913
            and r['cluster'] == 'doc_id' and r['cross_seed_coupling'] == 'independent'
            and r['backbone_seeds'] == [29] and not r['final_report']
            and r['population'] == population and r['left'] == [str(left)] and r['right'] == [str(right)]
            and all(math.isfinite(r[k]) for k in ('mean_difference','lower95','upper95')),
            'Invalid paired comparison')


def panel(phase, corpus, vocab):
    root = OUT/phase
    if phase == 'stage1':
        trained = STAGE1_ARMS
        evaluated = ('base',)+trained
    else:
        trained = evaluated = stage2_arms(read_json(DECISION))
    for arm in trained:
        g = read_json(root/f'validated_train_{arm}.json')
        ck = checkpoint_meta(root/'train'/arm/f'checkpoint-{977 if phase == "stage1" else 3815}')
        require(g['success'] and g['checkpoint_hash'] == ck['model_sha256'], 'Changed/missing training gate')
    for arm in evaluated:
        g = read_json(root/f'validated_eval_{arm}.json')
        require(g['success'] and file_hash(root/'eval'/arm/'metrics.json') == g['metrics_sha256']
                and file_hash(root/'eval'/arm/'segments.jsonl') == g['segments_sha256'],
                'Changed/missing evaluation gate')
    for right in (('isolated','shuffled') if phase == 'stage1' else ('isolated','shuffled','base','grad')):
        report_check(root/'reports'/f'contextual_vs_{right}.json', root/'eval/contextual', root/'eval'/right)
    if phase == 'stage1':
        d = read_json(DECISION)
        stage2_arms(d)  # Validate seed/policy/writer even on exclusion.
        require(d['corpus_hash'] == corpus.meta['manifest_hash'] and d['vocabulary_hash'] == vocab.hash
                and d['cluster'] == 'doc_id', 'Wrong Delta data/policy')
        expected = (d['results']['hit_vs_contextual']['upper95'] < 0
                    and d['results']['hit_vs_shuffled']['upper95'] < 0
                    and d['results']['miss_vs_contextual']['upper95'] <= .002
                    and d['overall_safeguard'])
        require(d['include_delta'] == expected, 'Delta criterion mismatch')
        if expected:
            require(delta_policy_matches(d, SEED, COMMON_HASH), 'Delta gate cannot authorize own seed')
    if phase == 'stage2' and 'delta' in evaluated:
        for right in ('contextual','isolated','shuffled'):
            report_check(root/'reports'/f'delta_vs_{right}.json', root/'eval/delta', root/'eval'/right)
    return dict(trained_arms=list(trained), evaluated_arms=list(evaluated),
                delta_decision_sha256=file_hash(DECISION))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inputs','tables','train','eval','panel'))
    p.add_argument('--phase', choices=('stage1','stage2'))
    p.add_argument('--arm')
    a = p.parse_args()
    torch.set_num_threads(8)
    require(code_hash() == CORE, 'Unreviewed current core')
    corpus, vocab = Corpus(DATA/'corpus'), Vocabulary.load(DATA/'vocabulary.npz')
    common = checkpoint_meta(COMMON/'common/checkpoint-15259')
    require(common['model_sha256'] == COMMON_HASH and common['seed'] == SEED
            and common['corpus_hash'] == corpus.meta['manifest_hash']
            and not corpus.meta['engineering'] and corpus.budget == PILOT
            and len(vocab.keys) == PILOT.slots, 'Wrong seed29 inputs')
    require_coverage(PREP/'coverage.json', corpus, vocab)
    report = dict(success=True, action=a.action, phase=a.phase, arm=a.arm, seed=SEED,
                  corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash, core_hash=CORE)
    if a.action in ('inputs','tables'):
        require(a.phase is None and a.arm is None, 'Unexpected phase/arm')
        report.update(inputs(corpus,vocab,common) if a.action == 'inputs' else tables(corpus,vocab,common))
        dest = OUT/f'validated_{a.action}.json'
    else:
        require(a.phase is not None, 'Choose phase')
        arms = STAGE1_ARMS if a.phase == 'stage1' else stage2_arms(read_json(DECISION))
        root = OUT/a.phase
        if a.action == 'train':
            require(a.arm in arms, 'Unscheduled training arm')
            kw = dict(seed=SEED, common_root=COMMON, table_root=OUT/'tables', core_sha=CORE)
            if a.phase == 'stage2':
                kw.update(stage1_root=OUT/'stage1', delta_decision=DECISION if a.arm == 'delta' else None)
            report.update((train1 if a.phase == 'stage1' else train2)(root,a.arm,corpus,vocab,common,**kw))
        elif a.action == 'eval':
            require(a.arm in (('base',)+arms), 'Unscheduled evaluation arm')
            ck = common if a.phase == 'stage1' and a.arm == 'base' else checkpoint_meta(
                root/'train'/a.arm/f'checkpoint-{977 if a.phase == "stage1" else 3815}')
            report.update(validate_eval(root,a.arm,corpus,vocab,common,checkpoint=ck,seed=SEED,table_root=OUT/'tables'))
        else:
            require(a.arm is None, 'Unexpected panel arm')
            report.update(panel(a.phase,corpus,vocab))
        dest = root/f'validated_{a.action}{("_"+a.arm) if a.arm else ""}.json'
    write_json(dest, report)
    print('SEED29_REPLICATION_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
