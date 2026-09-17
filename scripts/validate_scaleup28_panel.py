"""Fail-closed input gate for the Phase-B panel; local files only, no GPU calls."""
import shutil

import torch
from ccm.cli import code_hash
from ccm.contracts import require, read_json, write_json
from ccm.keys import Vocabulary
from ccm.runtime import checkpoint_meta
from ccm.studies import SCALEUP, SCALEUP_BUDGET, open_corpus, require_vocabulary
from scaleup28_config import (A2_EVENT, CORE, DATA, DATA28, EXT_MANIFEST_HASH,
                              MAPPING_HASH, OUT, PANEL_OUT, THETA, THETA_HASH,
                              THETA_META_HASH, VOCAB_HASH)


def main():
    torch.set_num_threads(8)
    require(code_hash() == CORE, 'Unreviewed current core')
    require(shutil.disk_usage('/mnt/local').free >= 400*1024**3,
            'Need at least 400 GiB free for tables and the panel checkpoints')
    ext = open_corpus(DATA28, verify=True)
    require(ext.meta['manifest_hash'] == EXT_MANIFEST_HASH and not ext.meta['engineering']
            and ext.budget == SCALEUP_BUDGET, 'Extension corpus binding mismatch')
    vocab = Vocabulary.load(DATA/'vocabulary.npz')
    require(vocab.hash == VOCAB_HASH and vocab.mapping_hash == MAPPING_HASH,
            'Vocabulary hash drifted')
    require_vocabulary(ext, vocab)
    theta = checkpoint_meta(THETA)
    require(theta['model_sha256'] == THETA_HASH and theta['metadata_hash'] == THETA_META_HASH
            and theta['phase'] == 'common' and theta['arm'] == 'base' and theta['seed'] == 17
            and theta['step'] == theta['total_steps'] == 38147 and theta['world_size'] == 8
            and theta.get('study') == SCALEUP and not theta['engineering']
            and theta['corpus_hash'] == ext.meta['manifest_hash']
            and theta['backbone_contract']['config']['num_hidden_layers'] == 28,
            'theta_10B_28L contract mismatch')
    done = read_json(OUT/'complete.json')
    require(done.get('success') is True and done.get('event') == A2_EVENT,
            'a2 queue is not verified complete')
    report = dict(success=True, theta_hash=THETA_HASH, extension_manifest_hash=EXT_MANIFEST_HASH,
                  vocabulary_hash=vocab.hash, ordered_mapping_hash=vocab.mapping_hash,
                  core_hash=CORE)
    write_json(PANEL_OUT/'validated_panel_inputs.json', report)
    print('SCALEUP28_PANEL_INPUT_GATE_PASS', report, flush=True)


if __name__ == '__main__':
    main()
