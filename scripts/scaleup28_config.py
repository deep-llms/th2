"""Fixed paths/contracts for the authorized 28L/12B Phase-A queue (plan v3 §23.A)."""
from pathlib import Path
from pilot_overnight import DATA, PREP, ASSETS, PROJECT, PYTHON, ALL, SPARE

CORE = '214d57c4816b37298000051da8a0ef16fabdfce1f2c870e8e5262729e35b1582'
STUDY = 'scaleup28-12b-v3'
FOLLOWUP = 'pilot12-shallow-followup'
SEED28 = 17
SHALLOW_SEEDS = (17, 29)
REVISION = '6a8734bc69fefcbb7735f4f9250f43e4cd7a442e'
CORPUS_HASH = '10746d723b6098729cb8824d52f377edb04a97fed69f1106101afdf1763059d5'
VOCAB_HASH = 'bff81f51cb1a9c03ade925996743fa1575a92560d74f4dcb960deb683da7d6c1'

RAW = Path('/mnt/local/_data/deep-llms_th2/data/raw')
# a2 recovery roots (2026-09-16). Bumped to _a02 after the first a2 attempt
# aborted in the GPU clear (buggy identity check) and left a stale session/root.
DATA28 = Path('/mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260916_a02')
OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a2_20260916_a02')
SESSION = 'ccm_scaleup28_phase_a2_20260916_a02'
# Prior interrupted queue: completed seed-17 section is kept and verified; the
# interrupted seed-29 section and the partial data root are removal targets.
OUT_A1 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a_20260915_a01')
DATA28_PARTIAL = Path('/mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260915_a01')

COMMON17 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01')
COMMON29 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01')
STAGE1_17 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_stage1_seed17_20260913_a01')
STAGE2_17 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_stage2_seed17_20260913_a01')
REP29 = Path('/mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01')

# Current burn owner to disarm cooperatively before any GPU reclaim.
PREVIOUS = REP29
OLD_SESSION = 'ccm_replication_seed29_20260914_a01'
OLD_SCRIPT = b'scripts/pilot_seed29_replication.py'
OLD_EVENT = 'seed29_replication_verified_and_burns_active'

COMMON_ROOT = {17: COMMON17, 29: COMMON29}
COMMON_HASH = {17: '844c0b0320e88c446f19d2ebd028857ba5b8a4a4e61d2ced740c017eaf8249cf',
               29: 'da85b43b1509f0f8b5e807f951b624126f17d673edd505b8f041d8413f104038'}
STAGE1_ROOT = {17: STAGE1_17, 29: REP29/'stage1'}
TABLES = {17: COMMON17/'tables', 29: REP29/'tables'}
TABLE_HASH = {(17, 'shallow'): '4b85e3bd4a25f0f82397c596ae64ef8767e35fab7dd7553b24258c7d033c7e7b',
              (17, 'contextual'): '923c1c507b424a19f46fb6b453f028d3c203a7a63413d21c767d2f799e4e5a0b',
              (29, 'shallow'): '67c9cf9238afed63bbb363c09f3c808f1dcf0e19e2e4899a70137c1029fc490a',
              (29, 'contextual'): 'df798cf42d136f6aab6ea36611557afed314b79c59b017d800d725cdbf762ef3'}
CONTEXT_EVAL = {17: STAGE2_17/'eval/contextual', 29: REP29/'stage2/eval/contextual'}
CONTEXT_EVAL_GATE = {17: STAGE2_17/'validated_eval_contextual.json',
                     29: REP29/'stage2/validated_eval_contextual.json'}
CONTEXT_EVAL_CKPT = {17: 'fe85b8d5e4ec41d909a58772ccbbe8f6b3b6d1117b2d1f97bdd978344fd23368',
                     29: '086ae1b1bcb9e21a059e2da676befd4e9a4ef6eac974f0894c95ba470df2e67e'}

# Pre-launch decision 1(c), 2026-09-15: 32-update pipeline smoke first, then a
# 1,024-update stability smoke past the 763-update warmup. Decision 3a: common
# checkpoints every 5,000 updates plus the final one.
SMOKES = (('common-pipeline', 32), ('common-stability', 1024))
COMMON_SAVE_EVERY = 5000

# Phase-B panel (plan v3 §23.B), against the verified theta_10B_28L.
PANEL_OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_panel_20260917_a01')
PANEL_SESSION = 'ccm_scaleup28_panel_20260917_a01'
THETA = OUT/'common-base/checkpoint-38147'
THETA_HASH = '8a370d274306089b8dd3e543cf3b124d9738e106ec8b731259d3876ec21b0bab'
THETA_META_HASH = '9dfee9f71c8a406227f22ddf7a705e2882af6a74d4321d0060183858fb9034df'
EXT_MANIFEST_HASH = 'b9e8c3bb38410f65e689face074d39f85f21cf7b002ec63d5329b7e103dba8aa'
MAPPING_HASH = '4777e0c0ecf0cf1c08569c1ac410d64580b7a99121b7478c76443b1ac1e668fb'
# Previous owner for the panel's cooperative disarm: the completed a2 observer.
A2_EVENT = 'scaleup28_phase_a2_verified_and_burns_active'
A2_SCRIPT = b'scripts/pilot_scaleup28_phase_a2.py'


def shallow_root(seed):
    from pilot_gpu_ops import check
    check(seed in SHALLOW_SEEDS, 'Only seeds 17 and 29 have authorized Shallow follow-ups')
    return OUT/f'shallow12_seed{seed}'
