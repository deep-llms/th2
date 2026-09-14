"""Fixed paths/contracts for the authorized seed-29 remaining offline pilot."""
from pathlib import Path
from pilot_overnight import DATA, PREP, ASSETS, PROJECT, PYTHON, ALL, SPARE

SEED = 29
OLD_CORE = '055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9'
CORE = 'bbbd831476fd942d1131754ad3b57fd0323ed7f41aa03b8cb783bea497f8a2ff'
COMMON_HASH = 'da85b43b1509f0f8b5e807f951b624126f17d673edd505b8f041d8413f104038'
COMMON = Path('/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01')
OUT = Path('/mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01')
PREVIOUS = Path('/mnt/local/_outputs/deep-llms_th2/ccm_a3_seed17_20260914_a01')
OLD_SESSION = 'ccm_a3_diagnostics_20260914_a01'
SESSION = 'ccm_replication_seed29_20260914_a01'
STAGE1_ARMS = ('contextual', 'isolated', 'shuffled', 'shallow', 'delta', 'grad')
STAGE2_ARMS = ('base', 'contextual', 'isolated', 'shuffled', 'grad')
DECISION = OUT/'stage1/delta_decision.json'


def stage2_arms(decision):
    from pilot_gpu_ops import check
    check(decision.get('replication_policy') == 'per_seed' and decision.get('decision_seed') == SEED
          and decision.get('source_checkpoint_hash') == COMMON_HASH
          and type(decision.get('include_delta')) is bool, 'Wrong per-seed Delta decision')
    return STAGE2_ARMS + (('delta',) if decision['include_delta'] else ())
