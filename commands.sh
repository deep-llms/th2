#1 +60+a
#th2-swt-final5k-diagnostics-then-finetune-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
date -u
python - <<'PY'
from pathlib import Path
from scripts.reclaim_verified_burn import identity
from scripts.gpu_status import require_free
expected = [(199032, '168650771'), (215026, '168932038'), (227206, '169491835'), (227975, '169505946'), (227976, '169505950'), (227791, '169503251'), (227792, '169503254'), (227221, '169492061'), (226967, '169489549'), (227874, '169504377'), (227948, '169505631'), (227949, '169505632'), (227950, '169505634'), (227183, '169491796'), (227887, '169504801'), (227057, '169491176'), (227951, '169505635'), (227188, '169491815'), (227892, '169504857'), (227893, '169504861'), (227966, '169505683'), (227967, '169505686')]
for pid, start in expected:
    try:
        current = identity(pid)
        assert current['start'] != start, f'Old owned process still alive: {pid}'
    except (FileNotFoundError, ProcessLookupError):
        pass
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01')
assert (root/'eval/complete.json').is_file()
assert not (root/'complete.json').exists()
require_free(list(range(8)))
print('OLD_QUEUE_AND_ALL_OWNED_WORKERS_GONE_ALL_GPUS_FREE', flush=True)
print('PRESERVED_FINETUNE_RESULTS', len(list((root/'finetune').glob('*/result.json'))), flush=True)
PY
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_final_checkpoint_handoff.py -v
bash scripts/final_checkpoint_diagnostics_finetune_b200.sh /mnt/local/_outputs/deep-llms_th2/swt/final5k_diagnostics_finetune_20260909_a01
