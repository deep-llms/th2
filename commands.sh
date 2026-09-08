#1 +60+a
#th2-swt-qwen6-all-six-train-10k-then-burn-20260908-a01
set -euo pipefail
date -u
hostname
test "$PWD" = /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
command -v python
python scripts/verify_manifest.py verify --root "$PWD" --manifest resources/swt_qwen_launch_20260908.json
TASK_TEST_LOG=/mnt/local/_outputs/@PROJECT@/logs/swt_qwen6_all_six_prelaunch_tests_20260908_a01.log
test ! -e "$TASK_TEST_LOG"
mkdir -p /mnt/local/_outputs/@PROJECT@/logs
if ! CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s tests -v > "$TASK_TEST_LOG" 2>&1; then
  tail -n 120 "$TASK_TEST_LOG"
  exit 1
fi
tail -n 6 "$TASK_TEST_LOG"
echo SWT_DESTINATION_CPU_TESTS_PASSED
# The foreground workflow prepares shared CPU caches while existing burns stay
# active, then performs verified reclaim, two wait/free checks, production
# smoke, all six 10k-step runs, checkpoint verification and persistent burns.
bash scripts/train_capacity_b200.sh \
  /mnt/local/_outputs/@PROJECT@/swt/qwen6_allarms_10k_s42_20260908_a01 \
  B0 A128 A256 A512 C D
