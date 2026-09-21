#1 +60+a
#th2-tpbw-verify-envs-20260921-a01
set -euo pipefail
date -u
hostname
echo '--- conda roots present'
ls -d /mnt/local/conda*/envs/* 2>/dev/null || true
TASK_ROOT=$(ls -d /mnt/local/conda*/envs 2>/dev/null | head -1)
test -n "$TASK_ROOT"
for TASK_ENV in train_env eval; do
  TASK_PY="$TASK_ROOT/$TASK_ENV/bin/python3.11"
  test -x "$TASK_PY"
  "$TASK_PY" -c 'import sys, torch, transformers, datasets, accelerate; print("ENV", sys.executable); print("  python", sys.version.split()[0], "| torch", torch.__version__, "| cuda_available", torch.cuda.is_available(), "| devices", torch.cuda.device_count()); print("  transformers", transformers.__version__, "| datasets", datasets.__version__, "| accelerate", accelerate.__version__)'
done
"$TASK_ROOT/train_env/bin/python3.11" -c 'import scipy, wandb, pyarrow, huggingface_hub, entmax; print("train_env extras OK: scipy", scipy.__version__, "pyarrow", pyarrow.__version__)'
"$TASK_ROOT/eval/bin/python3.11" -c 'import lm_eval, sentencepiece; print("eval extras OK: lm_eval", lm_eval.__version__)'
echo TPBW_ENV_VERIFY_DONE
