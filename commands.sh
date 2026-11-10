#1 +20+a
#th2-wc69-family-f10-cpu-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/eval/bin/python3.11 - <<'PYCODE'
from pathlib import Path
from importlib.metadata import version
import lm_eval,json
from scripts.gpu_process import inspect
print('F10_READONLY_GPU',json.dumps(inspect()),flush=True)
root=Path(lm_eval.__file__).parent/'tasks'
for name in ['triviaqa/default.yaml','triviaqa/utils.py','mmlu/continuation/_continuation_template_yaml','mmlu/continuation/_mmlu.yaml']:
 p=root/name
 print('TASK_SOURCE',str(p),p.read_text() if p.is_file() else 'ABSENT',flush=True)
print('PACKAGES',{k:version(k) for k in ['lm-eval','torch','transformers','datasets']},flush=True)
for p in [Path('/mnt/local/_data/deep-llms_th2'),Path('/mnt/local/.cache/huggingface/datasets')]:
 print('DATA_ROOT',str(p),[x.name for x in p.iterdir()] if p.exists() else 'ABSENT',flush=True)
PYCODE
TASK_OUT=/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-cpu-20261111-a01
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
cp -r eval efficient_llm scripts "$TASK_OUT/"
TASK_CMD="cd $TASK_OUT && exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f10_cpu --f9-root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-20261110-a01/results/a2_family_20261107/downstream --output $TASK_OUT/f10a >$TASK_OUT/cpu.log 2>&1"
tmux new-session -d -s a2_family_f10_cpu_20261111_a01 "$TASK_CMD"
tmux set-option -w -t a2_family_f10_cpu_20261111_a01 remain-on-exit on
echo F10A_CPU_SUBMITTED
