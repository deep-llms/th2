#1 +240+a
#th2-pq5r-dot-l34-4x-launch-20261128-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
date -u
hostname
out=/mnt/local/_outputs/deep-llms_th2/dot_l34_20261128
src=$out/src
base=$out/run
data=/mnt/local/_data/deep-llms_th2/dot_data_20261128
PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
if [ -e "$src" ] || [ -e "$base" ]; then echo L34_ALREADY_LAUNCHED; cat "$base/jobs.json" 2>/dev/null || true; exit 0; fi
echo "60a0af0bae9d2553b70e546845183491089954a2210be5b11ed8ea4cb396a534  $out/inspection-a01.json" | sha256sum -c -
test -x "$PY"
command -v tmux >/dev/null
# frozen code copy: later runner pushes must not change the code under the running jobs
mkdir -p "$src/docs/results/local_rows_20261126"
cp -r efficient_llm scripts eval run_experiments.py "$src/"
cp docs/results/local_rows_20261126/manifest.json "$src/docs/results/local_rows_20261126/"
(cd "$src" && find efficient_llm scripts eval run_experiments.py docs -type f -name '*.py' -o -type f -name '*.json' | sort | xargs sha256sum > "$out/src.sha256")
wc -l "$out/src.sha256"
jobs="0:huginn_r3_lr28_4x 1:huginn_r3_lr28_4x 0:dotpair_rand_inj_t512_si_lr14_4x 1:dotpair_rand_inj_t512_si_lr14_4x 0:dotfull_rand_inj_t512_si_lr14_4x 1:dotfull_rand_inj_t512_si_lr14_4x 0:dotshared_rand_t512_si_lr14_4x 1:dotshared_rand_t512_si_lr14_4x"
cmd="$PY -u scripts/node_dot_queue.py --base $base --rows $data/norm/local_rows --requests $data/downstream/requests.json --inspection $out/inspection-a01.json --inspection-sha256 60a0af0bae9d2553b70e546845183491089954a2210be5b11ed8ea4cb396a534 --jobs $jobs > $out/supervisor.log 2>&1"
tmux new-session -d -s dotl34_20261128 -c "$src" "$cmd"
tmux set-option -w -t dotl34_20261128 remain-on-exit on
echo LAUNCHED_TMUX dotl34_20261128
sleep 200
tmux ls 2>/dev/null || true
tail -n 30 "$out/supervisor.log" | cut -c1-300
ls "$base" 2>&1 | head -20
cat "$base/jobs.json" 2>/dev/null || echo NO_JOBS_YET
ls -la /mnt/local/_gpu_guard/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
