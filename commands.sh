#1 +240+a
#th2-pq5r-dot-l47-d1536-4x-launch-20261128-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
date -u
hostname
prev=/mnt/local/_outputs/deep-llms_th2/dot_l45_20261128
out=/mnt/local/_outputs/deep-llms_th2/dot_l47_20261128
src=$out/src
base=$out/run
data=/mnt/local/_data/deep-llms_th2/dot_data_20261128
PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
receipt=$prev/inspection-a01.json
if [ -e "$out" ]; then echo L47_ALREADY_LAUNCHED; cat "$base/jobs.json" 2>/dev/null || true; exit 0; fi
python3 -c "import json,sys;t=json.load(open(sys.argv[1]));sys.exit(0 if t.get('native_restored') else 1)" "$prev/run/terminal.json"
echo "81cef97c719c3e2cb603a138e47b25e7f131f0acbb00538c6de539aa9136bea9  $receipt" | sha256sum -c -
test -x "$PY"
command -v tmux >/dev/null
mkdir -p "$src/docs/results/local_rows_20261126"
cp -r efficient_llm scripts eval run_experiments.py "$src/"
cp docs/results/local_rows_20261126/manifest.json "$src/docs/results/local_rows_20261126/"
(cd "$src" && find efficient_llm scripts eval run_experiments.py docs -type f \( -name '*.py' -o -name '*.json' \) | sort | xargs sha256sum > "$out/src.sha256")
wc -l "$out/src.sha256"
jobs="0:dotpair_rand_inj_lin_si_coda_lr14_d1536_4x 1:dotpair_rand_inj_lin_si_coda_lr14_d1536_4x 0:huginn_r3_lr28_d1536_4x 1:huginn_r3_lr28_d1536_4x 0:huginn_r3_lr14_adamw_d1536_4x 1:huginn_r3_lr14_adamw_d1536_4x 0:dense8_sq_lr10_fa_d1536_4x 1:dense8_sq_lr10_fa_d1536_4x"
cmd="$PY -u scripts/node_dot_queue.py --base $base --rows $data/norm/local_rows --requests $data/downstream/requests.json --inspection $receipt --inspection-sha256 81cef97c719c3e2cb603a138e47b25e7f131f0acbb00538c6de539aa9136bea9 --jobs $jobs > $out/supervisor.log 2>&1"
tmux new-session -d -s dotl47_20261128 -c "$src" "$cmd"
tmux set-option -w -t dotl47_20261128 remain-on-exit on
echo LAUNCHED_TMUX dotl47_20261128
sleep 200
tmux ls 2>/dev/null || true
tail -n 30 "$out/supervisor.log" | cut -c1-300
cat "$base/jobs.json" 2>/dev/null || echo NO_JOBS_YET
ls -la /mnt/local/_gpu_guard/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
