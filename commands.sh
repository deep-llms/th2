#1 +40+a
#th2-pq5r-dot-data-prep-20261128-a02
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
date -u
hostname
out=/mnt/local/_outputs/deep-llms_th2
src=$out/dot_src_20261128
prep=$out/dot_prep_20261128
data=/mnt/local/_data/deep-llms_th2/dot_data_20261128
if [ -e "$src" ] || [ -e "$prep" ]; then echo PREP_ALREADY_LAUNCHED; cat "$prep/status.json" 2>/dev/null || true; exit 0; fi
pkg="$PWD/resources/dot_src_20261128.tar.xz"
echo "1545a267d990db65efe5a138caca59d11918550a80e33bbf5b295da3a3d044e0  $pkg" | sha256sum -c -
PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
test -x "$PY"
test -x /mnt/local/conda-py311/envs/eval/bin/python3.11
mkdir -p "$src" "$prep"
tar -xJf "$pkg" -C "$src"
cmd="CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/node_dot_data_prep.py --root $data --out $prep --workers 24 --python $PY > $prep/driver.log 2>&1"
if command -v tmux >/dev/null; then
  tmux new-session -d -s dotprep20261128 -c "$src/src" "$cmd"
  echo LAUNCHED_TMUX dotprep20261128
else
  (cd "$src/src" && setsid nohup bash -c "$cmd" </dev/null >/dev/null 2>&1 &)
  echo LAUNCHED_SETSID
fi
sleep 25
tmux ls 2>/dev/null || true
cat "$prep/status.json" 2>/dev/null || echo NO_STATUS_YET
tail -5 "$prep/driver.log" 2>/dev/null || true
ls -la "$data" "$data/tokenizer" 2>&1 | head -20
ls -la "$data/raw/dclm" 2>&1 | head -8
find "$data/downstream" -maxdepth 4 2>&1 | head -20
