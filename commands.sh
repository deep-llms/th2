#1 +60+a
#th2-ohmv-anchor-long-20261122-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/anchor_long_20261122.tar.xz"
echo "8135a551dd8d819b8864487cfa50d2750e8a58e67491e9b898ffb737f7fed89f  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/anchor_long_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=/mnt/local/_outputs/deep-llms_th2/embedding_anchor_fit_20261122_a01/execution/source
cd "$PYTHONPATH"
exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_anchor_long.py" "$extra"
