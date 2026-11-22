#1 +60+a
#th2-ohmv-anchor-loop-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/anchor_loop_20261123.tar.xz"
echo "48ee21df6cedcfd29dbe20fae284ba03aba7a175b774a476f35454538467d4ce  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/anchor_loop_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=/mnt/local/_outputs/deep-llms_th2/anchor_long_20261122_a01/execution/source
cd "$PYTHONPATH"
exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_anchor_long.py" "$extra" --loop
