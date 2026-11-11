#1 +60+a
#th2-wc69-rebuild-metadata-pack-20261111-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
hostname
date -u
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u scripts/a2_export_rebuild_metadata.py
