#1 +30+a
#th2-joint-v2-guard-source-20260923-a01
set -euo pipefail
date -u
ls -la /mnt/local/_gpu_guard
sha256sum /mnt/local/_gpu_guard/polite_burn.py
sed -n '1,300p' /mnt/local/_gpu_guard/polite_burn.py
/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11 -u -m scripts.inspect_burn
