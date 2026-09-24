#1 +30+a
#th2-joint-v2-collect-verify-burn-20260924-a01
set -euo pipefail
date -u
PYTHONPATH="$PWD" /usr/bin/python3 -u -m scripts.collect_joint_results --root /mnt/local/_outputs/@PROJECT@/joint-v2-b200-20260923-a03 --output /mnt/local/_outputs/@PROJECT@/joint-v2-results-export-20260924-a01
