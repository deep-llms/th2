#1 +30+a
#th2-readonly-gpu-guard-directory-20260911-a02
set -euo pipefail
date -u
hostname
id
ls -ld /mnt/local
if test -e /mnt/local/_gpu_guard || test -L /mnt/local/_gpu_guard; then
    ls -ld /mnt/local/_gpu_guard
    ls -la /mnt/local/_gpu_guard/
    find /mnt/local/_gpu_guard/ -maxdepth 2 -type f -printf '%p (%s bytes)\n'
else
    echo GPU_GUARD_DIRECTORY_ABSENT_ON_WORKER
fi
date -u
echo GPU_GUARD_READONLY_LOG_REVIEW_COMPLETE
