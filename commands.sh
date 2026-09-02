#1 +60+a
#th2-readonly-inspect-mnt-local-outputs-20260902-a13
set -euo pipefail

echo '=== exact read-only inspection of /mnt/local/_outputs ==='
date -u
hostname

echo '=== parent directory ==='
ls -lad /mnt/local /mnt/local/_outputs 2>&1 || true
stat /mnt/local/_outputs 2>&1 || true
df -hT /mnt/local /mnt/local/_outputs 2>&1 || true

if [[ -d /mnt/local/_outputs ]]; then
    echo 'OUTPUTS_EXISTS=yes'
    echo '=== immediate contents ==='
    find /mnt/local/_outputs -mindepth 1 -maxdepth 1 -printf '%y %p\n' \
        | LC_ALL=C sort
    echo '=== contents through depth 3 ==='
    find /mnt/local/_outputs -mindepth 1 -maxdepth 3 -printf '%y %s %p\n' \
        | LC_ALL=C sort | sed -n '1,2000p'
    echo '=== total size ==='
    du -sh /mnt/local/_outputs
else
    echo 'OUTPUTS_EXISTS=no'
fi

echo '=== GPU state unchanged ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 READONLY OUTPUTS INSPECTION COMPLETE'
