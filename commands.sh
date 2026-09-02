#1 +60+a
#th2-readonly-inspect-mnt-local-data-20260902-a14
set -euo pipefail

echo '=== exact read-only inspection of /mnt/local/_data ==='
date -u
hostname

echo '=== parent directory ==='
ls -lad /mnt/local /mnt/local/_data 2>&1 || true
stat /mnt/local/_data 2>&1 || true
df -hT /mnt/local /mnt/local/_data 2>&1 || true

if [[ -d /mnt/local/_data ]]; then
    echo 'DATA_EXISTS=yes'
    echo '=== immediate contents ==='
    find /mnt/local/_data -mindepth 1 -maxdepth 1 -printf '%y %s %p\n' \
        | LC_ALL=C sort
    echo '=== contents through depth 6 ==='
    find /mnt/local/_data -mindepth 1 -maxdepth 6 -printf '%y %s %p\n' \
        | LC_ALL=C sort | sed -n '1,4000p'
    echo '=== total size ==='
    du -sh /mnt/local/_data
    echo '=== immediate child sizes ==='
    du -sh /mnt/local/_data/* 2>/dev/null | LC_ALL=C sort || true
else
    echo 'DATA_EXISTS=no'
fi

echo '=== GPU state unchanged ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 READONLY DATA INSPECTION COMPLETE'
