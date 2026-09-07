#1 +30+a
#th2-readonly-machine-identity-ls-20260907-a01
set -euo pipefail
date -u
hostname
uptime
if [ -r /proc/sys/kernel/random/boot_id ]; then
    printf 'boot_id: '
    cat /proc/sys/kernel/random/boot_id
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv || true
fi
for TASK_PATH in \
    /mnt/local \
    /mnt/local/_data \
    /mnt/local/_data/@PROJECT@ \
    /mnt/local/_models/@PROJECT@ \
    /mnt/local/_outputs/@PROJECT@ \
    /mnt/local/conda-py311/envs; do
    printf '\nPATH: %s\n' "$TASK_PATH"
    if [ -d "$TASK_PATH" ]; then
        ls -lah -- "$TASK_PATH"
    else
        printf 'MISSING: %s\n' "$TASK_PATH"
    fi
done
echo 'TH2 READONLY MACHINE IDENTITY LS COMPLETE'
