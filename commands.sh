#1 +20+a
#th2-wc69-backup-fill-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
CONTROL=/mnt/local/_outputs/deep-llms_th2/wc69-preservation-control-20261111-a01
test ! -e "$CONTROL"
test ! -e /mnt/local/_backup/deep-llms_th2/wc69-preservation-20261111-a01
mkdir "$CONTROL"
cp scripts/a2_preserve_wc69.py "$CONTROL/copy.py"
cat > "$CONTROL/run.sh" <<'RUN'
#!/bin/bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
cd /mnt/local/_outputs/deep-llms_th2/wc69-preservation-control-20261111-a01
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u copy.py > copy.log 2>&1
rc=$?
printf '{"returncode":%s}\n' "$rc" > exit.json
exit "$rc"
RUN
tmux new-session -d -s wc69_backup_fill_20261111_a01 "bash $CONTROL/run.sh"
echo BACKUP_COPY_SUBMITTED
sleep 5
tail -n 5 "$CONTROL/copy.log"
