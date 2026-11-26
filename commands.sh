#1 +20+a
#th2-ohmv-backup-check-20261126-a01
set -u
date -u
# Read-only: did the pre-restart backup finish?
B=/mnt/local/_backup/deep-llms_th2_20261126
tail -50 "$B/progress.log" 2>&1
wc -l "$B/checkpoints_manifest.txt" 2>&1
du -sh "$B" 2>&1
tmux ls 2>&1 | head -5
