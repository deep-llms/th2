#1 +60+a
#th2-ccm-panel-backup-archive-20260918-a01
set -euo pipefail
date -u
hostname
echo '--- queue alive?'
tmux has-session -t ccm_scaleup28_panel_20260917_a01 2>/dev/null && echo PANEL_SESSION_PRESENT || echo PANEL_SESSION_ABSENT
tmux display-message -p -t ccm_scaleup28_panel_20260917_a01 '#{pane_dead}' 2>/dev/null || true
tail -c 700 /mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_panel_20260917_a01/status.json || true
echo
echo '--- archiving all small results (no weights, no token data)'
TASK_EXPORT=/mnt/local/_outputs/deep-llms_th2/ccm_panel_backup_20260918_a01
test ! -e "$TASK_EXPORT"
test ! -L "$TASK_EXPORT"
mkdir "$TASK_EXPORT"
cd /mnt/local/_outputs/deep-llms_th2
TASK_RC=0
tar --exclude='*.pt' --exclude='*.tokens' --exclude='*.sqlite' --exclude='*.sqlite-journal' --warning=no-file-changed -czf "$TASK_EXPORT/results.tar.gz" ccm_scaleup28_panel_20260917_a01 ccm_scaleup28_panel_20260917_a01.handoff.log ccm_scaleup28_phase_a2_20260916_a02 ccm_scaleup28_phase_a2_20260916_a02.handoff.log ccm_scaleup28_phase_a_20260915_a01 ccm_scaleup28_phase_a_20260915_a01.handoff.log || TASK_RC=$?
test "$TASK_RC" -le 1
cd "$TASK_EXPORT"
split -b 24m -d -a 3 results.tar.gz results.tar.gz.part
tar -tzf results.tar.gz > members.txt
sha256sum results.tar.gz results.tar.gz.part* > sha256.txt
ls -la
wc -l members.txt
echo CCM_PANEL_BACKUP_ARCHIVE_DONE
