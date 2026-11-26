#1 +150+a
#th2-ohmv-backup-copy-20261126-a01
set -u
date -u
# Backup before the B200 restart (user request): copy into /mnt/local/_backup/deep-llms_th2_20261126 in a detached
# tmux session. Pass 1: every run's small files (results, receipts, frozen sources, logs, row arrays; no .pt).
# Pass 2: the repo checkout. Pass 3: terminal checkpoints of the DOT / full-method line and its baselines, newest
# first, without in-run duplicate copies. Data is not copied (reproduction: docs/A2_DOT_DATA_REBUILD_20261118.md).
# Read-only on sources; nothing is deleted or moved; GPUs are not touched.
B=/mnt/local/_backup/deep-llms_th2_20261126
if [ -e "$B" ]; then echo BACKUP_DIR_EXISTS; tail -5 "$B/progress.log" 2>/dev/null; exit 0; fi
mkdir -p "$B"
cat > "$B/backup.sh" <<'SH'
set -u
B=/mnt/local/_backup/deep-llms_th2_20261126
O=/mnt/local/_outputs/deep-llms_th2
log() { echo "$(date -u +%FT%TZ) $*" >> "$B/progress.log"; }
log START
rsync -a --prune-empty-dirs --exclude='*.pt' --exclude='*.pt.*' --max-size=4g "$O/" "$B/outputs/"; log "PASS1_SMALL_FILES rc=$?"
rsync -a /mnt/local/deep-llms_th2/ "$B/repo/"; log "PASS2_REPO rc=$?"
for run in dot_f22_20261126 dot_f21_20261126 dot_f20_20261126 dot_f19_20261126 dot_f18b_20261126 dot_vs_inkling_20261126_f17 dot_ffn_control_20261126_f16 dot_f15_20261126 dot_t_ablation_20261126_f14 dot_vs_loops_2x_20261126_f13 dot_vs_dense8_2x_20261126_f12 dot_vs_dense8_20261126_f11 dot_fast_attn_20261125_f10 dot_narrow_tmlp_20261125_f09 dot_lr_high_20261125_f08 dot_lr_fairness_20261125_f07 dot_depth_scale_20261125_f06 dot_depth_scale_20261125_f05 dot_tmlp_20261125_f04 dot_shared_heads_20261125_f03 dot_compact_t_20261125_f02 dot_fixed_improve_20261125_f01 dot_t_loops_20261125_t01 dot_recipe_lm_20261125_a07 dot_stop_20261124_a06 dot_sandwich_20261124_a05 dot_sandwich_20261124_a04 dot_sandwich_20261123_a03 dot_sandwich_20261123_a02 dot_sandwich_20261123_a01 dot_headroom2_20261124_a01 dot_headroom_20261124_a01 dot_synth_20261125_s11 dot_synth_20261125_s10 dot_synth_20261124_s09 dot_synth_20261124_s08 dot_synth_20261124_s07 dot_synth_20261124_s06 dot_synth_20261124_s05 dot_synth_20261124_s04 dot_synth_20261124_s03 dot_synth_20261124_s02 dot_synth_20261124_s01 loop_screen_20261123_a01 loop_screen2_20261123_a01 loop_screen2_20261123_a02 loop_screen4_20261123_a01 loop_screen4_20261123_a02 loop_stop_20261123_a01 loop_stop2_20261123_a01 moe_loop_20261123_a01 moe_variants_20261123_a01 moe_dyn_20261123_a01 moe_dyn_20261123_a02 moe_scale_20261123_a01 sandwich_20261123_a01 router_screen_20261123_a01 translator_screen_20261123_a01 translator_screen_s1_20261123_a01 domain_screen_20261123_a01 scale_round_20261123_a01 anchor_loop_20261123_a01 anchor_long_20261122_a01; do
  [ -d "$O/$run" ] || { log "SKIP_MISSING $run"; continue; }
  rsync -a --prune-empty-dirs --exclude='backup/' --include='*/' --include='*.pt' --exclude='*' "$O/$run/" "$B/outputs/$run/"
  log "CKPT $run rc=$? bytes=$(du -sb "$B/outputs/$run" | cut -f1)"
done
(cd "$B/outputs" && find . -name '*.pt' -printf '%s %p\n' | sort -k2) > "$B/checkpoints_manifest.txt"
log "DONE checkpoints=$(wc -l < "$B/checkpoints_manifest.txt") total=$(du -sh "$B" | cut -f1)"
SH
command -v rsync >/dev/null || { echo NO_RSYNC; exit 1; }
tmux new-session -d -s backup_20261126 "bash $B/backup.sh"
sleep 120
echo '== progress'; cat "$B/progress.log"
du -sh "$B" 2>/dev/null
tmux ls 2>/dev/null | grep backup_20261126
