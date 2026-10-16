# CE7454 overnight evaluation and monitoring

User authorized evaluation on both dev and official test, concurrent with the
existing eight B200 fits. Do not stop training to evaluate. No new architecture
fits are queued by this watcher. Deadline:17October2026 15:55SGT (07:55UTC).

## Two independent components

- B200 evaluator: `q359-ce7454-interim-watch-20261016-a01`, evaluates the latest
  available checkpoint from every arm once per30minutes, beginning immediately.
  GPU0 only, one evaluator at a time, batch8,10% allocator-memory ceiling.
  It never signals trainers, writes their checkpoints, or changes their schedule.
- Dev-machine timer: tmux `ce7454_agent_watch_20261016`, script
  `scripts/ce7454_resume_watch.py`, resumes the existing conversation
  `01a120fe-ea0e-7030-a337-dd3f52f3c625` via Codex CLI. First scheduled tick
  19:45:18UTC16Oct /03:45:18SGT17Oct, then30minutes; deadline06:30UTC17Oct.
  Local state/prompt/logs: `temp/ce7454_resume_watch_20261016/` (ignored).
  Timer PID2384904 at installation. Verify PID identity before management.
  The timer was verified waiting; the first actual CLI resume was not yet
  observed at installation. It depends on this dev machine, CLI authentication,
  and model availability. No promise of a phone notification.

## Verified smoke

Commit `a6bf32b`, 19:10:30–19:10:56UTC16Oct. All8 trainer PIDs remained the
same; no inference gradients or model/BatchNorm state changes. Immutable
checkpoint copies are evaluated; stable stat/read/load rejects a concurrent
in-place checkpoint write. Snapshot hashes are recorded.

| Arm | Checkpoint | Dev accuracy | Official test accuracy | Inference time | Peak PyTorch allocated |
|---|---|---:|---:|---:|---:|
| GRIP | base256 epoch296 |95.78%|94.20%|9.75s|1.72GB|
| Haar strong | native512 epoch182 |96.67%|90.33%|5.58s|0.41GB|

All900dev and1500test images were scored by each. These are different stages,
not final or matched-budget results. Dev labels are estimates; official test
labels were supplied by the user. Test metrics are reported separately and
must not determine training changes. Source-holdout accuracy is not dev accuracy.
Small temporary throughput interference is expected when sharing GPU0; zero
runtime overhead is not claimed. Training weights/optimizer are not accessed.

## Remote paths and handoff

Root prefix: `/mnt/local/_outputs/deep-llms_th2/`.

- Original study: `q359-ce7454-scratch-20261016-a04/supervised/run/study/`;
  four `*_base`450epoch fits, then their own40epoch `*_full512` stages.
- New study: `q359-ce7454-band-20261016-a01/study/`; four450epoch native512 fits.
- Periodic watcher: `q359-ce7454-interim-watch-20261016-a01/watch.log`,
  `watch/latest_status.json`, `watch/owner.json`, `watch/released.json`.
- Each `watch/cycle_NNN_ARM/` has immutable checkpoint, complete dev/test JSON
  predictions, and `result.json`. Preserve these; do not export all weights
  merely to monitor. Compact `INTERIM_RESULT` lines include predictions in
  exact frozen manifest order and the checkpoint SHA. `WATCH_STATUS` logs
  all arm progress and GPU occupancy.
- Original supervisor owns the eight-GPU guard and final burn. The future
  original evaluation gate requests watcher STOP and waits up to30seconds for
  `own_evaluator_exited:true` before its normal final evaluation proceeds.
  The existing companion4–7 release handshake remains. The watcher also stops
  when original ownership disappears or at02:00UTC17Oct.
- Watcher stop at first original final evaluation is expected; do not restart
  it blindly. The normal final evaluators still predict both dev and test.
- Installation commit `0a50083` verified19:15:19UTC, watcher PID1271652;
  gate hash `a6e35ed1a516677b2fa3534fff21449fc5f032dc500e6a840578acf05e4aa0c3`.
  First periodic GRIP epoch306 predictions completed, all8 trainers advanced.
  Receipt: docs/results/ce7454_interim_watch_install_20261016.json.

## Each scheduled review

1. Read current task and latest user direction. Check both worktrees/remote head;
   preserve unrelated dirt and do not overwrite an unacknowledged command.
2. Read Dropbox q359, using ignored `temp/dropbox_q359_folders.txt`. If needed,
   submit a unique read-only `#1` printing the watcher log, current training
   logs, final result status and supervisor/burn receipt, or `#2 -f-` exporting
   those exact small files. Never repush the executable command unchanged.
3. Score downloaded compact results locally:

```bash
python3 ce7454/src/score_interim_log.py \
  --log temp/remote_logs/UNIQUE.log \
  --manifest temp/ce7454_eval_upload/evaluation_manifest.json \
  --dev-labels /disk/thuat/CE7454/cache/dev_labels_best_estimate.json \
  --test-labels /disk/thuat/CE7454/cache/test_eval_labels.json \
  --output temp/ce7454_eval_upload/UNIQUE-scores.json
```

4. Save small verified result summaries and update this document/current task.
   Assess dev trajectories and remaining time. Do not prune from a single early
   checkpoint, compare unmatched stages as final results, or optimize against
   test accuracy. Fix only demonstrated evaluation bugs without altering training.
5. Verify final per-arm outputs and burn restoration before declaring completion.
   Set commands#0 after acknowledged jobs. Stop the local timer by creating its
   ignored `STOP` file once everything is verified; never kill arbitrary processes.

No uploads or outbound network from GPU code. Labels remain local. Do not touch
other agents' CE7454 local GPU experiments or publish a contest submission.
