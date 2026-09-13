# Authorized overnight pilot

User approved a persistent script and monitoring on 2026-09-13. This is a
run-specific workflow on `thiennh-p6-8mgy-worker-0`, not a reusable launcher
for another node. Launch only once using `scripts/launch_pilot_overnight.sh`.
All research code remains unchanged from the tested, live preparation snapshot.

## Sequence and verification

1. While CPU preparation runs, check its exact PID/start time and GPU burn
   health every five minutes. Do not interrupt preparation or its burns.
2. Require the final preparation shell log marker, ended shell, completion
   report and GPU observation file. Independently verify full corpus hashes,
   budgets, Base assets, vocabulary and pre-reader development coverage.
3. Stop only verified original burn workers using pidfds, never PID 1, launchers,
   tmux servers, name matching or arbitrary groups. Wait 30 seconds/check free,
   then wait 30 seconds/check free again. Explicitly activate `train_env`.
4. Train the common model from scratch, seed 17, 12 Qwen3 layers, all eight
   B200s. Locked budget: 15259 updates x 262144 tokens = 4000055296 tokens.
   BF16, fp32 master AdamW, activation checkpointing, microbatch 8 and loss
   chunk 1024 match the tested settings. The actual runtime uses torchrun;
   no unrelated Accelerate configuration is copied.
5. Inspect live task/GPU snapshots every minute. After exit zero, wait/check
   free, validate final model/optimizer hashes, finite weights and all 15259
   training records/counters/LRs. Restore original burns and verify twice.
6. Run the existing single-GPU compiler on GPU 0; use original communicating
   burns on GPUs 1-7. Compile the full 1B-token role into shallow, contextual,
   delta, isolated and shuffled tables. No reader-training arms follow yet.
7. Require exit zero, exact compile token count, checked table artifacts,
   matched counts/provenance and shuffled/contextual consistency. Restore
   original all-eight-GPU burns and verify twice before publishing completion.

The common checkpoint is saved every 1000 updates and at update 15259,
including optimizer masters/moments. Resume is not implemented. Failures stop
the sequence without retrying or deleting anything. Burns may be restored only
after proving the corresponding GPUs free; a live child or unknown ownership
is preserved and reported, never killed as recovery.

## Paths and sessions

- Data: `/mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01`.
- Preparation reports: `/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01`.
- Workflow root: `/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01`.
- Persistent workflow tmux: `ccm_pilot_overnight_20260913_a01`.
- Its log is the workflow root path plus `.handoff.log` (a sibling file).
- `status.json`: fresh heartbeat with stage, UTC, task log tail and GPU snapshot.
- `common/checkpoint-15259`, `validated_common.json`: verified common output.
- `tables/`, `validated_tables.json`: verified compiler output.
- `complete.json`: Steps 1-3 verified and all-eight burns verified at that time.
  Current burn liveness still requires a fresh heartbeat, not this old marker.
- Burns: runner's original `/tmp/llm_pretrain_burn.py`, SHA256
  `3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857`.
  Low memory (~2.5 GiB observed), bf16 GEMMs and a scalar NCCL all-reduce each
  loop. It is not the enhanced resource file. Burns use independent tmux
  sessions `ccm_pilot_20260913_a01_bN` and separate rendezvous ports.

## Before starting the next GPU job

After successful completion, the supervisor keeps checking burns every five
minutes and can restart an entirely free burn group. **Disarm this observer
first** by creating the exact workflow-root file `STOP_IDLE_WATCH`, then wait
for its tmux pane to exit (at most five minutes), verify it exited, and only
then stop verified burn workers. This prevents a new job racing a burn restart.
Do not kill the tmux server or issue a broad process-name kill.

## Dev-side monitoring

Use `#2` to export small logs/state, then the dev-only Dropbox helper. Never
upload/download directly from B200. A local hourly monitor can run in tmux,
but must stop if another user/agent changes the execution Git head, or if the
runner reports infrastructure errors. It must never resubmit `#1`, kill jobs,
or rewrite the scientific source. Persist observations under ignored `temp/`.
The dev monitor is `scripts/monitor_pilot_local.py`, local tmux session
`ccm_pilot_monitor_20260913_a01`. Reports go to
`temp/pilot_monitor_20260913_a01/`; it exports only the existing workflow
status/handoff log and preparation log, not model weights. Create `STOP` in
that local monitor directory before making another execution push; also verify
the monitor exits. Unexpected Git HEAD changes make it stop, not overwrite work.
An interactive assistant is not guaranteed to wake after the chat ends; the
persistent scripts provide the unattended behavior.
