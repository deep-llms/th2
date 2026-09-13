# Current task

Status: full pilot data preparation started on 2026-09-13 at 01:36:51 UTC.
New authority: user approved unattended Step 1 -> Step 2 common training ->
Step 3 offline table compilation, plus persistent burns and monitoring.
Handoff commit `0750f56` is deployed and ARMED: verified from the B200 log at
02:32:50 UTC, persistent tmux alive, waiting for preparation. All eight original
burn workers 18141-18148 still at 98% utilization / 2510 MiB. Preparation had
scanned 2026000 documents and written 1602396747 / 5040134656 tokens (31.79%).
Evidence: `temp/overnight_armed_20260913.log`. Training has not started yet.
Local hourly monitor: `scripts/monitor_pilot_local.py`, tmux session
`ccm_pilot_monitor_20260913_a02`, reports under
`temp/pilot_monitor_20260913_a02/` (verify its first snapshot before assuming active).
Create that local directory's `STOP` file to relinquish its Git control before
making another execution push. It also stops automatically on an unexpected
HEAD change. The local monitor only submits #2 and pulls small status/log files.
The first monitor's snapshot verified a fresh 02:37:51 UTC heartbeat and all
eight burns active. It was stopped locally to correct retrieval of the sibling
handoff log (exported in the parent Dropbox folder); no B200 task was stopped.
Remote preflight passed; full preparation/validation is not complete yet.
Prior B200 smoke passed; see `B200_SMOKE_20260913.md`.

## Active step 1 — full scientific data, not training

- User authorized prepare/validate full data, build vocabulary and verify coverage.
- Execution commit: `a4facbd`, th2 main; job
  `th2-ccm-prepare-full-pilot-20260913-a01`.
- Canonical launch script: `scripts/prepare_pilot_data.sh`, called by `commands.sh`.
- CPU only (`CUDA_VISIBLE_DEVICES` empty), `train_env`; do not stop GPU burns.
- Expected node: `thiennh-p6-8mgy-worker-0`; script refuses a different hostname.
- Data root: `/mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01`.
- Reports/logs: `/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01`.
- Uses existing verified English raw Parquet and pinned Base assets; no download,
  old-split reuse, deletion, architecture change, training or model compilation.
- Uses the default locked PILOT budget, no `--engineering` or budget override.
  Common 4000055296 tokens (compile 1000000000 and adapt 256114688 included),
  continuation 1000079360 tokens, dev/val 20000000 each.
- Sequential stages: prepare → validate-data → vocabulary (262144 keys from
  compile only) → dev coverage → final checksum/budget/batch validation.
- Report `complete.json` appears only after the final validator passes, including
  eligible dev hit rate >= 20%. A failed guard must not trigger capacity changes.
- Outputs must be fresh; do not automatically delete or retry a partial run.
- Preserve the deployed `ccm/` source while preparation runs. Its byte hash is
  verified again at completion. The later overnight authorization below permits
  common training only after preparation passes all gates.
- Startup evidence: `temp/step1_start_20260913_a01.log`, pulled from the job's
  01:38:01 UTC snapshot. Correct node, `train_env`, pinned Base asset hashes,
  full quotas and available disk passed preflight. All eight GPUs showed
  98% utilization / 2510 MiB at startup; this CPU job did not stop their work.
- Latest control file is the read-only burn inspection `04fabf8`, not another
  preparation/training launch. For preparation progress, request its specific
  log or `/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01/prepare.log`;
  the runner's latest job log now describes GPU inspection.
- Progress verified from the 01:40:00 UTC snapshot: 49000 source documents
  scanned, 38839996 tokens written across the six roles. This is past the
  mandatory source-hash verification and actively tokenizing; no traceback in
  the snapshot. Local log: `temp/step1_progress_20260913_a01.log`.
- Progress-request commit: `cad99b5`. Full preparation, vocabulary and coverage
  remain pending; do not describe the dataset as complete yet.
- Latest snapshot (2026-09-13 01:56:52 UTC), requested by `1700ea1`: 677000
  documents, 535896620 / 5040134656 tokens written (10.63%). No traceback or
  error signature; still in preparation. Local log:
  `temp/step1_progress_20260913_a02.log`. Since the 01:40 snapshot, observed
  rate is approximately 491k tokens/s. Linear tokenization ETA is 04:30 UTC
  (about 2.55 hours after the snapshot), not an ETA for all of step 1: vocabulary
  counting, coverage and final validation still follow and are not yet timed at
  full scale. Do not claim completion or a firm full-pipeline ETA.
- Burns reverified on 2026-09-13 at 01:59:47–01:59:59 UTC, two samples ten
  seconds apart: all eight GPUs 98% utilization / 2510 MiB, workers 18141–18148
  under original `/tmp/llm_pretrain_burn.py` launcher 18072; script checksum
  matched and tmux pane was alive. No processes signaled; preparation untouched.
  Local evidence: `temp/burn_check_20260913_0200.log`.

## Previous smoke scope and evidence

- Objective: implement and carefully review the frozen Paper-2 offline pilot.
- User authorized review and, if needed, safely stopping B200 burns for testing.
- The prior turn authorized real-data verification and a B200 performance smoke.
- Downloaded only pinned Base config/tokenizer via controller `#d`; no model weights.
- Verified B200's 50 English Parquet hashes against the pinned official manifest.
- Created a fresh 2,752,512-token engineering corpus; old data is untouched.
- The handoff safely stopped the exact original burn workers, performed two
  free-GPU checks, completed the payload, verified all GPUs free, and restored
  the same original burn script in a persistent tmux session.
- Preserve the research specification and existing machine workloads.

## Configuration and evidence

- Development directory: `/disk/thuat/context_compiled_memory` (not yet a Git repository).
- Execution checkout: `/tmp/th2-commands-only-20260911-q6dcOW`, `origin` =
  `git@github-share:deep-llms/th2.git`, branch `main`; shared code remains here too.
- Submitted smoke commit: `3e1d642`; subsequent `#2` commits retrieve logs only.
- Node: `thiennh-p6-8mgy-worker-0`, eight B200s; environment `train_env`,
  PyTorch 2.14.0+cu130, Transformers 5.9.0. No pytest required for the smoke.
- Data: `/mnt/local/_data/deep-llms_th2/ccm/smoke_20260913_a01`.
- Outputs: `/mnt/local/_outputs/deep-llms_th2/ccm_smoke_20260913_a01`.
- Smoke entry: `scripts/ccm_smoke_handoff.sh`; real-data payload:
  `scripts/ccm_smoke_payload.sh`. Uses torchrun, not Accelerate.
- All completion checks passed: payload exit zero, `payload_verified.json`, five
  full-capacity systems cases, eight tiny CUDA pipelines and persistent burns.
- Local results: `temp/b200_smoke_results_20260913/`, all archive/member hashes
  verified. Latest burn observation: 01:00 UTC, all eight at 98% utilization,
  2510 MiB/device; session `ccm_burn_20260913_a01`, launcher 18072.

## Authorized overnight handoff — 2026-09-13

See `OVERNIGHT_PILOT_20260913.md` for the exact sequence and disarm instructions.
Canonical scripts: `scripts/launch_pilot_overnight.sh`, `pilot_overnight.py`,
`pilot_gpu_ops.py`, `validate_pilot_handoff.py`.

- Preserve `ccm/` bytes while preparation is running.
- Require preparation's terminal log marker, ended exact shell identity,
  complete artifacts, checksums, pinned assets, full quotas and coverage guard.
- Train the seed-17 common memory-free Qwen3-12L model: 15259 updates,
  4000055296 tokens, all eight B200s, bf16, microbatch 8, loss chunk 1024,
  activation checkpointing and fp32 master AdamW. This is 4B, not 5B.
- After verified training, compile all five offline tables on GPU 0 while
  the original communicating burns occupy GPUs 1-7. No reader adaptation or
  Stage-2 arm training is authorized by this handoff.
- Output root: `/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01`.
- No automatic retry/resume/cleanup. Save model + optimizer every 1000 updates
  and at 15259. Exact resume is not implemented; do not claim otherwise.
- Restore original all-eight-GPU communicating burns after verified completion.
  The persistent idle observer must be disarmed before reclaiming GPUs for a
  subsequent job (see the overnight guide).

## Next action

Check the current step-1 log via Dropbox; request fresh snapshots with `#2`,
never by re-pushing the executable `#1`. Verify all stage results and pull the
small completion/coverage/manifest artifacts when available. Report runner/AWS
errors to the user without resubmitting or cleaning. The overnight handoff is
the only authorized automatic follow-on; verify it is armed. Do not reuse the
smoke's run-specific burn-stop helper.

## Handoff

`ccm/` contains the research implementation. This folder is not yet a Git
repository; the approved source snapshot and execution history are on th2.
Keep source changes synchronized deliberately; never push an old executable
`commands.sh` merely to refresh a log.
