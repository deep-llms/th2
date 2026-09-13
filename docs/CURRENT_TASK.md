# Current task

## Active request — launch Stage-2 matched continuation, 2026-09-13

User authorized five seed-17 arms sequentially, with artifact verification,
dev evaluation and original communicating burns afterward. Stage-1 is now
confirmed complete at 17:16:25 UTC (fresh child-folder handoff export), and
its burns were verified at 17:47:51. Earlier stale-status notes below are
historical, not current failures. Last runner status acknowledges `08b0f21`
successfully after a transient SSH failure; no new job has yet been pushed.

Stage-2 implementation/launch checks in progress. See `STAGE2_RUN_20260913.md`.
Canonical folder: `/disk/thuat/context_compiled_memory`; execution checkout:
`/tmp/th2-commands-only-20260911-q6dcOW`, main, origin deep-llms/th2.
No research-core modifications, old-output/cache deletion or locked-val use.
Stop only verified original burn workers AFTER the Stage-1 idle observer is
disarmed through its own `STOP_IDLE_WATCH` and its exit verified.

## Historical request — sequential Stage-1 arms, 2026-09-13

Completion check requested at approximately 17:45 UTC via commit `08b0f21`
(read-only #2, remote main confirmed). Repeated Dropbox checks over several
minutes returned unchanged status/export files: `_RUN_STATUS_.log` is still
modified at 16:23:11 UTC and does not acknowledge the new commit. Latest
workflow heartbeat remains 16:22:57. Completion, evaluation results and current
GPU/burn state are therefore UNVERIFIED, not failed. No explicit new AWS or
project error was returned. Do not relaunch, clean or signal anything to fix
this visibility issue. Execution head is `08b0f21`; evidence:
`temp/stage1_done_export_status_20260913_a04.log` and
`temp/stage1_status_20260913_1747.json` (stale heartbeat).

Latest check: read-only export `3612926`, heartbeat 16:22:57 UTC. Contextual,
Isolated, Shuffled, Shallow and Delta each completed 977 steps and passed
their full training gates (unchanged backbone/table, paired initialization).
Their subprocess wall times were 16.55 / 16.42 / 16.44 / 16.43 / 16.51 minutes,
excluding the following wait and validator. Grad is running on all eight GPUs,
at least step 250/977 (65536000 tokens), finite NLL/gradients, with roughly
14–16 minutes of training remaining at the observed rate. Evaluation has not
started. Evidence: `temp/stage1_status_20260913_1623.json` and
`temp/stage1_handoff_20260913_1623.log`. No workload was modified. Execution
head is now `3612926` (#2 only); older startup observations below are historical.

User authorized the next arms. Launch commit `18c89f6` submitted
`scripts/launch_pilot_stage1.sh` to th2; startup is now verified, not merely
pushed. Input gate passed at 14:47:58 UTC. Old observer exited before burn
workers 31343–31350 were stopped through verified pidfds. All eight GPUs
passed free checks at 14:49:00 and 14:49:30. Contextual launched at 14:49:30;
fresh export `dfe0c85` confirms 131 contiguous, finite updates / 34340864 tokens.
Its recorded contract is Stage-1, seed 17, bf16, world size 8, total 977 steps,
reader peak LR 5e-4 and the exact completed common checkpoint hash. GPU worker
PIDs are 32359–32366, one per GPU; latest heartbeat is 14:52:31 UTC.
Evidence: `temp/stage1_handoff_20260913_a01.log`,
`temp/stage1_workflow_20260913_a02.json`,
`temp/stage1_contextual_run_20260913_a02.json`, and
`temp/stage1_contextual_train_20260913_a02.jsonl`.
Plan: six Stage-1 arms, 977 steps each, eight GPUs sequentially; then Base +
six full dev evaluations, paired primary comparisons, and verified burns.
No Stage-2, locked-val use, new compiler, cleanup or optimizer resume.
See `STAGE1_RUN_20260913.md` for settings, paths and the observer stop contract.
New workflow root: `/mnt/local/_outputs/deep-llms_th2/ccm_stage1_seed17_20260913_a01`.
The old dev a04 monitor exited after observing Steps 1–3 complete; it no longer
owns execution Git. The new remote handoff records per-minute status itself.
79 local tests passed before deployment; `ccm/` core remains unchanged.
Execution head is `dfe0c85`; commands.sh is now a read-only #2 export, not a
second launch. No local hourly push monitor is active. The remote persistent
sequence owns subsequent arms/evaluation/gates/burn restoration; do not start
another GPU job while it runs. Future reclaim after completion must disarm
the Stage-1 root's `STOP_IDLE_WATCH`, not only the old completed workflow.

## Latest B200 completion — verified 2026-09-13, 14:17 UTC

Fresh read-only export `0bac963` confirms the original single-GPU compiler
exited 0 at 14:10:26 UTC after processing exactly 1,000,000,000 tokens.
Reported compilation wall time: 17849.15 seconds (4h57m29s), excluding process
startup. All five table artifacts (shallow/contextual/delta/isolated/shuffled)
passed the table gate at 14:11:48, with corpus/vocabulary/checkpoint identities
checked. All eight GPUs passed the free check at 14:12:20; the handoff then
verified burn startup and recorded `verified_steps_1_2_3_and_burns_active`,
`success: true`, at 14:13:01. Steps 1–3 are now complete; no reader training
has been launched by this status check.

Evidence: `temp/b200_workflow_fresh_20260913_1416.json` and
`temp/b200_handoff_fresh_20260913_1416.log`. The local a04 monitor below will
exit after retrieving this terminal state. The remote idle-burn observer
remains active: create its workflow-root `STOP_IDLE_WATCH` and verify it exits
before an authorized future job reclaims the GPUs. Do not stop anything for
a status request.

## Latest verified state — 2026-09-13, dev compiler benchmark

Step 1 finished and passed its final data gate at 05:43 UTC. Step 2 finished
at 09:09 UTC (15259 updates / 4000055296 tokens) and its model/optimizer/log
checks passed at 09:10 UTC. Compilation started at 09:12 UTC; the latest
observed B200 heartbeat at 12:45 UTC still showed it running on GPU 0, with
original communicating burns on GPUs 1-7.

User authorized a **dev-only speed benchmark**, not cancellation/replacement
of the B200 compiler. See `COMPILER_BENCHMARK_20260913.md` and
`tests/benchmark_compiler.py`: CPU/GPU accumulation, larger batches, length
grouping and four-GPU NCCL merging tested at full pilot dimensions. Measured
3.08x / 6.00x / 9.60x sample speedups for GPU accumulation / added grouping /
four GPUs respectively. These are A100 engineering timings, not B200 ETAs.
No `ccm/` production source, B200 workload or execution command was changed.
The existing local hourly monitor remains responsible for read-only #2 pulls.

At approximately 14:15 UTC the old local monitor (`ccm_pilot_monitor_20260913_a02`)
gracefully relinquished Git control via its local `STOP` file. Replacement
`ccm_pilot_monitor_20260913_a04` requests an immediate fresh snapshot, then
continues hourly for up to 12 checks (or stops on complete/failure). Reports
and its `STOP`/expected-head files are now under
`temp/pilot_monitor_20260913_a04/`; log:
`temp/pilot_monitor_20260913_a04.log`. This changes only dev-side snapshot
scheduling, not the B200 workflow or its GPU processes. Use this new monitor's
STOP file to relinquish Git control before any other execution push.

Follow-up equivalence audit completed on four dev A100s: fixed-state replay
and whole-original-batch distribution both reproduce all bf16 lookup tables
exactly on the sample, though raw master statistics have rounding differences.
Length grouping changes forward numerics and is NOT an identical-output
replacement (up to 0.7852% relative L2 mean error). Preserve original physical
batches for the conservative candidate. See the benchmark report's audit
section. No B200 job or production source was changed.

## Historical overnight launch and early observations

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
`temp/pilot_monitor_20260913_a02/`. Its first full snapshot is verified: four
downloaded files (including the sibling handoff log), local SHA256 checks passed.
Remote heartbeat 02:42:52 UTC: preparation 1896188056 tokens / 2396000 documents,
still waiting, all eight original GPUs 98% / 2510 MiB. Local tmux pane alive.
Evidence: `temp/pilot_monitor_20260913_a02/20260913T024223Z/report.json`.
The next hourly snapshot is due about 03:42 UTC. Latest execution head after
the monitor's first push: `fcc767c` (#2 only). Do not push over the monitor.
Create that local directory's `STOP` file to relinquish its Git control before
making another execution push. It also stops automatically on an unexpected
HEAD change. The local monitor only submits #2 and pulls small status/log files.
The first monitor's snapshot verified a fresh 02:37:51 UTC heartbeat and all
eight burns active. It was stopped locally to correct retrieval of the sibling
handoff log (exported in the parent Dropbox folder); no B200 task was stopped.
Local final suite: 75 tests passed; `temp/overnight_local_tests_20260913.log`.
These latest observation-only notes are local; no further execution push was
made because the hourly monitor now owns its expected-HEAD guard.
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
