# Current task

## Seed29 remaining replication authorized — 2026-09-14

User authorized compilation, all six Stage1 arms, required Stage2 arms,
full D_dev evaluation/comparisons, and original burns afterward in one queue.
Explicit clarification: apply the unchanged Delta inclusion thresholds
separately to seed29 (new `per_seed` policy), not the older seed17-only screen.
No common retraining, seed43, D_val or cleanup. See
[the queue protocol](SEED29_REPLICATION_20260914.md) for commands and guards.
Fresh root: `/mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01`.
Existing seed29 theta_4B is reused only after full validation. A3's exact
observer is disarmed cooperatively before verified burn-worker reclamation.
CPU tests: all 69 passed, including a tiny real seed29 Delta pipeline.
Launched in `2c035fa`; startup export `28562d1` verified locally.
Input/checkpoint/optimizer validation passed. A3 observer exited cooperatively;
only its verified burn-worker pidfds were stopped, followed by two 30s free checks.
Compilation started 21:22:07 UTC; latest retrieved heartbeat 21:24:09 UTC:
GPU0 compiler PID 68604; GPUs1–7 original burn workers 68078–68084 at 98%,
seven-rank NCCL rings connected. Stage1/Stage2 are queued, not finished.
Current owner is `ccm_replication_seed29_20260914_a01`; do not reclaim through
the historical A3 owner below. Use this queue's STOP_IDLE_WATCH after completion.
Local startup evidence: `temp/seed29_replication_live_a01/`.

## A3 complete, original all-eight burns active — 2026-09-14

A3 completed and was verified locally. All 13 result files, three archive
parts, original replay/segment alignment, and both original bin marginals passed.
See docs/MEMORY_DIAGNOSTICS_RESULTS_20260914.md, A3 section. Contextual beats
Base/Isolated/Shuffled in all 25 joint cells but loses to Grad in all 25.
V5 improves more than V1 against those three controls in each frequency band;
this is exploratory, coarse-bin evidence, not a causal variance effect.
No per-cell CIs/regression, new training, D_val, or seed29 replication.

Latest original all-eight burn heartbeat: 20:04:12 UTC, workers 65728–65735,
98% utilization on every GPU; one shared eight-rank NCCL communicator.
Current owner: tmux `ccm_a3_diagnostics_20260914_a01`,
script `scripts/pilot_a3_diagnostics.py`. Before a later authorized reclaim,
create its exact
`/mnt/local/_outputs/deep-llms_th2/ccm_a3_seed17_20260914_a01/STOP_IDLE_WATCH`,
verify voluntary observer exit, then follow GPU_SAFETY.md.
The older A1/A2 observer is now disarmed; do not target it again.
Canonical results: `outputs/a3_results_20260914_a01/results/`.
No additional experiment is queued; await user direction.

## A3 authorized and prepared — 2026-09-14

User requested A3 and an update to the existing diagnostic results document.
Five seed-17 Stage-2 checkpoints, normal evaluation on full D_dev only:
Contextual, Isolated, Shuffled, Grad, Base on GPUs 0–4. Shared compile-derived
frequency/Contextual-variance quintiles; per-key and per-segment cell records.
No training, frozen research-core change, D_val, or A1/A2 output modification.
62 execution-checkout CPU tests passed (dev-only Dropbox helper on PYTHONPATH).

Entry: `scripts/launch_a3_diagnostics.sh`; controller:
`scripts/pilot_a3_diagnostics.py`; research diagnostic:
`scripts/frequency_variance_diagnostics.py`.
Fresh run root: `/mnt/local/_outputs/deep-llms_th2/ccm_a3_seed17_20260914_a01`.
Must verify A1/A2 completion and heartbeat, disarm its exact observer, then
reclaim verified burn workers, two 30s/free checks, five two-batch smoke jobs,
then full evaluation. Keep an original burn group on GPUs 5–7. Reap all five
jobs and check free before restoring all-eight original burns, CPU summary and
result packaging. Final `complete.json` requires all gates and active burns.
Launch commit `0ae7e33` was pushed and acknowledged by the runner. The launch
log confirms frozen-input preflight passed at 19:44:28 UTC, then the exact
previous observer was asked to exit voluntarily. Read-only export `e01167d`
requests live status, five smoke reports and available completion/results.
Full remote completion was verified at 19:53:03 UTC: all five real-data smoke
and full jobs exited zero; the grid passed pairing/original-marginal checks.
Original all-eight burns were restored before CPU summary/packaging. Heartbeat
19:57:06 UTC confirms workers 65728–65735, 96–98% utilization, one per GPU;
the burn log verifies one shared eight-rank NCCL communicator.
Export `0b6ba32` retrieved completion/burn evidence but skipped the export
folder. Exact-file export `842181c` requests manifest and three <25MB parts.
Local full-result verification and report update are complete (see entry above).

## A1/A2 diagnostics completed and verified — 2026-09-14

See [the diagnostic results](MEMORY_DIAGNOSTICS_RESULTS_20260914.md).
All three Stage-2 seed-17 models passed full 20M-input-token D_dev diagnostics,
with zero normal-replay discrepancy. All 12 new metrics/segment file hashes,
ordered segment identities, original normal loss sums, target counts and
summary point estimates were verified locally; the Contextual overall
10,000-replicate bootstrap was recomputed with an exact match.

Normal → memory-off overall NLL:
Contextual 3.213114 → 3.216956; Isolated 3.214406 → 3.215295;
Grad 3.208273 → 3.217307. Every off-minus-normal and off-minus-Base overall
95% CI is positive. This shows inference-time dependency, not a causal
partition of training effects. Mean contribution/hidden norm ratios are
5.82%, 2.01%, 8.68%, respectively. A3 and seed-29 memory replication remain unrun.

Workflow completion: 19:07:25 UTC. Final exported heartbeat: 19:08:26 UTC,
all eight original burn workers 63356–63363, one per GPU, 98% utilization.
The burn log verifies all eight ranks initialized the same NCCL communicator.
Historical owner at A1/A2 completion (now disarmed): tmux `ccm_memory_diagnostics_20260914_a01`,
script `scripts/pilot_memory_diagnostics.py`. Before a later authorized
reclaim, disarm this workflow through its own
`/mnt/local/_outputs/deep-llms_th2/ccm_memory_diagnostics_seed17_20260914_a01/STOP_IDLE_WATCH`
and verify its exact voluntary exit, then follow GPU_SAFETY.md.
The previous seed-29 observer is already disarmed.

Canonical local results: `outputs/memory_diagnostics_20260914_a01/`.
Launch `f8aa4a4`; read-only exports `933bd16`, `333e104`, `47c264e`.
No training/checkpoint/cache/core changes or cleanup. Final handoff-log
download returned Dropbox HTTP 409; the earlier handoff snapshots, successful
complete marker, and final burn heartbeat are available and verified.
This retrieval limitation is not a diagnostic/job failure.

## A1/A2 diagnostics submitted — 2026-09-14

User authorized running the cheap diagnostics after reviewing the next-step
plan. Execution commit `f8aa4a4` adds external-hook memory-off/contribution-norm
diagnostics for completed seed-17 Stage-2 Contextual, Isolated and Grad on full
`D_dev`, plus paired document CIs. No new training, A3, or locked validation.
See `MEMORY_DIAGNOSTICS.md` for exact measurements, guards and output paths.
All 57 execution-checkout CPU tests passed with the dev-only Dropbox helper
on PYTHONPATH; an initial all-suite invocation lacked that local helper.
The frozen `ccm/` code hash is unchanged.

Submitted to `deep-llms/th2` from `/tmp/th2-commands-only-20260911-q6dcOW`.
Canonical copies of all new source/tests are in this project. Remote preflight
passed. Export `933bd16` confirms all three two-batch real-data smoke tests
passed, with exactly zero normal-replay discrepancy and finite norm statistics.
Full diagnostics launched on physical GPUs 0–2; heartbeat 18:58:22 UTC confirms
those three independent jobs and the original five-GPU burn group on 3–7.
Full completion is not yet verified. Output:
`/mnt/local/_outputs/deep-llms_th2/ccm_memory_diagnostics_seed17_20260914_a01`.
Three independent eval GPUs, five spare original burns; all-eight original
burns restored before CPU bootstrap and checked before success marker.

## Results summary written — 2026-09-14

`docs/PILOT_RESULTS_SUMMARY_20260914.md` consolidates verified seed17
Stage1/Stage2 results and seed29 common training: all 12 evaluations, six
saved CIs, coverage/hit/miss/gate/frequency/variance diagnostics, optimizer
parameter counts, runtime/memory, compilation accounting and limitations.
It distinguishes a favorable single-seed contextual-control result from
the incomplete multi-seed criterion, and documents Grad's better NLL.
No new evaluation, remote command or training was submitted for this summary.

## Results pulled and verified — 2026-09-14

All 310 result files are local under
`/disk/thuat/context_compiled_memory/outputs/pilot_results_20260914_a01/results/`.
The sibling README indexes the four run roots; download/ retains seven archive
parts and a SHA256 manifest. All part/archive/file hashes passed. All 12
evaluations match original remote validation gates, all 13 training logs have
the expected final step/token counts, and all six comparison reports exist.
Includes training/evaluation logs, per-segment metrics, reports, completion
gates, checkpoint JSON configs/metadata and compilation artifact metadata.
Excludes large model/optimizer/table binaries and data; not a checkpoint backup.
Packaging `da103ee`, read-only export `f7290cf`; no GPU signal/restart.
Remote packaging's final original all-eight burn ownership/health check passed.
Local verifier: `scripts/verify_result_bundle.py`; manifest:
`outputs/pilot_results_20260914_a01/download/manifest.json`.
Combined archive SHA256:
`35a345113fd344cd9023f16b2ff55a118f590813149e29d66d37a9f92f7ac44e`.
No new experiments launched. Commands are #2, not a packaging/training rerun.

## Verified completion — 2026-09-14

Fresh read-only export `9e2ac82` confirms the entire seed-17 Stage-2 workflow
completed successfully at 23:57:05 UTC Sep13, including its panel gate.
The queued handoff observed the same burn group for 181.63 seconds, passed
predecessor/data checks, verified the old observer exited at 00:01:27 UTC,
and reclaimed only verified burn workers before two 30s/free checks.

Seed-29 common training launched at 00:02:29 UTC Sep14, exited zero at
03:26:10 (3h23m41s training subprocess), passed final validation at 03:27:13,
and recorded completion with original all-eight-GPU burns at 03:27:54.
Verified 15259 updates / 4000055296 input tokens, seed29; model SHA256
`da85b43b1509f0f8b5e807f951b624126f17d673edd505b8f041d8413f104038`.
Fresh handoff heartbeat at 10:58:53 UTC confirms original burn workers
46137–46144, one per GPU, 98% utilization and 2510 MiB/device. No failure
events in the handoff. Do not stop them merely to retrieve results.

Seed29 compilation, reader adaptation and continuation have NOT run.
They require the next authorized workflow. Future reclaim must first disarm
the seed29 root's STOP_IDLE_WATCH and verify its exact observer exit.
Evidence: `temp/seed29_done_handoff_20260914.log`,
`temp/seed29_done_complete_20260914.json`,
`temp/seed29_done_validated_20260914.json`,
`temp/stage2_done_complete_20260914.json`,
`temp/stage2_done_validated_20260914.json`.
One parallel Dropbox status-file fetch returned HTTP429; completion and fresh
GPU health were verified from the successfully downloaded full handoff log.
Commands remain read-only #2, execution head
`9e2ac823cff760d5772645ac89af8ad67453fac7`. No new training or GPU action
was submitted for this completion check.

## New active request — queue seed-29 common pretraining, 2026-09-13

User explicitly authorized seed 29's first/common phase after the current
Stage-2 workflow completes, irrespective of scientific wins/losses. Require
the full workflow completion, original eight-GPU burns observed for three
minutes, Stage-2 observer STOP_IDLE_WATCH + verified exit, then verified burn
worker stopping and two 30s/free checks. Train independent seed-29 common
backbone for 15259 updates/4B, validate and restore burns. No compilation,
adaptation/continuation, seed43, locked-val or model changes in this queue.
Implementation complete: execution suite 47 passed; shell syntax passed.
Pushed launch commit `67cd8d27c7eb6d5df27cd32d41aae086d437da31`.
The earlier SSH hostname-resolution failure recovered through the runner;
no duplicate queue was submitted. Launch log confirms
`CCM_SEED29_QUEUE_ARMED` at 23:08 UTC. Fresh read-only export `8e99e11`
confirms a healthy queue heartbeat at 23:46:21 UTC, waiting for full Stage-2
completion with no failure or GPU action. Remote `commands.sh` is now #2,
not another launch. Current execution remote head:
`8e99e11e284c5c66b72dbd4302767822d040ce07`.
All five Stage-2 training arms exited zero and passed their training gates;
Grad finished at 23:25:23 and passed validation at 23:26:36 UTC. Base,
Contextual and Isolated dev evaluations passed; Shuffled eval is active
at 23:45:51. GPU0 runs eval; GPUs1–7 have original burn workers
41918–41924, 2508 MiB/device and 98% utilization in that snapshot.
Evidence: `temp/seed29_queue_launch_recovered.log`,
`temp/seed29_live_handoff_a01.log`, `temp/seed29_live_status_a01.json`,
`temp/stage2_live_seed29_queue_a01.log` and
`temp/stage2_live_seed29_queue_a01.json`.
No reminder is needed for the remote queue: it waits for full completion,
observes burns for 180s, disarms/verifies the predecessor observer's exit,
reclaims verified workers, performs two 30s/free checks, then trains only
seed29 common and restores burns after validation. Seed29 training has NOT
started in this latest observation. No separate dev-side auto-push monitor
is active. Next manual check should use fresh #2 exports, not re-run #1.
See `SEED29_QUEUE_20260913.md`.
The current Stage-2 job is NOT stopped, relaunched or edited by this request.

## Active request — launch Stage-2 matched continuation, 2026-09-13

Latest read-only progress export `04a4728`, heartbeat 20:27:42 UTC:
Base and Contextual both finished all 3815 updates and passed training gates.
Subprocess wall times: Base 56m27s; Contextual 58m25s (each excludes the
following 30s wait/validator). Isolated launched 20:25:41, at least step 30,
all eight ranks active; Shuffled and Grad remain queued. No training failure.
Estimated whole-sequence completion including dev evaluation and gates:
approximately 23:55 UTC Sep13–00:25 UTC Sep14, subject to Grad throughput.
Evidence: `temp/stage2_handoff_20260913_2028.log`,
`temp/stage2_status_20260913_2028.json`. Only #2 was pushed; no workload altered.
Current execution head: `04a4728ef949e06095c35718c7b45e8709097e2b`.
Earlier authorization required result review before replication. The new
request above now permits seed-29 common pretraining unconditionally on the
scientific result; further seed-29 phases, seed 43, locked-val use, Stage-3
online writing and 28-layer scale-up remain outside this queue.

Earlier launch observations follow.

User authorized five seed-17 arms sequentially, with artifact verification,
dev evaluation and original communicating burns afterward. Stage-1 is now
confirmed complete at 17:16:25 UTC (fresh child-folder handoff export), and
its burns were verified at 17:47:51. Earlier stale-status notes below are
historical, not current failures. Last runner status acknowledges `08b0f21`
successfully after a transient SSH failure.

Stage-2 launch `b8d1712b0d2f70b60b93faf4ad7c22da31d0e9c6` acknowledged.
Read-only export `ac6c0e2` verifies input gate success at 18:27:04, Stage-1
observer voluntary exit at 18:27:35, and stopping ONLY burn workers 35999–36006.
Both all-eight-GPU free checks passed at 18:28:06 and 18:28:36. Base-Continue
launched at 18:28:36 with the correct eight-rank Stage-2 command.
Fresh export `35f3377` confirms 105 contiguous finite optimizer updates,
27525120 input tokens, the correct LR schedule, bf16, seed 17, 3815 total
updates, unchanged common source hash and research-core hash. Latest heartbeat
18:31:37: launcher 37994, workers 38000–38007 (one per GPU), 13544–14156 MiB
device memory. Instantaneous utilization varies across ranks; all eight are
assigned to this DDP training job. Evidence:
`temp/stage2_handoff_20260913_a02.log`, `temp/stage2_status_20260913_a02.json`,
`temp/stage2_base_run_20260913_a02.json`, `temp/stage2_base_train_20260913_a02.jsonl`.
Execution head is `35f337796cad36117c86311249e1a4c3ef9d7c02`; commands.sh is
read-only #2, not another launch. No separate local monitor is active; remote
persistent Stage2 owns sequential training/eval/verification/burn restoration
and minute heartbeats. Future reclaim must disarm the Stage-2 root observer.
Do not repush #1, clean outputs/caches or relaunch an active job.
Canonical suite 86 passed; execution suite 39 passed (dev-only Dropbox-monitor
test excluded because its client is not deployed). No ccm bytes changed in
execution: `055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9`.
See `STAGE2_RUN_20260913.md`.
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
