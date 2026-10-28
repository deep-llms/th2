# Remaining Phase 0A preparation — 28 October 2026

The current NEM S1 queue keeps its frozen source and recipe. The new code is
`scripts/a2_remaining.py`; no numerical model, loss, Trainer, data order,
optimizer, selection rule, or GPU supervisor was changed.

Implemented and CPU tested:

- Import reviewed, SHA-pinned result/baseline/terminal receipts. Reject changed
  files, incomplete queues/burn handoffs, duplicate IDs, nonfinite metrics,
  incorrect eight-rank sample order, wrong LR schedules and failed save/reload.
  Historical provenance remains intact. An original baseline may predate a
  coordinator-only commit; runtime/data/spec identity and its loss must match.
- Select each parent's LR from its own complete three-point tuning grid. After
  all three S1 pairs succeed, subtract their exact 30 run IDs from the registered
  matrix: 28 fresh fits remain (18 short S2/S3/S4, six main S2, two NEM long S1,
  and two T14 repeats). No automatic retry import; an unexpected historical
  failed/retried fit requires explicit handling rather than silent retraining.
- Identify missing S72/S188 surveys. Never freeze S2 while a branch that passes
  pinned T3 identity is unmeasured. Exclude only T1/T3 failures, never diagnostic
  loss flags. Require both valid annealed parents. Reuse the original S2 function.
- Build the existing runner's prerequisite manifest: up to six eight-GPU branch
  surveys, CPU set selection, then one representative numerical job covering the
  union of frozen patterns. It has no heal or throughput jobs. CPU jobs omit the
  `gpus` key, as required by the proven runner.

Validation: all 58 A2 tests pass, including eight new tests. A local replay of
20 actual archived DCLM/FWE fit receipts passes all import checks. Test logs:
`temp/a2_remaining_tests_20261028_final.log`; actual-receipt replay helper and
local-only lock: `temp/a2_replay_history.py`, `temp/a2_known_history_local.json`.
The full 30-fit plan is tested with synthetic records; NEM final receipts do not
exist locally yet. Do not describe NEM or S2 as completed.

Before GPU execution, still required:

1. Retrieve NEM's successful final artifacts and independently verify burns.
2. Build/review a remote-path history lock from actual receipts, including the
   original DCLM/P300 standalone survey; pin all actual artifact hashes.
3. Connect the prerequisite manifest to the existing native launcher, verifying
   the six branch weight/code files, cumulative budget, copied Accelerate cache,
   fresh burn PID/start identities, free GPUs and automatic verified burn restore.
4. After S2/numerical checks, create the current-provenance preflight/baselines
   for remaining heals and use the exact pending list. The old generic queue
   still runs its full matrix; do not call it unchanged and repeat S1/grid fits.

The helper prepares/validates work, not a complete new training launcher.
Existing historical numerical byte/AST locks remain enforced. No new GPU work
was submitted in this turn, only a read-only health request.

At approximately 12:59 SGT, the runner reported an SSH hostname-resolution failure
for health submission8920863. All remote writes are frozen pending operator/system
recovery. This does not establish whether NEM training stopped. Keep outputs and
workloads intact; local code/tests may continue, and the existing watcher may
check read-only receipts. Do not resubmit commands as an infrastructure repair.
