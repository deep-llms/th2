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
3. The prerequisite manifest is now connected to the existing native launcher
   (`--stage remaining-prereqs --history LOCK`). It verifies branch weights/code,
   current native provenance, cumulative worst-case timeouts plus cleanup reserve,
   then reuses Accelerate copying, fresh burn identities/free-GPU checks and
   native automatic burn restoration. Actual GPU execution awaits NEM handoff.
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

Recovery addendum, 13:26 SGT: the same pending health8920863 subsequently ran
successfully (snapshot13:08, receipt13:11); no retry was submitted here. NEM
continued through five grid fits. Remote-write freeze is lifted on this evidence.
The new launch gate has CPU tests for exact required assets, completed-history
dependencies and budget rejection. Its first actual GPU prerequisite stage is
still pending; numerical core and the running NEM frozen source are unchanged.

Validation after launcher integration: 59 A2 tests pass; actual20-receipt replay
still passes. Log: temp/a2_prerequisite_tests_20261028.log. No GPU execution
claim for this new stage. Numerical byte/AST locks still pass.

Local continuation preparation, 14:00 SGT: scripts/a2_remaining_heals.py now
implements exact pending-fit execution after a current all-pattern gate/baselines.
It subtracts the30 prior fits, keeps each dataset/parent's selected LR, preserves
T14 permutation suffix through the single allowed half-LR retry, rejects infra
failures, checks every successful result's coverage/schedule/baseline/roundtrip,
and reserves600seconds for handoff before the Phase0A cap. It does not acquire
GPUs or launch by import. Four new CPU tests exercise all28 pending runs, explicit
subsets, retry identity, caps, non-divergence failures and malformed receipts.
Actual full-pattern gate/baseline assembly and launcher connection for these heals
remain pending; the prerequisite stage is separately connected and ready to
validate real history. Do not describe this executor as already running.

Final preparation:64A2tests pass and all30actualcompletedfits+6annealedsurveys
replay successfully. The original DCLM/P300survey predates the launch-scope
amendment/data-spec metadata field: recognize only its exact reviewed artifact
SHA, verify both historical byte/ASTlocks, and check the approved spec transition
without rewriting originalprovenance. Changed survey values are rejected by test.
Node-path history lock: resources/a2_completed_s1_20261028.json.

14:34 SGT integration: `scripts/a2_heal_stage.py` verifies reviewed successful
prerequisite hashes, rechecks BI selection, requires identical numerical
code_files/runtime/data for check reuse, preserves the original check in its
envelope, verifies historical T8/recipe evidence, scores six native all-pattern
baselines, then invokes only the28 pending fits. It is connected as launcher
`--stage remaining-heals --prerequisites LOCK` (plus existing history/prior/grid).
No old matrix/grid fits are repeated. Worst-case original-fit timeouts plus
baselines/cleanup must fit remaining18h Phase0A cap; each retry is separately
budget checked. T15 diagnostics are recorded and never change eligibility.
70 CPU tests pass, including numerical-provenance/tampered-receipt rejection,
all-six-baseline CLI ordering, registered28-run execution and retry/T14 checks.
Actual prerequisite final receipt and GPU execution remain pending; current
branch job keeps its frozen source602fbce. Log temp/a2_heal_stage_tests.log.


At14:37SGT the first branch stage had six successful surveys but failed in CPU
S2 assembly: the exact original DCLM/P300 survey predates parent_sanity metadata.
The fix accepts that omission ONLY for its original locked SHA; pinned history
already requires a successful parent pre-baseline, and healing requires it again.
It never marks arbitrary missing sanity metadata as passed. Actual30fit/12survey
replay now selects S2=[1,2,3] DCLM/FWE, [1,2,5] NEM. All71CPUtests pass.
Original failed supervisor and successful survey receipts are retained under
results/a2_branches_20261028/final_surveys. Recovery reuses all six surveys through
resources/a2_completed_surveys_20261028.json and executes only CPU freeze + GPU
numericalcheck. No numericalcore or scientific selection rule changed.
