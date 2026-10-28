# A2 continuation and wake-up plan — 28 October 2026 SGT

The owner asks to continue real experiments, estimate completion and schedule a
worker to wake this same conversation to inspect/continue authorized experiments.
Active design: A2_pilot_spec.md v2.1. This execution authorization extends the
completed first-grid boundary; it changes no scientific recipe or decision rule.

## Immediate bounded stage

Run DCLM/P1T survey and unhealed S1 baseline, then its three registered 114-update
LR-grid fits. Reuse the completed DCLM/P300 grid and selected LR3e-4. Run both
parents on S1 for 190 updates (0.05B) and 762 updates (0.2B), using each parent's
tuning-selected LR. Seven new fits, plus only the specified one half-LR retry
per diverged fit. Each budget starts from original parent weights, fresh optimizer
and its own full schedule. No grid checkpoint continuation. Keep old outputs.

The actual S1 numerical receipt, P300 baseline, T8 and original data are reused
with exact artifact SHA locks, unchanged numerical source/definitions and runtime
checks. An explicit envelope retains the entire original receipt/provenance;
it does not claim a new numerical execution. No new removal patterns in this stage.
P1T identity/file hashes, finite forward, RoPE and FP32/AMP precision remain checked.
Only orchestration and execution eligibility change. Shared model/loss/Trainer,
packing/order, optimizer, seeds and distributed runtime remain unchanged.

Measured fits: ~1.2 seconds/update + ~95 seconds loading/scoring/saving per fit.
Expected total 65–80 minutes from queue start, including P1T survey and baseline;
retry adds roughly 4–20 minutes depending on budget. Bounded coordinator hard cap
3 hours, reserving normal supervised cleanup. Charge all elapsed work to Phase0A,
including prior322+833.47seconds and new tests/scoring. This is not the full pilot.

Launcher: scripts/launch_a2_preflight.py --stage s1-pair.
Coordinator: efficient_llm/a2/pair.py. CPU validation: existing43 A2 tests and5new
scope/reuse/terminal-contract tests pass. A sixth test simulates the full coordinator
and verifies exactly seven new fits, selected LR and no repeated P300grid.
No training implementation changed.
Root: /mnt/local/_outputs/deep-llms_th2/6y3o-a2-s1-pair-20261028-a01.
Session: a2_s1_pair_20261028_a01. Current submission/ack lives in CURRENT_TASK.

## Wake-up policy

Reuse scripts/queue_watch.py, queue into the existing thread (never a new agent),
with first check10:55:29SGT and30-minute repeats, ending15:50:29SGT. State and prompt
in ignored temp/a2_watch_20261028; bounded6-hour wall-clock window. A delivery test
must distinguish app-server acceptance from an actual delivered acknowledgement.
No long polling or waiting in a model turn. Always report Singapore times and ETA.

At a wake: read latest user corrections, CURRENT_TASK and this plan; retrieve the
pending runner receipt first. Inspect actual terminal artifacts, LR/data coverage,
save/reload, and communicating burns, not merely PIDs. Archive small verified
results with checksums. No duplicate launches; commands#0 after acknowledged work.
Only after a healthy handoff may the next preregistered Phase0A dependencies/runs
be prepared/launched (e.g. the next dataset's S1 pair), reusing completed work.
Keep18-hour Phase0A cap and15-hour report trigger, account for all stages. Do not
start work expected beyond this watch window without a new explicit schedule.
No new architectures, extra LR grids, altered budgets or deferred research phases.
Phase0B requires its registered prerequisites; this window prioritizes Phase0A.

Each new GPU queue uses the existing safety supervisor: exact freshly inspected
burn worker identities, free-GPU recheck, copied Accelerate config + accelerate env,
and automatic verified communicating burns on success/failure. No unknown kills,
no changes to environments/driver, no GPU-node internet. Freeze remote writes on
controller/AWS/auth/connectivity failure; never use cleanup as infrastructure repair.
Stop the local timer when planned work ends or user cancels. A delayed snapshot
alone is not a reason to stop it. Do not restart old KV recovery watches.

## Continuation after DCLM — 28 October, 11:05SGT

DCLM S1 complete and burn handoff verified; see A2_DCLM_S1_RESULTS_20261028.md.
Extend existing pair coordinator with explicit --dataset fwe/nem; both new
parents get their own scientific survey/pre-baseline and3LR-grid fits before
short/main S1. Ten fits per next dataset; no DCLM grid rerun or changed recipe.
All50CPUtests pass, including full mocked10-fit order on each new dataset.
Actual numerical core is byte-identical; historical checks retain exact hashes.
Next FWE estimated80–95minutes; conservative prior Phase0A ledger5200seconds
(actual completed queues4811.77seconds plus setup/handoff reserve). Same3hhardcap.
Worker's existing30min schedule and15:50:29SGTdeadline remain; no duplicate timer.
