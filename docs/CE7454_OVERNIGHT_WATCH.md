# CE7454 overnight evaluation and monitoring

User authorized evaluation on both dev and official test, concurrent with the
existing eight B200 fits. Do not stop training to evaluate. No new architecture
fits are queued by this watcher. Deadline:17October2026 15:55SGT (07:55UTC).

## Two independent components

- B200 evaluator: `q359-ce7454-interim-watch-20261016-a01`, evaluates the latest
  available checkpoint from every arm once per30minutes, beginning immediately.
  GPU0 only, one evaluator at a time, batch8,10% allocator-memory ceiling.
  It never signals trainers, writes their checkpoints, or changes their schedule.
- Dev-machine timer (repaired): tmux `ce7454_queue_watch_20261016`, script
  `scripts/ce7454_resume_watch.py`, queues a follow-up to the existing conversation
  `01a120fe-ea0e-7030-a337-dd3f52f3c625` through `codex queue --remote unix://`. Next scheduled tick after repair:
  20:15:18UTC16Oct /04:15:18SGT17Oct, then30minutes; deadline06:30UTC17Oct.
  Local state/prompt/logs: `temp/ce7454_queue_watch_20261016/` (ignored).
  The old exec/resume timer failed with an active-writer conflict at19:45 and
  was stopped. The replacement uses the existing app server; a real test message
  was delivered and acknowledged20:09:48UTC. The20:15:18UTC scheduled message
  started a real monitoring turn, verifying the wake path end to end.
  It depends on this dev machine, the app server, CLI authentication,
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

## First complete periodic round —19:15–19:17UTC16October

All eight arms passed. Same eight trainer PIDs before/after, all progressing.
Total round106seconds including process startup and checkpoint snapshots;
maximum reserved PyTorch memory3.03GB (GRIPstats). GPU0 epochs temporarily
rose from28.1s to36.2–37.9s during shared evaluation. No training interruption
or weight/BN/optimizer changes; zero performance interference is not claimed.

| Arm | Epoch / training crop | Dev accuracy | Official test accuracy |
|---|---|---:|---:|
| grip_nodown | 306 / 256 | 97.44% | 94.60% |
| band_strong | 191 / 512 | 94.56% | 92.73% |
| band_ibn_strong | 175 / 512 | 95.22% | 85.53% |
| srresnet | 231 / 256 | 95.56% | 92.20% |
| band_moderate | 208 / 512 | 97.22% | 62.07% |
| phase_strong | 190 / 512 | 92.00% | 89.93% |
| grip_nodown_stats | 267 / 256 | 94.78% | 81.93% |
| srresnet_stats | 232 / 256 | 94.67% | 91.73% |

Evidence: docs/results/ce7454_interim_cycle000_20261016.json. These are
intermediate checkpoints at unequal stages; no arm replaced or stopped.
Across even a few epochs scores fluctuate, so do not prune from one reading.
Next remote round starts approximately19:45UTC; local review scheduled19:45UTC.

## Scheduled review —20:17UTC /04:17SGT

The repaired timer delivered its test20:09UTC and started the20:15UTC review.
All eight trainers and both coordinators remain active; no final evaluation
or burn handoff yet. All24 evaluations across cycles0–2 passed; no reported
evaluation failures. Each used900dev/1500official-test images.

Latest completed cycle2,20:15–20:17UTC:

| Arm | Evaluated epoch / crop | Dev accuracy | Official test accuracy |
|---|---|---:|---:|
| grip_nodown | 432 / 256 | 96.56% | 95.93% |
| band_strong | 295 / 512 | 90.00% | 88.53% |
| band_ibn_strong | 270 / 512 | 89.67% | 85.80% |
| srresnet | 327 / 256 | 95.78% | 95.20% |
| band_moderate | 322 / 512 | 98.00% | 65.20% |
| phase_strong | 293 / 512 | 94.56% | 95.60% |
| grip_nodown_stats | 378 / 256 | 95.89% | 94.53% |
| srresnet_stats | 328 / 256 | 98.11% | 92.93% |

Training at20:17UTC: GRIP435/450, GRIPstats379/450, SR329/450,
SRstats328/450, Haarstrong298/450, HaarIBN272/450, Haarmoderate323/450,
phase295/450. Original four still require their planned40epoch full512 stage.
Using recent base epoch times plus the preflight full512 timings, the last
training stage is estimated near23:05UTC/07:05SGT; allow until roughly07:30SGT
for final evaluations and variation. This is an estimate, not completion.

Recommendation: preserve the current schedules. SRstats98.11 and moderate98.00
differ by one dev image, below confidence warranted by estimated labels. Haar
strong and IBN dev results fluctuate substantially; no evidence to replace the
baseline from these snapshots. No tests are used for training selection, no
fits stopped or restarted, no new architectures. Next review04:45SGT.

Evidence: docs/results/ce7454_watch_review_20261016_a02.json.

## Scheduled review —20:47UTC /04:47SGT

All eight training arms continue, and all32 interim evaluations across cycles0–3
passed (900dev/1500official-test each). Original supervisor and periodic watcher
remain active; no final results, study-complete markers or burn handoff yet.
GRIP completed its450epoch base and automatically entered the planned40epoch
full512 fine-tune, now13/40. Its new trainer PID1289808 replaces the completed
base trainer as expected; the other seven trainer PIDs are unchanged. No manual
training interruption, model/code/environment change or new evaluation job.

Latest completed cycle3,20:45–20:47UTC:

| Arm | Evaluated epoch / stage | Dev accuracy | Official test accuracy |
|---|---|---:|---:|
| grip_nodown | 12 / full512 fine-tune after base450 | 91.56% | 91.13% |
| band_strong | 348 / native512 scratch | 97.78% | 92.07% |
| band_ibn_strong | 318 / native512 scratch | 94.22% | 89.67% |
| srresnet | 375 / base256 | 97.67% | 94.27% |
| band_moderate | 378 / native512 scratch | 97.89% | 70.73% |
| phase_strong | 345 / native512 scratch | 96.22% | 95.53% |
| grip_nodown_stats | 433 / base256 | 95.56% | 94.73% |
| srresnet_stats | 376 / base256 | 97.78% | 94.87% |

GRIP's early fine-tune dev91.56% is below the previous base snapshot96.56%.
This is a real observed drop, not evidence that full512 fine-tuning has finished
or failed: only12/40epochs were evaluated and the LR was restarted as planned.
Continue the fixed stage, then compare final dev performance; no test-based
selection. Haarstrong recovered on dev, confirming earlier snapshot instability.

Live progress20:47UTC: GRIP13/40full512, GRIPstats434/450base, SR377/450base,
SRstats376/450base, Haarstrong350/450, HaarIBN320/450, moderate380/450,
phase346/450. GRIP expected to finish training05:30–05:40SGT; last SR stage
about07:05SGT, allow~07:30SGT for final evaluation and timing variation.
These remain estimates. User's possible later all8GPU single-model training
is discussion only; no such follow-up is queued. Existing handoff preserved.

Evidence: docs/results/ce7454_watch_review_20261016_a03.json. Commands#0
after acknowledged receipts. Existing timer continues; next review21:15UTC.

## Scheduled review —21:18UTC /05:18SGT

All eight arms remain active, all40 interim evaluations across cycles0–4 passed
with900dev/1500test each. No reported error, final-result or complete marker.
GRIP and GRIPstats completed450base and entered their planned40epoch full512
fine-tunes; current epochs31 and11. GRIPstats new trainer PID1297316 is the
expected stage transition. Original supervisor, companion and watcher continue.

Latest completed cycle4,21:15–21:17UTC:

| Arm | Evaluated epoch / stage | Dev accuracy | Official test accuracy |
|---|---|---:|---:|
| grip_nodown | 30 / full512 fine-tune after base450 | 97.00% | 95.53% |
| band_strong | 400 / native512 scratch | 96.00% | 94.00% |
| band_ibn_strong | 365 / native512 scratch | 93.89% | 91.00% |
| srresnet | 423 / base256 | 97.33% | 95.47% |
| band_moderate | 435 / native512 scratch | 97.78% | 71.47% |
| phase_strong | 397 / native512 scratch | 95.89% | 95.47% |
| grip_nodown_stats | 10 / full512 fine-tune after base450 | 95.67% | 93.80% |
| srresnet_stats | 424 / base256 | 97.56% | 95.47% |

GRIP dev recovered from91.56% at fine-tune12 to97.00% at30. This supports
waiting for the planned endpoint; it does not establish a final improvement.
No method or checkpoint selected from official-test scores. Dev labels remain
estimates; all interim comparisons involve different training stages.

Live21:18UTC: GRIP31/40, GRIPstats11/40; SR427/450base and SRstats426/450base;
Haarstrong405/450, IBN369/450, moderate439/450, phase400/450. GRIP training
estimated05:33–05:40SGT, moderate~05:24SGT, last SR stage~07:05SGT; allow
~07:30SGT for final evaluation/variation. No final completion claimed.

Next review05:45SGT may find periodic watcher stopped: the first original
final evaluator intentionally requests STOP and waits for its release. Do not
restart it. Retrieve normal final prediction files and verify companion/burn
handoff. No new jobs, schedule or environment changes. Commands#0 after
acknowledged export. Evidence: results/ce7454_watch_review_20261016_a04.json.

## Scheduled review —21:46UTC /05:46SGT: first final results

Three fixed-endpoint evaluations passed, with checkpoint identity and exact
900dev/1500test filenames verified locally. Prediction-file SHA256 matches
remote receipts; finite logits, argmax predictions and counts also verified.

| Arm | Completed training | Final dev accuracy | Final official test accuracy |
|---|---|---:|---:|
| GRIP |450base256 +40fine-tune512|96.78% (871/900)|96.13% (1442/1500)|
| Haar strong |450native512|96.78% (871/900)|94.53% (1418/1500)|
| Haar moderate |450native512|97.89% (881/900)|71.67% (1075/1500)|

These are final checkpoints, not selected best interim epochs. Dev labels are
estimates; official test is reported separately and not used to modify training.
GRIP's final dev is similar to its recent late-base snapshots; the fine-tuning
recovery does not establish a meaningful dev improvement by itself. Preserve
all endpoints for comparison; remaining five arms have not finished.

Current: GRIPstats25/40fine-tune, SR6/40fine-tune, SRstats5/40fine-tune;
HaarIBN414/450, spatialphase449/450. SR and SRstats completed450base before
their new trainers. No reported errors. Original supervisor and companion
remain running; both study-complete receipts and companion-release absent.

The periodic watcher stopped successfully21:32:04UTC/05:32SGT at the original
final-evaluation gate; own_evaluator_exited:true verified. This is intentional.
Do NOT restart it. GRIP evaluation exited and its CPU gate is waiting for the
companion's verified release. GPU guard still belongs to original PID1236924.
GPUs0,4,5 currently free; GPUs1,2,3,6,7 train. Do not start burns or additional
fits on free GPUs: preserve the existing all8 supervisor handoff. No burn
receipt yet. Existing dev review timer continues; next22:15UTC/06:15SGT.

ETA: IBN~06:09SGT, GRIPstats~06:15SGT, SR/SRstats~07:03–07:06SGT;
allow~07:30SGT for all final evaluations/variation. No training/eval restart,
model/env changes or new fits. Commands#0 after verified receipt. Evidence:
results/ce7454_watch_review_20261016_a05.json. Full predictions saved locally
in ignored temp/ce7454_eval_upload/final-review-a05/.

## Scheduled review —22:16UTC /06:16SGT: six final endpoints

Six final evaluations now verified. Newly finished: GRIPstats, HaarIBN and
spatialphase. All900dev/1500test predictions pass exact-name, SHA, checkpoint,
finite-logit and argmax verification. Interim watcher remains correctly stopped;
its40 earlier results were rescored, no new interim jobs.

| Arm | Completed training | Final dev | Final official test |
|---|---|---:|---:|
| grip_nodown | 450base256 +40fine-tune512 | 96.78% | 96.13% |
| grip_nodown_stats | 450base256 +40fine-tune512 | 94.67% | 94.73% |
| band_ibn_strong | 450native512 | 93.56% | 91.20% |
| band_moderate | 450native512 | 97.89% | 71.67% |
| band_strong | 450native512 | 96.78% | 94.53% |
| phase_strong | 450native512 | 96.44% | 95.60% |

Only SRNet and SRNetstats remain: both19/40epochs full512 fine-tuning after
450base. GPUs1/2 hold their two known trainer processes; other GPUs are free.
No errors reported. Both GRIP endpoints and all four companion endpoints have
saved passing final evaluation receipts, not merely vanished processes.

Companion controller and study status passed at22:09:53UTC/06:09SGT; verified
release says GPUs4–7 free. Original supervisor remains running; original-study
completion and burn receipt absent because SR training continues. Guard and
automatic eight-GPU burn handoff unchanged. Do not start extra jobs/burns.

Dev-only interpretation: GRIPstats and HaarIBN did not improve over GRIP on
these final endpoints. Finish the two SR variants before overall conclusions.
Dev labels remain estimates; official test reported separately, not used to
select training changes. No new architectures or continuation fits queued.

Remaining training~47minutes from06:16SGT, expected07:03–07:05SGT, with
final evaluation and burn verification roughly07:15–07:30SGT. Existing timer
continues, next06:45SGT. Commands#0 after receipt. Evidence:
results/ce7454_watch_review_20261016_a06.json; full local predictions in
ignored temp/ce7454_eval_upload/final-review-a06/.

## Scheduled review —22:47UTC /06:47SGT: final two fine-tunes

SRNet33/40 and SRNetstats32/40 full512 epochs, after450base each. Same known
trainer PIDs1305002/1305226 onGPUs1/2,99–100% approximate utilization;
134.4/135.1seconds per epoch. No errors. Six final prediction sets reverified
against receipts and local labels; results identical to06:16SGT table above.

Companion remains passed/released, periodic watcher remains stopped as intended.
Original supervisor/guard active; original complete and burn receipt absent.
No GPU-management or training changes. Expected remaining training16–18minutes
from06:47SGT (~07:03–07:05SGT), then final evaluation/burn verification.
Allow~07:15–07:30SGT. Existing timer next07:15SGT should check all final scores,
original exit/completion and actual advancing burn collective cycles. Do not
stop timer or infer completion from PID absence before verification.

Commands#0 after receipt. Evidence: results/ce7454_watch_review_20261016_a07.json;
local log temp/remote_logs/ce7454-watch-review-a07.log. Dev-only decisions;
no new fits, no changes based on official-test results.
