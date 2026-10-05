# Project notes

## Clean restart verified, P1-block preprocessing (2026-10-05)

User-authorized clean restart launch7d9f0b1 verified by monitor47215cd at
10:42:40UTC. The earlier V1-first screen was intentionally stopped via its
identity-rechecked supervisor (110b1b6); all8 training workers exited, GPUs
were verified free and communicating burn restored before the new launch.

New launch copied resources/accelerate_config.yaml to the active environment's
/dev/shm/.cache/huggingface/accelerate/default_config.yaml and checked accelerate
env:8GPU/MULTI_GPU/BF16. Seven targeted tests and existing nine-arm smoke artifact
validation passed. Known burn89414–89421 was reclaimed by verified identities;
all8 GPUs free at10:38:08UTC before the supervised cleanup/training queue.

Cleanup finished10:39:42UTC, removed exactly screena01 and325 English dataset
cache-*/tmp-* files (1,021,643,627,160bytes), preserving all351 source files.
Old screen output absence verified; completed FA4 A reference preserved and
its baseline validator passed. No environment/model/training/loss changes.
Cleanup/source-preservation receipts are saved outside the deleted directory.

Fresh P1-block job started10:39:42UTC, workers90422–90429. At10:42:40UTC it was
running normal train.py tokenization with160 workers,14% of36,595,514 documents.
Cache rebuilding is active; optimizer updates have NOT yet been verified.
This is a fresh output/run, not a checkpoint resume. Later arms reuse the newly
rebuilt deterministic cache. Same seed42,2500 updates each/all8 GPUs,
full28600/warmup1430,micro16/GAS4,FA4 isolation/reset positions/EOS,checkpoints off.
Order:P1-block→P3-block→P1-lambda0→P3-lambda0→P1-flow→V1→V3.
Existing A2500 is reused only for comparison. Per-arm validators/final reports
and automatic verified communicating burn recovery after success/failure remain.

Active root:/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02.
Run state/logs:supervised/run; guard/supervisor remain active. Do not relaunch.
Next action:read-only preprocessing/training progress and rebuilt fingerprint
comparison with A (train6e4708f1e818fb44,eval08432871cf987d61) once config is saved.
Evidence:artifacts/proxy-fa4-restart-seed42-20261005/ and
temp/proxy-restart-monitor-a02.log (SHA256
d68b40eef6b114a89ccf4d9cf68802b776b89701e7824b495833c22b64b1cf63).
commands.sh returned to inactive#0; detached training remains running.

## Seed-42 FA4 screen running (2026-10-05)

Launchb88e455 deployed queue tooling7f9b2d9; startup monitor9215134 confirms the
first job V1 training on all8 B200 GPUs. Observed10:05:26UTC (18:05Singapore),
workers85721–85728, ~99% GPU utilization/113782MiB device usage. Reached update28;
logged updates10/20 have finite losses12.09/11.85 and grad norms2.108/2.951,
steady update time~2.08s. Saved V1 common recipe matches completed FA4 A exactly.
Actual FA4 train receipt:28 forward and28 Q/K/V gradient calls, BF16, native GQA,
document isolation and reset positions. Training active; completion not claimed.

Sequence:V1→V3→P1-lambda0→P1-block→P1-flow→P3-lambda0→P3-block, seed42 only,
2500 updates EACH/all8 GPUs each. A2500 reused as comparison reference, not model
initialization; all seven are fresh from-scratch runs with shared recipe. Full
schedule28600/warmup1430, micro16×GAS4,1,048,576tokens/update,seq2048,EOS packing,
checkpointing off; unchanged attention_bench/NCCL/offline W&B environment.

Accelerate resource config copied to interpreter cache and `accelerate env`
verified. Known burn workers84505–84512 reclaimed by exact identity; all GPUs
verified free10:03:33UTC before queue. Four updated tooling tests and validation
of all9 existing smoke artifacts passed on-node before reclaim. Baseline CPU
validation finished successfully before V1. Every new arm has a checkpoint/runtime
validator before next training; final matched-arm report included. Existing
supervisor automatically restores verified communicating burns after success or
failure. No additional installation or backend numerical tests.

Run:/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a01.
Job logs/arm folders under supervised/run; live state supervised/supervisor.json,
queue state supervised/run/run.json; final marker supervised/run/complete.json.
Startup evidence:artifacts/proxy-fa4-screen-seed42-20261005/ and
temp/fa4-screen-monitor-a01.log. commands.sh restored#0 without stopping tmux
training. Next step is read-only progress/result monitoring; do not relaunch.

## Seed-42 FA4 proxy screen authorized (2026-10-05)

User requested real P1/P3 training on B200, explicitly selected seed42 first and
same settings as completed FA4 A. Queue scope: V1,V3,P1-lambda0,P1-block,P1-flow,
P3-lambda0,P3-block, each2500 steps/all8 GPUs. Reuse existing matching FA4 A2500;
no P3-flow or additional seeds. Full28600 schedule/1430warmup, micro16×GAS4,
seq2048, EOS document boundaries, per-document positions and FA4 isolation,
checkpointing off, offline W&B/NCCL settings unchanged.

Read-only preflight29d283b passed: expected node and only approved burn workers;
pinned environment; old A complete with optimizer/scheduler/all8 RNG states;
current smoke effective recipe matches baseline beyond known smoke-only settings.
Recomputed full validation fingerprint08432871cf987d61; train6e4708f1e818fb44.
Baseline config SHA256:2d198d6491282c7d8e748c63b372d5dac1efbe3eaa8bbd73c9e7d2189118f8db.
Receipt:temp/fa4-screen-preflight-a01.log. Plenty of disk (23446GiB free).

Queue tooling now allows matching FA4 baseline reuse, with a CPU backend check
before any training; final comparison still checks full recipe/data. Runtime
validator accepts explicit steps/arms and validates each arm before the next.
Four focused CPU tests passed after queue/validator updates; training/model/loss
code is unchanged from the successful B200 smokes. Previous fresh full suite142
passed before these tooling-only changes.

Prepared fresh output:/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a01.
Launch will copy/verify Accelerate config, revalidate baseline and smoke artifacts,
reclaim only verified burns, wait/recheck all8 GPUs free, and run sequentially
under train_then_burn. Automatic communicating burn restoration after success or
failure. No environment/driver changes. Launch observation pending.

## FA4 review findings verified; test fixes (2026-10-05)

Reviewed FA4_PROXY_REVIEW_FINDINGS_20261005.md against source and recorded B200
evidence. Findings confirmed. Reproduced the stale error-message assertion and
package-import error in tests/test_resume_staging.py; fixed both tests locally.
No model, attention kernel, loss, queue policy, environment or B200 workload changed.

Correction: the prior142-test pass predates the last train.py error-message edit;
it was incorrectly attributed to exact deployed commit73907b4. The old CPU receipt
is marked superseded, preserving its original result and stating this limitation.
A fresh full142-test run now passes (159.643s runner time), loading modules as
`tests.test_*` without PYTHONPATH=tests. Source SHA256 values were checked unchanged
before/after the run and recorded in
artifacts/proxy-fa4-validation-20261005/cpu-tests-review-corrected.json.
The previously recorded real B200 numerical/smoke results remain valid.

G2 is a confirmed tooling limitation: all FA4 baseline reuse is blocked, even a
matching FA4 A. Recommended next queue change: allow matching FA4 reuse, checking
backend before GPU training and retaining full recipe/data/seed/step matching.
Simply removing the guard and trusting the final report is insufficient: current
preflight reports on A alone and would not detect a backend mismatch until after
expensive training. This policy change is not implemented by this review.
G3's default-A behavior is confirmed and retained; explicitly select all screen
arms. FA4 remains the selected backend recommendation; no new GPU comparisons
are called for by these findings. commands.sh remains inactive#0. No remote push.

## FA4 proxy validation completed (2026-10-05)

Code73907b4, launch35332c1, final read-only monitorf9ed212. FA4 now supports A,
V1/V3 and all six P1/P3 variants through attention_backend=fa4. Shared projections,
RoPE, document resets, revision-7 targets/losses and HF Trainer remain intact.
CLI/recipe defaults stay SDPA; select FA4 explicitly and uniformly across a screen.
No backend switch on resume; legacy Deep-KV arms remain outside this FA4 path.

The earlier 142-test CPU pass preceded a final error-message edit; it did not
verify the exact deployed source. See the review correction below. Independent
varlen oracle tests cover nonzero gates,
checkpoint replay, auxiliary gradient routing, isolation and exact single-rank
resume. Eight-rank CPU DDP also passed normalization synchronization, data-order
resume and global-batch gradient scaling (maximum parameter roundoff7.45e-9).
On B200 all8 same-weight BF16 full-model cases passed: full-gradient relative
L2=0.537–0.573%, proxy-only=0.431–0.464%, max objective difference0.0001221,
max auxiliary difference0.00000334, exactly zero document leakage. Gates were
nonzero and block/flow auxiliary weight0.1; no optimizer updates in numerics.

All9 sequential three-step Trainer smokes passed on eight GPUs EACH, including
optional P3-flow. Micro16×GAS4, seq2048, EOS/reset/isolation, checkpointing off,
full28600 schedule/warmup1430; only smoke cutoff3/logging1/eval32 changed. Shared
recipe/data comparisons passed; optimizer/scheduler/eight RNG states, r7 buffers,
updated gates and actual Q/K/V FA4 backward receipts verified. Peaks103.62–128.76
GiB/GPU. No environment/driver installation. These are readiness checks, not
full-run convergence evidence or a launched research screen.

Queue finished09:14:48UTC; supervisor succeeded and automatic burns verified
09:15:59UTC. Fresh09:18:40UTC check confirmed known workers84505–84512 across
all8 GPUs,100% utilization,155212MiB each, all-rank readiness and advancing
collective cycles/payload; guard released. commands.sh restored#0 after this check.
Output:/mnt/local/_outputs/deep-llms_th2/proxy-fa4-smoke-20261005-a01.
SHA256-verified evidence:artifacts/proxy-fa4-validation-20261005/.
Report:PROXY_HEADS_IMPLEMENTATION_20261004.md, FA4 extension/results sections.

## FA4 proxy validation in progress (2026-10-05)

Operator approved extending the tested FA4 backend to P1/P3 and matched controls,
with local tests and B200 smoke validation only. Production model changes remove
A-only backend guards; native/replaced K/V, targets, EMS, packing and HF Trainer
remain shared. Q/K/V backward receipts now cover proxy gradients. Pending r7
normalization changes from prior turns are included in this deployment.

Local focused tests passed (11). Eight-rank CPU DDP with an independent varlen
oracle passed A/P1-flow/P3-block/P3-lambda0: exact normalization buffers and data
order on resume, parameter roundoff <=7.45e-9, global-batch gradient differences
<=7.45e-9. Evidence: temp/fa4-proxy-ddp-a01/verified.json. Historical CPU run: 142 tests passed in 159.94s before the final error-message
edit. This was incorrectly attributed to exact commit73907b4; see review correction.
B200 plan: eight full-model same-weight cases with nonzero gates and separate
proxy-gradient thresholds, then nine sequential three-step runs, each on all
eight GPUs (A, V1/V3, six P variants including optional P3-flow). Full 28600-step
schedule/1430 warmup, micro16×GAS4, seq2048 and existing full training cache;
32-row evaluation for smoke only. No environment installation or full screen.
Unique output: /mnt/local/_outputs/deep-llms_th2/proxy-fa4-smoke-20261005-a01.
Copy/verify Accelerate config, identify/reclaim only approved burns, verify free
GPUs, use existing supervisor for automatic burn restoration on any outcome.

## Repeated SDPA200 completed:74% norm gap (2026-10-05)

Authorized single SDPA200 repeat completed successfully, launch15cc5cf / final
monitor607af11. Exact historical e06d5f9 source restored in isolated output; same
old8192-row train/512-row validation pools, seed42, eightGPU BF16 micro16×GAS4,
28600 schedule/1430 warmup. Full per-rank token/segment stream hashes and LR
histories match originals. Old data/results, envs and local P1/P3 work untouched.

SDPA repeat versus original: max pre-clipping norm gap73.9992% at185 (norms
0.66423291/1.15576005), mean5.1968%, first>=3% at137,57 updates>=3%. Old FA4 versus
original SDPA: max48.0984%, mean4.1875%, first>=3% at137,49 updates>=3%. Repeat max
training-loss gap0.0145850; common held-out original/repeat/FA4 losses
7.11075908/7.10265799/7.10742352. Thus large trajectory norm gaps also occur without
changing backend. This does not identify the cause of the separate2500-step63%
gap or establish multi-seed/full-schedule equivalence. Old trajectory thresholds
remain failed for both comparisons; measurement/execution succeeded.

Train job533.04s, completed08:37:59UTC. Burns restored08:39:10UTC; fresh08:39:50UTC
verified known8 workers51261–51268 with advancing collectives, guard released.
commands.sh restored#0. Evidence:`artifacts/sdpa-repeat-200-20261005-a01/` and
`temp/sdpa-repeat-monitor-a03.log`; full details in
DOCUMENT_ISOLATION_INVESTIGATION_20261004.md Section12. All rank stream hashes
also match original downloaded receipts; comparison JSON hash matches run
manifest. Current revision-7 P1/P3 changes remain uncommitted and undeployed.

## Repeated SDPA 200-step control launched (2026-10-05)

User authorized one200-update SDPA-only repeat matching the old diagnostic.
Launch15cc5cf; output `/mnt/local/_outputs/deep-llms_th2/sdpa-repeat-200-20261005-a01`.
Original benchmark source e06d5f9 restored to isolated historical_source directory
using a reviewed patch; all8 source-file SHA256 values match that commit. Exact
old8192-row train /512-row heldout packed datasets reused and token/segment hashes
verified. Seed42, eight GPUs, micro16×accum4, full schedule28600/warmup1430; fresh
random initialization, stop200. Local/source-snapshot and node CPU tests5 passed.
Accelerate resource config copied and `accelerate env` verified. Verified burns
47839–47846 stopped, all GPUs free before launch. Supervisor restores burns after
success/failure. Last observation08:31:27UTC:8 workers48809–48816, about45 updates,
finite losses/norms; no completion claim yet. Monitor d86896b. Compare helper
checks every rank's full input stream and LR history against original SDPA and
FA4; measures large gaps without mislabeling them as execution failures. Old
3%/0.01 trajectory limits remain visible in output. This warmup-only repeated-pool
control is not a recreation of the later2500-step63% observation.

## Final trained-weight FA4 check passed (2026-10-05)

Authorized final check completed on B200, launch e2b5bfa, monitor eae9f2c.
Both checkpoint-2500 models × train/validation × micro2/16 passed same-weight
backend checks, repeats and micro2 FP32 references. Production micro16 gradient
relative L2=0.7323–1.2730%, cosine>=0.999919, loss difference<=0.000066042.
FP32 gradient errors similar: dense0.9224–1.2157%, FA40.9089–1.1907%. All parameter
fingerprints and both source checkpoint file hashes unchanged. No optimizer
updates; diagnostic source is the committed production Arm A, without deploying
local revision-7 P1/P3 changes. See DOCUMENT_ISOLATION_INVESTIGATION_20261004.md
Section11 for full methods, results, limits and evidence hashes.

Final checks completed08:02:47UTC (98.70seconds). Automatic burns restored and
verified08:03:58UTC; fresh08:05:22UTC check confirmed workers47839–47846 across all
8 GPUs with advancing collective progress, guard released. Environment unchanged;
Accelerate config copied/verified. commands.sh restored#0. Receipt:
`temp/proxy-final-check-monitor-a02.log`; exact-hash-verified extracted artifacts:
`temp/proxy-final-check-results-a01/`. No more backend checking is needed to answer
this baseline question; multi-seed/full-schedule/custom-arm claims remain separate.

## FA4 versus dense numerical review (2026-10-05)

Read-only retrieval `88bddbd` compared all250 training log points and all monitor
and final evaluations. Configs/fingerprints match except backend, and logged LR
is identical. Training-loss mean absolute gap0.00330542, maximum0.04351387;
final full validation3.47794143 dense /3.47717338 FA4. Same-weight production
preflight gradient relative L2=0.54365%, cosine0.99998523. Separately evolved
trajectory gradient norms differ more: mean13.2256%, max63.2908% atupdate980.
This is similar convergence, not identical gradients/weights or proof of
full-schedule equivalence. No new2500-checkpoint matched-weight GPU test was run.
See DOCUMENT_ISOLATION_INVESTIGATION_20261004.md Section10 for full evidence and
limitations. Receipt:temp/fa4-dense-numerical-review-20261005-a01.log. No GPU process
or environment changes; only read-only commands pushed, then restored to#0.
Local revision-7 P1/P3 changes remain uncommitted and undeployed.

## FA4 A completed; automatic burn verified (2026-10-05)

Read-only status commit `867d031` verified successful completion of all queued
jobs, final validation and step2500 checkpoint (model, optimizer, scheduler,
Trainer state and all eight rank RNG files). Run root:
`/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01`.
Training job finished06:50:13UTC /14:50:13 Asia/Singapore. Full schedule remains
28600; result status `stopped` is the requested2500-update cutoff, not a crash.
2,621,440,000 input tokens, final4882-row validation LM loss3.4771733830242613
versus dense A3.4779414257087833 (delta−0.000768042684522019). Config comparison
passed with only attention backend differing. One seed does not establish an
accuracy advantage. Trainer runtime5298.1764s (~88m18s), versus dense6187.4524s;
recent updates~2.074s. Peak allocated111264339456bytes.

Automatic burn verified06:51:24UTC. Fresh07:16:26–07:16:38UTC inspection found
approved workers46080–46087, all eight GPUs100% utilized, guard released and
collective progress advancing1690→1710 cycles /1876.49→1898.70GiB. No process
was stopped or environment changed by this check. commands.sh restored to#0.

Receipt:`temp/fa4-completion-status-20261005-a01.log`, SHA256
`abc50f27209a1704ef3c0c57db5f1dfa8e7e6cef72bf76b9407637b02bd181f5`.
Extracted JSON artifacts:`temp/fa4-completed-results-20261005-a01/`; result JSON
SHA256 verified against remote run manifest:
`5394fe8132cd528be31669ef6fbd3c3fc3fbf72ab7044db6e4bd37a31382cd4f`.
Only status commands were pushed; local revision-7 P1/P3 work remains uncommitted
and was not deployed or executed on B200.

## Revision-7 follow-up review passed (2026-10-05)

Rechecked target indices/increments, normalization order, block/flow gradient
routing, accumulation/DDP loss scaling, checkpoint replay, saved config matching
and the unchanged dense-SDPA/packing path. No new production-code defect found.
Strengthened tests for unequal microbatch sizes/means (pooled within-step variance,
EMA across steps), immediate post-update buffer synchronization before any DDP
forward/evaluation broadcast, and rank-local data order across resume.

Full offline suite: **138 tests passed in144.624s**
(`temp/proxy-r7-review-full-a01.log`). Eight CPU ranks passed for A, P1-flow,
P3-block and P3-lambda0 (`temp/proxy-r7-review-ddp8-a02/verified.json`):
- identical pre-interruption checkpoints and rank-local input order;
- bitwise-identical mean/variance immediately after both bootstrap passes and
  each optimizer-step update across ranks;
- bitwise-identical saved buffers after resume;
- maximum resumed parameter difference3.725290298461914e-9 (also vanilla A),
  and distributed/global-batch gradient difference at most the same magnitude.

The initial eight-rank check rejected this parameter difference because it
incorrectly required bitwise equality for every parameter after restarting DDP.
It was measured before adjusting the test: checkpoint2 matched exactly, with
only tiny differences after resumed update3. This is consistent with changed
floating-point reduction order. The revised test retains exact buffer checks,
checks data order explicitly, and bounds parameter differences with rtol2e-6 /
atol2e-8 while recording the observed maximum. No training implementation or
normalization tolerance was changed to make the test pass. Previous two-rank
bitwise results remain valid for that test, not a universal DDP guarantee.

Compilation/whitespace checks passed. No B200 access, push, environment change or
GPU launch; commands.sh remains#0. Full-size revision-7 P1/P3 GPU smoke/capacity
and runtime backend checks remain required before screening.

## Revision-7 P1/P3 implemented locally (2026-10-05)

User approved the updated spec and all normalization recommendations. P1 now
standardizes detached MLP window sums; P3 uses detached FP32 deep-band increments
h_b-h_(layer-1), independently normalized per (proxy,deep) pair before averaging
and EMS. Default full-depth buffers: 12 P1 / 58 P3 pairs. Static masks are rejected.
Per-channel mean/variance, floor0.01, clip±10, epsilon1e-6 and EMA momentum0.99
replace the old masked RMS targets. Cosine remains default; Smooth L1 beta1 is
available only as an explicitly selected ablation.

Initialization reuses the first rank-local training microbatch for two no-grad
forwards, all-reducing means then squared deviations. Shifted sums/squares/counts
accumulate outside checkpointed functions and update once per optimizer step,
including lambda-zero arms. Buffers freeze within steps/recomputation/evaluation.
Per-buffer clip/floor/variance/lag diagnostics are recorded in Trainer/W&B logs.
Target version r7, hyperparameters, quantity and index sets are saved; strict
resume rejects old targets or hyperparameter changes. No migration override.
A/V saved config/state compatibility is retained; report matching ignores only
irrelevant target metadata for controls and remains strict across proxy arms and
seeds. The completed dense A remains the matching reference; FA4 A is separate.

Validation on local sampling_b200 env, CUDA hidden:
- 29 focused model/Trainer/FA4 tests passed in37.520s.
- Full regression:136 tests passed in132.101s
  (temp/proxy-r7-full-tests-20261005-a01.log).
- Final six T9/accumulation tests passed in4.841s after adding BF16 statistics
  coverage and guarding zero-variance diagnostic denominators
  (temp/proxy-r7-final-t9-tests-a02.log).
- 500-update FP32 vs FP64 maximum relative errors: mean8.77e-7, variance1.89e-6.
  After1000 steps of4096 tokens, 100000 held-out tokens: max abs mean0.007692,
  channel variance range[0.987434,1.010180]. Freeze/clipping/detachment tests pass.
- Two-rank CPU train/resume checks passed for A/P1-flow/P3-block/P3-lambda0:
  all resumed weights/buffers bitwise equal, mean/variance bitwise equal across
  ranks; accumulated distributed/global-batch gradient maximum delta3.73e-9.
  Receipt:temp/proxy-r7-ddp-20261005-a01/verified.json.
Compilation and whitespace checks passed. An intermediate P3 resume test exposed
list/tuple serialization of target indices; fixed before final passing tests.

Changes remain local; no push, runner command, B200 process/environment change or
P1/P3 GPU run. commands.sh stays#0. The previously launched FA4 A is untouched;
its last verified remote state is recorded below, not refreshed in this task.
Next deployment step is a separate full-size eight-GPU P1/P3 smoke/capacity check
before screening. Local tests establish semantics, not GPU throughput or research
benefit. See PROXY_HEADS_IMPLEMENTATION_20261004.md for the updated usage guide.

## FA4 A tests passed; production training verified (2026-10-05)

Launch commit `90c5e11`, monitor `dfa465e`. At 13:22:22 Asia/Singapore
(05:22:22 UTC), fresh FA4 A had reached update13, with eight train.py workers
41445–41452 on all eight B200s. Recent update time 2.069s, finite loss/gradients.
Root: `/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01`;
production arm `supervised/run/baseline/seed-42/A`, cutoff2500. Not finished.

Accelerate config copied to `/dev/shm/.cache/huggingface/accelerate/default_config.yaml`
and `accelerate env` verified 8/BF16. Verified burn workers34928–34935 stopped;
all eight GPUs had zero compute PIDs/memory at05:18:55 UTC before tests.
No train_env/driver changes. attention_bench core package versions match.

Production-model numerical gate passed: same weights/packed text, loss absolute
difference0.0000419617, hidden relative L2 0.56898%, logits0.63636%, gradient
relative L2 0.54365%, cosine0.99998523. Cross-document output and embedding-gradient
leakage both exactly zero. This is a startup check, not a long-run equivalence claim.
Separate eight-GPU three-update smoke passed: exact saved scientific config/data
fingerprints match completed dense A after removing backend and smoke logging
frequency; all8 RNG/optimizer/scheduler/model artifacts and FA4 forward/backward
receipts verified. Peak allocated111264339456 bytes, eval loss12.1131795165.

Supervisor will validate final artifacts and restore communicating eight-GPU
burns after success/failure, provided the machine remains available and owned
cleanup/free-GPU verification succeeds. No completion/burn restoration claimed yet.
Expected output `supervised/supervisor.json` and `supervised/burn-verified.json`.
commands.sh returned to#0; detached training continues. Next action: read-only
status/collect after training, not another launch. Evidence:
`temp/fa4-launch-receipt-20261005-a01.log` and
`temp/fa4-startup-monitor-20261005-a01.log` (SHA256
`6f5a56d0b57a4bfd5884f89be75d81f9873ea0b84c7989dcb1eec10d9190bc9d`).

## FA4 A launch requested (2026-10-05)

User authorized testing and fresh training of A with FA4, matching the completed
2500-update dense A (`proxy-baseline-A-2500-20261005-a01`). Read-only B200
preflight passed: node tjx3, eight verified burn workers, matching core library
versions in attention_bench and train_env, FA4 4.0.0b33 import/pip checks passed.
Only recipe differences are attention_backend and output_dir.

Submission uses `proxy-baseline-A-fa4-2500-20261005-a01`: same seed42, micro16,
accumulation4, 8 GPUs, sequence2048, EOS, document isolation/reset positions,
28600-step schedule, 1430 warmup, cutoff2500 and checkpointing disabled. Copy
Accelerate config to its actual cache and run accelerate env before supervised
verified burn reclaim. Gates: full-model same-weight SDPA/FA4 CUDA numerical and
isolation comparison, separate three-update eight-GPU smoke, exact saved config
and data-fingerprint comparison to dense A, checkpoint/FA4 receipt checks.
Only successful gates permit a fresh production run. Supervisor restores the
communicating eight-GPU burn on completion or failure after owned cleanup.

Local gate regression: eight tests passed in 3.789s; shell syntax, embedded
Python, queue manifest and compilation checked. Current GPU gate results and
production launch remain unverified until remote receipts arrive. No live
environment/driver reinstall and no P1/P3 launch or loss changes.

## Attention option renamed (2026-10-05)

The public CLI/JSON option and model attribute are now attention_backend.
Use --arm A --attention_backend fa4. Recipe, queue selection, runtime audit,
documentation and tests use the new name. Default SDPA still omits the field
from saved scientific configs, preserving existing dense-checkpoint resume.
FA4 remains supported only for A; this rename does not expand model support.
All 27 focused FA4/proxy-training/Trainer tests passed in 64.333s
(temp/attention-backend-rename-tests-20261005-a01.log). Queue generation,
compilation and diff checks passed. No GPU job or remote push; commands.sh stays#0.

## Optional FA4-isolated baseline implemented (2026-10-05)

User requested an arm-A variant using FA4 document isolation. train.py now
accepts attention_backend=fa4 for A only, selecting the same ProxyModel/
ProxyTrainer baseline flow and unchanged packing, EOS, reset positions, loss,
optimizer and schedule. Native GQA/QK norm/RoPE feed the pinned FA4 varlen API;
one document-fragment layout is reused per microbatch without a dense mask.
The original dense SDPA screen remains the default and its saved config identity
is preserved. Switching attention backends on resume is rejected.

Recipe: baseline_a_fa4.b200.json. Queue CLI defaults to A only for this recipe;
use --seeds 42 for one run. The environment must already provide the separately
pinned flash-attn-4==4.0.0b33 from envs/attention_bench.txt. No fallback. Results
identify the backend/package and first-GPU-microbatch call/gradient receipts,
separately from SDPA receipts. These are not numerical-equivalence certificates.

All 28 focused CPU tests passed in 36.654s using an independent per-fragment
attention stand-in. Queue manifest and compilation/diff checks passed. Full
offline suite: 131 tests passed in 142.696s
(temp/fa4-baseline-full-20261005-a01.log). The checkpoint-enabled receipt
check also passed: 16 forward/recompute calls and eight query-gradient callbacks
for the eight-layer CPU fixture, with no dense mask. No B200 job or environment change; commands.sh stays#0.
Actual FA4 CUDA smoke/throughput validation remains before a real run.

## Proxy review fixes implemented (2026-10-05)

F1: calibration strictly loads either DeepKV-A or ProxyModel-A checkpoints.
Proxy block outputs are observed directly; every block must account for all
calibration tokens with finite statistics. The existing baseline is usable
without changing its weights or dropping checkpoint keys.

F2: proxy_heads.b200.json now explicitly disables decoder/LM/aux activation
checkpointing to match the completed baseline. Reports ignore channel-mask
fields for vanilla controls, but enforce identical path/receipt for every
P1/P3 arm and across seeds. Other recipe matching and resume checks stay strict.
Optional make-jobs/report-seeds --reuse-baseline SEED=/absolute/path/to/A
references an existing baseline, validates artifacts/seed/cutoff and skips its
training job. Final comparisons still enforce the complete shared recipe.

F3: default screen seeds are 42,1042,2042; initialization algorithms and explicit
seed overrides are unchanged. One reused A means 23 new training jobs.
Focused local CPU suite: 21 tests passed in 31.125s. Full offline suite: 124 tests passed in 138.135s
(temp/proxy-review-fixes-full-20261005-a01.log). Queue CLI dry-run and
compilation/diff checks also passed.
No B200 action or calibration run; commands.sh stays#0. Before the screen,
calibrate the completed A, put the shared mask path into the launch recipe,
and smoke-test full-size proxy capacity with the chosen execution flags.

## Baseline completed; eight burns verified (2026-10-05 03:19 UTC)

Read-only collector 998de33 verified the fresh original Qwen arm A/seed42
completed exactly 2500 updates with the full 28600-step schedule unchanged.
The three-update full-size smoke passed; all seven queue stages returned zero.
Final held-out LM loss: 3.4779414257087833 over 9,981,660 eligible targets
(4882 rows). Input tokens trained: 2,621,440,000. Trainer runtime: 6187.4524 s
(about 1 h 43 min); last update interval about 2.435 s. Peak allocated GPU
memory: 118,780,560,384 bytes. Use actual updates/runtime for throughput;
Trainer's generic train_steps_per_second uses the full scheduled step count
and is misleading after a forced cutoff.

checkpoint-2500 contains model weights, optimizer, scheduler, trainer state
and RNG files for all eight ranks. Supervisor training_status=ok. Automatic
burn handoff succeeded at 01:41:56 UTC. At 03:18–03:19 UTC, the same verified
burn workers34928–34935 under launcher34859 occupied all eight B200s at100%
utilization, collective cycles advanced6590 to6610, and the guard was released.
No further training was launched. commands.sh returned to inactive#0.

Evidence: temp/baseline-done-status-20261005-a01.log, SHA256
1578d04101d2cfb5c6ff54d0cf8380fcee99b68abc3d788703c263b2760e340a.
Remote root: /mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01;
model/checkpoint directory: supervised/run/baseline/seed-42/A.

## Fresh isolated baseline launched (2026-10-05)

User authorized one original Qwen3-0.6B arm A, seed42, from scratch to2500
updates on all8B200s. Launch3eec7f9 uses the proxy-screen boolean dense SDPA
path with no proxy heads/auxiliary loss, EOS boundaries and reset positions;
micro16/accum4/seq2048, full28600 schedule, warmup1430, all checkpointing off.
A separate3-update same-shape/schedule smoke must pass before the fresh baseline.
Existing train_then_burn supervisor provides ownership-verified reclaim, guard,
free-GPU checks and automatic verified burn recovery after success/failure.

At07:43 Singapore startup was verified through dataset-cache tokenization28%;
optimizer updates/smoke completion were still pending. Accelerate bytes/env and
reviewed source hashes verified; original burns21008–21015 stopped and all GPUs
free before new training ranks25274–25281. Root:
/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01;
model output supervised/run/baseline/seed-42/A. See CURRENT_TASK.md for receipts.
Initial read-only preflight assumed Parquet; the pool is actually the original
saved Arrow shards, already supported by train.load_text. No data/model change
or GPU stop occurred in that failed preflight. No resampling/download required.

## Proxy code review follow-up (2026-10-04)

Selecting any proxy arm now selects the matching proxy-screen A path for the
entire queue and rejects legacy B–G/bottleneck mixtures. V3 widening uses the
actual sequence length, consistent with its MAC report. Duplicate report seeds
are rejected. These fixes leave the default 2048-token recipe unchanged.
Independent reduced-width 28-layer target/loss/gradient reference checks pass,
along with all121 offline tests and repeated eight-process CPU resume/scaling
checks. Details/evidence: PROXY_HEADS_IMPLEMENTATION_20261004.md. B200 full-size
memory/dispatch/throughput preflight remains pending; no GPU job launched.

## Proxy-head screening uses revised dense SDPA contract (2026-10-04)

The active proxy_heads_P1_P3_spec_v3.md is revision 4, explicitly confirmed by
the user: boolean dense same-document causal SDPA, document-local positions,
and P3 EMS resets from the same document IDs. All new controls share this path.
Legacy A–G results are not matched controls. P1/P3 replace native KV groups;
they do not add the older extra attention branch.

New ProxyModel/ProxyTrainer adapters reuse the existing Trainer loop and data
pipeline. Mu bootstraps before the first update and changes once per optimizer
step after global reduction, including lambda-zero controls; checkpoint replay
never mutates it. Auxiliary lambda ramps over 250 absolute completed updates;
default maximum .1. Gates have no weight decay. Mask calibration is optional;
the no-checkpoint fallback explicitly records no excluded channels. All buffers
and calibration-file identity participate in checkpoint/resume correctness.

The new recipe/queue defaults to eight arms, seeds42/43/44, cutoff2500 with
full28600 schedule/warmup1430 and micro16/accum4/seq2048 on eight GPUs per job.
Legacy queue defaults remain ABCD. See PROXY_HEADS_IMPLEMENTATION_20261004.md
for commands, architecture counts and validation. Local distributed resume and
global-gradient checks and all 118 offline regression tests passed; new B200 capacity/throughput validation and real
screening have not run. commands.sh remains #0.

## Isolation follow-up review passed (2026-10-05)

Reviewed production packing/collation, dense and auxiliary masks, loss eligibility,
cache identity and resume guards after 2d51ea0. No additional production-code
change was needed. Added two regression tests connecting production tokenization,
packing and collation to all 11 models: literal source EOS stays in its real
document, later outputs/input gradients cannot depend on an earlier document,
and empty/single-token documents plus chunk continuations have correct positions
and target masks. The full offline CPU suite passed all 103 tests (97.028 s).
An independent oracle also matched every token/document ID for 2,005 shuffled
documents over two workers, batch-1000 map boundaries and empty documents (770
packed rows). Evidence: temp/isolation-review-full-suite-a01.log and
temp/isolation-map-boundaries-review-a01.log. Previous eight-process CPU resume
validation remains applicable; no production implementation changed in this
review. No B200 job launched; commands.sh remains #0.

## Dense SDPA document isolation enabled (2026-10-05)

User selected dense SDPA isolation for the next experiment series. `train.py`
and `deep_kv.b200.json` now default to `isolate_documents=true` for every arm,
in training and evaluation. Shared EOS tokenization carries real document IDs
through the existing two-map packing/cache pipeline. A small collator resets
positions per packed document fragment. Existing Context/model/loss code applies
same-document causal/strict-past masks and omits cross-boundary LM targets.
Literal source EOS tokens do not create false boundaries. No model, optimizer,
schedule or sampling changes. Implicit causal mode is rejected with isolation.
Isolation is part of the recorded data recipe; changing it on resume is rejected
even with performance-change permission. Use fresh outputs for isolated runs.
HF builds a separate boundary-aware cache during normal training; no manual
preprocessing or sampling rerun is required. It adds per-token segment metadata.
See DEEP_KV_TRAINING.md for behavior. Earlier entries saying production isolation
is disabled describe the state before this change. No B200 job was launched for
this change; commands.sh remains #0.

Validation: 31 local tests passed (real entry-point train/eval for all 11 arms,
cache reuse/rebuild and token preservation, isolation semantics, bottleneck losses,
resume, and audit regressions). Eight-process CPU BF16 micro16/accum4 checks for
A and Consumer-Aware-Align passed: resumed versus uninterrupted max parameter
difference 9.313225746154785e-10; masked Consumer-Aware DDP versus global-batch
reference max difference 3.725290298461914e-09. Python compilation/diff checks
passed. Logs: temp/production-isolation-tests-a02.log and
temp/production-isolation-ddp-a01.log; receipt:
temp/deep-kv-isolated-resume-a01/resume_verified.json. These local integration
checks do not replace a future B200 launch preflight.

## SDPA full-model audit completed (2026-10-04)

C1/C3 from SDPA_BACKEND_CHECKS_20261004.md completed on tjx3, launch ab62618.
All 88 cases passed: 11 arms, checkpointing on/off, cross/isolated masks,
train/eval, micro16/seq2048/BF16. Every observed forward and fused backward
used cuDNN; no math fallback. Full-objective gradients finite, call counts exact,
parameter fingerprints unchanged. Independent single-GPU cases across eight
GPUs; not DDP throughput. Tiny two-update actual Trainer smoke verified one
automatic receipt per phase. Future DeepKVTrainer invocations record first
train/eval CUDA dispatch on rank zero; train.py links the receipts.
Backbone API Q/K and additive masks were FP32 under BF16 autocast; retained
training tensors included BF16 attention/mask shapes. Aux API mask was boolean.
See Section 8 of the checklist for runtime, complete evidence, and limitations.
Results downloaded and all 104 manifest entries verified. Burns automatically
restored, collective progress verified 15:36:08 UTC, live workers 21008–21015
confirmed 15:39:13 UTC. No production isolation/pinning change. C2 conditional,
C4 optional, C5 deferred. commands.sh returned to #0 after export.

## Current document-isolation recommendation (2026-10-04)

See the [consolidated investigation report](DOCUMENT_ISOLATION_INVESTIGATION_20261004.md) for the goal, code map,
all test stages, timings, numerical results, and unresolved trajectory drift.
After discussing the 48% gradient-norm gap, the current recommendation is dense
SDPA isolation for real experiments when minimizing uncertainty is the priority.
This is not a production switch or launch authorization. Same-weight FA4 checks
passed; the 200-update trajectory comparison remains failed and its cause is
unproven. Production isolation remains disabled; commands.sh is #0.

## Same-weight trained FA4 verification (2026-10-04)

Follow-up 3d5070b/a02 passed all eight cases: both step-200 checkpoints, training
and held-out data, micro2 and micro16, seq2048. No optimizer updates. Full-gradient
relative L2 FA4 vs dense 0.181%–0.532%; at micro16 0.181%–0.266%. Both BF16 paths
have similar errors against FP32 math (dense 0.539%–1.209%, FA4 0.540%–1.163%).
Repeated full backward captures vary even on the same backend: dense up to
0.302%, FA4 up to 0.111%. Parameters unchanged and all original gates passed.
This supports numerical drift, not a demonstrated large FA4 backward error, as
a plausible explanation for the earlier training divergence. It does not prove
causation or cover auxiliary arms; production attention is unchanged.
Burn recovery and live workers verified. TRAINED_ATTENTION_CHECK_20261004.md
contains evidence and limits. Initial attempt failed only on diagnostic loader
handling of cloned tied tensors; fixed with strict alias consistency validation.

## Longer isolated-attention comparison (2026-10-04)

Two matched 200-update eight-B200 runs completed. Strict trajectory gate failed:
max loss gap 0.01066065 > .01 and relative gradient-norm gap 48.098% > 3%.
All input streams and LR sequences match; all losses/norms finite. Divergence
appears after the first 100 updates. These are different trained weights, so
the norm gap alone cannot identify a kernel gradient error. Common held-out
losses remain close (dense 7.1107591, FA4 7.1074235); evaluating the same final
FA4 weights through both backends differs by only 2.58e-6. Same-weight backward
agreement at the final checkpoint remains untested. Initial gradient gate passed.
Full update medians 2.42939s dense / 2.07476s FA4; repeated 16.78M-token pool,
original warmup, random initialization, baseline arm A only. Do not claim this
longer strict check passed or promote production FA4 based on it. Burns restored
and verified. Details: DOCUMENT_ATTENTION_STABILITY_20261004.md.

## Full eight-B200 training comparison for document isolation (2026-10-04)

Disposable daea6f6 benchmark completed11:08:14 UTC, all5 modes x30 optimizer
updates, with automatic eight-GPU burn recovery verified. Full random-init
Qwen3-0.6B arm A, HF Trainer/Accelerate, micro16/accum4/seq2048/BF16, activation
checkpointing off, LM chunks128. Same16.78M-token real English test pool and
seed42; first-update input hashes identical across modes. Dataset repeats.
Median full update intervals after5 warmup measurements: previous explicit
cross-document SDPA2.42576s; implicit causal SDPA2.06230s; FA4 cross2.08819s;
dense isolated SDPA2.42524s; FA4 varlen isolated2.07173s. FA4 isolation therefore
takes0.457% more time than fast causal SDPA and14.595% less than the previous
explicit-mask implementation. Peak allocated memory110.62GiB for dense modes,
103.62GiB for implicit/FA4. These are full baseline training steps, including
DDP and optimizer; startup/evaluation/saving excluded. Single short run per
mode, not a convergence study or performance result for auxiliary arms.

Full-model numerical gate passed: isolated FA4 vs BF16 dense gradient relative
L2=0.5408%, logits=0.6364%, loss gap4.196e-5. Against FP32, full-gradient errors
FA4=1.2314%, BF16 SDPA=1.2331%. Both isolated methods have exact zero output
leakage and embedding-activation gradients across document boundaries. Loss
decreased12.12->10.85 for cross-document modes and12.12->10.98 for isolation;
all30-step loss/gradient traces finite. Blocking documents changes the objective
and gradients deliberately; use matched masks for kernel correctness checks.

Experimental scripts/benchmark_document_training.py only; production train.py,
packing and custom arms unchanged. Boundary metadata tracks source document IDs
(including appended EOS), not token-ID guessing. Its Trainer normalizes by the
true global target count across ranks/GAS; this matters with isolated documents.
Four CPU tests include real HF Trainer full-batch/accumulation update equality.
Config/tokenizer on new node downloaded via e21d47f with pinned asset hashes.
Final weights and result files remain in document-training-20261004-a01 under
the project output root. See CURRENT_TASK for full receipt/hash information.

## API-based document isolation benchmark (2026-10-04)

FA4 flash_attn_varlen_func with document cu_seqlens and FlexAttention with
same-document causal block masks require no hand-written kernels. Experimental
adapters live in scripts/benchmark_packed_attention.py, not production code.
Strict-past auxiliary attention is expressible by per-document Q[1:], K/V[:-1]
with ordinary causal varlen and a zero first output. CPU tests verify outputs
and gradients including singleton documents.

B200 ca21c7b / packed-kernels-20261004-a02 passed all FA4/Flex-FA4 reference and
isolation checks in BF16, normal and strict-past, single/equal/ragged documents.
Pinned FA4 beta 4.0.0b33 returns a tuple; Flex-FA4 on B200 needs block size 256.
Separate attention_bench environment leaves train_env unchanged. For B2, L2048,
GQA16/8, D128, ragged normal forward+backward: implicit 0.339 ms, dense isolated
0.711 ms, FA4 varlen 0.525 ms, Flex-FA4 0.965 ms. Ragged strict-past: dense
0.775 ms, FA4 0.695 ms, Flex-FA4 0.970 ms. Wrapper movement overhead included,
metadata setup/compilation excluded. These do not estimate full training speed
at production microbatch size. No production attention policy was changed.
Completed 10:07:27 UTC with all eight GPU burns restored and collective progress
verified. Details and evidence hash are in CURRENT_TASK.md.

## Document isolation feasibility on tjx3 B200 (2026-10-04)

tests/test_document_isolation.py passed on local CPU/A100 and B200 FP32/BF16,
all11arms. Existing Context segment masks isolate decoder and auxiliary paths;
cross-document outputs match exactly and input-embedding gradients are zero.
Packed/separate parameter gradients agree in FP32; BF16 uses1% tensor norm/max
bounds because separate GEMMs round reductions differently. No production
segmentation enabled. Model tests exercise active auxiliary output weights.
scripts/benchmark_document_attention.py measured B200 implicit causal0.2535ms
versus document-isolated0.6239ms (B2,H16,L2048,D128,BF16,forward+backward).
All modes used cuDNN SDPA; explicit causal was0.6238ms. This is attention-only,
not a full-model throughput estimate; mask construction is excluded.
Completed e14a630 at09:38:50UTC with automatic eight-GPU burn recovery verified.
New node uses torch2.14.1+cu130 and driver580.167.08. Actual downloaded English
Arrow splits load through train.py with36,595,514 training/11,822 validation docs.

## Environment recipes and old-pool release (2026-10-04)

Keep env recipes simple: original direct dependency lists with exact versions;
torch2.14.1 for train/eval, sentencepiece0.2.2 for eval. Removed unused entmax.
Historical sampling_b200 locks remain unchanged; no reinstall was performed.
Public nht10/cx_sampled_old finished upload with 457 verified files (including
manifest), about156.5GB. Completion release2c425f4e1d4467008a0afce00de0ae8a33c8e834.
Local ordered-text hashes verified; old B200 document/shard counts match but
cross-machine full text-hash equality was never established. This is prepared
unpacked text, not training-token cache. No prepare_data.py rerun is needed.
Runner source sync lacks .git: do not require git rev-parse in GPU-node audits.

## Replacement B200 tjx3 installation (2026-10-04)

User supplied replacement Dropbox folder; use label th2-tjx3 and ignored
temp/dropbox_tjx3_folders.txt. Do not use78gg receipts as live inventory.
GitHub install commit1d3b68b matches local source and its runner log reports
successful installation of both train_env and eval at07:25UTC (15:25SGT).
Specifications keep Python3.11, transformers5.9.0, datasets4.8.5 and
accelerate1.13.0; torch is unpinned and installation resolved2.14.1 instead of
the prior2.14.0. Full old-machine package equivalence is not established.
Read-only runtime audit8f350d8 is pending; see CURRENT_TASK for scope and
retrieval. No new-machine training or GPU management was authorized/performed.
commands.sh is a pending CPU/import verification#1, not an installation retry.

## Task-Aware-Align continuation to 10,000 launched and verified (2026-10-01 10:33 UTC)

User requested Task-Aware-Align (seed 42) for 5,000 more updates to compare
with B/F/G at 10,000; B has no full evaluation or retained checkpoint at 7,500.
Native continuation as 0ce6bff. Read-only preflight c0eb025 passed at 10:20:57
UTC: pinned versions, 23 TiB free, guard enabled, only approved burn workers,
bottleneck recipe.json unchanged, and source checkpoint-5000 complete (all 13
files; eval 3.208638, seed 42). Launch d4de734 at 10:23 UTC
(th2-78gg-TA-10000-20261001-a01) copied/verified the Accelerate config, stopped
only verified burn workers 524137-524144, staged checkpoint-5000 with checksums
(stage-resume ok 10:26:00, 16 files), and resumed with the bottleneck recipe
unchanged except stop_after 10000 (schedule 28600, warmup 1430,
checkpoint_layers/lm off, checkpoint_aux on, lm_chunk 128). Training code is
unchanged since the bottleneck launch.
Status 86d1beb at 10:32:48 UTC: arm running at update 5,129, about 2.8 s/update;
saved config seed=42/data_seed=42, train fingerprint adb88539d2924dc2. Estimated
finish about 14:20 UTC plus evaluation/burn handoff. Original 5,000 outputs are
untouched. Root: /mnt/local/_outputs/deep-llms_th2/deep-kv-TA-10000-20261001-a01.
Evidence: temp/TA-10000-preflight.log, temp/TA-10000-launch.log,
temp/TA-10000-status-a01.log. commands.sh restored to #0.

## F seed 123 complete; burn verified (2026-10-01 06:58 UTC)

Read-only audit 69e2144 (th2-78gg-F-seed123-completion-20261001-0657) ran at
06:57:42 UTC; the runner recorded it OK. Queue complete.json matches run.json and
every artifact hash passed: smoke ok; F finished 06:30:16 UTC; comparison ok.
F stopped exactly at 2,500 / 28,600 updates (2,621,440,000 input tokens);
checkpoint-2500 holds nonempty weights, optimizer, scheduler, training args,
trainer state and eight RNG files. Config seed=123/data_seed=123, train
fingerprint c430c71d175af59e (same as A/B seed 123), eval fingerprint unchanged.
Held-out LM loss on the same 4,882 rows: F 3.5080745944491065
(route KL 0.16541, message 0).
Paired seed-123 contrasts: F-B +0.00152, F-A -0.00275 (B 3.50655, A 3.51082).
Seed 42 for reference: F-B +0.00446, F-A -0.00039.
F is worse than B in both seeds and better than A in both; the size of each gap
varies by seed. Two seeds only.
Automatic burn handoff verified: guard released, eight approved burn workers
with new progress. No process was changed by the audit.
Evidence: temp/F-seed123-completion-20261001-0657.log (SHA256
f3b8020dce862b81e596bacdc06760ee7ba1c441e0a0937e41d2d89dbbf7f296).
commands.sh restored to #0.

## F seed 123 replication launched and verified (2026-10-01 03:45 UTC)

User requested a fresh F at seed 123 for 2,500 updates, run as A/B seed 123 was.
Launch ff81e56 (th2-78gg-F-seed123-2500-20261001-a01) is 28da1bb with arms=['F']
and new root/session/job names only. Recipe = deep_kv.b200.json with only
seed/data_seed set to 123 (branch seed 124, PYTHONHASHSEED=123). Training code is
unchanged since the A/B seed 123 launch, so F is paired with A/B seed 123
(same data, order and backbone initialization). F seed 42's saved config differs
only in seed, train fingerprint, and three later pilot fields whose defaults
reproduce the earlier behavior.
Read-only preflight 7070c7a passed 03:30:53 UTC: pinned versions, inputs, 23 TiB
free, guard enabled, only approved burn workers 511147-511154. The launch copied
the Accelerate config (accelerate env: MULTI_GPU, 8, bf16) and stopped only those
verified workers. Root: /mnt/local/_outputs/deep-llms_th2/deep-kv-F-seed123-2500-20261001-a01.
Status 2c509f3 at 03:44:44 UTC: smoke ok 03:37:51 (10->12 resume, F route checks);
smoke step-10 peak allocated 21.17 GB/rank. arm-F running at update 92, about
4.0 s/update; F's train_config has seed=123/data_seed=123, max_steps 28600,
warmup 1430, train fingerprint c430c71d175af59e (same as A/B seed 123), eval
fingerprint 0568ce654afc3fdb. All eight ranks carry PYTHONHASHSEED=123; tmux pane
alive. Estimated finish about 06:30 UTC plus final evaluation/burn handoff.
Evidence: temp/F-seed123-preflight.log, temp/F-seed123-launch.log,
temp/F-seed123-status-a01.log. commands.sh restored to #0.

## A/B seed 123 complete; burn verified (2026-10-01 03:04 UTC)

Read-only audit af40d10 (th2-78gg-AB-seed123-completion-20261001-0303) ran at
03:03:42 UTC; the runner recorded it OK. Queue complete.json matches run.json and
every artifact hash passed: smoke ok; A finished 00:03:35 UTC; B finished
02:28:28 UTC; comparison ok. Both arms stopped at exactly 2,500 / 28,600 updates
(2,621,440,000 input tokens). Both checkpoint-2500 folders hold nonempty
weights, optimizer, scheduler, training args, trainer state and eight RNG files.
Held-out LM loss on the same 4,882 rows (seed 123): A 3.5108211549941237,
B 3.5065545880332287, B-A -0.004266566960895.
Seed 42 for reference: A 3.510730336181336, B 3.505886970597037, B-A -0.00484337.
Seed-to-seed change: A +0.00009, B +0.00067. B beats A by 0.0043-0.0048 in both
seeds, so the branch-only advantage replicated; two seeds only.
Automatic burn handoff verified: guard released, eight approved burn workers
with new progress. No process was changed by the audit.
Evidence: temp/AB-seed123-completion-20261001-0303.log (SHA256
6456ec93ef6dd04f7fbeaec89e65b0688bb6b22e7d322d92f6828a02ddf6f8e9).
commands.sh restored to #0.

## A/B seed 123: A complete, B running (2026-10-01 01:33 UTC)

Read-only audit 542ff8f (th2-78gg-AB-seed123-completion-20261001-0132) ran at
01:33:04 UTC; the runner recorded it OK. Queue run.json: smoke ok; arm-A ok,
finished 2026-10-01 00:03:35 UTC; arm-B running. A stopped exactly at 2,500 /
28,600 updates (2,621,440,000 input tokens). checkpoint-2500 holds nonempty
weights, optimizer, scheduler, training args, trainer state and all eight RNG
files. A's held-out LM loss on the same 4,882 evaluation rows is
3.5108211549941237, versus 3.510730336181336 for seed-42 A (+0.00009).
A and B configs both record seed=123/data_seed=123, train fingerprint
c430c71d175af59e and eval fingerprint 0568ce654afc3fdb. B's latest checkpoint is
1500; its log showed update 1,545 at about 3.5 s/update, so B should finish
around 02:30 UTC plus evaluation/save. GPU guard is held during training as
designed; live workers are the eight training ranks. No process was changed.
Evidence: temp/AB-seed123-completion-20261001-0132.log (SHA256
bde60643f946aae509e403203a5927136d959f783220d616e398a6dec8fdeeda).
commands.sh restored to #0.

## A/B seed 123 progress verified (2026-09-30 22:20 UTC)

Read-only status b2c80da (th2-78gg-AB-seed123-status-20260930-a04) ran at
22:19:50 UTC; the runner recorded it OK. The full-shape A/B smoke passed at
21:42:31 UTC. The production queue is running: arm A was at update 660 of the
2,500 cutoff (schedule 28,600), about 3.3 s/update, logged LM loss 5.357 at
step 660, with checkpoint-500 present. B has not started. The tmux pane is alive.
A's saved train_config has seed=123, data_seed=123, max_steps=28600 and
warmup=1430. Its train fingerprint is c430c71d175af59e, different from the
seed-42 runs' adb88539d2924dc2 as intended; the eval fingerprint is unchanged
(0568ce654afc3fdb). All eight worker processes carry PYTHONHASHSEED=123.
Rough estimate at this speed: A finishes about 00:05 UTC and B about 02:30 UTC
on 2026-10-01, before evaluation/handoff overhead. No process was changed.
Evidence: temp/AB-seed123-status-a04.log (SHA256
4f8402fa3516984f4ffcfb47d11c824b3f7d61c7ddfe126da30f9149c1f7edd7); the
previously unsaved a03 output is temp/AB-seed123-status-a03.log.
commands.sh restored to #0.

## Authorized fresh A/B seed replication (2026-09-30)

User requested fresh A and B, each cutoff 2,500, with all training seeds changed.
Selected seed=123 and data_seed=123, with PYTHONHASHSEED=123. Backbone and global
Python/NumPy/Torch RNG use123; B auxiliary initialization retains its independent
seed+1 stream (124). Existing sampled text splits and fixed evaluation prefix
remain unchanged; training shuffle and sampler both change to123. No resampling.
Original deep_kv.b200.json recipe is unchanged except seed/data_seed; full schedule
28,600, warmup1,430, micro16/accum4/world8/seq2048, original checkpointing/mask/chunk
settings. Each fresh run processes2,621,440,000 input tokens. Smoke checkpoints
will not initialize production. Queue: A/B full-shape10->12 smoke/resume, fresh
A2500, freshB2500, comparison, automatic burn handoff on success/failure.

Local CPU checks passed matched/new/repeatable initialization and RNG/shuffle/
sampler checks plus exact generated job arguments. Nine smoke/supervisor tests
passed (initial invocation typo corrected; no code issue). Read-only B200
preflight f6a7ab3 passed21:32:31 UTC: only approved burn workers487231–487238 under
487162; guard enabled, pinned packages and input assets present, >23TiB free.
Launch will copy/verify Accelerate config, run accelerate env, reidentify burns,
stop only verified workers and require all eight GPUs free before smoke.
Root: /mnt/local/_outputs/deep-llms_th2/deep-kv-AB-seed123-2500-20260930-a01.
Next: verify actual launch, smoke gate, seed configs and fresh production start.

## Both bottleneck arms complete; burns verified (2026-09-30 21:17 UTC)

Read-only audit fedcfb0 confirms both aligned arms completed exactly 5,000
updates / 5,242,880,000 input tokens, retaining schedule 28,600 and warmup 1,430.
Consumer finished 20:57:56 UTC (4h11m including startup/save/evaluation).
All queue jobs passed, complete.json matches run.json, and all declared artifact
hashes passed. Both checkpoint-5000 folders contain nonempty weights, optimizer,
scheduler, training args, trainer state and all eight RNG files.
Validation LM losses on the same 4,882 rows: Task 3.208638078874859;
Consumer 3.2137939965533175; historical B at 5,000 is 3.208608250096011.
Consumer is worse than B by 0.0051857464573065; Task is essentially tied.
Consumer auxiliary message CE gain is 0.015386735621624403, which is not an
improvement in the main LM loss. These are single-seed exploratory results.

Supervisor completed without error and automatically verified burns at
20:59:07 UTC. Audit rechecked all eight burn worker identities and new collective
progress (cycles 1210 -> 1230); GPU guard was released. No training restart or
process signaling occurred during inspection. Final small artifacts retrieved
and source-hash verified under artifacts/bottleneck-final-20260930/.
Raw log temp/bottleneck-completion-20260930-2115.log SHA256:
9e12496dc02d7a9f2eb73b8d0bad3d7df7f9a1a2ef5241b05fb93b412fc26533.
commands.sh restored to #0. No further training is queued by this experiment.

## Bottleneck status verified (2026-09-30 20:49 UTC)

Read-only audit 102f004: Task-Aware-Align completed exactly 5,000 updates and
5,242,880,000 input tokens at 16:46:48 UTC (3h56m wall time). Validation LM loss
3.208638078874859 over 4,882 rows. Full checkpoint-5000 weights, optimizer,
scheduler, training args and all eight RNG files verified present/nonempty;
result hash matches the successful queue receipt. Consumer-Aware-Align is still
running: live step 4,847/5,000 at 20:49:52 UTC, approximately 2.96s/update,
finite losses/gradients, eight train.py workers at 91–99% GPU utilization.
Estimated completion 20:58–21:00 UTC including save/evaluation; unverified ETA.
Supervisor remains running; burns have not restarted while training owns GPUs.
Automatic burn handoff remains pending. No restart or process signaling occurred.
Small artifacts pulled and source-hash verified under
artifacts/bottleneck-status-20260930-2046/. Raw snapshot SHA256:
4affd3b52bedfc0be9c0dbaaa406777b85459ff022a42acdf108b5e5e8d25184.
commands.sh restored to #0; the independent supervised queue continues.

## Bottleneck B200 launch verified (2026-09-30)

Final startup auditeb7ee19 at12:58:42 observed Task153, finite losses/gradients,
~2.77s/update and all8GPUs98–99%. Exact remote jobs/recipe files were pulled and
hash-checked: both arms have cutoff5000 and no resume argument; queue order is
smoke -> fresh Task -> fresh Consumer -> compare. Commands deactivated to#0;
independent supervised queue remains live. Evidence final-startup/ within the
artifact root below; raw audit SHA256
1a35118217b1c526666dfb33e3654c0cdd15aeccf0ba7683a727690bfc799bb9.

Launchaa1ce75 uses existing supervisor and selected-arm extension of the smoke
helper. Both aligned arms passed full Qwen3-0.6B, seq2048, micro16/accum4/eightGPU
BF16 smoke and native10->12 resume. Smoke gate passed12:50:48 UTC. Initial
measured update times were Task2.763s / Consumer2.930s; peak reserved memory
115.78 /119.37GiB, with >8GiB headroom on every rank. These short measurements
exclude long-run save/eval overhead. Shared inference parameters598540416;
training-only parameters Task19578880 / Consumer155713536.

Fresh Task-Aware production started12:50:48 and reached40 updates by12:53:28
(~2.78s/update, finite losses, warmup LR8.182e-6). All eight GPUs had one train.py
worker at97–99% utilization,120490MiB each. Consumer follows sequentially; each
cutoff5000 means5,242,880,000 input tokens. Both data fingerprints match prior
B/F/G exactly. Schedule28600/warmup1430 and original optimizer/data recipe stay
fixed. Decoder/main-LM checkpointing are off, auxiliary checkpointing on.

Accelerate copied to /dev/shm/.cache/huggingface/accelerate/default_config.yaml,
byte-verified and checked via accelerate env. Only known burn workers457634–
457641 were stopped after identity checks; allGPUs verifiedfree12:45:09. Final
burn recovery is automatic after queue success/failure under the unchanged
train_then_burn supervisor. No environment/package/driver modifications.

Root /mnt/local/_outputs/deep-llms_th2/deep-bottleneck-5000-20260930-a01.
Local hash-verified evidence artifacts/bottleneck-launch-20260930-a01/ includes
source digests, preflight/reclamation/free-GPU receipts, smoke completion and
production config. Smoke receipt also matches the outer queue artifact hash.

## Authorized two-arm bottleneck run (2026-09-30)

User selected the two aligned versions, each fresh for5,000 optimizer updates
(5,242,880,000 input tokens). NoAlign controls are not part of this launch.
Keep schedule28600/warmup1430, micro16/accum4/world8/seq2048, original text,
cache/order/seeds/EOS, optimizer/clipping, explicit attention mask, LM chunk128.
Use checkpoint_layers=false/checkpoint_lm=false/checkpoint_aux=true identically
for both arms: new training-only vocabulary readouts add memory. This requires
the production-shape smoke gate before fresh production outputs are created.
The existing smoke helper now accepts selected arms and validates bottleneck
loss denominators/components, main-LM evaluation, parameters, all-rank memory
and10->12 resume. Nine smoke-gate/burn-supervisor CPU tests passed locally.
Existing train_then_burn owns the entire smoke->training->comparison->burn flow.
Read-only B200 preflight2da76ed passed12:39:19 UTC; actual launch still needs
verification. No environment/package/driver changes are requested.

## Bottleneck second review (2026-09-30)

No training-logic correction required after specification/Trainer review and
independent mathematical checks. Fixed test_bottleneck sibling imports to use
tests.test_deep_kv/tests.test_train, supporting module invocation and discovery.
Expanded tests verify losses and every parameter gradient against a literal
dense consumer reference at blocks5/21 (22 tiny layers, query width2x residual),
plus all eight recomputation-flag combinations under BF16 with restored AdamW
and scheduler state. Full CPU suite83 passed in81.074s; local evidence in
temp/bottleneck-review-full.log. No model/optimizer/data/recipe changes and no
remote operations. B200 full-model memory/throughput remains unmeasured for the
new arms; earlier eight-process CPU distributed results are recorded below.

## Bottleneck variants implemented locally (2026-09-30)

`deep_bottleneck_task_and_consumer_v2.md` now has implementation/usage notes.
Arm IDs are Task-Aware-NoAlign/Align and Consumer-Aware-NoAlign/Align. All share
fresh identical backbone, P, E, code decoders and auxiliary output initialization;
each aligned/control pair has identical full initialization. Codes are fixed
width128, parameter-free RMS eps1e-6, SmoothL1 beta1, alignment weight0/.3.
The deep extractor reads the detached residual before block21; the predictor
reads block5 normalized attention input. Actual deep codes never enter LM logits.

Task CE trains E plus the independent code-to-vocabulary readout. Consumer CE
trains E plus the independent hidden-to-vocabulary readout, using functional
detachment of the live consumer weights and detached native query/shallow state.
Alignment targets are detached. Consumer query eligibility also requires a
strict-past source. All reductions are FP32; Trainer's item-count hook normalizes
each loss over its own global accumulated count. Existing A–G recipes remain.
New eval_loss means LM only, with eval_objective/components separate; consumer
no-message CE is evaluation-only. Result files include parameter counts and
invocation throughput/peak CUDA memory. Inference uses the shared predicted-code
branch; extractor/readout can be removed when auxiliary objectives are disabled.

Local full suite81 passed; eight-rank CPU BF16 native save/resume passed four
arms (max parameter difference1.86265e-9); unequal-mask DDP+accumulation update
matched global reference within3.72530e-9. These are tiny-model correctness tests,
not B200 capacity/performance evidence. No new run horizon or launch authorized
in this implementation request; no new runner command was submitted.

## B/F/G complete at 10,000; automatic burns verified (2026-09-30 05:23 UTC)

Read-only audit47508e2 confirms all queue jobs passed, complete.json equals the
successful run receipt, and final comparison passed. Each arm stopped at exactly
10000 / schedule28600 steps,10485760000 cumulative input tokens. Final configs
match except arm; all use the same4882 evaluation rows. LM losses:
B3.0293528655393467, F3.029287303072595, G3.0303955152025916.
F-B=-0.0000655625; G-B=+0.00104265. B/F are essentially tied; G is slightly worse
in this single-seed LM comparison. No clear extra-loss advantage is established.

B finished2026-09-29 20:00:56 UTC, F2026-09-30 00:02:11, G04:22:53;
comparison completed04:22:53. Full checkpoint10000 weights, optimizer, scheduler,
training args and8 RNG files were verified present/nonempty for each arm.
All completion-manifest artifact hashes passed.17 small artifacts retrieved and
source-hash verified under artifacts/BFG-10000-final-20260930/; final configurations,
trainer states, result hashes and F/G weighted-objective identities checked locally.
Large model/optimizer checkpoints remain on B200 in the a02 continuation root.

Supervisor verified automatic burn recovery at04:24:04. Live audit05:23:11–05:23:24
confirmed8 approved workers,100% utilization,155212MiB/GPU, guard released and
advancing collective cycles4000->4020/payload4441.41->4463.61GiB. No handoff error.
Raw log SHA256:24e3faf9b05ac002ff886ecf56d61ac4f8719e30c772ada1d3d956c8d7e44304.
commands.sh restored to#0. No more training is queued; burns are running.

## B complete at 10,000; F running (2026-09-29 20:45 UTC)

Read-only audit018b098 at20:45:12 confirms B finished successfully20:00:56,
exactly10000 steps /10485760000 input tokens, full schedule28600. Held-out LM
loss3.0293528655393467 on4882 rows, down from3.208608250096011 at5000. The
additional5000-step run took3h29m12s including startup/evaluation/checkpointing.

F started20:00:56 and reached5907 by the snapshot; latest saved checkpoint5750.
Throughput~2.85s/update; finite loss/gradients and LR~0.0002824 at5900. All8 GPUs
have one train.py worker,94–99% utilization,127774MiB each. G remains queued at
its staged5000 checkpoint. Supervisor/tmux remain active, no failure reported.
Automatic burn recovery remains configured for queue success/failure.

Four small artifacts and live GPU inspection stored in
artifacts/BFG-status-20260929-2045/; B result source hash matches queue receipt.
Raw log SHA256:8b398e25db28380de288107e01b49f9d936a501f99c6e7631c76e811bb6c1e3b.
No training/process/config changes. commands.sh restored to#0 after inspection.

## B/F/G 10,000-step continuation running (2026-09-29 16:34 UTC)

See BFG_OPTIMIZED_RESUME_20260929.md. Retry launch4daa75c passed the unchanged
real-checkpoint numerical gate for all3 arms. Selected optimization disables
checkpoint_layers/checkpoint_lm/checkpoint_aux only; explicit attention mask and
LM chunks128 remain original. Native HF Trainer/Accelerate is unchanged.

Audit967843b at16:34:12 confirms production B at5040, finite loss/gradient and
continued LR~0.0002884. All8 GPUs have one train.py worker,98–99% utilization,
119192MiB each. Early B throughput~2.5s/update. B started16:31:44; F then G are
queued, each from original5000 to total10000. Full schedule28600/warmup1430,
micro16/accum4/world8/seq2048/data/seeds/EOS packing remain fixed.

Active root: /mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a02.
Production: production/run/continuation/{B,F,G}. Original5000 runs are preserved;
production staging SHA256s match controls/source. All24 optimized rank receipts
match restored model/optimizer/scheduler/RNG and next batches against controls;
next batches match across B/F/G. Parameter relative L2<=4.20e-6, first moment
<=0.002158, second moment<=0.00002041, LM delta<=4.1e-6; gate passed unchanged.

Accelerate resource/cache bytes and eight-GPU BF16 env verified. Previous a01
failure automatically restored verified burns16:16:04; retry freshly reclaimed
those identified workers, with all GPUs empty16:22:52. Existing supervisor will
restore/verify burns after this queue succeeds or fails. No env/driver installs.
49 small artifacts/source hashes verified under artifacts/optimized-resume-20260929-a02/.
Raw startup log SHA256:63dcc975edd3167c4bf7b830ab72858a74de3ddf930040de46611f3a19607f43.
commands.sh restored to#0 after inspection; the active tmux queue continues.
Next: read-only progress/completion checks; do not submit another training queue.

## Checkpoint-only gate passed; production staging (2026-09-29 16:30 UTC)

Retry4daa75c/e31c52d passes the unchanged real-checkpoint gate for B/F/G.
Parameter relative L2: B4.19410e-6, F4.04550e-6, G4.14861e-6 (<1e-5).
First-moment relative L2<=0.002158; second-moment<=0.00002041 (<0.03).
LM evaluation deltas B-4.09e-6/F+1.68e-6/G+1.38e-6 (<0.001). All ranks restore
exact model/optimizer/scheduler/RNG and consume identical next microbatches.
Only checkpoint_layers/checkpoint_lm/checkpoint_aux are disabled. Keep original
explicit attention mask and lm_chunk128; no gate thresholds were relaxed.

Audit16:30:59: gate and all3 optimized smoke jobs succeeded; stage-continuation
running in deep-kv-BFG-10000-20260929-a02. Production uses fresh original5000
copies, not smoke5001 outputs. Schedule28600/warmup1430/micro16/accum4/world8
unchanged; planned sequential B/F/G cutoff10000. Next: verify production B progress.
Retry preflight also verified prior failed-attempt burn handoff at16:16:04 and
all8 GPUs completely free at16:22:52 before the retry. Automatic final burn
recovery still wraps the production queue. Raw passing-gate log SHA256:
70a2aa1817d7e64ecf74862836e0fccb81e0fcaff86710eb63cdea15d4efdd4b.

## Initial optimized gate rejected; checkpoint-only retry prepared (2026-09-29)

See BFG_OPTIMIZED_RESUME_20260929.md. a01's six smoke runs passed all48 exact
restoration/data receipts, but combined B parameter difference1.12609e-5 exceeded
predeclared1e-5. Gate failed16:14:53; no production continuation ran. Automatic
burn recovery started; all8 burn workers observed16:15:59, final handoff pending
at that snapshot.73 source artifacts verified locally. Keep every gate tolerance.
Fresh a02 disables only all3 checkpoint types, retaining original explicit mask
and lm_chunk128. Reuse completed control receipts read-only, rerun3 optimized
checks, then continue from fresh original5000 copies to10000 only if gate passes.
Same8 GPUs/micro16/accum4/schedule28600/warmup1430 and automatic burn recovery.
Next: verify retry preflight/free GPUs, numerical gate, actual production start.

## Authorized optimized B/F/G continuation to 10,000 (2026-09-29)

See BFG_OPTIMIZED_RESUME_20260929.md. Local train.py exposes tested execution
switches with strict opt-in resume metadata and a full transition record. Native
HF Trainer/Accelerate loading/data skip/optimizer loop is unchanged.30 CPU tests
plus2 targeted gate tests passed. Prepared deep-kv-BFG-10000-20260929-a01:
real5000 checkpoints -> six old/optimized one-update checks -> numerical/state/
data comparison -> fresh source5000 copies -> sequential B/F/G to10000.
Micro16/accum4/world8/seq2048, schedule28600/warmup1430 remain fixed. All three
checkpoint switches off, implicit causal SDPA, LM chunks512. Original5000 outputs
remain intact. Automatic verified burn recovery wraps the full gated queue.
Next: submit once, verify preflight/config copy, fresh GPU reclaim, remote gate,
then actual production resume and progress. No remote success claimed yet.

## Checkpoint and capacity investigation complete (2026-09-29 15:01 UTC)

See B200_CHECKPOINT_CAPACITY_20260929.md. Fresh capacity-20260929-a01 completed
41 probes: 25 successful eight-GPU runs and 16 controlled CUDA OOMs. Best matched
1,048,576-token setting is microbatch 16 / accumulation 4 with decoder, LM-loss
and F/G auxiliary-loss checkpointing disabled, causal SDPA and LM chunks 512.
B/F/G times: 2.0906 / 2.4745 / 2.7053 seconds per update; peak allocated memory
111.71 / 113.15 / 121.64 GiB, reserved 112.65 / 116.03 / 124.28 GiB.

Largest passing microbatch with all checkpointing off: B26, F26, G24; next integer
OOMed in each case. Retaining loss checkpointing permits 30 for all three, with
31 failing. These limits leave little headroom and are not fixed-global-batch
production recommendations. Microbatch 32 OOMed without decoder checkpointing.
B with decoder checkpointing and microbatch 64 fits (81.67 GiB allocated), but
is slower at 2.6532 seconds/update than checkpoint-free microbatch 16.

21 CPU tests passed, including 56 arm/toggle combinations for loss, gradients,
restored AdamW moments/next weights and scheduler. Successful probes passed
CUDA BF16 objective/gradient validation on all eight ranks. Matched 18-update
final LM losses differed by at most 0.0000253, not bitwise equality. Original
production CLI/recipe, strict resume checks, environments and drivers remain
unchanged. The new model switches default on; only the benchmark CLI exposes
them. A future scientific resume with new settings still needs a tested,
recorded metadata policy; no production resume was launched here.

Queue completed 14:58:33; automatic enhanced burn handoff verified 14:59:39.
Live audit f353a59 at 15:01:32–15:01:45 confirmed all eight approved workers,
100% utilization, 155212 MiB/GPU, guard released, cycles 150→160 and collective
payload 166.55→177.66 GiB. 429 artifacts/source hashes verified locally under
artifacts/capacity-20260929-a01/. Raw audit log SHA256:
17b85a9b0a2dc1023df59530421eb522dbd4d851c75a02b972e2a18ef76734f5.
commands.sh restored to #0; no additional probes or training queued.


## B200 performance investigation complete (2026-09-29 12:55 UTC)

All14 intended performance cases completed across original performance-20260929-a01
and corrected one-case performance-fa4-20260929-a02. See B200_PERFORMANCE_20260929.md.
At fixed8 GPUs /1,048,576 tokens per update: B3.3662->2.2075s, F4.0102->2.8547s,
G4.3710->3.2174s using no decoder checkpointing, implicit causal backbone mask,
and LM chunks512. Peak~98GiB/GPU. Microbatch32 alone3.2795s (small/noisy gain).
FA4 matched-env2.2281s versus SDPA2.2077s: no speedup. Actual FA4 kernels verified.
Keep Trainer/Accelerate and SDPA; changing the custom model paths gives the gains.
Production train.py/model/training files and scientific checkpoints are unchanged.

FA4 initially failed in its precheck due to this agent's tuple-return adapter bug.
Fixed in a5ede38 with a regression test; all5 CPU tests passed. Fresh retry passed
CUDA BF16 objective/gradient validation and18 real eight-GPU updates. All112 rank
prechecks, global first-batch hashes and measured timings verified across14 cases.
158 small source artifacts SHA256/manifest verified locally under
artifacts/performance-20260929-a01/ and artifacts/performance-fa4-20260929-a02/.
Combined summary is in the first root. Original failed queue receipt is preserved.

Corrected queue completed12:53:42; supervisor burn handoff12:54:53. Live read-only
audit96305a0 at12:55:00-12:55:12 verified all8 approved workers,100% utilization,
155212MiB/GPU, collective cycles30->40/payload33.31->44.41GiB, guard released.
Raw final log temp/perf-fa4-final-20260929-a02.log SHA256
7d4f388de1021f8cc47c6e3abc57530fbd6b2a17432b8be5f9fd6acb85ac5aac.
commands.sh restored to#0. No further training/benchmark queued. A future production
optimization should preserve matching settings across scientific arms; no such
recipe change was made during this performance investigation.


## B/F/G 5,000-update continuation complete (2026-09-29)

Queue deep-kv-BFG-5000-20260929-a01 completed successfully at 11:27:12 UTC.
Each arm trained 5.24288B input tokens in total on the unchanged 28600-step
schedule, same data/order/batch, and same 4882-row final evaluation. LM losses:
B 3.208608250096011; F 3.2087102048666094; G 3.2090661402300626.
F-B is +0.000102; G-B +0.000458. Essentially tied; the extra functional losses
show no observed LM advantage at 5000 updates in this single-seed experiment.
No matched 5000-step Arm A run exists, so do not claim superiority to vanilla A.

Automatic enhanced burns started and passed supervisor verification at 11:28:22.
Read-only live check 75816db at 11:51:18-11:51:30 verified eight approved workers,
100% GPU utilization, 155212 MiB/GPU, advancing cycles/payload and all-rank
collectives. Source and completion-manifest SHA256s verified locally for final
results and comparison. Small receipts/configs/states are retained in
artifacts/deep-kv-BFG-final-20260929/; full weights remain on B200.

## B200 FlashAttention availability (2026-09-29)

Live train_env inspection at 06:13:57 UTC: PyTorch2.14.0+cu130 has built-in
FlashAttention compiled and the flash SDPA backend enabled; memory-efficient
and cuDNN backends are also enabled. Triton3.8.0 and cuDNN9.24.0.43 installed.
No standalone flash-attn/flash-attn-3/flash-attn-4 package or flash_attn module.
Current train.py selects SDPA, so absence of the external package alone does
not imply unfused attention. Actual per-call backend remains unprofiled.
CPU-only inspection 96b2f4e left training/environment unchanged; evidence in
temp/flash-attention-env-check-20260929-a01.log.

## B continuation result at 5,000 updates (2026-09-29)

B completed its additional 2,500 updates at 05:26:49 UTC, exit zero, total
5,242,880,000 input tokens. Held-out LM loss: 3.208608250096011 (previously
3.505886970597037 at step 2500), same 4882 evaluation rows. Result file hash
matches the runner receipt; state reports global_step5000/max_steps28600.
F then started automatically; read-only check ae585c5 at 05:33:38 UTC found
step2589 and all eight GPUs training at 99% utilization. G remains queued.
Evidence: artifacts/deep-kv-BFG-status-20260929-0533/ and
temp/BFG-status-20260929-0532.log. No 5,000-step cross-arm comparison yet.

## Native checkpoint continuation (2026-09-29)

User authorized B/F/G to continue from update2500 to total5000, retaining the
28600-update cosine schedule and1430-update warmup. A fresh sequential queue
uses scripts/stage_deep_kv_resume.py to copy full checkpoints plus saved
train_config.json; prior result.json becomes previous-result.json. Every
checkpoint file is checksum-verified. This preserves original outputs while
satisfying train.py's checkpoint-parent/output-directory check and the generic
runner's fresh-output contract. No weights-only restart or schedule reset.
tests/test_resume_staging.py exercises actual B/F/G Trainer continuation and
matches uninterrupted CPU weights exactly, preserving all source files.
Launch0ce6bff ran on B200; at03:07:21 UTC B reached2582 with finite loss and
LR0.0002988. F/G follow sequentially, each cutoff5000. Original checkpoint
files were all checksum-verified against their copies. Accelerate config/env
and the all-eight-GPUs-free prelaunch receipt passed. Small evidence is in
artifacts/deep-kv-BFG-resume-startup-20260929/; the unchanged supervisor will
restore/verify enhanced burns after completion or failure.


## F/G complete; automatic final burns verified live (2026-09-29 02:40 UTC)

User explicitly requested another read-only check after the SSH failure.
Retry 1f98d6f reached thiennh-p6-78gg-worker-0; access is restored. All four jobs
(smoke, F, G, compare) exited zero and complete.json is present. F finished
Sep28 22:59:58 UTC (2h52m09s); G finished Sep29 02:06:05 UTC (3h06m07s).
Both stopped at exactly2500/28600 updates, 2621440000 input tokens, as intended.
Both saved configurations match original D after removing only pilot.arm;
train/eval fingerprints, tokenizer, seed, packing, batch and schedule match.
Local receipt/result hashes match the queue's completion manifest. All saved
steps, token counts, 4882 eval rows and weighted loss identities validate.

Final held-out LM loss (lower is better):
A 3.510730336181336; B 3.505886970597037; C 3.5791965769364404;
D 3.5259151358008247; E 3.514632263741271;
F 3.5103442129966322; G 3.5105305372349864.
F/G are essentially at A, improve over D/E, and remain worse than B. G-F is
+0.0001863242383541852. Single-seed exploratory results; tiny differences do
not establish a robust gain. F route0.1812025474324223/message0;
G route0.18460066779232728/message0.006506484034108609.

Supervisor completed automatic handoff at 02:07:16 UTC with all eight GPUs
verified free before starting the enhanced burn; no handoff error. Read-only
inspection at 02:40:26-02:40:38 verifies eight approved burn workers, one per
GPU, 100% utilization and155212MiB/GPU. All-rank readiness/collective probe
passed; rank0 cycles advanced2240->2250 and payload2487.19->2498.29GiB.
No workload was stopped or launched by this status check.

Small original artifacts and live verification are retained in
artifacts/deep-kv-FG-final-20260929/ (recipe, supervisor, queue/completion,
comparison, F/G results/configs/trainer states, live-burn-check.json).
Raw export: temp/FG-final-check-20260929-0240.log. Large model/optimizer
checkpoints remain on B200 and were not downloaded. commands.sh restored to #0.

## Latest B200 check blocked by runner SSH access (2026-09-29 02:11 UTC)

User requested a fresh completion/status check. Read-only submission a9619f9,
job th2-78gg-FG-completion-check-20260929-0210, failed before a project log was
returned. Controller record at 2026-09-28 19:11:03 (UTC-7; 02:11:03 UTC Sep29):
FAILED(rc=255), <host>: Permission denied (publickey).
Evidence: temp/FG-controller-20260929-0211.log.
No current G result or final-burn verification was obtained. This is an access
failure, not evidence that training failed or the node died. Last verified
observation remains Sep28 23:03:48 UTC: F complete2500, G runningstep43, eight
active GPUs. Do not resubmit, reclaim, or repair infrastructure in response;
wait for operator restoration of runner SSH access, then retrieve final queue,
F/G results, supervisor and fresh live burn progress. commands.sh restored to
#0; no training/data/process changed by this check.

## F complete; G running (2026-09-28 23:03 UTC)

Read-only status ac31b69 on thiennh-p6-78gg-worker-0 at 23:03:48 UTC confirms
F exited zero at 2500 updates / 2621440000 input tokens, finished 22:59:58 UTC
(10328.79 seconds, about 2h52m). Its result status stopped is the intended cutoff;
trainer_state global_step2500/max_steps28600; checkpoints2250 and2500 remain.
Final eval: LM=3.5103442129966322, route=0.1812025474324223, message=0,
objective=3.564704977226359; 4882 rows / 9993454 target tokens.
Runner recorded F/result.json SHA256
160830425acb17a1c0cae4c497b9dd1cfc7356103b83a645806a7fc9f610c8d4.

G started automatically at 22:59:58 UTC; latest log step43/2500, about4.36s/update.
Step40 LM10.55, route0.2557, message0.2672, finite gradient norm2.144.
All eight GPUs show one worker each, 95-100% utilization, 26416MiB per GPU.
Supervisor/queue status running; final comparison and burn receipts not present,
as expected while G trains. Estimated finish around 2026-09-29 02:05-02:15 UTC,
subject to throughput/checkpoint/evaluation overhead. Automatic burn handoff
remains configured; do not report it verified before completion.
Evidence: temp/FG-status-20260928-2303.log (SHA256
d2cf6e062f5ed88991b3b0218ab54cc99a1906e403461ac93957b5ffb44f0b14).
No process/data/training code changed. commands.sh restored to #0.

## B200 dataset format checked read-only (2026-09-28 22:26 UTC)

User paused HF upload planning to clarify formats and repository layout.
Read-only command 04891fc on thiennh-p6-78gg-worker-0 confirmed
/mnt/local/_data/deep-llms_th2/data/raw contains exactly 75 .parquet files.
Sample footer: ar/ar_part_00002.parquet, 578336 rows, 12 row groups,
columns text/timestamp/url/source, all strings. No text rows were exported.
Prepared English train contains 35 save_to_disk dataset directories; eval one.
Their state.json references data-*.arrow; dataset_info.json exposes text:string.
The directories also contain training-generated cache-*.arrow, which are not
source dataset files. Current train.py uses load_from_disk on sorted shards.

Repository count and storage format are independent: either one repository or
per-language repositories can preserve save_to_disk folders for the same loader.
Parquet would instead require local load_dataset('parquet', data_files=...).
Runner #d --url supports pinned individual HF file URLs, while --hf-dataset has
no documented subset/split filter. Partial folder downloads must include all
files referenced by state.json plus dataset_info.json. Preserve explicit shard
order and token/source indexes for reproducible prefix selection.
Evidence: temp/b200-data-format-20260928-a01.log (SHA256
7689d3e1e600da984d70914fece11e69fc370c840340c099f48f6cd4fe1bdf5b).
No training code/data/process was changed and no HF upload started.
commands.sh restored to #0 after this inspection.

## F/G production running after successful B200 smoke (2026-09-28)

Launch commit 26e8e7fa5ca1ebf71a424577ec9a946ddb32d555 was accepted on
thiennh-p6-78gg-worker-0 at 19:57:48 UTC. Fresh preflight at 19:58:00
verified train_env, copied resources/accelerate_config.yaml to the actual tmux
cache /dev/shm/.cache/huggingface/accelerate/default_config.yaml, compared
bytes/SHA256, and ran accelerate env (MULTI_GPU, eight processes, BF16).
Only freshly identified approved burn workers were stopped via pinned pidfds.
At 19:59:05 all eight B200s had zero compute PIDs and zero allocated MiB.

Root: /mnt/local/_outputs/deep-llms_th2/deep-kv-FG-2500-20260928-a01.
Tmux session: deep-kv-FG-2500-20260928-a01. Production outputs: production/run/F
and production/run/G. Do not relaunch this queue or reuse this output root.
D/F/G smoke passed at 20:07:48 UTC: each arm trained to 10, resumed its own
checkpoint to 12, and passed loss/count/checkpoint/all-rank memory validation.
Smoke receipt SHA256:
d8ff35e8744227ee388a6e7f71ce9d2f6256d24de94f71e275378869294e8ab8.
The retrieved receipt matches the outer runner's artifact hash.

Smoke steps 3-10: D 4.278, F 4.918, G 5.262 seconds/update; F/D=1.150,
G/D=1.230. Maximum reserved memory across initial/resumed runs was under
24 GiB per GPU, with over 8 GiB headroom on every rank. All objectives finite;
F message loss exactly zero, G message loss positive, both routing query counts
129*2047=264063. Smoke losses are implementation checks, not scientific results.

Fresh production F started at 20:07:48 UTC. The 20:10 export verifies step 19,
about 4.0 seconds/update, finite step-10 losses/gradients and W&B offline.
Its saved train_config.json equals original production D after removing only
pilot.arm, including train/eval fingerprints adb88539d2924dc2/0568ce654afc3fdb.
G starts automatically after successful F, then F/G result validation/comparison.
Both use eight GPUs, microbatch16/accum4, 1048576 tokens/update, seq2048,
EOS document boundaries, 2500-update cutoff, full schedule28600/warmup1430.
Fresh output roots prevent smoke or prior-arm checkpoints initializing F/G.
Provisional combined duration about six hours (around 2026-09-29 02:00 UTC),
subject to production throughput/checkpoint/evaluation overhead.

The unchanged train_then_burn supervisor owns cleanup and automatic enhanced
burn restoration after success or failure, then verifies all eight ranks and
advancing collectives. Final burn for THIS queue has not yet occurred/been
verified. Expected session: deep-kv-FG-2500-20260928-a01-final-burn.
Next check: production/run/run.json, F/G logs/results, final complete.json,
comparison.json, production/supervisor.json and burn-verified.json/burn.log.
Small receipts/logs retained under artifacts/deep-kv-FG-launch-20260928/.
commands.sh returned to #0 after verification; local sampling remains untouched.

## F/G smoke and production launch authorized (2026-09-28)

User authorized new arms on B200 using the established launch workflow.
Applied the reviewed deep-key target cast to attention value dtype for F/G only;
60 CPU tests pass, including a new BF16 target-precision regression. Three smoke
gate tests pass. Production train.py and original recipe remain unchanged.
Fresh root: /mnt/local/_outputs/deep-llms_th2/deep-kv-FG-2500-20260928-a01.
Queue: production-shape D/F/G smoke (10 updates, resume to 12), then fresh F
and G at 2500/28600 updates, followed by FG result validation/comparison.
Smoke-only callbacks collect clean step timings and all-rank memory peaks;
minimum 8 GiB headroom required. The outer unchanged train_then_burn supervisor
reclaims only freshly verified burns and restores them after success or failure.
Accelerate config copy/byte verification + accelerate env precede reclamation.
All runs use eight GPUs, micro16/accum4, seq2048, EOS packing; W&B offline and
NCCL_NVLS_ENABLE=0. Commands and code are ready for submission; remote status
not yet verified. Local corpus sampling is untouched.

## F/G independent recheck passed (2026-09-28)

Re-read the functional-loss specification and reviewed forward captures,
KL direction/weights, head/query/feature normalization, strict-past masks,
query/teacher detach boundaries, BF16 math, checkpoint recomputation, Trainer
accumulation, per-example evaluation statistics, resume and queue/report logic.
No implementation bug found; no production-code change needed.

Added a regression against an independent per-query FP64 oracle with 259
positions, 16 query heads / 8 KV heads / head_dim=128, noncontiguous BF16
inputs, padding and segment boundaries. This crosses both default 128-query
chunk boundaries, exercises both SmoothL1 regions, and checks nonzero FP32
losses and backward recomputation outside autocast against reference gradients.
All 59 local CPU tests pass: temp/deep-kv-FG-recheck-all-20260928.log.
The existing eight-process BF16 micro16/accum4 F/G resume checks remain valid;
implementation is unchanged from f6ce7e5. train.py, packing, data recipe,
schedule, Accelerate config and runner commands are unchanged. No B200 work
was submitted. Actual B200 capacity/throughput still requires a GPU smoke test.

## Functional-loss arms F/G implemented and locally validated (2026-09-28)

Implemented docs/deep_route_kl_variant.md. F = LM + .3*KL(deep||pred);
G = LM + .3*KL(deep||pred) + .3*SmoothL1(message_pred,message_deep), beta=1.
No /2 and no raw K/V reconstruction in F/G. Same Arm D forward, parameters,
initialization and shallow/deep pair. Reuse exact rotated shallow queries and
predicted keys from the live auxiliary attention; native rotated block-21 K/V
supply detached targets. The query is detached only in auxiliary losses.
Strict-past/backbone masks and GQA mapping are shared; empty rows are excluded
before softmax. FP32 loss math, 128-query checkpointed chunks (all source keys),
valid-query/head normalization, plus head-feature averaging for messages.
Unique query slots and a regular sum avoid repeated-index CUDA atomic reduction.

HF Trainer/Accelerate integration logs route/message losses separately and
uses per-example query counts for distributed evaluation. Existing A-E metrics
and formats preserved. train.py, packing/EOS, recipe, seeds, schedule, batch,
Accelerate resource and launch shell are unchanged. --arms F G generates the
sequential F/G queue and result validation; reports include F-B, G-F, G-B plus
prior controls when present. Default queue remains A/B/C/D. No new loss knob.

Validation: all 58 local CPU tests pass (temp/deep-kv-FG-final-tests.log).
Independent dense oracle covers KL direction, GQA, masking/empty rows, message
beta/normalization, gradients and chunk recomputation. Checks cover no raw loss,
no auxiliary query/deep-target/F-value gradients, retained LM gradients,
unchanged forward after branch output becomes nonzero, native one-pass captures,
BF16, exact F/G coefficients, Trainer accumulation, cache order and resume.
Eight-process BF16 CPU full-vs-resume tests with micro16/accum4 and uneven 5-row
eval pass for F/G: max parameter deltas 2.33e-10 / 4.66e-10. Receipt:
temp/deep-kv-FG-resume-20260928-a02/resume_verified.json; log:
temp/deep-kv-FG-distributed-final.log. Independent 28-layer tiny-width audit
against pre-change committed D passes bitwise raw outputs/gradients in FP32
and BF16; F/G initial parameters and LM outputs match original D:
temp/deep-kv-FG-vs-original-D-audit.json. Generated queue list validated locally
at temp/deep-kv-FG-jobs-20260928.json (dev paths, regenerate on B200 for launch).

No B200 process was changed or new training launched. commands.sh remains #0.
GPU capacity/throughput of the new losses has not yet been measured; use the
real B200 recipe in a smoke test before a long run. Implementation remains local.

## Arm E completed; automatic final burn verified (2026-09-28)

Read-only check 7ce7ce9 at 18:41:18-18:41:30 UTC verifies E completed the
2500-update cutoff successfully at 16:01:48 UTC (2h24m50s elapsed), with
2,621,440,000 input tokens and the unchanged 28600-step schedule. Training and
E-result validation both exited zero. Result status "stopped" is the intended
fixed-schedule cutoff. Final trainer state is global_step=2500/max_steps=28600.
Both artifact hashes match complete.json remotely and after local retrieval.
E's saved configuration still matches D after removing only pilot.arm.

Final held-out LM loss: E=3.514632263741271, D=3.5259151358008247,
A=3.510730336181336, B=3.505886970597037, C=3.5791965769364404.
E improves D by .01128287 but remains .00390193 worse than A and .00874529
worse than B. Same ~10M-token evaluation set (4882 contexts). Single-seed
exploratory result; lowering alignment weight helped relative to D, without
beating baseline or the branch-only B. E weighted objective=3.639359351846798;
raw K=.47293925608269066 and V=.35857466462082377.

Automatic burn handoff succeeded at 16:02:59 UTC. Fresh inspection confirmed
one reviewed burn worker on every B200, 100% utilization, ~155212 MiB/GPU.
Collective cycles advanced 10720 -> 10730 and logical payload per rank
11902.97 -> 11914.07 GiB during the 12-second check. Guard hold was removed;
no handoff error. No process was stopped or training launched by this check.
Local results and receipts: artifacts/deep-kv-E-final-20260928/ (ignored).
Large checkpoints remain on B200 in the E a02 production/run/E directory.
commands.sh restored to #0; no new experiment queued.

## Arm E is running; launch verified (2026-09-28 13:39 UTC)

Launch commit 887b64b, fresh root:
/mnt/local/_outputs/deep-llms_th2/deep-kv-E-2500-20260928-a02
Tmux session deep-kv-E-2500-20260928-a02; supervisor under production/.
E queue started at 13:36:58 UTC. Export ae1324d at 13:39:18 UTC shows live
optimizer updates through step 23, no traceback/nonfinite/OOM/child failure.
Initial held-out LM loss ~12.12; step-20 LM loss ~11.44. These are startup
observations, not final results. About 3.3-3.5 seconds/update; expected finish
around 16:02 UTC based on D's ~2h25 runtime, subject to checkpoint/eval overhead.

Verified ordering: resource Accelerate config copied to the actual tmux cache
/dev/shm/.cache/huggingface/accelerate/default_config.yaml, byte/hash matched,
then accelerate env confirmed MULTI_GPU / 8 processes / BF16. Supervisor
freshly verified burn identities and source hashes, pinned PID handles, stopped
only eight burn workers (80257-80264), waited 30 seconds, and recorded all
eight B200 GPUs empty (zero compute PIDs, zero used memory) at 13:36:58 UTC.
Training started after that receipt and the runner's additional free check.

E's saved train_config.json matches completed D exactly after removing only
pilot.arm; includes model, seeds, full training schedule, data fingerprints
(train adb88539d2924dc2; eval 0568ce654afc3fdb), world_size=8 and
1,048,576 tokens/update. E coefficient .3, D=1; no checkpoint resume.
2500 cutoff / 28600 full schedule / 1430 warmup / micro16 / accum4 / seq2048;
EOS packing and HF caches unchanged. NCCL_NVLS_ENABLE=0, W&B offline.

Only E then E-result validation is queued. Unchanged train_then_burn supervisor
runs owned-child cleanup and verifies free GPUs before independent final burn
on success or training failure. Final session:
deep-kv-E-2500-20260928-a02-final-burn. Its existing handoff was verified on
both success/failure for A-D; E's future handoff is configured, not yet observed.
Check production/supervisor.json, burn-verified.json and advancing burn.log
collectives after completion; do not launch another job or burn concurrently.

Receipts/log/config comparison: artifacts/deep-kv-E-launch-20260928/ (ignored).
a01 failed only a preflight version assertion before any GPU management or
training; preserved separately. commands.sh restored to #0. Local corpus
sampling was not touched. Next action: read-only E status/result retrieval.

## Arm E preflight version assertion corrected (2026-09-28)

Attempt a01 (b923b68) stopped at the first preflight version assertion, before
Accelerate copy, GPU inspection/reclamation or training. The package metadata
reports torch=2.14.0; torch.__version__ includes the CUDA suffix 2.14.0+cu130.
No training output/cache was produced. Log: temp/arm-E-a01-launch.log.
Corrected those separate checks, preserving all training code and recipe.
Retry uses fresh root deep-kv-E-2500-20260928-a02 and fresh tmux session;
a01 is preserved. Waiting for remote a02 verification.

## Arm E launch authorized (2026-09-28)

User authorized E on all eight B200 GPUs, with Accelerate copy/verification,
verified burn-worker reclamation, free-GPU checks and automatic final burn.
Submitting fresh root /mnt/local/_outputs/deep-llms_th2/deep-kv-E-2500-20260928-a01.
The E-only queue reuses scripts/train_then_burn.py unchanged; its failure and
success handoffs were verified on this same node during the completed A-D run.
Bootstrap checks the original A-D recipe and GPU UUIDs, copies the Accelerate
config in the actual tmux context and runs accelerate env before reclamation.
Only the coefficient changes: E=.3, D=1. Same 2500 cutoff, 28600 schedule,
1430 warmup, microbatch 16, accumulation 4, eight GPUs, seq2048 and EOS packing.
Fresh E output; no D checkpoint resume. Remote launch verification pending.

## Arm E recheck against original D passed (2026-09-28)

Rechecked E against D and the pre-E D implementation from a23032c. On a
28-layer tiny-width CPU model with consumer 5 / target 21 and nonzero branch
output weights, original D, current D and E have bitwise-identical initial
parameters and raw forward statistics. Original/current D gradients also match
bitwise. Explicit tests pin coefficients D=1.0 and E=.3 and verify that E's
gradients equal LM gradients plus .3 times D's alignment gradient contribution.
Generated D/E training argv are identical except arm/output_dir/run_name;
GPU allocation is identical. train.py, recipe, packing, Accelerate config and
shell training launcher have no changes from a23032c. No runtime fix was needed.
Strengthened regression tests; all 54 local CPU tests pass.
Evidence: temp/deep-kv-E-vs-original-D-audit.json and
 temp/deep-kv-E-recheck-all-20260928.log. E remains local and unlaunched.

## Arm E implemented locally: D with alignment weight 0.3 (2026-09-28)

User requested E as an otherwise identical D. E follows exactly D's native
block-21 target path, consumer block 5, shared seed/initialization, architecture,
stop-gradient and training recipe. Its objective is LM + .3*(K_L1+V_L1)/2;
D/C remain weight 1. Weight is fixed by arm identity; no new arbitrary recipe
knob or training loop. Trainer loss, evaluation objective and statistics use
the same coefficient; separate K/V metrics remain unweighted. Existing A–D
checkpoint/config formats remain compatible and D-to-E resume is rejected.

Queue/report support explicit --arms E, --arms D E, or all five. Default remains
A/B/C/D. E-only queue uses the existing base recipe (2500 cutoff, 28600 schedule,
1430 warmup, microbatch 16, accumulation 4, all eight GPUs) and a fresh run root.
Only E and its result validation run; earlier arms are not implicitly retrained.
Generated example: temp/deep-kv-E-jobs-20260928.json (list validated locally).

Validation on dev CPU in sampling_b200: all 15 model/training tests passed,
including E native deep targets, shared initialization, unchanged LM path,
.3 auxiliary gradient scaling, gradient accumulation, cache/data order,
weighted evaluation, real train.py cutoff/save/resume and report selection.
Eight-process BF16 E test (microbatch 16, accumulation 4) passed uninterrupted
versus resumed training, maximum parameter difference 9.313225746154785e-10;
uneven five-row evaluation retains the .3 objective in both modes.
Logs: temp/deep-kv-arm-E-tests.log and temp/deep-kv-E-distributed-resume.log.
Receipt: temp/deep-kv-E-resume-20260928-a01/resume_verified.json.
Changes are local, not deployed. commands.sh remains #0; B200 and local corpus
sampling were not modified. This request implemented E; it did not launch it.

## Four-arm run completed; live burn verified (2026-09-28)

Read-only node inspection 55c39ac at 12:49:43–12:49:56 UTC confirms all four
arms finished successfully at global_step=2500 with max_steps=28600. Queue
and comparison completed at 10:45:37 UTC (9h 51m 58s since 00:53:39 launch,
including preprocessing). Every job exited zero. All five artifact hashes
match complete.json, and A/B/C/D saved configurations match after removing arm.

Final held-out LM losses on the same ~10M-token evaluation set:
A=3.510730336181336; B=3.505886970597037;
C=3.5791965769364404; D=3.5259151358008247.
D beats C by .05328144 but is worse than A by .01518480 and B by .02002817.
B is slightly better than A by .00484337. deep_gain_pattern=false. This is
one seed and a 2500-update cutoff, not evidence of a robust method advantage.

Automatic handoff succeeded at 10:46:48 UTC. Fresh inspection at 12:49 UTC
verified one reviewed enhanced-burn worker on each of all eight B200 GPUs,
100% utilization and ~155212 MiB occupied per GPU. During a 12-second check,
completed collective cycles advanced 8330 -> 8350 and logical payload per rank
9249.23 -> 9271.44 GiB. The supervisor exited successfully; guard hold removed.
No training or process-management changes were made by the status check.

Local results: artifacts/deep-kv-final-20260928/ (ignored): all four result.json,
comparison.json, complete.json, and full fresh node inspection. Reconstructed
individual JSON bytes independently match the remote runner's recorded hashes.
Large weights/checkpoints remain on B200 under the existing production/run root.
commands.sh restored to #0. No further experiments have been launched.

## Status snapshot at 07:17 UTC (2026-09-28)

Export 394ddf7 confirms A/B completed 2500 updates with exit code zero.
B finished at 05:55:43 UTC (8698 seconds elapsed); final held-out LM loss
3.505886971 versus A 3.510730336. B result SHA256 matches run.json:
652fc740148b1d216fcc918d35fc3901dc7d67dfada060fa8a5bd84875fc5157.
Saved A/B configurations differ only in pilot.arm, including identical train
and eval fingerprints. This is a small observed difference, not a robust
method conclusion; C/D final results remain pending.
C started automatically at 05:55:43, reached step 1397 in the fresh log;
latest logged objective 4.290, step LM loss 4.205, K loss .1015, V loss .08786
at step 1390. No traceback, CUDA OOM, ChildFailedError or NaN found in C log.
Speed ~3.4–3.5 seconds/update. D remains queued. Estimated C finish ~08:20–08:25
UTC and total queue ~10:45–11:15 UTC if D maintains similar speed.
No final burn handoff yet because training continues. Status checking made
no changes to training; commands.sh restored to #0.
Receipts: artifacts/deep-kv-status-20260928-0717/ (ignored).

## Status snapshot at 04:19 UTC (2026-09-28)

Fresh export 8e6af30 from deep-kv-2500-20260928-a02 confirms Arm A exited
zero and completed its requested cutoff of 2500 updates at 03:30:44 UTC.
A result SHA256 matches the runner receipt:
cd9ca808342131d96ca2130f47d3a26a9ec0db1e8b761550248eda59323b7d19.
Its held-out LM loss is 3.510730336 on 4882 rows / 9,998,336 input tokens;
input training budget 2,621,440,000, full schedule retained at 28600.
Arm B began automatically at 03:30:44 and reached step 835 in the exported
log. Recent speed ~3.4 seconds/update; logged training loss 12.07 at step 10
to 4.937 at step 830. No traceback, CUDA OOM, ChildFailedError or NaN found
in B log. C/D remain queued; no matched final method comparison yet.
B log has no repeated tokenizer-map progress (consistent with cache reuse).
Production supervisor has no final handoff yet, as the queue is running.
Receipts: artifacts/deep-kv-status-20260928-0419/ (ignored).
Status check made no training changes; commands.sh returned to #0.

## Real four-arm queue launched and handoff verified (2026-09-28)

At 00:53:39 UTC, Arm A started under the detached session
`deep-kv-2500-20260928-a02` on thiennh-p6-78gg-worker-0. Its log confirms eight
DDP ranks and active full-English tokenization through train.py (36,595,514
documents, 160 map workers). Optimizer updates and exact full packed capacity
are not yet verified; startup performs the normal capacity check after packing.
B/C/D and the comparison follow sequentially. All arms use cutoff 2500,
schedule 28600, warmup 1430, microbatch 16, accumulation 4, sequence 2048,
EOS boundaries, and 1,048,576 input tokens/update (2,621,440,000 per arm).

Accelerate resources config was copied and verified with accelerate env in
both the preflight shell and the actual tmux environment. The latter resolves
its HF cache to /dev/shm/.cache/huggingface/accelerate/default_config.yaml;
both configs report MULTI_GPU, eight processes, BF16. NCCL NVLS=0, W&B offline.

A deliberate queue failure on B200 proved the automatic burn handoff at
00:52:17 UTC: all eight ranks ready, collective probe sum 36, advancing cycles
and collective payload, ~85% device memory. It remained alive after its
supervisor exited. Only those freshly identified burn workers (52125–52132)
were then signaled via pidfds; all eight GPUs were verified free at 00:53:39
before Arm A. These PIDs are historical evidence, never reusable stop targets.
The first rehearsal had correctly blocked on interleaved unbuffered log lines;
removing unbuffered burn output fixed verification without changing the burn.

Production supervisor runs the existing run_experiments.py queue and restarts
an independent burn on success or training failure, after cleaning only its
owned descendants and verifying free GPUs. A first-arm failure stops the queue;
it never publishes success. If GPU ownership/cleanup cannot be verified,
it records handoff_error and keeps the guard disabled instead of competing.
Machine loss or killing the supervisor with SIGKILL cannot execute cleanup.

Remote root: /mnt/local/_outputs/deep-llms_th2/deep-kv-2500-20260928-a02
- Production: production/run/{A,B,C,D}, production/run/run.json and arm-*.log.
- Handoff state: production/supervisor.json; final burn: production/burn.log
  and production/burn-verified.json, tmux session with suffix -final-burn.
- Main pipeline log: the remote root path plus .log.
- Local launch receipts: artifacts/deep-kv-2500-launch-20260928/ (ignored).
- Validation: 53 CPU tests passed locally; 5 handoff tests passed on B200;
  live failure-to-burn rehearsal and eight-rank launch verified.

commands.sh is reset to #0 after launch; this does not stop detached training.
No final experiment results or optimizer-step progress are claimed yet.

## Authorized real four-arm launch at 2,500 updates (2026-09-28)

User increased the cutoff to 2500 updates per arm (2,621,440,000 input tokens),
keeping full schedule 28600, warmup 1430, microbatch 16, accumulation 4, eight
GPUs and context 2048. Updated the default recipe and queue examples. User also
requires automatic GPU burn after either successful completion or training
failure. Prepare a persistent supervisor that waits for the training processes
to release GPUs, then starts and verifies the idle burn. Fresh preflight must
copy resources/accelerate_config.yaml to the actual HF cache and run accelerate
env; stop only freshly verified burn workers and verify all eight GPUs free.
Preflight bbdbb2f passed on host thiennh-p6-78gg-worker-0: 8 B200s, expected
train/eval fingerprints, correct pinned environment, Accelerate cache copied
and accelerate env verified. Next command launches a detached pipeline: a
controlled failure proves automatic eight-rank burn restoration, then the real
A/B/C/D queue runs 2500 updates each. scripts/train_then_burn.py uses the existing
runner and cleans only its own children (including orphaned descendants) before
starting an independent tmux burn. It verifies all-rank readiness and advancing
collectives. Queue failure remains failure in run.json/supervisor.json.
Output root: /mnt/local/_outputs/deep-llms_th2/deep-kv-2500-20260928-a01.
First live handoff rehearsal correctly held training because Python -u
interleaved rank readiness log lines, although the burn was live and advancing.
Corrected only the burn launch to buffered stdout with explicit flush in the
existing burn source. Retry root deep-kv-2500-20260928-a02 first verifies the
failed rehearsal is terminal before releasing its exact guard hold. Training
has not yet started. Local suite: 53 passed; B200 handoff CPU tests: 5 passed.

## Full-schedule 5% warmup selected (2026-09-28)

User explicitly requested warmup based on the full training schedule to mimic
later full training. deep_kv.b200.json now uses warmup_steps=1430 (5% of 28600),
superseding the inherited 500-step setting. The selected cutoff remains 2000;
cutoff does not shorten the scheduler or warmup. Verified all four generated
queue commands carry warmup 1430, full schedule 28600 and cutoff 2000.
This recipe change is local; no remote launch in this follow-up.

## Launch environment and standalone test import fixed (2026-09-28)

Compared with the original Qwen shell script at 63bcc61: NCCL NVLS defaults to
0 and W&B stays offline. train.py now sets the NCCL default before distributed
imports, covering generated queues/direct launches; explicit NCCL overrides are
preserved. Existing shell exports and the deep2shallow W&B project remain.
The test sibling import now uses tests.test_deep_kv. Both local invocation modes
passed: six module tests and 48 discovery tests, including a fresh-process
launch-environment check. Logs: temp/deep-kv-launch-env-module-tests-20260928.log
and temp/deep-kv-launch-env-discovery-tests-20260928.log. Changes are local;
no remote launch or deployment in this follow-up.

## Four-arm B200 smoke completed and verified (2026-09-27)

All four arms match the active specification and passed the full-model CUDA/NCCL
smoke on eight B200s per arm: 28 layers, context 2048, BF16, microbatch 16,
accumulation 4, 1,048,576 input tokens/update. Each arm stopped at update 2,
then resumed the native model/optimizer/scheduler/RNG checkpoint to update 3.
Each consumed 3,145,728 input tokens and evaluated exactly 129 contexts
(264,192 input / 264,063 LM-target tokens), including uneven distributed shards.
A/B alignment losses are zero; C/D K/V losses and all LM losses are finite.
Both matched-arm comparisons passed. This short smoke is not scientific evidence
for the method; the production recipe remains 28,600 schedule / 2,000 cutoff.
Full-corpus packed capacity is still checked at real-training startup.

The initial launcher hit missing os.pidfd_open in B200 conda Python before any
signal. The tested Linux PID-handle fallback fixed this; 7dae5e3 successfully
stopped only reverified burn workers 501–508, left PID 1/launcher untouched, and
verified all GPUs free before launching. The repo Accelerate config was copied
to the actual default cache and accelerate env confirmed eight-process BF16.
B200 runtime: torch 2.14.0+cu130, Transformers 5.9.0, Accelerate 1.13.0.

Postflight 3b2a6a1 at 23:38:25 UTC verified every arm's checkpoint weights,
optimizer, scheduler and eight RNG files. Eighteen downloaded result/config/state
artifacts match source SHA256. Local evidence: artifacts/deep-kv-smoke-20260927/
(including smoke_complete.json and postflight.json); remote root:
/mnt/local/_outputs/deep-llms_th2/deep-kv-refactor-smoke-20260927-a01.
The owned guard marker was removed, restoring the controller's idle policy.
Latest GPU inspection showed all eight free, with no compute PIDs; this is NOT
evidence that an idle burn has already restarted. commands.sh is now inactive.
No full 2,000-step pilot or corpus preparation job was launched/changed.

## Smoke launcher compatibility correction (2026-09-27)

The first smoke submission 7b01e3f failed BEFORE any GPU signal or training:
B200 conda Python does not expose os.pidfd_open. The runner burn remained active;
the safety wrapper left its owned guard-disable marker pending review. Evidence:
temp/deep-kv-refactor-smoke-20260927-a01-initial.log. Added libc PID-handle fallback
for Python builds missing either PID-handle function, tested against an owned
local child process (including stale-handle rejection). Retry job a02 adopts only
this task's previous marker after verifying its launcher has exited, and repeats
all fresh PID/UUID/start-time checks. It keeps the same fresh a01 smoke output
path, which the first attempt never created. Model/training code is unchanged.

## B200 preflight passed; full-model smoke submission (2026-09-27)

Commit 7b59f57 preflight succeeded on thiennh-p6-78gg-worker-0 at 23:10 UTC.
Runtime: torch 2.14.0+cu130, Transformers 5.9.0, Accelerate 1.13.0, Datasets
4.8.5. Eight B200s and sampled English text are present (36,595,514 train /
11,822 eval documents). Copied resources/accelerate_config.yaml to
/mnt/local/.cache/huggingface/accelerate/default_config.yaml and verified via
accelerate env: MULTI_GPU, eight processes, BF16. 24 TiB disk free.
Evidence: temp/deep-kv-refactor-preflight-20260927-a01.log.

Inspected runner burn workers 501–508 under launcher 434; no training process
was observed. The authorized smoke command pins and rechecks these identities
with pidfds, signals only those workers, waits 30 seconds and requires free
GPUs. No process-group or name-based kill. A temporary guard-disable marker
prevents co-burn; restore only the owned marker after confirming GPUs are free.

scripts/smoke_deep_kv_b200.sh runs full 28-layer Qwen, context 2048, microbatch
16, accumulation 4, BF16, eight GPUs per arm. A fresh 20,000-document train /
2,000-document eval text subset uses normal HF packing/cache. Smoke-only full
schedule is four updates: run each arm to two, compare, resume each to three,
compare again; verify 129 eval rows and eight rank RNG checkpoints. This is a
runtime check, not a scientific pilot or evidence of method quality. Expected
output: /mnt/local/_outputs/deep-llms_th2/deep-kv-refactor-smoke-20260927-a01/
smoke_complete.json. No completion claimed until retrieved and verified.

## Follow-up correctness review (2026-09-27)

Resume is restricted to native checkpoints within the same arm's output
directory, with matching full schedule; B/C/D have identical tensor names so HF
weight loading alone cannot detect a wrong-arm restore. Cutoff and LM chunk
arguments require valid integers. Queue list arguments are emitted as CLI values;
the pinned HF report_to CLI accepts one integration, including a singleton list
in the recipe (an empty list disables reporting). All 46 local tests passed;
evidence: temp/deep-kv-refactor-review-tests-20260927.log. Packing is unchanged.
No B200 deployment or corpus preparation change during this review.

## Active: minimal four-arm training refactor (2026-09-27)

The four-arm specification is the active task. User requested removing the
unnecessary PCC infrastructure and using the proven train.py flow, changing
only the custom model/loss and required experiment behavior. train.py now owns
HF argument parsing, cached text preprocessing, seeded shuffle, Trainer
training/evaluation, and native checkpoint/resume. deep_kv holds the model,
unchanged packing helper, small Trainer adaptations, queue generator and report.
The obsolete PCC package, teacher/distillation/joint workflows, their configs,
launchers and tests are removed; source remains in Git at pre-refactor 63bcc61.

EOS packing was mechanically moved from pcc/packing.py to deep_kv/packing.py;
its bytes are unchanged. prepare_data.py, saved text, local preparation project,
and running corpus preparation are untouched. No new export/tokenization stage.

The authoritative recipe is now flat HF JSON in deep_kv.b200.json: 28,600 full
schedule steps, eight GPUs, microbatch 16, accumulation 4, 1,048,576 tokens/update.
Generate A/B/C/D/report with python -m deep_kv make-jobs --stop-after 2000.
The old deep_kv train/smoke/plan and --config/--output/--stop-after training CLI
are removed. train.py uses standard HF args or a JSON file, including stop_after.
See DEEP_KV_TRAINING.md for current commands; older entries below are historical.

Native HF checkpoints replace the previous hash/certification/rotation layer;
old custom checkpoint receipts are not compatible. Config/data matching remains,
without custom per-step history or checkpoint manifests. Resume can select an
explicit intact native checkpoint if the latest save was interrupted.

Removed the redundant disabled upload argument that the runner scanner flagged;
pinned HF defaults still disable uploads, local-only model/tokenizer loading
and offline W&B remain. No push or B200 launch during this refactor. Latest
observed remote status was BLOCKED on commit 96487e0 (controller 06:53:34;
Dropbox upload 13:53:37 UTC, 2026-09-27), not evidence that the GPU node was dead.
Current local commands.sh remains #0. B200 CUDA/NCCL smoke remains pending.

Validation completed: 45 retained local tests passed (10.755 s). The actual
train.py path passed exact single-process resume, cache-hit/rebuild token and
sampler-order checks, independent gradient-accumulation checks, and matched
four-arm cutoff/reporting. Eight-process CPU/Gloo training passed every arm
with actual BF16 projection outputs, microbatch 16 and accumulation 4; maximum
resumed/uninterrupted parameter difference was 1.862645149230957e-9. Uneven
five-context final evaluation was counted exactly once per context. An initial
eight-process check exposed a fresh-output race; an HF distributed-state barrier
now completes all ranks' output checks before rank-zero writes. Shell syntax,
compilation, diff checks and the byte-identical packing comparison passed.
Evidence:
 temp/deep-kv-refactor-all-tests-20260927-final.log
 temp/deep-kv-refactor-resume8-20260927-final/resume_verified.json
No GPU workload, remote push, or corpus preparation change was performed.

## B200 smoke blocked by runner connection (2026-09-27)

User authorized an eight-GPU B200 smoke, committing all local changes, copying
resources/accelerate_config.yaml to the actual HF Accelerate default config,
verifying accelerate env, and correctly stopping identified GPU burns first.
Committed the user's README/commands.md/config rename as 5e76192. Pushed all
pending code and preflight command in 96487e06babf6d256723f239430f87efee98a959 to
origin main (deep-llms/th2); remote head was verified. No temp/INSTRUCTION.md was
supplied; protocol follows the provided guides and successful runner history.

Runner status for job th2-78gg-deep-kv-smoke-preflight-20260927-a01 at controller
timestamp 2026-09-27 04:06:10 reports FAILED(rc=255):
 ssh: Could not resolve hostname <host>: nodename nor servname provided, or not known
This is a runner SSH/hostname failure before remote project execution. No current
B200 GPU ownership, environment/config installation, burn stop, or smoke outcome
was obtained. Do not infer GPU availability from the old sampling logs. No
resubmission, process signal or cleanup was attempted after this failure.
Wait for operator repair before a fresh preflight and smoke submission.

Evidence: temp/deep-kv-b200-smoke-status-poll1-20260927.log, SHA256
21e974ade9e4be4f4601ca32b51593f420f1258e468c74ed221dc6d7874d8ddd.
The submitted command is preserved in Git at 96487e0 and locally in
temp/deep-kv-b200-smoke-preflight-20260927-a01.commands.sh. Local commands.sh is
restored to #0. No further execution-remote push after the infrastructure error;
remote main remains at the failed preflight revision. Hardware smoke is pending,
not passed. All user source changes are committed; no real training started.


## Active: use train.py's cached text pipeline (2026-09-27)

User explicitly rejected the custom binary preparation stage. Removed Deep-KV
prepare/--data-dir, token/permutation files, prepared manifests and custom
SequentialSampler. The queue is now A/B/C/D/compare (five jobs). Each training
command loads sampled text and calls shared `pcc.packing.preprocess_dataset`,
also used by train.py: tokenization + grouping Dataset.map, main_process_first,
HF cache reuse, dataset.shuffle(seed=42), then native Trainer seeded sampling.
Training uses the full packed dataset, not a custom prefix. B200 training uses
160 preprocessing workers as the baseline script; validation uses one worker
to retain its audited 4,882-context budget. Source/worker/tokenizer/seed/software
settings must match across arms for identical packing and order on cache misses.

The schedule stays 28,600 updates, 1,048,576 tokens/update, optional matched
2,000-step cutoff, eight GPUs per arm. Startup checks full packed-data capacity.
Fingerprints and full run identity guard comparison/resume; old binary-data
runs/checkpoints are incompatible. Scoped offline W&B and model/loss remain.
Sampled Arrow text is unchanged. Earlier preparation instructions below are
historical and superseded; use DEEP_KV_TRAINING.md. No remote/GPU launch.


Verification: 15 focused CPU tests passed (16.585s), including real two-worker
HF cache hits/forced rebuilds, independent packed-token expectations and equal
Trainer batch order across A/B/C/D. Native-sampler eight-process BF16 resume
passed all arms (maximum weight difference 9.314e-10), including production
input_ids/attention_mask/labels columns. Four-arm CLI cutoff/report smoke passed.
Existing regression: 134 passed, 15 version-specific skips (149 run, 166.967s).
Queue parsing lists exactly A/B/C/D/compare; shell, compile and diff checks pass.
Evidence:
 temp/deep-kv-hfdata-tests-20260927-final.log
 temp/deep-kv-hfdata-resume8-20260927-final/resume_verified.json
 temp/deep-kv-hfdata-smoke-20260927-final/comparison.json
 temp/deep-kv-hfdata-legacy-20260927.log
 temp/deep-kv-hfdata-jobs-20260927-final.json
All work was local and CPU-only; CUDA/NCCL/full-context capacity remains an
on-machine check. No push, sampling, production tokenization or training launch.

## Prepared-data compatibility and offline logging (2026-09-27)

Prepared v2 files are reusable when updates, tokens_per_update, context,
eval_rows and data_seed match. Other recipe settings and microbatch may change
for a new run. The original full preparation recipe/config remains in the
manifest; source paths, model config hash, shape, packing, order and content
checks remain enforced. Full run recipe/config identity still governs resume
and matched-arm reporting. No token-file rewriting or manifest migration.

Offline W&B is explicitly initialized on rank zero with dir set to the arm
output directory, unless WANDB_DIR overrides it. The standard HF callback
attaches to the owned run, closed on normal or exceptional exit. This avoids
source-tree artifacts and stale directories/runs across repeated train calls.
Tests use the real offline SDK on CPU, including sequential arms, an override,
and an injected failure; reusable-data tests also check incompatible data and
changed-recipe resume rejection. All 15 focused CPU tests passed (13.609s),
plus compilation and diff checks. Evidence:
 temp/deep-kv-data-wandb-tests-20260927-final.log
No GPU/remote work or launch. See DEEP_KV_RECIPE_COUPLING_AND_WANDB_20260927.md.

## Trainer review: retain certified checkpoints (2026-09-27)

HF 5.9 checkpoint rotation runs before the custom on_save certification. A
reproduced interrupted-save/resume case with a later partial directory deleted
the last recoverable checkpoint before certifying its replacement. Disabled
native rotation; on_save now retains the two latest certified saves after
publishing the new certificate and metrics. Uncertified partial directories
remain ignored and may occupy additional disk space. Training settings and
the native HF optimizer/accumulation loop are unchanged.

Verification: 14 focused CPU tests passed (7.537s), including injected save
failure/recovery and exact resume with two loader workers. Eight CPU/Gloo
processes, BF16, microbatch 16, accumulation 4 passed interrupted vs continuous
training for all arms (max parameter difference 3.726e-9). Forward hooks verify
actual BF16 projection outputs. Evidence:
 temp/deep-kv-review-tests-20260927-final.log
 temp/deep-kv-review-resume8-bf16-20260927-final/resume_verified.json
Pre-fix reproducer: temp/deep-kv-review-checkpoint-repro-20260927.log.
Plan/shell/compile/diff checks passed; no push, B200 operation or real training.
CUDA/NCCL, fused optimizer and full-context capacity still require hardware.

## Active: Hugging Face Trainer / Accelerate migration (2026-09-27)

User requested following train.py and scripts/train_qwen3_0.6b_baseline.sh,
replacing only project-specific model/loss and necessary integration points.
The active deep_kv trainer now delegates its training loop, optimizer/scheduler,
accumulation, DDP, evaluation gathering and optimizer/RNG checkpoints to Trainer
(Transformers 5.9.0, Accelerate 1.13.0). Sequential jobs use accelerate launch;
scripts/train_deep_kv.sh provides the equivalent single-arm launch. Legacy
EmbHub scripts are unchanged examples, not the active Deep-KV entry point.

Baseline settings now apply: microbatch 16 x 8 GPUs x 4 accumulation, BF16,
AdamW 3e-4 / .9,.95 / wd .1 / clip 1, cosine_with_min_lr (.1), warmup 500,
seed/data_seed 42, save every 250 steps, log every 10, 8 data-loader workers
per rank, offline W&B. CUDA uses Trainer's fused AdamW; CPU tests use AdamW.
Full budget remains 28600 x 1048576 = 29989273600 tokens; the chosen queue stops
all arms at 2000 steps without shortening the LR schedule. English packing,
full-model scratch initialization, four architectures, target detach, fixed
monitor/final splits, shared input permutation and matched receipts remain.
Old prepared recipes/checkpoint.pt outputs are incompatible; sampled text is
unchanged. Prepare once for the new seed/recipe; no resampling is required.

Custom compute_loss returns correctly normalized microbatch means and disables
HF loss-kwargs scaling; Trainer and DDP apply accumulation/rank averaging once.
Evaluation gathers per-example sums/counts and removes padded repeats before
computing LM, K/V, and total loss. A save adapter handles tied embeddings.
Standard checkpoint-N directories retain two saves; deep_kv.json certifies all
required files with SHA256 after every rank finishes. Resume/report validate
identity, exact step, histories and hashes. A CPU-only optimizer restore adapter
normalizes Accelerate's cpu:0 device; CUDA restore stays native HF.

Verification: 13 focused CPU tests passed in 5.793s, including direct raw-gradient
accumulation equivalence and corruption rejection. Existing train_env regression:
147 tests run, 13 version-specific skips, 134 passed (165.808s). Eight-process
microbatch-16/four-accumulation cutoff/report passed; single-vs-eight weights
agree within 6.054e-9 and gradient norms within 1.193e-7. Eight-process BF16
interrupted/resumed training passed for every arm (max weight difference
3.726e-9). Initial tests exposed/fixed an output-directory creation race and
CPU indexed-device checkpoint loading. Uneven eval scalar loss was corrected
to use de-duplicated per-example statistics. Evidence:
 temp/deep-kv-hf-tests-20260927-final.log
 temp/deep-kv-hf-legacy-regression-20260927.log
 temp/deep-kv-hf-ddp-parity-20260927.json
 temp/deep-kv-hf-resume8-bf16-20260927-a02/resume_verified.json
 temp/deep-kv-hf-ddp8-20260927-a03/comparison.json
Accelerate CPU module-launch, generated queue parsing, shell syntax, compilation
and diff checks passed. All testing local/CPU; no push, B200 operation or real
training launch. commands.sh remains #0. CUDA/NCCL/fused-optimizer execution
and real-context throughput are not verified by CPU tests.
See DEEP_KV_TRAINING.md for current commands. Earlier backend/seed/warmup and
checkpoint notes below are historical and superseded by this section.

## Selected 2,000-step run and packing margin (2026-09-27)

User selected 2000 updates per arm, replacing the earlier 1000-step example:
2097152000 input tokens per arm, 8388608000 across A/B/C/D, eight GPUs per arm.
Use make-jobs --stop-after 2000; the full LR schedule remains independent.
Checked DEEP_KV_DATA_SHORTFALL_20260927.md against sampler code, both sampler
logs and the eval audit. Its raw token arithmetic is consistent; the expected
302-context deficit depends on a uniform/independent remainder model, not an
exact measured train packing count. Applied the recommended 28600-update /
1430-warmup recipe (29989273600 tokens, 14643200 contexts), giving an estimated
4818-context margin. Preparation still validates exact capacity before training.
No sampler/packing changes or repeated data. Previous 28610-recipe streams
cannot be reused; generated queues must use the revised recipe.
Updated launch examples to 2000 steps. All 13 focused CPU tests passed (5.963s);
plan, queue parsing, compilation and diff checks passed. Evidence:
temp/deep-kv-2k-shortfall-tests-20260927.log, temp/deep-kv-2k-plan-20260927.json
and temp/deep-kv-2k-jobs-20260927.json. No preparation, GPU training, process
management, or remote push. commands.sh remains #0.

## Deep-KV budget and cutoff decision (2026-09-27)

User overrides the original four-arm document's 1B/32K budget: plan approximately
30B English tokens with a 1M-token global batch, then optionally stop all arms at
the same optimizer iteration. Defaults are 28600 updates x 1048576 input tokens
= 29989273600 tokens per arm; 512 contexts of length 2048, eight ranks with 64
contexts each. Microbatch defaults to 1 (64 accumulation passes); any divisor
of 64 is accepted. Existing AdamW settings remain; 5% warmup becomes 1430 updates.

The current generic run_experiments.py handles process/artifact sequencing, not
trainer step counting. Deep-KV make-jobs now passes --stop-after N to all four
trainers and the report, checks exact stopped.json receipts, and advances to
the next arm. The default queue still demands full-budget complete.json.
Cutoff checkpoints are resumable with unchanged schedule/data order. Every
endpoint uses the full fixed evaluation split; reports explicitly record
compared_update, schedule_updates and training_complete. Synthetic tests remain
marked pilot_result false. The selected experiment cutoff is now 2000 updates;
pass --stop-after 2000 explicitly (the generic CLI default stays full training).

Preparation covers the full 30B budget even for cutoff queues (~120GB uint32
tokens/order), so longer runs keep identical shuffled data prefixes. Existing
sampled text needs no change; old 1B packed streams fail recipe validation.
Preparation must verify the full packed budget fits; it fails on insufficient
data rather than silently repeating examples. No full 30B preparation ran here.

Verification: 13 Deep-KV CPU tests (5.950s), 31 runner utility tests (1.281s),
eight-process Gloo cutoff/report and interrupted-vs-uninterrupted checks passed.
Evidence under temp/deep-kv-30b-*20260927*. Tests cover accumulation equivalence,
full-schedule LR at cutoffs, full evaluation, mismatched cutoff rejection,
continuing stopped arms, and recovery at an existing periodic checkpoint.
All changes/tests local; commands.sh remains #0; no push or real training launch.

## Deep-KV code review and recovery fixes (2026-09-27)

Reviewed the four-arm implementation against the supplied pilot specification.
Fixed two recovery edge cases: a resumed run retained its old stopped.json, and
resuming a final checkpoint after an interrupted results write did not restore
metrics.json. Resume now checks the run receipt against checkpoint identity,
clears the stop marker only after validation, and restores metrics before final
completion. Training also checks that the supplied identity matches arm/recipe.
No architecture, objective, token budget, data recipe or schedule was changed.

11 B200-environment CPU tests passed in 4.296s, including a failing-before/fixed-
after regression and exact single-process recovery after final-checkpoint save.
The new nonzero-branch test verifies identical B/C/D LM outputs and gradients
in float32 and bfloat16. Eight-process CPU/Gloo full-vs-resumed training passed
for all four arms (max parameter difference 7.451e-9). Evidence:
temp/deep-kv-review-after-20260927.log and
temp/deep-kv-ddp8-resume-review-20260927-a01/resume_verified.json.
The actual 28-layer, 600244352-parameter arm D also passed CPU Base equivalence
and finite forward/backward gradients with an eight-token sequence (13.33s):
temp/deep-kv-full-geometry-review-20260927.json. This is not a 2048-token GPU
capacity test. The real locally reproduced English eval split has 11822 docs,
10011667 tokens including EOD, and 4883 packed contexts: enough for the locked
4882-context budget (temp/deep-kv-real-eval-audit-20260927.json).
Existing regression suite passed in its original train_env: 145 tests run in
165.753s, with the 11 version-specific Deep-KV tests skipped there (134 passed).
Deep-KV tests passed separately in sampling_b200 (Transformers 5.9.0). Evidence:
temp/deep-kv-review-legacy-regression-20260927.log. All work was local/CPU; no remote changes,
training launch, process termination or GPU-management action. commands.sh is #0.

## Active: implement four-arm from-scratch Deep-KV pilot (2026-09-27)

User requested training code for docs/anticipatory_deep_kv_four_arm_pilot_v2.md.
Implemented dedicated deep_kv package; the legacy EmbHub trainer and pretrained
PCC mechanism are unchanged. No B200 training/deployment requested in this step;
commands.sh remains #0. All verification ran on local CPU with sampling_b200.
Arms A/B/C/D: Base / ExtraAttn-NoAlign / ShallowKV-Align / DeepKV-Align.
Consumer block 5; target block 5 or 21; native query reused, normalized pre-RoPE
native key and native value targets from the same forward; target-only detach;
strict-past auxiliary mask; zero output initialization; L1 coefficient fixed 1.
Backbone initialized from Qwen3 config, never pretrained weights. Shared initial
backbone across all arms and identical B/C/D branch tensors are fingerprinted.
Fixed 30518 updates x 32768 tokens = 1000013824 tokens per arm. Recipe choices
not specified by the document are explicit: AdamW 3e-4, beta .9/.95, wd .1,
1526-update warmup, cosine to .1 peak, grad clip 1; full details in
DEEP_KV_TRAINING.md. English-only sources in deep_kv.b200.json point to completed
B200 sampled data. One CPU preparation creates a shared fixed token stream with
current EOS packing, then the queue runs A/B/C/D with eight GPUs each and reports
matched LM contrasts. Periodic atomic optimizer/model/RNG checkpoints support
exact resume; a graceful stop preserves LR schedule and cannot count as complete.

Verification: 10 CPU tests passed in 4.013s (temp/deep-kv-tests-20260927.log),
including all six mechanism acceptance requirements, native block-5/21 targets,
Qwen query-width geometry, checkpoint gradients, bfloat16 forward/backward,
actual Arrow/tokenizer preparation, stream checksums, exact interrupted resume,
and eight-GPU queue budgets. Four-arm single-process and eight-process CPU/Gloo
smokes both completed including comparison.json; artifacts under
 temp/deep-kv-smoke-single-20260927-a03 and
 temp/deep-kv-smoke-ddp8-20260927-a03.
Initial distributed parity measured max parameter difference 3.204e-7 and max
LM-loss difference 6.812e-8 (temp/deep-kv-ddp-parity-20260927.json).
End-to-end reporting exposed integer model-config keys becoming strings in JSON;
canonicalized run identities before checkpoint/receipt creation, added regression
coverage, and reran both complete smoke workflows successfully. No GPU/NCCL
capacity/throughput test or real 1B-token training has run. Next launch requires
normal GPU ownership checks and a short authorized hardware capacity check.

## B200 sampling completed and verified (2026-09-27)

Read-only export cbedcd7 (th2-78gg-check-sampling-20260927-a02) returned both
sampling.log and sampling_complete.json at 2026-09-27 04:43:31 UTC.
Completion receipt success=true, completed_utc=2026-09-27T01:53:34.841949+00:00;
terminal log ends D2S OFFLINE CULTURAX SAMPLING OK, no traceback. Runtime from
19:59 UTC was approximately 5h55m. All six train/eval outputs were reopened and
validated as nonempty text-only datasets by the launch script. Output remains
/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B-Base; du -sh reports 146G.
Train documents / eval documents / train shards:
en 36595514 / 11822 / 35; vi 927135 / 9251 / 2; zh 964420 / 9802 / 2;
ru 805963 / 7963 / 2; de 974614 / 9596 / 2; ar 846723 / 8337 / 2.
Evidence: temp/b200-sampling-complete-20260927-a02.json and .log.
Receipt SHA256 f204e4bbe6897ce2c1d85c40e7095ab59417e066a0c74249e785cbb49b78339f;
log SHA256 3ee2de006ded252d2c76a35e624bfd169c55fb7d591da8a401dfc04bb34c6446.
This supersedes earlier B200 progress estimates. Local sampling status was not
checked in this request. No training launched or processes modified; restored
commands.sh to #0 after the successful export.

## B200 sampling status checked (2026-09-27 00:20 UTC)

Read-only export e5b69e6, job th2-78gg-check-sampling-20260927-a01,
completed successfully. Snapshot timestamp 2026-09-27T00:20:14Z.
Sampling has not completed: English file 29/50 processed, 24,978,593,652
counted tokens toward 30B train + 10M eval. Last completed shard flush in
snapshot is shard_0027. No other language has started in this log; no traceback
or terminal success marker. sampling_complete.json was not returned and is
absent from the exported folder. Approximately 71% of the total 35.06B-token
target; elapsed about 4h21m since 2026-09-26 19:59 UTC. Rough remaining estimate
2 hours, subject to language/IO speed. This is a progress snapshot, not a fresh
process-liveness check. No workload restarted/stopped. commands.sh restored #0.
Evidence: temp/b200-sampling-progress-20260927-a01.log (87354 bytes,
SHA256 6339a77867717678f8df2d5e64df08ebea9a5d9fd161be10979f220e0a5e1fee).

## Active: local reproduction of B200 sampling (2026-09-26)

User authorized a separate environment matching B200, then download and sampling
on this dev machine. Run root: temp/local-sampling-20260926-a01.
New conda environment: /home/users/thien/miniconda3/envs/sampling_b200.
B200 reference captured read-only via f4f4e92, job
th2-78gg-export-sampling-env-20260926-a01. Python 3.11.15; 160 exact conda
name/version/build/subdir records and 160 Python distribution versions captured,
plus first-16-document tokenizer digests for each language. Reference and locks
are under envs/sampling_b200.*. Installation and pip check succeeded. At
2026-09-26 23:34:18 UTC, verification confirmed all 160 Python package versions
(no extra distributions), all 160 conda builds, Python 3.11.15, source/asset
hashes, and exact first-16-document token IDs/counts for all six languages.
Evidence: <run>/environment_verified.json and <run>/pip-check.log. Original
train_env is unchanged. No GPU use; no changes to the B200 sampling process.
Pinned local download completed; full SHA256 verification is progressing in
tmux d2s_sampling_download_20260926_a01,
using nguyenhuuthuat09/CulturaX_sampled revision
b19d850278693d37113c197857cc6328fa5c6881. HF metadata matches all 75 committed
manifest hashes/166107112571 bytes. Existing verified en_part_00015 was hardlinked;
other 74 files downloaded under <run>/data/raw. Require download_complete.json
(status ok, all 75 SHA256 checked) before sampling. Download log: <run>/download.log.
Sampler source/manifests and pinned tokenizer assets are frozen under <run>/source
and <run>/tokenizer. prepare_data.py is unchanged. CPU-only workflow
scripts/run_verified_local_sampling.py was copied to <run>/source and launched
in persistent tmux d2s_sampling_local_20260926_a01. It currently waits for the
download receipt, then automatically rechecks the environment, dry-runs and
launches the identical offline sampler. Launch receipt: <run>/workflow_launch.json;
workflow log: <run>/workflow.log; actual sampler log: <run>/sampling.log.
Check sampling_started.json and the sampler log before claiming actual sampling
has started. Output: <run>/data/Qwen_Qwen3-0.6B-Base.
Its final sampling_complete.json will record document counts and ordered text
hashes per split; cross-machine full-output equality still needs B200 hashes.
No Hugging Face upload authorized/launched in this step. commands.sh set back
to #0 after the read-only environment capture; do not relaunch it.

## Document-end packing update (2026-09-26)

User authorized adding EOS document boundaries while retaining current packing.
Both train.py and PCC now explicitly append <|endoftext|> (151643 for the pinned
Qwen tokenizer) before concatenation/chunking. Resolve by token name; do not use
tokenizer.eos_token_id, which can be the chat marker 151645. Shared logic lives
in pcc/packing.py. Automatic tokenizer special tokens remain disabled.
1000-document batches, remainder dropping, context shuffle, full causal attention,
positions, and fixed update/input-token budgets are preserved. EOD tokens count
in those budgets and participate in the existing causal loss.
Future metadata records document_map_eod_v1; experiment entry points reject
old prepared NPZ policies. Low-level PreparedContexts remains able to read
historical artifacts. No existing cache cleanup or migration is needed on B200.
prepare_data.py and ongoing remote sampling are unchanged; no deployment or
training launch is part of this code change. Previous experiments remain results
of the earlier no-separator policy.
Validation: all 134 offline CPU tests passed (163.834s), including document
boundaries, unchanged remainder handling, prepared-policy rejection, shared
prefixes, and the real tiny pipeline/queue test. Log:
temp/eod-packing-regression-final-20260926.log. The initial broader run caught
an out-of-vocabulary ID in the new toy fixture; corrected the fixture to use
its existing vocabulary, then reran the full suite successfully. Actual pinned
Qwen tokenizer check also passed: [9707,1879,13,151643] for Hello world. despite
tokenizer.eos_token_id=151645. Python compilation and git diff checks passed.

## Qwen end-token verification (2026-09-26)

Historical decision: user initially accepted packing without separators;
superseded by the 2026-09-26 document-end change below. A CPU-only verified base-weight probe naturally generated endoftext
151643 in 1/8 greedy and 1/8 sampled runs; no im_end generated. Official Qwen
control-token docs state pretraining inserts endoftext between documents,
separately from the tokenizer call. See QWEN_BASE_EOS_PROBE_20260926.md.
All local GPUs were occupied; no GPU or B200 workload changed.

## Sampling runtime and reuse assessment (2026-09-26)

Read-only progress export 889eec6 completed; snapshot at 20:30:38 UTC shows
three English files processed, 2,588,403,725 tokens. Run began 19:59:00 UTC;
completion remains pending. Evidence: temp/sampling-progress-20260926-2030.log.
Previous identical sampler ran 2026-09-23 13:02:49 to 18:50:19 UTC (5h47m30s).
Current planning estimate: 6–8 hours total, subject to language/IO throughput.
Local CPU-only benchmark used the matching tokenizer.json, Transformers 4.57.1,
three 4096-document English batches: 9,674,000 tokens in 5.6718 seconds (~1.706M
 tokens/s). Extrapolation is 5.71h tokenization-only for 35.06B tokens; budget
6–10h sampling locally, excluding download/upload. This is not an end-to-end
benchmark: local package version differs from remote 5.9.0 and timings exclude
whole-file IO, shuffle, saves and non-English languages. Evidence:
temp/sampling-local-timing-20260926.json. Dev has 32 CPU cores, 251GiB RAM,
8.0TiB available disk at observation, enough capacity for this workflow.
User asks about preparing once on dev and publishing reusable sampled splits
on HF. Feasible; preserve train/eval Arrow directory layout and order, publish
source/tokenizer/software/seed provenance plus per-file hashes, pin published
revision and validate downloads. Raw text outputs still require training-time
tokenization/packing. No full local sampling or HF publication launched in this
assessment. GPU-node direct outbound upload is prohibited by AGENT_GUIDE;
large #2 exports are limited to 25MB/file, so dev-origin publication is the
straightforward route without a separate operator-provided bulk export path.
commands.sh restored to #0 after this read-only export; sampler continues.

## Active: authorized six-language sampling on 78gg (2026-09-26)

Preflight 1d94408 completed on 78gg at 2026-09-26 19:53:55 UTC: all 75 raw files,
166107112571 bytes, SHA256 and Parquet metadata verified (33.6s). Correct
language counts 50 en / 5 each other; sampled output absent, 24 TB free. The three missing
pinned tokenizer assets were downloaded via 208baa6 (identical to d0a71b5):
all three ITEM OK and overall OK at controller 12:57:31 on 2026-09-26.
No full model weights are needed for sampling.
Receipt on worker: /mnt/local/_outputs/deep-llms_th2/data_preparation/culturax_78gg_preflight_20260926_a01.json.
Evidence: temp/th2-monitor-78gg-20260926/1790452501265019095-_run-2026-09-26_19-53-11-th2-78gg-sampling-preflight-20260926-a01.log.

User requests running prepare_data.py using the established th2 history.
The CulturaX download completed (616cc39, controller OK at 12:39:27). Follow
9bbcaf9 for 75-file hash/Parquet verification, d0a71b5 for pinned tokenizer
prerequisites if absent, and b4f150d for the CPU-only offline sampling launch.
prepare_data.py and both manifests are unchanged from b4f150d.
Use train_env; seed 42, Qwen3-0.6B-Base tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd;
train targets 30B English / 1B each vi,zh,ru,de,ar; eval 10M tokens per language.
Raw input /mnt/local/_data/deep-llms_th2/data/raw; sampled output
/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B-Base/{train,eval}/<lang>.
Refuse existing sampled output; do not delete or silently resume partial data.
Sampling job th2-78gg-d2s-sample-culturax-qwen-20260926-a01 launched via
3dca9ddf34764bbfb4921796e9cedaa32ac1d242 at 2026-09-26 19:59:00 UTC.
commands.sh preserves b4f150d's sampling and validation commands, changing only
worker/job/log identity and adding a check of the verified raw-data receipt.
Expected completion artifact:
/mnt/local/_outputs/deep-llms_th2/data_preparation/culturax_qwen_78gg_20260926_a01/sampling_complete.json.
Require success=true and all six languages reopened as nonempty text datasets.
Startup verified from the snapshot exported at 2026-09-26 20:00:24 UTC:
raw-data receipt, train_env, offline tokenizer/config and all three SHA256 checks
passed; dry run passed; real sampler entered English sampling from 50 files.
No completed shard or terminal result is present in that initial snapshot.
Evidence: temp/th2-monitor-78gg-20260926/1790452829348526335-_run-2026-09-26_19-58-50-th2-78gg-d2s-sample-culturax-qwen-20260926-a01.log.
commands.sh is now #0 to prevent accidental relaunch on later source pushes;
this does not stop the already running job. Use a unique #2 to export sampling.log
and sampling_complete.json on a later status request; do not resubmit #1.
Existing GPU workloads remain untouched; sampling hides GPUs. This authorizes
sampling, not training. Completion remains unverified.


## Historical: CulturaX download on verified 78gg environments

Download submission 616cc39 is confirmed STARTED by the controller. Background
ID 2026-09-26_19-20-41, status timestamp 2026-09-26 12:20:44 (controller clock).
Completion is not yet verified. Evidence:
`temp/th2-monitor-78gg-20260926/1790450488108036992-_RUN_STATUS_.log`.
Read status from the new share; do not repush the active #d command.

Environment validation 631f77c passed on thiennh-p6-78gg-worker-0 at
2026-09-26 19:17:53 UTC: both environments imported all required modules and
reported no broken pip requirements. Both use torch 2.14.0+cu130/CUDA 13.0,
Transformers 5.9.0, datasets 4.8.5, accelerate 1.13.0; eval has lm_eval 0.4.10.
All eight B200s are visible and NCCL is available. About 25 TB free; the dataset
destination did not exist. Existing GPU workloads were only inspected.
Evidence: temp/th2-monitor-78gg-20260926/1790450345258291967-_run-2026-09-26_19-17-31-th2-78gg-verify-envs-before-download-20260926-a01.log.

Now submit the exact dataset-only #d command from 382992d to fetch the whole
nguyenhuuthuat09/CulturaX_sampled repository to /mnt/local/_data/deep-llms_th2/data.
The historical raw-data manifest covers 75 files/166107112571 bytes; download
completion and file verification are still pending. Do not resubmit the same
download to request status. The existing #d workflow handles network access.

User authorized checking the completed installation and, if correct, downloading
nguyenhuuthuat09/CulturaX_sampled using the established th2 history.
The new share reports both `OK env eval` and `OK env train_env`, followed by
`OK | install: 2 env(s)` for bf4cadd (controller timestamp 2026-09-26 12:13:35).
Now submit read-only imports/pip-check/device/destination checks adapted from
f678729, job `th2-78gg-verify-envs-before-download-20260926-a01`.
If they pass, use the identical #d directive from 382992d/c802234/cc338d4:
`--hf-dataset nguyenhuuthuat09/CulturaX_sampled /mnt/local/_data/@PROJECT@/data`.
No sampling, training, GPU allocation, process stop, or additional model download.


## Historical: install runtime on the replacement B200 (2026-09-26)

Active Dropbox share updated by the user on 2026-09-26: label `th2-78gg`,
with `th2` as an alias. The supplied URL is stored only in ignored
`temp/dropbox_folders.txt` (mode 0600). Read-only folder listing succeeded.
The share name identifies assignment thiennh-p6-78gg; live worker identity
still needs verification. `temp/poll_th2.py` now reads the new share and keeps
its observations in `temp/th2-monitor-78gg-20260926/`. The old `th2-tpbw`
entry and its local observations remain historical. No installation resubmitted.
The new share's _RUN_STATUS_.log matches commit bf4cadd and the requested
train_env/eval job: STARTED, then RUNNING (latest file modification
2026-09-26T19:08:53Z). Installation completion is still unverified.

User reports a new fresh B200 machine and explicitly requests changing
commands.sh to install the environment and pushing to deep-llms/th2.
User corrected the environment choice: install both train_env and eval using
`#i envs/train_env.txt envs/eval.txt +a`, job
`th2-install-train-env-and-eval-20260926-a01`, through origin/main.
Verified against th2 history: bd23724, 3c5fa74 and 189e0c3 all use this exact
installation directive. Both environment specifications are unchanged from
bd23724 (Python 3.11, fresh:true, Transformers 5.9.0). The earlier pcc_joint-only
submission c1595a6 was the wrong choice for this request; its installation
outcome remains unverified. This correction does not remove that environment.
Installation completion, CUDA build and the replacement worker's identity are
not yet verified. Do not reuse the dead worker's hostname/PIDs/storage state.
This submission installs dependencies only; no training, download or GPU stop.
Do not repush the executable installation command to refresh status.

The local restart finished successfully at 2026-09-25T15:30:59Z. Deep teacher,
LM and PCC each completed 6144 updates/201326592 input tokens. Final dev NLL:
feedback off 2.98561804, Deep 2.96830996, LM 2.97769621, PCC 2.97917756.
PCC recovered 37.2% of the feedback benefit but lost to matched LM; the queue
correctly stopped before seed two. Results are in
`temp/local-restart-20260925-a01/report-1-seed/`; final teacher/student weights
and inputs/source have verified network-filesystem backups under
`/home/users/thien/deep2shallow-backups/local-restart-20260925-a01/`.
All older running/blocked observations below are historical.


## Current authority: local restart after B200 loss (2026-09-25)

User confirmed the B200 machine is dead and authorized the proposed local
restart. This supersedes the wait-for-B200-restoration instructions below.
See PCC_LOCAL_RESTART_20260925.md. Four idle A100-PCIE-40GB GPUs were verified;
use all four sequentially, train_env, full 28 layers. One newly trained Deep
teacher -> same-checkpoint feedback audit -> matched LM/PCC students if positive;
second seed only if the first matched student pair passes. Fixed 201M tokens/run,
new fixed local split, no substitution of old B200 metrics for missing weights.

Local queue launched from tested source commit 7ec9b4f under tmux session
`pcc-local-restart-20260925-a01`. Root: `temp/local-restart-20260925-a01/`.
All 131 CPU tests passed (165.407s). Real four-GPU Deep capacity/resume passed:
matching replicas, early/late/branch updates, exact initial native equivalence,
checkpoint roundtrip and next-update resume. Updates took 3.216/2.880 seconds;
capacity/resume peak allocated 20.24 GiB. The first 6144-update teacher is running: verified at 04:40:49 UTC,
update 14, finite loss/gradients, ~2.9–3.1s/update, one worker per GPU
(PIDs 771852–771855 at this observation only). Invocation confirms resume=null
and stop_after=null; capacity weights did not initialize the scientific run.
Monitor root/pipeline.json and teacher-seed-0/runs/seed-0-Deep/train.jsonl.
Source is isolated in root/source; do not edit that snapshot or relaunch the queue.
commands.sh stays #0; no remote submission. Inputs/model/source have already
been SHA256-verified on the separate network home filesystem under
`/home/users/thien/deep2shallow-backups/local-restart-20260925-a01/`.
The controller copies new scientific checkpoints every 15s after they are saved
(every 256 updates); complete teacher weights are required in the backup before
student stages. A negative gate stops the queue; no local burn is launched.
No B200 access, GPU reclaim, or new burn is authorized/needed for this restart.
The older sections below describe historical scopes and states.


New explicit authority (2026-09-24): user permits reclaiming current vLLM/burn/
other GPU workloads after precise fresh ownership verification, then launching
the four-run correction-distillation stage. This overrides the earlier leave-
deepeyes-running note below. Preserve completed teacher checkpoints and outputs.
Submission 1d2d4b5 failed before remote execution: no Running worker pod for
thiennh-p6-tpbw (controller rc=5, observed 21:51 UTC). No GPU signals/training
occurred. Wait for infrastructure repair/new runner assignment; do not relaunch
against a missing worker. Local 124-test regression and fresh reclaim test passed.

Latest outcome: the six-run joint-v2 queue completed successfully at 2026-09-24
03:25:08 UTC. Deep passed both-seed comparison gates; no student training yet.
See PCC_JOINT_RESULTS_20260924.md. At 20:46 UTC live inspection confirmed a
polite burn under deepeyes (PID 80920, workers 80990–80997) plus vLLM servers
on all GPUs. These are separate workloads; do not reclaim or reuse the node
based on historical permissions/PIDs. Compact results are local in
artifacts/joint-v2-20260924/.

## Latest execution decisions (2026-09-23)

- User explicitly approved stopping the verified runner burn, then clarified
  that every experiment must use all eight GPUs. Do not revert to six independent
  single-GPU jobs or request burn-stop authorization again.
- User authorized longer training: fixed joint-v2-ddp uses 6144 updates / 201M
  tokens per arm, six sequential eight-rank DDP runs. Global batch is unchanged.
- See PCC_B200_LAUNCH_20260923.md for the versioned budget, source pool and launch.
- Controller gpu_guard.sh has a supported /mnt/local/_gpu_guard/DISABLED marker.
  Its power trigger can launch a co-burn even during training. The current queue
  has an authorized temporary marker lease, restored when its pinned PID exits.
  One-time reclamation scripts must never be reused with different process IDs.
- Existing run_experiments.py handles eight-GPU job allocation; custom PCC
  training adds DDP normalization, rank-safe I/O, exact eval ordering and RNG.

## Current deployment authority (2026-09-23)

- User authorized B200 training and pushes to https://github.com/deep-llms/th2.
  This overrides earlier local-only restrictions recorded below.
- Source checkout now follows the existing th2 main history, initially fa905b1.
  SSH alias github-share uses the existing configured key; no credentials in Git.
- Runner project deep-llms_th2, source /mnt/local/deep-llms_th2, data
  /mnt/local/_data/deep-llms_th2/data, models /mnt/local/_models/deep-llms_th2,
  results /mnt/local/_outputs/deep-llms_th2; Dropbox label th2-tpbw.
- Completed sampler report (18:50:19 UTC) confirms all six languages succeeded.
  English train has 35 shards / 36,595,514 documents; eval has 11,822 documents.
- Launch b4bc9f6 passed readiness and all three eight-GPU capacity checks,
  including Deep resume; scientific seed-0 Base started at 21:59:20 UTC.
  Active root is joint-v2-b200-20260923-a03; do not resubmit the launch.
- The remote sampler env specification has Transformers 5.9.0; joint training
  requires 4.57.1. Preserve B200-compatible CUDA PyTorch when configuring it.
- No temp/INSTRUCTION.md was supplied. Runner syntax was checked against the
  provided guides and current successful th2 install/download/run/export history.

- The runner syncs source without Git metadata. Do not run git rev-parse in
  node jobs; use controller commit records and file hashes for provenance.
- Core tokenizer.json and model config hashes match between the sampler and
  locked training revisions, but tokenizer_config differs (chat template and
  added-token metadata). Training always uses the complete locked revision;
  no whole-tokenizer equivalence is assumed from the core file hash alone.

## Historical local work

Status: joint full-model follow-up is implemented and ready for a bounded local
pilot; scientific launch is pending. Read `PCC_JOINT_READINESS_20260923.md`.
The earlier local four-pair pretrained effectiveness screen completed. Negative
deep-source result: all deep adapters beat Base, but the matched shallow adapter
beat deep in every pair (paired 95% CIs). No full probe launched. See
`PCC_PROMISE_SCREEN_20260923.md`. Earlier singleton permutation blocker remains.

## Purpose and success criteria

Follow research-contract section 11: determine whether privileged strict-past
deep-source attention helps a frozen Qwen3-0.6B-Base, then whether a shallow-only
student can recover the correction's benefit. First establish correctness and
run the four preregistered layer pairs against matched shallow-attention controls.
Full student/control training and development-locked test orchestration are now
implemented; the user confirmed this is the next coding stage, before scientific
diagnostic results are available. Local pretrained GPU execution is now authorized;
B200 remains outside the execution scope.

## Infrastructure

- User authorized local diagnostics on dev host `transformer1` on 2026-09-23
  because B200 is down. Four idle A100-PCIE-40GB devices were observed; GPU 0
  was used for bounded pretrained/real-context diagnostics and released. The
  subsequent effectiveness request authorized the completed four-pair screen,
  one pair per local GPU. Its run directory is
  `temp/pcc-promise-screen-20260923-a01/`; all workers exited successfully and
  GPUs were released. B200 and all remote operations remain out of scope.
- Local conda `train_env` cloned from `sparse_emb` on 2026-09-22. The clone uses
  Python 3.11.15, torch 2.7.1+cu118, Transformers 4.57.1, numpy 2.4.4.
  `sparse_emb` retains Transformers 5.9.0. These are CPU-tested development
  dependencies, not validated dependencies for B200 deployment.
- Joint readiness work added matplotlib 3.10.9 to `train_env`; `pip check`
  passed. All other model/data dependencies and `sparse_emb` remain unchanged.
- No Git repository is initialized in this checkout. Existing `commands.sh`
  contains an earlier sampling command and has been left unchanged.
- Earlier user context was an ongoing `prepare_data.py` sampling job on B200;
  the latest report is that B200 is down. Neither status was remotely verified.
  Do not access B200 or change the sampler. Local source review confirms raw-text train shards
  plus one approximate eval split. PCC now consumes these train/validation
  directories directly; optional independent test data is deferred. See
  `PCC_DATA_HANDOFF.md`.

## Reproducibility

- Research-contract revision `ddc928429ed09d9ad603fd762053d0434c15e865` remains
  the default; sampling assets use a different revision (`da87bfb…`). No
  substitution or equivalence has been established.
- `pcc/` loads completed sampled English train/validation directories directly.
  Shared deterministic tokenization/packing/order serve identical prefixes to
  screen and full probe. There is no export stage or new document-range split.
  Internal Arrow caches live in the run directory; source data is unchanged.
  Existing packed NPZ inputs remain supported by the low-level reader.
- Screen: four fixed pairs, paired initialization seed 1701, 128 updates,
  32,768 input tokens/update, 2M dev tokens, 1,000 paired sequence resamples.
- The local 2026-09-23 screen uses a disclosed exploratory split because the
  finalized B200 outputs are unavailable. Source is the user-specified
  `nguyenhuuthuat09/CulturaX_sampled`, pinned revision
  `b19d850278693d37113c197857cc6328fa5c6881`, English `en_part_00015.parquet`:
  rows [0,20000) train and [20000,30000) dev. Shared legacy packing, context
  shuffle seed 20260922, exact budgets, and microbatch 4 are fixed for every arm.
  This is not a reproduction of the B200 split or a confirmatory test. Per-pair
  workers defer selection to one joint call of the unchanged bootstrap rule.
- The student alignment primitive detaches teacher targets. Frozen-tail
  activation checkpointing preserves adapter gradients. No scientific result
  may be inferred from synthetic test metrics.
- Full probe: shared fresh seed 2901, 610 updates per arm, exact 1M-input-token
  fp64 calibration, 10M-token dev/test, 2,000/5,000 paired bootstrap resamples.
  Screen weights are never loaded; screen selection is rederived from losses.
- Conditional permutation pools targets over each global update, uses crossed
  context-position/norm deciles, and refuses singleton buckets. Its seed, ties,
  and mapping are fixed/documented in `PCC_EXPERIMENT.md` before real data use.
- Development decisions and parameter fingerprints are saved before any test
  read. Positive mocked-metric tests validate this state machine, not a gain.
- One-pass full-context student scoring is implemented and compared with the
  training tail path in bf16/fp32. Incremental generation caching is separate work.
- Sequential automation uses `pcc pipeline` for screen → gated full probe in
  one process/model load. The existing `run_experiments.py` wraps that command
  as one job and can queue independent experiments/validators after it.
  Negative scientific outcomes complete successfully; software failures stop
  the queue. There is no automatic retry/resume or pretraining launch.
- Pipeline inputs are supplied once in JSON; relative paths are relative to
  the config location. Dry-run planning does not load ML libraries or inspect
  model/data paths. `pcc.pipeline.example.json`, `jobs.pcc.example.json`, and
  `PCC_AUTOMATION.md` document the interface. Config now requires only
  `model_path`, `train_data`, and `val_data`, with optional `microbatch` and
  independent `test_data`. The previous exporter and document-range config
  were removed at the user's request.
- Packing now uses document_map_eod_v1: 1000 documents per batch, explicit
  endoftext (151643) after each document, per-batch remainder dropping, then
  context shuffling with seed 20260922. Entire supplied splits are preprocessed
  once per run before shuffle; internal caches are reused between stages.
- Missing test input permits development-only completion. Passing dev gates
  yields `validation_complete_test_not_supplied`; no test confirmation or pilot
  recommendation is made. A supplied test remains locked behind frozen dev
  decisions and all gates. Validation is never automatically reused as test.
- Exact input-token budgets and 128/610 optimizer updates remain fixed. A short
  eval split fails rather than resampling or borrowing data. Sampler overshoot
  and packing loss can make its nominal 10M eval budget insufficient; any
  reduction requires an explicit protocol change before scientific execution.
- The normal pipeline checks full train/validation budgets before the screen,
  so a short full-probe eval split fails before adapter training. It records
  exact input/target counts and canonical ordered-context SHA256 fingerprints
  in `input-check.json`, and verifies screen/full prefix equality. Test data
  is outside this check's interface and remains locked.
- `pipeline --check-only` runs model correctness preflight plus the same input
  checks and completes as `inputs_validated` without scientific training.
  Preflight uses disposable adapters for gradient checks. This is optional;
  no separate export/preparation job is required. It does not establish corpus
  authenticity, external sampling completion, full-context GPU capacity, or gain.
- The generic runner timeout limits wall-clock time; it does not implement
  matched optimizer-update stopping. Trainer loops implement that and save
  checkpoints at the fixed final update.

## Results and decisions

| Date (UTC) | Run / commit | Evidence location | Result | Decision |
|---|---|---|---|---|
| 2026-09-22 | Local environment | `train_env`; `pip check` | No broken requirements | Use clone for CPU tests; preserve `sparse_emb` |
| 2026-09-22 | CPU correctness | `tests/test_pcc.py`, `tests/test_utilities.py` | All 46 tests passed, including tiny bf16/fp32 Qwen and synthetic four-pair screen | Implementation checks pass; no claim of pretrained benefit |
| 2026-09-22 | CLI preflight | `temp/pcc-tiny-preflight-20260922-a01.json` | Passed on CPU with tiny random model | Require actual pinned-model/data evidence before scientific conclusions |
| 2026-09-22 | Review and fixes | 56 CPU tests; `temp/pcc-tiny-preflight-review-20260922-a01.json` | All tests and 42 CLI checks passed; broken LM path also rejected under `python -O` | Stronger implementation evidence; full acceptance coverage remains incomplete |
| 2026-09-22 | Full probe implementation | 70 CPU tests; `temp/pcc-full-probe-tests-20260922-a01.log` | All passed in 49.558 s, including 14 full-probe tests and reduced real adapter training | Implementation ready for pinned-model/data validation; no scientific evidence or B200 use |
| 2026-09-22 | Follow-up review / sampling boundary | 72 CPU tests; `temp/pcc-review-sampling-tests-20260922-a01.log` | All passed in 50.388 s; one-pass causality and sampler output/budget checks added | Preserve ongoing sampling; resolve separate dev/test assignments and context export after completion |
| 2026-09-22 | Sequential automation | `temp/pcc-automation-tests-20260922-a01.log`; `temp/pcc-automation-completion-check-20260922-a01.log` | 82-test suite passed in 83.371 s; additional final-log-failure test passed separately | Use `pcc pipeline` directly or queue it through the existing runner; B200 untouched |
| 2026-09-22 | Raw-text preparation | `temp/pcc-prepare-regression-20260922-a01.log`; `temp/pcc-prepare-focused-20260922-a01.log` | 93-test suite passed in 123.016 s; final 11 exporter tests passed in 38.788 s | Historical exporter validation; superseded by direct loading at the user’s request |
| 2026-09-22 | Direct sampled-data loading / export removal | `temp/pcc-direct-data-regression-20260922-a01.log` | All 87 CPU tests passed in 88.311 s; CLI dry-run/help and runner list passed | Existing train/validation directories feed the shared loader; no export stage or document-range config; test set optional; B200 untouched |
| 2026-09-22 | Full input checks before screening | `temp/pcc-input-check-regression-20260922-a01.log` | All 94 CPU tests passed in 92.146 s; check-only help/dry-run passed | Full budgets fail before training; ordered-context counts/fingerprints saved; optional check-only mode; test data and B200 untouched |
| 2026-09-23 | Real training path review | `temp/pcc-real-training-review-20260923-a01.log` | All 15 focused CPU training/probe tests passed in 37.843 s | Real optimizer loops already exist; plain pipeline trains adapters; clarified CLI/README and verified exact optimizer calls, learned weights, frozen backbone, and checkpoint replay |
| 2026-09-23 | Local A100 pretrained diagnostics | `docs/PCC_DEV_DIAGNOSTICS_20260923.md`; `temp/dev-diagnostics-20260923-a01/` | 42 pretrained checks passed; real teacher/shallow/PCC updates and exact one-pass equivalence passed; peak allocated 2.432 GiB | Target-permuted control blocked by singleton bucket 69; preserve rule/failure, explicitly resolve singleton policy; no scientific gain claim |
| 2026-09-23 | Local four-pair effectiveness screen | `docs/PCC_PROMISE_SCREEN_20260923.md`; `temp/pcc-promise-screen-20260923-a01/decision.json` | All eight adapters trained 128 updates, evaluated on identical 2M tokens; artifact audit passed. All deep arms improve over Base, but all lose to matched shallow controls with paired 95% CIs | `stop_negative_screen`; no selected pair or full PCC probe. Exploratory local split; B200 untouched; all local workers finished |
| 2026-09-23 | Joint full-backbone implementation/readiness | `docs/PCC_JOINT_READINESS_20260923.md`; `temp/pcc-joint-ready-20260923-a01/` | 109 CPU tests passed; Base/Shallow/Deep real 2048-token/global-32768-token checks passed; Deep GPU resume gave bit-identical next-step loss/weights; peak 11.338 GiB | Six-run manifest and 50.3M-token shared inputs ready; scientific training not launched; B200 untouched |

## Operational lessons

- A real 32,768-token global update can still produce a singleton crossed
  position/norm bucket. Local diagnostics confirmed bucket 69 had one target;
  within-bucket derangement failed as intended. This used a one-update teacher,
  not the full trained reference. Do not silently merge buckets or accept
  self-targets to make a diagnostic pass.

- Review found the original LM+alignment gradient check could mask a detached
  tail. It now tests LM-only gradients independently, with fault injection.
- A loose bf16 causality tolerance could hide small leaks. Same-shape causal
  invariance now requires exact equality; segment visibility has an independent
  reference and cross-segment/padding perturbation checks.
- Runtime checks use explicit exceptions, not optimization-removable asserts.
- Local checkpoint loading rejects incomplete/mismatched parameter loads.
  A directory's revision label still does not authenticate its weights.
- Provenance is saved before completion; JSON publication is atomic and does
  not overwrite existing artifacts. Input segment IDs cannot reconnect segments.
- Tiny bf16 checkpoints differed by about 1.1e-4 in a microbatch-shape comparison.
  The fp32 update comparison passed at atol=2e-7, rtol=1e-5 with unequal target
  counts, supporting correct global normalization. BF16 checkpoints are not
  promised identical across batch shapes. Keep matched arms on the same setting.

## Backup / portability

List which small result/config artifacts are retained locally, which large
artifacts remain on temporary remote storage, and the authorized backup method.
Keep shared code and docs on the development repository rather than only in a
temporary execution worktree. Track project guides except the local-only
`AGENT_GUIDE.md` and `DROPBOX_ACCESS.md`; never track populated
credentials or `temp/INSTRUCTION.md`.
