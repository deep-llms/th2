# Current task

## Authorized no-reuse control N-A — 2026-10-20

User approved one additional matched training run after reviewing the failed
quality gate. Follow `docs/BDOT_v2_spec.md` Phase A control N with curriculum A:
fresh random initialization, seed 0, 4,768 steps, same data/order, schedule,
batch and FA4/grouped implementation as completed P1-A; only `reuse=false`.
Every token must use each of the eight middle modules exactly once.
No R, Phase B, extra seeds or longer training are authorized by this request.

Execution target remains th2/main on `thiennh-p6-6y3o-worker-0`, Dropbox 6Y3O.
Fresh output: `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01`.
First inspect the completed pilot's current burns and shared assets. Reuse
existing generators, training and supervision; compare N's generated recipe
with the completed P1-A arguments. Copy/verify Accelerate, reclaim only known
burn workers, require all eight GPUs free, then train. Existing supervisor
restores and verifies communicating burns after success or failure. Preserve
all prior outputs, shared HF caches and all N checkpoints.

Preflight **f5ddec8**, observed 23:36:03 UTC, passed: eight known burn workers
165144–165151, guard released, prior T12/T14 and pilot complete, ~25 TB free.
Four local configuration/no-reuse/forced-choice-gradient tests pass. A local
generation compared with the actual remote P1-A arguments confirms only the
reuse flag and new output/config paths differ. Launch preparation will also
check training source hashes and complete data/validation recipes before GPU
reclamation. Status: submitting preparation and the supervised N-A job.
Evidence: `temp/pilot_N_20261020/`.

## Pilot complete — verified 2026-10-20 23:05 UTC

Concise experiment definitions and results:
[BDOT_PILOT_RESULTS_20261020.md](BDOT_PILOT_RESULTS_20261020.md).

Read-only status **7a4497a** resolved the earlier delayed acknowledgement.
T12, T14, D, P1-A and P1-B all exited successfully. The queue finished at
22:46:31 UTC; automatic burn restoration was verified at 22:47:42 UTC.
Fresh observation at 23:05:08 UTC confirms eight identified burn workers,
100% utilization on every GPU, all-rank readiness/progress records and the
supervisor guard released. No handoff error was reported.

All three runs completed the full 4,768-step schedule. Final validation uses
the same 9,704 sequences / 9,915,698 scored tokens:

| Arm | Final loss | PPL | Trainer runtime |
|---|---:|---:|---:|
| D | 3.49446352 | 32.9326 | 53m08s |
| P1-A | 3.61594091 | 37.1863 | 58m20s |
| P1-B | 3.63270401 | 37.8149 | 60m40s |

Each output retains all 20 checkpoints. Final checkpoint metadata is at step
4,768; model weights, optimizer, scheduler and all eight rank RNG files exist
and are nonempty. This inspection does not constitute a new resume test.
P1-A is better than P1-B by 0.01676 nats, but both fail Gate 0a's quality
condition (at most D + 0.01 nats). R/N and route interventions were not run;
no claim is made that the remaining gate criteria were evaluated. No further
experiments were launched.

Export **fa9f74e** was acknowledged with 12 files pulled, zero skipped. All
nine result-file SHA-256 hashes match the node receipt; supervisor/run/burn
JSON matches the read-only observation. Raw local results:
`temp/pilot_final_20261020/{D,P1-A,P1-B}/`, with handoff receipts under
`temp/pilot_final_20261020/supervised/`. Durable evidence:
[final receipt](results/bdot_pilot_final_20261020.json).
Remote root: `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01`.
Commands return to #0 after the acknowledged export. No model, environment,
data or GPU process was changed during this status check.

## P1-A loss-history analysis — 2026-10-20

User asked why P1-A loses to D. Read-only export a9f888f retrieved completed
D/P1-A metrics; no training changes or new GPU work. Same 1,024-sequence periodic
validation: P1-A minus D is -0.03406 nats at 400, +0.00055 at 500, +0.07546 at
1,000, +0.11797 at 2,500 and +0.11978 at 4,768. Full-validation gap is +0.12148
(12.92% higher PPL). The opening at 477 has exactly zero immediate validation
jump, and neither run has a flagged >0.1-nat within-phase spike.

Early exploration coincides with the initial regression; learned router
probabilities become very sharp. At update 1,000 routes are still almost all
home (off-default 0.05%) despite the loss gap. Later module concentration grows:
max/min load ratio 1.01 at 1,000, 11.96 at 2,500, 63.97 at 4,760. At 4,760,
module 6 receives 33.14% of choices and module 2 only 0.52%; uniform baseline
share is 12.5%. Final validation off-default share 45.21%; 99.86% of tokens
reuse at least one module (not 99.86% of routing decisions).

Interpretation: evidence suggests an optimization problem during hard-routing
exploration followed by concentrated/repetitive module use. Causality is not
proved, and no routing-specific bug is excluded by the dense-equivalence check.
Wait for the already-running soft-to-hard P1-B before choosing another action.
Evidence: `docs/results/bdot_pilot_A_analysis_20261020.json`; raw histories:
`temp/pilot_analysis_20261020/`. Commands return to #0.

## Pilot progress — observed 2026-10-20 21:47 UTC

Read-only status a984b0f confirms D and P1-A completed all 4,768 updates,
with successful job exit and validated completion receipts. All 20 checkpoints
are retained in each output. D took 53m08s in Trainer (54m job wall time),
P1-A 58m20s (59m13s job wall time). Full validation uses the same 9,915,698
scored tokens / 9,704 sequences:

| Arm | Final loss | PPL | Status |
|---|---:|---:|---|
| D | 3.49446352 | 32.9326 | Complete |
| P1-A | 3.61594091 | 37.1863 | Complete |
| P1-B | pending | pending | Running; step 100 at 21:47 UTC |

P1-A's loss is 0.12147738 nats worse than D. Its off-default share is 45.21%;
complete stability/route-intervention analysis awaits the full results.
P1-B began at 21:44:58 UTC and has finite, decreasing training loss on all eight
GPU workers. Estimate completion around 22:45–23:00 UTC, subject to hard-phase
speed. No intervention or new experiments. Automatic burn restoration remains
pending queue completion; supervisor guard is held by the live training queue.

Evidence: `docs/results/bdot_pilot_status_20261020_2147.json` and raw status log
under `temp/pilot_20261020/`. Execution commands.sh returns to #0 after this
read-only check. Next: verify P1-B completion, collect all final metrics, verify
burn restoration, then assess curriculum selection. R/N remain unqueued.

## Authorized Stage 0a launch — 2026-10-20

User authorized running the necessary experiments after dispatch verification.
Design: `docs/BDOT_v2_spec.md` v1.4 and its curriculum review response.
Queue: T12 (1,000-step horizon, stop 200/resume 100) → T14 (fresh native and
fixed B-DOT without FiLM, 500 updates each) → D → P1-A → P1-B. Failed checks
stop the queue and restore burns. R/N follow curriculum selection later; no
Phase B or extra seeds are launched automatically.

Pilot: random initialization, seed 0, 12 layers, FA4/grouped experts, all eight B200s,
sequence 1,024, microbatch 16 × accumulation 4 = 524,288 input tokens/update, 4,768 complete
updates, 96 warmup, cosine floor 0.1, peak LR 3e-4. No pilot cutoff. Evaluate every 100 steps plus
curriculum boundaries; save every 250 steps and retain all checkpoints. Shared existing
English train/validation split, EOS/isolation/reset positions, seed 0 packed order.
CPU preparation verifies input hashes and exact document overlap once, while
burns continue; no exported token dataset is introduced.

Execution checkout `/disk/thuat/DOT-th2-inventory`, th2/main, host
`thiennh-p6-6y3o-worker-0`. Read-only preflight 215b5c7 at 19:23 UTC passed:
prior burn identities match, guard released, assets/cache exist, ~25 TB free.
New output root `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01`.
Preparation session `dot_pilot_prepare_20261020_a01`; GPU supervisor session
`dot_pilot_train_20261020_a01`. Launch **d24bf55** acknowledged at 19:29 UTC; CPU preparation completed.
Read-only status **9d4e80a** observed at 19:36 UTC confirms preparation passed: all
3,008 hashes verified, 29,278,633 train sequences, zero exact document overlap;
1,024 periodic / 9,704 final validation sequences. Accelerate copied and verified
at `/dev/shm/.cache/huggingface/accelerate/default_config.yaml`. Known burn
workers 125740–125747 were reclaimed by identity, all eight GPUs empty at
19:35:27 UTC, then the supervisor started T12. Do not relaunch. Local tests of initialization,
fail-closed loss gate, full-schedule cache preparation and generated queue all
pass; config regression passes. Training model/loss/optimizer code is unchanged.
T12 passed at 19:39:12 UTC (eight ranks, 200 updates, horizon 1,000):
16 comparisons, zero loss differences, exact restored/final state.
T14 passed at 19:51:44 UTC: native/B-DOT last-100 training loss
5.26911510 / 5.26604510 and periodic validation 5.22482845 / 5.22071967.
Gaps 0.003070 / 0.004109 nats, both below 0.01. Initial weights match exactly;
validation uses identical 1,046,305 scored tokens. Native/B-DOT throughput
846,558 / 783,529 input tokens/s including eval/save.

D started automatically at 19:51:44 UTC. Status 86b659f observed at
19:52:47 UTC confirms step 20, finite decreasing loss, all eight GPU workers
active and supervisor ownership held. P1-A then P1-B are queued. Estimated
remaining wall time is about 3–4 hours, pending actual routed timings.
Evidence: `docs/results/bdot_pilot_launch_20261020.json`; raw files under
`temp/pilot_20261020/`. T12 and T14 summary hashes match runner completion
receipts. Final documentation push returns execution commands.sh to #0;
source commands.sh is also #0. The independent supervisor remains running.

Next: retrieve D/P1-A/P1-B results and final burn handoff, then apply the
spec's curriculum selection before proposing R/N. No extra automatic runs.
The supervisor will restore verified communicating burns on success or failure.

## Dispatch overhead optimization passed — 2026-10-20

User authorized profiling and reducing overhead while preserving
`docs/BDOT_v2_spec.md`. Implemented conservative home-route discovery bypass,
hard-ST filtering bypass, fixed-size counts, one reference scatter per stage,
top-1 argmax, nonduplicated RoPE and lazy diagnostic formatting. Architecture,
data, losses, FiLM, optimizer and environments remain unchanged.

Verification **a5f65aa** completed successfully on B200 6Y3O:
`/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-opt-20261020-a04`.
CUDA SDPA/FA4 old/new comparisons, checkpointed DDP, eight-rank A/B resume,
18 unprofiled timing jobs and three profiles passed. No tolerances loosened.
Earlier a01–a03 direct-home attempts failed a multi-update GPU comparison;
that shortcut was removed. Each attempt restored verified burns.

Mean seconds/update before → after: D **0.707021 → 0.664589**, hard grouped
**0.772241 → 0.733153**, soft grouped **0.999200 → 0.956848**.
Native reference **0.617564**. Time reductions 6.0%, 5.1%, 4.2%; hard remains
18.7% slower than native (10.3% slower than D). Default backend stays reference.
Profiler synchronizations: D 224→28, hard 260→96. Full report:
[dispatch optimization](BDOT_DISPATCH_OPTIMIZATION_20261020.md).

Accelerate was copied/verified before testing, known burns reclaimed by identity,
and all eight GPUs verified free. Automatic burns verified at **16:47 UTC**;
independent status **b9829d6** at **18:25 UTC** confirms workers 125740–125747,
released guard and collective cycles 6840→6850. No research pilot was launched.

All 14 files from export **6b5865a** were downloaded and the summary hash
verified against remote inspection and completion receipts. Durable evidence:
`docs/results/bdot_dispatch_optimization_20261020.json`; raw files:
`temp/dispatch_opt_20261020/results/`. The final full local suite passed
**48 tests in 321.35s**. Eight old/new CPU three-update cases have zero gradient
and final-weight differences; all 32 GPU cases also have zero differences.

This optimization task is complete. Final publication leaves both execution
and source commands.sh at #0. No more verification
launch is needed for these changes. Further research runs need their own scope;
the design's full-width pilot acceptance tests remain separate requirements.


## Grouped-expert comparison completed — 2026-10-20

The authorized reference-versus-grouped comparison on 6Y3O completed successfully
at 14:52:44 UTC. All CUDA numerical, two-rank distributed and eight-rank A/B
resume checks passed; all 14 complete-update timing runs passed. Both resume
suites have zero loss difference and exact model/optimizer/scheduler restoration
and final state across every rank. No research pilot was launched.

Mean seconds/update (reference → grouped): D 0.707639 → 0.713127;
hard 0.807290 → 0.768397; soft 1.026440 → 1.005195. Native mean 0.615911.
Grouped improves hard throughput by 5.1% and soft by 2.1%, with disjoint observed
repeat ranges. Default expert_backend remains reference; keep backend fixed on
resume. Report: [grouped comparison](BDOT_GROUPED_GEMM_20261020.md).
Design remains `docs/BDOT_v2_spec.md`; this comparison does not replace its
full-width 500-update T14 or revised 1,000-step-horizon T12 pre-pilot checks.

Launch `5648b88`, execution checkout `/disk/thuat/DOT-th2-inventory`, th2/main.
Root: `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-grouped-20261020-a01`.
Accelerate was copied and verified; only identified burn workers were reclaimed;
eight GPUs were confirmed free before testing. Automatic communicating burns
were verified at 14:53:50 UTC. Independent live observation at 14:55:56 UTC
confirmed workers 113695–113702, released guard and cycles advancing 160→170.
No environment, driver, prior baseline output or corpus was changed.

All eight requested result files were exported by `73a9bb5` and downloaded to
`temp/grouped_20261020/results/`. The final summary matches its remote and
completion-manifest SHA256. Durable receipt:
`docs/results/bdot_grouped_comparison_20261020.json`.
The two initial read-only preflight failures (inline syntax, absent Git metadata)
were corrected before launch; neither signaled GPUs. Final retrieval was delayed
but completed without a reported infrastructure failure. Return execution
commands.sh to #0 with the final report publication; source commands.sh stays #0.

## Complete pilot recipe implemented and locally checked — 2026-10-20

The user accepted the complete short schedule in `docs/BDOT_v2_spec.md` v1.4
and `docs/BDOT_curriculum_context_review_response.md`, with minimal changes to
the existing infrastructure. No new training launch is authorized in this turn.
Pilot D → P1-A → P1-B uses 4,768 updates, 96 warmup updates, ordinary HF packing,
seed-0 packed order and the existing English train/validation splits.
The generator, shared validation selection/overlap check, warmup guard and
transition evaluation are implemented. LR and routing use the same full horizon;
the pilot has no early cutoff. The prior independent-router-horizon proposal
below is superseded. See [implementation](BDOT_IMPLEMENTATION.md).

Local CPU verification covers the 500-update native/B-DOT equivalence fixture,
curriculum boundaries, transition evaluation, checkpoint resume and frozen
validation reuse by standalone evaluation. The 46-test ML suite passed.
The B-DOT suite passed 43/45 initially; both CLI failures were resolved and their
affected cases rerun successfully. One was a source-receipt change during the
test; the packed case exposed checkpoint tokenizer metadata and filtered-token
count checks, now corrected without changing packed data or training math.
Evidence: `temp/pilot_v14/bdot_suite.log`, `ml_suite.log`,
`cli_verified.log` (legacy CLI passed), `packed_cli_verified.log` (packed CLI
passed after fixes), and `manifest_resume_verified.log`.
Syntax checks pass for all 14 changed Python files.

No remote command, push or B200 training was submitted; commands.sh stays #0.
The real-pool overlap check and full-model B200 pipeline verification are still
pending before a research launch. CPU tests do not establish GPU throughput.

## Curriculum/context review requested — 2026-10-20

The user requested a document for another agent to review before choosing the
routed pilot schedule. See
[curriculum and context review](BDOT_CURRICULUM_AND_CONTEXT_REVIEW_20261020.md).
It confirms the full English pool/full 57,220-step LR schedule with cutoff 5,000,
distinguishes the existing complete short-pilot generator from the proposed
long-schedule routed screen, and explains the sequence-length speed comparison.
No training configuration/code changes or new launches were made for this review.

## Baseline completed — verified 2026-10-20 09:40 UTC

Native Qwen3 12-layer FA4 baseline finished **5,000 updates successfully** at
05:22:42 UTC. Runtime was 3,040.22 seconds in Trainer (50m40s), 3,083.11 seconds
for the complete baseline job (51m23s). Full schedule remains 57,220 steps;
all 20 checkpoints, every 250 steps through 5,000, are retained. The fresh
final-artifact check passed: weights, optimizer, scheduler, all-rank RNG files,
recipe and step metadata. Final held-out loss **3.46775378**, PPL **32.06463718**,
over 9,915,698 scored tokens in 9,704 sequences. Mean training loss 4.22683868
is the entire-run average, not the final validation metric.

Supervisor restored communicating burns by 05:23:48 UTC. Live check at 09:40 UTC
matched all eight workers 104144–104151 to the verified burn launcher, confirmed
the guard was released, and observed collective cycles advance 17,640→17,660.
No new training/evaluation was launched. commands.sh is returned to #0.
Final checkpoint: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02/prepared/baseline/checkpoint-5000`.
Evidence: `docs/results/native_baseline_final_20261020.json` and local
`temp/attention_study_20261020/final_status01.log` (read-only inspection e19e34d).
The authorized baseline workflow is complete; further method training needs
its own scope from the user. Historical launch observations follow below.


## Live baseline verified — 2026-10-20 04:35 UTC

The authorized setup/comparison/download/verification sequence is complete;
the fresh 5,000-update native Qwen3 12-layer baseline is actively training.
Execution root: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02`,
output `prepared/baseline`, supervisor session `native12_train_20261020_a02`.
Implementation/launch commit caea83d; subsequent commits only retrieve status.
**Do not launch it again.** Source/execution commands.sh are returned to #0.

FA4/train selected after stock torch2.8/FA2 and compiler-fixed FA2 comparisons.
All 3,008 English dataset files and six tokenizer/config files pass pinned
SHA256 checks. Caches reused in 8.917 seconds with 29,278,633 train and 9,704
validation sequences. Original train/validation split, EOS, within-document
causality and reset positions are verified. Protected train/eval environments
and the NVIDIA driver remained unchanged.

Corrected eight-rank resume parity passed (weights max abs 3.7253e-9,
optimizer 2.3283e-10; scheduler/eight-rank RNG match). Real-model smoke25 and
resume50 both passed with matching training/data receipts. The first a01 smoke
failed before training due solely to the modern-cuDNN metadata getter; its
outputs remain separate. Only that reader changed; no precision flags changed.

At 04:35:32 UTC, baseline had logged step 340 (loss ~6.228), checkpoint250
complete including optimizer, scheduler, all eight RNG files, and matching
recipe. Validation250: loss 6.60917368, PPL 741.869745 over 9,915,698 scored
tokens/9,704 sequences. GPUs 0–7 each have one training worker, 97–99% utilization,
54,110 MiB device usage; guard held by the supervisor. No run failure observed.

Recipe: seed0, seq1024, micro16 × accumulation4 ×8 GPUs =524,288 input tokens,
full schedule57,220/cutoff5,000/warmup1,145, BF16, tf32False, checkpointingFalse,
AdamW3e-4 cosine floor.1, eval/save250, retain all checkpoints. The supervisor
will restore and verify communicating burns automatically after success/failure.
Estimated completion around 05:30 UTC (roughly one hour total, including eval/
saving; estimate from early observed progress, not a completion promise).

Receipt: `docs/results/native_baseline_launch_20261020.json`; report:
`docs/ATTENTION_BASELINE_LAUNCH_20261020.md`. Raw local evidence in
`temp/attention_study_20261020/`, especially baseline_health01.log and
retry_smoke25.json/retry_smoke50.json. Next status check should read the existing
queue, final baseline-5000.json, supervisor and burn-verified receipts; require
step5000/artifact success and live verified burns before declaring completion.


## Authorized overnight baseline workflow — 2026-10-20

The user authorized: install a separate older-PyTorch/FA2 environment; compare
it with production FA4 and the locally built FA2+C++20 environment; select a
correct, fast runtime; download only `nht10/cx_sampled`'s English
`qwen3_0.6b_base_en_30B` subset; run needed training/resume checks; launch the
original native Qwen3 architecture with 12 layers for 5,000 optimizer updates.
The design remains `docs/BDOT_v2_spec.md`. This baseline has no routers/FiLM and
is randomly initialized, as the spec requires. No further research arms are
authorized. All eight GPUs are used for complete-update timings and training.

Tie preference: FA4 approximately stock FA2+older PyTorch, both ahead of the
patched-source FA2 build. A speed win must exceed repeat variability and 5%
on native full updates without hurting the relevant routed-case throughput.
Every eligible choice must pass numerical checks; the selected choice must
also pass distributed training/checkpoint/resume verification before training.

Keep train/eval/train_fa2, system CUDA and the NVIDIA driver unchanged. New env:
`train_fa2_t28`, torch 2.8.0 / CUDA 12.8 and official FA2 2.8.3.post2 wheel.
Check compiled SM100 support and imports before GPU use. If incompatible, leave
it unused and compare FA4 with the fixed FA2 build, per the user's correction.
Use the established verified-burn handoff and restore burns on completion/error.

Read-only preflight `8b56289` verified 6Y3O at 01:03:55 UTC: protected imports
pass; known guard/burn hashes match; eight original burn workers remain active;
zero volatile uncorrectable ECC. The new environment is absent. This replacement
node has no project dataset/model, so tokenizer/config assets must also be
downloaded through the controller. Source commands.sh stays inactive.

Proposed recipe: spec sequence 1,024, 524,288 input tokens/update, seed 0,
EOS/isolation/reset positions, AdamW 3e-4 and 2% warmup. An optional question is
pending about full schedule vs cutoff. If unanswered, preserve continuation:
full 30B-token schedule, stop_after=5000. Record exact schedule, data revision,
preprocessing and validation split before launch; reuse the pool's existing
validation split rather than holding out training documents a second time.

Execution checkout: `/disk/thuat/DOT-th2-inventory`, th2 main. Current study root:
`/mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/`.
Local evidence: `temp/attention_study_20261020/`.

Runtime update, 01:35 UTC: stock FA2 2.8.3.post2 installed offline in
train_fa2_t28 after SHA256 verification. torch 2.8.0 / CUDA 12.8, C++11 ABI,
SM100 and compiled FA2 imports pass. Protected train/eval/train_fa2 snapshots
remain identical. The controller cache lacked the wheel; initial #datasets
request failed before creating a destination. Established #models single-file
request 16adf56 succeeded, then f8f9b75 performed the isolated offline install.

Comparison 4e1b1f6 completed successfully at 01:48:24 UTC. FA4 selected:
mean native update 0.615090 s, stock FA2/torch2.8 0.626876 s, patched FA2
0.610053 s. All nine numerical checks and all twelve timing jobs passed.
All eight enhanced communicating burn workers 12643–12650 were verified,
with matching live ownership and guard released, by 512d624 at 01:48:31 UTC.
See `docs/results/attention_comparison_20261020.json` for the compact receipt.
The selected-runtime distributed resume and real-data smoke gates remain.

Plan: `docs/ATTENTION_BASELINE_LAUNCH_20261020.md`. Implementation committed in
f8f9b75: explicit public FA2 backend sharing FA4's document-layout transport,
comparison/asset/queue helpers, local 44 ML regressions + seven old-torch CPU
adapter checks + four selection tests + queue preparation contract passing.
Committed `scripts/native_baseline_pipeline.py` prepares exact Trainer
caches while burns remain active, then gates the verification/baseline queue.
Its handoff test passes; committed in 7495e85.

Submitting the English-only download after selection/burn verification:
`temp/attention_study_20261020/download_commands.sh`, 3,016 exact single-file
#models URLs (3,008 English files, two pool manifests, six tokenizer/config
files), pinned revisions and unique destination paths. Model weights are not
needed for random initialization. Full schedule defaults to 57,220 steps,
cutoff 5,000, warmup 1,145; optional schedule question remains unanswered.

Download update (~02:14 UTC): the initial single-file list would take many
hours because of controller per-file overhead. Its exact task
`2026-10-20_01-51-37` was stopped (41647a8); existing files retained. Repartitioned
into eight disjoint shard groups, preserving all 3,016 requested inputs.
Batch commands: ignored `temp/attention_study_20261020/download_batch_0.sh`
through `_7.sh`. All eight batches acknowledged by 02:17 UTC.

| Index | Commit | Controller task ID |
|---|---|---|
| 0 | 398324c | 2026-10-20_01-58-08 |
| 1 | 7675f26 | 2026-10-20_01-59-36 |
| 2 | 25edec4 | 2026-10-20_02-01-18 |
| 3 | 8d623e9 | 2026-10-20_02-08-49 |
| 4 | 2276f60 | 2026-10-20_02-09-50 |
| 5 | 1ee6e34 | 2026-10-20_02-11-59 |
| 6 | 5c8756a | 2026-10-20_02-14-37 |
| 7 | 28ff772 | 2026-10-20_02-16-05 |

Controller per-task progress lines can be stale. Read-only node inspection
15322e4 at 02:06 UTC showed groups 0/1/2 all receiving files despite their stale
controller lines. All six tokenizer/config SHA256s passed. All eight verified
burns 12643–12650 remained at 100% utilization, guard released. No node/process
damage observed. Source: `temp/attention_study_20261020/corpus_arrival01.log`.
A local archive fallback was briefly prepared, then its exact owned CPU process
was stopped and its output removed when concurrency was confirmed. Nothing was
uploaded to HF; no dataset source was changed.

Next: monitor all eight and verify actual
file arrival/completion. Then use the reviewed
`temp/attention_study_20261020/launch_baseline_commands.sh` to run
`scripts/native_baseline_pipeline.py` under production train in a tmux session.
Fresh root: `/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a01`.
Pipeline: verify every dataset/model SHA256 and document count → prepare native
12-layer jobs → CPU-only exact Trainer cache fill while burns continue → require
predecessor verified burn ownership → fresh inventory/Accelerate copy+env check
→ supervised eight-rank resume test → real smoke25/resume50 → fresh baseline5000.
The GPU supervisor always restores and verifies communicating burns on exit.
Keep working until baseline training actually starts; do not end at download
or cache preparation. No other research arms are authorized.

At 02:21:54 UTC, node inspection f668079 observed 463/3,008 data files at
expected size (~21.36 GB), every group advancing. Node has 192 CPUs, unrestricted
CPU quota, ~1.96 TB MemAvailable. Baseline preprocessing is fixed to 128 workers
(before any real cache/training), leaving CPU capacity for the node; this value
is saved in the data recipe and must match future runs. Cache-helper CPU smoke
using SDPA metadata passed twice with identical train/validation recipes and
cache reuse (2,052 train sequences, 64 validation). FA4 numerical checks remain
the separate completed B200 evidence. Initial local FA4 cache-fixture attempt
failed only because local train_env has no FA4 package metadata; no production
code change was needed. Source log: `cache_helper_sdpa_test.log`.

At 02:32:32 UTC, read-only inspection ffe57c7 showed 820/3,008 data
files present at the expected size; all eight transfer groups advance and all
six tokenizer/config hashes pass. The same eight burn workers remain healthy.
Source: `temp/attention_study_20261020/corpus_arrival04.log`.

The baseline pipeline now supports a bounded six-hour input wait before its
existing verification/preparation sequence. It authenticates the downloaded
manifest against the pinned SHA256, waits for every data/tokenizer file at its
expected size and unchanged timestamps over two observations 30 seconds apart,
then verifies every SHA256. Four CPU tests cover missing/partial inputs,
same-size rewrites, timeout and invalid metadata. It does not retry transfers,
modify inputs, or touch GPUs during waiting. Commit c52d603 submitted the prepared launch. The node acknowledged it at
02:35 UTC: PREDECESSOR_BURN_HANDOFF_VERIFIED and 945/3,014 data/model files
(~42.14 GB). Log: `temp/attention_study_20261020/baseline_wait_launch.log`.
Continue monitoring until actual baseline training starts. The fresh output
root remains the a01 root above. Background controller transfers stay active.

Monitoring update, 03:10:27 UTC: commit 7779ab5 read-only inspection shows
2,091/3,014 data/model files at expected size (~93.45 GB), preparation tmux
alive (PID 37192), no failure receipt, training session absent as expected.
All eight burns 12643–12650 remain at 100% with guard released. Evidence:
`temp/attention_study_20261020/pipeline_status02.log`. The prepared pipeline is
already submitted and waiting; do not submit the launch a second time. Latest
execution command only reads status. Continue until hash/cache/resume/smoke
checks pass and the real baseline advances.

All eight controller download batches ended by 03:45:21 UTC. Read-only
inspection 4df3fe2 at 03:45:55 UTC confirms assets-verified.json status passed:
all 3,008 dataset file SHA256s, metadata/model/tokenizer checks, 36,495,424
training documents across 751 shards and 11,717 validation documents in one
shard. Pipeline is preparing the exact native-baseline/cache configuration;
all eight original verified burns remain active. Evidence:
`temp/attention_study_20261020/pipeline_status04.log` and `batch_status25.log`.
Continue through CPU cache preparation, GPU resume/smoke gates and real training.

GPU queue started at 04:07:56 UTC after successful cache preparation
(1,230.74 seconds; 29,278,633 train sequences, 9,704 validation sequences).
The actual Accelerate cache in the tmux environment is
`/dev/shm/.cache/huggingface/accelerate/default_config.yaml`; the launcher
resolved that path, copied/byte-checked resources/accelerate_b200.yaml, and
verified the selected Python reports MULTI_GPU / 8 / BF16 / B200. It invokes
Accelerate via the explicit conda Python module, not the PATH console script.
Reclamation signaled only freshly verified workers 12643–12650, then logged
ALL_EIGHT_GPUS_VERIFIED_FREE. Guard is held by the active supervisor.

Selected FA4 eight-rank resume verification passed: max weight error
3.7253e-9, optimizer tensor error 2.3283e-10, scheduler and all eight RNG states
match. Seven-sequence distributed evaluation also passed, exercising an empty
rank. Receipts downloaded to temp/attention_study_20261020/: cache-ready.json,
launch.json, accelerate_env.txt, selected_accelerate_env.txt, reclaim.json,
resume-complete.json. The remaining automatic stages are real-smoke25,
resume50, and fresh baseline5000. Continue to verify actual baseline progress.


At 04:11 UTC, a01 real-smoke25 failed before the first optimizer update:
the legacy torch.backends.cudnn.allow_tf32 getter raises when HF sets
convolution precision to IEEE independently of RNN precision. This was a
resume-metadata compatibility bug, not a kernel/data/OOM failure. Supervisor
finished at 04:12:27 with all GPUs free before verified communicating burns
93042–93049, guard released. Full local evidence: smoke25-failure.log and
pipeline_status09.log under temp/attention_study_20261020/.

Fix: read public per-operator precision fields on modern torch without changing
any flags; retain legacy reader for older torch. Seven resume-recipe tests pass,
including the exact split-cuDNN-policy reproduction and mismatch rejection.
The GPU parity probe now explicitly uses tf32=False, matching the real recipe.
All data hashes and caches remain valid. Retry uses fresh a02 outputs, retaining
a01 evidence. Its separate burn-predecessor gate requires completed cleanup,
matching verified burn receipts/live ownership and released guard, even though
a01 training failed; runtime comparison still independently must pass. Five
download/handoff tests pass. The same eight-GPU parity, smoke25/resume50 and
fresh baseline5000 stages remain mandatory. No precision/runtime/data change.


Retry caea83d was acknowledged at 04:24 UTC. Inspection f87a983 observed at
04:27:09: hashes/documents pass again, exact cache reuse in 8.917 seconds with
unchanged sequence counts; Accelerate copied/verified in /dev/shm/.cache;
only workers 93042–93049 reclaimed and ALL_EIGHT_GPUS_VERIFIED_FREE logged.
Eight-rank resume gate running under active supervisor. Fresh a02 root:
/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a02.
Source evidence: temp/attention_study_20261020/retry_status01.log.
Do not submit a duplicate launch. Wait for parity, smoke25/resume50 and actual
baseline advancement. Current monitoring commits only inspect live state.

## Isolated FA2 installation complete — 2026-10-20

The user authorized installation from official GitHub/PyPI inputs while
preserving B200 6Y3O stability. FA2 **2.8.3.post2** is now installed in
`/mnt/local/conda-py311/envs/train_fa2` against torch **2.14.1+cu130**.
The wheel was built offline from the SHA256-verified PyPI source with
CUDA 13.0.88, C++11 ABI enabled and SM100 only. The successful build took
1,625 seconds (~27 minutes), using four compiler jobs and one NVCC thread;
peak sampled build-process RSS was 16.55 GiB. No GPU workload was launched
or stopped, and no production backend or driver was changed.

The first source attempt (`6d551cb`) failed on C++17 versus the newer torch
headers. Retry `2f8b97e` changed exactly four setup.py compiler flags to C++20,
without kernel-source edits. `c11e34a` added the missing `libcudart.so` linker
symlink inside train_fa2 only. `scripts/install_fa2_offline.py` now includes
both build compatibility fixes for reproducibility. The env text installs
prerequisites; FA2 itself still requires this separate offline wheel build.

Read-only final verification `d09dc58` completed at **00:14:41 UTC**:
compiled FA2 fixed/variable-length forward/backward interfaces import in a
clean subprocess; train/eval and existing train_fa2 package metadata are
unchanged. Production imports including FA4 pass. All eight original burn
workers 500–507 under launcher 433 remain active at 98% GPU utilization;
volatile uncorrectable ECC and cgroup OOM/kill counts are zero.

Artifacts: `/mnt/local/_outputs/deep-llms_th2/6y3o-fa2-build-20261019-a02/`
contains `build.log`, `result.json`, `linker_name.json`, and the built wheel.
Wheel SHA256: `a34f9f06f824a60e233df06dc60668526b97c3cdf12cf00bbc39478008c9aadd`.
See [installation receipt](results/fa2_install_success_20261020.json) and
[environment/build notes](FA2_B200_ENVIRONMENT_20261019.md) for local log paths,
checksums, exact wheel tag, and upstream support evidence.

**Remaining:** GPU numerical, throughput and resume tests are not performed
by installation/import verification. Do not adopt FA2 for production based
only on these receipts. Source and execution commands.sh return to #0;
no build, test or new workload remains queued by this task.

## Node health after the failed FA2 request — 2026-10-19

The user emphasized preserving B200 stability. Read-only inspection `161d804`
completed at **23:27:14 UTC** on 6Y3O. All eight GPUs respond with unchanged
UUIDs and driver 580.167.08, zero volatile uncorrectable ECC counts, and the
same GPU PIDs 500–507. Their parent 433 runs `/tmp/llm_pretrain_burn.py` through
system Python 3.10; all eight GPUs remain 98% busy. No processes were signaled.
`train` and `eval` match the earlier checked versions and again pass core/FA4
imports. Host available RAM is about 1.83 TiB and local free disk about 24.1 TiB;
cgroup OOM/kill counters are zero. No sign of damage was found in these checks.
This is not a full hardware stress test or a pre-install file comparison.
See [health receipt](results/b200_6y3o_health_20261019.json).

No further install, build or GPU test was launched. commands.sh returns to #0.
Future FA2 work must stay in `train_fa2` and a dedicated build directory,
preserve production environments/system CUDA/drivers, and bound compiler
parallelism after checking memory/disk availability.

The user also identified the official PyPI FA2 source distribution. Dev-only
inspection verified its published SHA256 and bundled CUTLASS header; the
GitHub tag source archive omits that header. Both setup scripts include SM100
build flags. An offline source build is another possible route, subject to
toolchain/dependency verification; an operator-supplied wheel is not the only
option. This corrects the overly restrictive earlier wording below. No source
archive has been downloaded to B200 and no source build has been submitted.

## FA2 installation failure diagnosed — 2026-10-19

The user requested the reason for the failed FA2 install. Read-only discovery
`6bb2024` located `/mnt/local/_env_status/train_fa2.status.json`; inspection
`d3434c5` at **23:18:04 UTC** returned its exact failure:
`result: missing`, `source: gap`, `overall: incomplete`. Its sole gap is
`flash_attn-2.8.3.post2+cu13torch2.14cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`.
Other requested packages report success. The node cache contains no matching
FA2 wheel. A fresh official release asset check also finds no matching wheel.
This confirms an unavailable requested prebuilt dependency, before FA2 kernel
execution; the controller's upstream wheel-store contents were not audited.

The install specification named the runtime-compatible artifact without first
establishing its availability. Reusing that prebuilt request requires a matching
torch 2.14.1 / CUDA 13 / Python 3.11 / C++11-ABI FA2 build made available through
the operator's wheel store; see the source-build alternative above. Do not
repeat the same failed request, substitute
an incompatible wheel, or change production torch/FA4 environments.

See [failure receipt](results/fa2_install_failure_20261019.json) and
[environment notes](FA2_B200_ENVIRONMENT_20261019.md). Local sanitized diagnostic:
`temp/dot_th2_6y3o/fa2_failure_detail_20261019_a01.log`. No reinstall, source
build, training or process stop was performed. Source and execution commands.sh
return to #0 with this diagnosis publication.

## Replacement-node Dropbox checks authorized — 2026-10-19

The user clarified that Dropbox checks may resume freely now that the new
6Y3O folder is available. This supersedes the earlier monitoring deferral.
The controller reports installation success for `train` and `eval` at
`e2dabb0`; the separate `train_fa2` request at `960bbc9` failed with exit 3.
Ordinary log export `2efa752` could not retrieve installer details because
there were no normal job logs yet. No installation was repeated.

Read-only inspection `5f06832`, job
`th2-6y3o-attention-env-inspect-20261019-a01`, completed on
`thiennh-p6-6y3o-worker-0` (started **2026-10-19 23:05:47 UTC**). All three
environments have the expected torch 2.14.1, CUDA 13.0, Python 3.11.15 and HF
pins; core imports pass. `train` and `eval` also import FA4 4.0.0b33 successfully.
Eval has lm-eval 0.4.13 installed; its task execution was not tested.
`train_fa2` has **no flash-attn package**, so installation is incomplete and
FA2 testing remains blocked. At this stage the installer error was unknown;
the subsequent diagnostic above establishes the missing-wheel result.

The node exposes eight B200s with driver 580.167.08 and 183359 MiB each. All
eight had active GPU processes; their identities were not checked or changed.
This inspection did not allocate tensors, test kernels or launch training.
Verified package/import data and raw-log hashes:
[receipt](results/b200_6y3o_environment_20261019.json). Raw files:
`temp/dot_th2_6y3o/runtime_20261019_a01.log` and `status_20261019_a03.log`.
Source and execution commands.sh return to **#0** with this receipt publication.

The subsequent diagnostic above resolves the installer error. Preserve
production environments and driver. CUDA numerical, throughput and resume
verification on this node remains separate future work.

## Isolated B200 FA2 environment — installation request, 2026-10-19

The user authorized creating `train_fa2` for a later FA2 comparison without
modifying the current training environment. Submit only
`#i envs/train_fa2.txt +a`, job `th2-fa2-isolated-env-20261019-a01`, to th2.
The Python 3.11 spec uses `fresh: false`, retains the existing torch/HF pins,
and requests the exact FA2 2.8.3.post2 CUDA 13 / torch 2.14 / C++11 ABI wheel.
The upstream release lacks this wheel; internal runner availability is unknown.
If missing, the operator must supply a matching build. Do not change torch or
reuse the local torch 2.8 wheel merely to complete installation.

See [installation and verification notes](FA2_B200_ENVIRONMENT_20261019.md).
Production environment specs, model code and drivers are unchanged. The request
failed and its partial runtime was inspected as recorded above; no GPU tests,
training or process stops are queued. The earlier Dropbox deferral is superseded
by the authorization above. Both commands.sh copies return to #0 after inspection.

## Local FA2 comparison — completed, 2026-10-18

The user asked whether stable FA2 should replace FA4, then requested testing on
the development machine only. FA2 2.8.3.post2 passes the local A100 kernel,
native-Qwen and routed B-DOT numerical, causality/isolation and activation
checkpointing checks. Two full-update timing repeats per model/backend give
FA2 throughput gains versus dense isolated SDPA of 11.22% for native Qwen,
9.76% for matched D, and 1.01% for hard mixed routing. The routed difference
is small relative to variation. All twelve runs completed; used GPUs are free.

This uses an isolated package overlay and the existing local torch 2.8.0+cu126
test runtime. Production model/backend code, train_env and drivers are unchanged.
The test reuses the FA4 packed-layout transport but executes the real FA2 kernel;
receipts label that substitution explicitly. There is no public FA2 training
option yet. This is not B200 performance, distributed/resume or long-run evidence.
See [report](FA2_LOCAL_VERIFICATION_20261018.md) and
[receipt](results/fa2_local_20261018.json). No new push or remote action occurred;
th2 still has the installation commit `e2dabb0`. Dropbox remains unchecked.

## Replacement node environment installation — 2026-10-18

The user confirms q359 is dead and requests an installation command pushed to
`deep-llms/th2` for the replacement node. **Do not monitor Dropbox until the user
asks.** The replacement hostname, GPU identity and Dropbox mapping are not yet
supplied; do not reuse q359-specific launch/preflight assumptions.

The execution checkout is `/disk/thuat/DOT-th2-inventory`, main. Its command is
`#i envs/train.txt envs/eval.txt +a`, job
`th2-dot-new-node-install-envs-20261018-a01`. This requests fresh Python 3.11
conda environments named `train` and `eval`. Both retain the existing model/data
library pins and add the old B200's validated FA4 4.0.0b33 (CUDA 13), CUTLASS DSL
4.8.0 and Quack 0.6.5. Eval uses the project's lm-eval 0.4.13 pin with its HF
extra and SentencePiece 0.2.2. The files are self-contained package lists.

Only installation is submitted. Installation success and CUDA functionality
remain unverified until the user requests a check. No training, download,
process stop, or automatic monitoring is queued. The old grouped verification
below has no retrieved final result; do not treat it as completed. DOT's source
commands.sh remains inactive; the execution checkout carries the install action.

## Grouped expert execution — implemented; remote verification incomplete, 2026-10-18

The user authorized implementing grouped GEMM and checking correctness and
complete-update throughput. `bdot/grouped.py` groups Q/O/MLP projections using
the pinned PyTorch CUDA BF16 primitive. `bdot.expert_backend` selects `reference`
(unchanged default) or `grouped`; original parameters/state names are preserved.
Only occupied experts enter autograd, preserving absent gradients for unused
parameters. CPU reference-kernel checks pass; real CUDA verification follows.

q359 read-only preflight `614c5d5` confirms the existing eight known communicating
burn workers and matching environment. Planned bounded verification reuses the
established supervisor, Accelerate installation/check, exact-PID reclamation,
free-GPU verification and automatic burn restoration. Output root:
`/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03/`.
Initial launch `ff5452b` passed projection/model gradient checks but stopped at
an overstrict fresh-AdamW coordinate tolerance (max 0.000600 at LR 0.000300).
The unchanged CPU BF16 reference independently reproduces the near-zero-gradient
first-step sensitivity. The revised probe records whole-update RMS drift and
checks identical existing moments separately. Initial supervisor restored and
verified all eight burns at 21:54:37 UTC. a02 passes real grouped projection and
all eight same-route SDPA/FA4 model cases (identical logits/losses; max relative
gradient difference 0.003245). A seeded fused-AdamW counter-device bug in the
DDP test fixture then stopped that suite; model code was unchanged. Launch
`ec19fe8` fixes the fixture and requires verified a02 burn restoration before a
fresh a03 preflight/launch. a03 passes CUDA numerics and all three checkpointed
two-rank BF16 probes (no-reuse, soft, home-only). Eight-rank A/B resume and
full-update benchmarks were submitted; their current status is unverified. Latest retrieved snapshot: 22:07 UTC;
local evidence `temp/grouped_a03_summary1.json`. Initial numerical evidence is
also in `temp/grouped_a02_summary1.json`.
At 22:20 UTC, Dropbox/controller status still ends at 22:07:25 UTC. Export
request `12881a3` and read-only live-status request `098061b` have no returned
receipt yet. No AWS/controller error has been returned, so the cause of the
missing updates is unknown. Do not infer completion, failure, or live burns.
The submitted supervisor restores/verifies burns on success or failure, but its
final handoff is not yet observed. Do not resubmit the GPU job. Retrieve the
pending receipt first; final source/results publication must use `#0`.
See [grouped verification notes](BDOT_GROUPED_GEMM_20261018.md).

User-authorized runner retry at 22:54 UTC: pushed `5c90b84` to th2 main,
verified the remote head, and submitted fresh read-only job
`th2-q359-dot-grouped-status-20261018-a02`. Its commands.sh includes an added
blank line, unique heartbeat/timestamp, existing result reads and bounded
nvidia-smi queries. It does not launch or stop GPU work. As of 22:58 UTC the
latest Dropbox listing and fresh `_RUN_STATUS_.log` download still end at
22:07:25 UTC; the retry commit has no returned acknowledgement. Local evidence:
`temp/grouped_retry_2257_list.jsonl` and `temp/grouped_retry_2257_status.log`.
Execution commands.sh remains this harmless read-only probe; DOT's source
commands.sh remains #0. Do not infer the current training/burn state from this
missing response, or resubmit the GPU verification job.

No pilot training is part of this optimization task. Existing scientific-review
fixes below are included in the version to be verified.

## Scientific-code review fixes — 2026-10-18

The user authorized fixing the verified findings in
[BDOT_PHASE_A_CODE_REVIEW_20261018.md](BDOT_PHASE_A_CODE_REVIEW_20261018.md).
All three main findings and four minor issues are addressed locally:

- Bypass forced-choice routers, leaving their gradients absent so AdamW cannot
  decay weights or apply momentum. Mask forced rows out of the z-loss numerator;
  retain the global all-token-step denominator.
- Periodic validation logs target and current-curriculum policies, without
  exploration. `eval_spike_loss` uses curriculum loss; final evaluation uses
  target-window hard routing.
- Separate optimization-objective labels from token-weighted LM aggregates,
  including checkpointed cumulative sums and logging cursors. Add interval
  throughput and peak GPU memory metrics.
- Align standalone/study boundary-batch routing and RNG; retain the GPU guard
  after an uncertain burn reclamation; specify matched D for the runtime gate.

The original 100-update CPU reproduction now stays at **0% off-default** with
z-loss enabled (previously **52.9419%**). Local regression and distributed test
evidence is recorded in [the fix receipt](results/bdot_phase_a_fixes_20261018.json).
The spec and implementation guide document these explicit policy amendments.
No remote job, process stop, deployment or research training was submitted.
DOT's commands.sh remains #0.

Before a pilot, revalidate the changed training paths on B200/FA4. The earlier
GPU evidence below predates these fixes. Strict resume remains enabled: earlier
code's checkpoints are not silently migrated to this changed objective/state
schema. Decode/prefill performance measurement remains separate unfinished work.

## Route-study implementation — historical, before the fixes above

The user requested the Phase-A route-study exporter, then asked to check
[BDOT_PHASE_A_CODE_REVIEW_20261018.md](BDOT_PHASE_A_CODE_REVIEW_20261018.md).
The exporter now extends the existing packed evaluator: one model/data load,
paired as-trained/fixed/random full-validation losses, 4,096-sequence route
statistics, a 256-sequence NPZ artifact and identity-checked sample comparisons.
It supports unpadded distributed evaluation. See
[usage](BDOT_IMPLEMENTATION.md#route-study). Local validation passed: **32 B-DOT
tests and 8 native evaluation tests**, including two-rank CLI coverage, an empty
rank and FA4 adapter wiring against a CPU reference kernel. See
[evidence](results/bdot_route_study_review_20261018.json). No B200 commands,
deployment or research training were submitted.

The review initially reproduced finding 1: home-only
curriculum A plus z-loss gives 46.6309% off-default evaluation routes at update
50 and 52.9419% at 100; the zero-z-loss control stays at 0%. The review follow-up
records the optimizer/normalization details needed for a fix. The evaluation
policy ambiguity and objective-vs-LM training-log labeling are also confirmed.
The exporter work did not change training; the later fixes above address these
findings.

## GPU FA4 resume verification — passed

The user explicitly requested the remaining B200 resume check. Launch `82f6428`
on q359 completed the real train.py/Trainer FA4 test on all eight GPUs. The
uninterrupted 200-step run and fresh-process restarts (100→200 and 20→40) match
exactly on all ranks: data, RNG, both routing branches, loss, model, AdamW,
scheduler and routing state. Maximum loss difference is 0.0. This used the
existing FA4 defaults and a 12-layer reduced-width synthetic fixture; no model
or attention-kernel change was needed. See [resume report](BDOT_FA4_RESUME_20261018.md).

Accelerate was copied and verified. All eight GPUs were free at 19:48:20 UTC
before the test. Supervisor and automatic burn handoff finished successfully at
19:53:15 UTC. Final read-only inspection `7c49881` at 19:56:35 UTC verified
eight active burns, advancing collectives and released ownership. The final
source/results publication uses #0. No research training is queued.

Remote root: `/mnt/local/_outputs/deep-llms_th2/q359-dot-resume-20261018-a01/`.
Local evidence: `temp/bdot_b200_resume_20261018/`.

## B200 verification complete — 2026-10-18

The user authorized publishing DOT to `deep-llms/th2` main and running GPU
verification on q359. DOT is published from `/disk/thuat/DOT-th2-inventory`;
local DOT remains the source directory and its commands.sh is inactive.
Launch commit `c3ce8d1` completed the synthetic suite: **16 required checks pass**.
This includes full-size native equivalence, real FA4 forward/backward,
checkpoint/isolation checks, two-rank NCCL accumulation and eight-GPU Trainer
benchmarks. One optional natural near-tie route comparison changes discrete
paths across backends; separate same-route checks pass. Keep the backend fixed.

Hard-routing throughput is 89.2% of matched dense D with SDPA and 85.9% with
FA4, exceeding the 60% runtime gate in this setup. Relative to native Qwen these
are 78.1% and 75.4%. These are short synthetic measurements, not pilot results.
See [B200 report](BDOT_B200_VERIFICATION_20261018.md) and
[machine-readable summary](results/bdot_b200_20261018.json).

The selected existing eval_fa4 environment matches the training pins. Accelerate's
resource config was copied bytewise and verified as eight-process BF16. Exact
known burn workers were reclaimed and all eight GPUs verified free before tests.
Supervisor restoration finished successfully at 19:27:32 UTC. Final read-only
inspection `d0a24cc` at 19:30:53 UTC confirms all eight communicating burns remain
live, collectives advance and ownership is released. Training losses are finite
and benchmark source hashes match. A parser-only issue in the preceding check
was fixed; no GPU tests were restarted. The final code/results push uses #0.

Remote artifacts:
`/mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03/`.
Local receipts: `temp/bdot_b200_verification_20261018/`.
No research pilot, corpus download, package reinstall or driver change occurred.
The existing environment's older lm-eval version was not used or validated.

## Previous implementation and setup evidence

Updated 2026-10-18. The Phase A foundation is implemented and CPU-validated;
local A100 numerical, NCCL and SDPA throughput checks have also been run.
No research training or dataset download has been launched. The inventory and cleanup history is recorded below; subsequent B200
verification is summarized above.

Design: [BDOT_v2_spec.md](BDOT_v2_spec.md), **v1.3**, and
[review response](BDOT_v2_review_response.md). The current architecture is a
12-layer Qwen3 model trained from scratch; older architecture descriptions are
superseded. Later phases remain conditional on Gate 0a.

## Implemented

- Native Qwen3 dense reference and routed middle with step-owned K/V, FiLM,
  fixed/learned/random choices, reuse on/off, exact feasibility tables,
  curricula A/B, balancing and correctly normalized router loss.
- Reuses train.py, HF Trainer/Accelerate and strict checkpoint/resume. Adds
  only the model-specific Trainer behavior; dynamic unused experts use DDP.
- Frozen validation document IDs and deterministic document stream, EOS,
  isolated attention and reset positions. Continuous packing drops only the
  stream's final remainder; the original baseline packing remains available.
- Configurations D, P1-A/B, R, N-A/B; sequential queue preparation, packed PPL,
  route interventions, native lm-eval wrapper and a full-update benchmark tool.
- Recreated the outdated local train_env with the user's authorization.

See [implementation and commands](BDOT_IMPLEMENTATION.md) for scope and tests.
Initial local results are in docs/results/bdot_phase_a_local_20261018.json.
The follow-up review fixed nonpersistent buffer restoration during checkpoint
loading. Expanded checks (25 tests plus checkpointed hard/soft CPU DDP probes)
are in docs/results/bdot_phase_a_review_20261018.json. See
[GPU verification](BDOT_GPU_VERIFICATION_20261018.md) for subsequent CUDA evidence,
runtime limitations and a borderline default-SDPA AdamW check. Pinned FA4 backward
explicitly rejects this dev machine's A100 (SM80); forward-only results are separate.

## Remaining before a pilot

- Revalidate pinned-runtime FA4/DDP/resume after the scientific-review fixes.
  Earlier GPU tests passed for the preceding code version.
- Measure decode/prefill speed separately; training throughput is not inference
  throughput.
- Run the implemented route exporter on actual pilot checkpoints for Gate 0a.
- Pin the actual dataset/tokenizer and record the validation split/manifest.
  Existing English cx_sampled_old on B200 can be reused; preserve its old
  training/validation split if selected. No new download is inherently needed.
  No scientific performance claim follows from the tiny local test fixtures.

## Assigned execution target and cleanup preparation — 2026-10-18

The user explicitly assigned the previous `deep-llms/th2` GitHub repository,
`main` branch, q359 B200 and its existing Dropbox folder to DOT. The user asked
for a separate minimal checkout containing only commands.sh and a first
read-only inspection of the B200 project directory.

Execution checkout: `/disk/thuat/DOT-th2-inventory`. Commit `4831f9a` descends
normally from previous head `624d63a`; its tree contains only commands.sh.
No force push or change to the local DOT/old-project source trees was used.
Job: `th2-q359-dot-project-inventory-20261018-a01`, `#1 +30+a`.
It inventories `/mnt/local/deep-llms_th2` without deletion, file-content dumps,
process stops, GPU work or environment changes. Inventory completed successfully
at 18:32:47 UTC on `thiennh-p6-q359-worker-0`; the returned script SHA256 matches
the submitted commit. There are 28 top-level entries: old source directories
and files remain, alongside controller logs/status files. Synchronization did
not remove old files deleted from GitHub. No B200 files have been deleted.
Commit `5e395c7` restores commands.sh to #0; GitHub still contains only that file.
Local evidence: `temp/dot_th2_inventory_20261018/` and
[verified receipt](results/dot_th2_inventory_20261018.json).
The user subsequently authorized cleaning this remote project directory.
Cleanup job `th2-q359-dot-project-cleanup-20261018-a01`, commit `1676c12`,
removes the inventoried old source entries and all resources except the known
live burn script. It preserves commands.sh, runner logs/status files and
resources/llm_pretrain_burn.py. It refuses unexpected paths, mounts, non-burn
GPU work or live project scripts and verifies GPU identities before/after.
Local fixture checks passed for scoped deletion, external symlink preservation,
runtime preservation, refusal paths and idempotence. Cleanup completed at
18:53:02 UTC: all 41 listed entries removed; all eight GPU burn workers and their
launcher retain the same PID/start-time identities. The only remaining entries
are commands.sh, _run_log_, _dl_active.txt, _install_active.txt,
_previous_run_status.log and resources/llm_pretrain_burn.py. Separate data,
model, output and conda directories were untouched. Execution commit `162455c`
replaces commands.sh with #0 and a comment, removing the destructive body.
Evidence: [cleanup receipt](results/dot_th2_cleanup_20261018.json) and
`temp/dot_th2_inventory_20261018/cleanup_a01.log`.

DOT's own commands.sh remains #0. Research training has not been authorized
or launched by this cleanup/inventory request.
