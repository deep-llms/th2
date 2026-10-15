## P7 tuning implemented locally — 15 October 2026

User requested all discussed variants and safe throughput optimization, not a
new launch. Implemented14 new named architectures:10 targetgrad (2/3-block,
weighted, SmoothL1, K-only, V-only, nonlinear transforms, sparse,2/8 query heads)
and4 LM variants (sparse/K-only/V-only noaux and per-query gated V mixing).
Prepared20case sequential manifest including4 lambda sweeps and A/dense-noaux
seed2042 replication. All fresh2500steps,125scheduler warmup,250aux ramp,
8GPUs pertrain, saveall every250; manifest gates numerics/allsmokes before full
training. No B200 push, stop, launch, environment/config/cache changes.
commands.sh remains#0. Existing active queue unchanged; prior status below is
historical, not a fresh machine observation.

Value-only collapses duplicate keys into one attention call; gated values use
one native-length call with expanded GQA entries and independent query gates.
No measured B200 throughput claim. Saved metadata/report/reload/receipts support
all variants. Parent old scientific definitions and train.py loop preserved.
New variants keep fixed-four-block custom kernels disabled; parent lambda cases
retain its previously validated optimizations. GPU numerical/DDP gates pending.

36unique targeted test methods passed across recorded runs (some repeated).
All14 newarms passed actual Trainer exactresume/reload/report and2rankGloo.
FP32/BF16 referencebackend gradients, independent equations, isolation,
checkpoint replay, attention-call counts, gate logging, metadata rejection,
newrecipe and receipt checks passed. Existing P7 regression suite passed.
Evidence:docs/results/p7_tuning_local_checks_20261015.json; definitions/usage:
docs/P7_TUNING_VARIANTS_20261015.md. All20cases exceed the earlier12hour estimate;
select launch scope separately. Unrelated user documentation remains untouched.

## Exploration status — 15 October 2026, 12:18 UTC / 20:18 SGT

Read-only requesta2b381c verified q359-p7-exploration8-20261015-a03 healthy.
A and targetgrad completed/validated7500steps, all30checkpoints each retained.
Full validation: A loss3.23501613/PPL25.4068; targetgrad loss3.24556588/PPL25.6762.
Targetgrad is+0.01055nats worse in LMloss; downstream not yet evaluated.
Training runtime A8134.672s, targetgrad8413.6666s (~3.43% longer).
P7-lm running6914/7500 at~1.17s/update,~12–15min remaining; saved through6750.
All8B20098–100%utilization,~65.9–67.0GiB used. Supervisor/queue running; no failure.
Five2500-step variants remain (sparse,keyonly,valueonly,noaux,nexttoken), then
downstream. Wholequeue remaining roughly4–6hours provisionally.

Receipt temp/remote_logs/p7-exploration-status-20261015-1218.log SHA256
053b94c0d33b0abc15dca293bf126458cb6964594a59d7f8a160eb615b6dfb6d.
Targetgrad result docs/results/p7_exploration_targetgrad_7500_20261015.json.
No workload/config changes; commands.sh returned to#0. Unrelated docs preserved.

## Exploration status — 15 October 2026, 09:29 UTC / 17:29 SGT

Read-only requestaca0c97 verified activeq359-p7-exploration8-20261015-a03.
A completed/validated7500steps; full4882-row validationLMloss3.23501613,
PPL25.4068, trainruntime8134.672s (2h15m35s). All30 checkpoints250..7500 retained.
Targetgrad is5860/7500, ~1.092s/update, ~30–35min to completion. Latest saved5750;
all23 checkpoints250..5750 retained. All8B20096–99%utilization,~64.73GiB used each.
Supervisor/queue running, no failed stage. Six remaining arms:lm7500 then
sparse/keyonly/valueonly/noaux/nexttoken2500each; downstream follows. Do not
compare targetgrad training loss to A's full held-out result. Remaining queue
provisionally7–9hours; other-arm timings/evaluation still pending.

Receipt:temp/remote_logs/p7-exploration-status-20261015-0928.log SHA256
2cc57bf2a80085e6ff59c7129878d6407300ade28a81a4872f985d92cec5789f.
A compact result:docs/results/p7_exploration_A_7500_20261015.json.
Only read-only inspection performed; no workload/config/data/env changes.
commands.sh returned to#0. Unrelated user docs remain unstaged.

## P7 eight-arm queue running; production verified — 15 October 2026, 05:28 UTC

Active root:/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03.
Launch4358d84. All7 CUDA proxy gates and all8 three-update/eight-rank smokes plus
validators passed; original-control recipe/data gate passed (train fingerprint
6e4708f1e818fb44). Production A began05:23UTC (13:23SGT). Verified05:28UTC:
A238/7500, finite decreasing LM loss (step230 loss6.079, stepLM6.033),1.058s/update,
8train.py workers943677–943684 under launcher943666, all8B20099%utilization and
about61.94GiB used each. No production failure. Retention0(save all)/250,
warmup375 and7500-step schedule,seed/data42,micro16/GAS4 confirmed in config.

Queue:A7500 → targetgrad7500 → lm7500 → sparse2500 → keyonly2500 → valueonly2500
→ noaux2500 → nexttoken2500 → nine-benchmark downstream evaluation. All fresh,
12layers,FA4; all8GPUs per training job. Expected wholequeue roughly11–13hours,
new-arm full timings pending. Existing supervisor automatically restores and
verifies burns on success/failure. commands.sh returned to#0, which does not
stop the independent active tmux/supervisor. Do not resubmit the launch.

Successful launch copied/verified Accelerate8GPU/bf16 with sourceSHA
923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5.
Explicitly verified a02 failed-gate burns restored/live before retry, then
reclaimed exact workers913630–913637. All8 GPUs free05:12:35UTC before gates.
Failures a01(no.git shell check) and a02(wrapped numerical-row format) preserved;
neither reached a Trainer step. Corrected loader validates before reclamation;
5local queue tests passed16.429s. No training data/cache/env/driver changes.

CUDA value-only merge vs original duplicate entries: gradientrelL2 .004978SDPA,
.004840FA4; predictor .003950/.003851; both passed existing limits. Full new-arm
FA4vsSDPA gradientrelL2 .00567–.00602. Results saved
 docs/results/p7_exploration_numerics_20261015.json.
Evidence:temp/remote_logs/p7-exploration-health-20261015-a06.log SHA256
c8ab942c80955e1abae7195fb81ed1fc506b709b994f7ab4c680efd43d51e970;
temp/remote_logs/p7-exploration-production-20261015-a01.log SHA256
63b8f1058567afd1b5888389ddb4f1db391ff8e6e3fb696fab69fafedb74affb.
Unrelated user docs preserved unstaged. Next action is read-only progress check.

## P7 numerical-input correction and fresh retry — 15 October 2026 (SGT)

Launch49875aa reached preflight: pinned env and Accelerate MULTI_GPU/8/bf16
verified, exact burn workers911718–911725 reclaimed, free-GPU receipt saved.
Only numerics ran and failed with KeyError input_ids: teacher probe rows are
{index,data} records, not bare packed rows. No Trainer step ran. Supervisor
restored burns; retry explicitly requires completed failed-gate handoff, live
verified burn identities/collective progress and released guard before launch.

Fixed numerical loader to accept both formats, select a genuine multi-document
row deterministically, and validate all packed fields BEFORE GPU reclamation.
Tokens/labels/segmentation and scientific recipes unchanged. Added local format
regression. Fresh rootq359-p7-exploration8-20261015-a03; preserve a02 failure.
Evidence temp/remote_logs/p7-exploration-health-20261015-a01.log,
SHA2568cb90d9aad99c9921d102ca71eb5a6b3232a4f4fb2e8d3f1f73545b44da65644.

## P7 exploration submission retry — 15 October 2026, 05:04 UTC

First launch19812e5 exited before nvidia-smi/tmux/preflight or any GPU signals:
runner source copy has no.git, so the diagnostic git rev-parse HEAD failed.
No training/output/cache was created by that shell. Removed this source-copy
assumption; retry uses fresh q359-p7-exploration8-20261015-a02 root/session.
Implementation6ce50b3 and scientific config unchanged. Receipt:
temp/remote_logs/p7-exploration-launch-20261015-a01.log,
SHA2565b57c35e3db5f613b763909bc7bf1232ee2cb7ad6f91373477c1d3d2abeb29bb.
This is a delivered shell error, not a controller failure. Remote retry pending.

## Eight-arm P7 exploration authorized — 15 October 2026 (SGT)

User now explicitly requested launch of the prepared eight runs. Supersedes
preparation-only restrictions below. Fresh root intended:
/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a01.
Source implementation6ce50b3. Three fresh7500-step controls (A,targetgrad,lm),
five fresh2500-step variants (sparse,keyonly,valueonly,noaux,nexttoken), then
nine-benchmark evaluation. Seed42/proxy43,12layers,FA4,8GPUs per training job,
all checkpoints retained every250. No changes to existing outputs or caches.

38-stage manifest gates numerics and all distributed smokes before full runs.
Predecessor:q359-p7-target-integrated-20261015-a01, must be complete with verified
burn handoff. Preflight checks env/data/eval/disk, copies and verifies Accelerate,
inspects known GPU ownership. Existing supervisor revalidates exact burn worker
identities before signals, waits/rechecks free GPUs, restores/verifies burns on
queue success/failure. Launch log lives beside root so prepare can create a
fresh root itself. GitHub fetched, no divergence; previous status has no new
controller error. Submission/remote confirmation pending at this entry.

## P7 exploration review and value-only optimization — 15 October 2026 (SGT)

Preparation only remains in force; commands.sh stays #0. No push, launch or
B200 process/environment changes. Reviewed short-arm priorities: noaux, sparse,
key-only, value-only, next-token (most speculative). Eight-run scope/order and
fresh 7500/2500-step schedules unchanged. Reasoning in
docs/P7_EXPLORATION_QUEUE_20261015.md.

Value-only now merges duplicate-key values as (native+proxy)/2 and uses one
native-size SDPA/FA4 attention call across all heads: 12 rather than 16 calls
for the 12-layer model. Same real-valued function/gradient, not bitwise BF16.
Saved memory_entries version2 records implementation. Updated MAC estimates,
receipt validator and prepared CUDA gate; gate compares against original
duplicate entries on both backends before production. CUDA timing/acceptance
pending; do not report a measured whole-training speedup.

31 targeted CPU tests pass, including independent attention/gradient equations,
BF16, full-model equivalence, exact Trainer resume and two-rank runs for all
five arms, queue/receipt checks, consumer/combined/optimized-targetgrad regression.
Initial regression invocation had a misspelled test module; corrected module
passes. Logs temp/p7-exploration-review-*-20261015.log. No environment changes.

## P7 exploration prepared, NOT launched — 15 October 2026 (SGT)

User requested overnight exploration, then explicitly selected preparation only.
commands.sh remains #0. Do not launch or reclaim GPUs without new authorization.
Five new P7-simple-lm variants: sparse sites, proxy keys only, proxy values only,
no auxiliary loss, and last-site next-token CE through a detached current
final norm/LM head (packed stride8, same-document labels, global selected-token
normalization). Main LM trains predictors in every new arm. Target/shallow
auxiliary routes remain detached. Existing train.py entry point unchanged.

Schedule finding: original12layer runs completed full2500step cosine schedules
with125warmup and10%minimumLR. Proposed ordinary resume to10k would not preserve
that schedule. Prepared clean alternative: fresh A/optimizedtargetgrad/lm at7500
steps,375warmup; five new variants2500steps,125warmup. Seed/data42/proxy43,
8GPUs micro16 GAS4 seq2048 EOS/resetpositions/FA4. Save all checkpoints every250.
Provisional budget11–13hours including overhead; new CUDA timings pending.

38stage queue: numerical gate,8smokes+validators,preproductionrecipe/data gate,
8fullruns+validators,matchedsummary,9benchmark numerical/smoke/full evaluation.
Existing verified-burn supervisor and copy/verify Accelerate preserved. No
remote submission, B200 inspection/work, environment/driver change or deletion.
New code is PREPARED, not GPU-validated: mandatory CUDA/DDP gates remain queued.

Local evidence: six new testgroups138.347s incl actual Trainer exactresume and
2rankGloo runs for all5arms; additionalBF16 group1.687s. Broader32tests:28passed,
4blocked by missing local lm_eval (no installs). Existing consumer/combined
parents10tests121.486s passed. Final4queue checks16.041s passed. Shellsyntax and
diffcheck passed. Previewtemp/p7-exploration-prepared-final-20261015/jobs.json.
Details/definitions/limitations:docs/P7_EXPLORATION_QUEUE_20261015.md.

## P7 custom target-kernel optimization complete — 15 October 2026, 02:23 UTC

Implemented opt-in proxy_fused_targets (Triton four-block target sum,
normalization/clipping, moment partials and first-order backward). Use with the
existing compile_auxiliary/lse_attention/compile_entries optimizations. Default
false; targetgrad/A only; CUDA torch2.14. Checkpoint Parameters and optimizer
groups unchanged. Flag saved; resume toggles require explicit performance-change
allowance. No architecture, loss weights, gradient routes, data order, schedule,
environment or driver changes. Head-batched FA4 rejected: about2.06% slower.

Five50step eight-GPU trials: controls1.098117/1.096338s, target1.087154s:
+0.9266% throughput, 10.07ms/update saved, matched-trial peak0.375GiB lower.
Control drift0.162%. Normal-CLI final pair: A1.049351s, P71.086724s; remaining
P7 overhead3.5615% (previous4.43%). Final peaks58.38/62.21GiB. All-rank endpoint
data hashes agree; compilation counters stable. These are short steady-state
measurements, not a full2500step result or downstream equivalence claim.

Initial kernel failed the exact-clamp-endpoint oracle before training; fixed
to pinned torch2.14's strict derivative, retained failed artifacts. Added explicit
NaN propagation.13CUDA primitive cases pass on every rank. Target-only kernel
matched previous optimized loss/hidden/logits/all gradients exactly in both
shapes; running-buffer relativeL2<7.4e-9. Final combined optimizations vs ORIGINAL
checkpoint: loss abs.000171661, hidden relL2.006238, logits.002448,
gradients.0126126, unchanged acceptance thresholds. All8cases passed.
Nine CPU tests passed35.978s incl real Trainer resume; updated13-case interpreter
regression passed12.988s. Ordinary training never imports experimental head
batching. Production integration:d1b3336.

Final root:q359-p7-target-integrated-20261015-a01. All4jobs, summary and supervisor
succeeded; complete marker exists. Burns restored/verified02:23:40UTC. Read-only
verification02:23:49UTC:8correctworker identities, approvedparents, advancing
collectives, logage5.75s, GPUguardreleased. Final receipt:
temp/remote_logs/p7-target-completion-final-20261015-a01.log
SHA256d62c085ae54486c52ce360a86fe3e9cddf227e3bf3f96eac6bbdba0e46065e60.
Raw gate/summary hashes checked against recorded job artifacts. Results:
docs/results/p7_kernel_numerical_20261015.json,
docs/results/p7_kernel_trial_20261015.json,
docs/results/p7_target_original_numerical_20261015.json,
docs/results/p7_target_integrated_20261015.json.
Report:docs/P7_TARGETGRAD_SPEED_20261015.md. No long training queued.
commands.sh returned to#0. Unrelated user docs left untouched.

## P7 custom target-kernel integration — 15 October 2026

CUDA gates passed after fixing torch2.14 clamp endpoint semantics. Target-only
kernel exactly matches tested loss/hidden/logits/all parameter gradients;
updated running-buffer relativeL2 below7.4e-9. All12primitive cases pass on8ranks.
Nine CPU tests passed35.978s, including actual Trainer resume and interpreter.
Added a13th nonfinite-value case and explicit NaN propagation; interpreter
regression passed12.988s. CUDA integration gate will rerun the updated kernel.

Five timing runs completed: controls1.098117/1.096338s, target_kernel1.087154s,
head_batch1.120300s, combined1.109421s. Target gain0.927%, better than both
controls by0.84–1.01%; control drift0.162%. Head batching rejected as slower.
Keep target preparation opt-in with proxy_fused_targets; defaultfalse, limited
to targetgrad/A and torch2.14 runtime. Unchanged parameters/optimizer state;
flag persisted; toggling resume requires explicit performance-change allowance.
Final normal-CLI A/P7 pair plus direct-original-checkpoint gate planned at
q359-p7-target-integrated-20261015-a01, following verified trial burn handoff.
For this small gain, launch requires >0.5% gain against EACH control and <0.5%
control drift; actual timing values remain reported without rounding up.
Copy/verify Accelerate, reclaim only known burns, verify GPUs free, automatically
restore/verify burns afterward. No long scientific training.

## P7 custom target-kernel boundary fix — 15 October 2026

First CUDA primitive gate rejected the experimental kernel before any model
training/timing. torch 2.14.x clamp has zero derivative at either exact endpoint;
the first implementation incorrectly used inclusive comparisons. Reproduced
locally with torch2.14.0 and remotely with pinned2.14.1. Changed kernel mask to
strict comparisons and added positive/negative endpoints, zero and outside
clipping tests. Tolerances unchanged. Original failed root and logs preserved.
Retry fresh root:q359-p7-kernel-trial-20261015-a02. Additional launch guard
requires the failed primitive-only run's terminal state, restored verified burns
and released GPU guard before the ordinary known-burn reclaim/free-GPU protocol.

## P7 custom-kernel optimization study — 15 October 2026

User requested further throughput optimization, including a custom kernel if
useful. Test head-batched FA4 (independent native/proxy heads in one call) and a
Triton four-block target-preparation kernel, separately and together. Preserve
joint-softmax derivatives, live target gradients, detached predictor input and
injection, objective, statistics, data order and checkpoint parameters.
Compare against the current validated compiled-entries implementation.
CPU route/state tests, CUDA primitive oracle and eight trained-checkpoint gates
precede accepted-candidate 50-step eight-GPU Trainer timings. Experimental only
until correctness and a measured gain are established. No long training or
changes to environments/drivers. Fresh root:q359-p7-kernel-trial-20261015-a01;
predecessor:q359-p7-entries-integrated-20261015-a01. Copy/verify Accelerate,
inspect/reclaim only verified authorized burns, verify GPUs free, then use the
existing supervisor for automatic restoration/verification on success/failure.
Remote submission pending.

## P7 projection optimization complete — 15 October 2026, 01:30 UTC

Follow-up measured another1.60% throughput gain from compiling existing proxy
normalization/K/V/RoPE entries. Opt-in --proxy_compile_entries true, alongside
--proxy_compile_auxiliary true --proxy_lse_attention true. All defaultfalse;
only targetgrad/A accepted. No changed parameter names/shapes, optimizer groups,
architecture, aux objective/routing, data/order, environments or drivers.
Performance flag persisted and toggling on resume requires explicit allowance.
Target-loss fusion gained0.42% and packedKV1.50%; neither promoted to production.

All8CPU tests passed18.618s: objective/gradient routes, optimizer/state, exact
prototype/production outputs+gradients, actual Trainer save/resume and legacy
compatibility. Per-test compiler caches isolated for retained-graph probes.
All8CUDA cases passed against optimized control AND directly against original
eager checkpoint; fullseq2048/micro16 accumulated grad relativeL2 1.2629%,
hidden .6238%, logits .2452%, loss absolute .000181675. Thresholds unchanged.

Final integrated normalCLI50step pair: A1.049507s/update, P71.095953s/update,
remaining overhead4.43% (previous5.90%). Peak59.67/61.28GiB. All-rank endpoint
data hashes agree and compiler counters stable. Full report/limits/artifacts:
docs/P7_TARGETGRAD_SPEED_20261015.md. Only32evalrows/50steps; no full2500step
or downstream equivalence claim. Approx6.7% cumulative throughput vs original
P7 from earlier round; additional1.60% measured with current bracketing controls.

Root:q359-p7-entries-integrated-20261015-a01. All4stages/supervisor succeeded,
complete marker verified; burns restored/verified01:29:58UTC. Read-only collector
01:30:13UTC: all8worker identities/approved parents match, collectives advance,
logage1.05s, guardreleased. Receipt:temp/remote_logs/p7-entries-completion-20261015-a01.log
SHA2567209fb90a44c8f49dd6fdcc00fb333207920c20fe6026262f9aa4d812a5581ff.
A timed #1 snapshot was static; #2 +a fetched its completed log. Do not infer
continued training from an unrefreshed snapshot. No retry/cleanup was needed.
Production integration46adec6. commands.sh returned to#0. Nothing queued.

## P7 projection compilation integration — 15 October 2026

Fusion trial has finished four of five50step jobs. Preliminary medians:
optimized reference1.115236s, fused_aux1.109494s, entries1.096631s,
packedKV1.097682s; final bracketing control pending. Prefer entries: compile
original existing method without changing K/V GEMM structure or target path.
Opt-in proxy_compile_entries implemented, defaultfalse, restricted to targetgrad/A.
Serialized recipe, explicit performance-change-on-resume, unchanged state names.
Eight CPU tests passed18.618s including actual Trainer save/resume and exact
prototype/integration matching. Test compiler caches isolated between separate
single-backward and retained-graph probes; no runtime/compiler settings changed.

Final verification root:q359-p7-entries-integrated-20261015-a01. Launcher waits
for successful fusion trial and restored/verified burns/guardrelease, requires
entries measured gain>1%, copies/verifies Accelerate, reclaims only identified
burns, verifies all8GPUsfree. First compare accumulated optimizations directly
to ORIGINAL eager trained checkpoint (unchanged thresholds), requiring entries
acceptance. Then normal CLI50step A/P7 pair, all8GPUs, no prototype patch for
training. Auto restore/verify burns afterwards. Submission pending.

## P7 further fusion: numerical gates passed — 15 October 2026, 01:09 UTC

Launch36fafe3; read-onlydf275fb confirms all8CUDA cases pass for fused_aux,
entries and packed-KV candidates against the existing BOTH optimization.
Fullseq2048 fused_aux: exact LM forward/buffers, grad relativeL2 .0018493.
Compiled entries/packedKV: loss absolute delta .000003338, hidden relativeL2
.0062045, grad relativeL2 .0126110. Thresholds unchanged. Source checkpoint
hash unchanged. Eight CPU tests passed15.986s. Accelconfig copied/verified
SHA923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5.
Verified known-burn reclaim01:07:15UTC, all8GPUsfree01:07:45UTC.
50step candidate timings running; no speed conclusion or production adoption yet.
Root:q359-p7-fusion-trial-20261015-a01. Auto burn restoration configured.
Numerical report:docs/results/p7_fusion_numerical_20261015.json.

## P7 further fusion study — 15 October 2026

User accepted testing further method-preserving optimizations. Experimental
candidates: fuse target normalization/clipping with cosine (live target grads),
compile proxy entries, and compile entries with a packed K/V GEMM retaining the
original separate Parameters. Compare against the previously validated BOTH
optimization, not original eager. No production defaults or scientific runs change.
CPU numerical/gradient/state checks first; then eight trained-checkpoint CUDA
cases (short/checkpointed and full2048/micro16), followed by accepted-candidate
50step eight-GPU Trainer timings with bracketing controls/data-order hashes.
Fresh root:q359-p7-fusion-trial-20261015-a01. Require completed integrated-study
handoff; copy/verify Accelerate; inspect/reclaim only known burns; verify free
GPUs. Existing supervisor restores/verifies burns on completion/failure.
Remote submission pending. Preserve existing checkpoints/environments.

## P7 throughput optimization complete — 15 October 2026, 00:19 UTC

User requested optimization of P7-simple-targetgrad, lambda0.1. Implemented
opt-in proxy_compile_auxiliary (Inductor cosine fusion) and proxy_lse_attention
(live-LSE joint softmax without duplicate queries). Defaults remainfalse;
validated for this arm, harmless for A, other arms rejected until validated.
Flags recorded in recipes; toggling on resume requires explicit
allow_performance_change_on_resume. State names/shapes, optimizer grouping,
loss/gradient routes, schedules, document isolation and data order unchanged.
BF16 rounding differs: numerical equivalence, not bitwise future weights.

Eight-GPU12layer/2048/micro16/GAS4 timing, steps20–49 of disposable50step runs:
oldP7 controls1.169309/1.169598s; cosine1.137384s, LSE1.146433s,
both1.112883s =5.08% throughput gain (4.84% less update time).
Normal train.py integration repeated A1.049506s / optimizedP71.111457s:
remaining overhead5.90%. Same per-rank endpoint data hashes; stable compiler
counters. Original train fingerprint6e4708f1e818fb44, short-test eval32rows.
Six CPU tests, combined-flags actual-entry resume and legacy FA4 baseline
resume passed. Eight trained-checkpoint CUDA gates passed at short/checkpointed
and fullseq2048/micro16 shapes; full-shape gradient relativeL2 .370% cosine,
~1.26% LSE/both within pre-existing3% tolerance; cosine LM forward/buffers exact.
Only disposable test outputs written; original research weights/environments
untouched. Full report/use flags:docs/P7_TARGETGRAD_SPEED_20261015.md.

Integration root:q359-p7-speed-integrated-20261015-a01. All3jobs/supervisor
successful; complete marker verified. Burns restored/verified00:17:24UTC.
Read-onlya9ec9e5 at00:19:54UTC verifies all8live approved worker identities,
advancing collective progress, logage7.88s, guardreleased. Final receipt:
temp/remote_logs/p7-speed-final-burn-20261015-a01.log; SHA256
6281217fe27e5412e92b3060708560973f2347c3c6acfbfebb41825c4ebb6483.
Code integrationa1a40f0. commands.sh returned to#0; nothing queued.

Historical profile postprocessing failed only because it expected complete
instead of intentional stopped. Both training jobs succeeded; all recorded
artifact hashes reverified, corrected summary saved separately. Original
failure history preserved. No rerun/overwrite of those completed profiles.

## P7 optimized integration test — 15 October 2026

All eight trained-checkpoint CUDA gates passed. Preliminary medians: old P7
1.1693s, cosine1.1374s, LSE1.1464s, both1.1129s; final eager control/summary
still finishing. Production CLI now has opt-in proxy_compile_auxiliary and
proxy_lse_attention, defaults false. CPU exact prototype/integration checks,
actual train.py save/resume, legacy baseline resume passed; new flags are
serialized and require explicit performance-change allowance on resume.
No model state, optimizer, scheduler, data or environment changes.

Queue one final50step A/P7 pair via actual integrated flags, no prototype
patch. Freshroot:q359-p7-speed-integrated-20261015-a01. Launcher MUST wait
for successful q359-p7-speed-trial-20261015-a01 and verified burns/guard
release, then require measured combined gain>1%, copy/verify Accelerate,
inspect/reclaim known burns, verify8GPUsfree. Supervisor restores/verifies
burns after exit. Existing checkpoints untouched. Submission pending.

## P7 throughput candidates — 15 October 2026

Profile A/P7 jobs both succeeded (50steps), but summary incorrectly required
complete instead of intentionally stopped. Fixed; preserve failed summary
history. Burns verified/live ownership at07:51:33SGT, guard released.
Trial preflight revalidates every recorded profile artifact hash and recomputes
summary into NEW root, accepts only this exact documented postprocessing failure.
Rank0medians1.0492/1.1702seconds; all-rank aggregation pending preflight.
Three CPU tests pass, including checkpointing and LM/aux routing. Candidate
cosine compile, live-LSE merge, both: full CUDA gates before real Trainer timing,
50steps each with eager controls before/after. No production/default changes.
New root:q359-p7-speed-trial-20261015-a01. Auto burn restoration verified by
existing supervisor. No long scientific training. Report:
docs/P7_TARGETGRAD_SPEED_20261015.md. Submission pending.

## P7-targetgrad throughput optimization authorized — 15 October 2026

User requested optimization, including custom kernels if useful. First profile
A and P7-simple-targetgrad with existing12layer/8GPU/micro16/GAS4/2048/FA4
recipe in fresh disposable50step runs. Retain production checkpoints/configs.
Audit pinned FA4 b33 LSE backward support before an attention candidate.
Then validate candidate forward/backward, live target gradients, normalization
state and updates; measure complete eight-GPU Trainer throughput. No new long
scientific training or environment changes. Profile root:
q359-p7-speed-profile-20261015-a01. Predecessor teacher probe completed with
verified burns. Copy/verify Accelerate, reclaim only identified known burns,
verify all8GPUs free; existing supervisor restores/verifies burns after exit.
Queue construction and shell syntax passed locally; remote submission pending.

## P7 teacher investigation complete — 15 October 2026, 07:16 SGT

Frozen-weight probe passed for P7-simple, P7-simple-lm, P7-simple-deep and
consumer+cosine, 12 layers/seed42/step2500. All four sites tested separately
on identical 512 validation rows (1,046,840 scored tokens per checkpoint).
Full teacher-response replacement worsens LM loss in all 16 model/site cases,
consistent in both row halves. 10% movement yields no clear improvement;
all negative-delta exploratory intervals include zero. Some LM-trained and
deep-arm sites worsen even at 10%. Opposite-direction controls also show no
established improvement. Decision: do not spend another training sweep simply
strengthening imitation of this teacher. This local frozen-teacher test is not
an oracle/recurrent upper bound and does not rule out other proxy designs.

Protocol, all per-site results and caveats:docs/P7_TEACHER_INVESTIGATION_20261015.md.
Exact summary:docs/results/p7_teacher_probe_20261015.json; SHA256
0acb49fc27dacf0daab550e6c242cedf6e29fb56fcdf1c8fa0363201fa86e269.
Smoke 36.31s/full 57.89s; completed 07:12:21 SGT, no training. All 16 worker reports
verify unchanged state, exact identity, and zero consumer-objective mismatch.
Accelerate copy/env verified, original validation fingerprint 08432871cf987d61,
known burns reclaimed and all eight GPUs verified free before evaluation.
Supervisor restored/verified burns 07:13:32 SGT. Read-only ae122b5 at 07:16:24 SGT:
all eight live workers match ownership/approved parents, advancing collectives,
log age 2.64s, GPU guard released. commands.sh returned to #0; nothing queued.
Root:q359-p7-teacher-probe-20261015-a01. Receipt:
temp/remote_logs/p7-teacher-completion-20261015-a01.log, SHA256
c96a1bd0f8bab4aa54daaae48562c5876af754c5336125cc58128e09fd12b616.

## Consumer-teacher investigation authorized — 15 October 2026

User requested only decision-relevant investigation of the current design.
Run frozen-weight single-site teacher-response interventions for P7-simple,
P7-simple-lm, P7-simple-deep and consumer+cosine, existing seed42/2500 weights.
No new training or downstream sweep.512 fixed validation rows per checkpoint,
alpha +.1/-.1/1 at sites2/4/6/8; two halves per model across eight GPUs.
Protocol and interpretation:docs/P7_TEACHER_INVESTIGATION_20261015.md.
Four local tests passed5.682s, including eight model/backend combinations,
independent targets/teacher/downstream references, causality/isolation,
state preservation and coverage rejection. Used sampling_b200; legacy local
train_env was correctly rejected for transformers4.57.1, no env was modified.
Fresh root:q359-p7-teacher-probe-20261015-a01. Preflight checks completed
consumer queue/burn handoff, hashes and original validation fingerprint;
copy/verify Accelerate, inspect/reclaim known burns, verify all8GPUs free.
CUDA smoke must pass before full probe; supervisor restores/verifies burns.
Remote execution pending. Preserve existing checkpoints and user document edits.

## Consumer screen complete — 15 October 2026, 06:20 SGT

Read-only f7b1558 at06:20:18SGT verifies all21queue stages succeeded, all four
fresh12layer runs reached2500steps, validators passed, downstream numerical/smoke
and full9benchmark evaluations passed (diagnostic=false). Evaluation completed
06:00:27SGT; supervisor exited0 and verified communicating burns06:01:38SGT.
All8current burn workers match recorded ownership and approved parents, collectives
advance, log age1.82s, GPU guard released. No queued follow-up or fine-tuning.

| Arm (seed42) | PPL | MC mean (%) | BLiMP (%) | LAMBADA (%) |
|---|---:|---:|---:|---:|
| A (existing baseline) | 36.2769 | 45.5798 | 74.2866 | 14.9233 |
| Consumer-only | 36.1319 | 44.5497 | 67.1985 | 16.3594 |
| Consumer + cosine | 36.0724 | 45.1166 | 69.3821 | 15.6802 |
| Consumer + targetgrad | 37.4417 | 44.2936 | 74.1239 | 15.4085 |
| Targetgrad lambda0.01 | 36.0871 | 45.2179 | 70.2806 | 16.3206 |

MC mean is the unweighted average of the seven multiple-choice benchmarks.
Consumer-only, consumer+cosine and targetgrad lambda0.01 improve PPL versus same-seed A, but none
beats A on MC mean. Consumer+cosine is best PPL in this batch; its BLiMP score
is4.90percentage points below A. Lower targetgrad weight recovers PPL but does
not retain the earlier lambda0.1MC advantage. These are single-seed results,
not evidence of a broad downstream improvement.
Root:q359-p7-consumer4-20261015-a01. Receipt:
temp/remote_logs/p7-consumer-completion-20261015-a01.log, SHA256
3f06791b09d8aea030ff6683412722e447702a98222d96aae7699047929a3049.
Compact local results:temp/results/p7-consumer4-20261015.json.
commands.sh returned to#0. No new workload launched.

## Consumer queue ETA — 15 October 2026, 03:39 SGT

Read-only c59c929 at03:39:16SGT: consumer-only reached2500 and validator passed
(55m01s including startup/save/evaluation). Consumer+cosine is1586/2500,
~1.28s/update, all8GPUs99%, finite losses/gradients. Consumer+targetgrad and
targetgrad lambda.01 remain queued, followed by nine-benchmark evaluation.
Estimated full completion about06:00–06:15SGT today, allowing saves/evaluation;
this is an estimate, not completion. Automatic burn restoration stays configured.
Consumer-only log reports eval LM loss3.58718 (rounded); full result extraction
and downstream interpretation remain pending.
Evidence:temp/remote_logs/p7-consumer-eta-20261015-a01.log, SHA256
eda3955c9e74933e6cda4f51f170e237ca7f3efa6c12cfb1365a94bf6eeb4898.
Root:q359-p7-consumer4-20261015-a01. commands.sh returned to#0; queue untouched.

## P7 consumer screen running — 15 October 2026, 02:11 SGT

Launch36e5f71; read-only63cd00e at02:11:46SGT verifies all four CUDA numerical
cases, all four eight-GPU3step smokes and their validators passed. Existing
numerical thresholds were unchanged; overall gradient relativeL2 about0.0055.
Production consumer-only started02:08:51SGT and reached97/2500steps,
~1.27s/update, finite losses/gradients; all8GPUs98–99%,68538–69716MiB each.
Next: consumer+cosine, consumer+targetgrad, targetgrad lambda.01, then same9tasks.
Rough full-queue estimate about4hours including evaluation; not a guarantee.

Data/environment/evaluation preflights passed. Accelerate resource copied and
verified, SHA256923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5.
Exact known burn workers770790–770797 reclaimed; all8GPUs verified free at
02:03:25SGT. Supervisor owns guard; automatically restores/verifies communicating
burns after completion/failure. No new environment installation or data cleanup.
Root:/mnt/local/_outputs/deep-llms_th2/q359-p7-consumer4-20261015-a01.
Receipt:temp/remote_logs/p7-consumer-health-20261015-a03.log, SHA256
c416719f19bbf9af13acdf3427594e512980eb0e13fc2442afa199ce8f00fd19.
commands.sh returned to#0; detached queue continues. Do not relaunch.

## Consumer-aware P7 four-run screen authorized — 15 October 2026

User requested further method exploration, consumer-aware experiments and runs.
Implement three P7-simple consumer variants, then run these fresh cases in order:
consumer-only(.1), consumer+cosine(.1), consumer+live-target cosine(.1), and
existing P7-simple-targetgrad with lambda.01. All12layers,2500steps,seed42,
proxy seed43, existing1M-token/2048/FA4/BF16/micro16/GAS4recipe. Same9benchmark
evaluations afterwards. Reuse existing baselines; no new fine-tuning/long run.
Definition and gradient contract:docs/P7_CONSUMER_SCREEN_20261015.md.

Local28regression/integration tests passed195.241s; final11targeted tests
passed3.838s. No train.py, optimizer, scheduler, dataset/packing or kernel changes.
Queue reuses confirmation/training/evaluation workflows; historic scopes frozen.
Remote launch and full-shape acceptance pending. Root:
/mnt/local/_outputs/deep-llms_th2/q359-p7-consumer4-20261015-a01.
Predecessor:q359-p7-confirmation3-20261014-a01, completed and burns verified.
Require predecessor success/burn release, env/data checks, Accelerate copy/env,
known-worker reclamation and all8GPUs free, CUDA numerics and all4smokes before
production. Existing supervisor restores/verifies communicating burns on exit.
No existing outputs or caches removed; preserve unrelated user edits.

## P7 confirmation complete — 15 October 2026, 01:47 SGT

Read-only commit7eac4f0 verifies all17stages succeeded: three fresh12layer
2500-step runs, validators, and full nine-benchmark evaluations (no fine-tuning).
Evaluation finished14October23:43:45SGT; burns restored/verified23:44:56SGT.
At15October01:47:54SGT all8live burn workers match the recorded ownership,
approved parents and advancing collectives; log age5.24s, GPU guard released.

| Arm | Seed | LM loss | PPL | Seven-task MC mean (%) | BLiMP (%) | LAMBADA (%) |
|---|---:|---:|---:|---:|---:|---:|
| A (previous baseline) | 42 | 3.591181 | 36.2769 | 45.5798 | 74.2866 | 14.9233 |
| A | 1042 | 3.590607 | 36.2561 | 45.0288 | 74.3627 | 15.9519 |
| P7-simple-targetgrad | 1042 | 3.614167 | 37.1204 | 45.2418 | 74.9955 | 15.1174 |
| P7-simple-targetgrad-lm | 42 | 3.638429 | 38.0320 | 44.1116 | 74.2642 | 15.1368 |

Compare within seed: targetgrad1042 is+0.213percentage points on MC mean
but+0.023560nats worse on validation loss versus A1042. Earlier seed42
MC advantage was+0.312pp. Two small same-direction results are encouraging,
not a demonstrated statistically reliable downstream win. Combined targetgrad-lm42
is-1.468pp and+0.047248nats versus A42; this test does not support that change.
Root:/mnt/local/_outputs/deep-llms_th2/q359-p7-confirmation3-20261014-a01.
Receipt:temp/remote_logs/p7-confirmation-completion-20261015-a01.log, SHA256
b65e043e9c336e5f423255ab1281cc13adcc4081217f4afabe05d26bda9ce80c.
Compact results:temp/results/p7-confirmation-20261015.json.
Full downstream summary SHA256:
3a91352acf251bce5cadd4fdd3e355c3830a16abde62218db1479d6a2f669f88.
commands.sh returned to#0. No new training or evaluation queued.

## P7 confirmation queue running — 14 October 2026, 21:08 SGT

Launch commit 7fe8d34; read-only receipt 229a7e6 at 13:08:23 UTC verifies
CUDA numerical acceptance and all three eight-GPU smokes plus validators passed.
Fresh A/seed1042 production started at 13:04:21 UTC and reached step181/2500;
all eight GPUs active at 98–100%, about 63,428 MiB each, 1.06s/update.
Next: P7-simple-targetgrad/seed1042, then P7-simple-targetgrad-lm/seed42,
each fresh12layer2500steps, then the same nine downstream benchmarks.
No production result is available yet. Do not relaunch this detached queue.

Data/environment/evaluation preflights passed. Accelerate config copied and
verified at /dev/shm/.cache/huggingface/accelerate/default_config.yaml,
SHA256 923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5.
Known burns reclaimed and all8GPUs verified free before training at13:00:16UTC.
Supervisor owns the GPU guard and restores/verifies communicating burns on exit.
Root: /mnt/local/_outputs/deep-llms_th2/q359-p7-confirmation3-20261014-a01.
Receipt: temp/remote_logs/p7-confirmation-health-20261014-a02.log, SHA256
b4cc8764f73165d05886feebf469dc9df70e7ee00f7778c5b2e4c6f42d795844.
commands.sh returned to #0; this does not stop the detached queue.

## Authorized P7 confirmation queue — 14 October 2026

User accepted three new runs: A/seed1042, P7-simple-targetgrad/seed1042,
and P7-simple-targetgrad-lm/seed42. Proxy seeds1043/43 follow the existing
seed+1 convention. Fresh12layer2500-step training, unchanged full screen recipe,
then same nine downstream benchmarks (evaluation seed42). No fine-tuning.
New combined arm keeps detached predictor inputs, permits LM -> predictor,
and permits cosine auxiliary -> live target/backbone (including graph paths
back to predictor through the target). No architecture/forward/init changes.

scripts/proxy_confirmation_queue.py reuses the overnight queue's numerical
checks,8GPU smokes,train.py and validators; the existing evaluation suite runs
automatically afterward using eval_fa4. Gate on completed remaining27evaluation
and its verified burn handoff; check eval dependencies/data before training,
copy/verify Accelerate, inspect/reclaim only known burns, verify8GPUs free.
Existing supervisor restores/verifies burns after completion/failure.

Local17test acceptance passed; final queue tests cover mixed seeds, matched
PPL comparisons, frozen downstream scope and predecessor/burn gates.
Fresh intended root:/mnt/local/_outputs/deep-llms_th2/q359-p7-confirmation3-20261014-a01.
Predecessor:/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01.
Remote launch and CUDA acceptance pending. Preserve all existing outputs/caches.

## All33 downstream checkpoint evaluations complete — 14 October 2026, 19:52 SGT

Read-only bcaa6be at19:52:02SGT verifies both queues succeeded: six original
shortlist checkpoints plus all27remaining final step2500checkpoints. Numerical,
smoke and full stages all passed, complete markers present, both supervisors
exited successfully. Remaining full stage finished18:33:33SGT; communicating
burns restored/verified18:34:44. Live ownership matches all8recorded burn workers,
parents are approved, collective progress validates and log was5.86seconds old.

All33runs now have the same nine-benchmark evaluation (seven multiple-choice
benchmarks plus LAMBADA/BLiMP reported separately). No fine-tuning was performed.
Remaining full summary SHA256:
131e8cea2f8de46e67082c8c6d61933d1a1308bb0c765d1daa54418ac5e6d758.
Local compact33checkpoint results:temp/results/proxy12-downstream-33-20261014.json.
Receipt:temp/remote_logs/proxy12-eval-completion-20261014-a01.log, SHA256
 d804533422cdc641be5802b6e71b18a5e6a22472d2e2882ca11daf96a30c4450.
commands.sh returned to#0. Next action: analyze/report the completed scores;
no new workload queued.

## Remaining 27-checkpoint evaluation launched — 14 October 2026, 17:31 SGT

Launch da91b12 accepted on q359, detached tmux pane PID704925. Read-only receipt
7f6227a at17:31 confirms all27 checkpoint/data/prompt preflights passed. Accelerate
resource copied to /dev/shm/.cache/huggingface/accelerate/default_config.yaml;
SHA256 923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5.
Exact approved burn workers704220–704227 under launcher704151 were reclaimed
at17:30:34. All8GPUs were verified empty at17:31:04, then numerics started.
At this snapshot the numerical stage is RUNNING, not yet verified passed.
Smoke (135 checkpoint/task groups), then full evaluation follow automatically
only on success; supervisor restores/verifies communicating burns after exit.

Root: /mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01.
Expected: supervised/run/full/summary.json status=passed, diagnostic=false,
27 checkpoint entries, then complete.json, successful supervisor and burn receipt.
commands.sh returned to#0; do not relaunch this detached queue.
Evidence: temp/remote_logs/proxy12-remaining-health-20261014-a02.log, SHA256
981899a6cce965b5b9c1e461a788799dc8abddc0435a76676f734eec72322314.

Predecessor six-checkpoint evaluation completed17:17:44SGT, all30full groups
passed; full stage847.997s. Burns verified17:18:55 before the new handoff.
Its full summary hash is c18c4bc8fe739ff5b7de7b12c5341f047d01adb2d82d40fc100b5c85a72341bb.
This action chose remaining-arm evaluation from the user's evaluate-or-finetune
request. No fine-tuning or new training is queued. All33final step2500runs will
have the same nine-task protocol once the remaining batch succeeds.

## Remaining 12-layer checkpoint evaluation authorized — 14 October 2026

User authorized evaluating all other checkpoints or fine-tuning the current six
while away. Chosen action: evaluate the remaining 27 final step-2500 checkpoints
(16 original screen runs and all 11 follow-ups), preserving distinct loss-weight
labels. Together with the current six this covers all 33 completed training runs.
No intermediate snapshots or historical 28-layer checkpoints are in this queue.

Same nine English benchmarks, shot counts, seed42, pinned eval_fa4 environment,
full splits and eight concurrent single-GPU workers. Each checkpoint first passes
real CUDA numerical checks and two-example task-group smokes. Fresh output root:
/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01.
Predecessor: q359-proxy12-eval-20261014-a01. Launcher waits for its successful
full summary, complete marker, supervisor completion, verified communicating
burns and released guard. Then it copies/verifies Accelerate, checks GPU ownership,
reclaims only approved burns, verifies GPUs free and evaluates. Existing supervisor
restores/verifies burns after success/failure. Training/model code is unchanged.

Local validation: all six orchestration tests passed, including original six-arm
compatibility, remaining-checkpoint selection, repeated architecture/weight labels,
and evaluation/burn handoff gating. Shell syntax and diff checks passed.
Launch submission/remote acceptance still pending; do not claim running yet.

## Evaluation ETA snapshot — 14 October 2026, 17:11 SGT

Read-onlya3ff667:19/30checkpoint/task-group jobs completed;7started,4pending.
A, P6-iso and P6-iso-deep have all five groups complete. No failure reported.
Latest HellaSwag scoring ~298requests/s; remaining P7 jobs and final summary/
burn restoration imply estimated finish17:20–17:25SGT, not verified completion.
Full stage remains running. commands.sh returned to#0; no workload changes.
Receipt:temp/remote_logs/proxy12-eval-eta-20261014-a01.log, SHA256
05509b8e5cca28af4939edb2631ea68af69756a3578fc708f8f816d64f053605.

## Full 12-layer downstream evaluation running — 14 October 2026, 17:04 SGT

Launch1ad321c accepted; detached tmux PID696398 remains alive. Receipt60ae197
at17:04:10 confirms all CPU preflight/prompt audits, all6real CUDA checkpoint
numerical checks and all30checkpoint/group smokes passed. Full stage is running
with8independent GPU workers. FA4/SDPA logit relativeL2 .00260–.00295;
padding delta0; active proxies and unchanged buffers verified. Smoke scores are
diagnostic only and must not be reported as benchmark results.

Exact approved burn workers695375–695382 were reclaimed; all8GPUs verified free
at17:00:50SGT. Accelerate resource copied and verified in eval_fa4's cache. No
environment/driver/training-weight changes. Local and remote full prompt audits
agree: max2042tokens(five-shot),1180(HellaSwag10),1232(ARC-C25),656(zero-shot).
No truncation. Existing supervisor will restore/verify communicating burns on
success or failure after cleaning only its own workers.

Root:/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-20261014-a01.
Expected result:supervised/run/full/summary.json,status=passed,diagnostic=false;
then successful supervisor.json and verified final burns. Current status is
RUNNING, not completed. Scope remains6arms/9benchmarks, no finetuning or paired
loss. commands.sh returned to#0; do not relaunch this queue.
Evidence:temp/remote_logs/proxy12-eval-health-20261014-a03.log, SHA256
46ee07af874c1fabeb8f88911777b764fc0e84a2fb4f260fa1b1de5f77ee4c75.

## Authorized 12-layer downstream screen — 14 October 2026

User requested evaluation after training. Read-only0f7d88a at16:44:49SGT confirms
all22original arms and all11follow-ups completed successfully, including result
validators and summaries. Follow-up summary completed15:59:52SGT; communicating
burn restoration verified16:01:03SGT. All8GPUs show the expected burn load.
All6selected step2500/12layer checkpoints and eval_fa4 dependency pins verified.
Receipt:temp/remote_logs/screen-eval-preflight-20261014-a01.log, SHA256
8a61165d97ed2c15dfd35276b6aee6340616e62f6f480d8fdb2e4c5454915eeb.

Scope frozen before downstream scores: A, P6-iso, P6-iso-deep,
P6-iso-adapter, P7-simple, P7-simple-lm(lambda.1). Same12layer2500step seed42.
Tasks: HellaSwag10shot; ARC-C25shot; PIQA/ARC-E/WinoGrande/BoolQ5shot;
SciQ/LAMBADA OpenAI/BLiMP(all67)0shot. Existing pinned harness scoring;
seven-task multiple-choice mean, language diagnostics separate. No finetuning,
paired-loss study, generation tasks or new pretraining authorized by this stage.

scripts/proxy_screen_evaluation.py reuses eval.suite commands/execute/collect
and existing downstream numerical/prompt checks. Eight independent GPU workers,
fresh outputs, full splits after six-checkpoint CUDA numerics and full-suite
2-example smokes. Checkpoint hashes, seed, architecture/depth, data provenance,
BF16/FA4 and identical benchmark prompts validated; no silent truncation.
Existing supervisor reclaims only verified burns, checks GPUs free and restores
communicating burns automatically on success/failure. Preflight copies/verifies
Accelerate although scoring itself uses independent workers rather than DDP.
Local review:16test run found only6subcase failures in one outdated delayed-arm
fixture(active-proxy assertion at step0). Updated test fixture to2500; all5focused
rechecks passed12.894s, including inference equivalence/strict loading across all
arms and three new queue/gate tests. Remaining15original tests passed. No model
or training implementation changed. Actual local zero/five-shot prompts fit;
remote preflight rechecks every task before any GPU reclamation.
Target fresh root:q359-proxy12-eval-20261014-a01. Launch receipt still pending.

Latest four follow-up PPLs: P6 lambda1=36.352774;
P6-iso-targetgrad lambda.01=36.153596;
P7-simple-sparse=36.189873; P7-simple-short=36.141397.
Best remains P6-iso-deep36.061986 vsA36.276876, single-seed screen only.

## B200 recovered; original screen complete, follow-up healthy — 14 October 2026, 13:28 SGT

Read-only request1bc81c5 succeeded. Previous hostname-resolution error was an
access failure; no restart was needed. Screen22 completed all22arms/validators
and comparisons at06:06:53SGT; supervisor verified communicating burns on all8
GPUs at06:07:59. Follow-up preflight passed, Accelerate cache hash matched the
resource, exact burn workers were reclaimed and all8GPUs were free at06:09:46.
CUDA numerics and all11eight-GPU smoke/validation pairs passed.

Follow-up has7/11production runs and validators complete; P6 lambda1 (case8)
is running at2343/2500, all8GPUs98–99% utilization. Remaining after this:
P6-iso-targetgrad lambda.01, P7-simple-sparse, P7-simple-short. No failed stages.
Estimated finish about16:00–16:15SGT today from observed49–51min/run; estimate,
not completion. Automatic communicating burns remain configured after exit.

Completed follow-up PPL (2500steps,12layers,seed42):
P7-simple-deep36.080857; deep-shallow36.071237;
P6-iso-consumer36.131321; consumer-only36.219153;
P7-simple-lm lambda.0336.093250, lambda.336.111052;
P6 lambda.336.325693. BaselineA36.276876; originalP6-iso36.117302;
best originalP6-iso-deep36.061986. These are single-seed screening results.

Evidence:temp/remote_logs/followup-results-20261014-a03.log, SHA256
0a72dde828bef0436a4556dfb5e12915c545b93fb914556638fa5a1ba8756cf0.
commands.sh returned to#0; running queue untouched. Next action read-only status.

## Status unavailable — 14 October 2026, 12:36 SGT

Read-only status request34ce131 failed before reaching q359: runner SSH could
not resolve the machine hostname (rc255). This is an access failure, not evidence
that training failed or completed. Latest verified state remains05:18SGT:
screen22 on final P7-simple-delayed; follow-up11 waiter alive and correctly
waiting for predecessor success and verified communicating burns. Completion,
handoff and new results remain unverified. No GPU workload was stopped or
restarted. commands.sh returned to#0. Do not submit a duplicate queue.
Evidence:temp/remote_logs/q359-status-20261014-0437.log, SHA256
6415b24356e20b3369ab517424a25751db48caebbfb7535ae54f60aa7cafe063.

## Automatic follow-up reconfirmed — 14 October 2026, 05:18 SGT

User explicitly reconfirmed that the new arms must follow the current run,
waiting for communicating GPU burns to be started and verified before reclaim.
Read-only command873b29b confirms the existing follow-up waiter is alive:
one tmux pane,PID581217,pane_dead0, exact launcher with the screen22 predecessor.
No duplicate queue was created. Screen22 is on its final arm P7-simple-delayed
(started05:17SGT);87stages passed. Follow-up logs WAITING_FOR_SCREEN_AND_BURN50
and predecessor_ready=False, as required. All11cases match the approved list;
launcher, queue, supervisor, model, shared-loss/Trainer helpers and Accelerate
resource hashes match reviewed local source.

Handoff: current screen+validators succeed -> all-rank communicating burns
verified and guard released -> fresh preflight/copy Accelerate/accelerate env ->
new guard and exact verified burn-worker identities -> stop only those workers
via pidfds ->30seconds and all8GPUs verified free -> CUDA numerics/all8GPU smokes
->11fresh2500-step runs sequentially,all8GPUs each. Supervisor restores/verifies
burns on completion or failure after cleaning its own workers. Unknown GPU work
or failed predecessor validation prevents launch. No current workload stopped.

Receipt:temp/remote_logs/followup-handoff-verify-20261014-a01.log,
SHA2564dc4d74b80361b2fca74a913249a51cd0ec26f64811ed35f286b7b70131a8bb1.
commands.sh returned to#0; independent tmux waiter continues automatically.

## Consumer review and exact overhead reductions — 14 October 2026

Both consumer-aware variants remain included as follow-up runs3/4; P7 deep and
its shallow-source control run first. Reviewed all arm families; implemented
reuse of per-forward valid counts, frozen consumer RMS/teacher FP32 values, and
removed eager allocation of unused Trainer masks/positions. Scientific recipes,
architecture, gradient routes and checkpoint formats remain unchanged.

Three new exact-comparison tests passed49.464s (A plus all47proxy arms,
FP32/BF16/checkpoint replay, gradients and AdamW state). All80 broader regression
tests passed847.649s, including distributed training/resume, independent consumer
objectives and document isolation. B200 throughput gain has not been measured.
B200 deployment3d469a7 verified14October05:12:08SGT: all5CPU tests passed
in22.502s, including exact outputs/gradients/AdamW across48architectures and
independent consumer response/isolation checks. Recorded source hashes match
the local reviewed files. Original screen still running P6-iso-sparse-delayed;
follow-up still WAITING_FOR_SCREEN_AND_BURN40. No GPU processes changed.
Pending training subprocesses load this reviewed source; actual CUDA numerical
gates and eight-GPU smokes still precede follow-up production. The already
running training process retains its imported implementation. No B200 throughput
percentage is claimed. commands.sh returned to#0; both queues remain independent.
Evidence:temp/remote_logs/shared-work-verify-20261014-a01.log,
SHA256a2fa26bee5b4931f22372b47adfbd24e1242c4b97d8cfb62f9f88a9ba396a496.


## Authorized overnight follow-up — 14 October 2026

User requested additional implemented arms to run after the current screen while
away about11hours, explicitly including lower/higher auxiliary weights and more
P7/P6-style variants. Eleven selected cases and exact definitions are recorded
at the top of PROJECT_NOTES.md. Consumer-aware experiments are now included.
Same12-layer2500-step recipe,seed42/data42/proxy43,all8GPUs/experiment,
FA4/document isolation/reset positions and all250-step checkpoints.

New queue uses scripts/proxy_overnight_queue.py and the existing launcher with
an optional predecessor root. It waits for screen22 success, all22validators,
communicating-burn verification and guard release. Fresh preflight then copies
Accelerate config/verifies env, checks pinned assets/data and GPU ownership;
existing supervisor reclaims only verified burns and verifies free GPUs. All9
architecture numerical cases and all11eight-GPU smokes must pass before fresh
production. Burns are restored on success/failure. Separate output root:
/mnt/local/_outputs/deep-llms_th2/q359-proxy-followup11-20261014-a01.
Predecessor:q359-proxy-12layer-screen22-20261013-a01. No old output/cache deletion.
Local new tests/validator passed. All35 broader regression tests passed in537.469s,
including the original24-arm Trainer/resume/report and CPU two-rank tests.
Six new tests passed (including the baseline-summary mismatch guard).
Launch045238b acknowledged14October04:49SGT. Independent tmux session
q359-proxy-followup11-20261014-a01 is alive (pane PID581217, pane_dead=0),
logging WAITING_FOR_SCREEN_AND_BURN. Predecessor still running
seed-42-arm-P6-iso-sparse-delayed; it was not interrupted. Follow-up training
and remote CUDA gates have not started yet. commands.sh returned to#0.
Evidence:temp/remote_logs/proxy-followup11-launch-20261014-a01.log,
SHA256 5fe4ce1748629f0d7126c1ed6f18f23d7b5e3a3bdf083fc5ec265d60bc92b867.

## Screen22 results — 14 October 2026, 03:10 SGT

Read-only command f340fb9 confirms 18 production arms completed at2500updates
and all18 validators passed; no failed stage. New finals since prior snapshot:
P6-iso-sparse-targetgrad loss3.619002/PPL37.3003;
P6-iso-sparse-deep loss3.587493/PPL36.1434;
P7-simple-targetgrad loss3.613619/PPL37.1001.
P6-iso-deep remains best: loss3.585239/PPL36.0620, versus A36.2769.
P7-simple-lm running at970/2500, about1.17s/update. Three arms follow:
P6-iso-delayed, P6-iso-sparse-delayed, P7-simple-delayed.
Single-seed12-layer screen only; no downstream evaluation in this queue.
No workload changes; commands.sh returned to#0. Supervisor running.
Evidence: temp/remote_logs/screen22-results-20261014-a09.log,
SHA256 871a8607e49fa3a4ae0f24a56f27c2b75e60419efa0d4f26472fd06b834d820d.
Parsed results: temp/screen22-results-a09.json.

## P6-iso-deep-shallow completed — 14 October 2026, 00:51 SGT snapshot

Read-only command ab3c246 verified P6-iso-deep-shallow completed 2500 updates
at 00:22:43 SGT, exit 0, validator passed. Wall time 49m10s. Final validation
LM loss3.5872892363/PPL36.1360, versus deep3.5852392901/36.0620 and
P6-iso3.5867720296/36.1173. The proxy-fed deep branch beats its shallow-feature
control by0.002050nats (about0.205%PPL); one-seed evidence only, with different
predictor LM-gradient routing as well as adapter input between these arms.
15 production arms complete; sparse-targetgrad running at1471/2500, six queued
after it. Supervisor running. No workload changes; commands.sh returned to#0.
Evidence: temp/remote_logs/screen22-deep-shallow-status-20261014-a08.log,
SHA256 8035b4df36d11c0c9bda3cd3c489d6a09c542eb89b512b633f26686628067ab7.
Parsed results: temp/screen22-results-a08.json.

## P6-iso-deep completed — 13 October 2026, 23:51 SGT snapshot

Read-only command bb24612 verified P6-iso-deep completed 2500 updates at
23:33:31 SGT, exit 0, and its validator passed. Wall time 49m35s.
Full validation (4882 rows): LM loss 3.5852392901, PPL 36.0620.
PPL is 0.592% lower than A and 0.153% lower than P6-iso; this remains a
single-seed 12-layer screen. Depth-current also completed: loss3.5858574634.
14 production arms are complete. P6-iso-deep-shallow is running at892/2500;
seven remain after it. Supervisor still running. No workload changes.
Evidence: temp/remote_logs/screen22-deep-status-20261013-a07.log,
SHA256 10d73d383d0bd58f2cf3ba1e0d528d22143dddd609b0b8935bf3755447462639.
commands.sh returned to #0; automatic final burn restoration unchanged.

## Screen22 queue status — 13 October 2026, 22:46 SGT

Read-only command cbca3a8 confirms 13 production arms completed and validated.
P6-iso-depth-current finished at 22:43:54 SGT; P6-iso-deep is now running,
at 67/2500 updates in the snapshot. Eight production arms remain after it:
deep-shallow, sparse-targetgrad, sparse-deep, P7-simple-targetgrad,
P7-simple-lm, delayed, sparse-delayed, P7-simple-delayed (P6-iso prefix
omitted for its suffix variants). Both consumer arms remain deferred.
Supervisor reports running. No workload changes; commands.sh returned to #0.
Evidence: temp/remote_logs/screen22-queue-20261013-a06.log,
SHA256 e818559dbd0222eff5ebedbf1a3f8ed894897d3314fd413263cca07e869b9905.

## Screen22 interim results — 13 October 2026, 21:56 SGT

Read-only command3d0ca03 verified12 completed/validated arms, all at2500updates,
matching train/eval fingerprints and seed42/data42. P6-iso-depth finished with
loss3.5868324/PPL36.1195, essentially equal to parent P6-iso36.1173. Next arm
P6-iso-depth-current running at35/2500. All8GPUs99%busy; no failed queue stage.

PPL ranking of completed arms: adapter36.0679, adapter-shallow36.1091,
shallowgrad36.1144, P6-iso36.1173, depth36.1195, P7-simple36.1257,
sparse36.1586, A36.2769, P6 36.3095, lowaux36.3243,
bothgrad37.6266, targetgrad37.6281. P6-iso prefixes are omitted in this sentence
for the named suffix variants. Best observed adapter gain vsA is0.576%PPL,
vsP6-iso only0.137%; one-seed screen, no robustness/downstream claim.
Target-gradient variants regress about3.72%PPL vsA in this configuration.

Evidence:temp/remote_logs/screen22-results-20261013-a05.log,
SHA256 0fff1c414040f428f184f074180e74a3db466d485318b7019999277a929b72c0.
Parsed results:temp/screen22-results-parsed-a05.json. Workload unchanged;
commands.sh returned to#0, automatic final burn restoration still configured.

## Screen22 live status and ETA — 13 October 2026, 21:37 SGT

Read-only command7132c00 confirms11/22 production arms completed and validated.
Newly completed since prior check: P6-iso-sparse, P7-simple, P6-iso-adapter and
P6-iso-adapter-shallow. Current12th arm P6-iso-depth at1572/2500, about1.12s/update;
10full arms remain afterward. All8GPUs95–99%busy, about65–66GiB device memory.
No failed stage; supervisor and checkpoint retention healthy.

Observed completed wall times46.1–50.2minutes/arm. Estimated remaining8.55hours
from21:37SGT, finish about06:10SGT14October; practical window05:30–07:00 if
throughput remains stable and no failure. Estimate uses each remaining arm's
third smoke-update timing scaled by median completed wall/(2500*smoke timing)
factor1.0536; delayed variants use20%dormant/80%parent active timing, plus current
remaining steps and final validation/burn handoff allowance. This is an estimate,
not a completion guarantee. Current arm should finish around21:56SGT.

Additional final PPL: sparse36.1586, P7-simple36.1257, adapter36.0679,
adapter-shallow36.1091 (A36.2769). Single-seed exploratory results only.
Evidence:temp/remote_logs/screen22-health-20261013-a04.log,
SHA256 5c69196353dabdf05b861762e04981db87e88d7615f52569e5223e5de8bbea9f.
No workload changes. commands.sh returned to#0; final burns remain configured.

## Screen22 live status — 13 October 2026, 10:02 UTC

Read-only command04ef4ae confirms q359 healthy. All21 CUDA numerical cases and
all22 eight-GPU training smokes/validators passed. Seven production arms finished
and validated at2500updates: A, P6-iso, targetgrad, shallowgrad, bothgrad, P6,
P6-lowaux. Eighth arm P6-iso-sparse running at570/2500, about1.09s/update;
14arms remain after it. All8GPUs active at96–99%, roughly64–65GiB allocated
device memory reported by nvidia-smi. No failed queue stage; supervisor active.
Completed arms retain all10 checkpoints from250 through2500.

Full-validation LM losses: A3.5911805111, P6-iso3.5867720296,
P6-iso-targetgrad3.6277512592, P6-iso-shallowgrad3.5866927335,
P6-iso-bothgrad3.6277105874, P6 3.5920788962, P6-lowaux3.5924878404.
These are the new12-layer2500-update screen, not historical28-layer results.
Evidence:temp/remote_logs/screen22-health-20261013-a03.log,
SHA256 fdaeb20220425336d2c200b7af62d34f55d0690437a75336eed6565de3ab484d.
No workload changed; commands.sh returned to#0. Final burns remain configured.

## B200 22-arm launch authorized — 13 October 2026

User approved the complete screen, deferring consumer-aware experiments. Run22
arms: all24 except P6-iso-consumer and P6-iso-consumer-only. Same12-layer random
initialization, seed42/proxy43,2500updates,125LR warmup,1,048,576tokens/update,
FA4 isolated documents/reset RoPE and all250-step checkpoints. No downstream.

Read-only preflight0093324 verified q359 at03:33:01UTC, all8B200s, pinned
attention_bench packages, expected model/data paths and20.6TiB free. Only the
approved communicating burn workers379567–379574 were observed. Reclamation
will recheck identities; no arbitrary process or process group is authorized.

Launch root q359-proxy-12layer-screen22-20261013-a01. Existing supervisor will
restore and verify communicating burns after success or failure. Preflight
hashes assets/data, copies resources/accelerate_config.yaml into its actual
cache and runs accelerate env. Numerical cases run in up-to-eight-GPU waves,
then each arm gets a separate three-update all8GPU training/validation smoke.
Only after all gates pass do22 fresh production runs start sequentially.
Production checkpoints never resume smoke outputs. Existing full-data packing
and HF caches remain shared and untouched.

Launch c743fa5 acknowledged. At03:45UTC all21 active CUDA numerical cases had
passed, including the new routes, adapters, deep branches and delayed arms.
A and P6-iso all8GPU three-update smokes and validators passed; targetgrad smoke
was running with finite losses/gradients. Remaining smokes and all production
fits proceed automatically under the91-stage manifest, stopping on any failure.
No production result yet. Initial burn workers were identity-verified and stopped;
all8GPUs recorded free at03:39:48UTC before any numerical/training work.
All352 English files (134,739,753,089bytes) passed release hashes. Accelerate
cache copy/byte equality and accelerate env confirmed8processes/BF16. Latest
receipt:temp/remote_logs/screen22-health-20261013-a02.log. commands.sh returned
to#0 after status retrieval; the independent tmux supervisor keeps running.

## 24-arm correctness re-review — 13 October 2026

Reviewed the current gradient routes, consumer objectives, delayed activation,
resume, sequential queue and result validation. No new training-logic defect
was found. Expanded the actual train.py integration test from13 to all24 arms:
each trains on a tiny synthetic CPU fixture, saves/resumes, reloads for inference,
and enters one matched report. Existing new-arm uninterrupted/resumed weights
remain checked for exact equality. Corrected superseded scope notes below.

All44 relevant local tests passed (15 screen/backend-validator tests and29
routing/extension/normalization/synchronization/queue regressions), plus the
expanded24-arm integration test rerun. These include two-rank Gloo, BF16,
checkpoint recomputation, independent gradient references and dormant optimizer
state. No actual CUDA FA4 or B200 capacity/throughput validation was performed.
No production training, GPU operations or push; commands.sh remains#0.

## Complete 24-arm screen implemented locally — 13 October 2026

User asked to implement the other arms to reach24; no launch requested.
Nine additions: P6-iso-sparse-targetgrad, P6-iso-sparse-deep,
P7-simple-targetgrad, P7-simple-lm, P6-iso-consumer, P6-iso-consumer-only,
P6-iso-delayed, P6-iso-sparse-delayed, P7-simple-delayed. Exact definitions and
the full numbered24-arm queue are at the top of PROJECT_NOTES.md.

All use the existing train.py/Trainer/data pipeline, random-init12layers,
2500updates and PPL-only reporting. The delayed variants retain500 baseline
updates inside that budget; proxies are frozen until introduction, and only
their auxiliary ramp starts then. A saved counter governs resume/evaluation.
Consumer-aware variants match a frozen shallow decoder's response to proxy
versus target (with or without cosine), adding two local block forwards/site.

scripts.proxy_12layer_queue --all-arms prepares24 sequential eight-GPU jobs
with validators and final reports. It does not launch them. Default9 and
--extensions15 remain compatible. Delayed runs re-audit attention at activation;
validators distinguish frozen teacher calls and actual gradient-carrying calls.

Local synthetic CPU checks include independent objectives/gradients, exact
resume, two-rank Gloo for every new arm, frozen dormant optimizer state,
FP32/BF16/recomputation, global loss scaling, isolation and FA4-layout references.
No real CUDA FA4 or B200 capacity/speed test performed. No training launched,
GPU workloads touched or remote changes pushed. commands.sh remains#0.
Unrelated user docs are preserved. Real-data launch remains a later step.

## Six additional paired arms implemented locally — 13 October 2026

Current request: review the new arms and implement other arms; continue the
implementation-only scope. No training launch, remote push or GPU management.
commands.sh remains#0. Unrelated user documents are preserved.

Added P6-iso-adapter / adapter-shallow, P6-iso-depth / depth-current, and
P6-iso-deep / deep-shallow. Each pair has identical additional capacity and
starts at P6-iso's forward function. The deep-proxy arm deliberately adds an LM
gradient route into its predictor; its shallow injection remains detached.
See PROJECT_NOTES.md for exact formulas, initialization and gradient contracts.
No additional backbone/attention pass; two-state depth history and short-lived
deep-message storage. Actual B200 speed and memory have not been measured.

`scripts.proxy_12layer_queue --extensions` prepares fifteen sequential
2500-step/12-layer PPL runs; default remains the earlier nine arms. Shared
Trainer, data/cache/packing, schedule and checkpoint rules remain in use.
New architecture metadata and trained extension tensors are checked by reports
and validators. This historical fifteen-arm scope is superseded by the complete
24-arm implementation above, including delayed and frozen-consumer variants.

Local synthetic CPU tests exercise active branches, gradient paths, isolation,
exact resume/reload, BF16, recomputation, compiled estimators, two-rank Gloo,
global loss scaling, queue/report validation and FA4-layout reference agreement.
CUDA FA4/B200 performance tests and real training have not been run.

## 12-layer proxy screen implemented locally — 13 October 2026

Current request: implement the first variants, improve throughput where safe,
and do NOT launch training. commands.sh remains#0. No remote pushes, GPU
management, real-data training or downstream evaluation in this implementation.

New arms: P6-iso-targetgrad, P6-iso-shallowgrad, P6-iso-bothgrad, P6-lowaux.
The first three expose the named auxiliary gradient paths with prediction
injection still detached. P6-lowaux is ordinary P6 with lambda_max0.03 instead
of0.1. Existing target statistics remain detached; target-adaptive gradients
pass through the actual MLP sum, standardization and clipping. Arm/routing
metadata and strict reload/resume/report validation included.

train.py supports an explicit random-init depth override. A separate
proxy_heads.12layer.b200.json uses12layers,2500steps,125LR-warmup steps,
1,048,576tokens/update, FA4, EOS/document isolation and all250-step checkpoints.
The auxiliary ramp remains250steps. scripts.proxy_12layer_queue generates nine
sequential arms (A, P6-iso, three routes, P6, lowaux, sparse, P7-simple), each
followed by validation, then PPL reports; it never executes the queue itself.
Adapters, delayed introduction and frozen-consumer methods were added in the
subsequent implementation entries above.

Throughput work: existing no-checkpoint/synchronization optimization recipe,
one backbone forward, shared target contributions, shared centered FP32 values
for differentiable targets and detached statistics. No speculative kernel change.
CPU tests cover independent gradient references, exact unchanged LM-only paths,
FP32/BF16/checkpoint/compiled regressions, Trainer save/reload and exact resume,
two-rank Gloo from lambda0 to active auxiliary loss, global accumulation scaling,
12-layer full-schedule validation, and FA4-layout reference equivalence (CPU
kernel substitute, not CUDA validation). No B200 speed/memory claims.

Details and queue-generation command are at the top of PROJECT_NOTES.md.
Next step is user review; any GPU numerical/performance smoke and production
launch need a subsequent launch request. Existing unrelated dirty docs preserved.

## Sparse downstream evaluation blocked by missing checkpoint — 13 October 2026

User approved inference-only evaluation of P6-iso-sparse at step2500, seed42,
against matched A/P6-iso, reusing compatible historical scores. No new training.
Read-only inventory command67a80c1 completed on thiennh-p6-q359-worker-0 at
06:34:50SGT. Searched all project output/model trees:672 weight directories in
outputs, none for sparse; the old proxy-followup-2500-20261008-a01 root is absent.
No sparse weights found in local project artifacts/outputs/temp or the known
/home/users/thien/deep2shallow-backups tree. Saved result/config/validation
receipts remain available, but do not contain the model parameters.

No GPU workloads stopped; all8B200 GPUs were100%utilized by the existing idle
workload at inspection. No evaluation or training launched. commands.sh#0.
Evaluation requires locating/restoring the original checkpoint plus recipe;
do not silently reconstruct it by new pretraining or substitute a20k model.
Evidence:temp/remote_logs/sparse-checkpoint-inventory-20261013-a01.log,
SHA256 d1524131f8c2209c2cc844c84d09c2602efb53b6e0afb296c6c89662fed655d2.

## Full result reports retrieved — 13 October 2026, 05:02 Singapore

Export command4023e8f acknowledged: all seven original JSON reports downloaded
into artifacts/paired-final-20261013/ and their SHA256 values match the remote
receipts: paired-F/Z/probes/position, precision, finetune-summary, expanded-summary.
commands.sh returned to#0. No new GPU work requested; prior completed status and
verified burns unchanged. User requested results, not another experiment.

At20k, held-out PPL A17.886315/P6-iso17.841602. Expanded eight-benchmark
multiple-choice mean A40.2798%/P6-iso40.0569%; canonical few-shot eight-task
mean49.5579%/49.4411%; single-seed42/LR1e-5 fine-tune eleven-score mean
76.9720%/76.2409%. These are descriptive means, not official combined scores.
Paired Z confirms improved correct-answer log likelihood, but not relative
correct-choice probability or accuracy. bAbI mean accuracy24.7122%/22.1089%.
One pretraining seed; do not infer across-seed robustness or downstream gains.

## Paired study complete — 13 October 2026, verified 04:55 Singapore

Root q359-paired-parallel-repair-20261013-a02 completed successfully at04:34:21SGT.
All100 GPU scoring jobs and all6 outer stages passed. Scoring took18m08s;
planning plus scoring plus four analyses took23m46s. All F/Z/probes/position
reports exist and their hashes match the successful runner records. Earlier
expanded benchmark and single-seed/fixed-LR fine-tuning are already complete.
Paloma remains the explicitly deferred suite (gated source unavailable).

Supervisor verified all8GPUs free, then restored communicating burns at04:35:27.
Fresh04:55:27SGT inspection: same workers379567–379574,100%utilization on all8,
advancing collective cycles through1330 and1476.77GiB collective payload.
No training/evaluation running or queued. commands.sh#0; do not relaunch.

Initial paired result: Z M1(correct-answer log likelihood/character) favours
P6-iso:mean gain0.002715998,95%CI[0.000970197,0.004431836];standardized test
p=0.0000999900,H1rejected. M2(correct-choice relative probability) gain
-0.000102400,p_std=.26757,H2notrejected. Zaccuracy:A47.7275%,P6-iso47.3353%;
Faccuracy:A49.5579%,P6-iso49.4411%. These are fixed-suite checkpoint-level
comparisons, one pretraining seed; likelihood benefit is not an accuracy gain.
Z/probes/position used mandated FP32 SDPA fallback; F retained production.

Evidence: temp/remote_logs/paired-completion-check-a01.log SHA256
50569cfd1f2e2d98fdc7bf2b734403f8b6b1417e37066315b4f2129199c757fe.
Locally extracted status and report summaries: artifacts/paired-completion-20261013/.
These are summaries, not complete copies of the four original remote reports.
Original report hashes:F e8298001125752b9eaf533073c4f4164e538799d6def2d7ed6c7c4acbdba744c;
Z808ea923455b73f50632695bbcea6261c162e4e4fb7417381db82721e752384f;
probes0b3a95d2db4ce8e7e8760f9b54c5af97ca677f649dc179673746af98eb8fb0ff;
position316d8cc839e2581b2912773ef097db538f303aec811efa38dc392933def7f6a1.

## Repaired paired run verified healthy — 13 October 2026, 04:15 Singapore

Current root: /mnt/local/_outputs/deep-llms_th2/q359-paired-parallel-repair-20261013-a02.
Launch d55a875; monitor0b153f9 / pull5895e81. All18group/140task loaders passed
on both dev and B200 before GPU reclamation. Accelerate copied/verified; only
verified burns reclaimed and all8GPUs empty before start at04:10:35SGT.
Precision30/30 score jobs completed; config alignment repaired and all strict
numerical checks ran, selecting FP32 SDPA for Z/probes/position. F retains its
production path. No thresholds or prompts changed.

At04:15:09SGT:41jobs completed (30precision+11full),8running,0failed. P20 F
XStoryCloze finished in27.87s, resolving the second failure. Cached-model reuse
is recorded model_loaded=false across finished full tasks. Eight distinct CUDA
workers373115–373122 on physical GPUs0–7; MMLU/BLiMP and remaining F tasks active.
The frozen study has100 GPU jobs total (30precision,67benchmarks/probes,3position),
then CPU reports. No further fine-tuning. Original completed summary is saved
locally in artifacts/expanded-single-seed42-20261013/finetune-summary.json.

Validation:18 paired regression tests (including real spawned HellaSwag
serialization/score equality) and3 parser/restart tests passed. All actual
TSV rows agree with canonical CSV reader. Estimate04:45–05:15SGT13October,
provisional from observed small-task timings; full position/report runtime not
measured yet. Automatic final communicating burn restoration unchanged.
Evidence temp/remote_logs/paired-parallel-repair-health-a03-final.log SHA256
 ae2d53215b8242aa5a9ac33e2ae34510b291e358352957bf046082841d19e267.
commands.sh returned to#0 after acknowledgement; do not relaunch.

## Second paired input repair — 13 October 2026, 04:08 Singapore

First fresh retry passed the previous config-comparison failure and completed
its precision gate. All five diagnostic tasks failed production equivalence;
the frozen rule correctly selected FP32 SDPA for Z/probes/position. No tolerance
was changed. Main scoring then failed on XStoryCloze: the raw manifest omitted
TSV separator, so datasets CSV parsing used commas. All worker children cleaned
up; communicating burns restored/verified by04:02:48SGT, PIDs371644–371651.

Fix the pinned source manifest to specify sep=tab. Retry preparation creates a
new manifest with that one loader-field repair and unchanged raw files/hashes;
all old manifests/results remain immutable. The paired protocol/scoring source
is unchanged from the callable fix. Actual dev TSV verification compares every
field, prompt, choice and target for360train/1511eval rows against csv.DictReader:
all pass. New parser regression and retry tests:3passed in4.138s. Added preflight
loads every actual paired task before burns are reclaimed. Fresh a02 retry uses
q359-paired-parallel-repair-20261013-a02, original reduced study as input; no fits
rerun, all precision/scoring outputs fresh. Still awaiting verified healthy run.
Evidence temp/remote_logs/paired-parallel-repair-health-a02.log SHA256
6b44ff18d24a79816dda689292a253f9c9e1ad5538b842e2ed33ce3022feb01c.

## Paired serialization repair and fresh retry — 13 October 2026, 03:56 Singapore

Read-only receipt at03:47SGT: fine-tuning summary passed at03:07:53SGT,
all56 reduced-queue stages successful (13new fits,22final scores). Paired GPU
scoring completed all30 precision jobs on8GPUs in149s, but comparison failed:
`Paired task configs differ`. Exact field diff identifies only HellaSwag nested
fewshot_config.process_docs repr addresses; other four precision task configs
match. This is serialization/provenance failure, not failed numerical tolerance.
The supervisor restored and verified communicating burns at03:12:10SGT;
workers367392–367399 still occupied all8GPUs at03:47SGT.

Repair: recursively serialize nested callable identities before saving harness
results, using existing protocol.serial; do not weaken config alignment, sample
identity or precision gates.18 paired regression tests passed in122.890s,
including actual cross-process HellaSwag config/score equivalence and rejection
of genuine config changes.2 fresh-retry safety tests passed in0.023s.

Authorized paired-only retry: q359-paired-parallel-repair-20261013-a01.
Fresh protocol and output, identical scientific settings and2,000 token rows;
only eval/paired/run.py implementation hash may change. All original artifacts
preserved. No fine-tuning or expanded-suite rerun. Six stages: paired plan,
8GPU evaluate, F/Z/probes/position analyses. Copy/verify Accelerate, verify
prior dead supervisor and burn handoff, inspect burn identities, reclaim only
verified workers, wait/recheck all8free, unchanged automatic final burn handoff.
Launch prepared; awaiting remote acknowledgement. No supported finish ETA yet.
Evidence: temp/remote_logs/paired-parallel-health-a01.log SHA256
2accc92fb068892119da93b5608fb43c510f1fcc087a6bba7fa9b6199e1d9c7c;
paired-config-diff-a01.log SHA256
6c0c848829a93bbbc3ac85cd3ff00e4382a9466b18a3ffd674780e3773f05d0b.

## Parallel paired deployment verified — 13 October 2026, 02:47 Singapore

B200 acknowledged the parallel scheduler deployment without stopping fine-tuning.
Protocol refresh passed, with exact scientific-rule equality and unchanged2,000
position rows (rows SHA2563867288ee637a70cd2c20c9dd20ac809233e49c94c14c9a9530842aca3ff6961).
New protocol SHA2566a59b182ad7e4bd8c395ddde374c01aa662d852551fc6d9a006084b0da02f52c;
new sample SHA25617e32080f720969170d6963e087c20e7fe66a232169e56071e95ae77f81e99ff
(changed only because it binds the updated protocol). Original inputs preserved.
Current evaluation was A MNLI matched; paired planning still has not started.
Its already queued CLI run automatically selects8workers from the allocation.
No B200 GPU throughput claim yet. Automatic supervisor/burn handoff unchanged.

Validation: all18 local paired/scheduler/protocol-refresh tests passed in108.297s,
including real spawned tiny-model scores equal to serial/cached scores, parallel
process isolation, crash cancellation, original precision fallback and resume.
All3 CPU process-management tests also passed on B200 (1.679s). These did not
acquire GPUs or interfere with current downstream evaluation.
Evidence temp/remote_logs/paired-parallel-enable-a01.log SHA256
 eb36fa6f46c5bdd5d9a9cb38bc779efdb34c6c15204ceb015e170cd80c5b5d36.
commands.sh returned to#0 after acknowledgement.

## Paired evaluation parallel scheduling — 13 October 2026

User explicitly requested efficient multi-GPU paired evaluation. Implemented
one persistent spawned worker per explicitly visible GPU, automatic8workers
in the existing GPU queue. `--workers 1` retains the serial reference. Independent
checkpoint/task jobs execute concurrently; long groups first, checkpoint affinity
and resident-model reuse. Precision subsets/repeats deliberately reload models.
A barrier and original artifact/precision validation still precede main scoring.
Three position jobs run concurrently; scoring batch sizes/formulas unchanged.
Worker status/timing/model-load records: paired/parallel-status.json and sibling
*.worker.log files. Failure/cancellation stops only owned spawned processes.

At02:41SGT, B200 had completed all13 remaining fits and was doing final test
scoring (P6-iso QQP); paired output did not yet exist. No workload stop is needed.
Refresh frozen execution hashes before paired planning with
scripts/refresh_paired_schedule.py; requires active fine-tuning and absent paired
output, exact reviewed implementation delta, unchanged science and position rows.
Preserve original protocol/sample files as *.before-parallel-20261013. Both
protocol/sample paths remain usable by the already loaded outer queue. Frozen
statistics/calibration sources are unchanged; do not rerun or weaken calibration.

Validation: spawned worker concurrency/isolation, failure/crash cleanup, original
precision fallback/resume checks, and tiny-model fresh vs cached vs real spawned
scores agree exactly. Full18-test regression now passes; see the verified deployment record above.
No measured B200 speedup yet; do not repeat the unsupported4–10hour ETA as fact.

## Reduced queue timing — 13 October 2026, 00:52 Singapore

P6-iso QQP fixed1e-5 is actively training, epoch1.172/3, loss0.1324 and finite
gradients. Eight GPUs occupied (~21–30GB,50–85% utilization). Original A QQP
fit took1139.8s (~19m); original SST-2 fits ~5m and CoLA/MRPC ~2m. No new
failure. Base fine-tuning/final-score ETA is roughly04:00–06:00SGT13October;
paired stage is not yet timed and may add4–10h, especially on FP32 fallback.
These are planning ranges, not measured completion promises.

Isolation reviewed: eval.finetune requires a fresh output directory, sets seed,
loads original source checkpoint and invokes trainer.train() with no resume.
Seven previous completed identical1e-5 fits are reused intentionally; remaining
fits have new optimizer/scheduler state. Partial3e-5 output is not reused.
Receipt temp/remote_logs/reduced-seed42-health-a02.log SHA256
 e11bd9f397370ba5891efa3dcce99e64ba6e0e57bd30fe353e650cba0ebefab7.

## Reduced queue verified live — 13 October 2026, 00:44 Singapore

New single-seed/fixed1e-5 queue is active. Seven complete fits reused (A/P6-iso
CoLA, SST-2, MRPC; A QQP), leaving13 fits. Reuse hash validation passed on B200.
Current first new fit is P6-iso QQP1e-5; all8 distributed workers initialized.
Do not report full study complete. Expanded benchmarks remain in original a03.

Old supervisor305589 was identity-checked and stopped via pidfd SIGTERM;
its own cleanup verified8free GPUs and restored communicating burns at00:37:53
SGT. New launch waited for verified handoff/dead pane, copied Accelerate to
/dev/shm/.cache/huggingface/accelerate/default_config.yaml and checked accelerate
env (8GPU, bf16). New supervisor stopped only verified burn workers344580–344587,
verified8free GPUs, and started the reduced queue. Automatic final burn remains.

Evidence: temp/remote_logs/reduced-seed42-health-a01.log SHA256
67892b05edcf5b1e5a3490a4565807c6fd2bb31b0ee61fcc9d46401295d5436b.
Launch receipt: temp/remote_logs/reduced-seed42-launch-a01.log SHA256
072fb80a54ca38baeefed41432c7bdb238b4402e0ae2af909905e814e41c027b.
Status command acknowledged; commands.sh restored to#0. Fixed rate chosen
before new final test scoring; previous3e-5 artifacts retained but excluded.

## Reduced downstream budget — 13 October 2026 (Singapore)

User requested base results first: one fine-tuning seed and one learning rate.
Use seed42, fixed1e-5 for A20/P6-iso20, three epochs as before:20 total fits and
22 final task/split evaluations. No seeds43/44, 3e-5 trial, or LR search. Existing
selection files are recomputed with the sole fixed-rate candidate. Single-seed
summaries have null standard deviation, not zero or fabricated multi-seed claims.

The original a03 supervisor was signaled via the existing identity-checked,
pidfd-based stop helper; its final cleanup/verified communicating-burn handoff
must finish before replacement. Original outputs/checkpoints remain intact.
New root: /mnt/local/_outputs/deep-llms_th2/q359-expanded-single-seed42-20261013-a01.
New launch log is adjacent: q359-expanded-single-seed42-20261013-a01-launch.log.

`reduce_expanded_study.py` builds a fresh queue, verifies recorded artifact hashes
and best-weight hashes, and reuses only successful identical1e-5 fits and passed
gates. Failed/partial fits restart under the fresh root. Completed expanded
benchmarks are retained; the paired study still follows fine-tuning unchanged.
`launch_reduced_expanded.sh` waits for the source stop receipt and dead supervisor
pane, copies/verifies Accelerate, inspects ownership, then uses the unchanged
safe burn-reclaim/free-check/final-burn supervisor. No remote success implied yet.
22 targeted CPU tests passed (11.232s), including continuation tamper rejection,
fixed-rate selection, one-seed summary, and existing three-seed behavior.

## Expanded evaluation status — 12 October 2026, 15:52 UTC

Expanded benchmarks remain complete (validated summary passed). Fine-tuning
has advanced beyond smoke/reload gates to full learning-rate search. CoLA
search/selection completed for both arms; SST-2 A search/selection and P6-iso
1e-5 completed. Current job: P6-iso SST-2 3e-5, eight workers initializing.
98 nested jobs passed, one current; no failed job in recorded state. These
counts include checks/selection, not 98 full training runs. Paired analysis
has not started. Supervisor remains running; do not report study complete.

Receipt: `temp/remote_logs/expanded-paired-status-a06.log`, SHA256
`bbd5a962ff932bb223a6ba6f818795a57bed9c03a26dfd1296ba7e41faf1268b`.
Read-only status request made no workload changes; commands.sh restored to #0.

## Expanded evaluation status — 12 October 2026, 14:35 UTC

All eight expanded benchmark jobs completed; full summary status passed at
14:04:50 UTC (2h06m). Expanded fine-tuning is running its train/save/reload smoke
checks: 35 subjobs passed, A STS-B reload is current. Full fine-tuning and paired
analysis are not complete. No failed job appears in this snapshot. GPU query
caught the transition into the next eight-worker reload, with no compute PIDs;
this is not an overall completion signal. The supervisor remains running.

Evidence: `temp/remote_logs/expanded-paired-status-a05.log`, SHA256
`6b60da18caf0ce5aa1de3747ccafb245a69e6b04c467024f2851a4ac808d3ca2`.
Full benchmark summary SHA256:
`868190a224e6131c0faeeb077b05a1f10a3d8e233de6f7a118fed361ab05a4db`.
Status command acknowledged; return commands.sh to #0. No workloads changed.

## Expanded evaluation status — 12 October 2026, 13:22 UTC

Fresh a04 read-only receipt confirms a03 is running. Both numerical gates and
expanded smoke passed. Six of eight full benchmark workers saved results:
zero/five/continuation groups for A20 and P6-iso20. Both GSM8K workers remain
active on GPUs2/6 (~40–44% utilization, ~32GB each); full collection has not
finished. Fine-tuning and paired analysis have not started. No failure is
reported in this snapshot. Do not call the overall study complete.

Evidence: `temp/remote_logs/expanded-paired-status-a04.log`, SHA256
`7e9126e5f815fcdb8582dde1cb6b9225c1c200494b68eb0acd19e34859aa1190`.
The receipt also confirms Accelerate verification, stopping only authorized
burn workers, and all eight GPUs free before a03. Status request made no GPU
changes. Restore commands.sh to #0 after its acknowledged receipt.

## Expanded smoke collector correction — 12 October 2026

The2,000-document sample passed full local reconstruction and B200 preflight;
its sample hash matches across machines. a02 copied/verified Accelerate, stopped
only the known burns, verified all8 GPUs free, and passed A20/P20 numerical gates.
All8 benchmark smoke subprocesses returned successfully, but collection rejected
GSM8K's canonical per-filter records as duplicate documents. This is a result
collector bug. lm-eval0.4.10 logs one record per document per scoring filter.

Collector now requires unique(task,document,filter), complete matching document
sets for each reported filter, and counts each document once. Scores/prompts and
harness filters are unchanged.12 targeted tests passed, including multi-filter
counts, duplicate/filter omissions and the position source/overlap checks.

Retry a03 first requires completed and verified a02 burn restoration, then
revalidates all saved real smoke outputs on CPU with the corrected collector.
Only if those pass does it start a fresh full queue. Preserve a01/a02 artifacts.
Fresh root: /mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03.
Evidence: temp/remote_logs/position-2000-health-a02.log and
 temp/expanded-filter-regression.log. Normal/paired evaluation metrics and model
code are unchanged; the frozen2,000-document cohort is retained.

Retry416cf2f acknowledged in a03 receipt: prior communicating-burn restoration
verified, all8 actual B200 smoke outputs pass the corrected collector, and fresh
a03 launcher is active in CPU preflight. Receipt observed19:56SGT12October:
`temp/remote_logs/expanded-a03-launch.log`, SHA256
606709b7515813bde54f7241a6a05703c2c32c56689b0b56139e0c48f8ed1dd0.
Do not report full evaluation complete. Next check a03/supervised/run/run.json
and its numerical/smoke/full status; commands.sh returns to#0 after acknowledgement.

## Position probe extension approved — 12 October 2026

User rejected the882-document fallback and authorized extending to2,000 with
English documents from nht10/cx_sampled. The new pool is already local. Its
validation split supplies866 additional unique2048-token prefixes (old882 plus
new866 gives1748 before training-overlap exclusions). Fill the remainder from
new-pool train shard000000 only after excluding exact full-text matches against
ALL36,595,514 old English training documents. That newer train split was not
used by these checkpoints. Exclude duplicate texts and duplicate2048-token
prefixes. This is exact-match screening, not a near-duplicate guarantee.

The CPU audit verifies all278 old training Arrow files against the published
manifest while checking candidate membership. Commit only row identities/hashes,
source pins and audit receipts; no documents or token arrays. Freeze before any
model scores. Use the same selection for A10/A20/P20. Scripts:
`prepare_position_extension.py` creates candidates/audit/selection;
`extended_position_sample.py` verifies and materializes them offline.

Downloaded only the new validation and first training Arrow shards (231,810,872
bytes) via87bdf8d, revision9d5b93963a98f0e576f4d8a8361f214a30f6de0a. Receipt8de54c2
verifies both B200 file hashes, old manifest hash and tokenizer hash match local.
See temp/remote_logs/position-extension-source-check-a01.log.

Final selection:881 original-validation documents +865 new-validation documents
+254 new-pool documents unused by these checkpoints =2,000. Scanned all36,595,514
old training documents and verified278 Arrow hashes; excluded4 matches among
3,555 candidates. Metadata/audit are frozen in
`resources/position_extension_selection_20261012.json` and
`resources/position_extension_overlap_20261012.json`. Both sources were checked
against immutable HF revision and B200 source hashes. No near-duplicate claim.

Status correction: the original a01 queue completed all245 benchmark downloads
but failed CPU preflight because historical F logs lack tokenizer_sha256. No
GPU evaluations started; burns293909–293916 remained active. Preserve a01;
relaunch into `/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a02`
with2,000 documents and the frozen selection. Recompute paired Suite F
with its existing canonical tasks/shots, rather than fabricating missing metadata.
Expanded benchmarks remain first, then GLUE/SQuAD fine-tuning, then paired study.
Paloma remains deferred. GPU handoff/burn restoration logic is unchanged.

Verification so far:20 expanded/paired regression tests passed; the final8-test
launch suite additionally covers full training-file audit and corruption rejection.
Evidence: temp/position-extension-final-tests.log and position-extension-study-final.log.
The following section records the superseded882-document launch.

## Expanded + paired 20k evaluation launch — 12 October 2026

User authorized both studies, expanded benchmarks first. Verified q359 live at
10:36 UTC: eight B200 GPUs running the approved communicating burn, no active
evaluation, expected eval_fa4 versions, and all three source checkpoints present.
The input manifest covers245 immutable HF files,263,061,811 bytes, with checksums
in `resources/expanded_inputs_20261012.json`. Initial download commit4337a35
used filenames where dataset URL directives require directories. Stopped only
that download batch (f3782f4), then repaired39 verified nested files in place
without deleting the originals (20cf364). The CPU waiter and eight communicating
burn workers remained active; no GPU evaluation had started. Corrected downloads
use directory destinations; MMLU/BLiMP/GLUE use whole-repository downloads whose
required files must match the same pinned manifest before consumption.
The downloaded native snapshots are used offline; XCSR alone is normalized.
Social IQa's legacy loading script is avoided using its pinned HF Parquet
conversion. No new benchmark data is published or included in Git.

Fresh queue root: `/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a01`.
Launch commitd996f54 is active as a CPU waiter. Corrected controller download
commitd4bb848 was acknowledged as batch2026-10-12_11-06-37. Read-only health
receipt1f18da7 confirms input progress212→215/245 at11:08:49–11:09:49UTC;
the corrected directory layout is reaching the waiting launcher. At this receipt,
GPU evaluation had not started. Original burn workers293909–293916 were verified
unchanged during repair; existing source checkpoints were not modified. Evidence:
`temp/remote_logs/expanded-layout-repair-20261012-a01.log` and
`temp/remote_logs/expanded-paired-health-20261012-a01.log`. Next check: all245
input hashes, preflight.json, supervised/run/run.json and numerical/smoke results.
Return commands.sh to#0 after acknowledgement; this does not cancel either
the controller download or detached study launcher.

Order: checkpoint numerical checks → expanded smoke → expanded full → GLUE and
SQuAD v2 fine-tuning → paired planning/evaluation → paired reports. Expanded
likelihood uses the established shot counts, excludes the eight already completed
few-shot tasks, and runs independent groups on the eight allocated GPUs.
Fine-tuning retains eight-GPU Accelerate/Trainer, development-only selection over
1e-5/3e-5, seeds42/43/44, all epoch checkpoints, and common fitted MNLI weights.
Use A/P6-iso seed1042 step20k; A10 is the original non-cooldown step10k reference.
Reuse the original20k Suite F sample artifacts after frozen-protocol validation.

Paloma is explicitly deferred: the existing HF credential lacks gated access.
An input-only audit of all11,822 original English validation documents found only
882 with at least2,048 tokens. Use all882, explicitly frozen before scoring;
do not duplicate documents or substitute packed sequences. The fixed14-task
paired Suite Z, hypotheses and statistical calibration are unchanged.

Local verification: all21 final CPU tests passed in48.907s, including real
harness/model artifacts, queued reuse/fallback, explicit position sample counts,
parallel allocation, offline XCSR and existing expanded protocols. Real-data
audits passed: Social IQa1954, CommonsenseQA1221, bAbI199/200/200 story clusters.
All1319 GSM8K five-shot prompts fit: maximum1561 prompt tokens +256 generation.
Evidence: `temp/expanded-launch-final-tests.log`, `expanded-real-audits.log`,
`expanded-real-paired-audit-20261012.json`, `expanded-gsm8k-prompt-audit.json`,
`position-pool-audit-20261012.json`, and `remote_logs/expanded-eval-preflight-20261012-a01.log`.

The launcher waits for controller file sizes, verifies all checksums, performs
CPU/data preflight, copies and verifies the Accelerate cache config, checks
approved burn ownership, then invokes the established train_then_burn supervisor.
Burns stay active while downloads/preflight run. The supervisor verifies free
GPUs before evaluation and restores/verifies burns after completion or failure.
No environment or driver installation/change is part of this launch.

## Paired evaluation follow-up review — 12 October 2026

Reviewed the paired adapter, statistics, queue and position probe. Fixed missing
validation around precision metadata (FP32 reference with TF32 disabled, BF16
production and identical repeat settings), and applied the frozen precision
decision to bAbI analysis as well as Suite Z. Restricted the descriptive
independent-question admission/filter tables to Suite Z; bAbI retains its
story-cluster tests and joint-bootstrap contrasts.

The queue now validates the position sample before launching any job, and again
before position extraction. Completed position outputs must match the planned
recipe, tokenizer and batch size too. Position comparisons reject mixed numeric
settings; the native-loss agreement check explicitly rejects NaN/Inf.

The statistical implementation and calibration code are unchanged; the saved
full calibration source hashes still match. Original evaluation/training code,
runner commands and environments are unchanged. No B200 job or push requested.

Verification: the 32-test CPU regression ran 31 checks successfully, with one
queue-fixture mismatch because the process had loaded the fixture before the
final position-reuse guard was added. Fresh execution of all 11 affected
statistics/queue and real-model entry-point tests passed, including that queue
case. Evidence: `temp/paired-review-regression.log` and
`temp/paired-review-final.log`. The broad run covers the existing evaluator,
few-shot protocol and expanded-suite behavior; the final run uses saved tiny
P6-iso weights in FP32 and CPU BF16. Real-data audits and B200 precision checks
remain required before launching the study. Freeze a fresh protocol after these
implementation changes.

## Separate paired evaluation implemented — 12 October 2026

User authorized the revision-3 paired design while explicitly retaining the
existing evaluation methods. Added `python -m eval.paired`; ordinary benchmark,
few-shot, fine-tuning and PPL entry points keep their scoring and defaults.
Reuse the existing checkpoint loader and harness adapter. Only the dataset
snapshot preparer gains an opt-in `--profile paired`; its default is unchanged.

Implementation/commands: [../eval/paired/README.md](../eval/paired/README.md).
Includes protocol/source hashes, strict project-artifact parsing, M1–M3/ACC,
fixed H1→H2 analysis, descriptive filtering and A10 progress ratios, joint
probe contrasts, story clustering, all 16 Paloma sources, position inputs/losses
and T8, representative precision subsets, BF16/FP32 gates and an ordered queue.
Queue completion reuse verifies checkpoint/recipe/data/protocol identities;
incomplete outputs stop rather than being overwritten. GPU management remains
in the established deployment workflow, not this evaluation package.

Local CPU verification: 30 regression tests passed (paired protocol plus original
evaluator/few-shot/scoring/queue behavior). Subsequent provenance and nonfinite
checks have targeted tests. Tests use actual lm-eval 0.4.10 artifact generation,
all paired task templates on fixtures, and saved tiny P6-iso weights in FP32 and
CPU BF16. These are implementation checks, not B200 numerical validation.
Full T4a–e calibration passed: 1,000 datasets/case, B=2,000 for T4a and 1,000
otherwise; all exact comparisons within 3 Monte Carlo SE. Story-level rejection
5.6%, versus 28.6% for the deliberately invalid example-level control.
Saved evidence: [PAIRED_LOGPROB_CALIBRATION_20261012.json](PAIRED_LOGPROB_CALIBRATION_20261012.json),
including source hashes, per-task/aggregate interval coverage and RNG settings.
Ignored detailed logs: `temp/paired-regression.log`, `paired-provenance-final.log`,
`paired-release-checks.log`, `paired-calibration-complete.log`.

No real benchmark download, checkpoint analysis, B200 job or remote push was
performed. Before a real study, freeze the final protocol, audit pinned real
data, freeze original validation documents, and run the numerical gates on the
target machine. Do not claim five precision tasks prove equivalence everywhere.

## Paired likelihood review and plan — 12 October 2026

User requested documentation of the review of the revision-2 downstream paired
analysis spec, reference Python script and temp/custom_tasks.zip. Findings and
staged plan: [PAIRED_LOGPROB_REVIEW_AND_PLAN_20261012.md](PAIRED_LOGPROB_REVIEW_AND_PLAN_20261012.md).
Recommend paired likelihood metrics alongside accuracy, with fixed-suite main
analysis and descriptive admission. Reproduced missing-subtask/count validation,
constant-effect and nonfinite-input issues. All five custom tasks loaded on
fixture data under pinned lm_eval 0.4.10; real-data/model validation remains.
Next proposed work is an adapter for current sample artifacts and statistical
tests, then reuse of saved few-shot samples. Documentation only in this turn;
reference files unchanged, no new analysis implementation or GPU jobs submitted.

## Missing benchmark support complete — 12 October 2026

Added requested SciQ, MMLU continuation (57 subjects), and Paloma (16 sources)
using canonical lm_eval 0.4.10 definitions. Existing snapshot preparation picks
up their pinned repositories/configs/splits through the shared task inventory.
The sequential suite adds zero-shot SciQ, five-shot continuation and zero-shot
Paloma. `paloma`, quoted `paloma_*`, and individual source names work.

Continuation accuracy is example-weighted and reported separately from standard
MMLU. Paloma uses unchanged custom-model rolling likelihood with logits caching
disabled; word/byte PPL and bits per byte are separate from accuracy averages.
Report descriptive selected-source means, not pooled PPL; bootstrap resampling
is disabled for Paloma. The HF corpus is gated; no agreements were accepted.

Local CPU checks passed: missing-benchmarks-tests.log (3 tests, including all
207 canonical/custom templates on fixture rows), missing-benchmarks-rolling-final.log
(1 test, A/P6-iso/P7-simple direct-forward comparisons, exact rolling token-count
oracle, unchanged weights/buffers and public-entry-point cache/scoring policy),
missing-benchmarks-regression.log (25 tests, existing evaluator/few-shot and
expanded protocols). Logs are under ignored temp/. Syntax and diff checks passed.
These are implementation tests, not real benchmark results or B200 GPU checks.
No training code, dependencies, commands.sh or remote workloads changed.
Usage/protocol details: eval/README.md. No new evaluation jobs launched.

## Expanded English benchmark implementation — 12 October 2026

User authorized implementation of the previously planned suite plus MMLU,
OpenBookQA, Social IQa and unsupervised BoolQ. No B200 launch was requested.
Added canonical lm_eval 0.4.10 task selection (57 MMLU subjects, 67 BLiMP
subsets), local XCSR adapters, greedy full-prefix GSM8K generation, and
sequential suite validation/benchmark-level averages. Added official GLUE
transfer tasks (including WNLI and both MNLI splits) and SQuAD v2 span transfer
through the existing Trainer, including cached sliding windows and development-
only no-answer calibration. The queue shares fitted MNLI weights between its
matched/mismatched final splits. Source checkpoints remain unchanged.

Networked dev preparation freezes HF source commits and exports checksummed
portable parquet snapshots. XCSR uses labeled English validation only; its
public test labels are hidden. GLUE/SQuAD use train-derived grouped development
splits and public validation as final holdout. No hidden-test leaderboard claim.
GSM8K retains the learned proxy, recomputes complete prefixes without a KV cache,
and rejects overlength prompts plus generation budget. Existing likelihood
truncation behavior remains unchanged and must be reported in context audits.

Local CPU evidence in ignored temp/: expanded-final-tests.log (9 tests,
including 133 real harness templates and 11 Trainer train/reload configurations),
expanded-trainer-final.log (recheck including shared MNLI reload and cached QA),
expanded-last-fixes.log (4 passed, including suite identity/count validation and
existing custom test-task compatibility), expanded-span-gate.log (A/P6-iso/
P7-simple span gradients versus CPU FA4 reference kernel). Combined regression
expanded-complete-regression.log ran 41 methods: 40 passed, one old-task-manager
compatibility error was fixed and its targeted rerun passed in last-fixes.
Two added checks also passed; 43 distinct test methods now have passing coverage.
Real-source/Qwen-tokenizer audit: expanded-real-audit.json verifies XCSR
1,000/300 labeled rows, TruthfulQA 817 rows, official CoLA/STS-B preprocessing,
and SQuAD span/offset roundtrip (synthetic perfect logits, not model accuracy).

No pretraining code/weights, commands.sh, remote environment or GPU workload changed.
commands.sh remains inactive. Before a real expanded study: deliver pinned data,
run the existing B200 numerical gates and smoke/reload checks, audit context
lengths (especially GSM8K), then launch only the authorized scope. See eval/README.md
for exact task protocols and CLI examples. No benchmark results were added.

## 20k downstream study complete — verified 12 October 2026, 10:31 SGT

Read-only receipt6ab8909: all15 stages passed, including all few-shot validators,
four fine-tuning studies and final cross-checkpoint validation. Held-out means
A/P6-iso: PAWS93.733/93.183%,NLI84.943/85.689%,STS-B84.851/84.818,
BoolQ75.280/75.525%. STS-B is100×mean(Pearson,Spearman). Three fine-tuning seeds
42/43/44,one pretraining seed1042. All selected LR3e-5 on development data.
Results remain mixed; detailed table/SD added to PROXY_RESULTS_SUMMARY.md.
Final summaries pulled and SHA-verified against runner artifact records.
Source weights unchanged; all epoch model checkpoints retained.

Supervisor completed22:13:47UTC11October (06:13:47SGT12October), no errors,
automatic communicating burns restored/verified. Live02:31:22/38UTC12October
checks: approved workers293909–293916,all8GPUs100%,cycles16830→16850 and
collective payload18687.22→18709.42GiB. Guard released; no further jobs queued.
Evidence artifacts/downstream-20k-final-20261012/; receipt
 temp/remote_logs/downstream-20k-health-a05.log SHA256
fd82f8f0f30af90bfdfc2eaaf083a5f1392568586725399e2553c2815e546bc2.
commands.sh returned to#0. No workloads changed during this read-only check.

## 20k downstream progress — verified 12 October 2026, 03:53 SGT

Read-only receipt bf9f5f8: full few-shot evaluations and all5/10/25-shot
validators passed for both checkpoints. First fine-tuning study A20k PAWS/NLI
is running NLI seed43 confirmation (epoch0.1385→0.22 over15s), finite losses,
all8GPUs active. A extended and both P6-iso groups remain queued. No final
held-out fine-tuning summary yet; supervisor253720 remains active, no errors.
Automatic communicating burn restoration remains configured for queue end.
Few-shot results pulled/checksum-verified and added to PROXY_RESULTS_SUMMARY.
Evidence artifacts/downstream-20k-health-a04/; receipt
 temp/remote_logs/downstream-20k-health-a04.log SHA256
f0587a33ac05b12e2650209cd9fd5b775ec5e159f5ead8cb6e0292359a99ecbf.
commands.sh returned to#0; current queue untouched.

## 20k downstream queue verified running — 12 October 2026, 03:06 SGT

Launch cbd8e25 on q359 is active under persistent supervisor253720.
Read-only receipt ea85d89 confirms all21 remote CPU tests passed, pinned eval_fa4
packages and dataset/prompt audits passed. Both20k checkpoint hashes match the
just-completed PPL study. Resources Accelerate config copied to actual
/dev/shm/.cache/huggingface/accelerate/default_config.yaml and accelerate env
verified (8GPU,BF16,MULTI_GPU; SHA923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5).
Only verified burn workers252590–252597 stopped; all8GPUs free19:04:34UTC.
Both checkpoint CUDA numerical gates and all5/10/25-shot smoke validators passed.
Full few-shot evaluation is now running; four fine-tuning studies follow:
A20k PAWS/NLI → A20k STS-B/BoolQ → P6-iso20k PAWS/NLI → P6-iso20k STS-B/BoolQ.
32 production fits using original protocol; final summary and automatic
communicating burn restoration remain pending. No final downstream scores yet.
Source weights unchanged; all fine-tuning epoch checkpoints retained.
Root q359-downstream-20k-20261012-a01; completion supervised/run/downstream-summary.json.
Evidence artifacts/downstream-20k-health-a03/ and
temp/remote_logs/downstream-20k-health-a03.log SHA256
5528d7207a36d0347bb12f15ecaba60efd2fd8c737d74b518f27430032188eca.
commands.sh returned to#0; persistent queue continues. Next action: read-only
progress check, validate final results and confirm communicating burn restoration.

## Downstream evaluation of both20k checkpoints authorized — 12 October 2026

User requests few-shot and fine-tuning evaluation at20k. Scope: A and P6-iso,
seed1042, source q359-proxy-resume20k-seed1042-20261011-a01/supervised/run.
Reuse the completed10k study's exact English protocols, manifests and prompts:
5-shot XNLI/XStoryCloze/PAWS-X/PIQA/ARC-Easy/Winogrande,10-shot HellaSwag,
25-shot ARC-Challenge; demonstration seed42. Fine-tune PAWS/NLI/STS-B/BoolQ,
LR1e-5/3e-5 development selection with seed42 then selected LR seeds43/44,
held-out testing only afterward.32 production fits, all8GPUs per fit, same
FA4/BF16,512tokens,global batch128, task-only losses, frozen normalization stats,
all epoch checkpoints retained. Source pretraining checkpoints remain read-only.

Only orchestration changed: checkpoint count derives from spec (two here;
three-source regression retained), launcher accepts explicit checkpoint requests.
Existing numerical gates, smoke/reload tests, data/prompt references, full
few-shot validators and cross-arm source/data-order checks remain mandatory.
Remote CPU tests and data audits run before verified burn reclamation. Copy
resources Accelerate config to actual cache and verify accelerate env, then
verify known burns, stop them by checked identities, wait and require free GPUs.
Automatic communicating burn restoration on success/failure via original supervisor.
Fresh root q359-downstream-20k-20261012-a01; final artifact
supervised/run/downstream-summary.json. Submission pending verification.

## Full intermediate checkpoint PPL complete — 12 October 2026, 01:55 SGT

Fresh retry q359-checkpoint-ppl-20261012-a02 completed all12 eight-GPU evaluations
and paired comparison successfully. A/P6-iso PPL respectively:
15k18.9221/18.8700;16k18.6847/18.6416;17k18.4656/18.4159;
18k18.2516/18.2096;19k18.0684/18.0192;20k17.8863/17.8416.
P6 advantage0.23–0.28% across all checked steps, small and not widening.
Same full4882-row validation,9981660 targets,fingerprint08432871cf987d61.
Both20k controls exactly match saved results; all model state/source checkpoint
checks passed. No training/optimizer updates. Tables added to PROXY_RESULTS_SUMMARY.

First attempt's output-directory race was fixed with global-rank0 reservation
and broadcast errors; local and remote3-test suites passed (including two-rank
late-arrival regression). Fresh retry copied/verified Accelerate config and
confirmed all8GPUs free before eval. Earlier scores matched exactly on retry.
Supervisor succeeded and restored communicating burns17:53:22UTC. Read-only
17:55:24/39UTC checks show workers252590–252597,all8GPUs100%,approved parents,
collective cycles150→170/payload166.55→188.76GiB. Guard released; no queue remains.
Hash-verified results under artifacts/checkpoint-ppl-final-20261012/.
Receipt temp/remote_logs/checkpoint-ppl-retry-status-a06.log SHA256
ebb194e489c451bdf1b7cf04b27b63781e647f0c7c45ddf4ac8030c19d87acfa.
commands.sh returned to#0; no further workload submitted.

## Checkpoint PPL retry after output-directory race — 12 October 2026

First queue passed A/P6-iso at15k,16k,20k (both20k controls matched exactly),
then A17k failed before evaluation because rank0 created its output while late
ranks still checked nonexistence. This is an evaluation-script race, not a
model/checkpoint or GPU infrastructure failure. Completed scores are preserved.
Fix: only global rank0 checks/reserves output after distributed initialization;
creation errors broadcast to all ranks. Added a two-process Gloo regression
forcing the late worker to arrive after directory creation, including stale
output rejection on both ranks. All3 CPU tests passed locally.
First supervisor restored/verified communicating burns and released guard;
workers244463–244470 showed all8GPUs100% and advancing collectives17:32/33UTC.
Fresh retry root q359-checkpoint-ppl-20261012-a02 reruns both final controls and
all15k..19k pairs with original preflight, all8GPUs and automatic burn restoration.
Source checkpoints and first evaluation outputs are untouched.

## Intermediate checkpoint PPL authorized — 12 October 2026

User requested retained checkpoint PPL at15000,16000,17000,... . Evaluate both
A/P6-iso at15000..19000 every1000; re-evaluate both20000 controls first, requiring
saved full-validation LM loss within1e-5 before proceeding. Original ProxyTrainer,
unchanged FA4 model forward and full4882-row packed validation fingerprint/counts.
No optimizer/scheduler or training calls. Verify model weights/buffers and source
checkpoint hashes unchanged. Independent eval outputs; no overwrite of checkpoints.
Two local CPU tests passed4.120s covering A/P6 saved final+intermediate checkpoints,
matched counts, unchanged state, fingerprint rejection and controlled queue order.
Remote CPU tests/pinned env/Accelerate copy+env precede verified burn reclamation,
wait/free checks and12 sequential8GPU evaluations plus comparison; existing
supervisor restores communicating burns on success/failure. Root:
q359-checkpoint-ppl-20261012-a01. Source q359-proxy-resume20k-seed1042-20261011-a01.
Final artifact supervised/run/summary.json. Submission pending; no extra training.

## Both20k complete; communicating burns verified — 12 October 2026, 00:57SGT

Read-only receipt67c5314 confirms all6 stages succeeded. A/P6-iso20000 final
validators and combined continuation validator passed. Full4882-row validation:
A NLL2.884035908911387/PPL17.886315243302665; P6-iso NLL2.881532900308893/
PPL17.841601625011805. P6 PPL about0.25% lower at equal steps/tokens (one seed),
with8.05% greater cumulative Trainer runtime. No cooldown used.
Each arm retains all41 checkpoints10000,10250,..,20000. Original and copied10k
checkpoint hashes unchanged; only retention differs from saved recipes.
P6 training finished00:44:32SGT; final comparison00:45:04; burns restored and
verified00:46:15. At00:57:32/47 all8GPUs100%, workers239661–239668 have approved
burn parents, collective cycles760→770 and payload843.87→854.97GiB. Guard released,
supervisor exited successfully; no experiments queued.21.408TiB free.
Results hash-verified locally under artifacts/resume20k-final-20261012/.
Receipt temp/remote_logs/resume20k-final-a04.log SHA256
55ef9f73a5e0afdb44b46f1a41e2751f3aa9edfe42d966dd8954f6999c89dece.
commands.sh returned to#0. No new training or process changes submitted.

## A20k complete; P6-iso14486 active — verified 11 October 2026, 13:14UTC

Read-only receipt5c5be12: A finished20000 at10:23:56UTC, native validator passed.
Full4882-row LM NLL2.884035908911387, PPL17.886315243302665. Resumed10000 updates
cost21057.2239s (5h50m57s Trainer runtime); job elapsed5h51m57s. All41 checkpoints
10000,10250,..,20000 retained. P6-iso started10:23:57UTC; live samples advance
14479→14486 over15s,2.24s/update, all8GPUs99% utilized in second sample. Latest
saved14250; all18 expected checkpoint directories10000..14250 present. Finite
loss/gradient values, no errors in sampled production logs,21.576TiB free.
ETA about16:45UTC (roughly3.5h remaining at snapshot), including periodic saves.
P6 full final PPL pending. Supervisor218376 owns guard; original28600-step
schedule/cutoff20000 unchanged. Final validations and communicating burns remain
configured; no additional training launched or existing work changed.
Artifacts hash-verified under artifacts/resume20k-status-20261011-1314/.
Receipt temp/remote_logs/resume20k-status-a03.log SHA256
be315c4fec954d1eb82556dd81f69f5e2471f3881c77564b1421121d4df97cd4.
commands.sh returned to#0. Next: P6 completion, comparison and burn verification.

## Original seed1042 continuation running — verified 11 October 2026, 04:35UTC

Launch22e3dc7 passed11 remote CPU tests, environment/Accelerate verification;
only verified burn workers217654–217661 stopped. All8GPUs free04:31:25UTC.
Both original10000 checkpoints copied and hash-verified (model, optimizer,
scheduler, training args, all8RNG states) in33.22s. A training started04:31:59;
latest samples advance10072→10079, about2.07s/update, all8GPUs98–99% utilized.
Resume transition permits ONLY retention2→0; source recipe/fingerprints match,
LR2.381e-4 at10070 follows original28600/1430 schedule, no cooldown/restart.
P6-iso10000→20000 is queued after A20000 and validation; no seed42 jobs remain.
Keep all250-step checkpoints; final full PPL/matched report and automatic
communicating burn restoration on success/failure. Estimate12–13h for both.

Read-only receipt452c895 archived under artifacts/resume20k-20261011-a02/.
Log temp/remote_logs/resume20k-health-a02.log SHA256
adb094eff92b97101be4639034925404c86f3cfe1365c3733e2ef0be6332d8d4.
First health probe used the unactivated shell's default Accelerate path and
stopped early; corrected probe verified actual recorded/dev/shm cache hash.
Training was unaffected. commands.sh returned to#0; tmux supervisor218376
continues. Next action is read-only progress/terminal-result verification.

## Original seed1042 pair to20k authorized; seed42 cancelled — 11 October 2026

User superseded the fresh seed42 queue: stop it, resume original pre-cooldown
A/P6-iso checkpoint10000 to20000. Verified supervisor203574 received SIGTERM;
owned workers exited, communicating burns217654–217661 restored04:24:41UTC,
GPU guard released. Receipt9b2770a confirms seed42 A checkpoints250..1250
preserved and no P6-iso stage started. Cancellation is recorded as failed with
"Supervisor received signal15"; this is the authorized cancellation, not a fault.
Evidence temp/remote_logs/stop-seed42-verified.log SHA256
9734ca639c20cfab1a2bbd8dbe833f973a827d245fed7f2bc2acc46876e722eb.

New root q359-proxy-resume20k-seed1042-20261011-a01. Source:
q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042/{A,P6-iso}/checkpoint-10000.
Copy/hash all model/optimizer/scheduler/eight-rank RNG state; never alter originals.
Replay exact original job argv, changing ONLY stop_after20000, save_total_limit0,
and explicit copied resume path. No cooldown, synchronization optimization or
compilation enabled. Original28600 LR schedule,1430warmup,FA4,EOS/isolation/reset
positions, micro16/GAS4/all8GPUs, training/data seed1042 and proxy seed1043.
A20k → validate → P6-iso20k → validate/full PPL/matched report. Retain every250
checkpoint (41 per continuation including copied10000). No additional cooldown.
Launch gates: verified cancellation/burn handoff, CPU resume tests, pinned env,
Accelerate cache copy+byte check+accelerate env, approved burn identity checks,
guard, stop only verified workers, wait and all8 free. Existing supervisor restores
communicating burns after success/failure. Local12 targeted tests passed4.179s,
including unchanged model/optimizer state under changed retention. Training code
unchanged. Launch submitted; remote preflight/training confirmation pending.

## Cooldown complete; seed42 A training — verified 11 October 2026, 04:18UTC

Receipt e746709 confirms all8 seed1042 cooldown stages succeeded. Final PPL:
P6-iso11000=19.2038743; A11000=19.2680412; time-matched A11875=18.9971154.
P6 improves0.333% at equal steps but is1.088% worse at equal actual time. Time
mismatch is only0.0262% (A6.5561s longer). All source checkpoints unchanged;
full4882-row validation and zero final LR passed. Communicating burns restored
and verified03:32:08UTC; seed42 preflight then passed, verified burns stopped
03:33:34UTC, all8GPUs free03:34:04UTC, training started03:34:05UTC. Accelerate
cache verified at /dev/shm/.cache/huggingface/accelerate/default_config.yaml.
Seed42 A advanced1242→1249/10000 across15s, about2.06s/update, no errors observed.
Its approved P6-iso10k and both1000-step cooldowns remain queued. Old seed1042
A/P6-iso10k checkpoints contain optimizer/scheduler/all8RNG files; reusable.
User proposes extending previous runs to20k. Recommended appending seed1042
continuations after the current seed42 queue; scheduling clarification pending.
No20k training submitted, no running job interrupted. Resume from original
pre-cooldown checkpoints, retain28600 schedule/1430warmup, all checkpoints.
Artifacts hash-verified under artifacts/proxy-status-20261011-0418/. Receipt log
SHA25683c3ed11148b0d6e3082e4b909a031581005785ca4a0b7e0797aefd92291c7c4.

## Seed-42 handoff re-audited — verified 11 October 2026, 02:12UTC

Read-only receipt39f1129 confirms parent cooldown still owns the guard (PID170012)
and all8GPUs; seed42 waiterPID203574 is live, still waiting, with no reclamation or
training output. Both samples show current training98–99% GPU utilization. Remote
launcher, waiter, supervisor and Accelerate source SHA256 match the reviewed local
files. Waiting requires parent success + completed validation + all-rank burn
readiness/advancing collectives + guard release. Only afterward: CPU/env/data
checks → Accelerate cache copy/byte check + accelerate env → fresh burn identity
inspection → exclusive guard hold +30s → PID-handle identity recheck/verified
workers only →30s → all8GPUs free → per-job free check → training. Automatic burn
restoration retains guard ownership until handoff finishes. No launch-order defect
found; production code/current jobs unchanged. Twelve local CPU handoff/queue tests
passed1.506s, including new regressions for PID identity reuse and wait/free order.
Receipt temp/remote_logs/seed42-handoff-check-a01.log SHA256
45647a9a95aa6346209a944855ffe3b35c575877ba7c4e2f0803d2ad630197b0.
commands.sh returned to#0; waiting and training processes continue unchanged.

## Seed-42 replication waiter verified live — 11 October 2026, 02:03UTC

Launch24fab0e received on q359. Log confirms WAITING_FOR_COOLDOWN_SUCCESS_AND_BURN
and live tmux pane (PID203574). Existing seed1042 cooldown continues; no new GPU
reclamation or training has begun. Node has23TiB available, sufficient for the
planned retained checkpoints. The waiter is persistent and requires successful
parent results, verified communicating burns and guard release before preflight.
Queue root q359-proxy-10k-seed42-20261011-a01; A10k → P6-iso10k → copied A/P6
1000-step cooldowns, native validators and full PPL. Seeds42/43; all8GPUs each.
Automatic final burns configured. Receipt temp/remote_logs/seed42-replication-launch.log
SHA25608c3a171b6fd6dc2f360fdd48e71b294da8cb2cc31cbe963c1a2b23310a1e240.
commands.sh returned to#0. Next status: parent cooldown result, then replication
preflight/seed-42-arm-A progress. No further user input required to start.

## Seed-42 overnight replication authorized — 11 October 2026

User selected A + P6-iso with seed42 (proxy initialization43), fresh10k fits and
1000-step cooldown branches, after the current seed1042 cooldown study succeeds
and restores verified communicating burns. Seed42 was used in the earlier2.5k
screen; this is a second10k seed alongside1042, not a third independent seed.
Queue: A10k → validate → P6-iso10k → validate/compare → copy/hash originals →
A cooldown to11k → P6-iso cooldown to11k → full4882-row PPL and final validation.
These are equal-step/token comparisons; no equal-time extension was requested.
Original28600 schedule/warmup1430, FA4/document isolation/reset positions/EOS,
1,048,576-token batch, micro16/GAS4, all8GPUs, no activation checkpointing.
Exact synchronization option on for both; cosine compilation off. Save every250
steps and retain all checkpoints. Native Trainer, models/losses remain unchanged.
Existing launcher gains optional seed/predecessor arguments; default1042 path
retained. New orchestration waits read-only, verifies completion/burn/guard,
checks CPU tests, env/data/storage/Accelerate, reclaims only verified burns and
uses existing train_then_burn automatic restoration for success/failure.
Fresh root: q359-proxy-10k-seed42-20261011-a01. Parent dependency:
q359-p6-cooldown-20261011-a01. Local15 tests passed13.175s. Launch pending;
final artifact will be supervised/run/cooldown/summary.json. No preexisting
output or cache will be deleted. Shared training code stays unchanged.

## Timing study passed; P6 cooldown active — verified 11 October 2026, 01:39UTC

Read-only receipt77ab71a confirms all14 synchronization-study stages passed.
Eight100-step timing fits: A2.061121→2.053581s/update, P6-iso2.228933→2.221586s;
throughput gains0.367%/0.331%. P6 relative overhead remains8.18% versus8.14%
reference. Exact-state gate passed; tiny timing gain is not a material overhead
reduction. Hash-verified result artifacts are in artifacts/p6-workflow-20261011/.
Benchmark supervisor restored verified communicating burns01:32:07UTC. Cooldown
preflight then passed, reclaimed those burns, and started01:33:54UTC. Preparation,
two-update smoke and smoke validator passed (step10002, both final group LRs0).
P6-iso cooldown advanced10055→10062 across15seconds; all8B200s98–99% utilized.
Next: P6 to11000 → A from10000 to11000 → A from10800 with measured time budget →
full fixed validation PPL and source-checkpoint identity checks. Optimization is
selected for all three branches by the predetermined threshold. Final PPL and
cooldown comparison remain pending; automatic burns restoration remains enabled.
Receipt temp/remote_logs/p6-workflow-health-a02.log SHA256
25b6360200911eac788201790594de1f2c0c6d7f439711cfb8c268b6512c013d.
commands.sh returned to#0; active queue and training code unchanged.

## Workflow healthy; timing repeats active — verified 11 October 2026, 01:13UTC

Read-only receipt ebdf2cc confirms the eight-GPU exact-state gate passed and
four of eight 100-update timing runs completed: A-reference, P6-reference,
P6-optimized, A-optimized. A-optimized-repeat is running; P6-optimized-repeat,
P6-reference-repeat and A-reference-repeat follow. Both snapshots show all eight
B200s at 99% utilization, 110448 MiB each; no traceback in the sampled logs.
Cooldown waiter remains live and has not started training. It will proceed only
after the timing validator succeeds and benchmark burns are restored. No final
speedup or cooldown PPL is available yet. Original checkpoints remain untouched
by this read-only check. Receipt: temp/remote_logs/p6-workflow-health-a01.log,
SHA256 3b376351398d956b476f1fe95b90287424ffa8f1a15543f9258b975d3b16bc7a.
commands.sh returned to #0; existing supervisors continue unchanged.

## Exact GPU gate passed; measurements active — verified 11 October 2026, 00:57UTC

Read-only receipt3e9c93a confirms copied-checkpoint reference and optimized P6
fits and exact validator all passed on eight B200s. After two resumed updates:
restored model/optimizer/scheduler/RNG, eight microbatch hashes per rank, clipped
gradients, final model/optimizer/scheduler match exactly under deterministic FA4.
Gate/rank0 receipts pulled with matching runner SHA256 into artifacts/p6-sync-20261011.
Production-FA4 timing study: A-reference100 updates complete, P6-reference starting;
six more100-step fits and max-rank interval validator follow. No final speedup yet.

Cooldown waiter3bdd3a1 is live (tmux PID170012), waiting read-only for successful
benchmark completion and restored burns. It will repeat CPU/env/Accelerate checks,
reclaim only verified burns, run two-update cooldown smoke, P6/A token-matched
1000-update cooldowns, then A10.8k with measured time-budgeted cooldown and full
validation. Failure prevents dependent stages; each supervisor restores burns.
No production cooldown has started. Current active roots:
- /mnt/local/_outputs/deep-llms_th2/q359-p6-sync-20261011-a01
- /mnt/local/_outputs/deep-llms_th2/q359-p6-cooldown-20261011-a01
Final result: second root/supervised/run/branches/summary.json.
Ten targeted CPU tests now passed; production training code frozen during timing.
commands.sh #0; persistent queues continue. Next check: timing summary, then
cooldown smoke/training progress. No new architecture arms authorized here.

## Controlled cooldown queue prepared — 11 October 2026

Nine CPU synchronization/cooldown tests pass, plus the strict recipe-transition
check. Cooldown stays behind the real eight-GPU exact-state gate and repeated
throughput study q359-p6-sync-20261011-a01. New persistent launcher waits without
touching GPUs until that study succeeds and restores verified burns; any failure
prevents cooldown launch. Then repeat env/Accelerate checks and safe reclamation.
Fresh q359-p6-cooldown-20261011-a01: copied-checkpoint two-step cooldown smoke →
P6-iso10k+1000 → A10k+1000 → compute A10.8k cooldown duration from actual cumulative
runtime → A time comparison → full fixed4882-row validation/report. Original
checkpoints retained unchanged, all periodic branch checkpoints kept. No new
architecture arms, no further fine-tuning jobs. Normal production defaults stay
unchanged; adopt exact sync option for all three only if measured criteria pass.
Cooldown source-code additions do not alter the running benchmark's training code.
Submission pending; do not infer GPU gate success or cooldown start from this entry.

## Exact synchronization trial submitted — 11 October 2026

Profile completed on q359; A2.06056s vs P6-iso2.22968s complete max-rank update
intervals. Profile artifacts pulled with matching SHA256. Original checkpoints
unchanged; eight communicating burns restored00:39:17UTC, verified00:42UTC.
CPU14 tests pass for exact synchronization candidate, cooldown fork/resume and
previous compilation code; two legacy resume/bootstrap regressions pass too.
Submit q359-p6-sync-20261011-a01: copied10k P6 deterministic-FA4 exact-state
resume gate, then eight100-step A/P6 reference/optimized fits in reverse-order
repeats. Original model/loss/EMA arithmetic unchanged, no compilation. Complete
update intervals include all statistics/logging/data-delivery callbacks. Same
safe Accelerate/env/burn procedure; no production cooldown launched yet.
Details: P6_SYNC_AND_COOLDOWN_20261011.md. Pending GPU acceptance/timing.

## P6 exact optimization and cooldown workflow authorized — 11 October 2026

User approved profiling → exact synchronization/diagnostic optimizations →
remeasure A/P6 → controlled cooldown branches → evaluation before new arms.
First stage: two disposable30-step eight-GPU A/P6 profiles, production FA4,
micro16/GAS4, original28600/1430 schedule, unchanged model arithmetic. Instrument
complete update callbacks and start-to-start intervals on all ranks; profile
CPU synchronization events and component kernels on rank0 at update6. Fresh
root q359-p6-profile-20261011-a01; source10k checkpoints remain untouched.
Existing launcher verifies environment/Accelerate, reclaims only approved burns,
checks all8 GPUs free and restores communicating burns on success/failure.
No cooldown has launched yet; implementation and strict resume validation follow.

## Further P6 speed trial completed — 11 October 2026

Four100-step eight-GPU fits completed successfully; target scaling/clipping fusion
adds only0.310% measured throughput by means (0.216% medians), essentially no
memory saving. Cosine-only2.171355s/update vs target+cosine2.164644s; final short
LM NLLs9.30654–9.30658. All batches/fingerprints match, no timed recompilation.
Target passes all four CUDA numerical cases with exact LM outputs/backbone+gate
gradients/moments/buffers. Predictor differences remain tiny as for cosine-only.
Gate compilation with preserved precision casts still FAILS; excluded from fits.
Recommendation: keep cosine-only as the supported experimental option; the extra
fusion is not worth production complexity on this evidence. No default changes,
no resumed research training, no environment/driver changes or cache deletion.

Launch c31fc36, root q359-p6-further-20261011-a01; six stages finished00:02:11UTC,
communicating burns restored00:03:22UTC. Receipt a7fe4d8 at00:03:54UTC verifies all8
burns100%, workers149790–149797. CPU tests7passed locally16.276s /B20016.708s.
Report:docs/P6_COMPILE_TRIAL_20261011.md. Artifacts verified by SHA256 and archived
in artifacts/p6-further-20261011/. Receipt2fa533f at00:07:20UTC confirms source
checkpoint unchanged, all8burns100% and advancing collectives. commands.sh now#0;
no experiments queued. Target numerical comparison records equal prior cosine
records for all four cases.

## P6-iso speed trial completed — verified 11 October 2026, 07:11 SGT

All eight stages passed: CUDA numerical checks, six100-step eight-GPU Trainer
runs, aggregate validator. Source checkpoint hash unchanged after completion.
Cosine-only compilation: mean step2.16813s versus eager2.22381s (2.57% throughput
increase); A2.05617s. Median comparison gives2.96%. All measured updates retained,
including two slower updates in compiled repeat2. Mean overhead versus A drops
8.15%→5.45%; peak allocated111.620→108.626GiB per GPU. Not speed parity with A.
At identical trained weights: exact LM outputs/backbone+gate gradients and
normalization buffers; maximum predictor-gradient relative L2 difference0.04461%.
Gate/code-normalization compilation FAILED criteria and was excluded. No new
compiler graphs in timing window; matching dataset fingerprints and first/final
update batch hashes across all ranks/runs. Production defaults remain unchanged;
trial wrapper is experimental, not an approved exact-resume switch.

Fresh timing models use backbone/data seed1042 and proxy seed43 consistently;
trained numerical source uses original proxy seed1043. Full original28600/1430
schedule, micro16/GAS4/8GPUs, seq2048/FA4/EOS/isolation/reset RoPE retained.
Queue finished23:06:44UTC; communicating burns restored23:07:55UTC. Final read-only
receipt3e95eae at23:11:51UTC verifies all8 burns100%, workers133155–133162, advancing
collective progress. Final4 CPU tests passed8.681s. No package/driver changes.
Artifacts SHA256 verified and archived in artifacts/p6-compile-20261011/.
Report:docs/P6_COMPILE_TRIAL_20261011.md. Source/root successful launch5b03fd6,
q359-p6-compile-20261011-a02. Initial failed root preserved (missing W&B in our
incorrectly selected eval environment, before training); retry used existing
attention_bench. commands.sh now#0; no further experiments queued.

## P6 compilation gate passed; fresh timing retry — 11 October 2026

GPU numerical results: cosine-only passes all four 64/2048-token/checkpointing
cases. LM outputs, backbone/gate gradients and normalization buffers EXACT;
maximum predictor-gradient relative L2 0.00044610 (0.0446%). Gate/code-normalization
compilation rejected: hidden relative L2 up to0.00411, gradients up to0.01932.
Only cosine proceeds to timing. Original checkpoint SHA unchanged.
Initial A timing job failed BEFORE training: eval_fa4 lacks wandb. Our launcher
selected the evaluation environment incorrectly. Use existing original training
attention_bench, verify wandb0.30.0 before GPU reclamation, reinstall nothing.
All8 burns restored and verified in health-a02 (workers110063–110070).
Fresh retry q359-p6-compile-20261011-a02 repeats gates in the training environment
and six100-step timing runs. First root/results/cache preserved.
Evidence:temp/remote_logs/p6-compile-health-a02.log. Submission pending.

## P6-iso speed trial authorized — 11 October 2026

User requested step1: reduce P6-iso overhead with correctness checks and real
training timing. Experimental function-only compilation; production defaults
and train.py/model untouched. Plan/local evidence: P6_COMPILE_TRIAL_20261011.md.
Fresh root q359-p6-compile-20261011-a01, source P6iso checkpoint10000 read-only.
Eight CUDA numerical cases gate six100-step eight-GPU Trainer timings (A/eager
P6/compiled cosine, reverse-order repeats), original schedule/data recipe.
Existing safe supervisor restores communicating burns on completion/failure.
No production continuation or environment changes. Submission pending.

## Recurrent real-target evaluation completed — 11 October 2026, 05:45 SGT

Retry218f166 passed six CPU tests, GPU smoke and128-row full-length diagnostic.
Both jobs exited0; finished05:42:20SGT. Burn restoration verified05:43:31SGT;
read-only receipt790a18d at05:45:57 confirms all8 burns100% busy, workers106225–106232.
Original checkpoint SHA and in-memory weights/normalization buffers unchanged.
Summaries and eight per-rank reports downloaded with verified SHA256; all128
unique row records reproduce aggregate scores.261712 scored tokens,2048 context:
parallel original P6 PPL20.4195259; native sequential20.4198265;
past-only predicted memory21.7096859; past-only real-target memory21.7203633.
Experimental pass uses NO learned predictor or current-token proxy injection.
The two past-only passes share the same update rule and differ only in memory
content. Real targets are slightly worse (+0.049% PPL); no evidence of benefit
from this intervention. This does not prove predictor fidelity because the
inference rule differs from training. Not the full10M validation dataset.
Artifacts:artifacts/recurrent-p6-20261011/{summary.json,paired-rows.json}.
Report:docs/PROXY_RESULTS_SUMMARY.md (new final section).
Evidence:temp/remote_logs/q359-recurrent-health-a05.log. commands.sh now#0;
no additional experiments queued. Source weights and original outputs preserved.

## Recurrent diagnostic retry — 11 October 2026

First launch79a2763 passed CPU preflight, copied/verified Accelerate, reclaimed
only approved burn workers and verified all8GPUs free. GPU smoke failed BEFORE
scoring: TrainingArguments device setup reset an already-created Accelerator.
Fixed driver setup order: TrainingArguments/device first, Accelerator second.
Added a real CPU runtime regression; model/target/cache code and numerical gates
unchanged. First output is preserved. All8 communicating burns restored and
verified in temp/remote_logs/q359-recurrent-health-a02.log, workers104874–104881.
Fresh retry root:/mnt/local/_outputs/deep-llms_th2/q359-recurrent-p6-20261011-a02.
Same smoke and128-row diagnostic; original trained checkpoint untouched.

## Authorized recurrent real-target evaluation — 11 October 2026

User explicitly chose real targets only: no current-token learned proxy in the
experimental pass. Implemented evaluation-only recurrence in eval/recurrent_p6.py;
production model/training are unchanged. Process token t with native current
residual and all available past K/V. After completing its decoder, sum MLP
outputs l..l+3 for each P6 site, use checkpoint-frozen normalization/clipping,
then original gate/RMS/residual injection to rebuild only token t's cached K/V
from its saved pre-injection residual. Publish for future tokens only; no replay.
Targets are generated by this same recurrence; document caches are isolated and
RoPE resets at boundaries. Real mode never invokes the learned predictor.
Controls: parallel FA4 original P6; sequential original P6; past-only predicted
memory (same removal of current injection as experimental real-target memory).
Cached paths use explicit rectangular SDPA; native controls gate numerical error.
This tests a changed inference rule, not an oracle bound or pure predictor error.

Five CPU tests passed (4.308s): full-depth FP32/BF16, native equivalence, explicit
attention oracle, target formula/KV publication, no learned predictor in real
mode, state preservation, zero gates, document reset, future-token causality.
Local log:temp/recurrent-p6-cpu-tests.log. Launcher repeats tests in eval_fa4,
checks exact checkpoint SHA, copies/verifies Accelerate config and uses existing
verified-worker burn reclamation plus free-GPU check and automatic burn restore.
Eight-GPU smoke:8rows x64tokens; diagnostic:128 deterministically spread original
validation rows x2048tokens (262144 input tokens, not full10M validation).
Source:P6-iso checkpoint10000, seed1042, SHA3e04cc0a27f1e0d5a1e97efdc1e34c11d4e821f9b34fb76898a0652fd8d28442.
Output:/mnt/local/_outputs/deep-llms_th2/q359-recurrent-p6-20261011-a01
Launch pending; no optimizer, no checkpoint writes, no library/environment changes.

## Downstream evaluation completed — verified 11 October 2026, 02:56 SGT

Receipt2052a68 confirms all18 outer stages completed successfully, including
all few-shot validators, six fine-tuning studies, and final cross-checkpoint
summary. Queue finished02:38:15SGT. Supervisor returned0, confirmed GPUs free,
and restored communicating burns02:39:25SGT. At02:56 all8 approved burn workers
102775–102782 remain at100% utilization. No further evaluation queued.
Final summary downloaded and checksum verified against run.json artifact;
local temp/q359-downstream-results/downstream-summary.json. All six per-study
summaries also saved locally. Source checkpoints remain unchanged.
Three-seed held-out means:
| Source | PAWS accuracy | NLI accuracy | STS-B correlation | BoolQ accuracy |
|---|---:|---:|---:|---:|
| A-10000 | 0.932333 | 0.836926 | 0.843316 | 0.739653 |
| P6-iso-10000 | 0.926333 | 0.839454 | 0.844724 | 0.722630 |
| A-10800 | 0.928667 | 0.840452 | 0.838765 | 0.735678 |
STS-B correlation is average Pearson/Spearman. These are three fine-tuning
seeds from one pretraining seed, not evidence of statistically significant
method superiority. P6-iso has mixed downstream results, no consistent win.
Evidence:temp/remote_logs/q359-downstream-health-a09.log. commands.sh now#0.

## Downstream progress — verified 11 October 2026, 02:11:40 SGT

Read-only receipt e5b7cd3: all few-shot evaluations and five of six fine-tuning
studies completed and validated. A-10000 and P6-iso-10000 are fully complete.
A-10800 PAWS/NLI completed; its final STS-B/BoolQ study is running initial gates.
No reported failures; final combined validation and automatic burn restoration
remain pending. Based on the earlier A extended study (~28 minutes), approximate
completion is02:40SGT, subject to remaining fits and validation.
P6-iso three-seed means: PAWS0.9263333, NLI0.8394544,
STS-B average Pearson/Spearman0.8447241, BoolQ0.7226300.
A-10800 available means: PAWS0.9286667, NLI0.8404524.
Completed summaries saved in temp/q359-downstream-results/finetune/.
Evidence: temp/remote_logs/q359-downstream-health-a08.log. commands.sh returned
to #0; persistent queue continues. The brief idle GPU snapshot is between gates,
not evidence of final completion. Next check final summary and restored burns.

## Downstream progress — verified 10 October 2026, 16:00:40 UTC

Receipt 67b5adc: few-shot remains complete for all three sources. A-10000 has
completed both fine-tuning groups, all tests and summaries passed. Three-seed
held-out means: PAWS accuracy0.9323333; NLI accuracy0.8369261; STS-B average
Pearson/Spearman0.8433158; BoolQ accuracy0.7396534. Summaries pulled into
local temp/q359-downstream-results/finetune/A-10000/.
P6-iso-10000/pairs is active, starting its second NLI LR search after the first
finished successfully. Eight distributed workers are loading the next fit;
zero utilization in that startup snapshot is not a completion signal.
P6-iso/extended and both A-10800 groups remain queued. No reported failures.
Evidence: temp/remote_logs/q359-downstream-health-a07.log. Commands returned to
#0; persistent supervisor continues and will restore burns at termination.

## Downstream progress — verified 10 October 2026, 15:02:44 UTC

Read-only receipt 63aa1ab confirms full few-shot evaluation completed for all
three checkpoints; 5/10/25-shot validators passed. Results pulled into
local temp/q359-downstream-results/. Fine-tuning remains active in the first
nested study, A-10000/pairs. PAWS confirmation fits and NLI seed43 finished;
NLI seed44 was progressing from epoch0.87 to0.92 between snapshots, finite
losses, all eight GPUs active. No final held-out fine-tuning summary yet.
Remaining A-10000/extended, P6-iso-10000/pairs+extended, and A-10800/pairs+extended
are queued. Supervisor remains active; burns are reserved for queue termination.
Evidence: temp/remote_logs/q359-downstream-health-a06.log. No training state
changed by this check. commands.sh returned to #0; persistent queue continues.

## Downstream queue verified running — 10 October 2026, 14:07 UTC

Launch 731e752 is active in persistent tmux on q359. Status receipt 4cec904
confirmed all three source-checkpoint CUDA numerical gates and all 5/10/25-shot
smoke validators passed. Full few-shot evaluation is running; the six nested
fine-tuning studies follow automatically. All 20 CPU tests passed in eval_fa4.
Dataset, split and demonstration audits matched the established references.
Accelerate configuration was copied and verified; only identified burn workers
40498–40505 were stopped, and all eight GPUs were confirmed free before launch.
Original checkpoints remain unchanged. All fine-tuning epoch model checkpoints
are retained. Supervisor restores communicating burns on completion or failure.
Evidence: temp/remote_logs/q359-downstream-health-a05.log.
Root: /mnt/local/_outputs/deep-llms_th2/q359-downstream-10k-10800-20261010-a01
Completion artifact: supervised/run/downstream-summary.json. No final downstream
results yet. commands.sh is inactive (#0); the persistent queue continues.
Next check: full few-shot validation and nested fine-tuning progress.

## Authorized downstream comparison of A-10k, P6-iso-10k, A-10.8k — 10 October 2026

User requested few-shot evaluation and supervised fine-tuning for these three
seed-1042 checkpoints. Original checkpoint files remain read-only. Retain the
established English protocol: HellaSwag 10-shot, ARC-Challenge 25-shot, and
XNLI/XStoryCloze/PAWS-X/PIQA/ARC-Easy/Winogrande 5-shot, demonstration seed 42.
Belebele excluded because its available demonstration pool overlaps evaluation.
Fine-tune PAWS-X, NLI, STS-B, BoolQ using the existing task-only HF Trainer:
LRs 1e-5/3e-5 selected on development with seed 42, selected LR repeated with
seeds 43/44, then final held-out tests. All fits use eight GPUs, same FA4/BF16,
512-token context and global batch 128; no proxy auxiliary loss or normalization
updates. Keep all epoch checkpoints (model-only, as the established protocol).
48 production fits plus numerical gates, smoke/reload checks and final testing.

Benchmark download8729087 retrieves 36 pinned files (110,765,643 bytes) through
controller #d. Existing two manifests/paths reused. No remote direct network.
New composition helper reuses the established evaluators and fine-tuning study,
with separate named study roots for duplicate architecture A at different steps.
Explicit source-step checks replace hardcoded 2500; few-shot validation checks
exact path/arm/step/weight SHA256 plus identical prompt hashes across models.
Global summary verifies matched fine-tuning data/order across all checkpoints.

Local 19 checks passed in the pinned sampling_b200 environment; harness prompt
test needs lm_eval, absent there. Complete 20-test suite must pass in eval_fa4
on B200 before GPU reclamation. Dataset/prompt audits must match existing
references. Burns continue during controller downloads and CPU checks; then
copy/verify Accelerate config, reclaim only known burns, require free GPUs,
run the gated queue, restore communicating burns on success/failure.
Root: /mnt/local/_outputs/deep-llms_th2/q359-downstream-10k-10800-20261010-a01
Launch verified running; see the newer status above. Original outputs unchanged.

## A time-matched continuation completed — verified 10 October 2026, 21:33 SGT

A reached10,800 updates; staging, fit, validator and timing report all exited0.
Fit finished20:54SGT; communicating burn restored20:55SGT. Read-only checkbf08a3a
at21:33SGT confirmed all8 burns100% busy, PIDs40498–40505, collective cycles
2490→2510. All expected continuation checkpoints10000,10250,10500,10750,10800
are present; original10k outputs unchanged. Retention fix passed final checks.
A cumulative Trainer runtime22,758.6961s vs P6-iso22,743.7338s:14.9623s or0.0658%
longer. A10800 loss3.007632362511146/PPL20.2394235344; P6-iso10000
loss3.019880417442181/PPL20.4888414296. Thus A is better at approximately equal
training time for this seed1042. A10000 PPL20.5686746736 remains available.
Evidence:temp/remote_logs/q359-A-time-match-health-a02.log; extracted results
in temp/q359-A-time-match-results/. commands.sh inactive; no training queued.

## A time-matched continuation verified running — 10 October 2026, 20:26 SGT

Launch1ec24d8 passed the real-entry CPU retention/resume equivalence test and
queue/report tests in B200's pinned environment before touching GPUs. Accelerate
config was copied and rechecked. Only verified burn workers32773–32780 stopped;
all8GPUs free at12:24:27UTC. Existing staging helper copied/checksummed16 files
from original A/checkpoint-10000; original10k A/P6-iso outputs remain untouched.
Read-only receipt74b661e confirms step10019→10026 in15s, finite losses3.029/3.042,
~2.06s/update, all8GPUs97–99%,110232MiB each. Resume transition contains ONLY
save_total_limit2→0; save_steps250, seed/data_seed1042, schedule28600,warmup1430,
ignore_data_skip=false. Target10800; validator and actual cumulative-runtime
comparison follow training. Supervisor restores communicating burn automatically.
Evidence:temp/remote_logs/q359-A-time-match-health-a01.log. commands.sh now#0;
persistent training continues. New periodic checkpoints are all retained.

## Authorized A continuation for matching P6-iso training time — 10 October 2026

User requested fixing retention then resuming A to match P6-iso compute time.
Target10,800 updates: round(10,000 *22,743.7338/21,059.0039), using measured
Trainer runtime. Preserve full28,600 schedule/1,430 warmup, seed1042, FA4,
all8GPUs, micro16/GAS4 and data order. Only retention changes2→0; save every250.
Source10k A is copied with all model/optimizer/scheduler/RNG files and checked
SHA256, into a fresh run; original results and checkpoints remain untouched.
Resume guard now allows/logs only retention changes without performance override;
scientific/data/schedule changes still fail. CPU real-entry retention test on
B200 gates GPU reclamation (dev env cannot run it: Transformers version mismatch).
Local queue tests pass. Copy/Accelerate config+env verification and existing
supervisor restore burns after success/failure. Final report includes measured
cumulative runtime difference;10,800 is an estimated time match, not exact.
Root:/mnt/local/_outputs/deep-llms_th2/q359-A-time-match10800-seed1042-20261010-a01
Submission/startup pending. Do not infer launch success from controller receipt.

## Checkpoint retention correction — 10 October 2026

User wanted all periodic checkpoints retained. The completed A/P6-iso10k
launch incorrectly inherited save_total_limit=2; the earlier claim that this
was agreed was incorrect. Verified remote listings contain only9750/10000
for each arm. Older deleted checkpoints cannot be restored by a config change;
no backup of them has been verified. Production recipes now explicitly use
save_steps=250 and save_total_limit=0 (HF Trainer disables checkpoint rotation).
The10k queue guard/test enforce unlimited retention. Smoke-test retention is
unchanged. These changes apply to future launches, not the completed run.
No rerun/resume or GPU workload is authorized by this correction alone.

## A/P6-iso 10k queue completed successfully — verified 10 October 2026

Delayed status requeste6f9beb returned at19:49SGT on10October. Both fits,
both validators, and matched comparison exited0. Each arm stopped as requested
at10,000 updates/10,485,760,000 input tokens, preserving28,600-step LR schedule.
A final eval LM loss3.023769271527453 (PPL20.56867467); P6-iso3.019880417442181
(PPL20.48884143), approximately0.388% lower PPL at equal updates, one seed1042.
Both evaluated4,882 packed rows/9,981,660 target tokens. P6-iso no-proxy loss
3.0842712254000397; disabling its proxy increases loss by0.06439080795785879.
Trainer runtime A21,059.0039s (~5h51m), P6-iso22,743.7338s (~6h19m);
job wall time including setup A22,119.31s, P6-iso22,810.66s.
A finished05:37SGT, P6-iso11:57SGT; burn restoration verified11:58SGT.
Checkpoints9750/10000 retained for both. Latest19:49SGT snapshot confirms
burn PIDs32773–32780 at100% utilization/155212MiB each; communicating burn
log has advancing cycles30610→30720. Supervisor training_status=ok, returncode0.
Evidence:temp/remote_logs/q359-10k-health-20261010-a02.log; extracted result
JSONs in temp/q359-10k-results/. commands.sh reset to#0. No further training queued.

## Live production health verified — 10 October 2026, 01:58 SGT

Read-only inspection1d76e2e confirmed A advancing from step3780 to3787 in15s;
stop target remains10,000 (progress bar denominator28,600 is the full LR schedule).
Recent loss3.30–3.33, grad norm0.13–0.15, LR~2.951e-4, steady update~2.06s.
All8 B200 GPUs97–99% busy,110448MiB each, same owned training PIDs10958–10965.
Checkpoints3500/3750 present, latest trainer_state at3750;23TiB disk available.
Supervisor/queue/tmux live, P6-iso still queued. No errors/nonfinite metrics in
inspected recent logs. Evidence:temp/remote_logs/q359-10k-health-20261010-a01.log.
No training configuration or running processes changed. commands.sh reset to#0.

## Fresh A/P6-iso 10,000-step production queue on q359 — 9 October 2026

User requested continuing the previously planned experiments after both 50-step
smokes passed. Submit fresh A then P6-iso, seed1042/proxy1043, 10,000 updates each,
full28,600 schedule/1,430 warmup, LR3e-4, all8GPUs, micro16/GAS4, sequence2048,
1,048,576tokens/update, FA4 document isolation/reset RoPE/EOS. No smoke checkpoints
or short schedules are reused. Executed with save every250 and retain2; this was an agent error, corrected below in the newer retention note.
Five stages: A fit/validate, P6-iso fit/validate, matched comparison.
Output root `/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01`.
Launcher now takes an explicit verified hostname instead of assuming the retired
node. It verifies the Qwen asset hashes and all352 English release files
(134,739,753,089 bytes) against dataset_manifest.json from public release
2c425f4e1d4467008a0afce00de0ae8a33c8e834, whose SHA256 is
`dbba73b7a95ebc530297bac5a2151be6292f080e80e9b3101302a905806dde9a`.
Checks fresh-cache disk space, copies/verifies Accelerate config and checks
versions/ownership before the existing burn-restoring supervisor reclaims GPUs.
Production data processing remains inside train.py with160 workers and shared
HF caches. Launch5920a03 was verified by startup inspectionbb71213:
all352 English files and all Qwen assets passed hash checks; Accelerate config
was copied and verified. The supervisor stopped verified burn workers9776–9783,
recorded all8GPUs at0MiB/no compute PIDs at15:28:31UTC, and started A.
At15:31UTC A was tokenizing the full36,595,514-document training split;
optimizer updates had not yet started. P6-iso is queued after A validation.
Supervisor PID10598 and tmux session were live; automatic communicating-burn
restoration remains armed for queue success/failure. Evidence:
temp/remote_logs/q359-10k-startup-status-a01.log.
Local queue/validator tests (3) and launcher shell/embedded-Python syntax checks
passed. commands.sh is now inactive (#0); the persistent queue keeps running.

## A/P6-iso 50-step full-model smoke passed — 9 October 2026

Launch61e761a finished both train.py fits and both checkpoint validators with
exit0. Final verificationfbb56e4 returned `BOTH_50_STEP_SMOKES_AND_LIVE_BURN_VERIFIED`.
Evidence: temp/remote_logs/q359-training-smoke50-status-a04.log; extracted result,
validation and loss-history JSONs in temp/q359-training-smoke50-results/.
All8GPUs, full28-layer model, micro16/GAS4,2048,FA4: both ran50updates/52,428,800
input tokens. Final held-out LM loss: A7.59823, P6-iso7.69922 (128 packed rows).
First/last logged training loss: A11.46297→7.61298; P6-iso11.48109→7.72558.
P6 auxiliary loss0.92603→0.15165; lambda reaches0.1. Steady step times about
2.05s A/2.22s P6-iso; peak allocated memory103.62/109.38GiB. No OOM/nonfinite
training failures. Validators checked step50, actual FA4 forward/backward,
optimizer/scheduler/all8RNG artifacts, normalization/gates and finite eval.
Matching train/eval fingerprints:127b9996aba89b6f/3718bcedcdba1921.
Both fits finished at15:04:19UTC; supervisor verified communicating burn at
15:05:30UTC. Follow-up verified workers9776–9783, enabled guard and advancing
collective cycles130→150. commands.sh is inactive; no scientific run launched.
These are functional smoke results with schedule100/warmup5/proxy-warmup10
and disposable text subsets, not a method ranking or long-training result.
Checkpoint presence/state was checked; this smoke did not test resumption.

## Authorized 50-step real-training smoke on q359 — 9 October 2026

User accepted a real training-entry smoke and allowed 50–100 steps. Queue runs
A then P6-iso for 50 updates each, seed1042 (proxy1043), all eight GPUs, full
28-layer model, microbatch16/GAS4, sequence2048, FA4 document isolation with
EOS boundaries/reset positions and decoder/LM/aux checkpointing disabled.
Disposable raw-text subsets: first300k train/2048 validation documents;
train.py performs normal tokenization/packing with shared HF caches in this
smoke folder. Smoke-only schedule100/warmup5, proxy warmup10, save/eval every25,
logging every5, final eval128 rows. Production recipe remains unchanged.
Each arm is checked by the existing validator with explicit schedule/warmup
expectations; checks include finite losses, actual FA4 forward/backward receipts,
optimizer/scheduler/all8 RNG files, target statistics/gates and memory usage.
CPU validator tests and queue/shell/Python syntax checks pass locally.
Output root: `/mnt/local/_outputs/deep-llms_th2/q359-proxy-training-smoke50-20261009-a01`.
Launch copies/rechecks Accelerate configuration, freshly identifies the reviewed
burn workers, and uses the existing supervisor to reclaim GPUs and restore the
communicating burn after success/failure. No long scientific run is included.

## Qwen assets verified on q359 — 9 October 2026

Download 99c251c completed all three requested files. CPU-only verification
08db35e checked every SHA256 against resources/qwen3_base_assets.json and loaded
AutoConfig/AutoTokenizer offline. Qwen3: 28 layers, vocabulary 151936, tokenizer
EOS 151643. Receipt: temp/remote_logs/q359-verify-qwen-assets-a01.log, Dropbox
14:51:16 UTC. These config/tokenizer assets support the existing random-weight
training recipe; pretrained weights were not requested for that recipe.
No further GPU smoke or training was launched. commands.sh is inactive.

## Qwen assets download on q359 — 9 October 2026

User authorized downloading Qwen assets. Controller request reuses the three
pinned files in `resources/qwen3_base_assets.json` (Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`): config.json, tokenizer.json and
tokenizer_config.json, at the existing recipe model directory. The current
recipe initializes random weights and needs these config/tokenizer files.
Download completion, SHA256 matches and offline loading must be verified.
User asked whether another smoke is needed: recommend only 2–3 updates through
the actual training entry to cover data packing, auxiliary loss, memory and
checkpointing; no additional GPU test is launched by this download request.

## q359 smoke completion and live burn confirmed — 9 October 2026

Read-only verification 25ca059 returned `FA4_SMOKE_AND_BURN_HANDOFF_CONFIRMED`.
The smoke job exited zero in 34.7 seconds; saved queue completion and supervisor
receipts agree. At 14:39 UTC, verified burn workers 6044–6051 were active on all
eight GPUs, with rank-0 collective cycles advancing 390 to 400 during the check
and the GPU guard enabled. Evidence: `temp/remote_logs/q359-smoke-final-verification-a01.log`.
The tmux exit-status watcher error did not affect the completed smoke or burn.
`commands.sh` is inactive after verification. Remaining full-model preparation:
download the pinned Qwen config/tokenizer, then test the actual training entry.

## q359 FA4 runtime smoke passed (9 October 2026)

Launch 6eb7654 ran the real pinned FA4 kernel on all eight B200 GPUs with
synthetic 2048-token inputs, 16 query/8 KV heads and head dimension 128.
Relative L2 errors against FP32 math SDPA: output 0.00197505, Q gradient
0.00250610, K gradient 0.00248637, V gradient 0.00228929. Exact document
isolation passed. Three Accelerate/DDP optimizer steps (accumulation 2)
produced identical gradients and updated weights across ranks.
The supervisor recorded training_status=ok and verified eight-rank burn
collective progress at 2026-10-09 14:33:33 UTC. Evidence:
`temp/remote_logs/q359-fa4-runtime-smoke-a01-full.log`.
The outer log watcher then errored on an empty tmux pane exit-status field;
this happened after test completion and burn handoff. A read-only follow-up
verifies saved completion receipts and advancing live burn cycles directly.
No kernel/full-training code was changed; full-model smoke still needs local
model configuration/tokenizer assets.

## q359 FA4 runtime smoke — 9 October 2026

User accepted the short FA4 forward/backward and distributed smoke test.
Preflight fbf0c00 verified runner burn workers 499–506 under launcher 432,
with known source/guard hashes. The existing supervisor rechecks identities,
stops only approved workers, verifies free GPUs, and restores the communicating
burn after success/failure. Runtime test uses synthetic 2048-token inputs,
16 query/8 KV heads, head dimension 128, BF16 FA4 versus FP32 dense attention,
exact document isolation, and three Accelerate/DDP updates with accumulation 2.
Accelerate config is copied from resources and verified with accelerate env.
This is a kernel/distributed runtime test, not full-model training.
Output root: `/mnt/local/_outputs/deep-llms_th2/q359-fa4-runtime-smoke-20261009-a01`.
The model config/tokenizer paths are absent; download them before full-model work.
Dataset download and all English Arrow split loads passed: 35 train shards /
36,595,514 rows and one validation shard / 11,822 rows (c15510f).

## Prepared dataset download on q359 — 9 October 2026

User authorized downloading `nht10/cx_sampled_old`. The controller `#d +a`
request uses the same destination as historical commit `2a334dc`:
`/mnt/local/_data/@PROJECT@/cx_sampled_old`. It matches the English train and
validation paths in `proxy_heads.b200.json`. Download the existing prepared
text pool; no sampling or training is requested. Completion and local file
integrity remain to be verified from the new machine's receipts and manifest.

## All four environments verified on q359 — 9 October 2026

Verification job `th2-q359-verify-envs-20261009-a01`, commit `fd5e5f9`,
returned `ALL_FOUR_ENVS_VERIFIED` from `thiennh-p6-q359-worker-0`.
Evidence: `temp/remote_logs/q359-verify-envs-a01.log` (Dropbox snapshot
2026-10-09 13:20:18 UTC). All pins match and pip check passes in train_env,
eval, attention_bench, and eval_fa4. Python 3.11.15; torch 2.14.1/CUDA 13.0;
eight visible GPUs and sm_100 support; distributed/NCCL available. Training
and lm-eval imports pass, as do FA4 4.0.0b33/CUTLASS 4.8.0 imports in the two
FA4 environments. This verifies installation/imports and CUDA availability,
not FA4 kernel execution or distributed training. No GPU workloads were stopped.

## New machine read-only runner check — 9 October 2026

User provided the new q359 Dropbox share (stored only in ignored local config)
and authorized a simple `nvidia-smi` submission to test runner/Dropbox response.
Job `th2-q359-nvidia-smi-check-20261009-a01` prints UTC time, hostname, GPU status,
and `TH2_Q359_NVIDIA_SMI_CHECK_OK` on success; `#1 +60+a` requests full logs.
No environment activation, process termination, or training is involved.
The prior install commit d11140d has a STARTED receipt; completion is unverified.
Verify the new job's exact receipt and output in the q359 share.

## New machine environment installation — 9 October 2026

User reports a replacement machine and authorizes environment installation via
the existing `deep-llms/th2` execution repository. Prepared controller request
`th2-new-machine-install-envs-20261009-a01` installs `train_env`, `eval`,
`attention_bench`, and `eval_fa4` with full logs. The FA4 training specification
now explicitly pins CUTLASS DSL 4.8.0, matching the validated runtime and launch
preflight. Other package versions are unchanged.

Installation success is pending runner logs and runtime verification. Obtain
the new Dropbox folder and verify each environment's imports and CUDA runtime
before proceeding. This request does not launch training. The old-node hourly
Dropbox monitor still targets the old share; it cannot establish new-node
status. Old hostname-specific launch/stop scripts must not be used unchanged
on the replacement machine. The authorized fresh 10k A/P6-iso pair remains
pending environment, dataset, machine identity, and GPU preflight checks.

## Historical: authorized replacement with fresh 10,000-step A/P6-iso pair — 8 October 2026

An **hourly dev-machine Dropbox monitor** is installed in user crontab at minute
0, using `scripts/monitor_b200_dropbox.py`. Its ignored state is
`temp/b200_hourly_monitor/status.json`; the first successful read at14:30 UTC
still reports the same13:40 UTC controller SSH DNS failure and no later runner
success. The cron job is read-only and does not submit runner commands or launch
training. Dropbox receipts are passive, so lack of new files cannot prove that
the B200 is still down, and a new file alone cannot prove GPU availability.
The fresh pair remains pending a verified runner recovery and live preflight.

The corrected recovery command was **not delivered to the B200**. Runner status
at13:40:32 UTC reports job `th2-tjx3-recover-A-P6iso-10k-launch-20261008-a02`,
commit75e0d47, `FAILED(rc=255): ssh: Could not resolve hostname <host>`.
This is a controller/SSH failure, not a project traceback. Per AGENT_GUIDE,
do not resubmit, kill, clean, or claim success until the operator restores
access. The earlier verified SIGTERM is known; its final burn handoff and any
new training launch remain unverified. The fresh five-stage queue is prepared
and locally tested but has no verified B200 execution. `commands.sh` is reset
to inactive `#0` while waiting for infrastructure repair.

At13:33 UTC the exact verified supervisor PID486855 received SIGTERM; the
controller log confirms the identity-checked signal. The first stop/launch
command remains in its handoff wait. Its receipt predicate mistakenly expected
the burn session with a `tjx3-` prefix, while the established supervisor uses
`proxy-followup-2500-20261008-a01-final-burn`. That predicate cannot complete,
so the first command will **not** launch new training. A corrected follow-up
command waits for the old supervisor's actual verified burn receipt and then
starts the fresh pair. Do not infer new training has started from the signal.

User changed priority: stop the current nine-run follow-up, preserve completed
results, and start fresh A and P6-iso at seed1042, each to10,000 total updates.
Both use all eight B200s sequentially, identical data/optimizer/full28,600-step
schedule and1,430-step warmup, FA4, and the same 1,048,576-token global batch.
P6-iso uses proxy module seed1043. This is fresh training, not a checkpoint
continuation. Keep save every250/retain2; no all-checkpoints policy was requested.

Read-only runner inspection 6fb8e9f at13:28 UTC found the nine-run queue still
running P7-simple-sparse. Supervisor PID486855 is the tmux pane process; all
eight GPU workers509914–509921 descend through its Accelerate launcher509908.
The first four P6-iso seed42 variants and validators are complete; remaining
stages have not been claimed complete. The supervisor/GPU identities must be
rechecked immediately before signaling. The replacement queue has five stages:
A, A checkpoint validator, P6-iso, P6-iso checkpoint validator, matched report.
The existing burn-restoring supervisor remains responsible for both handoffs.
No queue stop or new training launch has been verified yet.

## Nine-run follow-up status — 8 October 2026, 12:15:21 UTC

Read-only monitor438bdab: all four new P6-iso seed42 arms completed2500 updates
and passed independent checkpoint/backend/metric validation. New layernorm
result: NLL3.4661172128440887, PPL32.0122042481, Trainer95.87min. Earlier new
results: sparse32.1777, short32.1177, weighted32.0212. Original P6-iso32.0180;
layernorm is only0.018% lower, insufficient for a robust improvement claim.
All four use full4882-row validation; scores are in PROXY_RESULTS_SUMMARY.md.

P7-simple-sparse now training, latest visible step489/2500, ~2.23s/update.
All8GPUs active98–99%, workers509914–509921. Four fits remain queued:
P7-simple-short seed42, then A/P6-iso/P7-simple seed1042. No stage failed.
Automatic burn restoration remains configured. No workload stopped/restarted;
commands.sh #0 leaves the queue running. Do not relaunch.
Evidence:34 SHA256-verified artifacts in artifacts/proxy-followup-monitor-20261008-a04/;
all eight completed-job artifact hashes match queue receipts. Raw log SHA256:
14492d463097418b46e4e074634f3005c67b9b124829edd8cf77a176b5e58ae0.

## Nine-run proxy follow-up authorized — 8 October 2026

User authorized six new arms at seed42 and A plus the previous best P6/P7 at
second seed, all 2500 steps. Best is selected by prior validation PPL: P6-iso
and P7-simple. Second seed1042 follows the screening seed list; explicit proxy
module seeds43/1043, with Trainer/data/Python hash seeds42/1042 respectively.
Queue order: P6-iso-sparse, P6-iso-short, P6-iso-weighted, P6-iso-layernorm,
P7-simple-sparse, P7-simple-short (seed42), then A, P6-iso, P7-simple (seed1042).
Every fit uses all8GPUs; existing train.py/Accelerate, native FA4, micro16/GAS4,
2048 tokens, EOS/document isolation/reset positions, full28600/warmup1430.
No pretrained checkpoint resume: all nine are fresh from-scratch experiments.
Full4882-row evaluation; save250/keep2; checkpointing disabled as validated.

Existing sequential runner supports heterogeneous argv/arms/seeds. The thin
scripts/proxy_followup_queue.py builder produces20 stages:9 fits,9 individual
checkpoint/backend/metric validators,2 within-seed comparisons. It does not
pool different arms into a cross-seed report. One focused queue test passed;
production training/architecture/loss code unchanged. Fresh output root:
/mnt/local/_outputs/deep-llms_th2/proxy-followup-2500-20261008-a01.
Launch verifies pinned environment/data, copies/checks Accelerate config and
accelerate env, inspects/reclaims only approved burns, and verifies all GPUs
free. Existing supervisor restores communicating burns on success/failure.
Launch edfbf3f accepted. Monitor ed53943 at05:34:08 UTC verifies preflight,
Accelerate cache byte-copy plus accelerate env, known burn PID-only reclamation,
and all8GPUs free at05:32:14 before queue start. P6-iso-sparse is training through
step20 with finite loss/gradients (~2.17s/update), native BF16 FA4 train/eval
receipts, seeds42/42/module43, workers487207–487214 (97–99% utilization).
All9 generated CLIs also parsed successfully with the pinned local HF parser
in CPU mode; model settings resolved to the intended seeds. No cache/output
deletion. commands.sh #0 leaves the detached queue running; do not relaunch.
Evidence:10 SHA256-verified artifacts in artifacts/proxy-followup-monitor-20261008-a01/.
Log temp/proxy-followup-monitor-a01.log SHA256
c65b5e329ee8458b5d7e4d307942ba84a79528cbf7a661053f40ed3db5c2531e.
Next action: read-only status/results; supervisor restores communicating burns
after completion/failure. No completed research result yet.

## Authorized B200 proxy validation completed — 8 October 2026

All three requested stages passed: 36 exact old/optimized CUDA comparisons
(nine arms, math SDPA/deterministic FA4, checkpointing off/on), six 25-step
all-eight-GPU smokes, six verified checkpoint-24→25 resumes, and four paired
native-FA4 throughput measurements. Resume data/normalization/scheduler/RNG
matched exactly; parameter/optimizer residuals were below 9.50e-8 relative L2.
Update-25 data also matched across all six arms.

Measured throughput gains: A 0.03%, P6-iso 0.77%, P7-simple 1.11%, P7 0.92%.
One short pair per arm establishes only modest screening gains. P7 peak memory
rose from 116.3 to 118.5 GiB. Production defaults/environment/drivers unchanged.
All 31 continuation stages completed at 04:58:09 UTC. All eight communicating
burns restored and independently verified at 05:01:52 UTC. commands.sh inactive;
no long research runs launched. The logging-cadence checker was fixed without
retraining the completed smokes. See PROXY_B200_VALIDATION_20261008.md for results,
exact test modes, earlier failures and locally verified evidence.

Historical entries below describe the state at their respective checkpoints;
the completed CUDA validation above supersedes their pending-validation notes.

## Remaining exact proxy optimizations — 8 October 2026

Implemented items 1.2, 2.1 and 2.3 of PROXY_SPEED_OPPORTUNITIES_20261008.md:
forward-local typed convolution masks and slice accumulation, clipping counts
only on logging steps, and owned MLP-window storage reused after the second
contribution. No target, loss, parameter, normalization rule, checkpoint format,
attention kernel or training recipe change. Normalization moments and the same
distributed reduction layout remain active every step. Clipping diagnostics
are omitted when unmeasured, not reported as zero. Equivalent/fused/LSE/compile
proposals remain deferred pending their separate CUDA gates.

Forty-eight exact comparisons against saved pre-change code passed: twelve arms,
28 tiny blocks, FP32/BF16, SDPA/FA4 CPU oracle and checkpoint replay. Five expanded
optimization tests passed (32.199 s) and three new speed-safety tests passed
(6.265 s), including native BF16 convolution and actual Trainer/AdamW state
equivalence with clipping on/off, accumulation, first-step and epoch logging.
Non-step logging retains legacy per-step clipping collection.
Broader regression passed: 91 tests in 675.339 s, including CPU DDP, accumulation,
exact Trainer resume, earlier proxy heads, evaluation and fine-tuning. Total
99 tests passed across the selected invocations, plus the 48 full-depth cases.
Evidence: temp/proxy-safe-speed-20261008/.
No B200 access/deployment; full-size CUDA correctness/performance gates pending.

## P6/P7 execution optimization — 8 October 2026

Follow-up review found no production-code defect. Five optimization tests passed
in 23.695 s, including two new tests: exact old/new AdamW and normalization-state
trajectories through bootstrap, three updates and save/load; and FA4 checkpoint
replay for multiple outstanding forwards with different document/batch shapes.
FP32/BF16, CPU SDPA and FA4 oracle. Tests/docs only; no B200 access or deployment.
Evidence: temp/proxy-optimization-20261008/review-tests.log.

Removed repeated target-normalization work (batched scales, reused centered
values/token counts, skipped unused clipping counts), redundant P7 proxy-key
RoPE and repeated FA4 doubled-layout construction. Forward-local caches only;
no architecture, objective, parameter/checkpoint format, optimizer, data or
training-loop changes. All 16 full-depth before/after FP32/BF16 comparisons
matched every output and gradient exactly. All 94 tests passed across three
invocations: 3 focused optimization (8.655 s), 12 shared-head (11.481 s), and
79 regression tests (661.568 s), including CPU DDP, exact Trainer resume,
checkpoint evaluation and supervised fine-tuning. FA4 used the CPU oracle.

CPU operator counts confirm reduced work, but noisy timings do not establish
an overall speedup, especially for P7. Full-size CUDA/B200 numerical and timing
validation remains pending. No B200 access, deployment or environment change.
Details and evidence: docs/PROXY_OPTIMIZATION_20261008.md and
 temp/proxy-optimization-20261008/. New model variants remain local/untrained.

## P6-iso weighted and per-layer-normalized targets — 8 October 2026

Implemented P6-iso-weighted and P6-iso-layernorm as separate target-only changes.
Weighted uses coefficients 1.6/1.2/0.8/0.4 on the current and next three MLP
outputs. Layernorm applies parameter-free per-token hidden-channel LayerNorm
(epsilon 1e-6, FP32) separately to all four outputs before summing. Both then
use the existing running standardization/clipping and cosine auxiliary loss.
All twelve P6-iso locations, four-block windows (last window 24–27), predictor,
initialization, gate 0.1, isolated gradient routing and lambda schedule remain.
Both normalization-bootstrap passes use the transformed target; inference does
not construct targets. Metadata records the exact transformation and prevents
incompatible resume. Reports allow only the registered target differences.

Both use existing Trainer/Accelerate, SDPA/FA4, sequential eight-GPU queues,
checkpoint evaluation and optional supervised fine-tuning. The only train.py
change records the target metadata. No training-loop/data/packing/env change,
remote push, or B200 workload. CUDA validation and research runs are pending.
See docs/P4_P5_P6_IMPLEMENTATION.md for definitions and queue command. Generated
local review queue: temp/p6-target-variants-review-jobs.json (not submitted).

Local validation: all 55 regression tests passed in 403.788 seconds. Four
focused final checks also passed in 61.152 seconds. Coverage includes exact
full-depth targets and parent-equivalent initial LM outputs/gradients, FP32/BF16,
independent LayerNorm/bootstrap/loss/gradient calculations, document isolation,
recomputation/compilation, two-rank CPU DDP, accumulation, exact Trainer resume,
metadata tampering, sequential queues, checkpoint evaluation and task fine-tuning.
FA4 used the independent CPU attention oracle; actual CUDA kernels were not run.
Evidence: temp/p6-target-variants-regression.log and
 temp/p6-target-variants-final-targets.log. The first focused run exposed an old
test expectation (last target block 25) that was corrected to 27 for these
four-block, twelve-location arms before the passing regression.

## B200 STS-B/BoolQ completed — 8 October 2026, 00:01 UTC

Read-only monitor a825a1e on thiennh-p6-tjx3-worker-0 returned a fresh snapshot
at 2026-10-08 00:01:51 UTC. All 97/97 stages finished successfully: eight
numerical gates, eight smoke/reload/validation triples, 32 production fits,
eight development-only LR selections, 24 final evaluations and the summary.
The queue finished at 2026-10-07 21:21:03 UTC. Supervisor exit is zero; final
burn restoration was verified at 21:22:14 UTC. The fresh snapshot confirms the
same burn workers 401980–401987 on all eight B200s, each at 100% utilization
and 155212 MiB used; the burn log shows ongoing collective/GEMM progress.

Pulled 104 source-SHA256-verified artifacts into
artifacts/stsb-boolq-monitor-20261008-a10/. Additionally verified all 97 stage
artifact hashes against complete.json and summary means/stdevs against all
24 evaluation result files. Predictions remain on B200; the remote summary
stage recomputed their metrics and validated provenance/order. Final means:
STS-B correlation (Pearson/Spearman average) ×100: A 78.631, P6 80.003,
P6-iso 79.076, P7-simple 79.379. BoolQ accuracy (%): A 70.887, P6 70.387,
P6-iso 69.817, P7-simple 70.031. Three fine-tuning seeds, one pretraining seed;
P6 is strongest on STS-B, while A remains strongest on BoolQ.

Source log temp/stsb-boolq-monitor-20261008-a10.log SHA256
1c6445e0cda8b0e30d8e773fba9009a9fbd7c0bc7d13438bae312ef82b929846.
Monitoring used the status-only worktree; no new model code, training, stopping,
cleanup or environment changes were deployed. Remote main 03a96a2 deactivates
the acknowledged monitor (commands #0). Status-only commits merged locally.
The new P6/P7 variants remain local and untrained.

## P7 P6-variant follow-up review

No production-code defect was found. Added two independent full-depth regression
checks using 28 blocks with tiny hidden dimensions. In FP32 and BF16, on dense
SDPA and the CPU FA4 oracle, short exactly matches the parent's initial LM
hidden states and gradients; sparse exactly matches P7-simple with only the
omitted memory consumers bypassed. The independent cosine-loss calculation uses
nontrivial per-layer means/variances and clipping, confirms averaging over tokens
and active locations, and confirms that only predictor parameters receive
auxiliary gradients. Three full-depth checks passed in 9.575 seconds; strict
checkpoint reload and supervised gradient-routing checks passed in 3.995 seconds.
Logs: temp/p7-p6-review-full-depth.log and temp/p7-p6-review-reload.log.
The three integration checks (exact Trainer resume/report/queue, accumulation,
and screen-result validation) also passed in temp/p7-p6-review-integration.log.
That invocation included a misspelled evaluation test class; the correctly named
reload test passed in the separate two-test run above. Eight intended checks
passed across these runs; the previous complete 41-test regression also passed.
Actual FA4 CUDA kernels/full-size B200 validation remain pending. Existing
training code and B200 workloads were not changed during this review.

## P7-simple counterparts of the P6-iso variants — 8 October 2026

Implemented the two recent P6-iso ablations with P7-simple consumption:
P7-simple-sparse uses blocks 2,6,10,14,18,22 and four-MLP targets;
P7-simple-short keeps blocks 2,4,...,24 and two-MLP targets. Both retain four
query heads/two KV groups, a tokenwise isolated predictor, cosine-only loss,
auxiliary ramp to .1, no gate, and joint native/proxy attention. Sparse initial
predictor/projection weights match the retained parent heads. These are fresh
arms; no residual injection is added and the predictor remains auxiliary-only.

Training, queues, strict resume/report checks, evaluation and optional supervised
fine-tuning are integrated. Both SDPA and FA4 reuse the existing attention paths.
The CUDA checkpoint validator expects 34 attention calls for sparse and 40 for
short at full depth. Definitions: docs/P7_IMPLEMENTATION.md. Full-size CUDA
validation and research runs remain pending. No remote push or GPU workload was
launched; B200 continues its previously accepted evaluation queue. commands #0.

Local verification: all 41 regression tests passed in 353.541 seconds, including
full-depth target/initialization checks, gradient routing and accumulation,
two-rank CPU DDP, exact Trainer resume, report guards, evaluation and fine-tuning.
Three focused checks also passed (2.084 seconds). Logs:
temp/p7-p6-regression.log and temp/p7-p6-focused.log. Both attention interfaces
were tested on CPU; FA4 used an independent SDPA oracle, not its CUDA kernel.

## B200 STS-B/BoolQ fitting progress — 8 October 2026, 04:48 Singapore

Read-only monitor b855d9d verified the existing supervised-stsb-boolq-20261008-a03
queue at GPU time 2026-10-07 20:48:54 UTC (04:48:54 Singapore, 8 October).
67/97 stages complete, one running, no failures. All eight numerical gates and
all eight training/save/reload smoke validations passed. All 16 LR-search fits,
all eight dev-only selectors and 11/16 seed-confirmation fits completed: 27/32
production fits done. Every selector chose 3e-5. These are development choices,
not final test-set results.

Currently confirm-boolq-P6-44, after epoch 2, with workers 390794–390801 on all
eight GPUs; seven GPUs at 100% and GPU 0 at 0% in this instantaneous snapshot.
Four further fits follow: P6-iso and P7-simple on BoolQ, seeds 43/44. Then 24
final evaluations and the summary; no final evaluation files exist yet. Automatic
final burn restoration remains configured; training workers currently own GPUs.

Pulled 72 source-SHA256-verified artifacts into
artifacts/stsb-boolq-monitor-20261008-a09/. Source log:
temp/stsb-boolq-monitor-20261008-a09.log, SHA256
4551262176cd816c7ef0e885e8b33aa957ca861f1ddeb525e6983d18d4425f90.
Read-only deployment used /disk/thuat/deep2shallow-status, branch b200-status,
starting at deployed 4c570ea; no new P6 code was synced into the active queue.
Remote main is 8420052 with commands #0. Its status-only commits were merged
into local main; new P6 implementation/review commits remain local. No launch,
process stop, cleanup, or environment change on B200. Next: inspect the remaining
fits and final held-out evaluations.

## P6 variant follow-up review — 8 October 2026

Reviewed the P6-iso sparse/short implementation, gradient routing, target windows,
normalization, resume guards, queue arguments, and downstream wrappers. No
production-code defect was found; this review adds a stronger regression test.
The full 28-block test passed (2.997 seconds), using tiny hidden dimensions with
FP32 and BF16 on SDPA and the independent CPU FA4 oracle. Sparse matches the
parent with only excluded injection gates zeroed; short matches the parent's LM
output and gradients. Both produce the exact intended detached MLP sums at all
locations, ending at block 25, with the original auxiliary token denominator.

Resolved the previous missing-lm_eval test gap in ignored temp/p6-review-env,
a separate system-site-packages virtual environment over sampling_b200. Installed
lm_eval[hf]==0.4.10 there after checking the dry-run plan; existing training
environments and B200 were not modified. Core versions remain torch 2.14.0,
Transformers 5.9.0, datasets 4.8.5 and Accelerate 1.13.0. All 11 evaluation tests
passed in 75.652 seconds, including all four previously blocked tests. Together
with the prior 90 passes, the original 94-test set has no outstanding failures;
this is verification across runs, not a new combined suite invocation.
Evidence: temp/p6-full-depth-review.log, temp/p6-eval-review.log and
temp/p6-review-eval-install-plan.log. Full-size B200/CUDA validation of these
new variants remains pending. No new experiment or remote push in this review.

## P6-iso sparse and short-target variants — 8 October 2026

User requested implementation of the two proposed P6 follow-ups. New arms:
`P6-iso-sparse` retains six locations (2,6,10,14,18,22) and four-MLP targets;
`P6-iso-short` keeps the original twelve locations (2,4,...,24) and two-MLP
targets. Both retain P6-iso isolation, gate 0.1, auxiliary schedule, normalization,
and the existing train.py Trainer/Accelerate path. Sparse initialization matches
the parent's retained heads. Both support SDPA/FA4, sequential queues, strict
resume/target reporting, checkpoint evaluation and optional task fine-tuning.
No running queue or runner commands were changed; commands.sh remains #0.
Full-size CUDA validation and research runs for the new arms remain pending.
Definitions and usage: docs/P4_P5_P6_IMPLEMENTATION.md, P6-iso follow-ups section.
The prior B200 status below is historical, not a new monitoring result.

Local verification: 90 of 94 CPU tests passed in 481.898 seconds. All new-arm
checks passed: exact targets/placement and initial weights, LM/auxiliary gradient
routing, BF16/recomputation/compilation, document isolation, two-rank DDP, gradient
accumulation, exact Trainer resume, SDPA versus the independent CPU FA4 oracle,
checkpoint reload, queue arguments and report rejection checks. Four existing
evaluation-harness tests could not import the missing local lm_eval package;
no numerical/assertion failures were reported. The focused two-test run also
passed. Logs: temp/p6-variants-regression.log and temp/p6-variants-focused.log.
CUDA kernels were not exercised by these CPU tests.

## Runner recovered; numerical gates and smoke checks progressing — 8 October 2026

Read-only monitor 48e3b5e, GPU timestamp 2026-10-07 19:45:30 UTC
(03:45:30 Singapore, 8 October), verified the existing a03 queue is running.
The runner recovered and delivered the previously failed a07 monitor; no
training relaunch, process signaling, cleanup or numerical changes were made.

All eight fp32_task_v2 numerical gates passed. Five of eight arm/task pairs
completed distributed smoke training, reload and validation: A and P6 on both
tasks, plus P6-iso/STSB. P6-iso/BoolQ smoke was starting with eight ranks;
P7-simple smoke checks remain. Queue: 23 of 97 stages completed, none failed.
The instantaneous GPU snapshot caught the transition before workers attached;
it does not establish sustained GPU utilization. Full fitting and final
benchmark scores are not yet available. Automatic final burns remain configured,
not currently verified as running. Leave this queue in place.

Evidence: artifacts/stsb-boolq-monitor-20261008-a08/ (28 source-SHA256-verified
JSON artifacts), temp/stsb-boolq-monitor-20261008-a08.log SHA256
94e96b6e88734a2e061ca6626e0e1c14b48536465d516489da49cb66c0ef39f4.
Output root: /mnt/local/_outputs/deep-llms_th2/supervised-stsb-boolq-20261008-a03.
commands.sh returned to #0 after this read-only check. Next: inspect completion
of remaining smoke/reload checks and subsequent production fitting.

## Runner SSH/DNS failure; corrected queue status unknown — 8 October 2026

Final code correction and fresh retry pushed as d866142. Seventeen focused
CPU tests passed in12.556s (sampling_b200), including real Trainer reload and
classification offset-invariance/error-rejection checks. Production task model,
head precision, optimizer and data recipe unchanged. See SUPERVISED_FINETUNING.
Fresh output: supervised-stsb-boolq-20261008-a03 (97-stage queue).

Last successful monitor939da7f at03:28:50 Singapore: preflight and identical data
audit passed; Accelerate config copied/verified. Only verified burn workers
358890–358897 reclaimed at03:26:50; GPUs free at03:27:20. A/STSB gate passed;
A/BoolQ running. All-arm gates, distributed smoke/reload and production fitting
are NOT yet verified. Automatic final burns remain configured.

Monitor48861de (th2-tjx3-stsb-boolq-monitor-20261008-a07) failed at the runner's
SSH/DNS layer: `ssh: Could not resolve hostname <host>: nodename nor servname
provided, or not known` (rc255). Raw controller timestamp2026-10-07 12:31:35;
Dropbox status modified2026-10-07T19:31:38Z. Do not infer the GPU job stopped.
Evidence: artifacts/stsb-boolq-monitor-20261008-a06/ and
artifacts/stsb-boolq-monitor-20261008-a07-blocked/controller-error.json.
No resubmission, signaling, cleanup or backend/threshold change after this
infrastructure error. commands.sh restored locally to#0; no further execution
remote push attempted. Wait for operator repair, then read-only inspect the
EXISTING a03 queue before considering any new launch. Earlier a01/a02 output
roots are preserved and their failures are documented.

## Task-output acceptance correction prepared — 8 October 2026

Retry 969d197 failed before production at P7-simple/BoolQ real-input gate;
seven other gates passed. Raw logits relative error3.725%, but max probability
error .002215, CE difference .003893, hidden .825%, gradient1.502%. See
artifacts/stsb-boolq-monitor-20261008-a05/. Gate fp32_task_v2 replaces only the
classification raw-logit criterion with max probability difference<.01 AND
CE difference<.01 versus FP32 math; retain hidden<2%, gradient<5%, regression
output<2%, finite/gradient/buffer checks. Raw logits remain recorded. This is
an explicit metric correction, not proof of all long-run numerical behavior.
Training code/recipe unchanged. Fresh root supervised-stsb-boolq-20261008-a03
is being prepared; all eight gates and distributed smoke/reload still required.
Previous failures preserved. No new benchmark scores. More detail in
SUPERVISED_FINETUNING_20261007.md.

## STS-B/BoolQ FP32-reference retry prepared — 8 October 2026

User authorized fixing the failed acceptance checks. B200 diagnostic a9a8a75 /
monitor 671f089 reproduced the failure: A/BoolQ FA4 versus FP32 logits 1.067%,
BF16 SDPA versus FP32 2.271%; FA4 also passed on real training examples.
The gate now uses full FP32 math SDPA with TF32/autocast disabled as reference,
retains 2% logits/5% gradient limits and adds a 2% pooled-hidden check. Original
synthetic and real training fixtures must both pass. Training code/recipe are
unchanged. Sixteen focused CPU tests passed in 11.963 s in sampling_b200;
initial attempt in legacy train_env had an incompatible Transformers version.
No environments were modified. Evidence: temp/finetune-fp32-reference-tests-a02.log
and artifacts/finetune-numerics-monitor-20261008-a02/.
Fresh retry root supervised-stsb-boolq-20261008-a02, all 97 stages retained;
eight-GPU smoke/reload precedes production. Prior outputs preserved. Config
copy/accelerate env, exact data audit, identity-checked burn reclamation, free
GPU verification and automatic final burns remain required. Launch/production
status still needs verification. Full diagnosis: SUPERVISED_FINETUNING_20261007.md.

## STS-B/BoolQ stopped at numerical gate — 8 October 2026

Monitor `e525455`, 02:40:08 Singapore: queue failed at 01:59:06 Singapore,
before distributed smoke or any production fitting. A/STSB numerical gate
passed; A/BoolQ failed the FA4-vs-SDPA logit relative-L2 check: .02175061
(limit .02). Gradient relative L2 .01261112 passed its .05 limit. Inputs were
two synthetic sequences, not final benchmark examples. This is an acceptance
check failure; its underlying numerical cause has not yet been diagnosed.
No new STS-B/BoolQ scores and no changes to previously completed results.
Automatic communicating burn restoration verified at 02:00:17 Singapore.
Current workers 356295–356302 match that receipt, all eight GPUs at 100%;
burn log shows advancing cycles/collective payload. Seven source-SHA256-verified
artifacts: artifacts/stsb-boolq-monitor-20261008-a02/. Source log:
temp/stsb-boolq-monitor-20261008-a02.log, SHA256
8c211dd0343bda3c08c86eb8ede1d4280e46538e99760d4b014c3ffa3a1d79cb.
commands #0. Read-only check only; no relaunch, threshold change, process stop,
or cleanup. Next step is diagnose the backend comparison before a new launch.

## STS-B/BoolQ gated queue launched — 8 October 2026

Accepted launch `a76fb56`, root supervised-stsb-boolq-20261008-a01.
Monitor `982397e` at 01:59:05 Singapore confirms pinned environment/checkpoint
preflight and exact local/remote data audit passed. Accelerate config copied
and verified (8 GPUs/BF16). Only verified communicating burn workers reclaimed;
all eight GPUs free before queue start. A/STSB CUDA forward/backward gate
passed (FA4-vs-SDPA gradient relative L2 .00906345); A/BoolQ gate running.
Distributed smoke/reload and production fitting are NOT yet verified.
The accepted 97-stage detached queue automatically proceeds after its gates,
with automatic communicating burn restoration on either success or failure.
commands #0 leaves it running. Do not relaunch. Next action: read-only monitor
numerical gates, all eight task/arm smoke-reload validators, production progress.
Evidence: artifacts/stsb-boolq-monitor-20261008-a01/ (6 verified JSON artifacts).
Full protocol: docs/SUPERVISED_FINETUNING_20261007.md.

## STS-B and BoolQ fine-tuning authorized — 8 October 2026

User approved the recommended STS-B + BoolQ extension for A, P6, P6-iso,
P7-simple, starting from original seed42 step2500 pretraining checkpoints.
Two learning rates (1e-5/3e-5) on seed42, selected-rate confirmations43/44;
3 epochs per task, same eight-GPU HF Trainer recipe. STS-B scalar MSE head,
Pearson/Spearman reporting and their mean for selection. BoolQ cross-entropy
and accuracy. No SST-2/WiC or pretraining queued in this extension.
Final data manifest: resources/supervised_english_20261008_v2.json.
Following user clarification, STS-B uses sentence-transformers/stsb official
train/dev/test with public test labels (score 0–1 converted back to 0–5).
BoolQ uses public validation as final holdout and fixed grouped 10% of training
for development. Initial GLUE-only download 77a41a6 completed but is superseded;
no training used it. New data root supervised-english-20261008-v2.
All five final files downloaded successfully (62e4c50). Fourteen local tests
passed, including real train/reload, regression metrics and result validation.
Submitting 97-stage gated queue; production start still requires verification.
GPU burns stay running until verified reclaim after launch preflight. New root planned:
/mnt/local/_outputs/deep-llms_th2/supervised-stsb-boolq-20261008-a01.

## Task-specific few-shot evaluation complete — 8 October 2026

Monitor `d815726`, 01:28:45 Singapore: all 12 stages passed, including full
5/10/25-shot validators. Four original seed42 step2500 checkpoints, eight tasks,
90,784 scored examples; exact audited prompts match across arms, no truncation.
Queue finished 00:27:35 Singapore after 12 min 37 s (including GPU gates/smoke).
Descriptive eight-task mean (acc_norm where available, otherwise acc):
A 42.841%, P7-simple 42.702%, P6-iso 42.665%, P6 42.539%. A remains highest;
no established proxy advantage. Single pretraining/demonstration seed only.
Automatic communicating burns verified at 00:28:46; same workers 354407–354414
remain at 100% on all eight GPUs. Current log confirms advancing cycles and
collective payload. No evaluation/training queued. commands #0.
Evidence: 77 SHA256-verified artifacts in artifacts/fewshot-monitor-20261008-a01/.
Full per-task results: docs/DOWNSTREAM_EVAL_20261007.md. Do not relaunch.

## Task-specific few-shot evaluation running — 7 October 2026

Launch `686637b`; monitor `3999100` at 16:20:24 UTC verifies all four CUDA
numerical gates, smoke evaluations and three smoke validators passed. Full
evaluation running, GPU workers 353458–353465. A/P6/P6-iso/P7-simple original
seed42 step2500; HellaSwag10, ARC-Challenge25, other six tasks5. Exact matching
prompts, separate training demonstrations and zero truncation. No final scores.
Accelerate cache copied/verified. Reclaimed only verified burn workers
351289–351296 at 16:14:27 UTC; all GPUs free at 16:14:57. Automatic communicating
burn restoration remains configured after success/failure. commands #0 leaves
the detached queue running; do not relaunch. Evidence: 55 verified artifacts in
artifacts/fewshot-monitor-20261007-a04/. Root:
`/mnt/local/_outputs/deep-llms_th2/fewshot-2500-20261007-a01`.
Protocol and additional task shortlist: docs/DOWNSTREAM_EVAL_20261007.md.

## Task-specific few-shot evaluation authorized — 7 October 2026

User requested few-shot before further fine-tuning and clarified that shot
counts should follow task conventions. A/P6/P6-iso/P7-simple original seed42
step2500 checkpoints; HellaSwag10, ARC-Challenge25, Winogrande5; project-chosen
5-shot XNLI-en/XStoryCloze-en/PAWS-en/PIQA/ARC-Easy. No universal 1/5 sweep.
Belebele excluded (test-split demonstration pool). Fresh root fewshot-2500-
20261007-a01; numerical gates, prompt audits, smoke, full evaluation, automatic
communicating burn restoration. See docs/DOWNSTREAM_EVAL_20261007.md.
Launch accepted as `686637b`; receipt verifies pinned eval_fa4 environment,
benchmark hashes and copied eight-GPU BF16 Accelerate config. Remote prompt
audit started; GPU gates/full inference not yet verified. Read-only monitor
`f4298bf` submitted. No further weight updates authorized.

## P6-iso supervised evaluation complete — 7 October 2026

Monitor `6ef6e1d`, 23:41:08 Singapore: all 20 stages passed, including 8 full
fits and 6 final tests. Queue finished 23:21:40 Singapore (63 min 34 s).
Mean test accuracy ± sample SD across fine-tuning seeds 42/43/44: PAWS-X
90.717 ± .751%, English XNLI 78.955 ± .445%. Compared with A: -1.133 pp
PAWS-X, +.033 pp NLI; no convincing downstream gain. Both LR selections 3e-5.
Data/tokenizer and example-order hashes match A's completed evaluation.
Automatic communicating burns verified 23:22:46; same workers 351289–351296
active at 100% on all eight GPUs in current snapshot. No training queued.
Evidence: artifacts/p6iso-finetune-monitor-20261007-a03/ (26 verified JSON
artifacts). Full report: docs/SUPERVISED_FINETUNING_20261007.md.
commands #0. Do not relaunch without a new request.

## P6-iso supervised fine-tuning running — 7 October 2026

Launch `892fc80`, root
`/mnt/local/_outputs/deep-llms_th2/supervised-p6iso-20261007-a01`.
Monitor `001a821` at 14:22:39 UTC confirms CUDA numerical gate, eight-GPU
four-update smoke and strict reload all passed. Reload loss/accuracy, weight
hash and development-example-order hash match exactly. Production PAWS-X
seed 42 / LR 1e-5 is running on eight workers 343599–343606; finite logs
through epoch .3238. No final results yet. Queue: 8 fits, 6 final tests,
matched protocol; reuse completed A/P6/P7-simple comparisons.
Accelerate config copied/verified, environment/data/recipes checked. Only
verified burn workers 341142–341149 stopped; all eight GPUs verified free
at 14:18:06 UTC. Automatic communicating burns remain configured after either
success or failure. commands #0 leaves the detached queue running; do not relaunch.
Evidence: artifacts/p6iso-finetune-monitor-20261007-a02/ (7 verified JSON artifacts).
Full protocol: docs/SUPERVISED_FINETUNING_20261007.md.

## P6-iso supervised extension authorized — 7 October 2026

User clarified that the strongest P6 variant should also receive fine-tuning,
and confirmed proceeding with P6-iso. Earlier P6 was stronger on zero-shot
benchmarks; P6-iso had the best held-out LM loss. Run P6-iso only and reuse the
completed A/P6/P7-simple results. Same tasks, data, seeds, LR search and Trainer
recipe: 8 full fitting runs and 6 final tests, preceded by CUDA and eight-GPU
smoke/reload gates. Fresh root supervised-p6iso-20261007-a01. No new pretraining.

The classification wrapper enables end-to-end task gradients through P6-iso's
predictor, as for P7-simple. This is a downstream-only mode: forward values,
checkpoint keys, and default pretraining gradient isolation remain unchanged.
No auxiliary objective or normalization updates. Numerical, gradient and
save/reload tests must pass before production. Reverify/copy Accelerate config,
stop only verified authorized burns, require free GPUs, and retain automatic
communicating burn restoration after success/failure.

## Supervised study complete — 7 October 2026

Monitor `57be0de`, 13:41:06 UTC: all 58 stages passed, including 24 full fits
and 18 final tests. Queue ended 12:37:41 UTC; automatic communicating burns
verified 12:38:52, workers 341142–341149 still active at 100% on all eight GPUs.
Mean test accuracy (three fine-tuning seeds): PAWS-X A 91.850%, P6 90.300%,
P7-simple 90.917%; English XNLI A 78.922%, P6 79.188%, P7-simple 78.696%.
P6's small NLI gain is not an established robust win; one pretraining seed.
Full protocol/results: docs/SUPERVISED_FINETUNING_20261007.md.
Evidence: artifacts/finetune-monitor-20261007-a07/ (64 verified JSON artifacts).
No training queued. commands #0. Do not relaunch without a new request.

## Supervised study reviewed, still running — 7 October 2026

Monitor `790694a`, 11:07:15 UTC / 19:07:15 Singapore: 14/24 production fits
completed, all six LR selectors chose 3e-5 using development accuracy only.
Current: PAWS-X P6 seed 43, eight GPU workers 328924–328931. Final 18 tests
have not started; they follow all fitting. No failed queue stages. Six focused
CPU tests passed again; code review found no correctness defect. Earlier
zero-shot evaluation is complete. Details/evidence:
docs/SUPERVISED_FINETUNING_20261007.md and
artifacts/finetune-monitor-20261007-a06/ (33 verified JSON artifacts).
Do not relaunch. commands #0 leaves accepted queue and final burn handoff active.

## Supervised fine-tuning running — 7 October 2026

Accepted launch `0765d2e`; root
`/mnt/local/_outputs/deep-llms_th2/supervised-finetune-20261007-a03`.
Monitor `3717b86` at 17:41:09 Singapore verified all three CUDA gates and all
six distributed smoke/reload stages passed. A/P6/P7-simple reload development
loss/accuracy and weight hashes match exactly. Production A PAWS-X LR 1e-5
was beyond epoch 2 (logged epoch 2.655), all eight GPUs active, finite losses
and gradients. Development accuracy .8885 after epoch 2 is interim, not test.
No final comparison yet. Queue: 24 fitting runs, 18 selected-model final tests;
all use the documented matched protocol. Automatic communicating burns remain
configured after success/failure. commands #0 does not stop the detached queue.
Evidence: artifacts/finetune-monitor-20261007-a05/ (13 source-hash-verified JSON
artifacts); docs/SUPERVISED_FINETUNING_20261007.md. Do not relaunch the queue.

## Supervised B200 gates progressing — 7 October 2026

Accepted launch `0765d2e`, root supervised-finetune-20261007-a03. Monitor
`9cff5a2` at 17:34:45 Singapore: all three full-checkpoint numerical gates
passed; A/P6 eight-GPU four-update smoke and strict reload completed. Reload
loss/accuracy, selected-weight hashes and development-example hashes match.
P7-simple distributed smoke is in progress; production fitting not yet
verified. All eight GPUs were verified free at 17:28:12 after stopping only
known burn workers 314038–314045. Config copied and accelerate env checked.
Evidence: artifacts/finetune-monitor-20261007-a03/. Full protocol and queue:
docs/SUPERVISED_FINETUNING_20261007.md. Automatic final burns remain configured.

## Authorized supervised adaptation — 7 October 2026

User selected A, P6 (not P6-iso), and P7-simple for the proposed English
PAWS-X/NLI fine-tuning study. Implementing HF Trainer/Accelerate task head,
matched two-LR development search and three fine-tuning seeds, then final test.
Prepared gated submission `th2-tjx3-supervised-finetune-20261007-a03`;
execution not yet verified. Existing eval/ was inference-only.
Protocol: docs/SUPERVISED_FINETUNING_20261007.md. B200 gates/smoke must pass
before production. Retain automatic burn handoff and all-eight-GPU training.

## Downstream evaluation complete — 7 October 2026

Monitor `764d872`, 16:41:20 Singapore: all 15 stages and all 99 model–task
evaluations passed. Eleven checkpoint-2500 models, nine English tasks,
23,596 examples each, zero-shot seed 42, FA4. A leads both descriptive
macro-averages: raw accuracy 40.947%, normalized-where-available 41.113%.
No variant exceeds A overall; individual-task gains remain single-seed.
Full tables/protocol: docs/DOWNSTREAM_EVAL_20261007.md.
66 small artifacts hash-verified under artifacts/downstream-monitor-20261007-a08/.
GPU burns automatically restored and verified at 16:41:13; workers 314038–314045
remain active on all eight GPUs at 100%. No work queued; commands set to #0.

## Full downstream evaluation running — 7 October 2026

Monitor `963c243` at 16:27:02 Singapore: all eleven CUDA acceptance checks,
all eleven nine-task smoke evaluations, and smoke provenance/count validator
passed. Full evaluation began 16:26:22, first eight models active, P4/P5/P4-iso
queued. 23,596 examples per model; nine English tasks; zero-shot seed 42, FA4.
All 31 benchmark files verified against pinned hashes and local document hashes.
Separate eval_fa4 env verified; training envs unchanged. Automatic final burn
handoff remains configured. No scientific downstream results complete yet.
See docs/DOWNSTREAM_EVAL_20261007.md for paths and acceptance evidence.

## Authorized downstream evaluation — 7 October 2026

User authorized downstream evaluation of A plus all ten completed P4–P7 arms.
Use checkpoint-2500, FA4, zero-shot, seed 42, nine English tasks: HellaSwag,
XNLI validation, Belebele, XStoryCloze, PAWS-X, PIQA, ARC-Easy/Challenge,
WinoGrande. One checkpoint per GPU, up to eight concurrently. No retraining.
User also explicitly confirmed benchmark download. Controller download
`f025c70`, id `2026-10-07_08-06-59`, supplies 31 pinned raw files (104884425 bytes)
to `/mnt/local/_data/deep-llms_th2/downstream-english-20261007`.
Separate `eval_fa4` env installed by `4e07abc`; runtime verification pending.
Training/attention_bench/original eval environments remain untouched.

Required addition: verified local dataset mapping through existing task templates;
no Hub access from B200. Local nine-task raw-data/labels/document hashes passed,
reference under resources/downstream_english_reference_20261007.json. Ten existing
CPU groups passed; expanded nine-template test passed; added local-file test
passed after correcting its fixture to include the required training split.
Logs: temp/downstream-final-local-tests.log (one fixture error), then
 temp/downstream-local-files-test.log (corrected test passed).

Prepared gated launch `th2-tjx3-downstream-2500-20261007-a01`, root
`/mnt/local/_outputs/deep-llms_th2/downstream-2500-20261007-a01`.
Preflight waits for all files while burns continue, verifies environment/data,
then copies/checks Accelerate config and checks checkpoint recipes/ownership.
Supervisor rechecks/stops only authorized burn workers, verifies free GPUs,
runs eleven full-checkpoint CUDA numerical gates, an 8-example-per-task smoke
for all eleven models, validates counts/provenance, then full evaluation and
final validation. Automatic communicating-burn handoff on success/failure.
Do not claim full evaluation started until CUDA/smoke gates actually pass.

## Ten-arm queue complete — 7 October 2026

Monitor `a9289bf`, 15:55:24 Singapore: all ten arms completed 2500 steps;
all 63 stages and production validators passed. New best P6-iso loss
3.4662995700 versus A 3.4771733830; P7-simple 3.4745054830 is second.
P7-ems 3.4777271990; P4 all-head 3.4822870036. Single-seed, equal-token only;
downstream evaluation and seed confirmation pending. No new launch authorized.
Queue ended 15:37:34; automatic communicating burns verified 15:38:45.
Current snapshot confirms the same workers 308656–308663, all eight GPUs 100%.
47 small artifacts source-hash-verified; full report:
`docs/PROXY_REMAINING_RESULTS_20261007.md`. Commands returned to #0.

## Seven remaining-arm results verified — 7 October 2026

Monitor `5b3e750`, 11:19:26 Singapore: seven arms finished and passed validation.
New losses: P6 3.4782647578, P5 3.4838803344, P7-mlp 3.4773446524,
P7-kq 3.4777975959; A 3.4771733830, P7-simple remains best at 3.4745054830.
P7-ems at 1250/2500 on all eight GPUs; P4 then P6-iso remain queued.
No failures or queue changes. 33 small artifacts hash-verified; see
`docs/PROXY_REMAINING_RESULTS_20261007.md`. Single-seed/downstream caveats remain.

# Current task

## P6 production status — 7 October 2026

Monitor `a2b6e36`, 03:33:24 Singapore: P6 production at step 620/2500 with
checkpoint-500 saved, all eight GPUs active, finite loss/gradient logs and
~2.35 s/update. First three arms remain completed; six follow P6:
P5 → P7-mlp → P7-kq → P7-ems → P4 → P6-iso. No queue changes or failures.
Log: `temp/remaining-queue-status-20261007-a01.log`.

## B200 interim results verified — 7 October 2026

Read-only monitor `c642279` at 03:07:16 Singapore (6 October 19:07:16 UTC):
P7-simple/P7/all-head P4-iso finished and validated at 2500 updates. Held-out
LM losses 3.4745054830 / 3.4769314640 / 3.4896725928 versus A 3.4771733830.
Shared recipes/data match; the small P7-simple gain is single-seed/token-matched.
P6 smoke occupied all eight GPUs; seven production arms remain in the original
order. No failures or queue changes; automatic final burns are not due yet.
Fifteen small artifacts downloaded and hash-verified. Details:
`docs/PROXY_REMAINING_RESULTS_20261007.md`.

## Evaluation correctness review — 7 October 2026

Rechecked the evaluation implementation. Fixed invalid segment layouts that
could differ between SDPA and FA4: repeated noncontiguous document IDs, internal
padding holes, nonbinary masks and noninteger segment IDs now fail clearly.
Saved NumPy/tensor scores remain JSON numbers/arrays rather than strings; nonfinite
benchmark output is rejected. Invalid zero/one-token PPL windows fail before load.

Ten local test groups passed in 70.408 seconds, including all proxy arms,
strict restoration, unchanged buffers, BF16, padded SDPA/FA4 reference paths,
the real harness and the full benchmark CLI using the five registered English
task templates with synthetic local datasets. No actual benchmark downloads,
B200 execution, training changes, remote push or queue modifications. Full-model
CUDA downstream acceptance remains pending. Log: `/tmp/deep2shallow-eval-review-tests.log`.

## Local downstream evaluation support — 6 October 2026

Extended `eval/` for strict custom-checkpoint restoration and an inference-only
logits adapter over the existing full model path. Training forwards remain
unchanged: they compute chunked vocabulary logits and return loss statistics.
The adapter keeps proxy consumers active, disables auxiliary target/loss work,
and preserves document isolation/reset positions, including internal EMS padding.
Added English-only task selection (XCOPA has no English subset), saved evaluation
provenance/per-example scores, and fixed parallel option forwarding/failure status.
Perplexity now evaluates independent EOS-terminated documents; its context policy
is explicitly different from packed training validation. See `eval/README.md`.

All seven local test groups passed in 55.354 seconds, covering every proxy arm,
SDPA and FA4 reference-oracle paths, legacy A/B/F/G, plain HF checkpoints, BF16
autocast, strict state restore, a real Trainer checkpoint, actual harness scoring
on a local synthetic multiple-choice task, PPL CLI and launcher failures.
Used an isolated temporary venv over the local sampling environment; no existing
training environment changed. Full-model CUDA downstream smoke and benchmark
dataset availability remain unverified. No B200 action, push, or queue change;
the previously authorized ten-arm queue below remains the active remote scope.

## Remaining-arm production startup verified — 6 October 2026

Launch `258bde6`, job `th2-tjx3-proxy-remaining-2500-20261006-a01`.
Fresh monitor `e759c88` at 13:49:06 UTC confirms P7-simple production running
on all eight GPUs (workers 217157–217164), with finite LM loss/gradient logs.
The run started at 13:47:01 UTC, after its independent 25-step smoke and all
three validation gates exited zero. Production logs include two 10-step
logging intervals; no traceback. Nine further arms remain queued in the
operator-confirmed order below. No completed arm was retrained.

Accelerate resource config was copied byte-for-byte to the actual default
`/dev/shm/.cache/huggingface/accelerate/default_config.yaml`; `accelerate env`
verified MULTI_GPU, eight processes, BF16. Only approved burn workers
210967–210974 were stopped by verified PID handles, followed by the required
wait and a receipt showing all eight GPUs free before any CUDA job.
Environment/offline/NCCL settings match the existing launch recipe.

P7-simple actual CUDA SDPA/FA4 same-weight check passed: loss difference
0.0002040863, whole-model gradient relative L2 0.00592701, proxy-gradient
relative L2 0.00513579, zero cross-document output/gradient leakage.
The 25-step all-eight-GPU smoke validated model/optimizer/scheduler and all
rank RNG files, actual FA4 forward/backward receipts, finite diagnostics,
and peak allocated 114.2105 GiB. Matched-A recipe guard passed with training
fingerprint `6e4708f1e818fb44`. Smoke and production use separate fresh roots;
production was not resumed from smoke. No data/cache or old output deletion.

Root: `/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01`.
Production: `supervised/run/training/seed-42/P7-simple` (then the other arms).
The 63-stage supervised queue stops on any failure and automatically performs
owned-worker cleanup, free-GPU verification and communicating-burn restoration.
Final evidence will be `supervised/run/complete.json`, production validators,
comparison reports and `supervised/burn-verified.json`. Local startup evidence:
`artifacts/proxy-remaining-launch-20261006/`, checksummed small logs.
`commands.sh` is returned to #0; this does not stop the detached queue.


## Authorized remaining-arm queue — 6 October 2026

Operator explicitly confirmed ten fresh FA4 runs in this priority order:
P7-simple → P7 → P4-iso → P6 → P5 → P7-mlp → P7-kq → P7-ems → P4 → P6-iso.
Each uses all eight B200 GPUs, seed/data seed 42, 2,500 updates, unchanged
28,600-step schedule and 1,430 warmup; microbatch 16 × accumulation 4,
1,048,576 tokens/update, sequence 2048, EOS boundaries and reset positions.
No activation checkpointing. Reuse the completed FA4 A baseline; do not rerun
completed four-head P4 arms or older P1/P3 ablations.

Fresh read-only inspection d0a4027 at 13:36:28 UTC identified the tjx3 host,
eight approved communicating-burn workers (210967–210974), released GPU guard,
correct pinned attention_bench environment, intact local inputs, ~24.9 TB
free storage. Completed-run inventory confirms none of the ten selected arms
has a 2,500-step result. The four older P7 arms have only 25-step smokes.

Submitting job `th2-tjx3-proxy-remaining-2500-20261006-a01`, output root
`/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01`.
Preflight copies/checks Accelerate config then runs `accelerate env` before
verified burn-worker reclamation and all-eight-GPU free checks. Each arm has
cross-backend full-model numerics, a separate 25-step eight-GPU smoke and
saved-state/backend validator, plus a matched-A recipe/data-order check,
before fresh production training. Production checkpoint validation follows
each arm; matched baseline/all-arm reports finish the queue (63 stages total).
Any failure stops subsequent stages. Existing train_then_burn supervisor owns
cleanup and restores/verifies eight-GPU communicating burns after exit.
Production outputs: `supervised/run/training/seed-42/<arm>`; disposable smoke
outputs: `supervised/run/smoke/seed-42/<arm>`; never resume production from smoke.
No cache or prior output cleanup. Submission is not verified training progress.


## P7-simple implementation — 6 October 2026

Operator approved the controlled memory comparison: four query heads / two KV
groups; P4-iso tokenwise MLP and four-MLP-sum target; cosine-only auxiliary;
existing P7 joint native/proxy softmax and isolated estimator. New arm
`P7-simple` supports SDPA and FA4 through train.py and the sequential queue.
Predictor initialization matches P4-iso under the same module seed.
See [P7_IMPLEMENTATION.md](P7_IMPLEMENTATION.md) for the exact design and checks.
Careful review passed all 68 acceptance/regression tests: a fresh 66-test suite
plus two direct P4-control checks, including real Trainer resume and two-rank
CPU DDP. No training-code issue found. No new remote launch or GPU access.
Earlier B200 acceptance below does not cover this new arm.

## P7 review and B200 acceptance completed — 6 October 2026

All four P7 variants passed full-model CUDA SDPA/FA4 comparisons, separate
LM/cosine/relational gradient-route checks and exact document-isolation probes.
The optimized relational calculation matched the original objective and FP32
derivatives (relative L2 1.26e-7) and ran 3.25x faster for that component.
The existing 65-test CPU acceptance/regression run also passed.

Launch `2729792`, job `th2-tjx3-p7-checks-20261006-a01`, remote root
`/mnt/local/_outputs/deep-llms_th2/p7-checks-20261006-a01`: all ten 25-step
8-GPU Trainer runs (A and P7/P7-kq/P7-ems/P7-mlp on both backends), reports,
backend/checkpoint validators and profiling stages passed. Production recipe
microbatch 16 / accumulation 4 / sequence 2048 / global 1,048,576 tokens; full 28,600-step schedule,
warmup 1,430; diagnostic evaluation 32 rows. No production P7 training was launched.
P7 median update: FA4 2.6094s / SDPA 2.9736s; peak 116.27 / 125.89 GiB.
FA4 A 2.0691s. SDPA A 2.4286s. See [P7_IMPLEMENTATION.md](P7_IMPLEMENTATION.md)
for all variants, numerics, speed measurements and limitations.

Accelerate cache was copied and `accelerate env` verified 8-GPU BF16 before
verified burn-worker reclamation and a free-GPU check. No env/driver change.
Supervisor finished successfully at 10:17:19 UTC / 18:17:19 Singapore. Final fresh
inspection confirmed eight approved communicating-burn workers, advancing
collective progress, and guard re-enabled. Both prior P4 arms completed 2,500
steps and were preserved. Their small results were downloaded and verified.
P4-iso-4h LM 3.479563634; P4-4h 3.479081029; A 3.477173383: no single-seed gain.

Local evidence folder: `artifacts/p7-review-20261006/` (ignored); P4 and P7
archives contain metrics/configs/logs/receipts, not weights. The tested source
is committed to th2. Results are retrieved and verified; commands.sh is inactive (#0).
Next research step requires a separately requested production experiment;
these smokes establish execution/numerics, not long-run model quality.

## Initial P7 implementation (2026-10-06; historical local phase)

Operator requested `proxy_arm_P7_spec.md` Revision 2 for both SDPA and FA4.
P7/P7-kq/P7-ems/P7-mlp now use the existing Trainer/Accelerate pipeline,
versioned p7-r1 targets, isolated convolutional estimators and separate
token/query denominators. FA4 uses interleaved native/proxy entries with
duplicated queries and retained odd outputs: exact joint softmax, with extra
compute explicitly reported. Step-1000 evaluation records the cosine heuristic
and proxy attention mass. See [P7_IMPLEMENTATION.md](P7_IMPLEMENTATION.md).
Local acceptance included real CPU Trainer resume and two-rank DDP. The
subsequent actual CUDA/backend/throughput acceptance is recorded above.
Second review: batched relational KL, cached per-batch masks/indices/counts,
skipped unnecessary query sampling and fused P7 normalization reductions.
Rejected nonfinite auxiliary weights before training. All65tests passed in
319.199seconds, including loop/batched gradient equivalence, exact resume,
two-rank DDP and legacy regressions. CPU timing benefits vary by shape; B200
speed gains remain unmeasured. CPU FA4 oracle coverage is distinguished from
real kernel validation. Full results are in the implementation report.
The initial implementation was local-only; the newer B200 acceptance entry
above supersedes that phase.
Older remote status below is historical, not a fresh observation.

## Latest FA4 status (2026-10-06,14:23 Singapore /06:23UTC)

Read-only monitorea81118: P4-iso-4h completed2500updates and its checkpoint
validator passed. Finished13:09:56Singapore; whole job5820.273s (~97min),
Trainer5756.086s. Full4882-row validation LM3.4795636336261895, raw auxiliary
0.40820062034433374. Proxy-disabled LM3.5015848079660326, reliance0.02202117.
Compared with matched FA4 A3.4771733830242613, LM is higher by0.00239025;
this single-seed result does not show an improvement. Peak allocated110.13GiB.

P4-4h is running at1854/2500 (~74.2%),~2.302s/update. Latest training LM3.738,
raw auxiliary0.5892, grad norm0.1984; finite and no errors in inspected recent
log. Checkpoint directories1500/1750 present. All8GPUs have training workers,
97–99%utilization,~119462MiB used each; no burn overlap. Expected remaining
~25minutes of updates plus final evaluation/checkpoint/handoff, not a guarantee.
No training changes. Automatic final burn restoration remains configured.
Evidence temp/p4-fa4-status-20261006-a02.log SHA256
c2d3e60da8106cb0e038255d13d3c8c3931013c857491ecaf658fb48d149ea03.
P4-iso result/validator retained in artifacts/p4-fa4-production-20261006/.
commands.sh restored#0; no whole-queue completion claimed.

## Latest FA4 status (2026-10-06,13:06 Singapore /05:06UTC)

Read-only monitor9bd67e1: P4-iso-4h is running at2413/2500,~2.258s/update;
P4-4h remains queued. Latest logged training LM3.539, raw auxiliary0.4042,
lambda0.1, grad norm0.1666; finite, no errors in the inspected recent log.
Gates remain~0.970–0.985. Checkpoint directories2000/2250 are present; this
status check does not validate their contents or claim final completion.
All8GPUs have the expected training workers,97–99% utilization,117550MiB
used per GPU. No burn overlap. Training and automatic handoff were unchanged.
Evidence temp/p4-fa4-status-20261006-a01.log SHA256
f5a2bf94cf579ab32f50266c7cc4cc69cf27bded6ae15a08628ccc8170822329.
commands.sh restored#0 after this read-only request.

## FA4 production verified running (2026-10-06)

Launch565c695 / p4-fa4-four-head-2500-20261006-a01 passed preflight and
started fresh P4-iso-4h. Monitor0337db7 confirms step82, finite loss/gradients,
~2.25s/update and one training worker on each of8GPUs. Actual FA4 4.0.0b33
receipts have28forward calls and28each Q/K/V backward calls. Step80 logged
loss9.934, grad norm1.842; this is training loss, not held-out performance.
FA4 A baseline reuse validation passed. P4-4h remains queued after the first
arm and its checkpoint validation; each stops at2500 on the full schedule.

Accelerate was copied/byte-checked in attention_bench and accelerate env
verified8-GPU BF16. All8GPUs were verified free at03:32:55UTC before launch.
Validation supervisor terminal receipt confirms success and communicating burn
handoff at03:29:12UTC. Production reclaimed only those freshly verified workers.
Its supervisor will restore verified communicating burns after success/failure.
commands.sh is#0; the independent tmux training queue continues.

Production root: /mnt/local/_outputs/deep-llms_th2/p4-fa4-four-head-2500-20261006-a01.
Evidence temp/p4-fa4-production-monitor-02.log, SHA256
e9d6302e300ca22c7bf6e84053c72a5864f76bfb04c29d8ccb8d5a9a32de5ea7.
The earlier SDPA run is cancelled and preserved separately. No production
completion or scientific gain is claimed. No environment/driver reinstall.

## FA4 validation passed; fresh production submission (2026-10-06)

FA4 full-Qwen routed/reference, cross-backend gate1, and all8GPU25-step
A/P4-iso-4h/P4-4h smokes passed. Actual FA4 forward/backward receipts,
weights/optimizer/scheduler/all8RNG states, finite gradients and memory passed.
Cross-backend overall gradient relative L2:0.5073%/0.5142%; proxy-only0.3859%/
0.4310%; zero document leakage. Full evidence and profile breakdown in
P4_P5_P6_IMPLEMENTATION.md. Throughput2.0692/2.2533/2.2979s per update for
A/iso/P4; overhead8.90%/11.05%, above3%target and reported before launch.
Peaks103.62/112.37/113.88GiB. Model/source unchanged by the backend selection.
Evidence temp/p4-fa4-checks-monitor-04.log SHA256
22ebbfe7eb9af176d116ebb2eec8fc0fcd239de69f843e82ccf2b95e95591ce6.

Fresh production root /mnt/local/_outputs/deep-llms_th2/p4-fa4-four-head-2500-20261006-a01.
P4-iso-4h then P4-4h,2500updates each/all8GPUs,attention_bench,FA4,
seed42/module43,micro16/GAS4/2048,full28600steps/warmup1430,EOS/document
isolation/resetpositions. Existing FA4 A reused only for comparison; no smoke
or cancelled SDPA weights reused. Preflight requires completed check/burn
handoff and unchanged validated source; copies/verifies Accelerate, reclaims
freshly identified burns and requires all8GPUs free. Supervisor restores
communicating burns on exit. Submission is not training-start verification.

## Operator correction: use FA4 for P4 (2026-10-06)

The user confirms FA4 is the selected backend; the P4 specification's SDPA
wording was stale. Updated the specification and proxy_heads.b200.json to
explicit FA4. The CLI default remains SDPA for compatibility. Preserve the
existing SDPA output separately; do not change backend on resume. Replacement
P4-iso-4h then P4-4h must start fresh,2500 updates each/all8 GPUs, and reuse the
matching completed FA4 A. No change to the model, objective, data, seeds,
optimizer or schedule. Use the already installed attention_bench environment
(FA4 4.0.0b33); leave train_env and NVIDIA installation untouched.

Stop request05373dd targets only the freshly verified old supervisor via pidfd
SIGTERM; its established cleanup restores communicating burns. Verify terminal
handoff before the replacement. Required checks: CPU FA4 plumbing/gradient
routing/document isolation/save-resume, full-Qwen CUDA same-backend reference
comparisons, FA4 vs dense comparisons at gate1, all8GPU full-size25-step smokes
and checkpoint/backend/throughput validation. No long run until all pass.

## Production start verified (2026-10-06)

Launch f807f59 / p4-four-head-2500-20261006-a01 is active. Read-only monitor
6e7e49a confirms Accelerate copy/8-GPU BF16 verification, precise approved
burn-worker reclaim, and all eight GPUs at zero memory/no PIDs before training
at03:04:32UTC. Baseline reuse validation passed. P4-iso-4h reached step14;
step10 had finite LM12.01, auxiliary1.000, grad norm2.449, ~2.61s/update.
Eight workers own the eight GPUs, no burn overlap. Actual train/eval backend
is cuDNN SDPA with no math fallback. P4-4h remains queued after the first arm
and its checkpoint validation; each stops at2500 on the unchanged full schedule.
The independent supervisor will restore verified communicating burns on exit.
commands.sh is now inactive (#0); this does not stop the running tmux queue.
Evidence: temp/p4-production-monitor-01.log, SHA256
61e2fd31ac6b84dc9c660e03c1a2578601794393f280c756612523324cf675c3.
Production root: /mnt/local/_outputs/deep-llms_th2/p4-four-head-2500-20261006-a01.
Only the first arm's start is verified; no production completion is claimed.

## Routing fix verified; production launch (2026-10-06)

The corrected custom gradient routing passed all 38 selected CPU tests and
B200 validation in p4-routing-fix-checks-20261006-a02. Under controlled
numerical settings, reference-repeat gradients matched exactly. P4-iso-4h
matched the recomputation reference exactly for LM, auxiliary and combined
losses; P4-4h matched for LM and auxiliary losses, with a worst per-parameter
combined-gradient relative maximum of 0.315% (unchanged 1% threshold).
Both cross-document isolation checks passed. Deterministic settings were
used only for this numerical diagnostic, not the training smokes.

Normal dense cuDNN SDPA, full 28-layer Qwen, eight GPUs, microbatch16/GAS4,
2048 tokens: A and both P4 arms completed 25 steps, with finite gradients,
validated weights/optimizer/scheduler/eight RNG files and backend receipts.
Median seconds/update over unprofiled steps10–25: A2.4275, P4-iso-4h2.6136
(+7.67%), P4-4h2.6586 (+9.52%). Peak allocated GiB:110.62/117.13/118.64.
The 3% overhead target is not met; the measured breakdown is recorded in
P4_P5_P6_IMPLEMENTATION.md and was reported before production submission.
The check supervisor finished successfully and restored eight communicating
burn workers, with collective progress verified at02:57:19UTC.
Evidence: temp/p4-fix-monitor-04.log, SHA256
479573f1b78f3c3ac8654df0aa420d7a2a624e98519df8b59daf0e361dde5893.

Submitting the already authorized fresh2500-step runs, P4-iso-4h then P4-4h,
root /mnt/local/_outputs/deep-llms_th2/p4-four-head-2500-20261006-a01,
tmux tjx3-p4-four-head-2500-20261006-a01. Existing denseA is reused only for
comparison. Seeds42/module43, gates1, auxiliary lambda0.1/ramp250, full
28600-step schedule/warmup1430; EOS/document isolation/reset positions.
No smoke weights are reused. Launch requires completed CUDA checks and exact
validated model source hashes, copies/verifies Accelerate, reclaims only the
freshly identified approved burn, and requires all eight GPUs free. Each
arm gets checkpoint validation before the next job; the supervisor restores
verified communicating burns on success or failure. No environment or driver
installation was changed. Submission is not evidence of training progress.

## Fixing the auxiliary autograd graph (2026-10-06)

The user explicitly requested fixing the failed CUDA gate. Anomaly diagnostic
3b747d2 / p4-backward-trace-20261006-a01 identifies
ScaledDotProductCudnnAttentionBackward0 as the first NaN-producing operator in
P4-4h auxiliary-only custom backward. The recomputation reference passes on
identical weights/backend. This is a software/autograd/backend interaction;
there is no evidence of faulty B200 hardware. Dense SDPA is already in use.
Evidence temp/p4-backward-trace-monitor.log, SHA256
049117ea4fe978fa1017240b8973bf266cd0a4353dac5669cf2ca7c7177f077f.

CPU minimal reproduction proves returning None from the old shared two-output
node still invokes upstream backward. The corrected BlockMLP computes forward
activations once, then creates separate gradient nodes sharing saved activations.
The auxiliary node has an explicitly detached input, so there is no backward
edge into the backbone. Model state names, objective, gates, masks, seeds and
normal training loop remain unchanged. This also fixes the shared route for P6.
New regression requires zero upstream backward visits, not only zero/None leaf
gradients. All16 anticipatory CPU tests passed116.943s, including exact Trainer
resume, BF16, checkpointing, AOT compilation and two-rank DDP. Evidence:
temp/p4-graph-reproducer.log and temp/p4-structural-routing-tests.log.

Next supervised CUDA validation/full-size25-step checks:
th2-tjx3-p4-routing-fix-checks-20261006-a01. Same dense-SDPA backend/environment,
same tolerances, fresh outputs, previous handoff must be complete, Accelerate
copied/verified, approved burn-only reclaim and all-eight-free check, automatic
burn restoration on exit. Production remains unlaunched until all checks pass.


## Confirmed CUDA correctness failure — production remains unlaunched (2026-10-06)

Latest diagnosis identifies P4-4h auxiliary-only backward with aux_recompute=False
as the failing case. Forward LM/aux losses are finite (12.17263699/1.00022733),
but282parameter tensors have NaN gradients, including backbone and earlier
estimator tensors. The detached-input recomputation reference (True) passes
this same auxiliary-only finite/gradient-routing check. LM-only comparisons
pass; P4-iso-4h also passed the first diagnostic. Do not generalize these probes
to full training safety: no full-size25-step smoke or production optimizer
updates have run. Do not relax tolerances or bypass the failed custom route.
The underlying CUDA/autograd cause still needs isolation and a verified fix.

Spec proxy_arms_P4_P5_P6_spec.md §9 explicitly says stop and report on test
failure. Training is blocked at this gate. Final diagnostic supervisor receipt
at02:25:53.707380UTC confirms cleanup, all-eight-free checks and restored burn
with collective_progress_verified=True. Read-only monitor3a6daeb observed
all8GPUs owned by the new burn; no training was active. Evidence:
temp/p4-diagnose-monitor-02.log, SHA256
3beb0ec34b0c0930225a4c072f677f26f8daae434ef36200b15df280b42af370.

commands.sh returned to#0. The next work is diagnosing/fixing the custom
auxiliary gradient route, then rerunning unchanged CUDA acceptance and the
full-size smokes/profile before the two fresh2500-step runs. Preserve failed
outputs; both supervised jobs are terminal. Existing denseA checkpoint/data
and environments were not modified. Accelerate config copy was intentional.


## P4 CUDA acceptance blocked production; burns restored (2026-10-06)

Job add770c / th2-tjx3-p4-four-head-checks-20261006-a01 reached the GPU tests.
Accelerate was copied to the activated environment's actual cache path
/dev/shm/.cache/huggingface/accelerate/default_config.yaml (different from the
unactivated read-only preflight path), byte-checked and accelerate env confirmed
8-GPU BF16 MULTI_GPU. Exact approved burn workers were reclaimed; the supervisor
saved its all-eight-free receipt. No optimizer update or training smoke ran.

CUDA reference check: P4-iso-4h passed separate LM/aux/combined gradients and
cross-document isolation. Loss differences were0; worst per-parameter gradient
relative maxima were0.00914634/0/0.00927568. P4-4h LM passed at0.00824176, but its
aux-only capture hit the nonfinite-loss-or-gradient assertion. This is not a
tolerance failure. Failed before full25-step smokes/profile or2500-step runs.
Do not launch production or relax tests until the cause is understood/resolved.

Supervisor finished failure handling at02:19:57UTC, verified all GPUs free,
then restored eight approved communicating burn workers with advancing
collectives. Monitor 1ff14b0 evidence temp/p4-checks-monitor-01.log, SHA256
2678fc5d1b779b2c054d6afeb2014c718aa9fd6eaa32702316680a80a5c92628.
Bounded diagnosis submitted072381c, job th2-tjx3-p4-four-head-diagnose-20261006-a01,
output /mnt/local/_outputs/deep-llms_th2/p4-four-head-diagnose-20261006-a01.
It reruns P4-4h only and records reference/custom mode, finite losses, exact
nonfinite gradient names/counts. Same verified reclaim and automatic burn handoff.
No model/training-loop changes or tolerance changes. Production command is only
an unsubmitted local draft:temp/p4-four-head-training.commands.sh.

Local profiler-instrumentation regression:4tests passed2.105s, including BF16
routed/reference gradients, four-head GQA mapping, checkpoint/target logic and
both backend validators. temp/p4-profile-routing-regression.log.


## B200 recovered; four-head P4 launch checks (2026-10-06)

Read-only job th2-tjx3-p4-four-head-preflight-20261006-a02 at 02:10:42 UTC
passed on thiennh-p6-tjx3-worker-0: eight B200s, approved resource burn workers,
guard enabled, train_env torch2.14.1/transformers5.9.0/accelerate1.13.0/datasets4.8.5,
23,369GiB free. Existing dense A is intact at2500, LM3.4779414257087833;
train/eval fingerprints6e4708f1e818fb44/08432871cf987d61.
Evidence temp/p4-recheck-preflight-04.log, SHA256
adcb4a4316a890a80bc576c04da38bf71832a10b5b6c3c0e97573782f5a46aae.

Next authorized stage is job th2-tjx3-p4-four-head-checks-20261006-a01,
output /mnt/local/_outputs/deep-llms_th2/p4-four-head-checks-20261006-a01.
It copies/verifies Accelerate, verifies approved burn identities again, reclaims
only those workers and checks all eight GPUs free. Supervised queue:
full-Qwen CUDA routed-vs-recomputed gradient/document-isolation checks, then
25-step eight-GPU Trainer smokes for A/P4-iso-4h/P4-4h, component profiling at
step6 and unprofiled timing atsteps10-25. Eval/monitor32rows for these disposable
smokes only. Same full28600schedule,1430warmup,micro16/GAS4,seq2048,seed42.
The existing supervisor restores verified communicating burns on success/failure.
Inspect acceptance, memory, profile and throughput before fresh2500-step runs.
Training code/model is unchanged by launch tools. Profile instrumentation only
exists in the disposable wrapper process. Validator now also checks dense SDPA;
CPU tests cover both backends, missing rank RNG state, and math fallback rejection.


## B200 connectivity recovered; correcting read-only preflight (2026-10-06)

At 01:02:08 UTC the existing c01373a preflight reached the expected host
thiennh-p6-tjx3-worker-0. It stopped at `git rev-parse HEAD`: the runner syncs
source without a .git directory. No Python inspection or GPU action ran.
Evidence: temp/p4-recheck-preflight-03.log (SHA256
59cd805173faf581529889dd540030c5285576fad68b174f8c6c191ea0a22592).
Corrected project command removes the unnecessary remote Git query; new job
th2-tjx3-p4-four-head-preflight-20261006-a02 remains read-only. Continue the
already-authorized P4-4h/P4-iso-4h launch only after fresh validation.
Previous infrastructure-block entry below is historical.


## P4 four-head launch blocked by runner hostname resolution (2026-10-06)

User authorized training the latest P4-4h and P4-iso-4h variants on B200.
Planned established dense-SDPA recipe: all8GPUs per arm, seed42, micro16/GAS4,
seq2048, EOS/document isolation/reset positions, stop2500, full28600schedule,
1430warmup. Must copy/verify Accelerate config, run CUDA correctness/full-size
smokes and throughput checks, then train with train_then_burn automatic recovery.
None of the GPU stages has started.

Submitted read-only preflight c01373a, job
th2-tjx3-p4-four-head-preflight-20261006-a01. GitHub main verified c01373a.
Dropbox _RUN_STATUS_.log modified2026-10-05T23:15:38Z (Oct6 07:15:38Singapore)
reports FAILED(rc=255): ssh: Could not resolve hostname <host>: nodename nor
servname provided, or not known. Controller line is16:15:36, timezone unconfirmed.
Evidence:temp/p4-launch-controller-after.log, SHA256
5d83fd70f2090eb451bc2e074c4f80c455b85abe9f12287de75be4b633bb5da8.

This is infrastructure failure before remote shell execution, not a training
failure or evidence the GPU node died. No GPU stops, environment changes,
Accelerate-cache copy or training occurred. Per AGENT_GUIDE failure policy,
do not resubmit/change/kill/clean to work around it; wait for operator repair.
Local commands.sh restored#0 and saved preflight in temp/p4-four-head-preflight.commands.sh.
No further push; execution remote still has the read-only preflight c01373a.
A controller retry can only inspect, not train. After repair, inspect fresh
status, finish launch gates and submit the authorized supervised training queue.

## Four-query-head P4 variants implemented locally

Added P4-4h and P4-iso-4h. "Four heads" is interpreted as four query heads:
Qwen3 query heads13-16 (zero-based12-15), backed by KV groups7-8 (zero-based6-7).
All Q/K remain native within each proxy block; the other12query heads read
native values. Existing v_proj output rows are split between native and proxy
inputs. Estimator architecture/initialization, gates, targets, auxiliary loss,
normalization and isolation exactly match the respective P4/P4-iso parent.
The original all-head arms remain unchanged. Incompatible GQA ratios are
rejected. Arm identity protects resume; no extra model parameters or state keys.

Both names work with train.py, sequential eight-GPU queues, gate evaluation
and comparison reports, including partial-vs-full parent and isolation contrasts.
Dedicated16-query/8-KV CPU tests verify exact selected heads, unchanged Q/K and
native values/attention outputs on the other heads, gradient isolation, identical
initial parameter state, real Trainer resume and queue generation. All48selected
CPU regression tests passed in194.643s; latest report contrast assertion also
passed separately. Logs:temp/p4-four-head-regression.log,
temp/p4-four-head-focused.log and temp/p4-four-head-report.log.
No remote actions or training launch. commands.sh remains#0. CUDA smoke and
throughput measurements remain pending. Usage:docs/P4_P5_P6_IMPLEMENTATION.md.

## P4/P5/P6 follow-up correctness review

Fixed train.py's initial dataset shuffle to honor data_seed, falling back to
seed when unset. Previously the documented initialization-only A experiment
(seed1042/data_seed42) changed the first permutation despite fixing the Trainer
sampler seed. The real Trainer test now captures consumed batches: equal data
seed gives identical batches with different backbone weights; changing data
seed changes the batches. Existing equal-seed runs preserve their ordering.
Historical unequal-seed checkpoints will fail fingerprint validation if the
corrected shuffle changes their saved data, rather than silently resume.

Report validation now checks actual P4/P5/P6 target quantity, exact windows,
normalization, epsilon and cosine objective, plus recipe/result agreement.
A matching target-version label is no longer sufficient. Added BF16 tests for
separate LM/auxiliary gradients on all five arms with checkpointing and AOT
compilation, compared with the detached-input recomputation reference.

All46 selected CPU acceptance/regression tests passed in183.264s, including
2-rank Gloo, exact resume, new batch-order checks, and old P1/P3/FA4-reference
coverage. Log:temp/p4-review-regression.log. No model architecture change was
needed in this review. B200/CUDA smoke and throughput profiling remain pending.
No remote action or training launch; commands.sh remains#0.

## P4/P5/P6 implemented and locally verified

Implemented revision 3 of docs/proxy_arms_P4_P5_P6_spec.md using the existing
train.py/HF Trainer pipeline: P4, P4-iso, P5, P6, P6-iso. Dense SDPA isolation
and reset positions remain the defaults. P4/P5 inject all values; P6 injects
before the residual block. Versioned four-MLP targets, prescribed gate defaults,
independent module seed43, strict resume metadata, sequential queue selection,
reporting and saved-gate evaluation are supported. P5 defaults to isolation.

One-forward block routing uses separate LM/aux autograd outputs. The auxiliary
output needs independent storage: a tensor alias merged routes in eager mode,
and a view still merged routes in AOT Autograd. An output clone fixes both.
No estimator recomputation by default; optional recompute reference and compiled
estimator/injection/cosine paths are available. Normalization buffers use one
collective per update. Isolated estimators execute zero auxiliary gradients at
lambda0 so DDP reducer hooks remain valid. Gates are logged at step0.

44 local CPU tests passed in169.987s, including actual two-rank Gloo/HFTrainer
for P4-iso/P5/P6, exact save/resume for all five variants, checkpoint replay,
no cross-document leakage, gradient routing and accumulation, AOT compilation,
and existing P1/P3/FA4-reference regressions. Additional tightened custom-VJP
comparison passed: fp32 max-absolute error divided by the reference tensor's
maximum absolute value <=1e-6; BF16 <=1%. Logs: temp/p4-p5-p6-acceptance.log
and temp/p4-gradient-tolerance.log. This is CPU acceptance, not CUDA validation.

No B200 interaction, push or real-data training. commands.sh remains#0.
Before launch, still run the spec's GPU smoke and component throughput profile;
3% overhead is a target, not an observed result. CUDA Inductor also needs smoke
before enabling compilation. First screen order P4-iso then P6; P5 conditional.
Usage and caveats: docs/P4_P5_P6_IMPLEMENTATION.md. Legacy FA4 A cannot serve as
matched control for default dense SDPA. Extra initialization-only A seed must
keep data_seed42; historical make-jobs --seeds changes both seeds.

## Final P1 gates retrieved successfully (2026-10-06 Singapore)

Controller export98263d9 recovered: OK,2files pulled, published18:26:49UTC
(02:26:49Singapore). Downloaded trainer_state.json and result.json; final
step2500confirmed and result SHA256 matches original successful queue manifest.
Step2500 per-layer channel mean absolute alpha (12layers,1024channels each):
2:0.9754641;4:0.9748995;6:0.9746817;8:0.9775315;10:0.9818562;12:1.0029802;
14:0.9883882;16:0.9918040;18:0.9848090;20:0.9913205;22:0.9837440;24:0.9769256.
Overall mean absolute alpha0.98370038; layer means range0.97468168–1.00298023.
These are logged absolute means, not signed means or individual channel extrema.
Gates stayed near initial1; they did not collapse to zero. Exact eval aux
loss0.5590405854648716. Proxy-disabled LM loss3.525009881073549 versus enabled
3.4823192569156207, reliance increase0.04269062415792835. This trained model
uses the proxies but remains worse than A; reliance alone is not baseline gain.
Trainer runtime5912.0704s (98m32s), whole production job5976.716s (99m37s).
Receipts:artifacts/proxy-p1-alpha1-completion-20261006/final-gate-metrics.json
and result.json. Full downloaded state retained in temp/p1-alpha1-final-trainer_state.json,
SHA2567a6cce59761fc0336b77fa108b0e1df3ade5593516abe0c6c752e4daea4f57a8.
No new workload or GPU manipulation; commands.sh restored#0.


## Gate retrieval blocked by runner hostname resolution (2026-10-06 Singapore)

Dropbox _RUN_STATUS_.log updated2026-10-05T18:10:40Z (02:10:40Singapore)
now explicitly reports both requests FAILED(rc=255):
-54b3650, th2-tjx3-p1-alpha1-final-gates-20261006-a01
-98263d9, th2-tjx3-p1-alpha1-pull-training-metrics-20261006-a01
Exact error: ssh: Could not resolve hostname <host>: nodename nor servname
provided, or not known. Controller raw timestamps11:09:55/11:10:38 have an
unconfirmed timezone; use Dropbox UTC timestamp above for user-facing timing.
This is an infrastructure failure, not a model/logging failure. Per AGENT_GUIDE,
stop retries/submissions/process changes until operator repairs runner hostname
or DNS. Final alpha remains unknown locally; metrics were logged every10steps
and saved in remote trainer_state.json. Last successful monitor still shows
2500complete and communicating burns at17:43UTC. Cannot infer machine death.
Evidence:temp/p1-gate-dropbox-controller-latest.log, SHA256
3f5ca946ec4844ee77a562d635dfc94e60764ee83c6dd71012546a5410a7c5a8.
No new remote actions. Before any future push, review commands.sh: remote head
98263d9 still contains the failed#2 request; do not accidentally resubmit it.


## Final learned gates requested; remote read pending (2026-10-06 Singapore)

User asks for gate alpha at2500updates of the alpha-one P1 run. Cached final
monitor log includes only smoke initialization gate values; do not report those
as final gates. Read-only CPU checkpoint extraction submitted54b3650 at
17:50:45UTC /01:50:45Singapore, jobth2-tjx3-p1-alpha1-final-gates-20261006-a01.
It reads checkpoint2500 safetensors, checks12x1024gates, prints FINAL_GATES
with per-layer/overall mean,mean_abs,std,min,max and final evaluation metrics.
No GPU allocation, burn interruption, training or checkpoint mutation.
At17:58UTC controller still had no acknowledgment/new log; GitHub main verified
54b3650. Last controller acknowledgment is priorc568290. No reported error and
no evidence that machine failed. Do not resubmit to refresh. Await Dropbox
log, parse FINAL_GATES, retain small receipt, then restore commands.sh#0.
Pending script backed up attemp/p1-alpha1-final-gates-20261006.commands.sh.
User clarified that gates should already be logged. Confirmed callback logs
per-layer mean absolute alpha every10steps into trainer_state.json, including
2500. Submitted direct#2 export of final trainer_state.json and result.json as
th2-tjx3-p1-alpha1-pull-training-metrics-20261006-a01; no training/GPU command.
Checkpoint-read54b3650 still unacknowledged at18:04UTC. Direct exported state
is sufficient for final per-layer logged means; do not infer full channel ranges
from those means. Await published files and restore#0 only after acknowledgment.


## P1 alpha-one finished; communicating burns restored (2026-10-06 Singapore)

Read-only request c568290 acknowledged by controller. Fresh17:43:01UTC
(01:43Singapore) snapshot confirms all queue jobs exit0, identical run/complete
receipts, and passed production validator at exactly2500updates. Production
finished17:25:32UTC /01:25:32Singapore, job5976.716s=99m37s.
Held-out LM loss3.4823192569156207 versus previous P1-block3.477280233119416
and A3.4771733830242613: worse by0.0050390/0.0051459 respectively in this
single-seed comparison. Initial alpha1 alone did not improve held-out LM loss.
Same saved recipe/fingerprints except alpha initialization. Peak115.7626GiB.
Final validator checked checkpoint2500 model,optimizer,scheduler,all8RNGs,
finite state and actual FA4 receipts. No training was resumed or launched.
Supervisor verified all8GPUs free then restored communicating burn at
17:26:44UTC /01:26:44Singapore. Latest snapshot workers123842–123849, guard
released, burn cycles advancing960→1130 and payload1065.94→1254.70GiB/rank.
Evidence:artifacts/proxy-p1-alpha1-completion-20261006 and
temp/p1-alpha-finish-20261006-a01.log, SHA256
6216067ba527494d36e24e194efdd43a7c18eb9241a17bb47f205e6ffb374e1c.
Published snapshot filename/body followed the prior monitor format despite
controller acknowledgment of c568290; results above are from that fresh snapshot.
commands.sh returned#0. Values-only and shorter-target variants remain unrun.


## P1 values-only and shorter targets implemented locally (2026-10-06 Singapore)

User requested values-only P1 and shorter lookahead using2or3blocks. Added
proxy_kv_mode=v for P1 only; all queries/keys remain native within each
attention operation, selected value groups consume the proxy. Added3 to
allowed lookaheads and optional explicit proxy_layers, preserving original
12proxy blocks2,4,...24 for both shorter windows. Historical automatic
placement/defaultKV behavior remains unchanged. Auxiliary loss/normalization,
HFTrainer, FA4/SDPA interfaces, seeds and data pipeline are reused.

Nondefault options are saved and protected by resume recipe checks; defaults
are omitted for legacy checkpoint compatibility. Gate evaluation reconstructs
new settings. Usage and sequential queue recipe: P1_VALUES_AND_SHORT_LOOKAHEAD.md.
First comparison: values-only/k4 and native-KV/k2, each against the alpha-one
P1 control, same fixed layers. k3 supported as an optional separate experiment.
No launch/push/process changes; commands.sh remains#0. Previous remote P1
alpha-one run was not inspected in this turn. All24CPU tests passed in50.158s:
exact Q/K/native-value preservation, target window sums and fixed placement,
LM/aux gradient routing, isolation, zero-gate equivalence, SDPA versus varlen
CPU reference with checkpointing, real HFTrainer exact resume and gate eval,
CLI parsing and legacy regressions. Documented sequential queue accepted by
run_experiments.load_jobs. Test log:temp/p1-variants-tests-final.log.
Actual FA4 CUDA smoke remains required before these variants train on B200.

## Fresh P1-block with alpha initialized to one (2026-10-05)

User authorized fresh P1 training with alpha initialized to1. Keep auxiliary
weight0→0.1 over250steps unchanged to isolate initialization. New optional
proxy_alpha_init defaults0; nonzero recorded in saved recipe, default omitted
for older-checkpoint compatibility. Learned gates remain trainable/no decay.
16 local CPU tests passed: nongate state identical, immediate LM gradients,
real HF save/resume exact equivalence, legacy recipe and gate evaluation.

Submitting fresh3-step full-shape smoke followed by fresh2500-step P1-block,
all8 B200s, FA4,seed42,micro16/GAS4,full28600/warmup1430,EOS/document isolation
and reset positions; only initial gate differs from prior P1-block. Separate
smoke/training outputs; same existing cache, no deletions or checkpoint reuse.
Root:/mnt/local/_outputs/deep-llms_th2/proxy-p1-alpha1-2500-20261005-a01.
Production begins only after smoke reports and checkpoint/backend validation
pass. Launch copies/verifies Accelerate config, runs accelerate env, checks
pinned env and known burn identities; supervisor verifies GPUs free and
automatically restores communicating burn after success/failure.
Launch10fc15e verified by read-only monitors9c84dea/8f359f0. Both local and
B200 CPU checks passed. Accelerate copied to active /dev/shm cache and env
confirmed8GPU/BF16. Only identity-rechecked burn workers113219–113226 stopped;
all8 GPUs verified0MiB/no processes at15:44:34UTC before smoke. Smoke3steps
completed and validator passed, peak113.5184GiB, finite state/backend receipts.
Fresh production started afterwards (separate seed/reset/output, no smoke resume).
At15:49:05UTC all8 training workers117877–117884 active,97–99%utilization;
latest observed58updates,~2.31s/update. At50 LM loss10.38,grad norm2.015 finite;
gates remain~1. Saved production recipe/data fingerprints EXACTLY match prior
P1-block after removing only proxy_alpha_init=1. No final scientific result yet.
Expected roughly100minutes training, automatic burn recovery after end/failure.
Evidence:artifacts/proxy-p1-alpha1-20261005/ and temp/proxy-alpha-monitor-a02.log
(SHA2565838a49c61d0ce90aae2c82c5cf6b174922aa9c83923ba0d36a8eed7f05778b0).
commands.sh returned inactive#0; current training continues. Old canceled queue
remains canceled. Next action:read-only progress/final validation and burn check.

## Gate multiplier evaluation complete; training stopped (2026-10-05)

User requested canceling the active queue and evaluating2x/5x gates;10x omitted.
Stop13a31ba targeted only the verified queue supervisor via pidfd SIGTERM and
preserved all source outputs/checkpoints/data. The old queue will not continue.
Gate-evaluation launch66b62d7 ran scripts/evaluate_proxy_gates.py with production
ProxyTrainer on all8 B200s, FA4/BF16/micro16, completed checkpoint2500 for each
of P1-block/P3-block. No optimizer, training step, or checkpoint save occurred.

Each evaluation used the same4882rows/9,981,660target tokens, fingerprint
08432871cf987d61. Multipliers applied to learned per-channel alpha only:
1x→2x→5x→restored1x. Each pass also ran the existing proxy-disabled control.
Original1x and restored1x LM losses match EXACTLY for both arms; proxy-disabled
losses are identical across all passes. Full restored model-state and source
checkpoint hashes unchanged. Actual FA4 eval receipts saved; local and B200
CPU tests passed, including exception-safe restoration and real HF evaluation.

| Gate multiplier | P1-block validation LM loss | P3-block validation LM loss |
|---|---:|---:|
| 0 (disabled) | 3.4772731692609793 | 3.479060135913288 |
| 1 | 3.477280233119416 | 3.4789088241301345 |
| 2 | 3.4772899565613393 | 3.4789827767758883 |
| 5 | 3.4773086201176944 | 3.4807160064284948 |

Both2x/5x worsen LM loss; P1 effects are tiny, P3 at5x rises0.0018071823
versus1x. This does not support simply amplifying the trained gates; it does
not test retraining with a different gate initialization or auxiliary objective.
Baseline A at the same2500steps remains3.4771733830242613.

Both evaluation jobs exited0 with passed integrity summaries; queue completed
15:18:26UTC/23:18Singapore. Supervisor finishedok15:19:37UTC after verified
communicating burn recovery. Read-only monitor4de4db9 at15:22:37UTC/23:22Singapore
verified all8 approved burn workers113219–113226, guard released, and collective
cycles/payload advancing (through220cycles/244.28GiB per rank in snapshot).
No training remains active. commands.sh is inactive#0.

Remote:/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-20261005-a01.
Evidence:artifacts/proxy-gate-sweep-20261005/ (hash-verified original JSON receipts),
temp/proxy-gate-sweep-monitor-a02.log (89ee3dbec278b6b321b304739af5be23a4a6c362a6c6222f88d8e0a0856f800e),
and temp/proxy-gate-sweep-burn-verified.log (ed85fdf4ba338bcb1cb5bf88b173b3d930aeb0a8be4af8a7328ff3fd3772400e).
Await user direction before any further training or gate experiments.

## Training queue stopped; gate evaluation authorized (2026-10-05)

User explicitly requested stopping the current run and testing gates2x/5x,
with10x omitted. This is evaluation of completed P1-block/P3-block checkpoints,
not new training. Scope:original full4882-row validation, FA4/BF16/world8,
micro16, multipliers1,2,5,then restored1; existing proxy-disabled evaluation
repeated as a control. Source checkpoints/data/caches must be preserved.

Stop13a31ba verified supervisor identity/argv/guard and GPU-worker ancestry,
then sent SIGTERM through pidfd. Monitorcf20292 confirms canceled queue terminal
15:06:42UTC, all8 GPUs free before burn handoff, guard released and communicating
burn107399–107406 progressing at15:09:09UTC. Intentional cancellation, not an
unexplained training failure. Remaining training arms are no longer queued to run.

scripts/evaluate_proxy_gates.py reuses the production ProxyTrainer evaluation,
strictly restores saved checkpoint state, checks validation fingerprint/counts,
scales only in-memory alpha tensors, restores gates after each pass, and verifies
full model-state/checkpoint hashes unchanged. No optimizer/scheduler is created.
CPU tests passed for real tiny P1/P3 checkpoint evaluation and exception-safe
scaling/restoration. Production train/model/loss code unchanged.

Fresh evaluation root:/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-20261005-a01.
Launch copies/verifies Accelerate config and accelerate env, checks pinned env,
reclaims only verified burns, verifies all8 GPUs free, and evaluates both arms
sequentially under train_then_burn with automatic burn restoration on end/failure.
Launch submitted; remote results still pending. Do not restart the training queue.

## P3-block complete; P1-lambda0 running (2026-10-05)

Read-only monitor9f10dc9 at14:51:06UTC/22:51Singapore confirms P3-block2500
finished exit0 at14:38:56UTC/22:38:56Singapore; validator passed14:38:58UTC.
Final checkpoint contains model,optimizer,scheduler,trainer state and all8 RNG
states. Common recipe and train/eval fingerprints match A. No run modifications.

Held-out LM loss3.4789088241301345 vs A3.4771733830242613 and
P1-block3.477280233119416: no gain over A at this single-seed checkpoint.
Proxy-disabled LM loss3.479060135913288; reliance difference+0.00015131178315375,
a small measured proxy benefit at identical weights. Aux loss0.3033168538614671.
Trainer runtime7305.7404s=2h1m46s; whole job7376.64s=2h2m57s;
peak allocated126.5192GiB. Training steps2500,input tokens2,621,440,000.

P1-lambda0 started14:38:58UTC; last observed update296/2500,~2.24s/update,
all8 workers103472–103479 active. Common recipe/fingerprints also match A.
Queue continues P1-lambda0→P3-lambda0→P1-flow→V1→V3, with existing validators
and automatic communicating burn restoration after end/failure.
Evidence:artifacts/proxy-fa4-p3-block-completion-20261005/ and
temp/p3-block-status-20261005-a01.log, SHA256
9058544308659096e28c89c9a389daa4d14eae10a8c82b56d0de258c057ad351.
commands.sh restored#0. Next action:read-only remaining-arm progress/results.

## P1-block complete; P3-block running (2026-10-05)

Read-only monitor606a5b7 at13:41:34UTC/21:41Singapore confirms P1-block2500
completed with exit0 at12:35:58UTC/20:35:58Singapore. Per-arm validator passed
at12:35:59UTC. checkpoint-2500 has model,optimizer,scheduler,trainer state and
all8 rank RNG files; actual FA4 receipts and finite normalization/gates checked.
Common recipe and rebuilt train/eval fingerprints exactly match A.

P1-block held-out LM loss3.477280233119416 on4882rows/9,981,660targets;
A3.4771733830242613 (difference+0.0001068500951549). This single-seed result
is effectively tied, not evidence of an improvement. Auxiliary cosine loss
0.4119091441372323; disabling proxies gives LM loss3.4772731692609793,
reliance difference-0.00000706385843685 (negligible at this checkpoint).
Trainer runtime5915.4653s=98m35s; whole job6974.93s=116m15s including cache
rebuilding/loading and finalization. Peak allocated113.5184GiB.

P3-block started12:35:59UTC on all8 GPUs, current workers97665–97672,
last observed update1331/2500 at~2.88s/update. Its common recipe/fingerprints
also match A. Queue/supervisor still active; automatic burn handoff remains
configured for end/failure. No training/process/config modifications.
Evidence:artifacts/proxy-fa4-p1-block-completion-20261005/ and
temp/p1-block-status-20261005-a02.log, SHA256
076e67858cbe698728bfe6ad0804da6c058a31fd347fbcb3fa90f5c3b7469104.
commands.sh restored#0. Next action:read-only P3-block completion/results check.

## P1-block nearing cutoff (2026-10-05, 12:32:33 UTC / 20:32 Singapore)

Read-only monitor919682d: P1-block still running, last observed update2427/2500
at~2.32seconds/update; all8 GPUs97–99% utilization. No final result/validator
receipt yet. Latest logged LM loss3.539, auxiliary loss0.413, grad norm0.2204
at2420 are finite training metrics, not final held-out results. P3-block is next.
Rebuilt train/eval fingerprints6e4708f1e818fb44/08432871cf987d61 and common
recipe match completed A exactly. No process/config/training changes.
Evidence:temp/p1-block-status-20261005-a01.log, SHA256
85ae104e96adad4048f659c7cc56f0cc6de34b462fbb2666142ed80e397b4a44.
commands.sh returned#0; active queue unchanged. Next check:2500 checkpoint,
final evaluation/result and P1-block-validation.json before claiming completion.

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
full28600/warmup1430,micro16/GAS4,FA4 isolation/reset positions/EOS,activation checkpointing off.
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

## Authorized clean restart, P1/P3 first (2026-10-05)

User explicitly requested stopping the active seed42 FA4 screen, removing its
outputs and dataset cache-*/tmp-* files, then restarting with main P1/P3 first.
Stop110b1b6 identity-rechecked supervisor85650 and sent SIGTERM via pidfd.
Monitor5213e0e confirms terminal10:31:53UTC, all8 GPUs free before automatic
burn restoration, communicating burn89414–89421 advancing and guard released.
This is an intentional cancellation, not an unexplained training failure.

Replacement root: /mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02.
Order: P1-block, P3-block, P1-lambda0, P3-lambda0, P1-flow, V1, V3.
Same seed42,2500 updates each/all8 GPUs, full28600/warmup1430,micro16/GAS4,
FA4 isolation/reset positions, unchanged baseline recipe and no resume.
Completed A remains the comparison reference; source Arrow text is preserved.

The first CPU job inside the replacement supervisor removes only old screena01
and inspected English train/validation cache-*/tmp-* files. Inspection found325
cache files (1,021,643,627,160bytes),351 source files. Fail-closed source inventory,
symlink/open-file checks and source-preservation tests passed locally. This
forces normal train.py tokenization/packing to rebuild, then later arms reuse
that fresh cache. No separate data export. Small stop/cleanup receipts retained
outside the deleted output under proxy-fa4-restart-control-20261005-a01.

Launch copies/verifies Accelerate config, checks accelerate env and existing
smoke/recipe evidence, safely reclaims only verified burns, checks all8 GPUs
free, then performs cleanup and sequential training+validation. Automatic burn
recovery after queue success/failure remains enabled. Relaunch submitted;
actual cleanup/startup still needs remote verification. Historical a01 startup
below is superseded; do not resume its weights.

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

## Baseline pipeline startup verified (2026-10-05 07:43 Singapore)

Launch3eec7f9 is running on tjx3. The actual Accelerate cache is
/dev/shm/.cache/huggingface/accelerate/default_config.yaml; copied bytes/hash
and accelerate env confirm8GPU/BF16. Only verified burn workers21008–21015
were signaled; all8GPUs were verified free at07:38:34 Singapore. Reviewed
train/model/supervisor source hashes match the remote receipt exactly.
Eight train.py ranks25274–25281 are alive under25268. At07:43 the first
smoke invocation was tokenizing the full English pool (28%,10.08M/36.60M
documents) through the ordinary HF Dataset cache. No optimizer step or smoke
success is claimed yet. The supervised queue automatically runs the3-step
smoke, validates it, then starts a fresh A/seed42 run to2500; burns restore
on queue completion or failure. Full schedule28600/warmup1430 unchanged.

Evidence: temp/baseline-launch-a01.log and temp/baseline-startup-monitor-a01.log
(SHA2562ea3a32a6de98f98cd2e6afa33d72491498ab620981e184c786dc273f18d0f9e).
Read-only monitor4c1d579 watches for first baseline optimizer logs (30-minute
window). The independent supervisor and training do not depend on that monitor.
commands.sh returned to#0; this does not stop the detached run. Next status check
should inspect launch.log, supervised/run/run.json, smoke/validation.json and
baseline-seed-42-arm-A.log under the launch root. Do not relaunch over it.

## Original Qwen arm A launch authorized (2026-10-05 Singapore)

User requested fresh original-model training to2500 updates. Prepared one
seed42 arm A (proxy_screen=true for the matching boolean dense SDPA path),
all eight B200s, micro16/accum4/seq2048, EOS/document-local positions,
full28600 schedule and1430 warmup. Decoder/LM/aux checkpointing disabled,
using the previously validated throughput configuration. No proxy heads or
auxiliary objective in A. Three-update full-size smoke in a separate output
must pass checkpoint, finite-loss/gradient-log, SDPA and peak-memory gates
before the fresh2500-update stage. Existing supervisor restores and verifies
communicating burns on success/failure, with ownership-safe cleanup.

Read-only preflight d8fc122 verified tjx3, eight known burns21008–21015 under
20939, train_env torch2.14.1/transformers5.9.0/accelerate1.13.0 and pip check.
Its Parquet-only assertion was incorrect; 60d1732 verified the saved Arrow
shard layout already handled by train.load_text. No GPU worker was stopped
by either preflight. Inputs remain cx_sampled_old English and pinned Qwen assets.
Launch root: /mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01.
Production model path: supervised/run/baseline/seed-42/A under that root.
Launch command copies resources/accelerate_config.yaml to the actual HF cache,
compares bytes, runs accelerate env, then invokes the verified supervisor in
an independent tmux session. Launch3eec7f9 startup is verified; see below.

## Proxy implementation follow-up review passed (2026-10-04)

Fixed custom-queue baseline selection/mixed-family rejection and V3 widening
using the configured sequence length; duplicate report seeds now fail early.
Default 2048-token screening recipe is unchanged. Added independent 28-layer
P1/P3 target/loss/all-parameter-gradient oracles with 16Q/8KV, reduced hidden
width, nonzero gates/means and channel exclusions. All121 offline tests passed
in140.462s (temp/proxy-review-full-a01.log). Repeated eight-process CPU BF16
resume/global-gradient checks passed with the same maxima as the implementation
receipt; centering buffers match across ranks. New receipt:
temp/proxy-review-ddp-a01/verified.json. No B200 job; commands.sh remains#0.
Full-size capacity/throughput smoke still precedes real screening.

## P1/P3 proxy-head implementation (2026-10-04, revision 4)

User confirmed dense SDPA and positions reset per document in the updated
proxy_heads_P1_P3_spec_v3.md (now revision 4). Implemented the P1 and P3 models,
lambda-zero/block/flow options, widened V1/V3 controls, step-level distributed
centering, optional calibrated channel mask, logging and reliance evaluation.
Reuses train.py/HF Trainer/Accelerate, packing/cache and checkpoint/resume.
proxy_heads.b200.json creates the eight-arm, three-seed sequential queue:
24 training jobs, each eight GPUs, 2500-step cutoff, 28600/1430 schedule/warmup,
micro16/accum4/seq2048. No B200 launch; commands.sh stays #0.

Implementation and validation details: PROXY_HEADS_IMPLEMENTATION_20261004.md.
Final offline suite: 118 tests passed in 137.825s; log
temp/proxy-full-suite-final-a02.log. Compilation/diff checks passed.
Eight-rank CPU BF16 train/resume passed A/P1-flow/P3-block/P3-lambda0; max state
difference 1.862645149230957e-9; means match across ranks. FP32 global-batch
gradient checks passed at max3.725290298461914e-9. Receipt:
temp/proxy-ddp-a02/verified.json. Full-size B200 memory, dispatch and throughput
checks remain the next preflight; prior additive-mask SDPA results do not verify
the new boolean-mask proxy path. The optional offline estimability study and
cached decoding are outside this implementation.

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

## SDPA full-model train/eval backend audit authorized (2026-10-04)

User requested execution of SDPA_BACKEND_CHECKS_20261004.md, with updated Section 6
including evaluation. Implement C1/C3 first: all 11 arms, cross-document and dense
isolation, train/eval, micro16/seq2048, checkpointing on/off. Full Trainer objective,
no full-model optimizer updates; a tiny Consumer-Aware Trainer smoke verifies
automatic first-train/first-eval receipts. Existing real English benchmark pool.
Each GPU handles independent cases; this is not a DDP throughput experiment.
Production now records first actual train/eval dispatch on rank zero, without
changing its data/masks/objective. No production isolation or backend pinning.
C2 full-model FP32 remains micro2 in previous evidence; runtime is unchanged.
C4 optional, C5 trajectory control deferred while closing dispatch gaps.
Job sdpa-full-profile-20261004-a01 uses existing train_env, verified burn stop,
Accelerate copy/env verification and automatic eight-GPU burn recovery.

## Document-isolation investigation consolidated (2026-10-04)

See the [consolidated investigation report](DOCUMENT_ISOLATION_INVESTIGATION_20261004.md) for the goal, code map,
all test stages, timings, numerical results, and unresolved trajectory drift.
After discussing the 48% gradient-norm gap, the current recommendation is dense
SDPA isolation for real experiments when minimizing uncertainty is the priority.
This is not a production switch or launch authorization. Same-weight FA4 checks
passed; the 200-update trajectory comparison remains failed and its cause is
unproven. Production isolation remains disabled; commands.sh is #0.

## Same-weight trained-attention follow-up authorized (2026-10-04)

User requested further verification after the 200-step trajectory gate failure.
Compare both saved step-200 checkpoints with dense SDPA and FA4 on identical
training/validation batches, micro2 and micro16. Eight independent GPU cases;
no optimizer updates. Repeat each backend; micro2 also uses FP32 math SDPA.
Strict state loading, unchanged parameter fingerprints, matching inputs, full
parameter-gradient comparisons. Preserve .01 loss / 3% gradient / 2% output
limits; save all cases even on gate failure. Production training unchanged.
Initial 3279ab4/a01 attempt failed before GPU capture: safetensors.load_model
rejected Trainer checkpoints containing both cloned tied-embedding aliases.
Burns 18263–18270 restored and verified at 13:49:25 UTC. Added explicit tied
alias consistency validation, strict state loading and cloned-alias regression.
Retry a02 preloads both actual checkpoints on CPU before stopping any burn.
Retry 3d5070b/a02 COMPLETED: all eight same-weight backend gates and all four
FP32-reference cases passed unchanged thresholds. Full-gradient relative L2
FA4 vs dense: 0.181%–0.532%; micro16 specifically 0.181%–0.266%. Against FP32
math, dense BF16 is 0.539%–1.209%, FA4 0.540%–1.163%. Identical weights, matching
inputs, parameters unchanged. Repeat backward variation is 0.119%–0.302% dense,
0%–0.111% FA4, showing the full pipelines are not bitwise repeatable. This
supports accumulated drift as a plausible explanation of the earlier training
trajectory difference, without proving its cause or erasing the failed gate.
See TRAINED_ATTENTION_CHECK_20261004.md for scope and all measurements.

Supervisor passed=true at 13:57:24 UTC, burns 19173–19180 restored with advancing
collectives; collector 736802d verified live identities and guard release at
13:58:16 UTC. Original train_env/driver and production training untouched.
Archive exported via fbef742; SHA256
dc7d0645149fac6628bcf3ea510ffca99cfa0a6fb3557dcd512691bf93555680.

Verified all 21 source-file hashes after retrieval to
artifacts/trained-attention-check-20261004-a02/. Eight receipts match the summary;
final supervisor passed and burn collective progress passed. commands.sh is #0.

## Authorized 200-update isolated-attention stability comparison (2026-10-04)

User approved the proposed longer check: fresh dense SDPA isolated and FA4
isolated runs, 200 updates each, same full arm A/seed42/data/optimizer, original
28600-step schedule and1430 warmup. Eight GPUs per run, micro16/accum4/seq2048,
checkpointing off. Repeated16.78M-token benchmark pool, separate512-row (~1M
input tokens) held-out pool from the saved validation split. No production run.
Hash the entire token/document-ID stream per rank; compare all200 loss/gradient
norm records. Evaluate both weight sets through common dense SDPA, then also
evaluate the FA4 weight set through native FA4. Predeclared comparison limits:
max training loss gap <.01, max relative gradient-norm gap <.03, common-backend
held-out loss gap <.01, trained FA4 dense/native held-out gap <.01.
Use the existing separate attention_bench environment and verified worker-only
burn reclamation, resource Accelerate copy/env check, automatic burn recovery.
Prepared job th2-tjx3-document-stability-200-20261004-a01; outputs under
/mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01/benchmark.
Both models start fresh rather than resuming the30-update test weights.

Completed both 200-update runs; strict trajectory comparison FAILED. Full
per-rank input streams and LR sequences match. All losses/norms finite. Maximum
loss gap 0.01066065 (limit .01), relative gradient-norm gap 48.098% (limit 3%).
Norm gap first exceeds 3% at update 137, with 49 exceedances overall. Common
held-out losses 7.1107591 dense / 7.1074235 FA4; same FA4-trained weights evaluated
through both backends differ only 2.58e-6. This does not establish backward
agreement at trained weights or identify the cause of trajectory drift.
Median full update times 2.42939s dense / 2.07476s FA4. See
DOCUMENT_ATTENTION_STABILITY_20261004.md for setup, limits and evidence.

Summarizer failed on the numerical gate after saving comparison.json; both
training/evaluation jobs finished successfully. All 24 source artifact hashes
verified locally in artifacts/document-stability-200-20261004-a01/. Added local
analysis.json and training_comparison.png/.svg, preserving raw receipts. Burns
17301–17308 restored with collective progress verified at 13:30:02 UTC / 21:30
Singapore; read-only collector verified live workers and guard release.
commands.sh deactivated (#0). No production change or additional GPU run.
Recommended next diagnostic: same saved weights/batch with both backends,
compare full gradients; this has not been launched.

## Full training benchmark completed and burns verified (2026-10-04)

daea6f6 / th2-tjx3-document-training-20261004-a01 completed successfully at
11:08:14 UTC (19:08 Singapore). All five modes finished 30 real HF Trainer
updates on all eight B200s; 40 rank receipts passed. Same seed-42 random full
Qwen3-0.6B arm A, micro16/accum4/seq2048, 1,048,576 input tokens/update, BF16,
checkpointing off, LM chunk128, full LR schedule28600/warmup1430. Shared test
pool: 8192 packed rows /16,777,216 tokens, mean3.453 document fragments/row.
The pool repeats during 30 updates. No scientific production run was changed.

Median full update intervals, updates6-30, maximum across all ranks per update:

| Mode | Seconds/update | Peak allocated GiB |
|---|---:|---:|
| Previous explicit SDPA, cross-document | 2.42576 | 110.62 |
| Implicit causal SDPA, cross-document | 2.06230 | 103.62 |
| FA4, cross-document | 2.08819 | 103.62 |
| Dense SDPA, isolated documents | 2.42524 | 110.62 |
| FA4 varlen, isolated documents | 2.07173 | 103.62 |

FA4 isolation is +0.457% time vs fast implicit SDPA, -14.595% vs previous
explicit SDPA. Approximately equal to fast causal for this baseline workload;
do not extrapolate the sub-percent gap or arm A timings to all custom arms.
Intervals include forward/LM loss/backward/DDP/clipping/optimizer/data delivery;
startup, first5 updates, evaluation and final weight saves excluded. One short
run per mode; no long-run convergence or between-run confidence claim.

Correctness gate passed before training: all-parameter gradient relative L2
FA4 vs matched dense isolated BF16 =0.0054076; sampled logits =0.0063636;
absolute loss gap4.1962e-5. Against FP32, gradient errors FA4=0.0123142 and
BF16 SDPA=0.0123313. Exact zero cross-document hidden changes and embedding
activation gradients for both isolated backends; cross-document controls leak.
Cross-document FA4 vs SDPA gradient error0.0042305. Mask semantics intentionally
change outputs/gradients: dense isolation vs cross gradient difference0.73331.
All training losses/gradient norms finite. Cross loss12.11895 ->10.84888;
isolated loss12.12183 ->10.98176. Same-semantics FA4/SDPA loss traces agree closely.

Input first-update hashes match per rank across all modes; data/seed/order are
shared. Final trained weights remain in each mode's final_model/ under
/mnt/local/_outputs/deep-llms_th2/document-training-20261004-a01/benchmark.
Original train_env/driver untouched; separate attention_bench used throughout.
GPU burns10552-10559 restored with advancing collectives; read-only collector
070798b also verified live matching workers and released guard.
Collector evidence temp/tjx3-document-training-collect-a01.log SHA256
34f210bd9c3eafca9d1d0695cba799e76400c9c4e2e1e343460f8c8025026688.
Result-only archive SHA256
835c13e4c140901aa99a2599299655083bca8bc3de2a1a961a1c26d4d9caced7;
exported via e141726 and retrieved with the exact expected archive hash.
All 50 artifact hashes passed under artifacts/document-training-20261004-a01/;
includes 40 rank receipts, correctness, timing summary, trained-step records
and verified burn receipt. commands.sh restored#0.
Production FA4/document isolation is not enabled; only disposable benchmark
code was added. Future integration requires per-arm auxiliary/loss checks.

## Full training attention comparison authorized (2026-10-04)

User requests real full-model training-step timings and output/gradient checks
for document-isolated FA4 versus previous cross-document attention. Use a fresh
test output directory, the separate attention_bench env, full Qwen3-0.6B arm A,
real sampled English text, HF Trainer/Accelerate, eight GPUs, seq2048 and matched
micro16/accum4 (1,048,576 input tokens/update), decoder/LM checkpointing off.
Also compare dense and FA4 with identical isolation semantics to distinguish
kernel numerical errors from expected changes caused by blocking documents.
Read-only preflight checks local model/tokenizer assets before GPU reclamation.
Production entry point and previous scientific outputs must remain unchanged.
Known burn reclamation and automatic verified recovery remain authorized.

Implementation: scripts/benchmark_document_training.py, process-local arm A
adapter only. Four local CPU tests passed, including actual HF Trainer update
equivalence for accumulation vs full batch with uneven document target counts.
Read-only 46e2f36 found only the verified eight burns and missing model assets.
Controller e21d47f downloaded all three pinned config/tokenizer files successfully
at 10:52:26 UTC; the test preflight verifies hashes before use. Initialization
is random seed 42, matching train.py; this is not pretrained-weight finetuning.
Prepared job th2-tjx3-document-training-20261004-a01 with fresh output root
/mnt/local/_outputs/deep-llms_th2/document-training-20261004-a01/benchmark.
Five modes: previous explicit SDPA cross-document, implicit causal SDPA,
FA4 cross-document, dense SDPA isolated, FA4 isolated. Each trains 30 updates
on a shared small text pool; report updates 6-30, including DDP/optimizer/data
delivery, excluding startup/evaluation/checkpoint writing. Full schedule and
LR warmup remain 28,600/1,430. Save final weights/logs in each mode's directory.
Require full-model BF16 output/all-parameter-gradient checks, independent FP32
reference, and exact document output/embedding-activation gradient isolation
before timed training. Automatic burn recovery wraps the entire GPU queue.

## Packed-attention APIs verified on B200 (2026-10-04)

User accepted benchmarking efficient document isolation and asked whether
existing APIs avoid custom kernels. Yes: FA4 cumulative sequence lengths and
FlexAttention block masks handle it. Production attention/packing is unchanged;
no real training launched. Separate attention_bench env installed via e6352df:
torch 2.14.1+cu130, flash-attn-4[cu13] 4.0.0b33, cutlass-dsl 4.8.0.
Original train_env and driver were not changed.

Corrected benchmark ca21c7b / th2-tjx3-packed-kernels-20261004-a02 completed
10:07:27 UTC (18:07 Singapore). All FA4 and Flex-FA4 cases passed output/QKV
gradient reference checks and exact forward isolation: single/equal/ragged
documents, normal causal and strict-past, BF16 GQA 16/8 heads, D128, B2, L2048.
Maximum relative L2 error was 0.302%, relative tensor-max error 0.489%.
Strict-past varlen uses each document's Q[1:] with K/V[:-1], zero first output;
CPU tests also cover all-singleton documents and gradients.

Attention forward/backward milliseconds, 20 measured iterations:

| Layout | Implicit causal | Dense isolated | FA4 varlen | Flex FA4 |
|---|---:|---:|---:|---:|
| Single document | 0.339 | 0.710 | 0.536 | 1.009 |
| Four equal documents | 0.339 | 0.712 | 0.515 | 0.965 |
| Ragged documents | 0.339 | 0.711 | 0.525 | 0.965 |
| Ragged, strict-past | N/A | 0.775 | 0.695 | 0.970 |

Includes tensor conversion/gather/scatter overhead; excludes metadata setup
and initial compilation. B2 is a microbenchmark, not the production microbatch
or a full-model/eight-GPU training throughput result. Direct FA4 improves over
dense isolation here but does not match unsegmented implicit causal attention.
First run b391b37 also passed native varlen/Flex Triton; its FA4 adapters failed
because beta returns (output,lse) and B200 Flex-FA4 requires 256-token blocks.
Both adapters were corrected before a02; no failed candidate was promoted.

Accelerate config copied/verified; stopped only verified burn workers 5420-5427,
then required all eight GPUs free. Automatic burn recovery verified workers
6339-6346, 100% utilization, 155212 MiB/GPU, advancing collective cycles.
Evidence: temp/tjx3-packed-kernels-a02.log, SHA256
75063063c63e07a1a37411c8783ea77b1954dfd095124e56b90c3ed730fab9dc.
Remote root: /mnt/local/_outputs/deep-llms_th2/packed-kernels-20261004-a02.
commands.sh restored to #0. Any production integration still needs explicit
document metadata, boundary-target/loss-denominator handling, all-arm checks
and realistic full-step measurements. No hand-written CUDA kernel is needed.

## B200 document-isolation tests passed; burn restored (2026-10-04)

e14a630 / th2-tjx3-document-isolation-20261004-a03 completed09:38:50UTC
(17:38:50Singapore). Both FP32 and BF16 four-test suites passed across all11
arms, including exact isolation, zero cross-document embedding gradients,
packed/separate losses and parameter gradients, checkpointing on/off.
Attention-only BF16 B=2,H=16,L=2048,D=128,20 measured forward/backward iterations:
implicit causal0.2535ms/236.27MiB; explicit causal0.6238ms/252.27MiB;
document-isolated0.6239ms/252.27MiB. All three used cuDNN SDPA on torch2.14.1+cu130.
These are kernel timings, not full-training throughput or eight-GPU scaling.
Accelerate resource config copied and verified; accelerate env confirmed CUDA.
Stopped only verified/hash-approved original burn workers501-508, then verified
all eight GPUs free. Automatic recovery verified workers4194-4201,100% utilization,
155212MiB/GPU, all-rank readiness and advancing collective cycles.
Evidence temp/tjx3-document-a03.log, SHA256
0fd29bf33bcad3610dd498b82405b1a93eabeaeb4865744af29ef1c86e997002.
commands.sh restored to#0. Production packing/attention policy unchanged;
no actual training launched. Remaining implementation work is explicit document
boundary metadata and non-bottleneck loss-denominator review, then full-model
throughput measurement if user chooses to enable isolation.

## B200 document-isolation test (2026-10-04)

User authorized B200 tests, no real training. Submitted e14a630, job
th2-tjx3-document-isolation-20261004-a03. Rechecks known burn source hashes
and live worker identities, stops workers only via existing pidfd helper,
requires all GPUs free, runs FP32/BF16 all-arm tests and attention microbenchmark
on GPU0, then restores/validates eight-GPU enhanced burn in finally.
Result: /mnt/local/_outputs/deep-llms_th2/document-isolation-20261004-a03/result.json.
Wait for result including passed=true and burn; do not resubmit blindly.
a01 failed before GPU work because output parent directory was absent; a02
verified both actual B200 English split counts (36,595,514 / 11,822), copied
Accelerate config and ran accelerate env, then correctly refused occupied GPU0.
Evidence temp/tjx3-document-a02.log: worker501-508 under /tmp/llm_pretrain_burn.py
launcher434; eightB200, driver580.167.08, torch2.14.1+cu130, Transformers5.9.0,
Accelerate1.13.0, datasets4.8.5. No prior test stopped processes.
Reviewed tests now compare parameter gradients as well as losses. CPU FP32
passed; local A100 BF16 passed with a 1% tensor-norm and tensor-maximum error
bound for packed/separate gradient reduction rounding. Exact cross-document
output equality and zero cross-document embedding gradients remain strict.
Observed initial failing BF16 tensor norm deltas were0.26-0.32%; logs under
temp/document-isolation-reviewed-* and document-isolation-gradient-deltas.log.

## Local document-isolation feasibility tests (2026-10-04)

User requested local testing only; production packing and training behavior
are unchanged. tests/test_document_isolation.py exercises all 11 arms with
nonzero auxiliary output weights: output/embedding-gradient isolation with
checkpointing on/off, packed-versus-separate loss sums/counts, EOS ownership,
boundary target masking, and a leaking unsegmented control. Four tests passed
on sampling_b200 CPU FP32 and sparse_emb A100 GPU0 BF16 (torch2.7.1+cu118,
transformers5.9.0). No environment/driver changes or GPU process stops.
Commands: python -m unittest tests.test_document_isolation -v; GPU additionally
sets CUDA_VISIBLE_DEVICES=0 DOCUMENT_TEST_DEVICE=cuda:0 DOCUMENT_TEST_BF16=1.
Logs: temp/document-isolation-{cpu,a100-bf16}.log.
Attention-only BF16 benchmark, B=2,H=16,L=2048,D=128, four512-token segments,
3 warmups/10 measured forward+backward iterations on A100: implicit causal
2.24ms (SDPA flash), explicit causal6.54ms, document-isolated6.53ms (both SDPA
efficient). Peak allocated236/252/252MiB. Evidence:
temp/document-isolation-a100-attention.log. These are not full-model/B200 timings.
Implementation still needs real packing boundary metadata and loss-denominator
review for non-bottleneck arms; test-only EOS inference is not a production
boundary policy. Do not silently enable segmented attention for existing runs.

## Downloaded dataset compatibility check (2026-10-04)

Download 2a334dc completed successfully according to the runner receipt
(temp/tjx3-download-status-083501.log). User requests compatibility verification,
explicitly no training. Existing train.load_text supports the uploaded save_to_disk
shards. Updated only data_dir/eval_data_dir in deep_kv.b200.json to English
subsets/qwen3_0.6b_base_en_30B/{train,validation} under cx_sampled_old.
CPU-only commands.sh audit verifies trusted manifest, all manifested file sizes,
English file SHA256 hashes, real loader counts/order, and generated queue paths.
No model creation, tokenization, training, GPU stop or environment reinstall.
Wait for DATA_COMPATIBILITY_PASSED; model/tokenizer availability is reported
separately and is not implied by data compatibility. Historical launch helper
still targets old 78gg; do not use it unchanged on tjx3.

## Prepared old-pool download (2026-10-04)

User authorized downloading the already sampled old pool to replacement tjx3.
Verified public nht10/cx_sampled_old completion and remote manifest SHA256
dbba73b7a95ebc530297bac5a2151be6292f080e80e9b3101302a905806dde9a.
Release head 2c425f4e1d4467008a0afce00de0ae8a33c8e834; data revision
d62a1db19dc789f834fe2e84fbb7d55f529e16bb. About 156.5 GB, six languages,
30B English / 1B others, unpacked text Arrow shards with validation.
commands.sh now requests controller #d to /mnt/local/_data/deep-llms_th2/cx_sampled_old.
No resampling or environment reinstall requested. Verify download completion,
manifest/file hashes and adapt folder layout before training consumes this data.
The earlier environment audit ended at git rev-parse: the synced runner folder
has no Git metadata. Remove that assumption in the next audit; no environment
imports ran, so runtime readiness is still unverified. Evidence:
temp/tjx3-audit-upload-check.log and temp/tjx3-status-upload-check.log.
This supersedes the pending-audit note below.

## Replacement tjx3 environment check (2026-10-04 Singapore time)

User supplied a new Dropbox folder for deep-llms_th2_thiennh-p6-tjx3 and
requested environment/GitHub verification. Use label th2-tjx3 with
`--folders temp/dropbox_tjx3_folders.txt`; the private link is ignored locally.
GitHub main matched local install commit1d3b68b. Its unchanged specifications
install Python3.11 train_env/eval, pin transformers5.9.0/datasets4.8.5/
accelerate1.13.0, and leave torch unpinned. New runner installation log reports
OK eval, OK train_env and OK install:2envs; last update07:25:52UTC (15:25SGT).
It downloaded torch2.14.1, whereas the previous B200 runs used2.14.0.

Read-only runtime verification submitted as8f350d8 at07:46:44UTC (15:46SGT),
job th2-tjx3-verify-environments-20261004-a01. It inspects GPU inventory,
both interpreters/package versions/imports/pip check/Accelerate cache, and runs
one tiny CPU model equivalence test with CUDA_VISIBLE_DEVICES empty. No GPU
workload, installation, download, process stop or config-copy action submitted.
GitHub head confirmed8f350d8. As of the last poll, Dropbox still only exposes
the installation status; audit execution/result and actual new hostname/GPU
inventory remain unverified. Do not resubmit the audit or infer machine failure.
commands.sh deliberately remains the pending read-only#1; do not push another
commit with it active merely to refresh logs. Retrieve the audit, then restore#0.
Installation evidence: temp/tjx3-install-run-status-20261004.log, SHA256
7cedbe2ab78d6b79746fee6ee09d0ef1b51e9d35ad8792ed557a9042632850e6.

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

## Bottleneck production running after successful GPU smoke (2026-09-30 12:58 UTC)

Latest auditeb7ee19 at12:58:42 confirms Task-Aware at153 updates, ~2.77s/update,
all8GPUs98–99% /120490MiB, finite step150 LM7.908 and LR3.126e-5. Remote jobs.json
and recipe.json retrieved/hash-verified: both native production commands have
stop_after5000, full schedule28600/warmup1430, micro16/accum4, all8GPUs, fresh
separate outputs, and no resume_from_checkpoint. Consumer-Aware remains next.
Evidence artifacts/bottleneck-launch-20260930-a01/final-startup/; raw audit SHA256
1a35118217b1c526666dfb33e3654c0cdd15aeccf0ba7683a727690bfc799bb9.
commands.sh restored to#0 after verification; independent tmux queue continues.

Launchaa1ce75 passed full-shape eight-B200 smoke for Task-Aware-Align and
Consumer-Aware-Align, including native10->12 resume, finite component losses,
all-rank telemetry and >8GiB memory headroom. Smoke gate finished12:50:48 UTC;
its receipt SHA256 matches the successful outer queue artifact record.
Fresh Task-Aware production started12:50:48. Snapshot6d2290b at12:53:28 confirms
step40, finite losses/gradients, LR8.182e-6, ~2.78s/update, and one native train.py
worker per GPU at97–99% utilization /120490MiB. Consumer-Aware follows next.

Both stop at5,000 updates with schedule28600/warmup1430/micro16/accum4/world8/
seq2048. Main/decoder checkpointing off, auxiliary checkpointing on; explicit
mask/chunk128 retained. Production train/eval fingerprints equal prior B/F/G.
Accelerate resource/cache bytes and accelerate env verified8GPU BF16 at launch.
Approved burn workers457634–457641 were reidentified and stopped; all8GPUs
were verified empty12:45:09 before the smoke. Existing supervisor owns the
entire queue and automatic burn recovery after success/failure.

Root: /mnt/local/_outputs/deep-llms_th2/deep-bottleneck-5000-20260930-a01.
Production: production/run/{Task-Aware-Align,Consumer-Aware-Align}; smoke is in
production/run/smoke and is never used to initialize production. Source hashes,
receipts/configs and snapshots collected under artifacts/bottleneck-launch-20260930-a01/.
Startup audit SHA256 b5df94291654d6d4814663d0ae4d93af3e70186424cee8f4d386e0432cbd2dea.
Next: read-only progress/completion checks; do not launch a duplicate queue.

## Authorized bottleneck launch: two aligned arms, 5,000 steps (2026-09-30)

User authorized Task-Aware and Consumer-Aware, each5,000 updates on B200.
Selected Task-Aware-Align and Consumer-Aware-Align; both fresh initialization.
Read-only preflight2da76ed at12:39:19 UTC passed: eight approved burn workers
457634–457641 under457565, guard enabled, pinned torch2.14/transformers5.9/
accelerate1.13, model and English text paths present, >23TiB disk free.
Evidence temp/bottleneck-remote-preflight.log, SHA256
583a078e5d811c1289106a3e501510c83ef3f635862aa111053b2a0fea6f9c0c.

Prepared root /mnt/local/_outputs/deep-llms_th2/deep-bottleneck-5000-20260930-a01.
Queue: both production-shape smoke arms (10 updates then native resume to12),
then fresh Task-Aware-Align5000, Consumer-Aware-Align5000, matched LM comparison.
Production schedule28600/warmup1430/micro16/accum4/eightGPUs/seq2048 unchanged.
Decoder and main-LM checkpointing off; auxiliary-loss checkpointing on for both
new heads' memory headroom. Explicit attention mask and LM chunk128 retained.
Smoke gate requires >8GiB free on every GPU. Existing supervisor rechecks exact
burn identities, verifies free GPUs, and restores/validates communicating burns
after success/failure. Accelerate resource->cache copy plus accelerate env precede
reclamation. Next: verify actual launch, smoke receipts, and production progress.

## Bottleneck follow-up correctness review (2026-09-30)

Reviewed the four new arms against the specification and pinned Trainer code.
No training-logic defect found. Fixed sibling test imports so the focused suite
also runs as `python -m unittest tests.test_bottleneck` (previously failed before
any test). Added independent dense-attention/loss/gradient reference checks at
blocks5/21 with the actual Qwen attention-width ratio, and BF16 AdamW/scheduler
continuation checks for all eight checkpointing combinations across four arms.
All83 CPU tests passed; evidence temp/bottleneck-review-full.log. The isolated
reference also passed via module invocation (temp/bottleneck-reference-review.log).
Training implementation and launch recipe are unchanged. No B200 action; the
remaining operational step is still an authorized full-model GPU capacity smoke.

## Bottleneck task-aware / consumer-aware implementation (2026-09-30)

Implemented the four arms in `deep_bottleneck_task_and_consumer_v2.md`:
Task-Aware-NoAlign, Task-Aware-Align, Consumer-Aware-NoAlign, Consumer-Aware-Align.
They reuse train.py, native HF Trainer/Accelerate, EOS packing/cache, and the
sequential queue/report/staging tools. New model code stays in deep_kv/model.py.
128-wide normalized codes; independent extractor/readout; alignment weight 0/.3.
New runs require fresh matched initialization, not old B/F/G checkpoint loading.

Local CPU validation: full suite 81 tests passed; focused seven bottleneck tests
also passed after metric/staging review. Eight-process CPU BF16 micro16/accum4
training/resume passed all four arms, including evaluation deduplication over
five rows and uneven-mask distributed scaling against a global SGD reference.
Maximum resume parameter difference 1.86265e-9; masked distributed difference
3.72530e-9. Evidence: temp/bottleneck-regression.log,
temp/bottleneck-final-tests.log, temp/bottleneck-ddp.log, and
temp/bottleneck-ddp-20260930-a01/resume_verified.json (local ignored test outputs).

No new B200 workload submitted; commands.sh remains #0. Next operational step
is an authorized full-model B200 smoke/capacity test, especially for the larger
consumer readout. Choose a common training cutoff before launch; the new
specification does not select one. Existing B/F/G completion below is history.

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


## Authorized checkpoint/capacity investigation (2026-09-29)

User requests testing remaining loss checkpoint removal and maximum fitting
microbatch. Fresh capacity-20260929-a01 uses all8 GPUs and train_env, no installs.
Model adds default-on LM/functional-loss checkpoint switches; production CLI,
recipe, parameter schema and old resume checks remain unchanged. Benchmark-only
flags control them. 21 CPU tests passed, including all8 toggle combinations on
all7 arms: loss, gradients, AdamW moments, next weights and scheduler continuation.

Queue: B/F/G independent remaining-checkpoint ablations at16x4, then fresh-process
integer capacity searches for retained losses and fastest passing removal; bound64.
Only typed CUDA OOM receipts permit a failed attempt to continue. Unknown failure
or timeout aborts. Each successful attempt is18 real Trainer updates; timing6-18,
no profiler or weight saves. Fixed-global-batch cases16x4/32x2/64x1 verify matching
first512 packed input rows and fingerprints. Non-divisor batches use accumulation1
for capacity only; they are not scientifically matched training runs. Also tests
B with decoder checkpointing and micro64. Recommendations require >=8GiB reserved
memory headroom, a heuristic rather than a long-run no-OOM guarantee.

Launch copies and verifies Accelerate config/cache and env, inspects approved
burn identities, uses existing supervised reclamation/free checks, and restores
verified enhanced burns on success or failure. No scientific checkpoints touched.
Next: verify launch, collect all-rank results, then audit final live burn progress.

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


## Performance investigation: 13 cases verified; FA4 adapter correction (2026-09-29)

Original performance-20260929-a01 completed13 native/SDPA cases successfully.
FA4 failed in the tiny CUDA precheck because the benchmark adapter treated its
(output,lse) tuple as a Tensor. Verified pinned b32 source; fixed tuple unpacking
and added a regression test through actual Qwen forward/backward. All5 CPU
benchmark tests passed. Production model/trainer remains unchanged.

Failure audit9abc7c6 retrieved138 artifacts with verified source/manifest hashes;
local cross-rank timing/input-batch checks passed for all13 completed cases.
Artifacts: artifacts/performance-20260929-a01/. Burn recovery verified live at
12:45 UTC: all8 workers, guard released, cycles230->240/payload255.38->266.48GiB.
Current B3.3662s/update, microbatch32 B3.2795s, optimized B2.2075s;
F4.0102->2.8547s; G4.3710->3.2174s. Optimized path disables decoder checkpointing,
uses implicit causal backbone mask and LM chunks512; peak~98GiB/GPU.

A fresh one-case FA4 retry is prepared under performance-fa4-20260929-a02;
only the benchmark adapter changed. Same8 GPUs,18 updates, BF16 objective/gradient
precheck, config copy/env verification and supervised fresh burn reclaim/restore.
Do not rerun the successful13 cases or reuse their outputs as fresh training.

## Authorized B200 performance investigation (2026-09-29)

Install93a404d succeeded. CPU-only preflight634b56b at12:08:39 UTC confirmed
perf_env torch2.14.0+cu130, transformers5.9.0, accelerate1.13.0, datasets4.8.5,
FA4 4.0.0b32, CUTLASS DSL4.8.0, cuDNN9.24.0.43 and Triton3.8.0; FA4 imports.
All eight GPUs have1000W limits and NV18 connectivity. Four CPU tests passed,
including native-Qwen/causal-mask/chunk/checkpoint loss and gradient equivalence.

Benchmark launch prepared: performance-20260929-a01 under the usual outputs
root,14 sequential eight-GPU cases at18 updates each, measured updates6-18;
profile update3 separately. Cases: current A/native A; current B/no decoder
checkpoint/B LM chunk512/causal-mask B/combined fast B/microbatch32 B; current
and fast F/G; fast B repeated in perf_env, then FA4 backbone B. Auxiliary
attention and F/G losses stay unchanged. Both microbatch settings use1M tokens
per update (16x4x8x2048 versus32x2x8x2048); summary verifies identical first
global batches and records data fingerprints. Original full schedule and data
pipeline retained; monitoring/checkpoint overhead excluded from timed updates.
Each case runs BF16 loss/gradient validation before the real training probe.
Use existing supervisor to reclaim verified burns, verify GPUs free and restore
burns on success/failure. Production trainer/model files remain unchanged.

User requests diagnosis of B200 throughput versus the earlier H100/H200 run,
and authorizes a separate FlashAttention environment and short training tests.
Keep the completed scientific runs and production trainer/model unchanged.
Install envs/perf_env.txt through controller #i, verify its versions/imports,
then benchmark the existing path and isolated performance variants using all
eight GPUs, the same packed inputs, global batch, seed and optimizer schedule.
Measure warmed-up updates separately from profiling/startup/checkpoint overhead;
record actual attention kernels and peak memory. Validate changed paths against
the original loss/gradients. Existing authorization permits stopping freshly
verified burns; use the existing train_then_burn supervisor for automatic final
handoff on success/failure. No node outbound network or direct package installs.

## B/F/G complete at 5,000; automatic burns verified (2026-09-29 11:51 UTC)

Read-only check 75816db confirms all five continuation queue jobs exited zero,
complete.json equals the successful run receipt, and comparison passed. Each
arm stopped at exactly 5000 / schedule 28600 updates, 5242880000 input tokens;
saved configurations match except arm and evaluation uses the same 4882 rows.
Final held-out LM loss: B 3.208608250096011; F 3.2087102048666094;
G 3.2090661402300626. F-B +0.00010195477059848912;
G-B +0.0004578901340517305. Essentially tied in this single-seed comparison;
no observed LM advantage for the additional F/G losses at this cutoff.

B finished 05:26:49 UTC, F 08:20:07, G 11:27:11; comparison completed 11:27:12.
Supervisor verified all GPUs free and automatic burn handoff at 11:28:22.
Live inspection 11:51:18-11:51:30 confirms eight approved burn workers at
100% utilization, 155212 MiB/GPU, all-rank readiness and advancing collectives:
cycles1570->1580, payload1743.25->1754.36 GiB. No handoff error; guard released.

F route loss0.17738269914479182; G route0.17724789010716982,
message0.0058306707835078485; weighted objective identities validated.
Fifteen small artifacts were retrieved and SHA256-verified against the source;
all completion-manifest hashes also match (including the saved staging receipt).
Local artifacts: artifacts/deep-kv-BFG-final-20260929/.
Raw log: temp/BFG-completion-check-20260929-1150.log, SHA256
ac2c0748999e084a5ac9ffa7a168e2b30f6d45dc3dbc560c92d9c54c1624b470.
Large model/optimizer checkpoints remain on B200. This check changed no GPU
processes or training code. commands.sh restored to #0; no further training queued.

## FlashAttention environment checked (2026-09-29 06:13 UTC)

Read-only check 96b2f4e verified B200 train_env: torch2.14.0+cu130,
CUDA13.0, Triton3.8.0, cuDNN9.24.0.43. PyTorch reports built-in FlashAttention
available and flash/memory-efficient/cuDNN SDPA backends enabled. Standalone
flash-attn, flash-attn-3 and flash-attn-4 distributions are absent; flash_attn
module is absent. Build availability/enabled flags do not establish which
kernel the training inputs dispatch to; no kernel profiling performed.
No CUDA context was initialized and no packages or training code changed.
Evidence: temp/flash-attention-env-check-20260929-a01.log, SHA256
aa4fede1b9a0be312ecb9a58cad04a8c829eeffe208d265eac9956dc073752f2.
commands.sh restored to #0. B/F/G queue remains as previously configured;
this environment check did not take a new training-progress snapshot.

## B complete at 5,000; F running; G queued (2026-09-29 05:33 UTC)

Read-only check ae585c5 at 05:33:38 UTC on thiennh-p6-78gg-worker-0 confirms
B finished successfully at 05:26:49 UTC (2h24m53s for the continuation).
Result and trainer state both report step 5000 / schedule 28600; input tokens
5242880000. Final held-out LM loss is 3.208608250096011, down from
3.505886970597037 at step 2500, on the same 4882 evaluation rows.
Pulled result SHA256 matches the successful queue job's artifact receipt.

F started automatically at 05:26:49 UTC and reached step 2589 by the check.
At step 2580: LM loss 3.504, route loss 0.1836, learning rate 0.0002988;
finite gradient norm. Throughput about 4 seconds/update. All eight B200 GPUs
have one train.py worker each, 99% utilization and 25486 MiB used per GPU.
G remains queued. Supervisor is running; final comparison/completion/burn
receipts are absent as expected while training continues. Approximate finish:
F around 08:20 UTC; G/queue around 11:30 UTC, subject to overhead.

Evidence: temp/BFG-status-20260929-0532.log, SHA256
3984a28376cf856ba07a02ecf19cf0d3c1687a24192455bf2f84e74a6eec7826;
verified small artifacts in artifacts/deep-kv-BFG-status-20260929-0533/.
This check did not change training or GPU processes. commands.sh restored to #0.

## B resumed on eight GPUs; F/G queued (2026-09-29 03:07 UTC)

Follow-up57d5674 at03:07:21 UTC confirms B reached2582, finite training loss
about3.51, finite gradient norms and LR0.0002988 continuing the original cosine
schedule. All eight GPUs remain97-99% utilized,25292MiB each. No restart from
zero or warmup reset. Evidence: temp/BFG-resume-progress-a02.log, SHA256
9a58e32c874c5e6d151c42551f6fea7a55f4ae930e31a9f1ca93b91ebaf0ab55.
commands.sh restored to #0; queue continues unattended. Approximate completion
11:30 UTC Sep29 (allow11:00-12:00), using prior B/F/G throughput and overhead.

Launch 0ce6bff succeeded on thiennh-p6-78gg-worker-0. Startup inspection864b793
at03:03:11 UTC verifies B progressed from checkpoint2500 through step2507;
all eight GPUs have one train.py worker,99-100% utilization,25292MiB each.
Supervisor/queue running; staging job passed in49.21s, then B started03:01:56.
All three copied checkpoints match original file SHA256s, including optimizer,
scheduler and all eight RNG states. Original checkpoint trees remain untouched.

Accelerate config was copied to /dev/shm/.cache/huggingface/accelerate/default_config.yaml;
byte/hash comparison and accelerate env verified MULTI_GPU,8 processes,BF16.
Only the eight freshly identified approved burn workers were stopped; the
supervisor recorded all GPUs free before the queue. Configs/data fingerprints
match earlier runs; ignore_data_skip=false, max_steps28600, warmup1430,
microbatch16, accumulation4, seed42,1048576 tokens/update. Cutoff5000 only.

Checksum-verified small startup receipts are retained in
artifacts/deep-kv-BFG-resume-startup-20260929/. Full inspection log is
temp/BFG-resume-startup-check-a01.log, SHA256
b8fdda4a53fa060c60d3d032eaff8b437f2ef5b8d1a539014e31261103454c8d.
Queue remains B -> F -> G -> compare, with automatic verified burn handoff
after success/failure. Expected total duration about8-9h, subject to throughput.
First resumed loss/LR check passed as recorded above. Future completion
requires each result.global_step5000, comparison/complete.json and live burns.

## B/F/G continuation to 5,000 updates authorized (2026-09-29)

User selected 5,000 total updates: resume each arm's complete checkpoint-2500
for 2,500 additional updates, sequentially using all eight B200 GPUs per arm.
Keep max_steps=28600, warmup_steps=1430, data/order/seed, batch and losses fixed.
Read-only preflight 4dbf931 at 02:52:58 UTC passed for all three checkpoints:
315 model tensors, 314 optimizer states, scheduler last_epoch2500 and LR
0.0002989680996734328, all eight RNG files, matching data/config and ample disk.
Evidence: temp/BFG-resume-preflight-20260929-a01.log.

Launch job: th2-78gg-deep-kv-BFG-5000-20260929-a01.
Fresh output: /mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01.
Original B is in deep-kv-2500-20260928-a02/production/run/B; F/G are in
deep-kv-FG-2500-20260928-a01/production/run/{F,G}, under the same outputs root.
The first CPU queue job copies complete checkpoints/configs, checks every
checkpoint file's SHA256, and records resume_inputs.json; originals remain
untouched. Native Trainer resumes the copied checkpoints, preserving optimizer,
scheduler and RNG, with the usual batch skipping. No model/trainer changes.

Local actual B/F/G copied-checkpoint continuation test passed: bit-identical
weights against uninterrupted training, unchanged source files, refusal of
reused destinations and incomplete checkpoints (tests/test_resume_staging.py;
temp/BFG-resume-staging-tests.log, 1 test, 5.715s).
Launch copies/verifies Accelerate config and prints accelerate env, verifies
and stops only approved burn workers, then records all eight GPUs free.
The existing supervisor restores/validates enhanced burns after queue success
or failure. Next: verify launch receipts and B progress beyond update2500;
then restore commands.sh to #0. Submission alone is not a running-job claim.

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

## Authorized B200 smoke after four-arm review (2026-09-27)

User requested another specification review and an eight-GPU B200 smoke.
Reviewed A/B/C/D against the mechanism document and pinned native Qwen3 code;
no new mechanism discrepancy found. Existing 46-test and eight-process CPU/BF16
evidence applies. Preparing a fresh read-only GPU/process/data preflight,
th2-78gg-deep-kv-refactor-preflight-20260927-a01, including copying the repository
Accelerate config to the actual default cache and checking accelerate env.
No GPU stop or training is included in this first command. Fresh ownership
inspection must precede the previously authorized burn stop and smoke launch.
Latest retrieved status still concerns old 96487e0 (BLOCKED, controller
06:53:34, Dropbox modified 13:53:37 UTC); it does not establish current B200
connectivity. Evidence: temp/deep-kv-smoke-review-status-20260927-a01.log.

## Follow-up correctness review (2026-09-27)

Fixed explicit resume accepting another arm's checkpoint: checkpoint paths must
now belong to the selected output directory and match its full schedule. Invalid
noninteger cutoffs and nonpositive/noninteger LM chunks are rejected. Queue list
arguments now follow HF CLI parsing; report_to lists with zero/one integration
are supported, while multiple integrations give an explicit CLI limitation error.
All 46 local tests passed (11.556 s), including cross-arm rejection, valid resume,
and generated queue parsing. Evidence: temp/deep-kv-refactor-review-tests-20260927.log.
Packing remains byte-identical; no remote launch or sampling change. Existing
eight-process BF16 CPU evidence below remains applicable; B200 smoke is pending.

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

## Prepared-data reuse and W&B output fixes (2026-09-27)

Implemented the findings in DEEP_KV_RECIPE_COUPLING_AND_WANDB_20260927.md.
TokenStream now matches only updates, tokens_per_update, context, eval_rows
and data_seed against the stored recipe. Preparation still records the full
original recipe. Source matching excludes microbatch; source paths, model-config
hash, shape, packing and token/order integrity checks remain. Operational changes
can reuse existing v2 token files in a new run. Full run identities still govern
checkpoint resume and arm comparisons; changing only a cutoff already worked.

Rank zero explicitly creates a scoped offline W&B run under the arm output
directory, or the operator's WANDB_DIR override, after output creation. The
standard HF callback logs to that run. It closes on success/failure and does not
leak a default directory to subsequent calls in the same process. Ordinary CPU
training and other ranks do not initialize W&B. Tests enable the real offline
SDK on CPU to exercise HF logging, separate arm paths, override and failure
cleanup. No re-preparation, GPU work, remote operation or launch was performed.
All 15 focused CPU tests passed (13.609s); compilation and diff checks passed.
Evidence: temp/deep-kv-data-wandb-tests-20260927-final.log.

## Trainer review and checkpoint recovery fix (2026-09-27)

Reviewed the four-arm model, loss normalization, fixed input order, BF16,
Trainer/Accelerate integration, cutoff and checkpoint recovery. Reproduced a
failure where HF rotation counted an old partial save and deleted the last
certified checkpoint before certifying its replacement. Native rotation is now
disabled; the save callback retains two certified checkpoints only after the
new save is certified. Partial directories remain ignored, not automatically
deleted. The training loop and experiment settings are unchanged.

All 14 focused CPU tests passed (7.537s), including the reproduced save failure
and exact resume with two data-loader workers. Eight-process CPU/Gloo BF16
resume passed for A/B/C/D at microbatch 16 and accumulation 4; maximum parameter
difference was 3.726e-9. The worker now asserts actual BF16 projection outputs,
not only the requested precision flag. Plan, shell syntax, compilation and diff
checks passed. Evidence:
 temp/deep-kv-review-checkpoint-repro-20260927.log (failure before fix)
 temp/deep-kv-review-tests-20260927-final.log
 temp/deep-kv-review-resume8-bf16-20260927-final/resume_verified.json
 temp/deep-kv-review-plan-20260927.json
No remote push, B200 access or real training launch. commands.sh stays #0.
CUDA/NCCL, fused AdamW and real-context memory/throughput remain hardware checks.

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

## Active recipe override: 30B English / 1M-token updates (2026-09-27)

User superseded the document's 1B-token / 32K-token batch pilot budget.
Deep-KV now plans 28600 optimizer updates x 1048576 input tokens per update
= 29989273600 input tokens per arm, with eight GPUs per arm. Context stays
2048; 512 global contexts / 64 per rank. Default microbatch 1 accumulates 64
passes; any divisor of 64 is configurable without changing global batch.
Warmup retains 5% of the full schedule (1430 updates), followed by the same
cosine decay to 10% of peak. Mechanism, seeds and other optimizer settings stay.

plan/make-jobs/train/report accept --stop-after N. Generated queues apply one
cutoff to A/B/C/D and report; they require exact step/token stopped.json receipts
and then continue sequentially. All cutoffs evaluate the full fixed 4882-row
split and save resumable checkpoints. A comparison records training_complete
false at a cutoff, while a default full queue still requires full completion.
Resume keeps the full LR schedule and data order; cutoff is not part of the
immutable training identity. Preparation always covers the full budget (~120GB
packed storage) and rejects insufficient data rather than repeating it. Old
1B prepared streams are incompatible; sampled text/prepare_data.py are unchanged.
Full packed-budget availability still needs validation during preparation.

Verification: 13 focused CPU tests passed (5.950s), 31 runner utility tests
passed (1.281s), eight-process Gloo cutoff/report and interrupted-vs-full
training checks passed for all four arms. Evidence: temp/deep-kv-30b-tests-20260927.log,
temp/deep-kv-30b-queue-tests-20260927.log,
temp/deep-kv-30b-cutoff-ddp8-20260927-a01/comparison.json,
temp/deep-kv-30b-resume-ddp8-20260927-a01/resume_verified.json.
Plan, generated queue parsing, compilation and diff checks passed. CPU only;
no remote push, GPU test, preprocessing or real training launch. commands.sh #0.
See DEEP_KV_TRAINING.md for current commands; older budget notes below are historical.

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


## Active: authorized B200 correction distillation

User explicitly authorized stopping all currently GPU-using workloads, including
vLLM and burns, after fresh identity checks; verify all eight GPUs free and launch
the next stage. This supersedes the previous instruction to leave deepeyes alone.
Do not use name-pattern or arbitrary process-group kills. Exact targets must
match the newly recorded PID/start-time/command-hash/GPU identities.

The fixed next-stage plan is PCC_DISTILLATION_PLAN_20260924.md: two same-checkpoint
feedback-off audits, two bounded capacity/resume checks, then four sequential
eight-GPU student runs (LM/PCC × two seeds), 6144 updates each, frozen Deep parent.
Local nine focused model/DDP/report tests passed in 23.517s. Full local
regression passed 124 tests in 144.467s; the additional fresh-reclaim safety test
passed separately (0.003s). All checks were CPU-only and offline in train_env.

**Launch blocked by infrastructure**, observed 2026-09-24 21:51 UTC:
commit 1d2d4b5, job th2-distill-ownership-cpu-ready-20260924-a01, controller
record 2026-09-24 14:50:42: FAILED(rc=5), "no Running worker pod for job
'thiennh-p6-tpbw' (context=<ctx>)". No remote inspection/test/reclaim/student
training executed. Current GPU/node state is unverified. User was asked to
restore/reconnect this worker or provide its new runner assignment. Per
AGENT_GUIDE.md infrastructure rules, do not resubmit until the system is repaired.
commands.sh is #0; source and the concrete launch script are committed.

Once the worker is restored: fresh read-only ownership/CPU readiness first;
then, without reasking the already granted workload-stop authorization, verify
that source matches readiness, hold the guard marker, revalidate/pin workload
identities, stop those workloads, require all eight GPUs free, and execute the
audits/capacities/four-run queue from an immutable source snapshot. Verify parent
checkpoints/data still exist on the restored node; do not recreate or substitute
missing parents without reporting it.

## Completed: longer eight-GPU B200 pilot

Verified on 2026-09-24: all six runs and the CPU report finished successfully
at 03:25:08 UTC. Deep beat both controls in both seeds; both predefined gates
passed. See PCC_JOINT_RESULTS_20260924.md. No follow-up training has started.
Live inspection at 2026-09-24 20:46 UTC confirmed an eight-GPU polite burn
under the deepeyes runtime alongside eight vLLM servers. These other workloads
were left untouched. All 45 compact result files are downloaded and verified
under artifacts/joint-v2-20260924/. commands.sh is inactive.

## Historical launch and progress observations

Latest read-only observation: **2026-09-23 23:27:15 UTC**, commit 682be7e.
Seed-0 Base finished successfully at 22:51:15 UTC, all 6144 updates and
201326592 input tokens; final 2M-token dev NLL 2.93309613, checkpoint verified.
Seed-0 Shallow is running at update 4847/6144. Four subsequent experiments
remain pending. All eight GPUs have one training process each; the burn guard
remains disabled. No final report yet. Status check changed no training state.

The user explicitly authorized B200 training, pushes to deep-llms/th2, stopping
the verified GPU burn, and **all eight GPUs per experiment** (six experiments
sequentially). They authorized longer training. No further stop approval is
needed for the verified burn. Scientific training is running (launch b4bc9f6).

Current fixed design is joint-v2-ddp: 6144 updates × 32768 global input tokens
= 201326592 tokens/run, all 28 pretrained layers, Base/Shallow/Deep × two seeds.
Warmup 307, monitor/checkpoint every 256, shared 400000-document source pool,
fixed 2M-token final dev. See PCC_B200_LAUNCH_20260923.md for all settings.
This supersedes the earlier one-GPU / 1536-update execution plan.

Remote main was initialized preserving th2 history. Recent deployment commits:
84e6ccf source; 0ecab91 successful read-only inspection; c07767e isolated env
installation; 955d9e5 exact model download; e6747eb burn ownership inspection;
cf0a8b2 CPU readiness; b61baf7 result export. No force pushes.

B200 CPU readiness v1 completed: all six model hashes verified, dependencies
passed, all 110 existing tests passed (30.275s). The fixed 50M training/2M dev
cache is valid but is superseded by the requested longer budget. Local two-rank
Gloo tests of the new DDP path passed, including exact resumed weights.

Node thiennh-p6-tpbw-worker-0 has eight B200s. Runtime pcc_joint uses torch
2.14.0+cu130, Transformers 4.57.1; train_env/eval unchanged. All model assets
are at the pinned ddc928... revision. Sampling completed 18:50 UTC, English
train 35 shards / 36595514 documents, eval 11822 documents.

Burn ownership verified at 21:25 UTC: workers 498–505, parent 431, worker start
ticks 258980483, parent start ticks 258980356. Script hash
3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857.
It uses eight-rank NCCL. Reclaim script rechecks all identities and pins process
handles before signaling only those workers; it does not signal PID 1 or groups.
If any identity changes, stop and re-inspect instead of widening the kill scope.

Current experiment root:
/mnt/local/_outputs/deep-llms_th2/joint-v2-b200-20260923-a03.
Readiness root: joint-v2-readiness-20260923-a01. All capacity checks and Deep
resume passed; seed-0 Base reached update 352 by 22:04 UTC. Fixed monitor NLL
was 3.00215843 initially and 2.97496419 at update 256. Checkpoint written.
A controller guard launched new burn workers during scientific startup;
commit b0ea30d stopped only freshly verified workers 27017–27024 and preserved
all scientific workers. At 22:13 UTC, Base reached update 1108; monitor NLL
at update 1024 was 2.96816706. Commit 76279e8 submits the supported DISABLED
marker lease until the pinned scientific queue exits. At 22:16 UTC its log
confirmed GUARD_DISABLED_FOR_AUTHORIZED_QUEUE and exactly one scientific worker
on each GPU (27540–27547). Base reached update 1532; monitor NLL at update 1280
was 2.96475494. Final code-only push sets commands.sh to #0; the isolated queue
and CPU guard lease continue. Retrieve results with a fresh #2 request.
The scientific job is live: do not restart it or edit its source snapshot.

The following paragraphs record earlier deployment attempts chronologically.

Remote DDP regression (7fa177c) passed all 115 tests in 33.983s; all model
hashes verified again. Longer input preparation is running. Local CLI allocation
guard was additionally tested (two DDP tests passed in 22.101s).

Submitting th2-joint-v2-eight-gpu-20260923-a01: it waits for CPU readiness,
checks exact 201M/2M counts and config, reclaims only the previously authorized
burn after fresh identity checks, then executes the gated capacity/6-run queue.
No unconditional training begins before those gates. Stop receipt:
/mnt/local/_outputs/deep-llms_th2/joint-v2-burn-stop-20260923-a01.json.

Launch 31f9a31 verified the longer inputs, then exited before any signal because
conda Python lacks os.pidfd_open. No GPU training or burn-stop occurred.
Retry a02 uses distro /usr/bin/python3 for the standard-library reclaim helper,
first requiring both pidfd_open and pidfd_send_signal. Training remains in
pcc_joint. Fresh experiment root is joint-v2-b200-20260923-a02; the verified
readiness root remains joint-v2-readiness-20260923-a01.

Retry 7acccaf confirmed system pidfd support, but system Python could not
resolve the scripts namespace. It stopped before any signal. Add explicit
scripts/__init__.py and set the guard's PYTHONPATH to the synced project root.
Retry a03 uses fresh joint-v2-b200-20260923-a03 and burn-stop-a03 paths; inputs
remain unchanged. No scientific runs have started at this point.

Launch b4bc9f6 succeeded in reclaiming the authorized burn with process handles.
The 21:55:53 UTC job log records LONG_INPUTS_VERIFIED and VERIFIED_BURN_STOPPED,
followed by eight-rank NCCL initialization and Base capacity update 1 with
finite NLL 3.156401. No traceback in the first snapshot. Source is isolated in
joint-v2-b200-20260923-a03/source. Exporting capacity reports and initial queue
progress next; do not resubmit the launch or reclaim command.

## Historical local readiness and screen scope

Latest request: carefully review/fix code and make it ready to launch the agreed
joint-training plan. Full-model training was absent; separate `pcc.joint` code
has now been implemented. Review is complete: 109 CPU tests passed; all three
full-size A100 capacity checks passed; real Deep checkpoint resume reproduced
the next update's loss and all weights exactly. Do not launch the six scientific
runs as part of this review. See `PCC_JOINT_READINESS_20260923.md` for evidence
and the unexecuted launch command.
Use `train_env`; B200, `prepare_data.py`, and `commands.sh` remain untouched.
New input/config/artifact root: `temp/pcc-joint-ready-20260923-a01/`.
`jobs.json` is the generated six-run sequential manifest with CPU input and
report stages. All bounded GPU checks have exited. Recheck availability before
launching a new workload. The source
experiment plan is `PCC_JOINT_TRAINING_PLAN_20260923.md`.
Earlier frozen-screen evidence follows.

Status: requested effectiveness test completed successfully on the local A100s.
Scientific decision: `stop_negative_screen`; none of four pairs qualified.
Deep feedback improved over Base, but matched shallow attention was slightly
better for every pair, with all paired 95% CIs favoring the shallow control.
Run: `temp/pcc-promise-screen-20260923-a01/`; all workers exited and GPUs were
released. Artifact audit passed. See `PCC_PROMISE_SCREEN_20260923.md` for results.

## Authorized scope

- Implement research-contract sections 11–13: correctness checks, layer screen,
  frozen-backbone adapter training, and sequential experiment execution.
- Latest user instruction: "run the test that show a method is promise or not."
  They requested pretrained Qwen3 0.6B on this dev machine and supplied
  `nguyenhuuthuat09/CulturaX_sampled` as the data source. Run a meaningful matched
  training/evaluation screen locally, keeping B200 untouched (reported down).
- Completed four fixed pairs `(4,16)`, `(4,20)`, `(8,20)`, `(8,24)`, one per local
  A100. All use pinned Qwen3-0.6B-Base, paired seed 1701, 128 updates, global
  32768 tokens/update, microbatch 4, the existing optimizer/schedule, and shared
  fixed train/dev inputs. Evaluate once after training and apply the original
  joint 1000-resample paired bootstrap/tie-break rule over all nine arms.
- The finalized B200 splits are unavailable. As stated to the user before
  execution, this is an exploratory local split from the user-selected source:
  English shard `en_part_00015.parquet` rows [0,20000) train and [20000,30000)
  dev. Same legacy packing and seed 20260922 for all arms. Training consumes
  4194304 input tokens; dev consumes 2000000. No test split or full probe.
- `design.json` froze this design before training; `data-check.json` records
  exact counts and input fingerprints. Scripts and logs are under the run root.
  Each worker invokes production `screen()` for one fixed pair; per-worker
  selection is deferred, and the controller calls the unchanged `select_pair`
  once using all pairs. No scientific gate, schedule, or training formula changed.
- Do not access B200 or change `prepare_data.py`, `commands.sh`, or remote jobs.
  All GPUs were checked free before launch; controller stops only its own
  child workers if a software failure occurs. No sub-agents were spawned.
- No sub-agents. Use `/home/users/thien/miniconda3/envs/train_env/bin/python`,
  cloned from `sparse_emb`; the source environment remains unchanged.

## Current interface

For real training, fill in a copy of `pcc.pipeline.example.json` and run:

```bash
conda run --no-capture-output -n train_env python -u -m pcc pipeline \
  --config temp/pcc.pipeline.local.json --output temp/pcc-training-001
```

This runs actual adapter optimization (screen, then an eligible full probe).
Do not pass `--check-only` or `--dry-run` when intending to train. These commands
are documentation, not an authorized B200 launch. The pretrained backbone stays
frozen; `pcc/screen.py`, `pcc/training.py`, and `pcc/probe.py` implement the method.
The separate legacy `train.py` is not the PCC entry point.

`python -m pcc pipeline --config <json> --output <fresh-dir>` reads:

- `model_path`: pinned local model/tokenizer snapshot.
- `train_data`: completed sampler English train directory (sorted shards).
- `val_data`: completed sampler English eval Dataset, used for validation.
- `microbatch`: optional, default 1, divisor of 16 contexts/update.
- `test_data`: optional independent test Dataset, opened only after dev gates.

Before screening, the normal pipeline validates full train/validation budgets,
matched policies, vocabulary bounds, and screen-prefix equality. It saves
`input-check.json` with exact input/target counts and SHA256 fingerprints of
ordered model inputs (IDs, masks, positions, segments). A failure stops before
screening and leaves `failure.json` without a completion marker.

Add `--check-only` to run synthetic model preflight and those same input checks,
then finish with `decision=inputs_validated`. It does not read test data or
train experimental adapters; preflight uses disposable adapters for gradient
checks. `--dry-run --check-only` previews the plan without loading model/data.
Normal execution includes the data checks automatically; a separate check job
is optional, and check-only outputs are not a training resume/export artifact.

The separate `pcc prepare` command, document-range schema, exporter examples,
and exporter-specific tests/docs were removed. Shared loading replaces that
stage. Internal Arrow caches stay inside the run directory and leave sampler
inputs unchanged. Existing NPZ inputs/fixtures still use the low-level reader.

Screen runs 128 optimizer updates per arm; full probe runs 610 per arm. Both
use 32,768 input tokens/update and fixed stage initialization seeds. Shared
packing follows the legacy single-process 1000-document map recipe, without
special tokens/separators, then context shuffling with seed 20260922. Screen
inputs are prefixes of the full streams. No new document split is selected.
The generic runner timeout is a wall-clock failure limit, not an update cutoff.

Without test data, a positive dev run finishes as
`validation_complete_test_not_supplied`, with no confirmatory test or scaling
recommendation. Fixed token budgets remain enforced: short validation inputs
fail explicitly; no repetition, resampling, or training-data borrowing.

## Evidence and remaining limits

Latest run: `temp/pcc-promise-screen-20260923-a01/` completed with status `ok`,
decision `stop_negative_screen`, no selected pair, and passing artifact audit.
All eight adapters completed 128 updates; nine arms were evaluated on the same
2M-token validation slice. Base NLL was 3.02107552; deep NLLs ranged from
3.00111982 to 3.00483784. Every deep-minus-shallow contrast was positive with
its entire 95% CI above zero. No full probe or pretraining was launched. This is
negative evidence under the tested setup, not a universal impossibility claim.
Train/dev source rows were disjoint and fixed across all arms, but this was not
a reproduction of the unavailable B200 splits. Full details and evidence links:
`PCC_PROMISE_SCREEN_20260923.md`.

Before the effectiveness request, a forward-only pretrained test passed real-text
wrapper loss/logit comparison, all post-block hooks, and future-token invariance,
but failed the native HF cached-vs-full logits comparison at atol .02 / rtol .01
(max absolute difference .375). Evidence:
`temp/dev-diagnostics-20260923-a01/pretrained-only-failure.json`. It was preserved;
no tolerance was loosened or result relabeled as passing. The user steered work
to the effectiveness screen, which uses `use_cache=False` throughout and the
previously verified full-sequence path. Native cached decoding remains a separate
unresolved diagnostic and is not used for this screen.


Latest local GPU diagnostics: pinned pretrained preflight passed 42 checks.
Real 2048-token contexts / 32,768-token updates passed for teacher and paired
students, with exact one-pass equivalence and unchanged frozen weights. Peak
allocated/reserved memory: 2.432/2.980 GiB. The permuted control failed before
its optimizer step because crossed bucket 69 contained one of 32,752 targets.
A deterministic diagnostic follow-up confirmed the bucket occupancy and finished
independent checks; final status is `blocked_permutation`, not all-passed.
Evidence: `temp/dev-diagnostics-20260923-a01/real-context-followup.json` and
`pretrained-preflight.json`. Detailed scope/results and limitations are in
`PCC_DEV_DIAGNOSTICS_20260923.md`. Those earlier checks alone established no
scientific benefit; the subsequent screen result is described above.


Latest training review (2026-09-23): **15 focused training/probe tests passed in
37.843 seconds**, CPU-only and offline in `train_env`. Evidence:
`temp/pcc-real-training-review-20260923-a01.log`. The strengthened full-training
test runs real teacher/shallow/PCC optimizer steps on a tiny Qwen fixture,
checks exact step counts and token logs, verifies learned weights and frozen
backbone identity, and reproduces evaluation from saved checkpoints. CLI help
also passed. Only documentation/help and test assertions changed this turn;
the actual training loops were already implemented. No pretrained-data run,
B200 access, sampler modification, or remote submission occurred.


New focused validation passed six real-loader/model input-check tests (2.208 s)
and twelve pipeline tests (29.593 s), including the real CPU CLI/runner flow.
Logs: `temp/pcc-input-check-focused-20260922-a02.log` and
`temp/pcc-input-check-pipeline-20260922-a01.log`. The first focused test attempt
was stopped because the new test class omitted the usual one-thread CPU setup;
that fixture was corrected before the passing rerun. Full regression passed
**94 tests in 92.146 seconds**, with no failures or skips, in `train_env` with
CUDA hidden and offline loading. Evidence:
`temp/pcc-input-check-regression-20260922-a01.log`. The CLI check-only help and
dry-run examples pass without opening input paths. No B200 access, model/data
downloads, scientific training run, or sampler/runner-command changes occurred.

Earlier direct-loader validation:

Focused checks passed: three direct-loader tests and eleven pipeline/runner
tests. Logs: `temp/pcc-direct-loader-tests-20260922-a01.log` and
`temp/pcc-direct-pipeline-tests-20260922-a01.log`. They verify legacy packing
agreement, deterministic prefixes, unchanged source files, shortfall/overlap
rejection, and real CLI → pipeline → next queued job with synthetic Arrow data.
Full regression passed **87 tests in 88.311 seconds**, with no failures or
skips, under `train_env`, CPU-only and offline. Evidence:
`temp/pcc-direct-data-regression-20260922-a01.log`. This also covers the positive
dev-gates/no-test branch, preserving the existing held-out test lock when test
data is supplied. CLI help, example-config dry run, and runner manifest listing
passed. No model/data downloads, pretrained experiments, or B200 access occurred.

The pinned pretrained snapshot is now local; see
`temp/dev-diagnostics-20260923-a01/inputs.json` for exact paths. Model revision:
`ddc928429ed09d9ad603fd762053d0434c15e865`. Source dataset revision:
`b19d850278693d37113c197857cc6328fa5c6881`, file `raw/en/en_part_00015.parquet`.
Weight/shard SHA256 verification against pinned Hub metadata is saved in
`download-verification.json`. The cached non-Base Qwen variant was not used.

The dataset repository contains raw parquet files, not the completed sampled
train/validation datasets. Fixed split replication, usable full eval budget,
and the B200 sampling-tokenizer revision discrepancy remain unresolved. Local
checks use the first 16 complete contexts (2048 tokens each), no special tokens,
no context shuffle, for one update per teacher/shallow/PCC arm. This is not
scientific evidence of PCC benefit. B200 remains untouched.

See `PCC_AUTOMATION.md`, `PCC_DATA_HANDOFF.md`, `PCC_DIAGNOSTICS.md`, and
`PCC_EXPERIMENT.md` for current commands, behavior, and contract coverage.
