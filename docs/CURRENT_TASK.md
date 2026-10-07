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
