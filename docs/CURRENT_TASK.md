# Current task

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
