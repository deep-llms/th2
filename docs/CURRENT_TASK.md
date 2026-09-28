# Current task

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
