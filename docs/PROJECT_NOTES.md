# Project notes

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
