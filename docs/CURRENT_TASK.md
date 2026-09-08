# Current task

## Latest instruction — stop training and benchmark batches (2026-09-08)

The user explicitly authorized stopping the active six-arm queue and testing
larger per-device batches. Preserve all outputs/caches. Do not automatically
restart full research training or bypass its strict resume-config checks.
Compare batch16/accumulation4 against batch32/accumulation2, keeping8 GPUs,
effective batch512, BF16, sequence2048, the same actual Trainer and English
cache. Test all six arms in isolated processes,5 warm-up +20 measured updates.

**Stop confirmed at15:17 UTC; pulled at15:25 UTC.** Execution commit855bde4
eventually ran successfully. It stopped the exact queue109572 and all eight
verified workers112904–112911, waited30s, and verified all GPUs free plus
queue/launcher exit. Snapshot: every GPU0 MiB /0% utilization. The latest
retained checkpoint passing validate_resume_checkpoint was B0/checkpoint-6250.
No outputs/caches were deleted. Evidence:
temp/remote_logs/swt_stop_confirm_20260908_1525.log, SHA256
94d325171ac2784a63fa166e572b933ac9945fcf48431b7437e0dcf38cae949b.
The earlier response delay is historical; no model-code failure was reported.

On the user's follow-up check, continued the already-authorized benchmark:
execution commit39d5c9c submits fresh source/hash checks and destination CPU
benchmark tests, then scripts/benchmark_batches_b200.sh with fresh root
/mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_20260908_a01.
It rechecks free GPUs, copies/verifies Accelerate, waits30s/rechecks, then
profiles all arms. Submission is not yet confirmation that GPU tests started.
Do not relaunch full research training from this benchmark request.

Ancestry evidence: temp/remote_logs/swt_before_batch_stop_20260908.log.
At15:01 UTC it showed B0 workers112904–112911, Accelerate launcher112894,
and dedicated queue PID109572/start161614436 with exact argv
`bash scripts/train_capacity_b200.sh <a02-run-root> B0 A128 A256 A512 C D`.
The parent commands shell, tmux server88744, and PID1 are not stop targets.
These are historical identities; scripts/stop_capacity_queue.py verifies
queue start/argv and fresh worker ownership before signaling individual PIDs.
It disables only the dedicated queue, then stops verified training workers,
waits30s/checks all GPUs free and checks queue/launcher exit. No process-group
or name-based kills. The submitted command then checks retained checkpoint
state. B0 checkpoint directories through5250 existed at the ancestry check;
their final completeness has not yet been reported by the stop command.

Benchmark source: development826db18 (stop/helper/tests) andf1de3bd (shell queue).
The Python helper was deployed with855bde4; the shell is now deployed with
39d5c9c. `scripts/benchmark_batches_b200.sh` needs a fresh root
matching `/mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_*`, copies and
verifies Accelerate, waits30s/rechecks, and runs isolated profiles. It records
OOM as an unusable configuration; non-OOM failures stop for investigation.
It neither resumes training nor deletes files nor automatically starts burns.
Do not confuse benchmark_complete.json with research training completion.
Local checks passed:3 benchmark unit/Trainer tests,2 stop identity tests,
and a real two-rank CPU Trainer benchmark. Evidence:
temp/batch_benchmark_tests.log and
temp/batch_benchmark_ddp_review_20260908_a01/result.json.

## Active scope — 2026-09-08

User authorized correcting/verifying Stagewise, safely stopping verified GPU
burns, launching training, verifying completion, then restarting communicating
burns. The user explicitly selected all six arms in order:
B0 -> A128 -> A256 -> A512 -> C -> D. No dataset/checkpoint cleanup is authorized.

Launch preparation (2026-09-08): the incomplete-resume and label-smoothing
gaps are fixed. All checkpoints require optimizer/scheduler/RNG/weights;
nonzero label smoothing and model-only saving are rejected. DDP world size
and effective batch are recorded and checked on resume. 51 tests plus the
two-rank BF16 CPU smoke passed, including handoff ownership checks and
completion-validator rejection of nonfinite saved weights. Evidence:
temp/qwen_launch_final_tests.log and temp/qwen_prelaunch_fixed_ddp_bf16.json.

Read-only runner preflight 137fb5d completed at 12:36 UTC. Same B200 node,
eight known burn workers 99571-99578, parent 99482, /usr/bin/python3 -u
/tmp/llm_pretrain_burn.py. These are historical observations, not PIDs to kill
without fresh identity checks. The burn hash matches resources/llm_pretrain_burn.py.
35 English training shards (36,595,514 documents) and 11,822 eval documents
are readable; local Qwen3 tokenizer directory exists. swt imports confirmed.
Evidence: temp/remote_logs/swt_qwen_preflight_20260908_a01.log.

Workflow scripts: scripts/train_capacity_b200.sh prepares the shared HF map
cache while burns stay active, then reclaims freshly verified workers,
waits 30s/checks free, copies/verifies the eight-GPU BF16 Accelerate config,
waits 30s/checks free, runs a production-shape smoke, then sequential training.
Each arm: English, six layers, BF16 SDPA, batch16 x accumulation4 x 8 GPUs,
one-epoch cosine-with-min-LR schedule, stop-at-step10000, seed42, same cache.
After successful verification it writes training_complete.json and starts
the verified /tmp burn in a persistent tmux terminal; burn_verified.json
requires all eight ranks and advancing communication counters.
Development implementation is committed/pushed as64472ea. Execution commit
50340a8 submitted the six-arm workflow, but the runner blocked it before any
shell execution. Status: `BLOCKED | GUARDRAIL: outbound push pattern(s):
train.py:126:    if training_args.push_to_hub:`. That line rejects Hub uploads;
it does not perform an upload. This is a false-positive runner guardrail.
Evidence: temp/remote_logs/swt_launch_status_20260908_a01.log (Dropbox modified
2026-09-08T12:50:36Z; controller status line05:50:30). No burns were stopped,
no new cache preparation or training started.

Latest user instruction: remove the added upload-option guard, follow the
proven sparse-embedding default (Hub uploads disabled), and explicitly
document the outbound prohibition and this case in local AGENT_GUIDE.md.
The guard has been removed locally; there are no added upload calls or
launch flags enabling uploads. The launch manifest was updated and all20
source hashes verified. All51 tests passed again after removal
(temp/qwen_remove_upload_guard_tests.log); the inherited upload option remains
False and the launcher does not enable it. Development correction 792ea47 is
pushed to origin. The blocked a01 command did not create its intended run root.

## Authorized retry — observed 2026-09-08 13:25:50 UTC

Execution commit **1226141** was pushed to th2 for job
`th2-swt-qwen6-all-six-train-10k-then-burn-20260908-a02` after another local
51-test pass. Its source manifest passed, the node/environment and burn
ownership checks passed, and **all 51 CPU tests passed on B200 in 5.366s**.
The pipeline has entered CPU cache preparation: last exported progress was
563,000 / 36,595,514 training documents tokenized. Existing burns remain
untouched at this stage. GPU smoke and optimizer training have not yet been
observed; do not describe this as training already underway.

Fresh run root:
`/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02`.
Evidence: `temp/remote_logs/swt_retry_start_20260908_a02.log`, exported run
log dated 13:24:14 UTC, Dropbox modification 13:25:50 UTC. Local SHA256:
`9af005cf23464083739b3f4d7b2bae1248f2d8bdbe306be24cf2def8e22ae4c1`.
No outputs/caches were deleted. The shared cache is
`/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000`.

### Later verified progress — 2026-09-08 14:05:29 UTC

Read-only log exports a390631/2f9086a confirmed cache preparation completed,
the Accelerate configuration was copied/verified, and all six production
BF16/eight-rank GPU smokes exited successfully. The first training arm **B0**
is active: step1487/10000, training elapsed20m21s, approximately1.38 steps/s.
Step1480 training loss3.965; step1000 validation loss4.20 on9,829,694 targets.
The progress bar's28575 steps describe the full-epoch LR horizon, not the
10000-step cutoff. Evidence: temp/remote_logs/swt_progress_full_20260908_1405.log.
No pipeline-failure marker or traceback was present. Based on B0 and relative
production-smoke timings, provisional all-six completion is around03:00–04:00
UTC September9, excluding failures; revise from later arms' actual timings.

Read-only export176f1fa at14:30:34 UTC confirms continued B0 progress:
step3182/10000 after45m29s, still1.38–1.39 steps/s; step3180 loss3.623,
step3000 validation loss3.619 (same9,829,694 scored targets).
Evidence: temp/remote_logs/swt_progress_20260908_1430.log. The provisional
all-six finish window remains03:00–04:00 UTC September9.

Next: retrieve fresh progress, without resubmitting the executable command.
The running script automatically performs burn reclaim, both 30s/free checks,
Accelerate copy/verification, production GPU smoke, all six sequential arms,
completion validation, and communicating-burn handoff. Completion requires
`training_complete.json`; successful burn handoff additionally requires
`burn_verified.json`. A failed gate stops the pipeline without declaring
success or starting a burn over training. Report any failure before deciding
on recovery; do not assume any later stage succeeded from startup alone.

## Evaluation/fine-tuning preparation — local development only

User requested English-only evaluation/fine-tuning for this run, retaining
explicit language selection for future runs. New entry points:
`eval/eval_parallel.py`, `eval/eval_checkpoint.py`, `finetune/run_all.py`,
`finetune/train.py`. See EVALUATION.md for protocol, offline inputs, examples,
and known differences from historical fine-tuning. These files have not been
deployed to B200 or appended to the current training-and-burn workflow.
Pretraining source and the execution commands remain unchanged by this work.
Verify evaluation environment and actual benchmark snapshots on B200 before
any authorized full evaluation launch. Do not install into active `swt`.
Verification: all62 tests passed in109.606s using the isolated dev
`temp/evaluation_test_env` with pinned lm_eval0.4.10 and inherited swt ML
dependencies (no changes to swt). Evidence: temp/evaluation_final_full_suite.log.
Tests cover real offline local-parquet harness scoring for all six tiny model
types, BF16 CPU fine-tune updates, fine-tune/save/evaluate round trip, BPE
boundary matching, language coverage and queue failure propagation. No actual
B200 benchmark data or GPU evaluation throughput has been validated yet.

## Current implementation

- Model arms B0/A128/A256/A512/C/D now use Qwen3; stock B0 has 249,969,152 parameters.
- train.py uses HfArgumentParser, TrainingArguments, saved HF text datasets,
  multiprocess Dataset.map, cached tokenization/packing and ordinary Trainer.
- Existing sampled English Qwen data is the intended input; verify B200
  train/en and eval/en directories before launching.
- Superseded serial sampler, PackedTokens, train_capacity.py, GPT-2 manifests
  and its B200 preparation launcher are removed from active source.
- See CAPACITY_EXPERIMENTS.md for current architecture counts and commands.
- The historical research drafts are not the current execution specification.

## Remote state (last verified)

Serial sampler PID 106043 was stopped at 2026-09-08 08:42 UTC by runner commit
cb5b9d5. The first stop command failed before signaling because pidfd_open was
unavailable; the corrected command verified PID/start-time/argv, sent SIGTERM,
confirmed exit and stable output sizes. Evidence:
temp/remote_logs/swt_stop_sampler_20260908_a02.log.
Partial GPT-2 output remains on B200 (not deleted or reused):
/mnt/local/_data/deep-llms_th2/swt/english_gpt2_10b_seed0_20260907_a01.
All eight GPU burns remained active (155010 MiB, 100% utilization each).
These are historical sampler-stop observations; the Qwen cache-preparation
retry above is now active.

## Infrastructure

Development: /disk/thuat/stagewise_widening_transformer, main,
nguyenhuuthuat09/stagewise_widening_transformer.
Execution: deep-llms/th2, main, checkout /disk/thuat/th2_runner_clean_probe.
Machine: thiennh-p6-oish-worker-0, 8 B200; @PROJECT@ = deep-llms_th2.
swt environments: /home/users/thien/miniconda3/envs/swt (dev),
/mnt/local/conda-py311/envs/swt (B200). Both clone each host's sparse_emb.
Dev torch 2.7.1+cu118, B200 torch 2.14.0; Transformers 5.9.0 on both.

## Next gate

Local verification passed: 42 unit/integration tests
(temp/qwen_rewrite_final_tests.log), all-six-arm BF16 CPU and two-rank Gloo
model smoke, and a real two-rank CausalTrainer D run with cached text preprocessing.
Distributed evaluation scored exactly 378 synthetic targets with NLL
4.5103965777; standalone evaluation scored the same targets with NLL
4.5103966587 (difference about 8.1e-8). Tiny tests are not production throughput
or B200 CUDA evidence. Reports: temp/qwen_bf16_cpu_smoke.json,
temp/qwen_ddp_cpu_smoke.json, temp/qwen_trainer_ddp_run/result.json,
temp/qwen_trainer_ddp_nll.json.

Destination CPU tests, cache preparation, Accelerate configuration and
production GPU smoke have passed. The active training queue must still finish
and pass every checkpoint verifier before its final burn handoff.
