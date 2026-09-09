# Current task

## Authorized full English evaluation then fine-tuning — 2026-09-09

The user authorized evaluation followed by independent fine-tuning for all six
arms at steps 250, 500, 1000, 2000, 3000, 4000 and 5000 (42 checkpoints), then
emphasized downloading missing benchmarks first. Keep one GPU per independent
job, up to eight concurrent jobs. Retain the documented three tasks and seeds
42/123/456 (378 fine-tunes), BF16 forwards and unchanged task hyperparameters.
No full sweep has launched yet. No cleanup, training restart or burn restart
is part of this submission.

Read-only execution97f82b2 verified all42 checkpoint configs/weight files are
present, all eight GPUs free (0 MiB), and22 TiB available. Existing HellaSwag/ARC
match all11 of their reference entries. Five repositories are absent, comprising
122 missing reference files; there are zero hash mismatches. Executiona5b3177
submits controller#d downloads of nyu-mll/blimp, EleutherAI/lambada_openai,
baber/piqa, allenai/winogrande and aps/super_glue into the existing
`/mnt/local/_data/deep-llms_th2/benchmarks/hf/<org>/<repo>` layout.
Download completion, all133 reference hashes, and full offline task/split
coverage must pass before launching the sweep. Evidence:
`temp/remote_logs/swt_benchmark_inventory_20260909_c01.log`, SHA256
`3c902461dfbb28808ce469ccdce5788041fcd6a6c59f7c4f8c002f4d457a0940`.
Earlier no-full-sweep authorization statements below are historical.

Downloads completed successfully: all five controller items reported OK.
Executionfbfeb3b then verified all133 reference files, offline loading of all78
evaluation tasks and the three training splits (HellaSwag39905, ARC-Easy2251,
XNLI-English392702), and the74-task tiny B0/C CPU scoring smoke. All eight GPUs
remained free. Terminal artifacts exported via8792aff were pulled and matched
to their printed source SHA256 values: coverage
`b31d9d35a8a408f725f089a108f79bc61805f0be6f3b5ed7c6075aa3defc6dd9`, smoke
`7cdb16ec2a77713f3710d87f61bc2c431fcd84f2a9cd371b02a9c88a5994427a`.
Local evidence prefix: `temp/remote_logs/swt_benchmark_verified_20260909_`.

Prepared `scripts/eval_finetune_capacity_b200.sh` for fresh output
`/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01`.
It checks42 checkpoint steps/hashes, all benchmark splits and the common PPL
cache once, validates84 eval/378 fine-tune job plans, copies/verifies Accelerate
and checks GPUs before each stage. Eval success requires full benchmark sample
counts and9829694 English PPL targets; fine-tune success additionally checks
the unchanged task hyperparameters and full three-epoch update counts. Source
pretraining files are untouched. No process signaling/deletion/burn restart.
Local17 eval/harness tests passed without skips (101.741s); the expanded three
handoff tests also passed, including rejection of incomplete PPL/changed weights.
Shared handoff commit:a7412c4. Executiona6c0ab8 submits the authorized sweep;
GitHub head was verified. All source files match development (only commands.sh
differs), and all20 pretraining-manifest entries still match. The launch first
checks the handoff SHA256 and runs its three CPU tests on B200, then executes
the foreground script. Remote progress confirmation is pending at this entry.
Do not repush its active#1 command to poll; use read-only Dropbox/#2 exports.

## Benchmark precision fixed; corrected B200 smoke passed — 2026-09-09

User approved fixing the confirmed evaluation-only bug and rerunning smoke.
`eval.benchmarks.evaluate` now supplies explicit HFLM mixed_precision_dtype
(BF16 or None for FP32) and softmax_dtype=FP32 instead of outer autocast.
Pretraining, PPL, diagnostic model calls and finetuning optimizer code are
unchanged. The regression gate checks all six arms/both precisions, FP32
softmax and FP32 master weights; the production smoke also records actual
post-finetuning benchmark-forward dtypes. Shared fix commit: bbcce4e.

Verification completed:85 local regression tests passed (174.311s), and a local
real-data smoke passed. Execution5665bf6 deployed the fix and verified all four
changed-file hashes plus the unchanged20-file pretraining manifest. Its B200
tests passed:15 CPU integration tests (71.861s),12 CUDA precision cases (six
arms ×FP32/BF16), and all six actual checkpoint5000 smoke workers (exit0).
All18 checkpoint/task cases observedBF16 in both pre- and post-finetuning
benchmark forwards and in fine-tuning updates. Each used two optimizer updates;
all six arms passed2048-token synthetic PPL and fine-tuned save/reload, with
unchanged original checkpoint weight/config hashes. These are bounded smoke
tests, not research evaluation scores or full fine-tuning runs.

After workers exited and30s elapsed, all eight GPUs were verified free. No
burns were restarted. The terminal pipeline marker is
`SWT_CORRECTED_BF16_SMOKE_SUCCESS`. Summary was exported through5211a3a and
downloaded to `temp/remote_logs/swt_smoke_fixed_final_20260909_summary.json`;
its SHA256 matches the independently printed source hash:
`bf89a99c724622fa125f694921b8073e8d63d6af59a067d8dc5996376b5c04f7`.
Remote corrected smoke output:
`/mnt/local/_outputs/deep-llms_th2/swt/eval_finetune_smoke_20260909_b01`.
CPU/precision/pipeline logs and per-arm reports use the same local filename
prefix. Full English-suite benchmark downloads/verification remain a separate
prerequisite; no allocation diagnostics were run.
No full benchmark sweep, cleanup, training restart or burn restart is authorized.
Previous failed smoke outputs below are preserved, not overwritten or relabeled.

## Eval/finetune smoke completed; benchmark precision fails — 2026-09-09

Execution189b56e completed the bounded destination tests. All14 CPU integration
tests passed (71.551s). Six checkpoint5000 workers, one per GPU0–5, each passed
the2048-token synthetic PPL path (6141 targets), two real BF16 optimizer updates
on each of HellaSwag/ARC-Easy/XNLI, finite weights, before/after benchmark
coverage, and a HellaSwag fine-tuned save/reload with identical probe logits.
Fresh original-checkpoint reload was used for every task. All six original
checkpoint weight/config hashes remained unchanged.

The CUDA precision gate and all18 production checkpoint/task smoke cases
confirmed the existing bug: requested BF16 benchmark calls actually useFP32.
Fine-tuning forward calls useBF16 correctly. Thus overall smoke success=false;
no full research evaluation is authorized or ready. The evaluation wrapper has
NOT been fixed; awaiting user direction on that change. Missing new English
benchmark downloads are also a separate prerequisite to the full suite.

All eight GPUs were verified free after workers exited and a30s wait. Burns
were not restarted. Results were pulled via40d2d98/82bc041 and checked locally:
`temp/remote_logs/swt_smoke_final_20260909_summary.json`, SHA256
`cee1315c89700b93a3324afdbe9149e03c12ee7f4c9da86fed539a023464cec4`, plus
pipeline/CPU/precision logs and six complete per-arm reports in the same folder.
Remote smoke root: `/mnt/local/_outputs/deep-llms_th2/swt/eval_finetune_smoke_20260909_a01`.
These outputs include one smoke fine-tuned model per arm and private smoke
input/cache directories; they are not research results or pretraining outputs.
The source pretraining20-file manifest still matches. Shared eval/diagnostic
code was deployed for these tests, but no allocation diagnostics were run.

All six arms completed step5000 at01:49:11 UTC; `training_complete.json`
and all six checkpoint sets were recovered and verified in the07:09 UTC
live check. Same node/GPU UUIDs, retained data/model/env, host uptime19.4 days.
The burn verifier timed out, but the independent burn was running on all eight
GPUs (155010 MiB each) with advancing communication counters.

User now authorized stopping those burns and testing eval/finetune, not full
research evaluation. Execution440a53f failed before signaling: system Python
could not import the project stop helper. Corrected7ad9562 activated `swt`,
checked the helper's absolute path, verified one known launcher/eight worker
PID+start identities, and signaled only those workers. After30s all eight GPUs
were free, with0 MiB. Evidence: `temp/remote_logs/swt_stop_eval_preflight_20260909_a02.log`
(SHA256745a39b015374bf3b98ad337b6087b33b6568ef5bb58b12a8b62326860adc7f3).
No checkpoint/data/cache cleanup or burn restart was requested or performed.

Preflight found no `swt_eval`/`lm_eval` conda environment; `swt` lacks lm_eval.
Old benchmark snapshots are under `/mnt/local/_data/deep-llms_th2/benchmarks/hf`.
Execution08e801d installed separate `swt_eval` via `#i envs/swt_eval.txt +a`.
Install succeeded07:31 UTC; imports verified07:35 UTC: torch2.14.0+cu130,
Transformers5.9.0, datasets4.8.5, Accelerate1.13.0, lm_eval0.4.10. Accelerate
config was copied/compared before tests (workers themselves are single-GPU).
No direct node downloads; the `swt` training environment was not modified.

Local14 eval/finetune integration tests passed in103.002s, including all-arm
fine-tuning, CLI round trip and queue validation (`temp/eval_finetune_review_20260909.log`).
The independent precision gate reproduces the OPEN benchmark BF16 failure on
all six arms (`temp/eval_precision_before_20260909.json`); FP32 cases pass.
User was asked whether to fix this evaluation-only bug; do not claim it fixed.
New bounded `scripts/smoke_eval_finetune.py` tests real benchmark subsets,
two optimizer updates at each task's normal batch/sequence settings, fresh
checkpoint reload per task, save/reload and unchanged input checkpoint hashes.
It fails if requested benchmark BF16 is not observed. Its small-subset outputs
are smoke tests, never research scores. Destination results are recorded above.

## Checkpoint diagnostics implemented locally — not deployed (2026-09-08)

Implemented the reviewed token-frequency/spectra/gradient design. See
DIAGNOSTICS.md: freeze one diagnostic bundle via `eval.diagnostic_data`, then
run `eval.diagnostics_checkpoint` or add `--diagnostic-bundle` to the existing
eval queue. With all diagnostics selected, six checkpoints produce 30 jobs
(PPL, benchmarks, frequency, spectra, gradients per checkpoint), not 12.
The current training/burn workflow has not been modified or submitted again.

Counts refer explicitly to the complete selected packed training pool, not
the actual exposure prefix of a stopped checkpoint. Production diagnostics
check training/eval fingerprints and preprocessing against saved train_config.
All artifacts and comparisons bind to one frozen manifest; per-checkpoint
consumed token budgets are separately reported for full-batch first-epoch runs.
Probe gradients are newly measured at fixed saved weights without optimizer
steps, not reconstructed historical gradients. The future training observer
is opt-in and is not attached in train.py. No live training source was changed.

The benchmark precision review remains an OPEN gate: `eval.benchmarks.evaluate`
uses outer autocast which lm_eval overrides, so its requested BF16 calls are
actually FP32. User deferred the B200 precision test. The executable regression
gate is now `scripts/check_eval_precision.py`; its dev report
`temp/eval_precision_gate_20260908.json` confirms all six BF16 cases fail while
FP32 passes. No benchmark precision fix was silently applied. Configure the
harness explicitly and require this gate to pass before full B200 benchmark
evaluation. Direct PPL/diagnostic model calls are not affected by this bug.

Initial verification: 82 full-suite tests passed (172.501s), followed by 11
diagnostics tests (15.560s) including additional BF16 all-arm checks. Tests
cover SVD/effective products, masked shifted loss versus PPL, bucket/hash errors,
fixed-probe accumulation, tied additivity, no optimizer/model mutations and
unchanged tiny Trainer weights when the observer is enabled. The initial
diagnostic loss-chunk off-by-one was found and fixed before these passes.
Evidence: `temp/diagnostics_full_suite.log`, `temp/diagnostics_tests_final.log`.
Final regression pass: 83 tests passed in 171.526s
(`temp/diagnostics_final_full_suite.log`). The expanded 12-test diagnostics
suite then passed in 15.106s, including a real train.py CLI checkpoint whose
saved training/eval fingerprints and consumed input/target budgets match the
prepared bundle (`temp/diagnostics_checkpoint_provenance_suite.log`). All 20
active training-source manifest entries still match. These remain tiny CPU
checks, not a production B200 memory/throughput or distributed-observer test.
No B200 data preparation, diagnostics, eval, finetune or GPU action was launched.

## English benchmark expansion — dev verified, not deployed (2026-09-08)

Added BLiMP (all 67 official subtests and a separate unweighted suite mean),
LAMBADA-OpenAI (accuracy and target-word PPL), PIQA, WinoGrande, ARC-Challenge
and BoolQ to the existing HellaSwag/ARC-Easy coverage. Definitions/scoring use
unmodified lm_eval 0.4.10. The eight-family core has 74 task results; defaults
also retain four earlier English tasks, giving 78. Fine-tuning tasks are unchanged.
See EVALUATION.md for selection, local data layout and future #d commands.

Dev verification: all 72 regression tests passed, with no skips, in 157.678s
(`temp/expanded_benchmarks_full_suite_20260908.log`). A separate offline CPU
smoke checked the complete evaluation split counts for all 74 tasks and scored
two real examples per task on tiny random B0 and C models; both passed.
Evidence: `temp/english_realdata_smoke_20260908_a02/complete.json` and its log.
These are correctness checks, not research scores or B200 throughput evidence.

Pinned dev snapshots are in `temp/english_benchmark_snapshots_20260908_a01`.
The tracked `resources/english_core_benchmark_files_20260908.json` records seven
source revisions and 133 file hashes/sizes for later B200 verification. The first
real-data smoke failed because a hand-pruned WinoGrande snapshot lacked other
configs referenced by README format inference. Downloading complete WinoGrande
and SuperGLUE parquet snapshots resolved this; use whole-repo #d downloads.
No B200 download, deployment, training change or GPU action was performed.

## Latest instruction — authorized fresh 5k screening (2026-09-08)

Latest progress, pulled18:54 UTC: B0 started18:30:17 UTC after cache preparation
and production smoke. The18:53:39 UTC pipeline snapshot shows step1656/5000,
approximately1.39 updates/s, latest training loss3.904. Step1000 evaluation
completed: loss4.204, scored_targets9829694, runtime131.4s. No project traceback
or failure marker observed. A128/A256/A512/C/D remain queued. Estimated total
wall time for all six: approximately7–8h including initial cache, evaluation,
checkpoint and handoff overhead; finish roughly01:00–02:00 UTC September9.
Other-arm timing is extrapolated from short batch benchmarks, not completed
production runs. Evidence: `temp/remote_logs/swt_5k_progress_20260908_1854.log`,
SHA256 `ef3088f8709209d340023ed5d46fccc88c63274629fdc185d590ca908f652747`.
The displayed28575-step progress total belongs to the full-epoch LR schedule;
the screening callback still stops at5000. Execution `00138b1` only exports
the pipeline log; it does not signal/restart the job.

The user authorized restarting all six arms at5000 steps (approximately5.24B
input tokens per arm), with the normal GPU/config gates and post-success burns.
Shared implementation commit `b43aecf` adds SWT_STOP_AT_STEP (default10000) and
passes it consistently to preprocessing coverage checks, train.py's stop callback,
per-arm checkpoint verification and the aggregate completion marker. This launch
sets SWT_STOP_AT_STEP=5000. Model/trainer/data code and the full-epoch LR schedule
are unchanged; effective batch512, eight GPUs, BF16 and English-only are retained.
Eight local handoff tests passed, including5k completion and wrong-cutoff rejection.
New source manifest: resources/swt_qwen_launch_5k_20260908.json (all20 files verified).

Execution checkout contains shared commit `e486957` and launch commit `594b3ac`.
Fresh output: `/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01`.
Order B0 -> A128 -> A256 -> A512 -> C -> D. It first checks GPUs, waits30s/checks,
copies/parses/compares Accelerate, waits30s/checks, runs destination handoff tests,
then regenerates cache and runs production smoke/training through the existing
foreground queue. Every arm must verify checkpoint5000 before training_complete.json;
only then does the queue wait/check GPUs and launch/verify communicating burns.
Remote startup confirmed at18:09:41 UTC:5000-step cutoff banner, all20 source
hashes, fresh paths and all eight GPUs free. Accelerate was copied and verified
at `/mnt/local/.cache/huggingface/accelerate/default_config.yaml`; all eight
destination handoff tests passed. At the18:11:53 UTC snapshot,160-worker English
cache tokenization was616k/36.596M documents (~2%); GPU training had not started.
No project error appeared. Evidence: `temp/remote_logs/swt_5k_start_20260908_1812.log`,
SHA256 `f8e1a707d4a01d3d66ab0fb17d61f01bd5127205c85ebae30ecfb8bfcd83c287`.
Use read-only logs, never resubmit to poll.
Older cancellation/no-relaunch instructions below are superseded by this request.

## Latest instruction — cancel and clean a03 (2026-09-08)

The user requested cancellation and cleanup again, not a5k relaunch. Execution
`db212a7` inspected the live queue: PID152404/start163046836, parent152229,
Accelerate155673 and GPU workers155683–155690, all belonging to a03 B0.
The17:53 UTC snapshot showed B0 aroundstep841. Evidence:
`temp/remote_logs/swt_a03_cancel_ancestry_20260908_1755.log`.
Execution `872f4c4` submits the unchanged verified stop helper for that exact
queue identity, then confirms no remaining active Stagewise workers before
deleting only a03 output and qwen_en_map160_batch1000. It preserves sampled
Qwen data, tokenizer/model and unrelated HF benchmark cache. It verifies
absence of cache-/tmp- leftovers in Stagewise directories and all GPUs free.
Completion confirmed at17:58:34 UTC. The exact queue and eight verified GPU
workers were stopped; launcher exit and no active Stagewise workers were checked
before deletion. Both target directories are absent, and both Stagewise parent
trees have zero cache-/tmp- leftovers. Protected file inventories are unchanged.
All eight GPUs report0 MiB/0% and no compute processes. No training/burn restart.
Evidence: `temp/remote_logs/swt_a03_stop_cleanup_20260908_1800.log`, SHA256
`4e79102d851e45f746a540351ce7a61368bc99031b5a4b63781f83ccb5e2898a`.
Execution commands.sh is returned to inactive #0 after verified cleanup;
the earlier training-to-burn authorization is superseded by this cancellation.

## Latest instruction — fresh six-arm rerun submitted (2026-09-08)

The user authorized checking GPUs, rerunning training, copying Accelerate and
starting burns after verified successful training. Execution commit `f528935`
submits the unchanged tested `scripts/train_capacity_b200.sh` for
`B0 A128 A256 A512 C D`, with fresh root
`/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03`.
It uses batch16/accumulation4, eight GPUs, BF16, English only, seed42,
stop-at-step10000 and the full one-epoch LR schedule (no max_steps override).
The wrapper rejects live Stagewise queues and existing old/new output/cache,
checks GPUs, waits30s/checks, copies and verifies the actual Accelerate cache
config, waits30s/checks, and runs destination handoff tests before the queue.
The queue must regenerate the deleted English preprocessing cache, then repeat
GPU/Accelerate gates, run production smoke, train/verify all six arms, publish
training_complete.json, wait30s/check GPUs, and launch the verified communicating
burn in its own persistent tmux session. Failure prevents the burn handoff.
All20 source-manifest entries match both development and execution checkouts;
seven local handoff tests passed. Remote startup is now confirmed: the
17:22:11 UTC log verifies all eight GPUs free, source hashes and fresh paths;
Accelerate was copied and parsed at its actual HF cache path. All seven
destination handoff tests passed. By the17:25:40 UTC snapshot, English cache
tokenization was approximately10% (3.506M/36.596M documents), using160 workers.
No project error appeared in that log. GPU training has not started yet; the
foreground queue will proceed automatically after cache preparation and gates.
Evidence: `temp/remote_logs/swt_a03_start_20260908_1726.log`, SHA256
`76094ba147a3b7ad72ec9aa1fec4079d198bea76609105c3524de6070ceaf463`.
Use read-only log retrieval to verify; never repush the executable command to poll.
Previous cleanup/no-relaunch restrictions below are historical, superseded by
this explicit rerun authorization. No additional deletion is authorized.

## Latest action — authorized cleanup completed (2026-09-08)

After reviewing batch-test results, the user requested removal of the previous
run outputs and cache. Execution commit `f9a1c5f` completed cleanup at16:54:52 UTC;
`b294d8b` pulled its completion log without repeating deletion.
Removed these exact B200 directories, including all contents and temporary files:

- `/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02`
- `/mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_20260908_a01`
- `/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000`

All three were verified absent. B0/checkpoint-6250 is now deleted, not resumable
from that path. The sampled Qwen corpus, local Qwen tokenizer/model and unrelated
HF benchmark dataset cache were preserved; file paths/sizes/mtimes matched before
and after. All eight GPUs were free after cleanup. No training or burns launched.
Local benchmark JSON/logs remain available. Evidence:
`temp/remote_logs/swt_cleanup_complete_20260908_1657.log`, SHA256
`4619ceb2a540df633104d18d9d8a473cefe8a8a94d7a487aa6fd513999a65e8d`.
The following stop/benchmark notes are historical and do not describe retained
remote artifacts after this cleanup. A new run must regenerate preprocessing cache.

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
At15:28:42 UTC, its startup log confirmed source hashes, three destination CPU
tests passing, free-GPU checks and copied Accelerate. The first profile
`B0_b16_a4` started at15:27:53 UTC. All 12 profiles finished successfully at
15:44:49 UTC; results were pulled and validated at16:32 UTC. No OOMs. All
eight ranks recorded 20 finite positive measured update times per profile;
the data fingerprint and effective batch512 matched throughout. Batch32
improved median update throughput by only approximately0.7–2%, while peak
reserved memory rose from75.9–82.8 GiB to150.0–163.2 GiB per GPU. These short
measurements do not demonstrate a material end-to-end speedup. All GPUs were
free at completion (historical snapshot, not a live usage check).
Evidence: temp/remote_logs/swt_batch_progress_20260908_1629.log and
temp/remote_logs/swt_benchmark_complete_20260908_1632.json. Research training
remains stopped; B0/checkpoint-6250 is retained and no burns were started by
the benchmark queue. Detailed comparisons are in PROJECT_NOTES.md.
It rechecks free GPUs between isolated profiles and never resumes research
training. Refresh logs with #2, not by resubmitting its executable #1 command.
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
