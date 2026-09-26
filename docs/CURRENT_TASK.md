# Current task

## Active: authorized six-language sampling on 78gg (2026-09-26)

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
Current preflight job: th2-78gg-sampling-preflight-20260926-a01. No sampling
started yet. Existing GPU workloads remain untouched; sampling hides GPUs.


## Active: CulturaX download on verified 78gg environments

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


## Current: install runtime on the replacement B200 (2026-09-26)

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
