# Project notes

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
- Packing follows the legacy single-process map recipe: 1000 documents per
  batch, no special tokens/separators, per-batch remainder dropping, then
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
