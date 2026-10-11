# P6-iso synchronization and controlled cooldown

User authorized: profile → exact synchronization/diagnostic changes → remeasure
A/P6 → controlled cooldown branches → evaluation before adding more arms.

## Profile

Two disposable 30-update fits on eight B200s, original FA4, micro16/GAS4,
2048 tokens, seed/data seed1042, full28600 schedule/warmup1430. Update6 profiled;
updates10–25 provide a short preliminary timing sample. No compilation.
Fresh root: `q359-p6-profile-20261011-a01`; launch `63e9849`.

| Rank0 measurement | A | P6-iso |
|---|---:|---:|
| Complete start-to-start interval, seconds | 2.06055 | 2.22963 |
| CPU scalar reads in profiled update | 29 | 33 |
| CUDA stream synchronization calls | 38 | 42 |
| Complete statistics callback CPU time, ms | 1.419 | 2.348 |

P6 auxiliary cosine kernels total about23.31ms forward +43.89ms backward.
Estimator kernels total3.75ms forward +4.14ms backward. Other extra work lies
in decoder injection and target/statistics preparation. Kernel sums are not
critical-path wall times. CPU synchronization waits include useful GPU work;
the roughly611–666ms summed waits are **not** a removable-overhead estimate.
The existing production update timer excludes part of the parent callback.
The revised experimental timer includes it and records all-rank start-to-start
intervals, including logging/data delivery. Raw component profiles and rank0
timings: `artifacts/p6-profile-20261011/`.

## Exact candidate and acceptance

Opt-in `--proxy_sync_optimization true`; default remains false:

- Read `mu_initialized` once per `train()` after restore, and cache it after
  bootstrap. The actual normalization buffer remains unchanged.
- Compute normalization diagnostics only on logging updates; perform identical
  normalization updates and validation every update.
- Transfer reduced loss statistics once for both validation and logging.
- Disable HF's redundant NaN *logging filter*. The existing finite-objective
  check still runs on every microbatch before backward and raises on failure.

No changed attention, loss, gradient routing, reductions, EMA arithmetic,
clipping, seeds, packing, optimizer, numerical precision or compilation.
CPU tests compare actual A/P6 runs and resumes: exact final model, optimizer,
scheduler, RNG and logged diagnostics, plus invalid-loss/moment rejection.

GPU acceptance uses two actual eight-GPU updates from copied P6-iso10k
checkpoints, with deterministic FA4 for both implementations. Require exact
restored model/optimizer/scheduler/RNG, input batches, clipped gradients and
final model/optimizer/scheduler hashes. Then eight disposable100-step timing
fits compare reference/optimized A/P6, in forward/reverse order, with production
FA4. Report max-rank complete intervals21–90, retain outliers, verify first/last
batch hashes and final32-row LM NLL spread≤0.002 within each architecture.
This is an execution check, not evidence of improved learning.

## Cooldown implementation

`python -m deep_kv.cooldown --checkpoint ... --output ... --start S --steps D`
copies and checksums every checkpoint state into a fresh branch. Original
weights/optimizer/RNG/config remain unchanged. `train.py` accepts
`--cooldown_start S --cooldown_steps D --stop_after S+D` only on a verified
checkpoint fork. Keep original `max_steps=28600` and `warmup_steps=1430`.

The original scheduler multiplier is unchanged through stepS. For scheduler
index `s>S`, use `original(S) * max(0, (S+D-s)/D)`. Thus the first cooldown
optimizer update uses the saved anchor LR; after D updates the scheduler is
zero. All parameter groups preserve their LR ratios. Native Trainer restores
optimizer moments, RNG and data order. A recorded cooldown cannot be retimed or
removed on resume. Ordinary resumes remain strict. All checkpoints are retained.

CPU actual-Trainer tests cover A/P6 forks, LR boundaries, interruption and resume
with exact final state and batch order, and original-checkpoint preservation.

Planned comparison: P6-iso10k +1000 cooldown updates, A10k +1000 updates, and
the existing A10.8k checkpoint with a time-budgeted cooldown. Use **actual past
runtime**, not a retroactive optimized cost: A10.8k22758.6961s vs
P6-iso10k22743.7338s. Choose the last branch's duration using measured suffix
runtime, then report actual cumulative time and residual mismatch. Do not
claim a time match if the difference exceeds the declared tolerance. Full
fixed validation PPL is the first outcome; no new architecture arms are queued.

The GPU exact/timing checks have completed; cooldown training remains active
as of 01:39 UTC. See the latest results below.

## Execution status

Profile complete; all8 communicating burns restored00:39:17UTC and verified in
receipt72aab3c. Raw profile artifacts match runner SHA256. Exact/timing study
submitted as e70bbde; preflight passed and reference10k checkpoint restore was
running at00:48UTC. No model/loss code changes during this study.

Cooldown orchestration waits for successful study completion and burn restoration,
then verifies its own environment and Accelerate configuration. Any failed gate
prevents launch. A separate two-update, eight-GPU cooldown smoke must finish with
zero final LR before the1000-update branches. The time baseline duration uses
completed P6/A suffix runtimes and the actual original cumulative costs. Final
report verifies zero LR, complete optimizer/RNG checkpoints, matching data recipes,
unchanged source files, PPL and the actual residual time gap.

For this decision stage, evaluation is full fixed validation PPL; no new
fine-tuning studies or architecture arms are launched.

**Verified00:57UTC:** the two-update eight-GPU exact-state gate passed. Reference
and optimized restored model/optimizer/scheduler/RNG, every audited batch, clipped
gradients and final model/optimizer/scheduler hashes match exactly with
deterministic FA4. Gate and rank0 audit artifacts pulled with matching SHA256 to
`artifacts/p6-sync-20261011/`. Production-FA4 repeated timings are active (first A
fit complete). Cooldown waiter is live; no cooldown training or final evaluation
has completed yet.

**Verified 01:13UTC (receipt ebdf2cc):** four of eight timing fits complete;
A-optimized-repeat is active, followed by the remaining three repeats. All eight
GPUs show99% utilization. Exact-state gate remains passed. Cooldown waiter is
live and has not started training; final timing summary and cooldown PPL are
still pending. Read-only receipt SHA256:
`3b376351398d956b476f1fe95b90287424ffa8f1a15543f9258b975d3b16bc7a`.

## Completed timing results and cooldown progress — 01:39 UTC

All fourteen study stages passed, including the exact-state audit and repeated
production-FA4 timing validation. Complete max-rank intervals21–90, two fits per
condition, with outliers retained:

| Arm | Reference seconds/update | Optimized seconds/update | Throughput gain |
|---|---:|---:|---:|
| A | 2.061121 | 2.053581 | 0.367% |
| P6-iso | 2.228933 | 2.221586 | 0.331% |

The improvement is small and does not reduce P6's relative overhead:8.18%
optimized versus8.14% reference. It should not be presented as a substantial
speedup. Within-architecture short validation NLL spread is below0.00005;
these100-step fits are execution checks, not research quality comparisons.
The predetermined performance threshold selects the optimized path for all
three cooldown branches. Production defaults remain unchanged.

Benchmark burns were restored with verified collective progress at01:32:07UTC.
The cooldown supervisor subsequently passed its checks and started01:33:54UTC.
The real eight-GPU two-update smoke passed, with both final scheduler LRs zero
and finite full4882-row validation. P6 cooldown is active:step10055→10062 across
the two status snapshots, all eight GPUs98–99% utilized. Its target is11000;
A-token and A-time follow automatically. No final cooldown PPL is available yet.
Original source checkpoints are preserved; final validation will recheck hashes.
Burn restoration remains configured for success/failure.

Results were pulled and SHA256-verified into `artifacts/p6-workflow-20261011/`.
Read-only receipt77ab71a, local log `temp/remote_logs/p6-workflow-health-a02.log`,
SHA256 `25b6360200911eac788201790594de1f2c0c6d7f439711cfb8c268b6512c013d`.

## Seed-42 replication queued next

User approved fresh A/P6-iso10k fits with seed42 and proxy seed43, followed by
separate1000-step cooldowns of both. Seed42 appeared in the earlier2.5k screen;
this adds a second10k seed alongside1042. Same native training recipe, exact
synchronization option on, compilation off. Retain every250-step checkpoint.
This is an equal-token replication; no additional equal-time A continuation.

The persistent launcher waits for the current study's success and verified burns,
then repeats environment/data/storage/Accelerate checks and safe GPU reclamation.
Original10k checkpoints are copied with hashes for cooldown and checked again
afterward. Final summary: `q359-proxy-10k-seed42-20261011-a01/supervised/run/`
`cooldown/summary.json`. The existing supervisor restores burns on success/failure.
Local15 tests passed; the waiter must still be verified after submission.

**Seed-42 waiter verified02:03UTC:** launch24fab0e, live tmux PID203574,
`WAITING_FOR_COOLDOWN_SUCCESS_AND_BURN`. Node has23TiB free. It will start the
approved replication after predecessor success and verified burn handoff.
Receipt SHA256 `08c3a171b6fd6dc2f360fdd48e71b294da8cb2cc31cbe963c1a2b23310a1e240`.
The runner command is now inactive; persistent current and waiting jobs continue.
