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

GPU exact/timing checks and cooldown runs have not yet completed.

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
