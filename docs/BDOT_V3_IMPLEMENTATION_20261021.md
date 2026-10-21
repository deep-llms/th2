# B-DOT v3 implementation and launch

Specification: [v3 pilot plan](BDOT_v3_pilot_plan.md), inheriting v2 v1.4.
The user also approved three exploratory ablations and delegated launch choices.
There are **12 v3 training jobs plus the P1-A′ v2 closure: 13 jobs total**.

| Order | Run | Seed | Definition |
|---|---|---:|---|
| 1 | P1-A-prime | 0 | P1-A with probability-plus-beta selection only; no cap or PF |
| 2 | V2 | 0 | No reuse, radius 1, cap 2, probability selection, both PF maps |
| 3 | C2 | 0 | Random/no-reuse/radius 1/both PF, no curriculum or router fixes |
| 4 | V1 | 0 | V2 without PF |
| 5 | V3 | 0 | V2 with radius 2 |
| 6 | C3 | 0 | C2 with radius 2 |
| 7 | V4 | 0 | V1 with reuse |
| 8 | D | 1 | Matched fixed-path control |
| 9 | V2 | 1 | Same V2, model/routing seeds both 1 |
| 10 | C2 | 1 | Same C2, model/routing seeds both 1 |
| 11 | V5 | 0 | V4 with both PF maps: tests whether placement adaptation helps reuse |
| 12 | V6 | 0 | V2 with query-only PF: isolates adaptation to step-specific keys |
| 13 | V7 | 0 | V2 with MLP-input-only PF: isolates MLP-input adaptation |

V5/V6/V7 were judged before running. They complete useful controlled comparisons,
but may not overcome the capacity cost of reuse or may remove a necessary PF map.
They remain outside the predefined v3 gate candidate pool. No post-hoc promotion.

All jobs use the unchanged complete 4,768-update/96-warmup recipe, sequence 1,024,
microbatch 16 × accumulation 4 × eight B200s, FA4/grouped projections and the
same fixed data/validation order. Data seed remains zero for both model seeds.

## Code

- `bdot/routing.py`: optional `logit_cap`, `select_on`, `placement_film` and
  `placement_film_scope` fields. Defaults retain v2. These are flattened under
  `config.bdot`, following the existing config format.
- `bdot/model.py`, `bdot/grouped.py`: cap the full FP32 router logit before
  masking/softmax/z-loss. Mask `p + beta` again before selection. PF is applied
  after query norm/before RoPE, and after step FiLM/before the MLP. Only allowed
  off-home pairs own parameters; home paths bypass PF exactly. Grouped execution
  uses the same pair maps. Main LR, no decay for all PF parameters.
- `bdot/trainer.py`: every 100 updates logs module loads/beta, distinct-module
  counts, saturation and all placement shares alongside existing reuse/off-default
  metrics. Saturation includes sole-candidate choices; eligible-only saturation
  is also reported. Metrics describe the last complete global optimizer update.
- `train.py`: optional milestone-retention callback after native checkpoint and
  recipe saving. Keep steps 1,000/2,500/4,768 plus the latest complete checkpoint.
  Check all-rank RNG/optimizer/scheduler/recipe files before pruning only older
  checkpoints in the new run. The default retention behavior remains unchanged.
- `scripts/make_bdot_pilot.py --study v3`: explicit sequential train/evaluate
  pairs, isolated output paths and the seed assignments above.
- `scripts/bdot_v3_verify.py`: T1, frozen-v2 CUDA comparison, PF/RF SDPA/FA4
  numerical checks, eight-rank resume tests for V3/V5, full-update timing for
  each option set, and final-checkpoint loss/route carry-over for D/P1-A/N-A/R.
- `scripts/bdot_v3_pipeline.py`: verification precedes every research run;
  failures stop the batch. Reuses the existing GPU supervisor/Accelerate setup.
- `scripts/bdot_v3_report.py`: final tables, comparisons, declared selection/tie
  rules and two-seed gate. Missing conditional controls/seeds are reported as
  pending, never as a gate pass; further conditional jobs are not silently added
  to the agreed 13-job batch.

Each final checkpoint receives full-validation own/default/random interventions
and first-4,096-sequence usage/saturation diagnostics. As-trained loss must
reproduce the training result. P1-A′ is newly trained by this code, not a v2
checkpoint predating the new selection option.

The job wrapper permits one retry for recognized infrastructure failures, using
the latest complete optimizer checkpoint and unchanged scientific settings.
Code/numerical failures stop for review. Node disappearance cannot be recovered
without a live node. No environment, driver, old checkpoint or shared cache is
modified. Automatic communicating burns are restored after success or failure.

## Numerical qualifications

The cap bounds candidate gaps and probabilities when there are multiple choices.
It does **not** prove that parameter gradients never vanish: the derivative of
`tanh` can saturate, and sole-candidate decisions have probability one. Floating
point `tanh` may round exactly to ±1, so the numerical bound is `abs(z) <= c`.
No extra clamp or altered cap formula is introduced.

The plan's linked `BDOT_pilot_diagnostics_response.md` was absent. The existing
[pilot response](BDOT_pilot_results_response.md) specifies the same P1-A′ rule;
the v3 plan supplies the remaining experiment details.

## Local verification

- Full existing/new B-DOT suite: 70 tests passed before the final reporting test;
  targeted v3 suite including gate/queue logic: 11 tests passed.
- CPU frozen-v2 comparison: eight modes, three AdamW updates each, zero gradient
  and parameter differences; routes/loss accounting unchanged.
- V3 CPU fresh-process continuation across the radius-1/radius-2 curriculum:
  exact restored/final weights, optimizer/scheduler state, RNG, data and routes.
- PF counts: 86,016 parameters at radius 1 and 159,744 at radius 2. Identity,
  off-home-only gradients, optimizer groups, grouped/reference parity, activation
  checkpointing and save/reload tested for all three PF scopes.

Local evidence: `temp/v3_implementation_20261021/`. B200 preflight at
**15:33 SGT, 21 October** verified the prior communicating burn handoff, about
23,120 GiB free, and an immutable v2 source snapshot. GPU results follow below.

## B200 throughput checks

All twelve distinct option sets completed 5 warmup and 20 measured HF Trainer
updates on eight B200s, using the actual width, FA4, sequence 1,024 and
microbatch 16 × accumulation 4. These are synthetic complete training updates,
excluding preprocessing, validation and checkpoint saves; they do not predict
convergence or guarantee the same routing distribution after training.

| Configuration | Seconds/update | Native throughput |
|---|---:|---:|
| dense_native | 0.6158 | 100.0% |
| D | 0.6737 | 91.4% |
| V3 | 0.7546 | 81.6% |
| P1-A-prime | 0.7297 | 84.4% |
| V1 | 0.7377 | 83.5% |
| V2 | 0.7453 | 82.6% |
| V4 | 0.7377 | 83.5% |
| C2 | 0.7313 | 84.2% |
| C3 | 0.7642 | 80.6% |
| V5 | 0.7251 | 84.9% |
| V6 | 0.7385 | 83.4% |
| V7 | 0.7452 | 82.6% |

V3 passes the required 60% threshold. Its last measured update used 54.9%
off-home decisions; C2/C3 used 61.1%/76.9%, so the timing checks did exercise
placement conditioning and non-default dispatch. These shares describe the
last update, not the average over the timing interval. Detailed timing and
routing receipts are retained under `supervised/run/checks/T8_*.json`.

## GPU verification and live launch

All acceptance checks passed at **16:18:15 SGT, 21 October 2026**:

- Full-width T1 native-equivalence checks passed.
- Frozen-v2 CUDA comparison: all 32 cases have zero gradient and final-weight
  differences over three AdamW updates, including SDPA/FA4 and both execution paths.
- Twelve new PF/RF model cases and two grouped-projection cases passed the
  existing tolerances. Model logits/losses were identical; maximum relative
  gradient difference was 0.391%. Maximum first-Adam-update RMS difference was
  4.03%; with existing optimizer state, max parameter difference was 5.01e-7.
  These grouped/reference GPU paths are numerically compatible, not bitwise
  identical in their gradients.
- Eight-rank V3 and V5 resume tests: exact restored/final model and optimizer
  state, scheduler, RNG, batches, routes and loss. Each uses fresh processes
  and 60 updates on a reduced-width fixture with a 100-update schedule,
  resuming from both soft and hard phases; production width is covered by T1/T8.
- Final D/P1-A/N-A/R checkpoints reproduce the original full-validation loss
  exactly with both code versions; all 256 saved route samples agree exactly.

Implementation **8304de2**, launch **b6a8cf9**. The launch copied the resource
Accelerate config to the actual Hugging Face cache and verified MULTI_GPU,
eight processes and BF16 using `accelerate env`. The supervisor revalidated
known burn workers 189117–189124, stopped only those workers, waited and recorded
all eight GPUs free. No environment, driver, old checkpoint or shared cache changed.

P1-A′ began at 16:18:15. Health check **5221f19** at **16:23:27 SGT** found
**390/4,768 updates**, finite losses/gradients and eight active GPU workers.
The step-250 checkpoint contains weights, optimizer, scheduler, recipes and all
eight RNG files. Training-source hashes still match the launch recipe. The
latest ten-update mean LM loss was 4.95276; this early training metric is not
a validation/final comparison. Radius 0 is expected before update 477.

The thirteen runs and their evaluations proceed sequentially through the
independent supervisor. Based on previous pilots and these timings, allow
roughly 13–14 hours from launch, approximately **05:30–06:30 SGT, 22 October**,
if no retries are needed. This is an estimate, not a completion claim. Burns
will be restored and collective progress verified after success or failure.

Evidence: [launch receipt](results/bdot_v3_launch_20261021.json),
[downloaded raw receipts](../temp/v3_implementation_20261021/evidence/).
All 71 archive members and the archive hash match the remote manifest. Results
stay under `/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01`;
the automatic final comparison is `supervised/run/v3-results.json` / `.md`.
