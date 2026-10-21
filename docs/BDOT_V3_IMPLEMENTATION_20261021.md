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
23,120 GiB free, and an immutable v2 source snapshot. GPU verification and
research launch receipts will be recorded separately; these local checks alone
are not a claim that GPU validation or research training has completed.
