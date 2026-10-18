# B-DOT Phase A code review — 18 October 2026

**Current status:** the authorized fixes below resolve findings 1–3 and the four
minor issues. See [resolution and validation](#resolution-after-authorized-fixes).
The original reviews are preserved as historical evidence; their statements
that fixes are still open describe the code before this resolution.

A read-only review of the Phase A implementation against
[BDOT_v2_spec.md](BDOT_v2_spec.md) v1.3. No project code was changed and no
remote job was launched. The CPU suite (`python -m unittest discover -s tests/bdot`
in train_env) passed: 27 tests in 97 s.

Apart from the three findings below, the Phase A code matches the spec.

## Findings

### 1. Curriculum A cancels the default-path prior before routing opens (confirmed, high)

Affects P1-A and N-A.

- **Cause.** During curriculum A's home-only phase (first 10% of training),
  each step allows exactly one candidate. Then `log p[m*] = 0`, so the
  straight-through LM gradient to the router is exactly zero. The only router
  gradient is the z-loss (`bdot/model.py`, `RoutedStep.select`), which pushes
  the home logit toward 0, against the +1 `default_prior`.
- **Why the small coefficient doesn't help.** AdamW normalizes each parameter's
  update to roughly the learning rate however small its gradient is. The router
  weights `w` and the visit matrix `U` therefore move quickly until
  `w·x + U·c ≈ −1`, which cancels the prior. The bias itself barely moves
  (1.000 → 0.982 after 200 steps), so the change is easy to miss.
- **Evidence.** A tiny 12-layer model (d=128, S=8, r=1, curriculum A) was trained
  on CPU with AdamW (lr 3e-4, betas 0.9/0.95) at `progress = 0.05`, on random
  tokens, then evaluated with the target window r=1:

  | Training step | Off-default share at eval | Same run with `z_loss_coef = 0` |
  |---:|---:|---:|
  | 0, 5, 10, 25 | 0.000 | 0.000 |
  | 50 | **0.466** | 0.000 |
  | 100 | **0.529** | 0.000 |
  | 200 | **0.509** | — |

- **Consequences.**
  - At the 10% widening, about half of all tokens leave the default path at
    once, chosen by untrained router weights. P1-A then does not test "start on
    the default path, widen gradually" as the spec intends, and its comparison
    with P1-B is confounded.
  - Periodic validation shows this from about step 50 onward, while training is
    still home-only (see finding 2).
- **Not affected.** Curriculum B always has at least two candidates. R and D
  have no router.
- **Related.** Forced single-candidate steps in the no-reuse runs get the same
  z-loss-only gradient, but they are rarer.
- **Suggested fix (a spec decision).** The code follows the spec literally:
  z-loss covers every router decision. The simplest change is to not count a
  decision with only one allowed candidate as a router decision: no z-loss and
  no router update for it. That also keeps the reach-0 phase exactly the r=0
  model (spec §0, rule 5). Add a regression test that router parameters are
  unchanged after a home-only update.

### 2. Periodic validation ignores the curriculum state (medium)

`RoutingConfig.phase()` (`bdot/routing.py`) returns the target reach and hard
top-1 whenever the model is not training. Every-100-step validation therefore
runs:

- with r=1 during curriculum A's home-only phase;
- with hard top-1 during curriculum B's soft phase.

In both cases it measures a different model from the one being trained.
Gate 0a's spike criterion (more than 0.1 nats above the running minimum of
this validation loss) uses exactly these numbers. Combined with finding 1,
P1-A can show a false early spike or a distorted running minimum.

The spec does not say which window periodic validation should use. Choose and
record one: evaluate with the current curriculum reach, or log both values.
End-of-run evaluation should keep the trained target window, as now.

### 3. The logged training loss includes the router term (low)

`BDOTForCausalLM.forward` adds `z_loss_coef · z_sum / denominator` to
`output.loss`. HF's logged `loss` and the `train_loss` in `train_results.json`
therefore include the router term, while spec §5 says reported loss excludes
router terms. The clean value is logged as `lm_loss_last_update`, and
validation loss is clean. The difference is about 1e-3 nats; it is a
labelling issue, not a training error.

## Checked and found consistent with the spec

- **Windows and quotas:** absolute-window candidate masks and exact no-reuse
  feasibility tables, one per reach used by curriculum A; path counts 34 and 400.
- **Router:** fp32 with autocast disabled; +1 prior on the home module; weight
  std `0.1/sqrt(d)`; `U` initialized to zero.
- **Selection and gating:**
  - selection by `z + beta`, with ties going to the lowest module id;
  - straight-through gate, and the `scale` alternative;
  - soft top-2 weighted by `p` renormalized over the pair, with a
    single-candidate fallback;
  - visit counts, quota state and balance counts use the top-1 choice only.
- **Balancing:** one update per optimizer step, after backward, from
  all-reduced counts; default-path target; `beta` fixed within a step and
  during evaluation; checkpointed.
- **Loss normalization:** z-loss and LM loss use separate global denominators
  across accumulation and ranks, consistent with HF's `num_items_in_batch`
  handling and ×`num_processes` scaling.
- **Attention and module execution:**
  - step-owned norm, K/V and `k_norm`; RoPE applied once to K and per branch to Q;
  - FiLM after the module's MLP norm;
  - sparse dispatch, with dtypes correct under bf16 autocast.
- **Checkpoint replay:** route bookkeeping happens outside activation checkpointing.
- **Decoding:** KV-cache layer indices are consistent across stem, routed
  steps and coda.
- **Random streams:** separate data, model and routing streams. The training
  routing stream is counter-based and checkpointed; evaluation uses its own
  stream and never advances the training one.
- **Optimizer:** no weight decay on norms, biases (including router biases) or
  FiLM vectors.
- **Data:**
  - validation manifest taken from the end of the seed-0 document shuffle;
  - continuous packing that drops only the final tail;
  - segment IDs nondecreasing within each block;
  - training budget is a sequential prefix.
- **Pilot recipe:** 512 × 1024 tokens per update; 1,907 steps for 1B tokens;
  2% warmup; cosine decay to 10%; a 977-sequence (about 1M-token) validation
  prefix every 100 steps.

## Missing, not incorrect

Already listed in [CURRENT_TASK.md](CURRENT_TASK.md):

- `routes_sample.npz` and the 4,096-sequence route diagnostics;
- distinct paths per 10,000 tokens;
- cross-seed route agreement;
- decode and prefill speed.

Not listed there: spec §5 also asks for tokens/s and peak memory in the
every-100-step training log.

## Reproducing finding 1

Run with the train_env interpreter on CPU. Setting `z_loss_coef=0.` in the
`RoutingConfig` call gives the control row.

```python
import os, sys
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
sys.path.insert(0, '/absolute/path/to/DOT')
import torch
from transformers import Qwen3Config
from bdot.model import BDOTForCausalLM
from bdot.routing import RoutingConfig

torch.manual_seed(0)
c = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=384, num_hidden_layers=12,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_dropout=0.,
                tie_word_embeddings=True, rms_norm_eps=1e-6, max_position_embeddings=256)
c._attn_implementation = 'sdpa'; c.use_cache = False
c.bdot = RoutingConfig(n_stem=2, n_coda=2, S=8, r=1, curriculum='A').to_dict()
m = BDOTForCausalLM(c)
nodecay = lambda n: any(k in n for k in ('norm', 'bias', 'film_'))
opt = torch.optim.AdamW([
    dict(params=[p for n, p in m.named_parameters() if not nodecay(n)], weight_decay=.1),
    dict(params=[p for n, p in m.named_parameters() if nodecay(n)], weight_decay=0.)],
    lr=3e-4, betas=(.9, .95))
g = torch.Generator().manual_seed(1)
eval_ids = torch.randint(0, 512, (8, 128), generator=g)
for step in range(1, 101):
    m.train(); m.progress = 0.05  # curriculum A home-only phase
    ids = torch.randint(0, 512, (8, 128), generator=g)
    loss = m(input_ids=ids, labels=ids).loss
    opt.zero_grad(); loss.backward(); opt.step(); m.finish_update()
m.eval(); m.capture_routes = True
with torch.no_grad():
    m(input_ids=eval_ids)
print('eval off-default share:', (m.last_routes != torch.arange(8)[:, None, None]).float().mean().item())
```

## Follow-up verification — 18 October 2026

Finding 1 was independently reproduced on the dev CPU, using the configuration
above, the existing model/optimizer implementation and 100 updates per condition:

| Update | z-loss coefficient 0.001 | z-loss coefficient 0 |
|---:|---:|---:|
| 0 | 0% | 0% |
| 25 | 0% | 0% |
| 50 | 46.6309% off-default | 0% |
| 100 | 52.9419% off-default | 0% |

Raw evidence is in `temp/review_prior_reproduction.json` and
`temp/review_prior_reproduction.log`; the reproducible script is
`temp/check_review_prior.py`. This is an optimization-policy problem that the
previous numerical equivalence/resume checks did not test. Resolve it before
starting the scientific pilot.

The proposed fix needs one additional detail: a zero router gradient tensor is
not the same as no optimizer update. AdamW can still decay router weights (and
apply existing momentum). For the whole home-only phase, bypass router learning
so its parameters receive no gradients/optimizer updates. For forced decisions
in mixed no-reuse paths, omit their z-loss contributions. Specify explicitly
whether the z-loss denominator remains all token-step decisions or changes to
eligible multi-candidate decisions; the latter requires correct global
normalization across accumulation and ranks. Simply masking the numerator does
not implement the latter objective. This changes the stated loss policy and
has **not** been applied as part of the route exporter.

Finding 2 accurately describes current behavior, but target-window, hard-routing
validation is not inherently incorrect: it measures the eventual inference
policy. The ambiguity is using it as a curriculum-stability gate. Recommend
logging both target-policy loss and current-curriculum loss without exploration,
and explicitly selecting which defines a spike. End-of-run route studies remain
target-window, hard-routing evaluations as specified.

Finding 3 is confirmed: HF `loss`/`train_loss` are optimization-objective values;
`lm_loss_last_update` and validation loss are clean LM values. Recommend explicit
objective-loss labeling and a token-weighted LM training-loss aggregate over the
same logging interval. Do not relabel the last-update value as an interval mean.
The model objective and training-log behavior are unchanged in this follow-up.

The route exporter now implements the previously missing path statistics,
`routes_sample.npz`, and sample-aligned cross-seed comparisons. Usage and exact
metric definitions are in [BDOT_IMPLEMENTATION.md](BDOT_IMPLEMENTATION.md#route-study).
Decode/prefill timing and periodic training tokens/s/peak-memory logging remain
separate missing metrics; the exporter does not claim to implement them.

## Re-review — 18 October 2026

A second read-only review, after the route-study and B200 work. No code was
changed. The CPU suite passes: 32 tests in 163 s (`python -m unittest discover
-s tests/bdot` in train_env).

### Findings 1–3 are still open

No file changed between the two reviews in `bdot/model.py`, `bdot/routing.py`
or `bdot/trainer.py`. Their SHA-256 hashes are identical, and
`DOT-th2-inventory/bdot/model.py` is the same file. No project file changed
after 20:33 UTC. The reproduction script above still gives **0.529
off-default** after 100 home-only updates.

The suite passes because no test covers router behaviour during curriculum A's
home-only phase.

Fix checklist for the agent applying the fixes:

1. **Finding 1.**
   - During curriculum A's home-only phase (reach 0), bypass router learning
     entirely, so the router parameters get no gradient and no optimizer update.
     A zero gradient is not enough: AdamW still applies weight decay and
     leftover momentum.
   - Omit z-loss for any decision with a single allowed candidate. This
     includes forced steps in no-reuse paths.
   - Decide and record the z-loss denominator. Either keep all token-step
     decisions (masking the numerator only), or use only decisions with at
     least two candidates. The second needs a new count summed globally across
     accumulation and ranks, plus its own normalization test.
   - Add regression tests:
     - router parameters are unchanged after a home-only update;
     - the reproduction above gives 0% off-default at eval after the
       home-only phase;
     - the z-loss normalization still matches across accumulation and two
       ranks.
2. **Finding 2.** Log periodic validation with both the target window (as now)
   and the current curriculum reach, without exploration. Record which one
   defines a Gate 0a spike. End-of-run evaluation stays target-window, hard top-1.
3. **Finding 3.** Label HF `loss`/`train_loss` as the objective (LM + z-loss).
   Add an LM-only training-loss aggregate weighted by tokens over each logging
   interval. Do not present `lm_loss_last_update` as an interval mean.

### New code reviewed: no implementation errors found

- **Route-study exporter** (`eval/route_study.py`, plus changes to
  `packed_evaluation.py` and `eval/packed_ppl.py`):
  - Each rank's routes map back to the right sequence index. The loader gives
    rank r rows r, r+W, r+2W, …, and the collector uses the same mapping.
  - All-boundary batches are still run, so the diagnostic prefix includes
    every sequence.
  - The fixed and random interventions are checked to score identical targets.
  - Sample comparison refuses mismatched tokens, boundaries, layout or
    evaluation settings.
- **B200 tooling** (`scripts/train_then_burn.py`, `scripts/gpu_process.py`,
  and `scripts/bdot_b200_{preflight,verify,launch}.py`):
  - It signals only GPU workers whose launcher is a hash-verified burn script,
    through PID file descriptors, and never PID 1.
  - Cleanup is limited to descendants of the supervisor itself.
  - A zero exit without the expected receipt counts as a failure.

### Minor problems (low priority)

1. **Two random interventions are not interchangeable.** The route study's
   random pass also runs all-boundary batches, which advances the evaluation
   RNG counter. Its random routes therefore differ from a standalone
   `packed_ppl.py --routing_override random` run with the same seed, batch
   size and world size. Losses are comparable; exact routes are not. Document
   this, or skip the RNG advance for batches that are skipped in standalone mode.
2. **Two off-default shares can disagree slightly.** The study's top-level
   `off_default_share` includes all-boundary batches; ordinary evaluation
   skips them. The effect is negligible but unlabeled.
3. **`train_then_burn` can leave the GPUs without a burn.** If the burn workers
   are killed but `require_free` fails 30 s later, `stop_burns` raises before
   `supervisor.json` exists. `main` then releases the guard without starting a
   replacement burn, and relies on `gpu_guard.sh` to restart one on GPUs in an
   unverified state. Consider keeping the guard held whenever workers were
   signaled and no verified burn was restored.
4. **The runtime gate's denominator is undocumented.** `bdot_b200_verify.py`
   evaluates the 60% runtime gate against D, not native Qwen3:
   `hard_runtime_gate = hard_mixed / D`. That is defensible, since D is
   Stage 0a's dense reference, but spec T8 says only "dense". Record the choice
   with the Gate 0a criteria. Measured with SDPA: 89.2% of D, 78.1% of native.

Still missing, as listed in CURRENT_TASK.md: decode and prefill speed, and
tokens/s and peak memory in the every-100-step training log.

## Resolution after authorized fixes

All three main findings are real. Finding 2 is a policy ambiguity rather than
an intrinsically invalid target-policy evaluation. The user authorized fixing
the confirmed problems, and spec sections 3.7, 5, T8 and Gate 0a now record the
chosen semantics.

1. **Preserve forced-choice routers.** `RoutedStep.select` bypasses a router
   when all choices are forced. Mixed batches mask forced rows out of the
   z-loss numerator. Its denominator remains **all input-token × routing-step
   decisions**, globally summed across ranks and microbatches. The home-only
   optimizer callback also clears router gradients to `None`, so stale zero
   tensors or optimizer momentum cannot update them. Other rows/ranks with
   genuine choices can still train shared router parameters.
2. **Separate validation policies.** Periodic validation logs target hard-routing
   `eval_loss` and current-phase `eval_curriculum_loss`, with no exploration.
   The latter is also `eval_spike_loss`, the Gate 0a stability metric. Its phase
   is computed at completed updates / full schedule, i.e. the next update's
   policy. Identical policies reuse one pass. Final evaluation remains target
   hard routing. Modes, RNG, evaluation counter and saved routes are restored.
3. **Separate LM loss from the objective.** HF's aggregates are labeled
   `objective_loss` and `train_objective_loss`; the latter retains HF's native
   invocation/rank/resume semantics. `lm_loss` is token-weighted over the logging
   interval, and `train_lm_loss` over all updates including resumed ones. Sums
   and logging cursors are checkpointed. Existing last-update diagnostics retain
   their names. Validation PPL excludes router loss.

The minor issues are also resolved:

- Standalone B-DOT scoring now visits all-boundary batches just like the study.
  Random-route RNG advancement and whole-validation routing denominators agree
  for the same seed/layout. `routing_input_tokens` exposes the denominator;
  diagnostic-prefix statistics remain separately scoped.
- The supervisor retains its guard if worker signals may have been sent and
  burn restoration is unverified, even when failure precedes `supervisor.json`.
  Tests mock GPU inspection/signaling; no real processes were stopped.
- T8/Gate 0a explicitly use matched dense **D** as the 60% throughput denominator;
  native Qwen throughput is reported separately.
- Training logs now include interval wall-clock input tokens/s and peak
  allocated/reserved memory (maximum across GPUs, null on CPU). Wall time includes
  intervening validation/checkpoint overhead; T8 isolates training throughput.

### Verification

The same 12-layer, hidden-size-128 reproduction, optimizer, random input stream
and coefficient were run again on CPU:

| Update | Before, z coefficient 0.001 | After, z coefficient 0.001 | After, coefficient 0 |
|---:|---:|---:|---:|
| 0 | 0% | 0% | 0% |
| 25 | 0% | 0% | 0% |
| 50 | 46.6309% | **0%** | 0% |
| 100 | 52.9419% | **0%** | 0% |

Local checks:

- **37 B-DOT tests:** 36 passed in the full run; one old BF16 test expected a
  router invocation for the now-bypassed forced final choice. After updating
  that assertion, its targeted rerun passed. No numerical tolerance was changed
  in that test. New regressions cover existing AdamW momentum, forced-row z-loss,
  interval LM weighting, both validation policies, and exact save/resume across
  home-only → widening with validation enabled/disabled.
- **16 native baseline/evaluation tests** and **49 utility tests** passed.
- **Three two-rank CPU Trainer/DDP probes**, with nonreentrant checkpointing,
  passed: mixed forced no-reuse paths, fully home-only routing and soft top-two.
  They compare gradients and AdamW updates against a full-batch reference;
  maximum absolute gradient differences are 4.84e-8, 5.96e-8 and 5.96e-8.
- Route studies and standalone scoring agree even when a batch has no eligible
  LM targets; native evaluation/packing behavior remains covered by regressions.

Raw local logs, source hashes, reproduction values and both ranks' receipts are
indexed in [the fix receipt](results/bdot_phase_a_fixes_20261018.json).
These are CPU checks; the FA4 adapter tests use a CPU reference kernel. No B200
job or deployment occurred. Revalidate the changed training paths on B200/FA4
before the pilot. Existing strict resume guards must not be bypassed for older
code/checkpoints. Decode/prefill speed measurement remains separate unfinished
work; this fix does not claim to implement it.

## Independent check of the fixes — 18 October 2026

A read-only review of the fix diff, compared against the pre-fix copy in
`DOT-th2-inventory`. **No new problems found.** All three findings and the four
minor items are resolved as described above.

- **Finding 1:**
  - The reproduction script in this document now gives 0.0 off-default at eval.
    It gave 0.529 before the fix.
  - A separate probe confirms router gradients are `None` at reach 0 (reuse on
    and off) and non-zero on every step after widening.
  - One expected exception: in no-reuse runs the last step's router never
    trains, because that choice is always forced by its quota.
  - Router learning is bypassed with `grad=None` in `on_pre_optimizer_step`,
    which HF 5.9 calls after gradient clipping and before `optimizer.step()`.
    AdamW therefore applies no decay or momentum.
  - The z-loss denominator stays at all decisions, matching the amended spec §3.7.
- **Finding 2:** the curriculum policy reuses `phase(progress, training=True)`
  with exploration forced to 0. A separate pass runs only when the policy
  differs from the target. Evaluation mode, RNG, `eval_counter` and
  `last_routes` are restored. Final evaluation and route studies stay
  target-policy.
- **Finding 3:**
  - Interval and cumulative LM loss come from all-reduced, checkpointed sums
    with the correct `metric_totals` indices.
  - The logging cursor stays consistent across ranks, because HF calls `log()`
    on every rank.
  - `train_objective_loss` can be lower than `train_lm_loss` (observed in a
    resume test: 3.179 vs 3.361). This is expected: HF's aggregate is an
    unweighted per-step mean with resume semantics. Use `train_lm_loss` or
    validation loss for comparisons, as §5 now states.
- **Minor items:**
  - Standalone and route-study scoring both visit all-boundary batches.
  - The guard is held whenever workers were signaled and no verified burn was
    restored.
  - T8 and Gate 0a name D as the throughput denominator.
  - The training log reports tokens/s and peak memory.
- **Local runs** (train_env, CPU):
  - 37 B-DOT tests pass (171 s);
  - 49 standard-library tests pass.

The remaining pre-pilot items are unchanged:

- revalidate the changed training paths on B200/FA4;
- measure decode and prefill speed;
- pin the dataset and tokenizer.
