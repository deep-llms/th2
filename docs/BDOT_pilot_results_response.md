# B-DOT: response to the Stage 0a pilot results

Date: 2026-10-21. This answers "B-DOT pilot results — 20 October 2026". The spec ([BDOT_v2_spec.md](BDOT_v2_spec.md), version 1.4) is unchanged for now; section 5 says when it would change.

## 1. Verdict

**Gate 0a fails.** Criterion 5 requires the P1 winner's validation loss to be at most 0.01 nats above D's.
- P1-A is 0.121 nats above D, and P1-B is 0.138 nats above, twelve to fourteen times the threshold.
- A miss this large is not noise, so the second-seed rule for near misses does not apply.

**What follows from the plan:**
- Phase B is not built, and Stage 0b does not run.
- Nothing is tried as a rescue: no STOP, re-entry, larger model, longer run, other window or other K/V mode.
- The P1 winner is P1-A. Its loss is 0.017 nats below P1-B's, more than the 0.01 tie margin, and both reach the 5% off-default share. Please confirm that neither run had a spike (criterion 1).

**What the pilot does establish:**
- Routing is used: 45% and 38% of choices leave the default path.
- The overhead is small: the routed runs took 10–14% longer than D in wall-clock, including evaluation and saving. Criterion 6 (at least 60% of the native reference's throughput) should pass easily; please confirm with T8.
- T12 and T14 passed, so the training pipeline is unlikely to explain a 0.12-nat gap. A routing-specific bug is not yet excluded.

## 2. Why diagnose before writing up

A negative result is only useful if we know what failed. Three explanations fit the numbers so far, and they lead to different conclusions:

1. **A defect in the routing machinery,** such as a bug or a balancing mechanism that cannot act. This is fixable, and the idea itself would still be untested.
2. **Substitution.** With reuse on, a token can repeat strong modules and skip others, so it runs fewer distinct parameters than in D. Your observation that P1-A "concentrates work on a few modules" points here. Reuse-off routing (run N) cannot do this: every token runs each module exactly once.
3. **A structural cost.** A module shared by neighbouring steps may not specialise as well as a layer that always sits at one depth, whatever the router does. Then every routed variant loses to D, N included.

The spec already planned R, N and the route interventions as part of Stage 0a; they are what separates these explanations.

**One suspected defect needs checking first.** The spec adds the balance bias `beta` to the raw router logits `z` and takes its rate, 1e-3 per update, from loss-free balancing (Wang et al., 2024). That paper adds the bias to sigmoid gating scores, which lie between 0 and 1, so 1e-3 per update is calibrated to that scale.
- **Raw logits are unbounded.** P1-A routes for 4,291 updates (after its 477 home-only updates), so `beta` can move by at most about 4.3 per module, and only if it is pushed the same way on every update.
- **Consequence:** once the logit gaps between candidates are larger than that, balancing cannot change the choices, and the router can concentrate on a few modules unopposed.

## 3. What to run

### Step 1: analyses of the existing checkpoints (no training)

Use the final checkpoints of P1-A and P1-B and the final validation set. The route statistics use its first 4,096 sequences.

1. **Route interventions** (spec section 5): the validation loss with
   - the model's own routes;
   - `routing = fixed` (the default path);
   - `routing = random` in the same windows, with the fixed evaluation generator.
2. **Usage map:**
   - for each step `s` and module `m`, the share of tokens choosing `m` at `s`;
   - each module's total load against its target of 1/8;
   - the distribution of the number of distinct modules per token, and the reuse rate;
   - the off-default share per step.
3. **Balancing health,** from the checkpoints at updates 1,000, 2,500 and 4,768, and the logged loads:
   - `beta` and `load` per module;
   - a sign check: over the logged updates, `beta[m]` should fall while `load[m]` is above its target and rise while it is below;
   - at the final checkpoint, the median and 90th-percentile gap between the top-1 and runner-up logits among the allowed candidates, next to `beta`'s range (maximum minus minimum).
4. **Router scale:** the z-loss and mean `logsumexp(z)` over training.
5. **Correctness spot checks:**
   - every chosen module is inside its window;
   - the gate's forward value is exactly 1;
   - **router-gradient check:** on one batch, with the routes held fixed, compare the autograd gradient of the loss with respect to `z` against a finite difference. In the finite difference, the gate is `p(z + ε) / p(z)` with the denominator held constant: the same surrogate, written without `stopgrad`. The two must agree in sign and size. T3 only checked that this gradient is non-zero; a flipped sign would steer the router toward harmful routes.

### Step 2: complete Stage 0a's controls

Run these exactly as spec section 9 defines them, with P1-A's settings, 2.5B tokens and seed 0:
- **R:** `routing = random`, curriculum `none`;
- **N:** `reuse = false`, curriculum A.

Step 2 does not depend on Step 1, so the two can run in parallel.

## 4. How to read the results

These patterns support the write-up; they do not decide anything by themselves.

- **Default path vs own routes:**
  - if the trained P1-A does better on the default path than on its own routes (by at least 0.01 nats), the router is choosing harmful routes, which points to its learning signal;
  - if the default path is much worse, the modules have adapted to their routes.
- **P1-A vs R:** if R is within 0.01 nats of P1-A, learned routing is no better than chance (criterion 4 fails).
- **N vs D, and P1-A vs N:**
  - N close to D and P1-A far above N points to substitution;
  - N far above D points to the structural cost.

## 5. Decision rules

These rules are fixed now, before any of the new results. Apply them in order.

1. **A bug is found.** That means `beta` moves the wrong way, a route falls outside its window, the gate's forward value is not 1, or the router-gradient check fails. Fix it, rerun the affected acceptance tests, rerun P1-A and P1-B, and judge Gate 0a again from the start. If it fails again, apply rules 2 to 5 to the new runs. This is a correction, not a rescue.
2. **Balancing is too weak.** There is no bug, but at the end some module's load is below half or above twice its target, and the median logit gap is larger than `beta`'s range. Then run one corrected arm, **P1-A′**:
   - P1-A with selection by `argmax(p + beta)`, where `p = softmax(z)` over the allowed candidates;
   - everything else unchanged, including `balance_rate = 1e-3`.

   This puts `beta` on bounded scores, as in the method it comes from.
3. **N is close to D** (within 0.01 nats). Then also run **R_N**, random routing with `reuse = false`, as N's chance control.
4. **Pass.** Gate 0a passes only if a candidate (P1-A′ or N) meets all six criteria against its own chance control (R for P1-A′, R_N for N), judged on the mean of two seeds.
   - D and the chance control get a second seed as well.
   - Two seeds are required here whatever the margin, because these candidates are chosen after seeing the first results.
   - The passing configuration then becomes the main routed setting, and the spec is updated before Stage 0b.
5. **Otherwise, stop.** Write up the controlled negative result:
   - the runs D, P1-A, P1-B, R, N, and P1-A′ if it ran;
   - their route interventions and usage maps.

   Nothing else is tried: no other curricula, priors, balance rates, windows, K/V modes, model sizes or token budgets.

## 6. What to send back

- Confirmation that no run spiked, and T8 throughput for the native reference, D and P1-A.
- From Step 1:
  - the intervention table and the usage maps;
  - the balancing-health table with the sign check;
  - the logit-gap numbers and the spot checks.
- From Step 2: the final validation loss of R and N, with N's off-default share and usage map.
- The decision under section 5, before starting any further run.

## 7. Cost

- **Step 1:** evaluation only, well under an hour.
- **Step 2:** two runs of about an hour each on the 8-GPU node.
- **Conditional runs:** at most seven more. These are P1-A′ and R_N, plus second seeds for D, the candidates and their chance controls.
