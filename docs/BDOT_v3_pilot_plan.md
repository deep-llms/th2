# B-DOT v3 pilot: plan (9 runs)

Date: 2026-10-21. This plan is fixed before any v3 run. It is a new study with its own gate. It does not change the outcome of v2: Gate 0a failed, and that result is written up whatever v3 shows.

Everything not stated here follows [BDOT_v2_spec.md](BDOT_v2_spec.md) version 1.4 and the Stage 0a recipe:
- 2.5B tokens (4,768 updates), the same data order and validation sets;
- `compute_matched`, absolute windows, `kv_mode = step`, step FiLM, curriculum A, seed 0.

## 0. First, close v2

Run P1-A′ first, as already declared in [BDOT_pilot_diagnostics_response.md](BDOT_pilot_diagnostics_response.md). The v3 batch follows it without waiting for its result (section 11).

If P1-A′ meets all six Gate 0a criteria at seed 0, or misses one by less than that criterion's threshold, its two-seed confirmation is scheduled after the batch.

## 1. What v3 tests

The v2 diagnostics located the losses:
- **Reuse:** about 0.09 nats (P1-A vs N-A).
- **Modules running away from their home step:** about 0.036 nats (N-A's default path vs D).
- **Value of the learned order:** only 0.005 nats (N-A's own routes vs its default path).
- **Router behaviour:** routers saturated, and with reuse on they collapsed onto a few modules.

**The question:** keep the router unsaturated and give each module cheap conditioning for every step it visits. Can routing without reuse then match D, with routes worth at least 0.01 nats? And do the same fixes make reuse workable?

## 2. New options

Each option's default reproduces v2.

### 2.1 Router fixes (RF)

- **Logit cap:** `z ← c · tanh(z / c)` with `c = 2`.
  - It is applied to the full logit (`w · x + b + U · c_t`) before the softmax, the selection and the z-loss.
  - Logit gaps stay below 4. Among three candidates the top probability then stays below about 0.97, so the straight-through gradient (proportional to `1 − p[m*]` and to the other `p`) never vanishes.
  - Selection is still argmax, so the cap does not make routing less decisive.
- **Selection on probabilities:** `m* = argmax over allowed m of (p[m] + beta[m])`, where `p` is the softmax of the capped logits over the allowed candidates. This is P1-A′'s rule. The balance rate, counts and timing are unchanged.
  - With reuse off, the quotas fix every module's load at 1/8, so `beta` never moves; there, only the cap matters.
- **Config:**
  - `router.logit_cap: 2.0` (`null` = off);
  - `router.select_on: probs` (`logits` = v2).

### 2.2 Placement FiLM (PF)

When a token runs module `m` at a step `s` other than its home (`m ≠ s` in `compute_matched`), apply two zero-initialised affine maps specific to the pair `(m, s)`:
- **query,** after `q_norm` and before RoPE: `q ← q ⊙ (1 + γq[m,s]) + βq[m,s]`, with `γq` and `βq` of size `H · head_dim = 2,048`;
- **MLP input,** after the step FiLM: `u ← u ⊙ (1 + γu[m,s]) + βu[m,s]`, with `γu` and `βu` of size `d = 1,024`.

**Details:**
- **Home step unchanged:** at a module's home step nothing changes. The default path, D and every `r = 0` model are therefore untouched, and D needs no rerun.
- **Size:** one set per allowed off-home pair: 14 pairs at `r = 1` (86,016 parameters) and 26 at `r = 2` (159,744). That is under 0.05% of the model.
- **Training:** no weight decay and the main LR, like the existing FiLM vectors.
- **Rationale:** v2's step FiLM is shared by every module at a step. PF lets each module adapt, for each step it visits, its query (which must match that step's keys) and its MLP input. The Library of Layers paper applies a depth-dependent scale and bias after every norm for the same reason.
- **Config:** `model.placement_film: true`.

### 2.3 Wider window without reuse

Use `r = 2` with `reuse = false`. This allows 400 paths instead of the 34 that `r = 1` allows (only swaps of adjacent modules).

Under curriculum A:
- the window opens to `r = 1` after update 477 and to `r = 2` after update 954;
- exploration restarts at each widening and reaches 0 at updates 954 and 1,431.

## 3. Runs

### Stage a (6 runs)

All use the P1-A recipe unless the table says otherwise.

| Run | `reuse` | `r` | RF | PF | `routing` | Role |
|---|---|---|---|---|---|---|
| V1 | off | 1 | ✓ | – | learned | Step on the ladder: the cap without reuse (vs N-A) |
| V2 | off | 1 | ✓ | ✓ | learned | Candidate |
| V3 | off | 2 | ✓ | ✓ | learned | Candidate with more freedom |
| V4 | on | 1 | ✓ | – | learned | Candidate: do the fixes make reuse work? |
| C2 | off | 1 | – | ✓ | random, curriculum `none` | Chance control for V2 |
| C3 | off | 2 | – | ✓ | random, curriculum `none` | Chance control for V3 |

V4's chance control is v2's R (reuse on, `r = 1`, no PF), which is reused. RF does not apply to random routing.

### Stage b (3 runs, committed in advance)

So that every run can go in one unattended batch, the second seeds are chosen now rather than after stage a:
- V2 at seed 1;
- C2 at seed 1;
- D at seed 1.

V2 gets the second seed in advance because the tie rule of section 6.1 favours it. "Seed 1" means `seeds.model = 1` and `seeds.routing = 1`; the data order stays fixed.

If the candidate selected in section 6.1 is V3 or V4 instead, see section 6.2.

### Runs reused from v2

D, N-A, P1-A, R, and P1-A′.

Before using any of them, apply the carry-over check of the v1.4 spec (Stage 2):
- load the final checkpoint with the v3 code;
- re-evaluate it: the validation loss must match within 1e-3 and the route sample must be identical.

## 4. Before launching

- **T1** still passes with the new code.
- **T4** enumerates exactly 400 no-reuse paths at `r = 2`.
- **Window schedule:** a T13-style check confirms the `r = 2` boundaries of section 2.3.
- **Unit checks:**
  - every capped logit satisfies `|z| < c`;
  - selection uses `p + beta`;
  - PF is the identity at initialisation and has no effect at home placements;
  - PF parameters receive gradients only from off-home placements;
  - PF parameters are saved and restored.
- **T8** for V3, which has 5 candidates per step.

## 5. Logging (every routed run)

**Every 100 updates:**
- each module's load and `beta`;
- the mean number of distinct modules per token, and the reuse rate;
- the share of decisions with `p[m*] > 0.99`;
- the off-default share;
- the share of placements per `(module, step)` pair.

**At the end:** the Step 1 interventions (own routes, default path, random routes) and the usage maps.

## 6. Decision rules

### 6.1 Selection after stage a

- **Pool:** V2, V3 and V4. Keep those with no spike and an off-default share of at least 5%.
- **Pick:** the one with the lowest final validation loss.
- **Ties:** differences under 0.01 nats are ties. Ties go to the simpler run: V2, then V3, then V4.
- **Contingency:** if V1 beats V2 by at least 0.01, PF is dropped and V1 replaces V2 as the `r = 1` candidate. Its chance control (C2 without PF) is then run at seed 0 after the batch. V1 is then treated like V3 or V4 in section 6.2: it gets second seeds only if it meets the seed-0 condition there.

### 6.2 Second seeds

- **V2 selected:** its second seed and C2's are already in the batch.
- **V3 or V4 selected:** if it meets all six Gate 0a criteria at seed 0, or misses one by less than that criterion's threshold, run it and its chance control at seed 1 (2 more runs). Otherwise it cannot pass, because a candidate without two seeds is never judged.

### 6.3 Gate v3

The gate is judged on the mean of two seeds for the candidate, its chance control and D. All six Gate 0a criteria must hold:
1. no spike;
2. off-default share of at least 5%;
3. its own routes at least 0.01 nats better than its default path;
4. at least 0.01 nats below its chance control;
5. at most 0.01 nats above D;
6. at least 60% of the native reference's training throughput.

### 6.4 After the gate

- **Pass:** the spec is updated, with the passing configuration as the main routed setting, and the next stage is planned then.
- **Fail:** the B-DOT line stops, and v2 and v3 are written up together. Nothing further is tried: no other curricula, LR multipliers, caps, windows, sizes or budgets.

## 7. What each comparison shows

All of these are single-seed comparisons, so differences under about 0.01 nats are noise.

| Comparison | Shows |
|---|---|
| V1 vs N-A | Effect of the cap without reuse |
| V2 vs V1 | Effect of placement FiLM |
| V3 vs V2 | Effect of the wider window |
| V4 vs V1 | Cost of reuse once the router is fixed |
| V4 vs P1-A and P1-A′ | Whether the cap prevents collapse |
| V2 vs C2, V3 vs C3, V4 vs R | Learned routing vs chance |
| Own routes vs default path, for each run | Value of the learned routes |

## 8. Predictions (written before running)

- **V1 vs N-A:** within 0.01 of each other. Without reuse there is no collapse to prevent.
- **V2 vs V1:** V2 better by 0.005 to 0.02. PF should remove part of the 0.036 cost of running modules away from home.
- **V3 vs V2:** uncertain. More freedom could add routing value, or more cost.
- **V4:** better than P1-A but worse than V1. The cap should prevent collapse, but repeats still cost capacity.
- **Most likely overall:** no candidate passes, and the value of the learned routes stays below 0.01 nats.

## 9. Order and cost

Run everything as one batch, in this order, so that the most important runs finish first if the night is cut short:

1. P1-A′ (closing v2);
2. V2, then C2;
3. V1;
4. V3, then C3;
5. V4;
6. D, V2 and C2 at seed 1.

That is 10 runs of about an hour each, so about 10 hours of the 8-GPU node. The contingency of section 6.1, or a V3 or V4 selection under section 6.2, adds 1 or 2 runs after the batch.

## 10. What to send back

- **Per-run table:**
  - final validation loss and gap to D;
  - off-default share;
  - mean distinct modules per token;
  - smallest and largest module load;
  - share of saturated decisions.
- **Intervention table:** own routes, default path and random routes for every routed run.
- **Comparisons:** the section 7 comparisons.
- **Decisions:** the candidate selected under section 6.1, any extra seeds needed under section 6.2, and the gate decision.
- **Run status:** for every run, whether it completed, and any failure or resume.

## 11. Running the batch unattended

**Before launching:**
- Every section 4 check must pass.
- If a check fails and cannot be fixed, launch only the runs that do not need the failing feature, and report. For example, if PF cannot be made to pass, run P1-A′, V1, V4 and D at seed 1.
- Check free disk space. Keep only the checkpoints at updates 1,000, 2,500 and 4,768, plus the latest one for resuming, and delete the others once they are superseded. Twenty checkpoints per run for 10 runs may not fit.

**During the batch:**
- **Infrastructure failure** (crash, out-of-memory, preemption): resume the run from its latest checkpoint. T12 showed that resuming is exact.
- **Never change** a run's configuration, or the code of a feature that is already in use, while the batch runs.
- **A bug in the new code that affects results:** stop the batch and report. Do not fix it and relaunch without a review.

**At the end:** write one report following section 10, and apply sections 6.1 and 6.2.
