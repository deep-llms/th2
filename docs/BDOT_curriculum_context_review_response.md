# B-DOT: response to the curriculum-horizon and sequence-length review

Date: 2026-10-20. This answers "B-DOT prototype: curriculum horizon and sequence-length comparison" (2026-10-20). Every decision below is written into [BDOT_v2_spec.md](BDOT_v2_spec.md) version 1.4, and Appendix B of the spec lists the changes.

## Summary

The review is right on every factual point:
- the native baseline is valid;
- the routing curriculum follows `max_steps`;
- a 5,000-update cutoff of the 30B schedule would never reach hard routing;
- the faster updates come from smaller updates, not from cheaper tokens.

The decisions:

1. **Complete schedules, one horizon.** Every run used by a gate or a comparison has its own complete schedule. The LR schedule and the routing curriculum share one horizon, `max_steps = floor(tokens / batch_tokens)`. The proposal to keep the 30B LR schedule and add a separate routing horizon is declined for gate and grid runs.
2. **Curriculum transitions come after warm-up.** With complete schedules this holds by construction, and a config check enforces it.
3. **Stages 0a and 0b now train 2.5B tokens (4,768 updates).**
   - This is the Stage 2 budget, so a pilot run whose configuration matches a Stage 2 row becomes that row's seed-0 run.
   - Compute for Stages 0a, 0b and the Stage 2 grid rises by about 7 to 15%, depending on Gate 0b.
4. **One data recipe for every run from Stage 0a on:**
   - the prepared English pool, read in its `packed_shuffle` seed-0 order;
   - fixed subsets of its existing validation split;
   - one exception: Stages 4 and 5 interleave the two-hop text of spec section 10.3.
5. **D is the B-DOT model** at `r = 0` with `routing = fixed` and FiLM on. The native baseline becomes a reference, the subject of a new pipeline check (T14), and the bar for the runtime gate. It is not D.
6. **Validation during the curriculum is now defined,** and the spike rule restarts at each curriculum transition.
7. **The sequence-length analysis is accepted.**

## Q1. A completed screen, or an early look at the 30B curriculum?

A completed screen. Every run used by a gate or a grid comparison has a complete schedule:

- **Horizon:** `max_steps = floor(tokens / batch_tokens)`, which is 4,768 for 2.5B tokens at 524,288 tokens per update.
- **LR:** linear warm-up over `ceil(0.02 · max_steps)` = 96 updates, then cosine decay to 10% of peak at `max_steps`.
- **Routing curriculum:** `progress = completed updates / max_steps`, as `RoutingUpdateCallback` already computes, with the same `max_steps`.
- **Length:** the run trains all `max_steps` updates. An interrupted run is resumed, never shortened. A longer budget is a new run with its own schedule (as the 5B reruns of Stage 2 already are), never an extension of an old one. The one exception is Stage 1's two-hop task, whose LR is constant after warm-up (spec section 10.1).

Why not keep the long LR schedule and add a routing horizon:

- **The gates compare end-of-run losses at 0.01-nat resolution.** At update 5,000 of the 57,220-update schedule, the LR is still 99% of peak (2.97e-4). Every arm would be judged before any substantial LR decay. Decay usually lowers the loss by far more than 0.01 nats, and by different amounts for different models, so differences before decay are a poor guide to differences after it. This matters most for the routed arms, whose router and balancing behave differently at high and low LR.
- **It would test a regime no later stage uses.** With a 5,000-update routing horizon, the router finishes its curriculum and then keeps training at peak LR. Stages 2 to 5 use complete schedules, so a gate passed this way would not transfer.
- **It adds a second horizon** to record, check on resume and never stretch. A complete schedule needs none of that, and `make_bdot_pilot.py` already builds one.
- **Complete runs are cheap.** The native baseline trained 2.62B tokens in about 51 minutes on 8 GPUs.

The first 5,000 updates of a 30B-token routed run answer a different question (how a long run starts), which the plan does not ask. Such a run may be useful for exploration, but label it exploratory: it never feeds a gate.

`train.stop_after` is now defined in the spec (section 4.1). It ends a run after N completed updates without changing `max_steps`, so the LR and curriculum still follow the full schedule. It is allowed only in tests (T12, T14) and exploratory runs; gate and grid runs leave it unset.

## Q2. Should hard routing begin during warm-up?

No. Every curriculum transition comes after warm-up.
- **Automatic here:** with complete schedules this holds by construction. Warm-up ends at 2% of the run, curriculum A first widens at 10%, and curriculum B switches to hard routing at 15%.
- **Enforced:** the config loader now rejects a run whose warm-up does not end before its first curriculum boundary (10% for A, 15% for B), so a later change cannot break it.

Changing the routing regime while the LR is still ramping would confound two changes: if the loss then spiked, we could not tell which change caused it.

Curriculum B's soft phase does start at update 0, during warm-up. That is intended: the soft phase is how the router starts learning.

For the 2.5B-token runs the boundaries are below. They are counted, as in your review, in completed updates: the next update uses the new phase.

| Event | Completed updates |
|---|---:|
| LR warm-up ends | 96 |
| A: the window opens to `r = 1` (10%) | 477 |
| A: exploration reaches 0 (20%) | 954 |
| B: soft top-2 ends, hard top-1 starts (15%) | 716 |
| B: exploration reaches 0 (50%) | 2,384 |

The new test T13 checks these numbers. T12, the resume test, now runs curriculum B with `max_steps = 1,000` and `stop_after = 200`, so that exploration is on after the resume and the routing generator's restore is really tested.

## Q3. Is 5,000 updates / 2.62B tokens enough for the first gate?

Yes. We use slightly less, the Stage 2 budget: **Stages 0a and 0b now train 2.5B tokens (4,768 updates)**, up from 1B in spec 1.3. Change the default of `make_bdot_pilot.py` to match.

- **Why more than 1B:** Gate 0a can stop the project. At 1B tokens, curriculum B gets about 1,600 hard-routing updates; at 2.5B, about 4,050. With the shorter budget, a viable method could still look unused or no better than chance, and a wrong stop costs far more than the extra compute.
- **Why the Stage 2 budget exactly:** a Stage 0a or 0b run whose configuration matches a Stage 2 row (after the Gate 0b choices) becomes that row's seed-0 run, so it is not repeated. Stage 0a's seed-1 reruns likewise become seed-1 runs.

  Which runs carry over depends on Gate 0b:

  | Gate 0b outcome | Carried-over runs |
  |---|---|
  | Any | D → row 1; P3b's free-window `step` run → row 6; P3b's `lol_chunk` run → row 14; the P4 run with the chosen `visit_scale` → row 13 |
  | `step` K/V kept | Also N → row 2 |
  | `step` K/V and the absolute window both kept | Also the P1 winner → row 3 and R → row 7 |
  | Another window or `module_band` chosen | The winning P2 or P3a run becomes row 3, if its configuration matches |

  Before reusing a run, check it:
  - load its final checkpoint with the current code: parameter names and shapes must match;
  - re-evaluate it: its final validation loss must match the logged value within 1e-3, and its routes on the route sample must be identical;
  - confirm in version control that the training code for its features (router, balancing, curriculum, data pipeline) has not changed since it ran.

  If any check fails, rerun it.
- **Cost:** by the spec's planning estimates, Stages 0a, 0b and the Stage 2 grid cost about 610 to 660 H100-hours, instead of 573. The low end assumes Gate 0b keeps `step` K/V and the absolute window (seven runs carry over); the high end assumes only the four runs of the "Any" row carry over.
- **Ambiguous results:** handled as before. A criterion missed by less than its own threshold triggers a second seed at the same budget, never a longer run. Longer training happens only where the plan puts it: the 5B reruns of Stage 2, then Stages 3 to 5.

## Q4. Matched D, data and validation

**The comparison arms of Stage 0a.** All of them use the B-DOT code and share code, data order, validation sets, schedule, seeds, batch layout and evaluation cadence with the routed arms.
- **D:** `mode = compute_matched`, `r = 0`, `routing = fixed`, `step_film = true`, `kv_mode = step`, curriculum `none`. Row 17 (`step_film = false`) is the plain dense model and is not needed for Gate 0a.
- **R:** the P1 winner's window with `routing = random`, curriculum `none` and FiLM on. It is validated and evaluated with random routing drawn from the fixed evaluation generator.

**Data, for every run from Stage 0a on:**
- **Pool:** `nht10/cx_sampled`, subset `qwen3_0.6b_base_en_30B`, at a pinned revision recorded in every `config.json`. Spec 1.4 replaces FineWeb-Edu with this pool. Its 30B tokens cover every stage, Stage 5 included (10B and 20B tokens), so `sample-100BT` is no longer needed.
- **Order:** the prepared packed cache, shuffled at the sequence level with seed 0 (`packed_shuffle`), which is also the native baseline's order.
  - Every run reads it from the start, and a run of N tokens uses the first `floor(N / 524,288)` updates' worth of sequences.
  - Exactly one shuffle may define this order. The Hugging Face Trainer's default sampler reshuffles map-style datasets, so make sure it either is that one shuffle or is turned off.
- **Exceptions to the order:**
  - Stages 4 and 5 interleave the two-hop text: in every update, 10 of the 512 sequences are two-hop sequences, at positions drawn with seed 0 (spec section 10.3).
  - Runs that continue from a trained trunk read on from the first sequence after the trunk's last one: head training and labelling, and Stage 3's fine-tuned pruned baselines.
- **Validation:** the pool's existing validation split, packed the same way and shuffled once at the sequence level with seed 0. The validation manifest stores the sequence indices of both sets:
  - **Periodic set:** the first 1,024 sequences (1,048,576 tokens).
  - **Final set:** the first 10,240 sequences (10,485,760 tokens), or all of them if the split is smaller.
  - The route diagnostics use the final set's first 4,096 sequences, and the route sample its first 256.
- **Check once, before any training:** that no document appears in both splits, by document id or, where ids are missing, by content hash. Drop any validation sequence that contains a training document, and record the check.
- **`make_bdot_pilot.py`:** switch it from `document_stream` and its own manifest to this recipe before Stage 0a. Keep its complete-schedule logic.

**The native baseline cannot be D.** Its LR history differs: 1,145 warm-up updates, then 3,855 updates near peak LR. It also has no FiLM and no B-DOT code path. Spec section 6 now says that the native model is a reference and not run D. Spec 1.3 called it "Dense baseline … can serve directly", which invited the confusion.

**New pipeline check (T14, before Stage 0a, about 5 minutes on 8 GPUs):**
1. Build the B-DOT model in the row 17 configuration (`r = 0`, `routing = fixed`, `step_film = false`).
2. Give it the native model's initial weights with the T1 mapping. Re-creating the native model with its training seed should reproduce them. If it does not, initialise a fresh native model, copy its weights into the B-DOT model, and train both.
3. Train 500 updates with exactly the native run's recipe: same data order, batch layout and LR schedule. To reuse the completed native run, this means its 57,220-update schedule (1,145 warm-up updates) with `stop_after = 500`; the spec allows `stop_after` in this test.

**Pass condition:**
- **Losses:** the mean training loss over updates 401–500, and the periodic validation loss of the update-500 checkpoints, agree within 0.01 nats.
- **If they do not agree:** train the native model a second time with the same recipe. If the two native runs differ by a similar amount, the gap is numerical noise. Otherwise the two pipelines differ in data, masks, positions, LR or initialisation.
- **Speed:** record both runs' tokens per second; the difference is the cost of the B-DOT machinery at `r = 0`.

## Validation during the curriculum

Spec 1.3 did not say which routing periodic validation uses, and your two validation modes show why it matters. Spec 1.4 (section 5) defines it:

- **Periodic validation (every 100 updates):** the routing the model is currently trained with, without exploration.
  - **Learned routing:** soft top-2 during B's soft phase, otherwise hard argmax over the current window.
  - **Control runs (`fixed`, `hash`, `random`):** their own routing; `random` draws from the fixed evaluation generator.
  - **STOP and re-entry trunks trained with random policies:** policy `none`.
- **During B's soft phase:** also log the hard-policy validation loss. Its gap to the soft loss shows how far soft training is from the deployed policy.
- **At each curriculum transition:** validate the weights at the transition update twice, once with the old phase's routing and once with the new one's, and log the jump (new minus old). Measuring both on the same weights isolates the effect of the switch itself.
- **Spike rule (Stage 0a):** at each transition, the running minimum restarts from the new phase's loss at the transition update. A transition jump is reported, not counted as instability. If the jump does not recover, the final loss shows it anyway.
- **End-of-run evaluation:** without exploration: hard argmax for learned routing, and the controls' own routing for control runs.

## Sequence length and throughput

Section 4 of the review is correct:
- **Equal sequences, different tokens:** with the same micro-batch and accumulation, halving the sequence length halves the tokens per update.
- **Token-wise layers** therefore do half the arithmetic per update.
- **Attention under document isolation** scales with the fragment lengths, not with one square per packed block.
- **At equal tokens per update,** the arithmetic is about the same.

Your logged runtimes add one point. At equal tokens (2.62B), the current run took 50 min 40 s and the previous 2,048-token run about 45.3 min, so tokens per second fell by about 11%. The two runs differ in several ways, so this is not a controlled result, but the direction is expected:
- **Per-update costs:** with half as many tokens per update, the optimizer step, the gradient all-reduce, logging and the checkpoints every 250 updates are paid twice as often per token.
- **Smaller matrix multiplies:** each micro-batch's matrix multiplies are half as tall.

Two suggestions, neither of which changes any result:
- **Micro-batch 32 with accumulation 2,** still 512 sequences per update, if memory allows. The math is the same, and T11 covers accumulation.
- **Headroom:** 862k tokens/s on 8 GPUs, at about 2.2 GFLOP per training token, is about 240 TFLOP/s per GPU. If these are B200s, that is roughly a tenth of their dense bf16 peak. This does not affect the pilot, but Stages 2 to 5 would cost less with a faster trainer.

**T8 and Gate 0a criterion 6.** T8 now benchmarks three models: the native dense reference, D, and the routed model. All three use the same micro-batch layout, compile settings and logging, and only training updates are timed, without validation or checkpoint saves. Gate 0a's runtime criterion compares the routed model with the native dense reference, because that is the model B-DOT must compete with in practice. D's throughput is reported beside it.

## Assessment

Agreed: 12 layers and a 1,024-token context test the core bounded-routing hypothesis, its stability and its overhead. They do not show long-context gains, full-depth rankings or converged quality. Stage 5 returns to full depth.

## Spec changes (version 1.4)

- **Section 4.1:**
  - data pool and the `packed_shuffle` order (one shuffle only);
  - the shuffled validation split, its fixed sets and the disjointness check;
  - the schedule horizon (including epoch-based runs) and `train.stop_after`;
  - warm-up rounding and budgets;
  - the order for the two-hop mix and for continuation runs.
- **Section 4.2:** one horizon, transitions after warm-up (checked in whole steps), and the boundaries table for 2.5B tokens.
- **Section 5:** routing during validation for learned and control runs, the hard-policy diagnostic, transition jumps on the same weights, and named validation sequences for the diagnostics.
- **Section 6:** the native model is a reference, not run D; Phase A now passes T13 and T14.
- **Section 7:**
  - the test list before Stage 0b matches section 6;
  - T8 benchmarks three models;
  - T12 uses curriculum B with `max_steps = 1,000`;
  - new T13 (schedule horizon) and T14 (pipeline check).
- **Section 8:** the `data` keys (`dataset`, `subset`, `order`, `validation_manifest`) and `train.stop_after`.
- **Section 9:**
  - Stages 0a and 0b at 2.5B tokens;
  - carry-over of seed-0 and seed-1 runs into Stage 2, with its check;
  - D defined in full;
  - the spike rule at transitions;
  - the runtime bar;
  - Stage 2 notes (rows 1 and 17 use `step` K/V, the rows marked W, row 3 vs row 18 under `module_band`);
  - costs, with the measured throughput.
- **Section 10:** a fixed end for the two-hop extension and the curriculum during it; the two-hop mix as whole sequences interleaved at 2% of every update.
- **Section 11:** a longer schedule stopped early is never a reference or a comparison arm.
- **Section 12:** pitfall 17 (early cutoffs).
- **Appendices A and B.**

## Please report back before Stage 0a

- The pool revision hash, the validation split's size, and the result of the document-disjointness check.
- The GPU type behind the measured throughput.
- T13 and T14: for T14, both losses and both runs' tokens per second.
- Confirmation that `make_bdot_pilot.py` uses the packed cache, the existing validation split and a 2.5B-token default, and that nothing reshuffles the order.
