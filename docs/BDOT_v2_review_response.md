# B-DOT v2: response to the pre-implementation review

Date: 2026-10-18. This responds to "B-DOT v2: assessment and questions before implementation", a review of spec version 1.1. Spec version 1.3 later switched every layer to the Qwen3 architecture, trained from scratch; the decisions below still apply. Spec version 1.4 later moved Stages 0a and 0b to 2.5B tokens, changed the data pool, and added tests T13 and T14 before Stage 0a; see [BDOT_curriculum_context_review_response.md](BDOT_curriculum_context_review_response.md). Every decision below is now written into [BDOT_v2_spec.md](BDOT_v2_spec.md) version 1.2, and section numbers refer to that version. Appendix B of the spec lists all the changes.

## Summary

The review's assessment is accepted: run a small controlled pilot first, and expand only if it shows useful routing at acceptable runtime. Three decisions follow from it.

1. **Scope.** Accepted. The build is phased (§6).
   - Phase A contains only what the first pilot needs.
   - Everything else is built only after the pilot gate passes.
   - The comparisons are kept for later; none is dropped.
2. **What justifies expansion.** The first pilot tests viability, not a win (§9, Gate 0a). Gate 0a passes only if all six hold:
   1. no loss spike;
   2. routing is used (off-default share ≥ 5%);
   3. the learned deviations matter: forcing the default path on the trained model raises its loss by ≥ 0.01 nats;
   4. learned routing beats same-window random routing by ≥ 0.01 nats;
   5. learned routing is within 0.01 nats of the dense reference;
   6. routed throughput is ≥ 60% of dense.

   How results are read:
   - A result within noise triggers a second seed, never a scale-up.
   - If routing is unused or no better than chance, the project stops.
   - STOP, re-entry and larger models are never rescue runs.
3. **Data contract.** Attention is blocked by document and positions restart per document (§4.1). This matches your existing helper and the LoL paper, and it replaces the v1.1 choice of cross-document attention.

All other recommendations are accepted as proposed, with the details below.

## Prior work

- **Mixture of Attention Heads.** Agreed that shared K/V with per-expert Q/O is prior art. §1 now says so. It names B-DOT's distinct element as bounded, per-token, cross-step selection of a whole Q/O-plus-MLP module with step-owned K/V, and says the contribution must be shown as a quality–cost improvement.
- **ReXMoE.** Agreed. Grid row 18 (`route_scope = mlp`, MLP-only routing) is added, so row 3 vs row 18 measures what routing the attention projections adds (§3.5, §9 Stage 2).
- **DNA.** Agreed. §1 now reports the result per variant: top-1 2.754 and top-2 2.674, against a dense 2.720, with top-2 using more active parameters.
- **Library of Layers.** We hold the paper's full text, and §3.5.1 was checked against it: Algorithm 1, sections 3.2–3.3 and the appendix.
  - **Source:** §3.5.1 now cites these sections and quotes the instability sentence ("…this mismatch makes training unstable and harms performance"). The PDF is at https://openreview.net/pdf?id=O4y4PcNmYD.
  - **Deliberate differences:** `lol_chunk` uses our router, gate, FiLM and AdamW, where the paper uses router-probability scaling, FiLM after every norm and Muon.

## Answers to the questions

**Q1. Objective of the first pilot.**
- **The pilot (§9, Stage 0a):** a dense reference D, curricula P1-A and P1-B, then a same-window random control R and a no-reuse run N for the P1 winner.
- **Gate 0a:** the six criteria in the summary.
- **Default-path test (new, §5):** this is the route intervention Gate 0a uses. The random-route test stays a diagnostic, because a model can be highly sensitive to random routes even when its learned deviations are worthless.
- **A clear win over dense** at 1B tokens counts as a strong signal but is not required. A win is required later, at Gate 2 (2.5–5B tokens, 2–3 seeds).

**Q2. Order, substitution, or both.**
- **Separate claims (Stage 2 notes):** order and substitution are reported as separate claims. A reuse-on win shows that per-token module selection helps. Order is credited only if the matching reuse-off row also wins.
- **What moves:** §3.5 now states it explicitly: with `kv_mode = step`, a module is `W_Q`, `W_O`, the MLP norm and the MLP. The attention-input norm and K/V stay with the step. It is not an intact decoder block.

**Q3. Soft top-2 semantics (§3.6, §3.7).**
- The pair is chosen by `z + beta`, as for top-1, and weighted by the unbiased `p`, renormalised over the pair.
- Both deltas are computed from the same `h` and the same visit vector.
- Only the top-1 choice uses quota, and soft training is labelled a relaxation, not an exact permutation.
- Ties go to the lowest module id, including the two home experts in capacity mode.
- The straight-through gradient is described as a surrogate that cannot evaluate counterfactual modules, so exploration and the soft phase are real optimisation choices.

**Q4. Balancing and distributed training (§3.7).**
- **Counts:** per-step counts and their denominators are summed over micro-batches and ranks before shares are computed.
- **Empty steps:** a step with no active token is skipped.
- **Target:** `beta` moves toward a default-path target, not a uniform mean. The target is 1 for each step's home module, and 0.5 for each of the two home experts in capacity mode. The default path therefore never moves `beta`, with or without STOP and whether or not a later step is empty.
- **Fixed routing:** with `routing = fixed`, the router and balancing are not used at all, so a capacity model that always uses expert `2s` does not go through balancing.
- **Timing:** `beta` is updated once per training step, after the backward pass and from that step's counts, so routing only ever sees past load. It is fixed during the step and during evaluation, and it is checkpointed.
- **z-loss:** it is averaged over every router decision of an active token in both passes, normalised globally. Reported loss and perplexity exclude router terms.

**Q5. Matched compute (§3.11).**
- **Three measures, reported separately:** nominal hard-path LE, executed training FLOPs and wall-clock time.
- **Equal tokens:** grid comparisons are at equal tokens, not at equal total training compute.
- **Soft-phase cost:** your estimate is right. It costs about `2 - kv_frac` times the hard middle FLOPs, roughly +8% of block weight FLOPs over a run with a 15% soft phase.
- **Runtime:** §3.6 now notes that router latency can dominate despite tiny FLOPs, and that stopped tokens still occupy attention rows in training.

**Q6. Data and baseline contracts (§4.1, §9, T1).**
- **Data:**
  - **Document blocking:** attention is blocked by document, with per-document RoPE positions, and `lol_chunk` chunks are computed per document.
  - **Dataset revision:** pinned and recorded in every config.
  - **Validation:** a fixed manifest of document ids, chosen before any training. It is the same for every budget and stage, removed from `sample-100BT` in Stage 5, and never split with training.
  - **Budget rounding:** budgets round down to whole training steps, and the rounding is recorded.
- **FiLM:** row 1 now uses the same FiLM option as the routed rows, as do rows 9 and 11. Row 17 adds a plain dense transformer.
- **T1** now also compares all gradients and the parameters after one AdamW update.

**Q7. `module_band` cache (§3.6).**
- **Eager policy:** at each step, every token's state is projected with every module in `P(s)`, the set of modules any token could choose at that step. `P(s)` is defined for every window type, and the memory cost is stated.
- **Recompute ablation:** it is training-only and has no cached decoding.

**Q8. STOP and re-entry labels (§4.3, §4.4).**
- Both are now described as heuristics, and the claim that the STOP approximation errs in a safe direction is removed.
- Re-entry labels are noted to measure a joint intervention.
- Both heads are judged by the measured quality and speed of the deployed mixed policy.
- The alignment is stated: the state at position `t` goes with the loss of predicting token `t + 1`, and losses are supervision only.

**Q9. Development versus final testing (§10, Gate 1).**
- **Two-hop task:** it now has ID-dev/ID-test and OOD-dev/OOD-test splits. The LR choice and the extension decision use dev only, and Gate 1a uses OOD-test.
- **Order task:**
  - Exact accuracy is reported on the 64 held-out cases.
  - Task seeds (new permutations and held-out order) are separate from initialisation seeds and paired across configurations.
  - Gate 1b now uses held-out accuracy only.
  - Mutual information and a new route-swap test are diagnostics.

**Q10. Reproducibility and distributed training (§4.1, §7, §12).**
- **Random streams:** `seeds.data`, `seeds.model` and `seeds.routing` are separate, and evaluation has its own generator.
- **Activation checkpointing:** recomputation must reuse the stored routes.
- **New tests:**
  - **T11:** two ranks with gradient accumulation, a module used on one rank only and a module used on none, compared with a single process.
  - **T12:** save and resume with exploration on; routes must be identical and losses must match. The checkpoint holds `beta`, curriculum progress and the routing and data generators.

## Implementation boundary

The proposed boundary is accepted (§6, Phase A):
- **Infrastructure:** reuse the existing Trainer/Accelerate and checkpointing where their contracts match the spec, and implement the model separately from the Qwen example.
- **Model:** fixed-order equivalence, `step` K/V, bounded hard routing in `compute_matched` mode with absolute windows, and fixed, learned and random routing.
- **Routing extras:** reuse on and off, the two curricula with their router extras, and `step` decoding.
- **Before any pilot compute,** the following must pass for this subset: T1 to T5 and T8 to T12. That covers gradients, causality, masks, accumulation/DDP, resume and cached decoding.

The other K/V variants, windows and modes stay in the study but move to Phase B, after Gate 0a.

## Kept as designed

- **Step-owned K/V stays the default.** The review agrees it is the strongest idea; the central hypothesis is now stated explicitly in §1.
- **Gate 0a does not require beating dense at 1B tokens.** It does require matching dense within 0.01 nats, beating chance and showing that the learned deviations matter. A real win is required at Gate 2.

## Please report back before Stage 0a

- The dataset revision hash, and the token count and number of documents in the validation manifest.
- T8 throughput (routed vs dense) and the T11/T12 results.
- How DDP handles unused modules (unused-parameter detection, or another method).
- Any place where existing infrastructure contracts differ from the spec.
