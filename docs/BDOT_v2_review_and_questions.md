# B-DOT v2: assessment and questions before implementation

Date: 2026-10-18. Reviewed [BDOT_v2_spec.md](BDOT_v2_spec.md), version 1.1.
This is a review, not an amendment to the specification or authorization to train.
Recommendations below are proposals pending a decision.

## Assessment

**The core method is worth a small controlled pilot. The full experiment plan is
premature until that pilot establishes useful routing and acceptable runtime.**

The strongest idea is to separate computation step from module identity while
preserving a common attention context. With step-owned K/V, different tokens can
choose different Q/O/MLP modules without requiring their keys and values to come
from incompatible module-specific projections. Restricting choices to nearby
modules is a plausible way to make this easier to learn than unrestricted routing.
Unlike a proxy that approximates a deeper representation, the selected computation
is trained directly through the language-model objective.

However, common K/V projections do not make the input representations identical in
meaning or distribution. Earlier routing decisions still change those inputs.
Whether a module can work well at several nearby depths is the central hypothesis,
not something the K/V construction already guarantees.

The specification has valuable safeguards: fixed-order references, no token
dropping, explicit cache semantics, causality tests and measured-speed requirements.
Its main weakness is scope: routing, reuse, extra capacity, looping, STOP and
re-entry answer different questions. A positive result in one does not establish
the others, and implementing every variant before obtaining an initial signal is
expensive.

## Position relative to prior work

- **Shared K/V is established prior art.** Mixture of Attention Heads shares K/V
  while selecting attention experts with their own Q/O projections. B-DOT's
  proposed distinction is the use of bounded, cross-step module selection,
  including the MLP, rather than this projection split alone.
  [MoA, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278/).
- **Neighbouring-layer expert reuse and progressive expansion also have prior
  art.** ReXMoE reuses FFN experts across groups of adjacent layers and expands
  routing candidates during training. B-DOT should explain what routing the
  attention computation as well adds over that approach.
  [ReXMoE, sections 3–4](https://arxiv.org/html/2510.17483v1).
- **The summary of DNA needs qualification.** Its Table 3 reports validation loss
  2.720 for the dense model, 2.754 for top-one DNA, and 2.674 for top-two DNA.
  Table 2 lists larger total parameter counts for DNA and different active counts
  for top-two. Thus, “DNA trails a dense model at matched compute” is too broad;
  the result depends on the variant and matching criterion. This does not prove
  B-DOT will work, but neither does that paper establish a general failure of
  dynamic ordering. [DNA, Tables 2–3](https://arxiv.org/html/2506.22389v1).
- I could not retrieve the supplied LoL paper through its OpenReview link during
  this review. Its reported instability and the fidelity of `lol_chunk` remain
  unverified. Obtain the paper before treating that implementation as a faithful
  reproduction. [Supplied LoL reference](https://openreview.net/forum?id=O4y4PcNmYD).

This is a targeted literature check, not an exhaustive novelty review. The credible
contribution would be evidence that **bounded routing of attention/MLP modules,
with compatible K/V, improves a defined quality–cost tradeoff**. Combining known
components alone does not establish that contribution.

## Questions to settle before the core implementation

### 1. What result should justify expanding beyond the first pilot?

**Question:** Is the first objective better LM loss at comparable training cost,
or simply demonstrating stable, nontrivial routing?

Gate 0 currently selects a curriculum and can proceed even if routing remains on
the default path. That is useful diagnostic information but insufficient evidence
for a large grid. Off-default routing alone is also not a benefit: exploration or
balancing can induce it without improving prediction.

**Recommendation:** start with the existing dense reference and P1 curricula A/B.
If promising, prioritize a same-window random control and a no-reuse comparison.
Use validation LM loss, the existing route intervention diagnostic, and measured
runtime together to decide whether to expand. A result consistent with noise can
justify a second seed, not an automatic scale-up. This is a proposed change to
experiment ordering, not a silent replacement of Stage 0.

### 2. Are we testing order, module substitution, or both?

**Question:** Which claim is primary when `reuse=true` wins?

With reuse enabled, tokens may repeat some modules and omit others. Improvement
could come from expert selection or repeated useful computation rather than
changing the order of a fixed set of modules. With reuse disabled, exact quotas
provide a cleaner ordering test. Keep these interpretations separate.

For `kv_mode=step`, the reordered object is specifically Q/O plus MLP and its norm;
attention-input normalization and K/V stay attached to the step. Describe it this
way rather than implying that an intact conventional decoder block moves around.

### 3. What are the exact soft top-two semantics?

Sections 3.7 and 4.2 leave several consequential choices open:

- Is the pair selected using `z` or `z + beta`? Hard selection uses the latter.
- Are both deltas computed from the same incoming hidden state and visit vector?
- When `reuse=false`, only the top-one choice consumes quota, although both
  modules contribute to the state. Is exact-quota semantics intended only for the
  recorded path and subsequent hard phase, rather than all actual soft computation?
- How are ties resolved, including the two home experts in capacity mode?

**Recommendation:** explicitly select with `z + beta`, weight the selected pair
with unbiased probabilities, and compute both deltas from the same input. Preserve
the stated top-one quota rule if desired, but label soft training as a relaxation
of the hard architecture. Do not claim it executes an exact permutation.

The straight-through gate is also a **surrogate gradient**, not the exact gradient
of discrete module selection. It tells the router how scaling the selected delta
would change loss; it does not evaluate the unselected module's counterfactual
benefit. This makes exploration and the soft-to-hard transition substantive
optimization choices.

### 4. How should balancing handle empty steps and distributed training?

**Question:** What is `share_s(m)` when no token is active at step `s`? Should beta
remain unchanged for that update, or use a specified eligible-module target?

The statement that the default path is exactly balanced “with or without STOP”
requires qualifications. In compute-matched mode it holds when every step has
active tokens and chooses its own home. It does not automatically hold after an
entire later step becomes inactive. In capacity mode, always selecting expert
`2s` leaves every odd expert unused, so balance is not exact across modules.

**Recommendation:** define the policy explicitly; aggregate per-step counts and
denominators across all microbatches and ranks before computing shares. Keep beta
fixed during an optimizer update and during evaluation, and checkpoint it. Never
update beta from the current sequence before finishing that sequence's forward
pass: future assignments could then affect earlier decisions. The source method
uses historical load information for this reason.
[Auxiliary-loss-free balancing](https://arxiv.org/html/2408.15664v1).

Also specify the z-loss averaging population: active token-step pairs, all
token-step pairs with inactive entries zeroed, or equal-weight step means. These
are different objectives under STOP. Accumulation/DDP must reproduce the chosen
global normalization, and reported PPL must exclude router regularization.

### 5. What compute budget is actually matched?

**Question:** Does equal compute mean hard-inference weight FLOPs, total training
FLOPs including the soft phase, or measured elapsed training time?

The hard `step` model can match dense weight FLOPs, but curriculum B adds training
work. If K/V are computed once and shared between two candidates, soft middle
weight FLOPs are approximately `2 - kv_frac` times hard middle FLOPs, rather than
exactly two. At the defaults, this is 11/6; including four fixed layers and eight
middle steps gives about 1.56 times the hard model's block weight FLOPs during the
soft phase. A 15% soft phase adds roughly 8.3% over training under these simplified
assumptions. This excludes the LM head, attention-score operations and routing;
actual forward/backward time must be measured.

**Recommendation:** extend accounting to include both soft candidates and distinguish
nominal hard-path LE, executed training operations and runtime. Keep equal-token
comparisons, but do not call them equal-total-training-compute without adjustment.

Small router FLOPs do not imply small latency: token dispatch, small matrix
multiplications and synchronization can dominate. Likewise, computing attention
with zero queries for stopped tokens still executes those attention rows; it does
not realize all the savings implied by counting skipped modules.

### 6. Which data and baseline contracts should implementation follow?

**Questions:** Should EOS-separated documents attend across boundaries exactly as
the spec says? Should positions run continuously within each packed sequence?
Which immutable dataset revision and validation manifest apply across budgets?

The inherited helper defaults to document isolation and reset positions. The spec
instead requests EOS packing without isolation and document shuffle before
packing. These must be intentional configurations, not accidental inherited
behavior. Freeze the validation document set before selecting training prefixes;
avoid moving the held-out set with the training budget or splitting one document
between training and validation. Record token-budget rounding to optimizer steps.

T1 establishes forward equivalence with FiLM disabled. For a claim about routing
alone, decide whether the experimental reference also needs the same FiLM option:
trainable FiLM can change the transformation even when initial values are identity.
Also check backward gradients and an optimizer update against the reference;
matching initial logits is not enough to establish matching training.

## Questions for later variants, not blockers for a minimal pilot

### 7. What cache policy makes `module_band` exact?

A later query can select a module whose K/V were not needed by earlier queries.
Its cache still needs that module's projections of earlier tokens at the same
step. Specify either eager projection of every potentially needed module or
retention of earlier states for lazy projection. Relative windows make a
current-token-only candidate cache insufficient. Account for both memory and work.

Keep full-forward versus cached decoding tests. For re-entry with recomputed K/V,
define a separate state/cache contract rather than assuming the first-pass-cache
implementation remains valid.

### 8. Are STOP and re-entry labels only heuristics?

The claim that simultaneous-exit STOP labels necessarily favor later exits in a
safe direction is not established. Mixed exits change other tokens' states and
thus attention context; the approximation can err in either direction.

Re-entry labels generated by forcing 10% of tokens together similarly measure a
joint intervention, not independent per-token benefits. This may still be useful,
but deployment quality must validate it. Decision-head inputs must be available
before the decision; future answer losses are supervision only. State at position
`t` must be associated with the loss of predicting token `t+1`.

**Recommendation:** retain these as later hypotheses. First demonstrate useful
routing without either feature, then measure the actual mixed-policy quality and
speed rather than assuming the labels ensure a safe policy.

### 9. How will synthetic development and final testing be separated?

Section 10.1 chooses LR using the ID “test” set and extends training based on OOD
test improvement. Those sets are then development sets. Either rename them and
reserve a final evaluation set, or predefine choices without using final tests.

The held-out order task has only 64 distinct input symbols for one fixed operation
order. Repeated packed contexts do not turn those into thousands of independent
composition problems. Report exact accuracy on the 64 cases and variability over
independent permutation/task seeds. Separate task seed from initialization seed
and pair tasks across architectures.

Order–route mutual information is a diagnostic, not proof of useful composition:
the router can encode the instruction without executing the corresponding
operation. Interpret it alongside held-out accuracy and route interventions.

### 10. What reproducibility contract applies to stochastic routing?

Define separate data, model and routing random streams. Evaluation must preserve
training RNG. Random-route comparisons should use repeatable draws, and activation
checkpoint recomputation must reproduce the original routes. Resume must restore
beta, curriculum progress, routing RNG and all other optimizer-independent state.

Dynamic unused modules also require an explicit DDP policy. Test a module unused
on one rank but used on another, and a module unused globally. A successful
single-GPU forward does not establish distributed training correctness.

## Recommended implementation boundary

Reuse the existing Trainer/Accelerate and checkpoint infrastructure where its
contracts apply. Implement the specified model separately from the existing Qwen
example. Start with fixed-order equivalence, `step` K/V, bounded hard routing and
the two P1 curricula; validate gradients, causality, accumulation/DDP, resume and
cached decoding before spending pilot compute.

That boundary is narrower than the spec's current requirement to implement every
K/V variant before Stage 0, so it needs an explicit scope decision. It does not
mean dropping those comparisons from the eventual study.

My priority is to establish whether bounded learned routing buys useful prediction
quality at tolerable cost. If it does, the broader plan becomes justified. If it
does not, STOP, re-entry and a larger model should not be automatic rescue runs.

## Follow-up: review of the response, 2026-10-18

Read [BDOT_v2_review_response.md](BDOT_v2_review_response.md). Its decisions resolve
most of the substantive questions above. The phased build, explicit soft routing,
document isolation, stronger training-equivalence tests and pilot gate form a
reasonable basis for implementation. The following items remain.

1. **The local specification has not received the claimed update.** The response
   says every decision is in v1.2 and Appendix B, but the local file still declares
   v1.1. It still requests cross-document attention, the old build order and only
   T1–T10; it has no Appendix B. Please synchronize the specification before using
   it as the implementation contract. This review does not silently merge those
   changes or assume an unavailable v1.2 file.

2. **Qualify the balancing guarantee.** The proposed default-path target handles
   empty steps sensibly when both target and observed loads average over the same
   nonempty steps. However, in learned capacity mode, selecting only home experts
   does not imply exactly half the tokens select each one. If their observed
   shares are 0.7/0.3 and the targets are 0.5/0.5, beta changes even though every
   selection is a home expert. That is valid balancing behavior; the statement
   that the default path never changes beta is too strong. Fixed routing bypassing
   balancing does not establish this property for learned routing. State the
   aggregate target/load formula and make an all-empty update a no-op.

3. **Make the pilot comparisons unambiguous.** Gate 0a's random comparison should
   mean the separately trained control R evaluated with its own random routing,
   not randomizing the learned model at evaluation. Use repeated reproducible
   evaluation draws to assess R's sampling variability. Measure off-default share
   with exploration disabled. The forced-default intervention measures dependence
   on learned routes, not superiority to dense by itself; the other gate criteria
   remain necessary. N should inform the ordering-versus-substitution claim,
   without implicitly requiring N to win for a reuse-on pilot to pass.

4. **Route replay must retain the router's gradients.** Reusing discrete choices
   during activation-checkpoint recomputation is appropriate, but the recomputed
   forward still needs differentiable logits, straight-through gates and soft
   mixing weights. Do not reuse detached gate values. Replay also must not count
   assignments twice or update beta twice. Compare checkpointed versus ordinary
   backward gradients with exploration and soft routing enabled. This is an
   implementation requirement, not a request for another experiment arm.

5. **Scope the acceptance tests to Phase A explicitly.** Its supported subset does
   not include hash routing or all modes/K/V variants, whereas the current T5 and
   several other tests do. Run the applicable cases in Phase A and mark the others
   deferred, rather than declaring the entire old test matrix passed. In T3,
   disable z-loss and use multiple candidates so nonzero router gradients actually
   demonstrate an LM gradient through the gate.

The response's claim that LoL was checked against the full text is useful context,
but is not an independent verification by this review. Retain the stated deliberate
differences and describe `lol_chunk` as an adapted comparison.

**Conclusion:** agree with the revised direction. Resolve the specification
version mismatch and the small clarifications above; no new architecture search
is needed before implementing the limited pilot. No code changed or runs launched.

### User clarification after re-upload

The user replaced the specification with v1.2 and clarified that they were
reporting their own update, not requesting an edit. The earlier version mismatch
is resolved. The assistant's intervening v1.3 edits are superseded by the uploaded
v1.2 and must not be treated as accepted changes. The remaining findings above
are review recommendations only; the uploaded specification is authoritative.
