# Component and causal-feature diagnostics

Registered before execution, 18 November 2026. Authorized by the user's approval
of the revised diagnostic and explicit instruction to keep optimizing/evaluating/
trying. Parent idea: `docs/warmup_DOT_idea_sketch.md`. Prior studies are immutable.

## Questions and fixed comparisons

1. On the two frozen dense pilot checkpoints, exchange adjacent middle modules
   at positions 0/1, 1/2, or 2/3. Separately exchange attention+its input RMSNorm,
   FFN+its post-attention RMSNorm, or both (the whole block). Four paths including
   canonical per component, identical residual placement, six jobs in total.
   No fitting or selection from final losses. Report each swap and canonical;
   an average swap penalty is only a diagnostic, not a deployable model.
2. On frozen R/W × seeds 0/1, replace the previous router's causal stem feature
   with the normalized final hidden state of the canonical path at position t,
   predicting token t+1. Same head architecture, initialization, AdamW settings,
   eight passes, fit/validation IDs, target costs and best-validation selection
   as the preceding token screen. The final target is never in the feature.
   Reuse hash-verified per-token eight-path costs and old route selections.
   Compare deep vs stem, constant, token-ID, prefix and dense; retain oracle and
   probability ensemble as diagnostic bounds. Four jobs; no backbone updates.

Use prior indices: 4096 fit, 512 validation from disjoint training documents;
3475 DCLM and 4096 FWE final sequences, each 512 tokens with 32-token conditional
prefix/480 targets. Final sets are reused, hence exploratory. No claim of a
fresh confirmatory test or training-seed confidence interval. Use the original
paired document-cluster bootstrap, 2000 replicates/seed901.

## Integrity and compute

CPU tests: module identity and whole-block equivalence, manual composition,
future-token perturbation, loss/target shift, real small entrypoint, raw-array
replay and tamper rejection. Reuse tested fullstate head resume; additionally
exercise it for deep features. Production FP32/no TF32 checks canonical,
batching, causality and old-cache parity. Refuse fresh-output collisions or
source/data/checkpoint/cache mismatches. No environment/network mutation.

One physical GPU per job, at most eight concurrent. Use existing verified
native ownership/guard/admission/handoff helpers. Source and all input hashes
pinned before execution; output, head and backup hashes checked afterwards.
Full raw costs/features stay on node. Export only compact per-sequence losses,
route counts, reports, head weights and small proofs, under 25MB per artifact.

Count canonical deep forwards and head training, as well as the inherited
all-eight-expert cost. This is a feature diagnostic, **not** an efficient
conditional-dispatch or shared-KV implementation. Report wall time per job.

## Decision rules for the next autonomous iteration

- Deeper features are promising only if learned loss improves over the old stem
  router by at least 0.01 nats in both seeds on both domains. A win versus dense
  additionally requires the same margin over dense; these are separate claims.
- FFN interchangeability merits a training pilot if its mean three-swap penalty
  is below 0.02 nats and at most half both attention and whole-block penalties in
  both seeds/domains. Report all penalties regardless; this does not prove a
  learnable router or causality of one submodule in isolation.
- If neither passes, deprioritize order permutations and investigate adaptive
  depth/repetition, using an equal-compute dense control and a newly registered
  protocol. Method changes need rationale, tests and separate outputs, not
  edits to completed controls. Negative findings do not end the research task.

Watcher remains active across studies under the user's ongoing instruction.
Infra failures stop remote writes; diagnosed project failures are repaired.
