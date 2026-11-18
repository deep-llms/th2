# Conditional follow-up: slot-conditioned window blocks

Prepared while the registered window-order run is still active; no outcome is
assumed. Activate this follow-up only if WL fails its registered continuation
criterion. If WL wins, first confirm on unused document-separated final examples
instead. Preserve every window-order output and its frozen source.

Hypothesis: a shared block sees different hidden-state distributions at different
execution depths. The current depth embedding informs only the router. A small
per-slot affine input transformation may help the reusable blocks themselves.
This is an engineering hypothesis, not a claimed reproduction or novelty result.

For core slot d, let z=(1+gamma[d])*h+beta[d], initialized with gamma=beta=0.
The selected residual is h + gate * (Block(z)-z). Router features and choices use
h exactly as before. No token information is added; transforms are static learned
vectors. Four slots add 6144 parameters at width768. All arms instantiate them;
freeze them at zero in the no-FiLM controls. Initial forward outputs must match
the parent's checkpoint before any update. Parent balance and router weights are
retained; this is new training with a fresh optimizer, not an exact parent resume.

Registered arms, each seed0/1, 2000 new updates:
- G0/GF: initialize from the corresponding WG window checkpoint; fixed order,
  respectively frozen-zero / trainable slot affine.
- L0/LF: initialize from corresponding WL; learned order,
  respectively frozen-zero / trainable slot affine.
- D6C: initialize the existing full-attention D6; continue for the same new rows.

Use the already recorded adaptive-refinement 64000 training rows, shared across
arms, verifying no overlap with original pilot training rows and fit/validation
reservations. Context512, batch32, LR3e-4, warmup100, cosine to10%, same optimizer
and suffix480 objective. No LR/model selection from final scores. All other
routing/window settings stay at the parent values. Use ten independent jobs on
at most eight owned GPUs; no edits to the current active queue.

Evaluate all arms on the same two exploratory final sets with per-sequence NLL,
paired document CIs, routing statistics and measured full-context latency. A
promising learned-order signal requires LF to beat L0 and GF by .005nats and D6C
by .01nats in each seed/domain. Otherwise record a negative screen. A positive
result still requires untouched confirmation examples. Report all arms even if
one is negative. Do not compare only the best final row.

Required before launch: identity initialization, per-slot reference forward and
gradients, causal future-token independence, exact new-run fullstate resume
including parent router/balance state, real ten-job miniature pipeline/export,
node CPU tests, parent/source/data actual hashes and fresh GPU ownership. Require
GPU numerical gates and first40 fullstate continuation. New source/output and
compact result-only export; original/continued weights stay on node with verified
same-node backups and native handoff. Watcher remains active across studies.

Status: candidate protocol only; no new GPU job submitted.

Numerical amendment before admission: original window evaluation hit a grouped-batch FP32 causal threshold. This follow-up must await the recovery report and then use a fixed-shape all-expert reference for exact causal checks and primary NLL; retain grouped full-set NLL/route parity and both latency scopes. Do not claim all-expert reference timing is sparse execution. The trained grouped implementation remains separately flagged.
