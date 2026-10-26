# First hypothesis: joint selection of attention groups

26 October 2026 SGT. Proposed design for discussion. No real-model experiment,
router fit, architecture implementation or GPU job has been run for this idea.

## The question

Can a small router preserve pretrained model quality better by selecting a
**combination of GQA groups whose joint output approximates dense attention**,
instead of selecting groups by independent magnitude scores, at the same number
of active groups?

The primary objective here is batched decode latency. Persistent KV cache size
is unchanged. This candidate does not involve dynamic layer order, deep-state
prediction or proxy injection.

## Closest prior work and proposed distinction

Polar Sparsity trains attention routers using the heads with the largest output
norms as supervision. It also supports group selection for GQA. This is the
closest practical reference, including the sparse execution strategy.
[Polar Sparsity, section 4.2](https://arxiv.org/html/2505.14884v3).

Our hypothesis is that a norm-based target can spend the budget on outputs that
overlap or cancel, and ignores how the output projection transforms them.
Joint output preservation accounts for those interactions. Whether this matters
enough for real LM quality and a deployable router is unknown.

This is **not an established novelty claim**. Subset selection for head pruning
is longstanding: [Differentiable Subset Pruning](https://aclanthology.org/2021.tacl-1.86/)
learns a fixed-budget head subset, and reconstruction/compensation objectives
already occur in structured pruning. The potential distinction is per-token
GQA-group policies with joint teacher targets and a cheap causal predictor.
A fuller comparison is required before any publication claim.

## Minimal architecture

Keep the pretrained layer order, residual connections, MLPs, normalization, RoPE
and head projections. For each layer and token:

1. Compute a small router from the layer's normalized input, before attention.
2. Select a fixed-size subset of whole GQA groups. Every query head belonging to
   a selected KV group participates; unselected head outputs are zero.
3. Produce and store **all** K/V groups. A later token may select a previously
   unused group and require its history.
4. Read and execute attention only for selected groups, then apply the ordinary
   output projection. Keep any output bias exactly once.

Start with frozen backbone parameters and train only the router. Initially retain
dense QKV and output projection execution and count their costs.

For a model with eight KV groups, keeping six permits 28 subsets; keeping four
permits 70. A linear classifier over these choices can predict the whole subset.
These are examples, not a selected model or run. Begin with moderate sparsity.
Enumeration does not scale to arbitrary group counts: larger models would need
a bounded subset library or another selector, introducing another approximation.

## Teacher target and loss

Let `a_h(t)` be the dense output of query head `h`, and `W_O,h` its block in the
output projection. Define the contribution of KV group `g` as

`c_g(t) = sum_{h in group g} W_O,h a_h(t)`.

Ignoring the common bias, the full output is `y(t) = sum_g c_g(t)`. For retained
group set `S`, define

`D_t(S) = ||y(t) - sum_{g in S} c_g(t)||²`.

This measures error after the output projection. It is an intermediate-output
surrogate, **not a guarantee about LM loss or downstream correctness**.

If `Q_ij(t) = c_i(t)^T c_j(t)` and `z` marks omitted groups, then
`D_t(S) = z^T Q(t) z`. The cross terms are exactly what independent norm ranking
omits. This expression is exact for the isolated layer at fixed input and cache.

Collect teacher group contributions at sampled causal positions in calibration
documents. Candidate errors can be computed from one dense layer evaluation
per position; no full-model pass per subset is needed. Activation collection
and Gram construction still cost compute and must be reported.

A possible router objective is cross-entropy against
`q_t(S) = softmax(-D_t(S) / (tau * s_layer))`, with a positive, fixed per-layer
scale estimated on calibration data. Soft labels retain near-equivalent choices.
The router predicts `p(S | normalized_input_t)`; inference selects its argmax.
Teacher outputs and actual candidate errors are unavailable to the deployed
router. Specify scale, temperature and tie handling before a fitted run.

Separate calibration/router-fitting documents from evaluation. Report error
regret relative to the best subset; classification accuracy alone penalizes
nearly tied choices unnecessarily.

## Algebra checked locally

Two four-group examples, keeping two groups, show that even ranking projected
contribution norms can be suboptimal:

| Group contributions | Largest-norm choice: squared error | Joint optimum: squared error |
| --- | --- | --- |
| `(10,0), (-10,0), (0,1), (0,1)` | First two: 4 | Last two: 0 |
| `(10,0), (10,0), (0,9), (0,9)` | First two: 324 | One horizontal + one vertical: 181 |

All six subsets per fixture were enumerated with standard-library Python; direct
error equaled the Gram quadratic exactly. These are mathematical counterexamples,
not measurements of head geometry or model improvement.

## Correctness and execution

Use the same causal selection policy during prefill and decoding. A dense
reference may compute all head outputs and mask unused outputs to check quality
and cached-decode equivalence; it is not a fast implementation.

Real sparse execution must avoid unselected groups' attention work and historical
KV reads, while still writing every group's current KV. It must initialize
skipped output slots, preserve positions and document boundaries, and use
deterministic selection/ties. The router cannot access future tokens or full
attention outputs. Different prefill and decode policies would require a
separately specified hybrid inference evaluation.

Always-dense layer exceptions must be identical across methods. Fixed group
count approximately matches attention work, but router/dispatch costs can differ.
Measure complete decoding with identical precision, cache layout and backend.
An active fraction does not imply a proportional speedup.

## Smallest informative test — not launched

Before LM adaptation, use one selected pretrained model and fixed held-out
documents. Compare at the same group budget:

- A static subset selected on calibration data.
- Independent magnitude selection, including projected-group norms so that
  gains cannot be attributed solely to applying `W_O`.
- Joint subset selection using actual dense contributions, **offline only**.

The third is a best-case local reconstruction reference. Its advantage for its
own objective is guaranteed by construction; what matters is gap size and
whether held-out LM predictions improve. Teacher-fixed activations do not
establish whole-model quality. Verify the altered model on its own causal states,
using dense same-input references only for the diagnostic choices. Do not time
this oracle as a deployable model or call it an oracle for end-to-end NLL.

If there is a useful LM advantage, fit a small router and a capacity-matched
magnitude-supervised router. Keep the classifier family and execution budgets
matched to isolate the target's effect; include the original independent-score
router as the practical reference. Evaluate all layers together on the sparse
trajectory, then measure decode latency and a fixed downstream suite.

These comparisons test one mechanism, rather than unrelated architecture fits.
Model, data, budget and hardware are still unassigned.

## Reasons this may fail

- Real GQA models may have little local reconstruction headroom.
- Lower output MSE may preserve irrelevant directions and harm important logits.
- Dense-teacher targets may not transfer to sparse multi-layer execution.
- Current hidden states may not predict history-dependent subset choices well
  enough; the offline reference could be unattainable with a cheap router.
- Group selection can remove a critical query head together with an unimportant
  one; whole-group removal is a coarse intervention.
- Router/dispatch overhead may erase the attention savings.
- Static selection or a simpler smaller-attention architecture may match the
  router, and existing subset-pruning methods may cover the eventual contribution.

Do not expand to more losses or backbone training simply because the first test
disappoints. Establish useful, predictable headroom at the intended serving cost.

## Other candidate mechanisms already have close precedents

- Per-head local/full switching: [All-or-Here Attention](https://arxiv.org/html/2512.22562v1).
- Token/layer sliding-window/full switching: [Switch Attention](https://arxiv.org/html/2603.26380v1).
- Local-first decoding with a learned decision to recompute globally:
  [On-Demand Attention](https://arxiv.org/html/2609.20734v1).

These are existing alternatives, not new inventions of this project. The present
candidate is an incremental, falsifiable hypothesis; it has not earned a claim
of novelty, improvement or justification for substantial training.
