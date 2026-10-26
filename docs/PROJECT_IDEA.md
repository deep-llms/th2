# Efficient LLM architecture — initial project brief

26 October 2026 SGT. This brief records the user's direction from the discussion;
it is not a finalized architecture specification.

## Objective

Develop an LLM architecture that preserves useful model quality while reducing
inference latency, computation or memory. Higher accuracy is not required if
similar quality is achieved at meaningfully lower measured cost.

This is a new project, separate from B-DOT's dynamic layer ordering and earlier
deep-to-shallow proxy work. Ordinary fixed layer order is compatible with the
objective. Reuse the template's training, evaluation and runner helpers where
appropriate; do not inherit the old projects' custom architectures.

## Directions under consideration

- Selective attention heads or GQA groups.
- Shared or reduced KV representations, including cross-layer reuse.
- Attention experts, potentially combined with feed-forward experts.
- Better allocation of attention resources under a serving budget.

Combining existing techniques alone does not establish novelty.
The [literature review](EFFICIENT_LLM_RESEARCH_20261026.md)
covers relevant prior work and causal/hardware constraints.

## Current assessment

Structured compression or adaptation of a pretrained GQA model is the current
starting point. The user approved the initial diagnostic/router fit and now
further bounded research experiments. See CURRENT_TASK.md for exact scope.

The first concrete candidate is documented in
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md): predict joint
subsets of GQA groups using post-projection reconstruction targets. Real-model
results favor joint over independent supervision but still lose quality to dense;
see ROUTER_PILOT_RESULTS_20261026.md. The next authorized screen preserves local
attention and reduces global-history access: CONTEXT_BUDGET_PLAN_20261026.md.
Neither result establishes novelty or practical efficiency yet.

Keep KV storage, KV reads, attention FLOPs and whole-model latency separate.
Benchmark a real serving workload against an appropriate optimized baseline.
Use meaningful LM/downstream evaluation and account for adaptation cost. Tests
on an existing small model may establish implementation correctness without
establishing quality or performance at another scale.

## Next design deliverable

A concise method specification identifying the closest prior work, the precise
technical difference, why it should help, its failure modes and the smallest
informative comparison. Model, data, context length, quality tolerance and compute
budget should follow that design rather than being inherited from examples.

## Initialization

User requested a direct copy of `/disk/thuat/new_project_template` into a sibling
of `/disk/thuat/DOT`. The new working folder is `/disk/thuat/efficient_llm`.
At initialization only project documentation was customized. Subsequent authorized
implementation, deployment and experiment history is recorded in CURRENT_TASK.md.
