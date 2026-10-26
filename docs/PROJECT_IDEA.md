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

No one of these has been selected. Combining existing techniques alone does not
establish novelty. The [literature review](EFFICIENT_LLM_RESEARCH_20261026.md)
covers relevant prior work and causal/hardware constraints.

## Current assessment

Structured compression or adaptation of a pretrained GQA model is a candidate
starting point. We still need a specific weakness in existing methods, a proposed
mechanism addressing that weakness, and a small experiment that could disprove
the hypothesis. The user has not approved training other methods simply to search
for a direction.

The first concrete candidate is documented in
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md): predict joint
subsets of GQA groups using post-projection reconstruction targets. It remains
a proposal requiring critical review; no real-model evidence or novelty claim
has been established.

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
Only project documentation is customized at initialization. No training, remote
push, GPU job or environment installation is part of this setup.
