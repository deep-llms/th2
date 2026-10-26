# Next screen: preserve heads, reduce global-context reads

26 October 2026, SGT. User authorized further research and bounded B200 tests.
This replaces the earlier pause after the matched head-removal pilot. No backbone
adaptation, new environment, or long training sweep is included.

## Why this direction

The learned 6/8 head-removal router still increases PPL from 18.9674 to 23.4644.
It removes useful local computation as well as expensive history access. Instead,
keep every head active and give some groups **first four sink tokens plus the
most recent 128 tokens**, including the current token. Other groups retain full
causal history. Use original absolute RoPE positions and deduplicate overlapping
sink/window ranges. The model retains all native KV; this screen tests reduced
reads/attention work, not reduced cache capacity.

This direction has strong precedents, so it is not a novelty claim:

- [DuoAttention](https://arxiv.org/html/2410.10819v1) divides heads into retrieval
  and streaming groups and optimizes their execution/cache layout. Its learned
  static allocation uses distillation and synthetic retrieval data. Our natural
  text calibration is a simple control, not a reproduction of that method.
- [All-or-Here Attention](https://arxiv.org/html/2512.22562v1) learns per-head,
  per-token local/global decisions with task training and a sparsity penalty.
  Our screen freezes the backbone and isolates the router supervision target.
- [On-Demand Attention](https://arxiv.org/html/2609.20734v1) uses local-first
  inference and conditional global recomputation with a learned recall decision.
  We avoid whole-model recomputation and compare group versus layer allocation.
- [Switch Attention](https://arxiv.org/html/2603.26380v1) is another close
  local/global switching precedent. A positive screen would still need a clear
  contribution beyond combining these existing mechanisms.

## The mechanism under test

For a fixed input/cache in a layer, compute full and local head outputs. Let
`d_g = W_O,g (a_full,g - a_local,g)` for GQA group `g`. If `S` uses global
attention, the exact output discrepancy before the residual addition is
`||sum_{g not in S} d_g||²`. Compare this joint cost with the independent cost
`sum_{g not in S} ||d_g||²`.

Two identical linear classifier banks predict one of the 70 four-of-eight
subsets from the normalized layer input, before attention. They share initialization,
data, scale, optimizer and schedule; only teacher cost differs. This predicts
which groups need history, without predicting the attention output itself.
Teacher/oracle calculations are diagnostics and not a deployable fast path.

## Fixed experiment

Frozen original 28-layer Qwen3-0.6B-Base; existing verified English CulturaX pool;
native SDPA/BF16. Keep the previous pilot's 1,024 training prefixes and 256
development prefixes, each with 2,048 next-token targets. The earlier 128-document
diagnostic's validation documents remain excluded. The 256-document set is reused
development evidence, not a newly untouched test set.

Teacher features every 16th position: 131,072 examples. Native HF Trainer,
eight GPUs, 256 examples/device, 512 optimizer updates/eight feature passes,
AdamW LR 0.001, cosine, 26 warmup updates, weight decay 0.01, seed 42.
No clipping couples the two banks. Backbone weights receive no gradients.

Evaluate these eight policies on their own causal model trajectories:

| Policy | Global budget | Allocation |
| --- | --- | --- |
| Dense | 100% | Original model |
| Local only | 0% | Sink + recent in every group |
| Fixed groups | 50% | Best four-group subset per layer by mean training joint cost |
| Fixed layers | 50% | All groups global in 14 layers; rank training whole-layer local error / dense output energy |
| Periodic layers | 50% | Global attention in even layers, local in odd layers |
| Joint oracle | 50% | Best current full/local joint reconstruction subset; expensive reference |
| Learned joint | 50% | Pre-attention classifier, joint cost target |
| Learned independent | 50% | Matched classifier, independent cost target |

Also freeze 32 validation prefixes with 8,192 targets before GPU reclamation.
They may overlap the 2k validation documents; report the count. Reject overlap
with router-fitting documents. Preparation additionally selects eight long
training prefixes that are **unused** by this study. Use bounded pinned source
shards, not a claim of uniform sampling from the entire pool. Freeze all allocations
and settings on training data before scoring either validation scope.

Report per-document NLL, PPL and paired document-bootstrap CIs. Long-document PPL
does not establish long-range retrieval preservation or downstream accuracy.

## Correctness and measured performance

`context_attention.py` implements both-branch quality references and an executable
static policy. The latter reorders current QKV into contiguous global/local groups
before writing the cache, attends only the global history slices, and slices local
KV to sink+recent before SDPA. Restore original head order before unchanged `W_O`.
Whole-layer policies need only one attention branch per layer. Cache reuse across
different policy/layout instances is rejected. Supported streams are contiguous,
unpadded, with automatic positions; this is not a general generation API.

Local tests compare sliced attention to explicit masked attention, optimized static
execution to the both-branch reference, full forwards to incremental DynamicCache
and StaticCache, prefix causality, teacher identity, restoration and actual two-rank
HF Trainer execution. Run the corresponding pretrained-weight FP32 numerical gate
on B200 before fitting; fail rather than relaxing a failed tolerance silently.

Measure **whole-model cached decode**, batch 1/8 × context 2k/8k, 32 measured steps,
three repeats, two GPU replicates per case. Dense, local-only and all three static
policies get fresh caches and workload warmups; rotate policy timing order.
No collectives occur during timing. Report actual wall latency, including policy,
projection, MLP and output-logit costs. Retain all KV allocation in every method.
No learned-router speed claim: its quality reference computes both branches.
This is eager HF timing, not optimized-serving or CUDA-graph performance.

Reuse the verified supervisor: hash assets before stopping burns, copy/verify
Accelerate config, inspect exact burn PID/start identities, reclaim only known
workers, verify all GPUs free, run gated jobs, then restore and independently
verify communicating burns on success or failure. No driver/environment changes.

Prefer the simplest policy on the measured quality/latency frontier. Do not fund
backbone adaptation or custom kernels unless this screen finds useful headroom.
