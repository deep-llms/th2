# Selective decode and retrieval gate

26 October 2026 SGT. The user approved continuing after the context-budget screen.
Reuse its frozen Qwen3 backbone, fitted routers, and allocations. No new training,
installation or download. This tests execution and a missing quality dimension.

## Execution

Single-token cached decode uses an inference-only Triton split-K kernel. Each
query head reads its GQA group's full history or its first four plus recent 128
keys, as chosen before attention. The kernel indexes native KV directly; it does
not copy the full cache or compute both branches. Partial softmax numerators and
denominators are accumulated in FP32 and merged stably. Local and global ranges
are disjoint when needed, and overlapping sink/window tokens occur once.

The softmax reduction follows the standard principle documented in the
[Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
Masked loads follow [Triton's memory contract](https://triton-lang.org/main/python-api/generated/triton.language.load.html).
This is an original small implementation, not a claim to improve FlashAttention.
All KV still remains stored. No cache-capacity or training-throughput claim.

Prefill retains the previous dense/reference implementation and routing semantics;
only single-token attention execution changes. Fixed/native SDPA paths remain
available. No variable-length batch, padding, paged cache or backward API is added.

## Gates and comparisons

1. CPU Triton interpreter tests: mixed groups, arbitrary valid lengths, noncontiguous
   KV, poisoned unused cache, and a real tiny-model cached trajectory. These do not
   substitute for compiled GPU tests.
2. B200 gate: compiled FP32/BF16 attention against an FP32 SDPA reference with the
   same operands, including padded NaNs and inaccessible-history NaNs. FP32 attention
   tolerance atol 3e-5/rtol 3e-4; BF16 atol .002/rtol .01. Whole-model FP32 cached
   dense/fixed/joint/independent comparisons use atol .002/rtol 3e-4. Stop on failure.
3. Measure whole-model BF16 decode for dense native SDPA, **dense using the same new
   kernel**, fixed groups with native execution, fixed groups with the kernel,
   joint-router two-branch reference and joint-router selective execution.
   Batch 1/8 × context 2k/8k; two GPU replicates per case, rotating policy order,
   one workload warmup, three repeats of 32 steps. Report native and same-kernel
   speed ratios separately; kernel improvements are not routing improvements.
   Compilation/prefill excluded from decode timing, no CUDA graphs/full compilation.
4. Compare BF16 own-trajectory next-token losses on 32 cached positions from eight
   fixed real validation documents (one per GPU). Report predictions and per-token
   NLL, reject absolute mean NLL difference above .01 nats per document. This is a
   numerical sanity check, not another full-corpus quality comparison.

## Retrieval sanity check

A seeded synthetic record lookup with eight balanced single-token answer choices
(chance 12.5%). Two demonstration lookups use different identifiers and non-color
values. Test records assign colors to eight random keys. Query a key after filler
text; all eight colors occur in records, so color frequency cannot identify it.
Target record at 10%, 50%, 90% of 2k/8k context or within the final 128 tokens.
There are 16 balanced examples per context/location cell: 128 cases total. The
same records/answers appear at different contexts/positions; cells are correlated.
Seed 9042, frozen before scoring. No prompt/answer fitting after inspecting results.

Compare dense, local-only, fixed groups, joint router, independent router and
selective joint execution. Score the final candidate token after cached prefill
and a one-token query step, report eight-choice accuracy and log probabilities.
Use dense and near-target controls to check whether the base model understands
this task; a failed positive control makes retrieval conclusions inconclusive.
This is a sanity check, not RULER/NIAH reproduction, free generation, or a standard
benchmark score. Report every cell, including failures, without cherry-picking.

## Safe run

Verify pinned prior-fit files/model and freeze the input bundle before reclaiming
GPUs. Reuse the existing supervisor: copy/verify Accelerate, identify exact known
burn workers, stop only those workers, verify all GPUs free, then GPU gate → all-eight
profile → all-eight retrieval. Automatic verified communicating burn on success or
failure. Preserve earlier outputs. Export hashes/results and return commands.sh to #0.
