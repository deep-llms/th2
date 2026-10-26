# Matched 6/8 router pilot

Approved 26 October 2026 SGT after the first
[joint-group diagnostic](JOINT_GROUP_RESULTS_20261026.md). Scope: fit two small
routers on a frozen pretrained model and profile dense cached decoding. No
backbone adaptation or sparse-kernel development is queued. Completed results:
[ROUTER_PILOT_RESULTS_20261026.md](ROUTER_PILOT_RESULTS_20261026.md).

## Comparison

Use the same original 28-layer Qwen3-0.6B-Base and pinned English CulturaX pool
as the first diagnostic. All eight K/V groups are produced and cached. At each
layer, a linear classifier reads the current normalized input **before attention**
and chooses one of the 28 possible six-group subsets. Dense attention followed
by output masking remains the correctness reference, not an efficient kernel.

Two banks have identical parameter count, initialization, examples, optimizer,
schedule and inference cost. Only their supervision differs:

- Joint: squared norm of the sum of omitted post-output-projection contributions.
- Independent: sum of the squared norms of those omitted contributions.

The second is a capacity-matched control, not a reproduction of Polar Sparsity's
router. Both use soft targets proportional to `exp(-cost / (0.5 * scale_layer))`.
The shared layer scale is the training mean of the joint costs' standard
deviation across subsets, floored at 1e-8. No evaluation-based tuning.

## Frozen protocol

- Seed 42; 1,024 training documents and 256 new validation documents. Prefixes
  contain 2,048 model-input positions plus the next-token target. Exclude all
  128 documents/prefixes from the earlier held-out diagnostic. Train/eval are
  disjoint by text and token-prefix hashes. The earlier 64 training documents
  may recur. This is a bounded long-document probe, not full-pool PPL.
- One frozen BF16 dense-teacher pass over 2,097,152 input tokens. Sample every
  sixteenth position: 131,072 training examples, each containing all 28 layers.
  Features are BF16; projected contributions, subset costs and routers are FP32.
- Native HF Trainer/Accelerate, eight GPUs, 256 examples/device, global batch
  2,048, 512 updates (eight passes through cached features), no accumulation.
  AdamW, LR 0.001, cosine schedule, 26 warmup steps, weight decay 0.01.
- Both independent router banks train in one process group. Sum their separate
  mean cross-entropies. Disable gradient clipping so one bank cannot rescale
  the other's gradients. No loss or optimizer touches backbone parameters.
- Save native checkpoints at steps 256 and 512, retaining both, plus a strict
  inference export. Reload and verify identical state; check trained-router
  causal and FP32 full/cached decoding consistency before evaluating.

Evaluate dense, static, raw-norm, projected-norm, joint oracle, learned joint
and learned independent policies on all 256 held-out documents. Each policy
runs on its own causal hidden states. Report token NLL/PPL, paired document
bootstrap intervals, and sampled reconstruction regret on each router's own
trajectory. The dense-teacher/routed-state distribution shift is a limitation;
no automatic teacher refresh or backbone fine-tuning follows this pilot.

## Cached-decode profile

Native dense SDPA, BF16, all 28 layers, HF StaticCache, batch sizes 1 and 8,
contexts 2,048 and 8,192. Eight GPUs run two independent device replicates per
case, without collectives in timed regions. Synthetic fixed token IDs; compute
one next-token logit vector per step. Exclude prefill from decode timing.

Each measurement has four warmup decode steps followed by 32 timed steps;
discard one full workload warmup and report three uninstrumented repetitions.
A separate pass records CUDA events around the native SDPA core without changing
its arguments or output. Verify exactly 28 core calls per decode step and FP32
StaticCache equivalence first. Wall latency is measured without timing hooks.

The estimate `1 / (1 - 0.25 * attention_fraction)` assumes attention-core time
scales linearly when keeping six groups. It excludes router/dispatch costs and
is **not measured sparse speedup or a universal performance bound**. Event hooks
add overhead; this eager HF result does not characterize an optimized serving
engine. QKV/output projections, MLPs, logits and dense KV storage are unchanged.

## Execution and decision

`efficient_llm/routers.py` implements targets, teacher hooks and inference.
`router_pilot.py` uses native Trainer and the existing diagnostic scorer.
`router_check.py` is the real-model GPU gate; `decode_profile.py` profiles decode.
`scripts/launch_group_diagnostic.py --study router` reuses the existing immutable
snapshot, asset hash verification, Accelerate cache installation, verified burn
reclamation, free-GPU checks, sequential jobs and automatic burn handoff.

Require both successful exit and validated artifacts. A failed check prevents
later jobs. Preserve all outputs on failure. The supervisor restores and verifies
communicating burns after success or failure, provided GPU ownership is safe.

Continue toward a sparse kernel only if the learned joint router has a useful
advantage over its matched control and the target decode workload offers enough
headroom to pay for routing. A short adaptation experiment would require a
separate decision. Negative results are grounds to stop this specific design.
