# Causal window-order follow-up

Decision after teacher-order: KD fixed order improves quality, but order learning
still loses to the KD control and its oracle has negligible headroom. Preserve
K2 as a useful stronger baseline. Replacing only the prefix router is not the
next hypothesis. Test learning localized routing jointly from scratch instead.
Motivation and primary-source boundaries: ROUTING_LITERATURE_RECHECK_20261119.md.
This is our small exploratory study, not a reproduction or a novelty claim.

Use existing six-layer 768-wide model initialization, original pilot per-seed
12000-update schedules, batch32, context512, prefix32, LR.0006/warmup300/cosine.
One full-attention stem, four core applications, one full-attention coda.
Core library contains four independent blocks. Each core depth partitions the
sequence into contiguous windows128 with alternating offsets0/64. Attention
inside each core block is causal and restricted to its own window. There is no
cross-expert KV sharing; the full stem and coda retain global causal context.
Do not change the historical training data convention silently: source packed
rows and suffix scoring remain identical to the pilot, including its document
boundary convention. Report that this differs from document-blocked variants.

Per-depth route features: first window uses only its first hidden token; other
windows use the mean of the means of preceding windows at the same depth.
A shared four-output linear router receives a depth embedding. Softmax logits
receive Gaussian noise0.1 during training only. Selection argmax includes a
non-gradient expert bias; selected softmax probability scales the whole block
residual delta. Update per-expert balance bias by.001*sign(mean_load-load) after
each training update. Initialize router weights/bias/depth embedding at zero.
No auxiliary route-label loss. Store balance buffers and RNG in fullstate.
No FiLM or depth curriculum in this first controlled screen.

Four arms times two seeds, eight new fits:
- WF: fixed library order0,1,2,3, residual coefficient.25.
- WR: pseudorandom per-window route based only on already visible token prefix,
  fixed residual coefficient.25; deterministic and batch-invariant at eval.
- WG: fixed order0,1,2,3 with learned router residual coefficient.
- WL: learned route and learned residual coefficient, with balancing/noise.

All instantiate the same blocks/router; freeze unused router in WF/WR. WG
separates learned gating from learned order. Evaluate historical full-attention
D6 seed0/1 checkpoints again without retraining. All predictions remain strictly
causal, including first/partial windows; no choice can depend on its scored
future suffix. Full NLL scoring remains conditional on first32 tokens.

Evaluate DCLM3475/FWE4096, route histograms/entropy/switch rates, parameter counts,
measured full-context latency and training throughput. Report active block-token
work and attention-window work, not assumed equal FLOPs. Gate a quality signal
on WL beating WG and WR by.005nats, and D6 by.01nats, in each seed/domain. These
reused finals remain exploratory; confirmation requires untouched examples.
Also retain prior K2 numbers as a nonmatched-budget reference, clearly labeled.

Required before admission: independent per-window reference equivalence and
backprop, future-token independence of hidden states/routes, partial-window
padding, batch invariance, optimizer/balance/RNG exact resume, actual eight-arm
entrypoint+report/export tests and GPU gradient/numerical gate. Benchmark first40
before full continuation. Preserve separate output/frozen source; validate input
hashes/native ownership. Source tree currently near25MiB: make space losslessly
with an explicit restore manifest if needed. Export only compact result arrays
and proofs, keep weights/fullstate on node with verified backup/native handoff.
Watcher9736 stays active. This protocol is preparation, NOT yet GPU-launched.
