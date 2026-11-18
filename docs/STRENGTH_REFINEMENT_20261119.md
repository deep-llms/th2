# Extra-block optimization diagnostic

Registered after budget-refinement: all 24 learned-vs-random gains were below
0.001 nats and all sparse body timings lost to dense. Preserve those results.
The next hypothesis is narrow: the identity-initialized seventh block may be
under-optimized by a 1e-4 peak LR over 32.768M tokens. Test whether useful extra
computation exists before fitting another router. This is an optimization
screen, not a claim about dynamic order or an equal-compute win.

Four new fits: frozen six-layer base, seventh block peak LR3e-4 or1e-3,
seed0/1. Start each from exactly the same local-order C checkpoint as the prior
F7; same copied-coda/zero-output initialization, same2000 updates, batch32,
context512, warmup50, cosine to10% peak, AdamW(.9,.95), wd0.1, clip1 and exact
recorded training row order. Only the peak LR differs. Reuse the old1e-4 F7
and C6 controls with their verified per-sequence arrays; do not train them again.
No body or router from earlier studies is modified.

Evaluate each endpoint on the same512 validation rows and two reused exploratory
final sets. Select LR from{1e-4,3e-4,1e-3} by validation refined NLL separately
per seed, then report both new settings and the selected pair. No selection by
final scores. All two seeds and both domains must improve on matched C6 by
at least0.01 nats to justify a stronger quality follow-up. Also report gain
against the original frozen base and low-LR F7, target-aware oracle headroom,
paired document intervals and actual training time. An oracle is not an
achievable router. No new routing fits unless this block-value screen warrants
one; a promising result needs fresh held-out confirmation and compute matching.

Reuse the tested trainer unchanged. Check input/source hashes, identity
initialization, frozen-base equality, fullstate resume, finite gradients,
BF16/FP32 and causal output gates. Each new job runs first40 then exact-state
continuation; save final weights/fullstate and small per-sequence results.
Report raw-cost replay on node, local summary replay,9 verified node backup
copies (four weight/fullstate pairs plus compact report), then native handoff.
Only compact results/proof leave the node. Keep watcher9736 across decisions.
