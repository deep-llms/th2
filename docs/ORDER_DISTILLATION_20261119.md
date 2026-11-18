# Teacher-anchored adjacent-order screen

Decision after shared-depth screening: RV's target-aware depth oracle remains
worse than D6 on both domains and seeds. More router capacity cannot close that
backbone gap. Return to six distinct layers and test whether a frozen dense
teacher preserves quality while teaching two interchangeable middle-layer orders.
This is a hypothesis, not evidence that distillation will win.

Use the same local-order C seed0/1 initialization and recorded 2000-update rows
as the adaptive-refinement study. Preserve all old outputs. Context512,prefix32,
batch32,AdamW(.9,.95),weight decay.1,clip1,peakLR3e-4,warmup100,cosine10%floor.
Freeze a separate copy of the initial dense model as teacher. All student
parameters remain trainable. Two paths: identity(0,1,2,3) and last-pair swap
(0,1,3,2), with the shared stem/coda unchanged.

Eight new fits, four arms times two seeds:

- C2: two identity passes, CE only.
- K2: two identity passes, CE +0.5 KL(teacher || student).
- P2: identity and swap, CE only.
- PK2: identity and swap, CE +0.5 KL(teacher || student).

Each step averages both path objectives, one optimizer update. Teacher always
runs identity and is detached, temperature1. Both CE and KL are calculated for
all arms, including zero KL coefficient controls, so measured coarse forward
work is matched; do not call this exact FLOP or time equivalence. Teacher
sees precisely the same causal input; loss scores suffix480 tokens only. Chunk
vocabulary projections and checkpoint chunk losses to bound GPU memory, without
approximating vocabulary distributions. No teacher gradient or teacher updates.

First compare validation identity/swap costs and oracle headroom. If swap never
improves any validation example, skip unnecessary router training and report the
fixed identity result. Otherwise fit a prefix-only two-action cost regressor,
using existing disjoint4096fit/512valid reservations. Choose head epoch on
validation only; never choose architecture/hyperparameters from final scores.
Evaluate all four arms, both paths, and applicable routers on DCLM3475/FWE4096.
These final sets are reused exploratory screens; any win needs untouched-data
confirmation. Show K2 vs C2 and PK2 vs P2 to isolate distillation, PK2 vs K2 to
isolate order, learned vs validation-best constant/random/oracle, and prefix
routing overhead. Constant six-block paths are equal depth, not automatically
equal latency. No generation-speed claims.

Continuation signal: PK2 learned NLL at least0.01 below C2 and0.005 below K2
in both seeds and domains, with learned beating its own best validation-fixed
route by0.001. Report all other comparisons regardless of thresholds. No claim
of a dynamic-order win from a distillation-only improvement.

Before launch: test exact chunked CE/KL/gradients against an unchunked reference,
teacher freeze, causal path/prefix independence, all four actual training arms,
fullstate resume, data/source locks, report/replay/export and GPU numerics.
Prepare fresh output/source admission and verified GPU ownership; do not change
previous frozen sources or rerun old DOT-RL. Only small result exports, fullstate
and weights on node with verified backups and native handoff. Watcher9736 stays
active across studies. Implementation is in progress, not launched.
