# Shared-block depth screen

Registered after the extra-block LR screen: both seeds selected1e-3 on
validation, but final gains over C6 remained0.00029–0.00187nats, below0.01.
Preserve all prior studies. This next study tests depth sharing trained from
scratch, not arbitrary permutations of already-specialized pretrained layers.
It is adaptive depth, not evidence for dynamic layer order.

Primary sources: [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
combines shared layers with adaptive recursion; [LayerSkip](https://arxiv.org/abs/2404.16710)
trains for exits using layer dropout and exit losses. These motivate training
variable computation explicitly; the following small experiment is our own
inference/design, not a reproduction or promised improvement.

Reuse the original pilot D6 seed0/1 checkpoints and train-row schedules. New
models initialize by constructing the identical seeded six-layer Model and
retaining only its first two middle blocks, keeping embedding/stem/coda exactly
matched. Stem→(middle0→middle1)^r→coda has4 unique blocks and effective4/6/8.
No recurrence across examples and no cross-route mutable KV cache.

Six new fits, seed0/1 each: D4 fixed r1; RF fixed r2; RV deterministic minibatch
cycle r1,r2,r2,r3.12000steps,batch32,context512,prefix32,peakLR.0006,warmup300,
cosine10%floor,AdamW(.9,.95),wd.1,clip1, same data/order as original D6. Thus
RV/RF/D6 have equal six block-token forwards averaged across training, but
parameter, optimizer and hardware time differ; report them, do not assert equal
FLOPs. D4 is a parameter-matched control. Do not retrain original D6 or old DOT.

Evaluate all eight models on the same DCLM3475/FWE4096 exploratory final rows.
RV also evaluates all three depths on4096fit/512valid rows, disjoint from both
pilot training orders and fit/validation document-disjoint. A prefix-only
router sees the last normalized depth1 state computed on32 prefix tokens only.
MLP d→128SiLU→3 predicts row-centered counterfactual NLL with MSE,30epochs,
AdamW.001/wd.01,batch256, private seeded permutations. Select epoch by validation
MSE. Then select a fixed scalar compute penalty from a registered log-spaced
nonnegative grid, minimizing validation chosen NLL subject to average full
block-token work≤6 blocks/token, including the four-block prefix pass. With
512/32 this means mean r≤1.875. If necessary use explicit always-shallow fallback.
No ranking across final examples, no future/target-dependent routing, no final
score selection. Report achieved final compute under distribution shift.

Controls: fixed depth2 from the same RV, validation-best constant depth,
validation-rate matched random, RF, D4, D6, target-aware oracle (unachievable).
Report every depth and both seeds/domains. A continuation signal requires learned
NLL≥0.01 below D6 in both seeds/domains, actual block-token work≤6 on each, and
learned≥0.005 below the same RV fixed-depth2. Also show paired document intervals,
route counts, measured full-context latency versus D6 including router and
partitioning. No generation-speed claim; confidence intervals are not across
training seeds. Any promising result requires untouched-data/compute confirmation.

Reuse tested trainer/checkpoint machinery in a new module, preserve sources and
outputs. CPU causality/dispatch gradient/init/fullstate/reservation/export tests,
GPU BF16/FP32 and first40 gates before six fits. Re-evaluate two existing D6
checkpoints without training. Automatic report/replay,15node backups (six
weight/fullstate pairs,two router fullstates,compact archive),native handoff.
Only small results/proof exported; watcher9736 continues through next decision.
