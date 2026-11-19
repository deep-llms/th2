# Adjacent-pair composition screen

Status: complete local pipeline; seven local and seven independently frozen CPU tests PASS. Not admitted or launched. This follows
`docs/FROZEN_ROUTER_CAPACITY_20261120.md` and the user's continued authorization
to test dynamic-order designs. Original objective:
`docs/warmup_DOT_idea_sketch.md`. Preserve all prior results.

## Decision and hypothesis

Frozen-router capacity did not pass its registered criteria. M's soft gains are
about 0.00005 nat, with hard outputs equal to dense. Larger router heads alone
have not established useful order selection. Existing four-way depth mixtures
can repeat a block and omit another. The next hypothesis is that restricting
choices to complete adjacent-pair compositions makes changes less destructive.
This is an untested hypothesis, not a prediction of a win.

The earlier teacher-anchored adjacent-order screen trained whole-sequence
identity/last-pair-swap paths and used a detached prefix cost router. This study
instead jointly optimizes causal token gates over two complete block
compositions at each of two pairs. It uses no teacher or KD. It is not a rerun
of that screen, but the earlier negative result reduces confidence.

Research context: [Routing Networks and the Challenges of Modular and
Compositional Computation](https://arxiv.org/abs/1904.12774) examines interactions
between module selection and module updates. [Library of Layers](https://arxiv.org/abs/2611.03526)
learns dynamic block selection with strided-window attention and reports
competitive language-model performance. Neither paper validates this paired
composition design. The proposed change and its interpretation are ours.

## Registered design

Six fits: C/G/M, seeds 0/1, from the same authenticated recovered C parent in
each seed. All backbone parameters are trainable, unlike the frozen-router
study. Keep stem, four middle blocks, coda, tied embedding/head and the actual
parent model dimensions. Length512, prefix32, suffix480.

For pair (A,B), compute u=B(A(h)) and v=A(B(h)), each with full causal attention
and the same positions. Blend their output states with a two-way gate evaluated
from the pair input h. Then apply pair (C,D) in the same way. Gate M has a
separate LN→Linear(d,128)→SiLU→Linear(128,2) head per pair; final heads zero.
G has only two global logits per pair. Both initialize canonical probability
0.99. C computes two identical canonical branches and averages them with equal
constant weights; this exercises the same number of block applications.

Every branch uses both blocks in its pair once. Each candidate forward performs
10 block applications including stem/coda, compared with ordinary dense6.
This matches coarse candidate count across the three training arms, not exact
FLOPs/time/parameters. No sparse implementation or generation-speed claim.
Hard projection selects one composition per token, but attention context can
contain states selected by other tokens at earlier pairs; it is not a single
whole-sequence permutation. Report that distinction explicitly.

All arms receive the same recorded recovery rows per seed, batch32, 2000updates.
These rows previously trained the parent: repeated exposure is a limitation,
not fresh-data confirmation. Fresh AdamW(.9,.95), wd0.1 for body, wd0 for
router; body peakLR3e-4, router peakLR1e-3, 100update warmup, cosine to10% of
peak, global grad clip1. C has no trainable router. Save all optimizer groups,
RNG, row cursor and schedules in fullstate. No checkpoint selection on final.

Evaluate terminal soft/hard policies on the same DCLM3475/FWE4096 exploratory
sets. Include ordinary dense-parent arrays as a descriptive reference; the
new equally continued C is the primary baseline. Report each seed, both
domains, all deltas/paired document CIs, gate entropy, hard choice frequencies,
and input dependence. Registered soft signal: M beats continued C by0.005 nat
and G by0.001 in each seed/domain. A discrete-order signal separately requires
these hard gains and nonconstant input-dependent choices. A soft-only gain
does not establish a discrete-order win. Preserve negative results.

## Required work before submission

Authenticate actual parent schema and four parent sidecars, registered row
identity/reservations, source locks and fresh native ownership. Test independent
branch and gate reference outputs/gradients, exact canonical hard initialization,
matched C body gradient, future-token causality for both modes with nonzero
heads, trainability of both pairs, exact optimizer/RNG/row/schedule resume,
all three actual trainer entrypoints, report/export/replay and backup pipeline.
Production GPU tests include BF16/FP32 agreement, exact canonical shape causal
check, first40 resume and full-set reference NLL/routes parity. Do not weaken
gates to obtain a result. No launch before these admission stages pass.

New output only; 6 fits plus paired controls, report/raw replay, 13 node copies
(6 weights, 6 fullstates, result archive), verified native handoff/guard release.
Only compact results/proof return locally. Keep reused-final/multiple-comparison,
two-seed, repeated-body-rows, conditional-suffix, compute and soft/hard flags.
Watcher9736 remains active; old DOT-RL stays paused. Runner/infrastructure is
unchanged. Local model/authenticated parent loader/trainer: six CPU tests PASS, including
three-arm entrypoints and exact model/optimizer/RNG resume across warmup.
Reservation/evaluation/report/queue and local true-frozen integration now PASS, including all six fits, unchanged parents, 13 backup copies, bounded export/replay and tamper rejection. Remaining: source-cap accounting, actual-parent admission and production GPU checks. Parent-only reference arrays are labelled P; newly continued C is the primary control.
