# Dense restart factorial diagnostic

This follows the completed `docs/PAIR_ORDER_CANDIDATE_20261120.md` under the user's authorization to continue controlled experiments. Status: six local CPU tests PASS, including six-fit integration and reused FH equivalence; frozen tests, actual-parent admission and GPU gates remain. Not launched.

Pair C lost about 0.054/0.058 nat versus its recovered parent on DCLM/FWE. This does not identify the cause: fresh Adam, a new high-LR schedule and repeated rows changed together. Diagnose retention before another router change.

## Fixed design

A 2×2 factorial crosses optimizer state (fresh F / inherited I) with LR policy (high H / low L), two seeds. Reuse the completed Pair C runs as FH; train only FL, IH, IL (six new fits). P is the unchanged recovered C reference. No rerun of Pair C or old DOT-RL.

All cells use the same recovered C weights, recorded 64,000 rows per seed, batch32, length512, prefix32, 2,000 updates, dense Pair C duplicate-canonical forward (10 candidate block applications), AdamW betas(.9,.95), wd0.1 and clip1. High is the original 100-update warmup/cosine peak3e-4 to3e-5. Low is constant1e-5, the actual parent terminal LR. This isolates the LR-policy package, not peak LR alone. All rows repeat parent training exposure; that factor stays fixed and is not identified here.

For I, authenticate the completed recovery fullstate and map Adam moments and step counters by reconstructed source parameter names into the identical dense body. Verify complete state coverage, tensor shapes/dtypes/finiteness, group settings and step count (pre-recovery updates plus recovery updates). Never equate the phase counter with cumulative Adam steps. Fresh and inherited arms use the original Pair C model-construction/RNG recipe; only Adam state and scheduled LR differ. Save exact new-phase optimizer/model/RNG/cursor state; this is a new experiment, not an unchanged continuation of the parent.

## Analysis, without final-set selection

Evaluate terminal checkpoints on the same exploratory DCLM3475/FWE4096 arrays. Report all four cells and parent per seed/domain, paired document CIs, LR effects FH−FL and IH−IL, reset effects FH−IH and FL−IL, and interaction (FH−FL)−(IH−IL). Positive deltas mean worse NLL. A practically consistent retention signal requires at least0.005nat improvement over FH in each seed/domain; also report every delta to P. No choice of best checkpoint, seed or final-set subset. This is a diagnostic, not an order/router win.

## Admission and delivery

CPU tests: actual parameter-name Adam mapping, direct continued-update equivalence, reset/factor separation, all schedules, exact fullstate resume, tamper rejection, all-six pipeline/control identity/export/replay. Authenticate six actual parent loads and FH recipe/data/source/runtime compatibility before deployment. Production gates: numerical/causal/initial dense identity, first40 resume, independent full-set reference parity, current source/data/GPU ownership. Preserve prior outputs. New namespace `restart_factorial_20261120_a01`; six fits, 13 same-node backup copies, verified native handoff. Only compact results/proof local; weights/fullstates stay on node.

Keep reused-final/multiple-comparisons/two-seed/repeated-rows/conditional-suffix/compute-accounting flags. Historical FH reuse adds execution-time/runtime sensitivity; check its runtime and forward recipe rather than assuming matching code names. Watcher9736 remains active across this diagnostic and later method work.
