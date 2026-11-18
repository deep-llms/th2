# Window-order evaluation recovery, 19 November 2026

Scope: evaluation only of the eight completed window-order checkpoints and two historical D6 controls from `docs/WINDOW_ORDER_20261119.md`. No training, parent mutation or replacement of the original failed outputs. GPU admission, guard ownership and native handoff reuse the existing tested helpers.

The original grouped implementation exceeded its FP32 hidden-prefix causal threshold of 2e-5 for WR-s0, WR-s1 and WL-s1 (2.19345e-5, 2.33352e-5, 2.09808e-5). All eight training terminal states and actual weights/fullstate SHA were verified. The queue failed closed and restored native workers. The working hypothesis is route-dependent batch regrouping roundoff; it is not accepted as an explanation without an independent fixed-shape audit.

## Preregistered audit

The canonical evaluator sends every padded window to each of four experts at the same fixed batch shape, then selects the recorded causal route. It has identical model parameters, masks, positions, gates and route policy. It computes all experts and is explicitly not a sparse speed implementation.

Require exact zero future-prefix hidden changes at cuts 1,32,128,511 and exact unchanged prefix routes. Require reference/grouped hidden maximum error <=1e-4 and relative L2 <=1e-5 on original and perturbed inputs; record the original grouped causal discrepancy without erasing its failed gate. Over every final-set sequence in both domains, require identical route arrays and NLL absolute difference <=2e-4. Fail closed if any requirement fails. Canonical NLL is primary; retain grouped NLL and per-sequence deltas. These new parity tolerances do not retroactively relax the original causal gate.

Use the unchanged 3475 DCLM /4096 FWE sequence identities, token normalization, paired cluster confidence intervals and original continuation criterion. All eight models plus both D6 controls receive the same new evaluator. Two seeds and reused final sets remain exploratory. Training precision and unequal compute remain caveats; successful FP32 inference checks do not establish mathematical equivalence of every original training update.

Freeze source and 35 input/lineage hashes before CPU admission; node validates all actual bytes. Local tests cover forward/gradient agreement, exact causality, ten evaluation jobs, unchanged parent trees, replay, tamper rejection and 17 verified same-node backup copies. Weights/fullstate remain on node; only bounded result archive and backup proof are exported. Report original failure metadata and new proof side by side. Native handoff and owned guard release must be verified before the next experiment.

FiLM is only a candidate follow-up until this audit and final report resolve the parent study. Watcher remains active across studies.
