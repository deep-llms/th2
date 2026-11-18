# Single-window counterfactual diagnostic, 19 November 2026

Candidate after the slot-affine screen completes and its registered criterion is negative. No new training. Preserve every previous checkpoint/output. User authorizes continuous optimization and diagnostic experiments.

Question: do the trained pool blocks offer useful alternatives that the causal router fails to choose, or is another router unlikely to help? The slot-affine effect observed so far is tiny. Diagnose the operator choice directly before changing its learning algorithm again.

Frozen GF and LF endpoints, seed0/1. Two intervention modes per endpoint, eight independent GPU jobs: (a) replace one selected expert and retain its factual residual gate; (b) replace one expert and recompute the selected probability gate. All other decisions recompute naturally from the intervened hidden states. Use the canonical fixed-shape all-expert evaluator, not grouped execution. No gradient update or optimizer exists in this diagnostic.

Use only the first128 recorded validation sequences from the existing adaptive-refinement eval reservation; no final-set selection. For each of18 window/depth decisions and each of4 expert identities, measure full suffix480 NLL. Record all72 intervention costs plus factual NLL and factual routes. Every intervention that forces the already selected expert must reproduce factual NLL exactly. A frozen D6C same-seed baseline is evaluated on identical examples. Its repeated outputs across jobs must match.

The per-example minimum over factual and all interventions is a **label-using, single-intervention oracle**, not a deployable causal router. Do not sum gains across windows or claim a multi-intervention optimum. Report paired document-cluster intervals for best single-edit gain and oracle-minus-dense, positive-gain fraction, and selected-action distribution. Mean oracle headroom above.005nats in both LF seeds suggests testing counterfactual router supervision next; if it remains worse than dense, this alone is insufficient for a claimed quality win. Compare both gate modes; no best-mode final-test selection.

Required tests: exact identity for factual choices, causal intervention forward behavior, cost enumeration against separate forwards, no checkpoint mutation, real miniature eight-job report/archive/replay, tamper rejection. Freeze parent report/weights/data/source hashes; check current GPU ownership and runtime before launch. Source/data checks at both ends. Backup only the compact diagnostic archive; parent weights retain verified slot-affine backups. Native handoff/ownedguard release, bounded export/local replay, persistent watcher. All reused-validation/two-seed/oracle/unequal-compute/weights-node-only caveats remain.

Status: candidate; no CPU admission or GPU job submitted.
