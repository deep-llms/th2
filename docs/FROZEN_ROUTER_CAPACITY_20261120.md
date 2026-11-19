# Frozen-backbone router capacity screen

Status: local implementation and tests; not admitted or launched. Predecessor: `results/exposure_recovery_20261119_a02/REPORT_FINAL.md`. Continue the user-authorized dynamic-order research; preserve every previous output and old DOT-RL pause.

Recovery reduced SF's exposure penalty, but SF remains worse than dense C by 0.001306 nat DCLM and 0.001479 nat FWE (two-seed means). S without exposure remains nearly canonical, with only approximately 0.00012 nat soft improvement and worse hard projection. These reused exploratory scores do not establish a mechanism win.

The next hypothesis is that updating the backbone together with the router lets the backbone absorb the LM gradient while the router remains near its initial canonical policy. Test router learning with the backbone frozen. This is a new hypothesis, not an explanation established by the preceding results. A more expressive causal router is evaluated against both a global policy and the previous shared linear router.

## Registered arms and common recipe

Use the two authenticated terminal recovery **C** parents (2000 recovery updates after their original 2000-update continuation). Extract the dense backbone only, retaining the parent's numerical tensors exactly. All embeddings, six transformer blocks, normalization and tied LM output weights are frozen. No body optimizer or gradients. Preserve gradients through frozen blocks with respect to their inputs: do not detach the hidden states or put the whole backbone forward under `no_grad`.

Six fits, G/L/M × seed0/1:

- G: four depth-specific global 4-way logits.
- L: the same global logits plus the previous shared LayerNorm/linear current-token router.
- M: global logits plus separate LayerNorm → Linear(d,128) → SiLU → Linear(128,4) heads for each of four depths. Output layers start at zero. This compares a depth-specific nonlinear head package to L, not an isolated parameter-count effect.

All arms initialize canonical probability **0.99**, other choices each 0.01/3, and use the same terminal C backbone. Hard argmax initially reproduces the dense parent exactly. No uniform floor, temperature adaptation, route balance loss or target-derived router input. Gates depend only on the current causal hidden state and depth. Soft forward still computes all 16 middle candidates, plus stem/coda. Hard projection is an evaluation diagnostic and carries no sparse-speed claim.

Use 2000 updates, batch32, length512/prefix32, suffix480 CE. Fresh router-only AdamW betas(.9,.95), weight decay0, LR1e-3 with100-update linear warmup and cosine to1e-4, clip1. Same recorded recovery `train-s0/1.npy` and derived `recovery_train` bytes within each seed, independent of arm. These are body-training rows reused for router fitting; explicitly report this. Final/validation reservations are not used for fitting. Do not select LR, architecture, checkpoint or stopping using final scores. Save exact optimizer/RNG/row-position fullstate; first40 interrupted/resumed checkpoint must match.

Evaluate both existing DCLM3475/FWE4096 exploratory domains: unchanged dense C (reuse its verified arrays), G/L/M soft and hard terminal policies, route histograms and noncanonical/input dependence. Compare M to G, L and C. Registered soft continuation criterion: M beats C by0.005 nat and G and L by0.001 nat in both seeds/domains. A discrete-order claim separately needs the same hard comparisons and noncanonical input-dependent routes. Report all signed deltas and paired document CIs regardless of criteria. Gains from soft mixing alone are not evidence for useful discrete ordering.

## Admission and completion

Test causal masking including changed future tokens, zero-head initialization, exact parent hard outputs, independent mixture/reference gradients, frozen tensor identity before/after optimizer steps, actual three-arm entrypoints and exact fullstate resume. Verify authenticated production C parent/config/hash, common row hashes, fresh GPU ownership, source/runtime, BF16/FP32 numerical agreement and fullset independent-reference parity before accepting results. Frozen weights must remain byte-equivalent tensor values to the parent in every terminal model; router state is the only learned component. A frozen backbone does not make an expensive all-candidate forward free; record actual throughput and work separately.

Keep fresh outputs, source locks, report/raw replay, verified node backups and native handoff. Download only compact results/proof; large weights and states remain on node. Keep reused-final/multiple-comparison, two-seed, repeated body-training rows, unequal compute, conditional-suffix and soft-versus-hard limitations. Watcher9736 remains active across this and subsequent authorized iterations. No GPU submission until trainer, reservation/evaluation/report/queue integration and frozen production admission pass.
