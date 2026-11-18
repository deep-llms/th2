# Next screen: causal adaptive final refinement

Status: complete local implementation; nine CPU tests passed, including six-arm miniature train/eval/report/backup/export replay and exact body/head resume. Node admission and GPU numerical gates are required before production fits. User explicitly authorizes continuing optimize/evaluate/try.
Parent idea: `warmup_DOT_idea_sketch.md`, now testing its adaptive-depth direction.

## Why change direction

The completed component/deep-feature diagnostic did not pass any registered
continuation criterion. Deep features change NLL by less than 0.001 nats;
component permutations damage dense checkpoints. Preserve these negative results.
The next hypothesis is to retain the six-layer canonical backbone and learn an
optional seventh residual refinement block. Adding useful computation is a
separate hypothesis from permuting already specialized layers.

CALM studies confidence-driven early exits and explicitly addresses missing
past hidden states: [paper](https://arxiv.org/abs/2207.07061). MoR combines shared
blocks with adaptive recursion and explicit KV handling:
[paper](https://arxiv.org/abs/2507.10524). Neither paper establishes a win for our
small model. Our choice is an engineering inference: placing the optional block
last makes its K/V depend only on the available canonical hidden states, so
skipping its output at an earlier token does not remove K/V needed later.
This is adaptive extra depth, not a reproduction of either paper.

## Fixed initial screen

Initialize both seeds from the completed local-order C checkpoints (the best
canonical dense control), not the weaker original pilot checkpoints. Three fits
per seed, six in total, 2000 updates × batch32 × context512 =32.768M input tokens:

- C6: canonical six-layer continuation, CE only.
- F7: frozen C backbone + trainable optional seventh block. Copy its initial
  attention/MLP from the coda, then zero attention output and FFN down projections,
  making the initial refinement exactly identity. Optimize refined CE only.
- J7: same identity-initialized extension, jointly tune backbone and extension;
  loss =0.5 canonical CE +0.5 refined CE, keeping both exits trained.

Base/body LR1e-4, AdamW betas(.9,.95), weight decay0.1, clip1, warmup50 and the
existing fixed schedule. No early stop from final scores. Same unique training
row order per seed across arms. Reserve new training/router-fit/validation rows
excluding prior pilot/local/token-router training and routing rows. Router-fit
and validation documents must be disjoint. Prefer new final documents; if not
available, keep the reused-final exploratory flag rather than quietly reuse.

After fitting, freeze all bodies. For each F7/J7 endpoint fit a two-action
causal router on canonical final hidden state h_t to choose base or refinement
for predicting x_(t+1). Fit4096/validate512 rows, eight epochs, MLP768→128→2,
AdamW.001/wd.01, full-information expected NLL on detached counterfactual costs.
Keep best validation hard NLL epoch. Final evaluation cannot select epoch,
threshold or seed. Controls: always base, always refinement, random routing
with validation-matched selection rate, and target-aware oracle diagnostic.

## Real dispatch and fair claims

Compute the canonical backbone once. Optional-block K/V use its input for all
past positions in each sequence with any selected query. Compute Q/attention
output/FFN only for selected positions, retaining original positions and causal
mask. No cross-sequence mixing, no future routing scores or target features.
Skipped output positions retain base hidden state. Both the all-selected and
arbitrary-mask outputs must match dense-then-select independently; future-token
perturbations must preserve earlier outputs/routes. No shared mutable K/V among
incompatible paths. This screen uses full contexts, not a generation-speed claim.

Report parameter counts, train tokens, actual seconds, base and extra
block-token work, router and K/V overhead, and sparse-vs-dense measured latency.
Equal token budgets are not equal FLOPs: F7 has no backbone backward, J7 has a
second CE head. Do not claim equal-compute superiority. If promising, the next
confirmation must match measured compute and use untouched data/additional seeds.

A quality signal requires learned routing at least0.01nats below C6 on both
seeds/domains. An adaptive-depth signal separately requires ≤0.01nats worse than
always-refine with ≥25% of refinement queries skipped in both seeds/domains.
A deployment speed claim additionally requires measured latency improvement;
sparse Python dispatch may lose despite fewer MACs. Report failed criteria.

Before GPU launch: finish trainer/fullstate and data reservation, prove CPU exact
resume and real entrypoint, verify BF16 against FP32 gradients/causality, then
fresh source/input/ownership admission. Keep all artifacts and node-only weights;
small reports/replay/backup/native handoff. Keep the existing watcher across this
and later iterations; a failed screen does not stop the research task.
