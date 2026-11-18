# Warm-order causal LM pilot — 18 November 2026

Status: authorized autonomous research, implementation and CPU testing; not yet launched.
Input idea: `docs/warmup_DOT_idea_sketch.md` (SHA256 63ed8b0729b1e7d59de482d000b2def44710a4f092d779fff3debe6caaa3c729).

## Research and choice

[Library of Layers](https://arxiv.org/abs/2611.03526), [full paper](https://arxiv.org/html/2611.03526v1), already studies routing through shared transformer blocks and variable depth. Its treatment of causal routing and attention compatibility is directly relevant. Dynamic layer order alone is therefore not a novelty claim. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) studies learned compute allocation, a related but different question. [Routing Networks](https://arxiv.org/abs/1711.01239) provides a modular RL precedent. [LayerShuffle](https://arxiv.org/abs/2407.04513) is evidence for vision, not proof that shuffled causal LMs win.

First isolate two questions: does shallow warm-up improve a reorderable LM, and can a causal context router exploit its order differences? Enumerating eight actions gives exact counterfactual training losses. Use these full-information labels first instead of a noisy RL reward; RL/joint finetuning remains an authorized later experiment if measurable headroom exists. Do not select methods using test outcomes and then present that test as untouched confirmation.

## Model and controls

Six decoder blocks: fixed stem, four reorderable middle blocks, fixed coda; width768,12 query heads,4 KV heads,64 head dimension, SwiGLU2048, tied50304 vocabulary, RoPE and QK normalization. Reuse tested DOT block implementation. No STOP or repeated block initially. Eight paths are four rotations of forward/reverse order; each middle appears twice at each position. All scored routes use six blocks.

Use single-document512-token windows. The first32 tokens are observed context; score exactly480 suffix targets. Route features come only from the prefix through the shared stem. Thus the route cannot observe scored tokens. This is **conditional suffix language modelling**, not a result for unrestricted generation, per-token routing or full-sequence perplexity. Route features add a32-token stem pass; report this cost. Group samples by whole-sequence route to retain conventional causal attention and avoid incompatible mixed-order KV histories.

Train D (fixed order), R (balanced varied orders from start), W (warm-up) at seeds0 and1. W first10% updates use one middle block, next10% two, remaining80% all four. Stem/coda and selected middle blocks all train; no independently trained models are spliced. R/W use deterministic balanced path cycles, not stochastic random draws. Each seed uses identical initialization and sample order across arms. Initial budget12000 updates ×32×512 =196,608,000 input tokens,184,320,000 scored tokens. AdamW LR0.0006, betas0.9/0.95, decay0.1, clip1, warmup300 updates then cosine to10%. The real entry point must be benchmarked before launch and any budget change recorded before outcomes.

Token budgets match; warm-up uses fewer block applications. Log actual block-token counts and wall time, including routing-label collection and head fitting. This screening does **not** establish the sketch's equal-total-compute success criterion. A promising result requires a subsequent matched-compute confirmation before such a claim.

## Data and router fitting

Reuse verified OHMV DOT E2 DCLM train2B for backbone and routing-fit rows, with explicit immutable selected row indices and hashes. Original DCLM tuning/final document partitions supply dev/test, preventing same-document fragments from being split into dev/test by a new random window split. Filter any window with a document start after position0; no modification of restored arrays. Select enough eligible train windows without replacement; distinct8192 training-domain windows supply router labels. Dev2048, test8192 windows (reduce only if insufficient, with counts recorded before runs).

For each frozen R/W checkpoint, evaluate all eight orders on routing-fit/dev/test rows and save per-row NLL, row IDs and prefix features. Only routing-fit labels train the router. Tiny LayerNorm/MLP128/8 head, soft cost targets at temperature0.02 nats; select head epoch by mean dev NLL from actual selected routes, not classification accuracy. Best constant route is chosen solely on routing-fit data. Dev/test labels never update model/head parameters. Test is read only after head selection is frozen.

Report dense canonical, R/W uniform mean, canonical, best training-selected constant, learned router and diagnostic per-example oracle. Oracle is explicitly unavailable at deployment. Same-backbone learned-vs-constant/uniform isolates routing; W-vs-R isolates curriculum with stated compute difference. Two seeds and paired document-cluster bootstrap intervals; retain all outcomes. Initially require learned to improve best constant by≥0.01 nats on both seeds and W-learned to be within0.01 of D before expanding. A miss is informative, not a reason to relabel a diagnostic oracle as learned performance.

## Operational gates

Data full READY/golden hashes and a2_env/pipcheck verified from health2dee062. No old experiment resumes. Before GPU launch require fresh native burn worker identity/ancestry, source/input checks, CPU tests (causality, loss/gradients, save/resume and rejection), then B200 real-entrypoint numerical/throughput tests. Stop only freshly verified native worker PIDs per GPU_SAFETY, never PID1/launcher/groups. Freeze source and manifests outside sync tree. Unique output root, full optimizer/RNG/data-position checkpoint; no blind restart. Automatically train→fit/evaluate→report/small backup→verified native handoff. Remote receipt before further writes. Weights/fullstate stay on node. Existing watcher4091122 remains until experiment/report/replay/backup handoff.
