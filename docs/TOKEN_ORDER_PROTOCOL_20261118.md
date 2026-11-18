# Token-wise causal order selection, 18 November 2026

Status: registered before new labels/results. Autonomous follow-up authorized by the user; builds on `docs/warmup_DOT_idea_sketch.md`. Preserve every prior study. No old DOT-RL resume or backbone retraining.

## Question and limits

The prefix router chose a single order for 480 target tokens. Its sequence oracle did not beat dense; local CF/PG selected canonical everywhere. Test whether token-wise causal features reveal useful order specialization. This is a hypothesis, not a diagnosed bug in previous runs.

Reuse the six frozen pilot D/R/W checkpoints, seeds 0/1, 76M parameters, six blocks, 512 tokens, 32-token unscored prefix. For each R/W, the eight existing full-depth dihedral orders are separate causal order experts. At input position t, shared-stem state h_t selects the order predicting token t+1. No target, future token, counterfactual loss, or oracle choice enters inference features. Freeze all backbone parameters.

This is an **all-route computation experiment**: each order processes its own entire causal history, with separate intermediate/KV semantics. Selecting token predictions does NOT implement a cheap mixed-path transformer or share incompatible KV states. Count all eight forwards, the extra stem forward, cache generation, router training and evaluation. No equal-compute, inference speedup or cheap sparse-dispatch claim. If quality improves, efficient implementation is a separate experiment.

Relevant primary research: [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) studies token-level recursive computation and KV handling; [Dr.LLM](https://arxiv.org/abs/2510.12773) studies dynamic layer routing. Neither establishes this experiment's success or novelty.

## Locked data and fitting

Use exact restored, golden-hash-verified GPT-NeoX DCLM/FWE assets. All arms share the same indices. Reserve 4096 fitting and 512 validation single-document 512-token windows from the DCLM train stream, excluding all pilot training/router-fit windows and local-order training/router-fit windows. Seed 55190 selects fit; seed 55191 selects validation from documents absent in fit. Training documents may have occurred in old backbone training; only new windows are promised. Neither final-set domain is used for router fitting/selection.

Reuse precisely local-order DCLM3475/FWE4096 final windows, without inspecting new scores to select examples. This is explicitly exploratory: these final sets have already informed the research direction. Two domains and two seeds are a screen, not an untouched confirmatory experiment.

At each fit/validation position, compute FP32 target NLL for all eight orders and detached FP32 causal stem features. Fit LayerNorm(768)→Linear(128)→SiLU→Linear(8) with AdamW lr0.001, weight_decay0.01, batch4096 tokens, eight passes, seed 55200+backbone_seed. Objective is expected NLL under softmax routing, using detached complete counterfactual costs; subtracting per-token mean cost is only a variance/scale offset. Select the epoch with lowest validation hard-greedy NLL (earlier wins ties). No hyperparameter search or test selection. Save optimizer/generator/head state each epoch, and verify interrupted versus uninterrupted training on CPU.

Constant route is selected using fit mean NLL. Diagnostic token-ID-only lookup chooses from fit costs with 32-observation smoothing toward global route costs and a global fallback, then freezes. Existing prefix router is preserved and evaluated as a separate historical control.

## Comparisons

Primary: hard learned token router versus same-seed dense D, and versus fit-selected constant, independently on both domains. Record per-sequence NLL, token counts, document-cluster paired 2000-resample 95% CIs (seed901), route counts and switches. A promising quality screen needs both seeds improve by at least0.01 nats versus constant and dense on both domains. Report any mixed outcome without calling it a win.

Controls: canonical, uniform expected NLL, fit-selected constant, frozen old prefix router, token-ID-only selector. Uniform probability ensemble is a separate diagnostic (log-sum-exp of probabilities; not mean NLL). Per-token oracle is an impossible target-aware diagnostic, not a deployable method. No test-time candidate selection.

## Correctness, execution and handoff

FP32/no autocast/no TF32 for labels and evaluation. Test exact target shift, sequence-NLL parity, future-token perturbation of causal features and route probabilities, batch/singleton parity, detached costs/backbone immutability, expected-loss gradient direction, token-ID lookup no target input, resume identity, real CPU entrypoint and independent report replay. Run corresponding production numerical/causality probes before each backbone's collection. Fail gates explicitly; do not loosen thresholds to admit results.

Six independent jobs on physical GPUs0–5, one checkpoint per GPU. Keep common safe ownership/guard/native handoff machinery. Admission verifies node/runtime/source/data/checkpoints and previous native identities; guarded hold/reinspection before exact owned-native reclaim. Frozen source, unique output. Automatic report→node replay→small backup→native restoration→owned guard release. Only result/proof/router-head archive under25MB exported; raw token arrays, cached features and original backbone weights stay on node. Existing backups are referenced with actual hashes. Watcher every30min remains until handoff verification.
