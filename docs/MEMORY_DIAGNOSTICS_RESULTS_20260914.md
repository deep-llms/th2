# Memory-off and contribution diagnostics — seed 17

Completed and verified on 2026-09-14. These are **post-hoc A1/A2 diagnostics**
from `ccm_next_steps_after_seed17.md`, not new training or primary preregistered
comparisons. All three completed Stage-2 checkpoints are at step 3815.
Only `D_dev` was used; no locked validation or seed-29 memory training.

## Main finding

All three trained models still benefit from their memory branch at inference.
Disabling it raises NLL and also makes each model worse than Base-Continue.
Contextual has a larger inference-time dependency and contribution magnitude
than Isolated; Grad has the largest of both. This supports a live memory
contribution, not an explanation in which the measured improvement is entirely
independent of memory after training. It does **not** partition training effects
causally, establish a specifically deep-computation advantage, or supply an
independent-backbone replication. Grad still has the best normal NLL.

## A1: Overall NLL

Base-Continue NLL: **3.214783136**.
Positive differences below mean that memory-off is worse. NLL units are nats/token.

| Model | Normal | Memory off | Off − normal | 95% CI | Off − Base | 95% CI |
|---|---:|---:|---:|---|---:|---|
| Contextual | 3.213114 | 3.216956 | +0.003841 | [+0.003776, +0.003907] | +0.002173 | [+0.002120, +0.002225] |
| Isolated | 3.214406 | 3.215295 | +0.000888 | [+0.000858, +0.000919] | +0.000511 | [+0.000485, +0.000537] |
| Grad | 3.208273 | 3.217307 | +0.009034 | [+0.008841, +0.009239] | +0.002524 | [+0.002443, +0.002608] |

The intervals use 10,000 paired source-document (`doc_id`) bootstrap replicates,
seed 20260913, conditional on the fixed seed-17 backbone. They are not uncertainty
across independently trained models. All overall intervals above are positive.

## Hit/miss populations

All arms retain identical original target labels in both passes. There are
19,972,074 scored targets: 12,942,338 hits and 7,029,736 misses; 7,001,813 misses
are eligible-but-unselected keys. A miss can change when memory is disabled
because earlier hit positions influence its causal context.

| Model | Population | Normal NLL | Memory-off NLL | Base NLL | Off − normal | 95% CI |
|---|---|---:|---:|---:|---:|---|
| Contextual | hit | 3.213921 | 3.218285 | 3.215595 | +0.004364 | [+0.004283, +0.004446] |
| Contextual | miss | 3.211629 | 3.214509 | 3.213289 | +0.002880 | [+0.002803, +0.002955] |
| Contextual | eligible miss | 3.201759 | 3.204650 | 3.203426 | +0.002891 | [+0.002814, +0.002967] |
| Isolated | hit | 3.215242 | 3.216226 | 3.215595 | +0.000984 | [+0.000947, +0.001022] |
| Isolated | miss | 3.212868 | 3.213579 | 3.213289 | +0.000712 | [+0.000672, +0.000751] |
| Isolated | eligible miss | 3.203006 | 3.203721 | 3.203426 | +0.000714 | [+0.000675, +0.000754] |
| Grad | hit | 3.207820 | 3.219203 | 3.215595 | +0.011383 | [+0.011128, +0.011652] |
| Grad | miss | 3.209106 | 3.213815 | 3.213289 | +0.004709 | [+0.004594, +0.004826] |
| Grad | eligible miss | 3.199229 | 3.203957 | 3.203426 | +0.004728 | [+0.004612, +0.004846] |

All population-specific `off − Base` effects and intervals are also retained
in the machine-readable summary; this table focuses on the direct off/on test.

## A2: Actual contribution magnitude

FP32 L2 norms at the existing block-2 read point, before residual addition.
Exactly 12,942,338 original hit targets contribute to each distribution.
The ratio is computed **per token**, not as a ratio of aggregate means.
No denominator was below the 1e-12 safety clamp.

| Model | Quantity | Mean | Median | p10 | p90 |
|---|---|---:|---:|---:|---:|
| Contextual | Contribution L2 | 1.6435 | 0.8906 | 0.1807 | 3.4040 |
| Contextual | Pre-memory hidden L2 | 27.4765 | 26.4712 | 22.2474 | 33.9706 |
| Contextual | Contribution/hidden (%) | 5.8189 | 3.2269 | 0.7140 | 12.1513 |
| Isolated | Contribution L2 | 0.5842 | 0.2594 | 0.0714 | 1.2262 |
| Isolated | Pre-memory hidden L2 | 27.5783 | 26.5592 | 22.3340 | 34.1131 |
| Isolated | Contribution/hidden (%) | 2.0133 | 0.9534 | 0.2807 | 4.2231 |
| Grad | Contribution L2 | 2.4474 | 1.8475 | 0.2817 | 5.0500 |
| Grad | Pre-memory hidden L2 | 27.9119 | 26.8570 | 22.6822 | 34.5147 |
| Grad | Contribution/hidden (%) | 8.6824 | 6.5677 | 1.0710 | 18.0132 |

Mean norm ratios are approximately **5.82% Contextual, 2.01% Isolated, and
8.68% Grad**. These describe the injected vector magnitude, not a percentage
of accuracy explained or causal attribution. Gate values alone do not capture
this magnitude because the value projection also matters.

## Correctness and execution evidence

- 57 CPU tests passed, including model-hook invariance and mock-only safe
  handoff/failure tests. The dev-only Dropbox helper was supplied on PYTHONPATH
  for the existing local-monitor tests; no network access occurred in tests.
- All three real-checkpoint, two-batch B200 smoke tests passed before full runs.
- Full evaluation covered exactly 20,000,000 input tokens and 27,926 segments
  per arm. Every segment/population normal loss sum matched its original
  evaluation **exactly**, not merely within the acceptance tolerance.
- No checkpoint, training weight, source evaluation, data cache, or frozen
  `ccm/` file was modified. The original code hash remained
  `055f0518853e76487a4e2f31b81f1441113d4fceab322b5e211359a7d58f2fa9`.
- GPUs 0–2 ran independent diagnostic jobs; GPUs 3–7 ran one original burn
  group. The three full children exited zero at 19:04:23 UTC, after about
  7m02s dispatcher wall time (each evaluation body about 404–405 seconds).
- Only verified burn-worker pidfds were signaled, after the old observer
  voluntarily exited. Both initial 30s/free checks and final free checks passed.
- The all-eight original `/tmp/llm_pretrain_burn.py` group was restored before
  CPU bootstrap. Its log contains eight `nranks 8 ... Init COMPLETE` entries
  for the same NCCL communicator. Completion was recorded at **19:07:25 UTC**;
  heartbeat **19:08:26 UTC** confirms eight distinct workers, one per GPU,
  each at **98% utilization**. This is the original small-memory burn, not the
  enhanced 85%-memory resource copy.

Memory-off still computes the reader before zeroing its contribution, so these
runs are **not** a reader-free speed benchmark. A3 frequency × variance and
seed-29 replication remain separate future work.

## Artifacts

Canonical local result root:
`outputs/memory_diagnostics_20260914_a01/`.

- `full/summary.json`: all NLLs and 24 paired effect intervals.
- `full/{contextual,isolated,grad}/diagnostics.json`: norm distributions,
  source identities, exact-replay checks, and four per-arm result-file hashes.
- Each arm's `normal/` and `memory_off/`: metrics and full paired segment records.
- `complete.json`, `status.json`, and `ccm_memory_diag_20260914_a01_b2.log`:
  completion and original burn evidence.

Implementation/launch commit `f8aa4a4`; read-only exports `933bd16`,
`333e104`, `47c264e`. Diagnostic scripts are also present in the canonical
project's `scripts/` directory. See [the diagnostic procedure](MEMORY_DIAGNOSTICS.md).

Summary SHA256:
`9c2e9f092d3cc9535e30066363d0fd1fd6f089e8698f985513c7784497e5bbaa`.
