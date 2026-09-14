# Memory diagnostics A1–A3 — seed 17

Completed and verified on 2026-09-14. These are **post-hoc A1/A2/A3 diagnostics**
from `ccm_next_steps_after_seed17.md`, not new training or primary preregistered
comparisons. A1/A2 use three models; A3 uses all five Stage-2 models, each at step 3815.
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
runs are **not** a reader-free speed benchmark. A3 is reported below;
seed-29 memory replication remains separate future work.

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

## A3: Joint frequency × within-key variance — completed 2026-09-14

A3 evaluates all five seed-17 Stage-2 step-3815 checkpoints with their normal
memory behavior: Contextual, Isolated, Shuffled, Grad and Base. Each replays
the same full 20M-input-token D_dev. The grid uses only the **12,942,338 hit
targets**, not all targets, with identical bins and target assignments for
every model.

### Bin definition

Frequency is the key's D_compile count; variance is its compiled Contextual
within-key residual variance. Both use unweighted quintiles over all 262,144
selected keys, computed separately, then crossed into 25 cells. The variance
cutoffs are global, not recomputed inside each frequency band. Equal values
stay together (`searchsorted(..., side="right")`); no test-loss-driven grouping.

- Frequency cutoffs: 502, 663, 960, 1776.4 (last value rounded).
  Integer-count bands: <502; 502–662; 663–959; 960–1776; ≥1777.
- Variance cutoffs, rounded: 24.393661, 30.446532, 35.023790, 40.272482.
- Rows/columns 1→5 mean low→high. NLLs and differences are token-weighted
  within cells, even though the cutoffs are unweighted across selected keys.
- All 25 cells are populated. 261,835 selected keys appear among scored hits.
  The highest frequency band accounts for **75.02%** of all hit targets;
  cells range from 83,549 to 3,116,758 targets.

### All 25 cell results

Differences are **Contextual − comparator**, in nats/hit-target.
Negative favors Contextual. These are descriptive point estimates, not
25 independent significance tests.

| Frequency | Variance | Hit targets | − Base | − Isolated | − Shuffled | − Grad |
|---|---|---:|---:|---:|---:|---:|
| 1 | 1 | 105,520 | -0.003233 | -0.002165 | -0.003321 | +0.012202 |
| 1 | 2 | 96,344 | -0.003968 | -0.003200 | -0.004010 | +0.014069 |
| 1 | 3 | 88,652 | -0.003472 | -0.002738 | -0.003464 | +0.014347 |
| 1 | 4 | 83,549 | -0.003456 | -0.002924 | -0.003384 | +0.014410 |
| 1 | 5 | 88,902 | -0.003820 | -0.002884 | -0.003673 | +0.017408 |
| 2 | 1 | 134,049 | -0.002309 | -0.001707 | -0.002340 | +0.011159 |
| 2 | 2 | 124,435 | -0.002742 | -0.002024 | -0.002751 | +0.012770 |
| 2 | 3 | 117,533 | -0.003256 | -0.002550 | -0.003252 | +0.011928 |
| 2 | 4 | 109,759 | -0.002908 | -0.002156 | -0.002915 | +0.014069 |
| 2 | 5 | 115,398 | -0.004310 | -0.003597 | -0.004333 | +0.014769 |
| 3 | 1 | 174,020 | -0.002510 | -0.001625 | -0.002484 | +0.008546 |
| 3 | 2 | 167,774 | -0.002778 | -0.002309 | -0.002809 | +0.010387 |
| 3 | 3 | 166,308 | -0.002787 | -0.002191 | -0.002825 | +0.010932 |
| 3 | 4 | 162,504 | -0.002644 | -0.002028 | -0.002717 | +0.011943 |
| 3 | 5 | 158,167 | -0.003073 | -0.002330 | -0.003075 | +0.014874 |
| 4 | 1 | 251,154 | -0.001781 | -0.001355 | -0.001791 | +0.006673 |
| 4 | 2 | 268,961 | -0.001883 | -0.001329 | -0.001835 | +0.008905 |
| 4 | 3 | 276,947 | -0.002156 | -0.001616 | -0.002148 | +0.009258 |
| 4 | 4 | 279,443 | -0.001991 | -0.001501 | -0.002060 | +0.009295 |
| 4 | 5 | 263,826 | -0.002831 | -0.002287 | -0.002850 | +0.010854 |
| 5 | 1 | 945,921 | -0.000811 | -0.000568 | -0.000814 | +0.002951 |
| 5 | 2 | 1,319,766 | -0.001106 | -0.000867 | -0.001115 | +0.003702 |
| 5 | 3 | 1,763,043 | -0.001052 | -0.000844 | -0.001012 | +0.004234 |
| 5 | 4 | 2,563,605 | -0.001303 | -0.001063 | -0.001314 | +0.004482 |
| 5 | 5 | 3,116,758 | -0.001795 | -0.001473 | -0.001779 | +0.005290 |

### Interpretation

1. **Contextual beats Base, Isolated and Shuffled in all 25 cells**, but loses
   to Grad in all 25. Its advantage over the frozen-table controls is not
   confined to one rare cell. The grid does not overturn Grad's superiority.
2. **The high-variance pattern persists within coarse frequency bands.**
   V5 has a larger Contextual improvement than V1 against each of Base,
   Isolated and Shuffled in every frequency band. V5 is the largest improvement
   in four of five frequency bands; F1 is an exception (V2 is strongest).
   Thus the marginal pattern is not explained solely by different proportions
   of these five frequency bands. It is not a monotonic variance curve.
3. **Lower-frequency selected keys generally show larger per-hit gains.**
   For example, Contextual − Isolated is −0.002884 at F1/V5 versus −0.001473
   at F5/V5. These are low-frequency *selected* keys, not unselected rare keys.
   Individual cells do not show a strictly monotonic frequency relationship.
4. **No causal variance conclusion follows.** Coarse bins leave residual
   frequency and difficulty differences. No per-cell confidence intervals,
   multiple-testing claims, or exploratory per-key regression were fitted.
   Per-key sums/counts and per-document segment records are saved for a later
   adjusted analysis. This remains post-hoc D_dev evidence from one backbone,
   not independent-seed replication or locked-validation evidence.

Bottom line: A3 strengthens the descriptive evidence that Contextual's small
advantage over the fixed-table controls is broad, including high-variance keys
within frequency bands. It does **not** establish that variance causes the gain,
that deep computation is necessary, or that Contextual beats trainable Grad.

### Verification, runtime and artifacts

- 62 CPU tests passed before launch, including new binning, repeated-key
  accumulation, masking, and all-five-child handoff tests.
- All five real-checkpoint two-batch smoke tests passed before full evaluation.
- Every full replay covered 27,926 identical segments and exactly 20M input
  tokens per arm. Every normal segment/population loss matched its original
  evaluation exactly (maximum discrepancy **0** in all five arms).
- Per-key and per-segment cell counts/sums agree. Both grid marginals reproduce
  the original frequency/variance deciles merged in adjacent pairs. Hashes,
  checkpoint identities, ordered segment identities and all cell contrasts
  were also checked locally against the original results.
- Five independent GPU jobs ran together on GPUs 0–4 while the original burn
  group used GPUs 5–7. Full-panel wall time: **4m11s**, 19:47:00–19:51:11 UTC;
  per-arm evaluation bodies: 230–234 seconds. No training/core changes.
- Original all-eight burns were restored before CPU summary/packaging.
  Completion: **19:53:03 UTC**. Latest exported heartbeat: **20:04:12 UTC**;
  workers 65728–65735, one per GPU, each at 98% utilization and 2510 MiB used.
  All eight ranks initialized the same NCCL communicator
  `0x6df2f2ed4e3fbc36` using original `/tmp/llm_pretrain_burn.py`.
- All three archive-part hashes, the archive hash and all **13 result-file
  hashes** passed local verification. The initial folder export was skipped;
  exact-file exports successfully retrieved the three size-bounded parts.
  This was a retrieval issue, not an evaluation failure.

Local root: `outputs/a3_results_20260914_a01/`.

- `results/full/summary.json`: all five NLLs and four contrasts in every cell,
  plus selected/observed key counts.
- `results/definition/bins.{json,npz}`: exact cutoffs and per-key cell mapping.
- `results/full/{arm}/{diagnostics.json,statistics.npz}`: replay/identity checks,
  FP64 per-key and per-segment cell loss sums, INT64 counts, segment/document IDs.
- `results/VERIFIED.json`: local verification record; `download/`: manifest
  and three verified archive parts. No model weights or training data exported.
- `first_export/`, `final_export/`, `latest_burn_status.json`: smoke, handoff,
  completion and burn evidence.

Remote root: `/mnt/local/_outputs/deep-llms_th2/ccm_a3_seed17_20260914_a01`.
Launch commit `0ae7e33`; read-only exports `e01167d`, `0b6ba32`, `842181c`.
Canonical verifier: `scripts/verify_a3_results.py`.

A3 summary SHA256:
`465ccdc5affe80b2de64b03b06f14ecf6e3033a04b32af7d47422d19a906a53b`.
