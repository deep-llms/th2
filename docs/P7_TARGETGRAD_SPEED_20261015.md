# P7-simple-targetgrad throughput study

Scope: optimize the existing 12-layer, lambda=0.1 arm without changing its
architecture, objective, gradient routing, data order or effective batch size.
No new long scientific training. Default behavior is unchanged; validated optimizations are opt-in.

Previous 2,500-step training runtimes were about 45.3 minutes for A and 50.4
minutes for P7-simple-targetgrad: approximately 11% overhead in both seeds.
These include training-loop overhead and are not isolated attention timings.

## Candidates

`scripts/p7_speed_trial.py` provides process-local experimental switches:

- **cosine:** compile the existing cosine loss with Inductor, retaining live
  target gradients. No detached-target approximation or new loss formula.
- **lse:** compute native attention on all heads and proxy attention on the
  four selected heads. Merge selected outputs with
  `sigmoid(LSE_native - LSE_proxy)`, keeping both LSE gradients live. This is
  the same joint softmax in real arithmetic, without duplicated queries.
  FP32 merging and BF16 attention outputs can change rounding.
- **both:** combine those two independent changes.

Installed FA4 4.0.0b33 exposes `return_lse=True` and passes `dlse` into its
ordinary varlen backward. This is audited from the actual B200 installation,
not assumed from another wheel. No FA version, environment or driver changes.

## Acceptance and measurement

CPU tests compare an independent joint-attention oracle, all five Q/K/V/proxy
input gradients, document isolation, causality, full-model LM and auxiliary
gradient routes, normalization buffers and an AdamW update. They also reject
the deliberately incorrect detached-LSE gradient. All four tests passed, including zero/tiny norms and fully masked rows.

CUDA acceptance uses the existing trained step-2,500 checkpoint, deterministic
FA4 backward, short/checkpointed and full 2,048-token/micro-batch 16 cases. Eager/eager
controls run alongside the three candidates. Backbone and proxy gradients
are checked separately using the existing FA4 model thresholds: loss absolute
error <0.01, output relative L2 <2%, gradient relative L2 <3%. Cosine-only must
retain exact forward outputs and normalization buffers. These are numerical
equivalence checks, not a claim of bitwise training equivalence.

Accepted candidates then run disposable 50-step jobs through the real
eight-GPU HF Trainer: micro-batch 16, accumulation4, sequence 2048, unchanged schedule,
optimizer, seed42 and dataset. Timing uses the slowest rank's start-to-start
interval for updates 20–49, excluding profiling, compilation, saves and eval.
Eager controls bracket the candidates. Per-rank sample hashes check data order
at both ends; compiler counters must remain stable during measurement.

Each supervised stage copies/verifies Accelerate, reclaims only identified
burn workers, verifies all GPUs free, then restores/verifies communicating
burns after success or failure. Existing training checkpoints are read-only.

## Results so far

The original A and P7 50-step profiles completed successfully. Their final
summary script initially misclassified intentional force-stopping as an error;
this reporting check was fixed. The original failure history is retained, and
every recorded result/timing artifact hash was reverified before recomputing
[the summary](results/p7_speed_profile_20261015.json) in the new trial root.
Burns were restored/verified after that stage.

All-rank median intervals: A **1.049218 s**, P7 **1.170189 s** (**11.53%**
overhead). P7 cosine forward/backward sums to **36.56 ms** of GPU kernel time
per update. Kernel sums are explanatory, not independently additive wall time.

[CUDA acceptance](results/p7_speed_numerical_20261015.json) passed for all three
candidates in both tested shapes. Eager/eager was exact. At sequence 2048 and
micro-batch 16, cosine-only retained exact forward outputs and buffers, with 0.370%
overall gradient relative L2 difference. LSE/both had about 1.260% gradient
relative L2 difference, 0.623% hidden relative L2, 0.246% logits relative L2,
and 0.000175 absolute total-loss difference. Predictor gradients were checked
separately and passed. Source checkpoint was unchanged.

## Complete candidate timing

All five jobs completed, all per-rank data hashes agreed, and no recompilation
occurred in updates 20–49. [Full timing report](results/p7_speed_trial_20261015.json).

| Implementation | Median seconds/update | Throughput gain vs eager controls |
|---|---:|---:|
| Eager, before | 1.169309 | — |
| Compiled cosine | 1.137384 | 2.82% |
| LSE attention | 1.146433 | 2.01% |
| Both | 1.112883 | 5.08% |
| Eager, after | 1.169598 | — |

The eager controls differ by 0.025%. The combined result reduces elapsed update
time by 4.84% (throughput increases 5.08%). Relative to the initial A measurement,
P7 overhead falls from 11.5% to about 6.1%. These are steady-state short-run
measurements, excluding compilation and checkpoint/evaluation overhead, not
measured completion times for a full 2,500-step experiment.

## Normal training interface

Use the existing P7-simple-targetgrad recipe and add:

```text
--proxy_compile_auxiliary true --proxy_lse_attention true
```

The same keys work in the JSON recipe passed to the existing sequential queue.
Both default to false. They are validated for P7-simple-targetgrad; A accepts
these settings without changing its computation. Other proxy arms fail closed
until validated. Dense SDPA retains its original attention path; the LSE option
is used only with FA4. Cosine compilation does not change the LM forward path.

Production changes are in `deep_kv/fa4.py`, `deep_kv/proxy_memory.py`,
`deep_kv/proxy.py` and the CLI/configuration guards in `train.py` and
`deep_kv/report.py`. Experimental benchmarking remains in `scripts/`.
No parameter names/shapes, optimizer grouping, data/packing, schedule or loss
weights change. Both flags are saved in the recipe; enabling/disabling them
on resume requires `--allow_performance_change_on_resume true`. Existing
checkpoints with absent flags mean false and retain their original behavior.
Normal optimizer/scheduler/RNG/data restoration is unchanged. Floating-point
rounding differs, so identical future weights are not promised.

Six local tests passed, including exact prototype/production outputs and
parameter gradients, actual train.py save/resume, and checkpoint state names.
An additional combined-flags real-entry resume and legacy FA4 baseline test
passed. The [final integrated eight-GPU pair](results/p7_speed_integrated_20261015.json)
also passed through normal CLI flags, without runtime prototype patches:

| Arm | Median seconds/update | Peak allocated GPU memory |
|---|---:|---:|
| A, unchanged | 1.049506 | 59.67 GiB |
| P7, both optimizations | 1.111457 | 61.41 GiB |

P7's remaining update-time overhead over A is **5.90%**. Data hashes at both
ends agree across arms/ranks, and graph counters are stable during timing.
Both use the original train fingerprint `6e4708f1e818fb44`; validation uses
32 rows for these disposable tests. All original research checkpoints remain
untouched. The default settings for ordinary launches are still disabled.
All integration jobs and the summary succeeded; complete marker exists.
The supervisor restored and verified eight communicating burns at
2026-10-15 00:17:24 UTC. Read-only verification at 00:19:54 UTC confirmed
all eight worker identities/approved parents, advancing collectives and a
released GPU guard. Final receipt:
`temp/remote_logs/p7-speed-final-burn-20261015-a01.log`, SHA256
`6281217fe27e5412e92b3060708560973f2347c3c6acfbfebb41825c4ebb6483`.

Code integration: `a1a40f0`. Final run root:
`/mnt/local/_outputs/deep-llms_th2/q359-p7-speed-integrated-20261015-a01`.
No long training is queued. `commands.sh` is returned to inactive mode.

## Follow-up fusion experiment

The next bounded trial compares the validated combined optimization against:

- `fused_aux`: defer target standardization/clipping until the compiled cosine
  calculation. Preserve the live target gradient, frozen normalization scales,
  original running-moment updates and clipped-channel diagnostics.
- `entries`: compile the original proxy normalization, projections and RoPE.
- `kv`: compile those entries with one concatenated K/V GEMM; the original
  separate Parameters, initialization, optimizer groups and state names stay
  unchanged. There is no persistent concatenated weight cache.

These remain process-local experiments in `scripts/p7_fusion_trial.py`. CUDA
acceptance and full-update timing must pass before recommending any production
change. The reference now includes BOTH earlier optimizations. No additional
throughput gain is established yet. Output: `q359-p7-fusion-trial-20261015-a01`.

The [completed follow-up timing](results/p7_fusion_trial_20261015.json) gives:

| Extra change over the previously optimized P7 | Seconds/update | Throughput gain |
|---|---:|---:|
| Control before | 1.115236 | — |
| Fused target normalization + cosine | 1.109494 | 0.42% |
| Compile existing proxy entries | 1.096631 | **1.60%** |
| Compile packed K/V entries | 1.097682 | 1.50% |
| Control after | 1.113034 | — |

Reference drift is 0.20%; gains use the mean of the two reference medians.
All five 50-step jobs succeeded, endpoint data hashes match across all ranks,
and compiler counters are stable throughout the measured interval. Final
32-row LM losses are 7.9669915 / 7.9669381 / 7.9669659 / 7.9669596 / 7.9669719
in the table order. This remains a short-run numerical check, not full-length
PPL/downstream equivalence evidence.

Choose **compiling the existing entries method**. Packed K/V gives no extra
measured benefit, and the 0.42% target-loss fusion gain does not justify its
deferred-target plumbing. Those alternatives remain isolated research scripts,
not production behavior. The new production flag is `--proxy_compile_entries true`,
used alongside `--proxy_compile_auxiliary true --proxy_lse_attention true`.
It defaults false, is restricted to the validated arm (and unchanged A), is
recorded in recipes, and requires explicit performance-change allowance when
toggled on resume. It compiles the existing bound method; it does not wrap the
module or rename/reinitialize Parameters. Legacy absent flags mean false.

[Refinement CUDA tests](results/p7_fusion_numerical_20261015.json) passed at
short/checkpointed and full 2048/micro16 shapes. An additional
[direct comparison against original eager P7](results/p7_entries_original_numerical_20261015.json)
also passed for the accumulated optimizations: at full shape, total-loss
absolute difference 0.000181675, hidden relative L2 0.6238%, logits relative
L2 0.2452%, overall gradient relative L2 1.2629%. Predictor/backbone checks
passed separately; thresholds unchanged. Eager/eager control is exact.

Eight CPU tests passed in 18.618 seconds, including exact prototype/integration
output and gradient comparison, gradient routing, optimizer/state checks, actual
Trainer save/resume, and old-recipe compatibility. One initial combined test
run reused an AOT donated-buffer graph from a single-backward test in a later
retain-graph gradient probe. Isolating compiler caches between tests fixed this
test-process interaction; no production compiler settings were changed.

Fusion trial completed with successful supervisor status and verified burn
restoration at 01:21:18 UTC. Final normal-CLI A/P7 repeat is running in
`q359-p7-entries-integrated-20261015-a01`, after a verified burn handoff and
Accelerate/free-GPU checks. No full scientific training is queued.
