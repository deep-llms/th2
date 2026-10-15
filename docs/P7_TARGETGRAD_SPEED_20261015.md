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
restoration at 01:21:18 UTC. The [final normal-CLI repeat](results/p7_entries_integrated_20261015.json)
completed in `q359-p7-entries-integrated-20261015-a01`, after a verified burn
handoff and Accelerate/free-GPU checks:

| Arm | Seconds/update | Peak allocated GPU memory |
|---|---:|---:|
| A | 1.049507 | 59.67 GiB |
| P7, all three production optimizations | 1.095953 | 61.28 GiB |

Remaining overhead over A is **4.43%**, versus 5.90% before this follow-up.
All-rank endpoint data hashes agree and compiler counters remain stable.
The prototype projection result (1.096631 s) reproduces through the real CLI
(1.095953 s). These are complete steady-state update intervals in disposable
50-step runs, not total wall time including compilation, save and evaluation.
Compared with the original P7's earlier 1.169-second controls, total throughput
improvement is approximately 6.7%; that cumulative estimate spans the two
study rounds, while the additional 1.60% uses this round's bracketing controls.

Use the ordinary recipe plus:

```text
--proxy_compile_auxiliary true --proxy_lse_attention true --proxy_compile_entries true
```

No long training is queued. All four final stages succeeded; complete marker
exists. Supervisor exited zero and restored/verified burns at 01:29:58 UTC.
Read-only verification at 01:30:13 UTC confirms eight live workers matching
recorded ownership, approved parents, advancing collective progress, log age
1.05 seconds and released GPU guard. Receipt:
`temp/remote_logs/p7-entries-completion-20261015-a01.log`, SHA256
`7209fb90a44c8f49dd6fdcc00fb333207920c20fe6026262f9aa4d812a5581ff`.
The collector's early Dropbox snapshot was refreshed with `#2 +a`; an unchanged
snapshot was not evidence that training was still running. No relaunch occurred.
Code integration: `46adec6`. `commands.sh` returned to inactive `#0`.

## Custom-kernel follow-up (in progress)

Current reference is the validated compiled-cosine/LSE/compiled-entries path.
Three experimental candidates are being checked:

- `head_batch`: place native and proxy heads into one FA4 call (20 Q heads,
  10 KV heads, unchanged GQA ratio), split the outputs, then perform the same
  live-LSE joint-softmax merge. Heads remain independent inside attention.
- `target_kernel`: a custom Triton kernel sums the original four FP32 block
  contributions, subtracts the frozen channel mean, scales/clips the target,
  and computes moment/clipping partials. A second reduction combines moments.
  Backward applies the pinned torch clamp derivative and inverse scale to all
  four operands. Shared FP32 contributions retain overlapping-window gradient
  accumulation before the upstream BF16 cast. Original cosine loss unchanged.
- `kernel_both`: both candidates together.

The custom autograd path supports first-order training only. It is CUDA-only,
requires contiguous FP32 target contributions, and is not enabled by ordinary
training. Statistics/bootstrap-only passes keep the existing implementation.
Moment reduction order can change rounding; this is checked separately from
LM forward equivalence. No approximate gradient, detached LSE, changed target,
or new parameter is introduced.

Eight CPU tests passed in 23.170 seconds, including reference implementations
of all candidates, isolated LM/auxiliary routes, optimizer/statistics updates,
checkpointed execution and existing production save/resume. These CPU checks
do not execute the Triton kernel. CUDA must additionally pass its independent
primitive oracle (irregular dimensions, clamp boundaries and overlapping
windows), followed by the trained-checkpoint full-model gates. Only accepted
candidates enter 50-step eight-GPU Trainer timings with bracketing controls.
Launch `77bf1e3`, root `q359-p7-kernel-trial-20261015-a01`.

The first CUDA primitive gate correctly rejected an endpoint-gradient mismatch
before model testing or benchmarks: pinned torch2.14.1 gives zero derivative at
exact clamp boundaries, whereas the initial kernel used inclusive comparisons.
Local torch2.14.0 reproduced it. Fixed to strict comparisons, strengthened both
endpoint tests, and kept all tolerances unchanged. Failed a01 artifacts remain;
retry uses a fresh a02 root after verified burn restoration.

The corrected CUDA primitive oracle passed all 12 cases on every rank, and all
three full-model candidates passed both short/checkpointed and full2048/micro16
gates ([raw report](results/p7_kernel_numerical_20261015.json)). The target-only
kernel matched loss, hidden states, sampled logits and **all parameter gradients
exactly** against compiled entries in these checks. Updated mean/variance buffers
had relative L2 below 7.4e-9; moment reduction order still prevents a bitwise
long-run equivalence claim. Head batching changed overall gradients by 1.073%
relative L2 at full shape, within the unchanged 3% limit.

Preliminary full-update medians: first control1.098117s, head batching1.120300s,
target kernel1.087154s, combined1.109421s. Head batching is slower and will not
be promoted. The target-only kernel is about1% faster; final bracketing control
and integrated production verification remain pending. Nine local tests passed
35.978s, including the standalone Triton interpreter regression and actual
Trainer save/resume with a reference implementation of the opt-in target path.


### Completed custom-kernel timing

[All five trials](results/p7_kernel_trial_20261015.json) passed, with identical
per-rank endpoint data hashes and stable compiler counters in updates20–49.

| Candidate | Seconds/update | Throughput change vs bracketing controls |
|---|---:|---:|
| Current optimized control, before | 1.098117 | — |
| Head-batched FA4 | 1.120300 | -2.06% |
| Custom target kernel | 1.087154 | +0.93% |
| Both candidates | 1.109421 | -1.10% |
| Current optimized control, after | 1.096338 | — |

The target kernel saves about10ms/update and0.375GiB peak allocated memory in
these trials. Its gain against the individual controls is0.84–1.01%; controls
drift0.162%. The gain is small, so it remains optional. The integration gate
requires >0.5% gain against each control and <=0.5% control drift. Head batching
is rejected as slower; ordinary attention is unchanged by the new option.

Diagnostic32-row validation LM losses after50steps: controls7.966937/7.966930,
target kernel7.966979. These are small diagnostic evaluations, not a downstream
or long-training equivalence claim. The trial completed successfully and
restored/verified eight burns at02:12:04UTC before the final integration launch.

Use `--proxy_fused_targets true` with the existing three optimization flags.
It defaults false, supports the validated P7-simple-targetgrad arm (A stays
unchanged), and requires the torch2.14 runtime. Normal training requires CUDA;
first-order autograd only. A version guard prevents silently reusing its clamp
subgradient with another torch release. Resume toggles require the existing
`--allow_performance_change_on_resume true` acknowledgement. No parameter/state
names, optimizer grouping, schedule, data packing/order or loss weights change.

A final review added explicit NaN propagation to the Triton min/max operations
and a13th primitive test for nonfinite forward values and their clamp gradients.
The interpreter regression passed12.988s. Final CUDA integration reruns this
updated kernel against the original checkpoint implementation, then runs A/P7
through ordinary CLI flags. Production integration commit:d1b3336; root:
`q359-p7-target-integrated-20261015-a01`. Completion evidence pending.
