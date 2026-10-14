# P7-simple-targetgrad throughput study

Scope: optimize the existing 12-layer, lambda=0.1 arm without changing its
architecture, objective, gradient routing, data order or effective batch size.
No new long scientific training. Production remains unchanged pending results.

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
the deliberately incorrect detached-LSE gradient. All three tests passed.

CUDA acceptance uses the existing trained step-2,500 checkpoint, deterministic
FA4 backward, short/checkpointed and full 2,048-token/micro16 cases. Eager/eager
controls run alongside the three candidates. Backbone and proxy gradients
are checked separately using the existing FA4 model thresholds: loss absolute
error <0.01, output relative L2 <2%, gradient relative L2 <3%. Cosine-only must
retain exact forward outputs and normalization buffers. These are numerical
equivalence checks, not a claim of bitwise training equivalence.

Accepted candidates then run disposable 50-step jobs through the real
eight-GPU HF Trainer: micro16, accumulation4, sequence2048, unchanged schedule,
optimizer, seed42 and dataset. Timing uses the slowest rank's start-to-start
interval for updates20–49, excluding profiling, compilation, saves and eval.
Eager controls bracket the candidates. Per-rank sample hashes check data order
at both ends; compiler counters must remain stable during measurement.

Each supervised stage copies/verifies Accelerate, reclaims only identified
burn workers, verifies all GPUs free, then restores/verifies communicating
burns after success or failure. Existing training checkpoints are read-only.

Results: pending CUDA gates and complete-step timing. Do not enable these
experimental switches in a production experiment on the basis of CPU tests.
