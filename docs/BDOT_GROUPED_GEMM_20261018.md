# Grouped expert GEMM verification — 2026-10-18

Status: CUDA numerical and two-rank distributed checks pass; eight-rank resume
and complete-update throughput results are unverified: the latest returned
remote snapshot is 22:07 UTC, with no newer controller receipt as of 22:20 UTC. No research pilot
was launched. The default expert backend remains `reference`.

## Implementation

`bdot/grouped.py` batches expert Q/O and SwiGLU gate/up/down projections through
PyTorch's native CUDA BF16 `torch.nn.functional.grouped_mm`. This is a grouped
matrix product with one weight matrix per selected token group. It does not
replace the model with a dense product that evaluates every expert for every token.

Set `bdot.expert_backend` to `grouped` in the model configuration; `reference`
retains the original loop. Parameter objects/names, routers, step-owned K/V,
attention, normalization, activation, FiLM, loss and optimizer definitions remain
unchanged. Only occupied experts enter each stack. In particular, unused
parameters keep `grad=None`, preserving AdamW's skip behavior, including its
momentum and decay. Single-expert steps use ordinary linear layers. Multi-expert
grouped execution requires CUDA BF16. Routing still has host synchronization;
this change does not remove all dispatch overhead.

The strict resume recipe records the backend and implementation source. Keep a
run's backend fixed; this work does not authorize silent migration of older
checkpoints. Default experiment configs are unchanged.

## Numerical and optimizer checks

Local validation: 40 B-DOT CPU tests and 49 utility tests pass. CPU grouped tests
substitute only the GEMM primitive and compare hard/soft routing, no-reuse,
checkpoint replay, save/load, gradients and optimizer updates. They do not
validate the CUDA kernel.

On B200, both irregular `[1,17,65]` and regular `[128,256,384]` token groups
match ordinary linear layers exactly for outputs, input gradients and weight
gradients in the projection fixtures. An unoccupied expert retains no gradient.

All eight model fixtures produce identical logits, losses and route choices.
They deliberately separate router margins to avoid conflating kernel error
with discrete decisions at near ties. Checkpointing is enabled in the no-reuse
cases. The table reports relative whole-gradient error, fresh-AdamW whole-update
error, and maximum parameter difference with identical existing moments:

| Attention | Reuse | Routing | Gradient difference | First update difference | Existing-state update max abs |
|---|---|---|---:|---:|---:|
| sdpa | yes | soft | 0.291% | 2.83% | 3.48e-07 |
| sdpa | yes | hard | 0.178% | 2.21% | 3.91e-07 |
| sdpa | no | soft | 0.324% | 3.12% | 4.02e-07 |
| sdpa | no | hard | 0.287% | 3.27% | 3.27e-07 |
| fa4 | yes | soft | 0.275% | 3.24% | 3.43e-07 |
| fa4 | yes | hard | 0.162% | 2.25% | 2.31e-07 |
| fa4 | no | soft | 0.145% | 1.83% | 2e-07 |
| fa4 | no | hard | 0.236% | 2.60% | 3.33e-07 |

These results are numerical agreement within BF16 precision, not bitwise
training equivalence between expert backends. A fresh AdamW first update can
amplify roundoff near zero gradients into an individual +/-learning-rate change.
The maximum coordinate difference is about 0.000600 at LR 0.000300, despite the
small whole-gradient difference. An independent CPU BF16 test of the unchanged
reference backend also fails an overly strict first-step coordinate tolerance
when comparing full batches with microbatch accumulation. We therefore report
fresh-update drift separately and additionally compare optimizers initialized
with identical nonzero moments. That does not establish identical long-run
convergence between backends.

All three two-rank BF16 tests pass with checkpointing: hard no-reuse, soft and
home-only routing. They compare full-batch and accumulated gradients, load
counts, balancing, absent gradients and AdamW updates. Existing moments are
used for the BF16 optimizer comparison; FP32 tests retain the fresh optimizer.

## Eight-rank resume and throughput

Pending. The suite uses actual `train.py`/HF Trainer for A/B curriculum restarts,
including phase transitions and the periodic dual-policy validation. It compares
data, RNG, routes, losses, model state, optimizer, scheduler and cumulative
metrics across uninterrupted and restarted processes on every rank.

The timing suite uses eight B200s, FA4, the full Phase-A 12-layer dimensions and
151,936-token vocabulary, microbatch 16, accumulation 4, sequence length 1,024,
524,288 input tokens/update, isolated 128-token documents and reset positions.
Checkpointing is off; five warmup updates precede 50 measured updates. Timings
include forward/backward, clipping, AdamW, scheduler, routing bookkeeping and
weight stacking. They exclude preprocessing, validation and checkpoint writing.
The hard/soft diagnostic sets router prior to zero to exercise dispatch and
holds curriculum progress fixed; these are synthetic timings, not convergence
results or inference benchmarks.

## Operations and evidence

The existing pinned q359 environment is reused without installation or driver
changes. Accelerate's eight-process BF16 config is copied and checked before
launch. The established supervisor verifies known burns, reclaims only their
worker PIDs, waits and requires free GPUs, and automatically restores/verifies
communicating burns on completion or failure.

Initial attempt `ff5452b` stopped at the overstrict first-AdamW test; burn handoff
was verified at 21:54:37 UTC. Attempt `ffd19ea` passed all CUDA numerical cases
but exposed a test-fixture bug: seeded fused-AdamW step counters need the CUDA
device. Its burn handoff was verified at 22:01:50 UTC. Launch `ec19fe8` fixes that
fixture, explicitly checks the preceding handoff, and starts fresh output a03.
The grouped model implementation is unchanged across these attempts.

Remote root: `/mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a03/`.
Local receipts currently include `temp/grouped_a03_summary1.json`,
`temp/grouped_a02_numerics.log`, `temp/grouped_a03_launch.log`,
`temp/grouped_all_cpu.log` and `temp/grouped_utilities.log`.

At 22:20 UTC the result export (`12881a3`) and read-only status request
(`098061b`) have not returned. The last controller receipt is 22:07:25 UTC;
there is no returned infrastructure error identifying a cause. Final resume,
throughput and burn-restoration status must be retrieved before claiming success
or recommending the grouped backend. Do not launch a duplicate verification.
