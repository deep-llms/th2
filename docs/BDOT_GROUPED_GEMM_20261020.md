# Grouped expert execution: B200 comparison, 2026-10-20

**Grouped matrix multiplication helps modestly: +5.1% hard-routing throughput
and +2.1% soft-routing throughput.** All numerical, distributed and resume
checks passed. The default remains `reference`; no research pilot was launched.

## Complete training-update timings

Eight B200s, FA4 4.0.0b33, PyTorch 2.14.1, BF16, 12-layer Qwen3 dimensions,
sequence 1,024, microbatch 16 × accumulation 4 × 8 GPUs = 524,288 tokens/update.
Each measurement uses five warmup updates followed by 50 measured updates.
There are two fresh-process repetitions, with reference/grouped order reversed.

| Case | Reference seconds/update, repeats | Grouped seconds/update, repeats | Mean reference → grouped | Throughput change |
|---|---|---|---|---|
| Native Qwen | 0.616282, 0.615541 | — | 0.615911 | — |
| Matched dense D | 0.706881, 0.708396 | 0.713568, 0.712687 | 0.707639 → 0.713127 | −0.8% |
| Mixed hard routing | 0.807108, 0.807473 | 0.769781, 0.767014 | 0.807290 → 0.768397 | **+5.1%** |
| Mixed soft routing | 1.027704, 1.025176 | 1.008153, 1.002237 | 1.026440 → 1.005195 | **+2.1%** |

Throughput change is `mean(reference time) / mean(grouped time) − 1`.
Grouped is faster in both routing repetitions; their observed ranges do not
overlap. These are two short repeats, not statistical confidence intervals.
Hard grouped reaches about 80.2% of native throughput.
Peak allocated memory rises by 0.315 GiB/GPU for hard routing and 0.327 GiB/GPU
for soft routing. D uses one expert per step, where grouping has no batching gain.

Timings include forward/backward, clipping, AdamW, scheduler, routing bookkeeping
and weight stacking through the existing HF Trainer. They exclude preprocessing,
evaluation and checkpoint writes. Inputs are synthetic, with isolated 128-token
documents and reset positions. Routed timings set the router prior to zero and
hold the tested phase fixed to exercise mixed dispatch; they are not corpus
training or convergence results.

## Correctness and interpretation

- CUDA BF16 projections and model output/gradient/optimizer checks passed for
  SDPA/FA4, hard/soft routing, reuse/no-reuse and checkpointed cases.
- All three checkpointed two-rank cases passed: no-reuse, soft and home-only.
- Both eight-rank curriculum A/B resume suites passed. They use a reduced-width
  12-layer fixture, horizon 120, restarts 60→120 and 10→20. This is not the
  separate v1.4 1,000-step-horizon T12 or full-width 500-update T14 requirement.
- Resume checks compare routes, inputs/labels, RNG, losses, weights, optimizer,
  scheduler and balancing state. Passing cross-backend BF16 tolerances does not
  imply bitwise equality or identical long-run convergence between backends.

The 32 rank/restart comparisons have zero loss difference and exact restored
and final model/optimizer/scheduler state. Cross-backend model logits match
exactly in the controlled-route fixtures; maximum whole-gradient relative error
is 0.3244%. Fresh AdamW whole-update difference reaches 3.27% because of
near-zero-gradient sensitivity; with identical existing moments, maximum
parameter difference is 4.03e-7. These are the tested BF16 tolerances, not a
claim of identical training trajectories between reference and grouped.

The implementation in `bdot/grouped.py` batches selected experts' Q/O and
gate/up/down projections. It still sorts and gathers/scatters tokens, stacks
occupied weights and performs host synchronization. The measured gain therefore
does not remove the gap to native Qwen. These timings do not isolate which of
those remaining operations dominates.

Grouped execution is a viable opt-in for fresh routed runs. Keep the chosen
expert backend fixed across checkpoint resume; no silent backend migration or
new environment is needed.

## Evidence and operational status

[Compact result receipt](results/bdot_grouped_comparison_20261020.json).
The complete downloaded summary and operational receipts are in
`temp/grouped_20261020/results/`. Raw returned logs are in
`temp/grouped_20261020/`, notably the a12/a13 status logs.
The immutable remote summary has SHA256
`d6db97599f13fcb78a87f7e0b455b9d4de2e594beecacc9a4fc5ae5854f2bf73`.
Launch commit: `5648b88`; execution root:
`/mnt/local/_outputs/deep-llms_th2/6y3o-dot-grouped-20261020-a01`.

Accelerate's actual cache configuration was copied and verified. Only verified
burn workers were reclaimed; all eight GPUs were confirmed free before tests.
No environment installation, driver change or existing checkpoint/data cleanup
occurred. Tests completed at 14:52:44 UTC. The supervisor restored and verified
communicating burns at 14:53:50 UTC. An independent 14:55:56 UTC observation
confirmed all eight workers, released guard and collective cycles advancing
160→170. Final retrieval was delayed but succeeded; all eight requested files
are local, and the summary matches the remote SHA256 and completion manifest.
commands.sh returns to #0 with publication of this report.
