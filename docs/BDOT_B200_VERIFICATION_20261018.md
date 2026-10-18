# B-DOT verification on eight B200 GPUs — 2026-10-18

All 16 required checks passed, including eight-GPU throughput. A separate
near-tie routing diagnostic changed paths between backends, as detailed below.
Automatic burn restoration and a subsequent live check both passed. No research
pilot was launched.

## Runtime and launch

q359, eight NVIDIA B200 GPUs. Reused the existing `eval_fa4` interpreter with
PyTorch 2.14.1/CUDA 13.0, Transformers 5.9.0, Accelerate 1.13.0, Datasets 4.8.5,
Tokenizers 0.22.2, Safetensors 0.8.0, FA4 4.0.0b33, CUTLASS 4.8.0 and quack 0.6.5.
No environment reinstall or driver changes. This environment has lm-eval 0.4.10;
the project's downstream-evaluation pin is 0.4.13 and was not validated here.

Published DOT to `deep-llms/th2`; launch commit `c3ce8d1`. The eight-GPU resource
config was copied byte-for-byte to the actual Accelerate cache and checked using
the selected interpreter's `accelerate env`: MULTI_GPU, 8 processes, BF16.
The supervisor rechecked exact known burn identities, stopped only their GPU
worker PIDs, waited 30 seconds and verified all eight GPUs free at 19:16:16 UTC.
Automatic burn restoration is part of both successful and failed exits.

Two preliminary attempts did not touch GPU work: the controller rejected a
redundant test assertion about a Hub option, then Python failed to resolve a
helper import. Removed the unnecessary assertion, retained the offline-network
test, and added scripts/__init__.py. Fresh preflight `0f37b2c` then passed.

## Numerical results

- Full-size 12-layer T1, FP32, TF32 disabled: native mapped-model checks pass with
  both math-reference and default SDPA. Maximum gradient errors are 1.24e-7 and
  1.04e-7; maximum parameter differences after AdamW are 6.38e-5 and 4.45e-5,
  below the unchanged 1e-4 threshold. Native-versus-native control also passes.
- Real FA4 forward/backward passes native and routed BF16 checks. With separated
  diagnostic router biases so both soft branches follow the same paths, maximum
  relative logit L2 error is 0.357%, and gradient L2 error is 0.548%. Both hard and
  soft routing, reuse on/off and nonreentrant recomputation are covered.
- Causal/document masking passes; cross-document input gradients are exactly
  zero. With fixed routes, unchanged document logits match bitwise.
- Two real CUDA/NCCL ranks with accumulation 2, unused experts and checkpointing
  pass the global-reference gradient/count/balance/update checks for hard and
  soft routing. Maximum gradient error is 1.34e-7.
- The B-DOT CPU suite passes in the node's actual runtime. Local helper tests
  pass: 8 baseline tests, 7 ownership/handoff tests and 2 reporting tests.

The **natural near-tie soft-routing diagnostic does not preserve identical
paths across backends**: 6/2,048 primary decisions differ (0.293%), with 15 total
primary/secondary branch differences. Relative logit/gradient L2 differences
are 2.79%/2.92%. That case is retained as a failed identical-path diagnostic, not
relabeled as a kernel equivalence pass. The separated-bias fixture is a separate
test configuration, changes no production initialization, and passes for all
four reuse/phase combinations. Keep the backend fixed within an experiment;
these results do not promise identical natural routes or training trajectories
across SDPA and FA4.

## Throughput

Full 12-layer configurations; actual HF Trainer forward/backward,
clipping, AdamW, scheduler and routing-state updates. Eight ranks, microbatch 16,
accumulation 4, length 1024: 524,288 input tokens/update. BF16, checkpointing off,
3 warmup + 10 measured updates, synthetic 128-token document fragments.

| Model / routing | SDPA tokens/s | FA4 tokens/s | SDPA s/update | FA4 s/update | SDPA / FA4 peak GiB* |
|---|---:|---:|---:|---:|---:|
| Native Qwen3 | 808,378 | 854,426 | 0.649 | 0.614 | 51.3 / 50.1 |
| Matched dense D | 708,018 | 750,419 | 0.741 | 0.699 | 53.8 / 52.6 |
| Mixed hard routing | 631,335 | 644,435 | 0.830 | 0.814 | 55.4 / 54.2 |
| Mixed soft routing | 483,153 | 509,955 | 1.085 | 1.028 | 66.9 / 64.7 |

*Peak PyTorch allocated memory on rank 0, not total NVIDIA memory or the
maximum across ranks. No CUDA OOM occurred.

Hard routing reaches **89.2% of D / 78.1% of native** throughput with SDPA
and **85.9% of D / 75.4% of native** with FA4. Both exceed the specified
60%-of-D runtime gate in this setup. Soft routing reaches 68.2% / 68.0% of D.
FA4 improves hard-routing throughput by only about 2.1% over SDPA here.
These are ten measured updates per case, not sustained long-run timings.

Native dense and matched D are reported separately because D itself has
generic-dispatch overhead. Mixed hard/soft cases explicitly
set the diagnostic initial router prior to 0 to exercise non-default routes;
production configs remain unchanged. These timings exclude preprocessing,
checkpoint saving and evaluation and are not convergence evidence.

## Evidence and remaining scope

Remote root:
`/mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03/`.
Local raw receipts: `/disk/thuat/DOT/temp/bdot_b200_verification_20261018/`.
Scripts: `scripts/bdot_b200_{preflight,launch,verify}.py`, the existing numerical
probes and full-update benchmark, and reused `scripts/train_then_burn.py`.

The launch is bounded to one hour with failure propagation and burn restoration.
The complete summary reports `passed`. Machine-readable results:
[summary](results/bdot_b200_20261018.json) and
[final handoff](results/bdot_b200_handoff_20261018.json).

The supervisor finished at 19:27:32 UTC. A separate read-only inspection at
19:30:53 UTC verified the guard released, one known burn worker on each GPU,
all eight GPUs at 100% utilization, and collective cycles advancing from 230
to 240 (255.38 to 266.48 GiB cumulative payload). No training workers remained.
The final inspection also checks benchmark source hashes and finite printed
training losses for every backend/model case. Its first attempt needed a parser
fix for Trainer's string-formatted loss values; no GPU tests were restarted.
A subsequent [eight-GPU FA4 resume test](BDOT_FA4_RESUME_20261018.md) passed
200-step checkpoint/route/RNG/optimizer comparisons exactly on a reduced-width
12-layer fixture. Actual corpus selection and the full Gate 0a route-study
exporter remain before a pilot.
