# P6-iso function compilation trial

**Completed on eight B200s, 11 October 2026.** Compiling only the existing
auxiliary cosine loss improves measured steady-state throughput by **2.57%**
using all measured update times, or **2.96%** using per-run medians. It saves
about **3.0 GiB** peak allocated memory per GPU. P6-iso remains slower than A.
The broader gate/code-normalization compilation failed the numerical criteria
and was excluded. **Production defaults and existing research runs are unchanged.**

| Version | Mean update (s) | Mean of run medians (s) | Peak allocated/GPU (GiB) |
|---|---:|---:|---:|
| A | 2.05617 | 2.05555 | 103.623 |
| P6-iso, eager | 2.22381 | 2.22229 | 111.620 |
| P6-iso, compiled cosine | 2.16813 | 2.15838 | 108.626 |

Each version ran **100 real Trainer updates twice**, all eight GPUs, in order
A → eager P6 → compiled P6 → compiled P6 → eager P6 → A. Statistics use
steps21–90 of each run (140 updates/version). All measured updates are retained:
the second compiled run had slower updates66 (2.929s) and73 (2.592s), explaining
its mean/median difference. No cause is established for these two pauses.
Mean P6 overhead versus A falls from **8.15% to5.45%**; median overhead falls
from8.11% to5.00%. This closes part of the gap, not all of it.

At identical trained step10000 weights, cosine compilation preserves **exact**
LM outputs, backbone/gate gradients and target-normalization buffers in all four
64/2048-token, checkpointing off/on cases. Maximum predictor-weight gradient
relative L2 difference is **0.04461%** (0.02962% at2048tokens), from floating-point
execution changes; predictor gradients are not bitwise identical. The CPU tests
also cover masked/zero/near-epsilon inputs and detached targets.

After100 fresh updates, 32-row evaluation LM NLL was9.306579/9.306574 for eager
P6 and9.306547/9.306553 for compiled P6. This is a short numerical sanity check,
not a downstream result or proof of identical long-run trajectories. Both runs
use the same original mathematical loss and gradient routing.

Both compiled timing runs captured only two graphs, with no new compiler graphs
during steps20–90. Dataset fingerprints and first/final update hashes (four
microbatches on every rank) match across all six runs. HF Trainer/Accelerate,
FA4, document isolation/reset RoPE/EOS, global batch1048576, micro16/GAS4 and
full28600-step/1430-warmup schedule are unchanged. Benchmarks use backbone/data
seed1042 and the generator's default proxy seed43 consistently across the four
P6 timings; they are fresh disposable runs, not continuations of the original
proxy-seed1043 checkpoint used for the numerical gate. Compiler cache is shared
within the trial; these timings do not measure fully cold compiler startup.

All six fits and the aggregate validator passed. Queue completed23:06:44UTC;
communicating burns restored and verified23:07:55UTC (11October07:07:55SGT),
workers133155–133162. Source checkpoints and environments remain unchanged.
Final local four-test suite passed in8.681s. Additional real train.py CPU wrapper
smokes ran three updates each with matching eager/compiled batches.

Results: [timing summary](../artifacts/p6-compile-20261011/summary.json) and
[full numerical checks](../artifacts/p6-compile-20261011/numerics.json).
SHA256 `ec35fd007ce2012dd0c013023a8870181121863543d8ae0356873349616587e8`,
verified against the runner's artifact record. Successful launch5b03fd6;
root `/mnt/local/_outputs/deep-llms_th2/q359-p6-compile-20261011-a02`.
Evidence: `temp/remote_logs/p6-compile-health-a14.log`; final read-only receipt
`temp/remote_logs/p6-compile-results-a01.log` confirms all eight burns at100%
at23:11:51UTC, advancing collective progress, unchanged source checkpoint SHA,
and the exact seeds/fingerprints above. Numerical artifact SHA256:
`6d24ba13e6c954bdb184425dd9a501bc9215fdad849d1fb1dc8f40f06338dbca`.

**Decision:** retain this as a tested, opt-in experimental wrapper. Do not enable
the broad existing `proxy_compile_estimator` switch: it also compiles functions
that failed this trial. No existing research checkpoint is automatically resumed
with a changed numerical implementation. A normal training option and a targeted
resume check would be needed before deploying cosine compilation to those runs.

## Original protocol and implementation

User requested reducing P6-iso overhead without changing its architecture or
objective. Production model/train.py/defaults remain unchanged. The experimental
wrapper `scripts/p6_compile_trial.py` compiles existing functions, never modules:
`cosine_loss`, and separately `cosine_loss` plus `gated_prediction`. Full-graph,
static-shape Inductor; no custom backward, precision reduction, checkpointing
change, predictor change or target change. The second candidate covers gate/code
normalization, not the entire residual injection chain.

Prefer cosine-only for the timing trial: P6-iso's auxiliary predictor is isolated
from the LM gradient. At identical weights, the LM outputs, LM/gate gradients and
normalization buffers must remain exact. Predictor gradients can differ due to
compiler FP32 reduction order and subsequent BF16 matrix multiplication.
This is a numerical-equivalence experiment, not an exact-resume guarantee.

Local tests: actual CPU Inductor cosine forward/backward including zero and
near-epsilon norms, masked tokens and detached targets; tiny full-model AOT
Autograd comparison with checkpointing on/off; gate failure/restoration tests.
All three passed, 29.649 seconds, sampling_b200; CUDA disabled. The installed
PyTorch emits a local CUDA-driver compatibility warning during CPU compilation;
no local GPU execution or environment changes are performed.

B200 plan, fresh root `q359-p6-compile-20261011-a01`:

- Copy/verify resources Accelerate config and run accelerate env. Verify pinned
  attention_bench packages (original training environment), source checkpoint SHA and known burn identities. Reuse
  existing supervisor for verified burn-worker stopping, all-eight-GPU free
  checks and automatic communicating burn restoration on success/failure.
- Eight numerical cases: two candidate functions selections, checkpointing on/off,
  64/2048 context, two real packed validation rows, exact P6-iso step10000 weights.
  Deterministic FA4 for these comparisons only. Outputs/statistics <=1e-4 relative
  L2 and peak-scaled absolute difference; each gradient <=1% by both measures.
  Cosine-only additionally requires EXACT LM outputs/gradients and buffers.
  Shape/dtype/missing gradient/nonfinite differences fail. Record all differences.
  Source weights remain read-only. Failure of cosine gate stops the timing queue.
- Actual train.py/HF Trainer: 100 updates each, A / eager P6 / cosine-compiled P6,
  then repeat in reverse order. Eight GPUs per run, native FA4, micro16/GAS4,
  context2048, global1,048,576 input tokens, seed1042, original28600 LR schedule
  and1430 warmup, all activation checkpointing off. Fresh disposable outputs;
  final checkpoints retained. Existing full-data HF cache reused, same order.
- Record synchronized existing Trainer update timings; use steps21–90. Verify no
  additional compiler graphs during measured window. Audit first/final batches
  on every rank, training fingerprints, peak memory and compiler counters.
  Small32-row final validation/monitoring bounds overhead; excluded from timings.

No full research continuation, environment reinstall, driver modification,
cache deletion, production compile switch or original-checkpoint write is part
of this trial. Even successful timing does not authorize changing a resumed
research run's numerical execution. Results pending.

## First CUDA results and startup correction

Launch f6c6499 numerical stage passed. Cosine-only: exact LM outputs, LM/gate
gradients, moments and normalization buffers at identical checkpoint weights.
Largest predictor-gradient relative L2 was0.00044610 at64tokens and0.00029616
at2048tokens, with checkpointing on/off. Auxiliary output differences were at
FP32 rounding scale. Both cosine cases at2048 had only statistics different,
relative L2 2.93e-8; LM and auxiliary summed loss were exact there.

Broader cosine+gate compilation was rejected, not benchmarked: hidden relative
L2 0.00363–0.00411, worst parameter-gradient relative L2 0.01837–0.01932.
The existing eager gate/code normalization remains unchanged.

The first A timing job failed before training because eval_fa4 lacked W&B. This
was our launcher environment selection error. No optimization was active in A.
All eight communicating burns were restored and verified (110063–110070).
The launcher now selects the existing original attention_bench environment and
verifies wandb0.30.0 alongside all pinned CUDA/model packages before reclaiming
GPUs. No environment or driver changes. Fresh root q359-p6-compile-20261011-a02
repeats gates and timings; prior outputs remain intact.
Evidence:temp/remote_logs/p6-compile-health-a02.log.

An additional local real train.py wrapper test completed three updates each with
eager and compiled cosine; actual first-update batch hashes matched.
Log:temp/p6-compile-trainer-cpu.log. No GPU use for that check.
