# P6-iso function compilation trial

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
