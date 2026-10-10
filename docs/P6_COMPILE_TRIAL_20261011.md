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
  eval_fa4 packages, source checkpoint SHA and known burn identities. Reuse
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
