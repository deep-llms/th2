# Proxy optimization CUDA validation and timing

Authorized scope: CUDA correctness, 25-step eight-GPU smokes/resume checks,
and before/after throughput. **No 2,500-step research runs.**

Initial output root: `/mnt/local/_outputs/deep-llms_th2/proxy-speed-validation-20261008-a01`.
Machine: `thiennh-p6-tjx3-worker-0`. Read-only inspection at 02:37:27 UTC on
8 October verified eight B200s, only the known communicating burn workers
401980–401987, and the expected attention_bench package versions. Inspection
log: `temp/proxy-validation-inspection-20261008-a01.log`, SHA256
`e87dc71acc1e3a8bb8e642598c0f7ca839bc55c328d5f07f2b58e0c44027025e`.

## Queue

1. Nine numerical gates: P6-iso-sparse, P6-iso-short, P6-iso-weighted,
   P6-iso-layernorm, P7-simple-sparse, P7-simple-short, P6-iso, P7-simple and P7.
   Full-size Qwen, two real packed 2048-token rows, BF16 CUDA, both forced math SDPA and
   FA4. Compare previous/optimized code at the same weights and checkpointing
   setting, separately with checkpointing off/on. Up to eight independent gate processes run concurrently, one per GPU.
   Two-pass bootstrap, hidden
   outputs, losses, every gradient, normalization moments and updated buffers
   must match exactly for math SDPA. FA4 outputs must match exactly; gradients
   use the explicitly revised FP32 bound below. Failures stop the queue.
2. Six new arms each train 25 steps on all eight GPUs. Validate finite metrics,
   backend receipts, schedule, normalization and checkpoint contents. Copy the
   complete step-24 checkpoint into a fresh resume directory, verify all copied
   state-file hashes, resume one update to 25, then compare with uninterrupted
   step 25: model/optimizer within the FP32 bound, normalization/scheduler/RNG
   exactly. Hash all four microbatches on each of eight ranks for update 25;
   require identical input IDs, labels and document metadata after resume.
3. A, P6-iso, P7-simple and P7 each receive previous/optimized 25-step runs,
   alternating pair order. Component profiler captures step 6; unprofiled
   steps 10–25 supply the existing synchronized update timer. Record median,
   all 16 timings and peak allocated/reserved memory. One pair per arm is a
   screening measurement, not a precise estimate of small speed differences.

All training uses the existing train.py/Accelerate recipe: microbatch 16,
accumulation 4, eight GPUs, 1,048,576 input tokens/update, sequence 2048,
document isolation/reset positions, seed 42, schedule 28,600, warmup 1,430,
activation checkpointing off, FA4. Logging stays every ten steps so the new
clipping optimization is exercised. Disposable eval/monitor subsets use 32
rows, and saves occur at 24 plus the forced final step 25.

The frozen reference consists of the only three production modules changed by
the optimization, copied exactly from commit `4f8208b`. Source hashes are checked
before import. `scripts/proxy_speed_validation.py` selects them only in the
disposable test process; ordinary training never imports the reference. Main
training/data code is shared. `scripts/profile_proxy_training.py` reuses the
existing callback timer without extra synchronization on timed updates.

`scripts/launch_proxy_speed_validation.sh` verifies inputs/environment, copies
and verifies the repo Accelerate configuration and runs `accelerate env`.
The established `train_then_burn` supervisor rechecks process identities,
stops only approved burn workers, verifies GPUs free, and restores/validates
communicating burns after success or failure. All roots are fresh; old research
outputs/caches are preserved. Current queue has 38 stages; the first runs nine independent numerical checks
across eight GPUs. Training jobs remain sequential, each using all eight GPUs.

Local preparation: four tests passed in 37.776 s, including actual old/new
train.py wrapper runs and identical saved tiny-model weights. A separate gate
control test checks pairing and failure behavior. Local tests use CPU only.
Remote completion and numerical/performance results remain pending.

## First remote gate: stopped, 8 October 02:57 UTC

The first SDPA/checkpoint-off P6-iso-sparse comparison failed strict gradient
equality. All outputs, losses and normalization statistics were exactly equal;
307 gradient tensors differed, maximum per-tensor relative L2 0.00784866
(0.785%), maximum absolute difference 0.00055997. No smoke or timing jobs ran.
The supervisor restored and verified communicating burns on all eight GPUs
(workers 405378–405385); fresh inspection at 02:59:53 confirmed them.
Evidence: artifacts/proxy-speed-monitor-20261008-a01/, source log SHA256
`deffe5ba5e3cfd40a8aae938be8a4685ac2380a53c295caefb1d4c8a75c285ad`.

This is not yet evidence of an optimization defect: earlier SDPA backend
checks documented non-bitwise repeated backwards. A separate disposable
repeatability diagnostic compares previous/previous and previous/optimized
at identical weights under SDPA, explicitly forced math SDPA and FA4. It
records differences without declaring acceptance or relaxing the study gate.
Root: proxy-speed-repeatability-20261008-a01. No production recipe change.

## Repeatability result and corrected strict gate

The diagnostic completed normally; all eight burns restored/verified again.
At identical weights on P6-iso-sparse:

| Backend | Previous/previous max gradient relative L2 | Previous/optimized | Outputs |
|---|---:|---:|---|
| Default SDPA | 0.00774436 | 0.00781674 | Exactly equal |
| Forced math SDPA | 0 | 0 | Exactly equal |
| Production FA4 | 0 | 0 | Exactly equal |

Thus the default SDPA comparison is limited by observed backward variability.
The fresh full study uses **math SDPA and production FA4 for strict equality**;
no numerical tolerance is added. This backend restriction applies only to the
numerical oracle; all smoke/resume/timing jobs retain production FA4.
This does not claim bitwise repeatability for default SDPA.

Evidence: artifacts/proxy-repeatability-monitor-20261008-a02/, source log SHA256
`3e5da76557bed17c4643c073c63e1a77711782221a6fca00daa285ea530a808c`.
Fresh full-study root: proxy-speed-validation-20261008-a02. Original failure
and diagnostic outputs are preserved.

## FA4 FP32 rounding and final acceptance rule

Full-study a02 stopped at the first FA4/checkpoint-off comparison: only
`backbone.model.layers.25.self_attn.q_norm.weight` differed, absolute maximum
3.63798e-12, relative L2 1.13150e-8. All outputs and math-SDPA gradients
(checkpointing off and on) were exactly equal. Burns restored and verified.
Evidence: artifacts/proxy-speed-monitor-20261008-a03/, source log SHA256
`055f9d49574858a6d77cde6a6ebaa99a5ef524ea961b53631268ece1eb0fde44`.
The earlier FA4 repeat matched bitwise; one repeat cannot establish universal
bitwise reproducibility. This residual is at FP32 rounding scale.

Explicitly revised rule for fresh study **proxy-speed-validation-20261008-a03**:
- Math SDPA remains exact for every output/gradient, with checkpointing off/on.
- FA4 outputs, losses, moments and buffers remain exact. Gradient differences
  must satisfy BOTH relative L2 <= 2^-20 and maximum absolute difference
  <= 2^-20 times that reference tensor's peak magnitude. This is eight FP32
  epsilons (under one part per million), not a BF16-scale tolerance. Shape,
  dtype, missing gradients and nonfinite values remain failures. All observed
  differences are retained in reports even when within the bound.
- Resume uses the same bound for model parameters/optimizer floating state,
  while normalization, scheduler, RNG and update-25 data hashes remain exact.
  Checkpoint-24 copies still require byte-identical hashes before loading.
- Production model code and training recipe remain unchanged. No research runs.

Local gate/bound/queue tests pass. The new data-audit wrapper is also exercised
by a real tiny-model step-24-to-25 resume test before launch.

Revised local validation: nine distinct tests passed across the final targeted
invocations/full-suite coverage, including the real 24→25 resume with identical
per-microbatch data hashes and model weights (45.225 s). The parallel-gate
control deliberately injects a failed arm and confirms that training cannot
proceed. Environment: sampling_b200, Transformers 5.9.0, CPU.

## Nine-arm CUDA result: eight passed, P7-simple-short isolated

Study a03 completed all nine numerical checks in 257 s. Eight passed both
math SDPA and FA4, checkpointing off/on. P7-simple-short passed math SDPA
exactly but failed FA4/checkpoint-off gradients: 134 tensors, maximum relative
L2 0.00494480 (0.494%). All outputs exactly match. No smoke/timing started.
All eight burns restored and verified (workers 409228–409235). Evidence:
artifacts/proxy-speed-monitor-20261008-a05/, source log SHA256
`73e983efeed546a7aa86c7111fd571a6c24d7a074ca0883dd278aec8c9501882`.

The installed 4.0.0b33 signature confirms deterministic=False by default and
a deterministic=True option. A targeted P7-simple-short previous/previous
and previous/optimized diagnostic now compares native FA4 and deterministic
FA4. It is measurement only; production settings and acceptance bounds are
unchanged. Fresh root: proxy-fa4-repeatability-20261008-a01.
