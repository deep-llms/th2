# Proxy optimization CUDA validation and timing

Authorized scope: CUDA correctness, 25-step eight-GPU smokes/resume checks,
and before/after throughput. **No 2,500-step research runs.**

Fresh output root: `/mnt/local/_outputs/deep-llms_th2/proxy-speed-validation-20261008-a01`.
Machine: `thiennh-p6-tjx3-worker-0`. Read-only inspection at 02:37:27 UTC on
8 October verified eight B200s, only the known communicating burn workers
401980–401987, and the expected attention_bench package versions. Inspection
log: `temp/proxy-validation-inspection-20261008-a01.log`, SHA256
`e87dc71acc1e3a8bb8e642598c0f7ca839bc55c328d5f07f2b58e0c44027025e`.

## Queue

1. Nine numerical gates: P6-iso-sparse, P6-iso-short, P6-iso-weighted,
   P6-iso-layernorm, P7-simple-sparse, P7-simple-short, P6-iso, P7-simple and P7.
   Full-size Qwen, two real packed 2048-token rows, BF16 CUDA, both SDPA and
   FA4. Compare previous/optimized code at the same weights and checkpointing
   setting, separately with checkpointing off/on. Two-pass bootstrap, hidden
   outputs, losses, every gradient, normalization moments and updated buffers
   must match exactly. A failure is recorded and stops the queue; no tolerance
   is relaxed automatically.
2. Six new arms each train 25 steps on all eight GPUs. Validate finite metrics,
   backend receipts, schedule, normalization and checkpoint contents. Copy the
   complete step-24 checkpoint into a fresh resume directory, verify all copied
   state-file hashes, resume one update to 25, then compare with uninterrupted
   step 25: model, optimizer, scheduler and all eight RNG states exactly.
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
outputs/caches are preserved. Expected 46 sequential stages.

Local preparation: four tests passed in 37.776 s, including actual old/new
train.py wrapper runs and identical saved tiny-model weights. A separate gate
control test checks pairing and failure behavior. Local tests use CPU only.
Remote completion and numerical/performance results remain pending.
