# Proxy optimization CUDA validation and timing

Authorized scope: CUDA correctness, 25-step eight-GPU smokes/resume checks,
and before/after throughput. **No 2,500-step research runs.**

## Completed results — 8 October 2026, 05:02 UTC

All three authorized stages passed. The continuation completed all 31 stages
at 04:58:09 UTC. No long research training was launched.

- **Correctness:** nine arms × two backends × checkpointing off/on = 36
  old/optimized pairs. Every output, loss, gradient, normalization statistic
  and buffer matched exactly. Backends were math SDPA and deterministic FA4;
  this does not establish bitwise reproducibility of native FA4 backward.
- **Training:** all six new arms completed 25 updates on eight B200s, with
  finite metrics and complete checkpoints. Peak allocated memory was
  108.7–114.2 GiB.
- **Resume:** all six checkpoint-24→25 checks passed. Normalization, scheduler
  and all eight RNG states matched exactly. Model/optimizer differences were
  tiny FP32 residuals: maximum relative L2 across arms was 9.50e-8, within both
  prescribed bounds. All 48 source/resume data-audit pairs matched byte for
  byte; update-25 data also matched across all six arms on every rank.

Native-FA4 throughput, median full optimizer-update time over steps 10–25:

| Arm | Before (s/update) | After (s/update) | Throughput gain | Peak allocated GiB, before → after |
|---|---:|---:|---:|---:|
| A | 2.0704 | 2.0698 | 0.03% | 103.6 → 103.6 |
| P6-iso | 2.2571 | 2.2399 | 0.77% | 111.6 → 111.6 |
| P7-simple | 2.3911 | 2.3648 | 1.11% | 114.2 → 114.2 |
| P7 | 2.6084 | 2.5846 | 0.92% | 116.3 → 118.5 |

These are whole training updates, not attention-only timings; startup,
checkpoint I/O and the component-profile update are excluded. One short pair
per arm is a screening measurement, so differences around 1% should not be
treated as a precise sustained speedup. P7 used 2.27 GiB more peak memory.
The component profile supports reduced work: P6 target setup/normalization
kernel sums fell from 55.60 to 37.46 ms; these sums are not wall-clock critical
path timings. Production FA4 settings, environment and drivers were unchanged.

Burn restoration passed at 04:59:21 UTC. An independent 05:01:52 UTC snapshot
confirmed all eight GPUs at 100% utilization, with workers 485959–485966 matching
the supervisor's verified communicating burns. `commands.sh` is left inactive.

Final evidence is in `artifacts/proxy-speed-monitor-20261008-a26/`:
`summary.json`, `supervised/run/complete.json`, the six `resume-checks/` reports,
all 96 data-audit files, per-arm benchmark reports, and burn/GPU receipts.
All completion-output hashes and all 16 reused source-output hashes were
independently verified locally. Numerical/smoke source reports are in
`artifacts/proxy-speed-monitor-20261008-a13/`.

Final Dropbox log: `_run-2026-10-08_05-01-41-th2-tjx3-proxy-speed-monitor-20261008-a26.log`,
local copy `temp/proxy-speed-monitor-a26.log`, SHA256
`07df85353653dd5a9c9fd36dba272f7ecf8e389e0a9a3bfd275fbbbb36a5f2c0`.
Source a13 log SHA256:
`7c0fa9b72453ede3a702c7927974329d003fb6ee1ccebad221bafefda2cf2404`.
Artifacts and raw logs are local, Git-ignored evidence; this report is tracked.

The sections below retain the protocol and failed-attempt history, including
native-kernel numerical variability and the corrected logging-cadence checker.

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
   deterministic FA4. Compare previous/optimized code at the same weights and checkpointing
   setting, separately with checkpointing off/on. Up to eight independent gate processes run concurrently, one per GPU.
   Two-pass bootstrap, hidden
   outputs, losses, every gradient, normalization moments and updated buffers
   must match exactly for math SDPA. Deterministic FA4 outputs must match exactly; gradients
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

The existing train.py/Accelerate recipe is retained: microbatch 16,
accumulation 4, eight GPUs, 1,048,576 input tokens/update, sequence 2048,
document isolation/reset positions, seed 42, schedule 28,600, warmup 1,430,
activation checkpointing off, FA4. Correctness smokes/resumes use a validation-only
`deterministic=True` kernel override; throughput runs use normal FA4.
The override is recorded in attention-runtime receipts and never affects
production defaults. Logging stays every ten steps so the new
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
Final completion and numerical/performance results are recorded above.

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

## Controlled FA4 result and final test mode

P7-simple-short diagnostic completed (105 s). Native FA4 old/old was exact;
old/optimized had four gradient tensors differing, maximum relative L2
7.50660e-8. Deterministic FA4 old/old and old/optimized were **exact** for
all outputs and gradients. The earlier 0.494% difference did not recur in this
serial control; this does not establish universal native-kernel repeatability.
Evidence: artifacts/proxy-fa4-repeatability-monitor-20261008-a01/, source log
SHA256 `137285522c5f7c3850395c4280b150fa167ee77ddd698976ccb38b3f8a19b312`.

Fresh final study: **proxy-speed-validation-20261008-a04**. Numerical checks
use math SDPA and deterministic FA4, checkpointing off/on. Six 25-step
correctness smokes and their resumes also use deterministic FA4, making a
strict resume comparison meaningful. All four before/after throughput pairs
use native FA4; the wrapper and summary reject deterministic timing runs.
The tight eight-FP32-epsilon bound is retained; no BF16-scale tolerance added.
This is a controlled correctness experiment, not a switch of production
backends/defaults. Native FA4 already produced identical forward outputs for
all nine arms in a03; its gradients are not claimed to be bitwise reproducible.

Burn restoration after the targeted control was independently confirmed by
artifacts/proxy-fa4-repeatability-monitor-20261008-a02/ (fresh snapshot
03:54:15 UTC). Four focused override/gate/queue/real-resume tests passed in
44.708 s before the next launch.

## Controlled numerical checks and six training smokes completed

Study a04: all nine arms matched **every output and gradient exactly** under
math SDPA and deterministic FA4, checkpointing off/on (36 old/new pairs).
All six 25-step, eight-GPU smokes completed successfully; peaks were 108.7–114.2
GiB. No architecture, objective or production training code was changed.

The post-run checker then failed because it required a loss log at step 25
although this study logs every 10 steps. This was a validator assumption, not
a training failure. Fixed it to require all scheduled logs (10 and 20 here),
while independently requiring step 25 in result, Trainer state and checkpoint.
Added coverage for non-aligned cutoffs and rejection of missing scheduled logs.

Continuation **proxy-speed-validation-20261008-a05** reuses the completed
a04 artifacts after checking their saved hashes, production-module hashes and
identical recipe. Source outputs remain read-only. Its 31 stages rerun the
corrected checker, perform the six resume comparisons and eight native-FA4
timing runs, with fresh output paths and automatic burn restoration.
Source evidence: artifacts/proxy-speed-monitor-20261008-a12/, source log SHA256
`a34cf455e9687451de20c38a6bf5d0accde34d39ee651696a6d5ac6fe97d34b3`.
Local checker regression passed (0.199 s); continuation/queue tests passed
(0.069 s), including rejection of corrupted source evidence.
