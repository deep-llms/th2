# A2 distillation performance study — 29 October 2026 SGT

User authorized profiling, safe optimization and torch.compile experiments after completed Phase0A. This is a discarded implementation benchmark, not Phase0B or additional scientific arms. Original `model.py`, `train.py`, runtime, data and all old results remain unchanged.

Representative: actual DCLM/P300 native OpenSci teacher (24 layers), S1 student (18 layers), FP32 student/BF16 AMP, native SDPA, full-vocabulary pure KL. Each candidate starts fresh with the same 16-update schedule, seed, 64×4096 tokens/update and real first 1024 packed training sequences. Audit actual eight-rank order; microbatch changes rank assignment, not the effective global batch.

Bounded candidates: original reference with one profiled update; deferred loss/order synchronization; fused AdamW; chunk 512; chunk 1024; microbatch2/chunk512; microbatch4/chunk256; combined deferred+fused+chunk512; same combined with compiled loss chunk; original reference repeat. No candidate changes decoder/loss definition or thresholds. Compile uses Inductor fullgraph on `kl_chunk` only, default mode, backward autocast explicitly off to match HF Trainer. No global decoder compilation or custom attention. Compilation caches live under this new output root; one compile thread per process.

Deferred mode retains GPU sample IDs and copies them once for exact order audit. Finite loss flags are aggregated across all microbatches and ranks before each optimizer update; nonfinite loss aborts before parameter modification. This differs from immediate detection, so it remains opt-in. Fused optimizer keeps the exact optimizer grouping, betas, epsilon, weight decay, clipping and schedule.

Checks before considering promotion:
- CPU native-model loss/gradient/AOT capture equivalence, 12-step accumulation and fused/deferred update equivalence, nonfinite abort before update, data-order and launcher contracts: 19 tests pass, `temp/a2_performance_all_cpu.log`.
- B200 actual first-update **clipped** gradients for every trainable parameter versus original (relative L2≤0.02, cosine≥0.9998); same teacher has no gradients. These are screening tolerances, not bitwise equivalence.
- Identical final small-set NLL evaluation after 16 updates, absolute difference≤0.002 nats; exact data coverage, LR and update count.
- Full actual Trainer timings; exclude first five updates including startup, full-gradient audit and the profiled update. Report mean/median of 11 steady updates, max-rank durations, peak memory and cold first-update compute time. Compiler overhead is not hidden as steady throughput.
- Original repeat quantifies timing/numerical variability. No default change based on a failed or marginal test, and no inference-speed claim.

Expected 15–30 minutes after startup (compile time not measured yet); each eager probe capped at450seconds, compile at900seconds, total jobs≤4950seconds plus supervised cleanup. CPU input hashes and fresh known burn identity checks, Accelerate config copy/env verification, free GPU confirmation and native automatic communicating burn restoration are mandatory. Any failed process stops the queue; numerical rejection is retained as a result, never promoted to acceptance.

References for implementation constraints: [PyTorch compile](https://docs.pytorch.org/docs/stable/generated/torch.compile), [compile profiling](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html), [non-reentrant checkpointing](https://docs.pytorch.org/docs/main/utils/checkpoint.html). Installed runtime and actual GPU tests determine compatibility.
