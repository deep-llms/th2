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

## Verified B200 results

All ten discarded probes completed 29 October 2026 at **02:18:36 SGT**. Supervisor exited successfully and restored burns at 02:19:47; independent health at 02:23:26 verified guard release, eight burn workers and collective progress 260→270 cycles. All 88 frozen source files match commit `7441fd0`; all ten result hashes match the terminal job receipts.

Use the **unprofiled reference repeat** (1.16083 seconds/update) for speed comparisons. The first profiled reference measured 1.24486; using it would overstate improvements. Steady measurements cover eleven updates per probe, not full-run elapsed time.

| Variant | Seconds/update | Throughput gain | Peak allocated GiB/GPU | NLL delta vs first reference |
|---|---:|---:|---:|---:|
| reference | 1.24486 | -6.8% | 41.67 | +0.000000 |
| deferred | 1.15259 | +0.7% | 41.67 | +0.000397 |
| fused | 1.14745 | +1.2% | 41.67 | +0.000255 |
| chunk512 | 1.07639 | +7.8% | 41.94 | +0.000054 |
| chunk1024 | 1.05847 | +9.7% | 42.42 | +0.000252 |
| micro2 | 0.98200 | +18.2% | 53.56 | -0.000064 |
| micro4 | 0.94989 | +22.2% | 75.83 | +0.000263 |
| combined | 1.04583 | +11.0% | 41.94 | +0.000024 |
| compiled_head | 1.01925 | +13.9% | 42.04 | +0.000314 |
| reference_repeat | 1.16083 | +0.0% | 41.67 | +0.000273 |

The best tested configuration is **microbatch 4, chunk 256**, using the existing production `heal --microbatch 4 --chunk 256` options. Global batch stays 262,144 tokens/update; accumulation falls from eight to two. It reaches 275,972 tokens/second versus 225,825, a **22.2% throughput increase / 18.2% shorter update**, with 75.83 GiB allocated per GPU. This recommendation does not require adopting the experimental fused/deferred wrapper or compiler. Original production defaults remain unchanged.

Compiled head/loss works on the actual runtime. Its 1.01925 seconds/update is only about 2.6% faster than its matching combined eager configuration (1.04583), and slower than microbatch4. Cold first-update compute was 5.11 seconds compiled versus 2.34 combined eager; full startup/audit time is excluded from steady timings. A microbatch4+compile combination was not tested.

Every variant passed the fixed numerical gates, exact effective global data order, LR schedule and teacher-no-gradient checks. Microbatch4 gradient relative L2 was 0.0006573, cosine 0.999999784 and final small-set NLL delta +0.000263 nats. Reference repeat itself differed by +0.000273 nats. These are 16-update screening results for DCLM/P300 S1, **not bitwise equality or evidence of unchanged long-training quality across every parent/pattern**. No scientific fit was launched.

Artifacts: [validated summary](results/a2_performance_20261029/final/summary.json), [terminal verification](results/a2_performance_20261029/final/terminal-verification.json), and [SHA receipts](results/a2_performance_20261029/final/SHA256SUMS). Full raw local receipt: `temp/a2_perf_health01.log`, SHA256 `3cdb0a97b1517696f735c065b46785a3fcc1d161233d7bd06ef64bbcbec6bdc7`; extraction `temp/a2_perf_health01/`. Node root: `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-performance-20261029-a01`. Full profiler and gradient tensors remain on node; the health receipt contains only a profiler-text tail, so no operator-level bottleneck attribution is claimed from it.

## Authorized MBS8 follow-up

User requested measuring larger microbatch after the first results. New separate suite `mbs8`: microbatch4/chunk256 reference → microbatch8/chunk128 → microbatch8/chunk256 → repeat microbatch4/chunk256. All four use unchanged native `KDModel`/`HealTrainer`, no profiler/compiler/fused/deferred additions. Four discarded16-update probes, same fresh DCLM/P300/S1 weights, 64sequences×4096 tokens/update, original small-set/first-gradient thresholds and exact order audits. Accumulation2→1, no global batch or schedule change. Expected8–12minutes after startup, hard4×450seconds plus supervised cleanup. Predecessor is verified performance-a01 burn owner; native launcher freshly verifies exact workers/start times and GPU availability, restores communicating burns on success/failure. Original ten-probe suite and scientific results preserved.

### MBS8 results — complete

All four probes completed **29 October03:11:27SGT**; successful supervisor/burn handoff03:12:33. Independent health03:31:29 verified89frozenfiles (`04ab566`), four artifact hashes, releasedguard, eight knownburnworkers676151–676158 and advancingcollectives1310→1320cycles. Dropbox HTTP500 during an earlier check was transient; user-requested retry succeeded, then read-only health `e817f2f` returned these results. No benchmark restart.

| Native configuration | Seconds/update | Tokens/second | Throughput vs MBS4 repeat | Peak allocated GiB/GPU |
|---|---:|---:|---:|---:|
| MBS4/chunk256 first |0.94893|276,251|+0.04%|75.83|
| MBS8/chunk128 |0.93273|281,049|+1.78%|115.49|
| MBS8/chunk256 |0.91553|286,332|+3.69%|116.46|
| MBS4/chunk256 repeat |0.94934|276,132|reference|75.83|

MBS8/chunk256 is the fastest measured configuration, but the gain is **modest:3.7% throughput** for **40.6GiB extra allocated memory/GPU**. Both MBS4 references differ by only0.04%, supporting stable timing within this short experiment. No candidate reaches the existing5% automatic-screening improvement threshold, so the summarizer correctly leaves `fastest_screened_candidate` null. Keep MBS4 as the conservative recommendation; MBS8 is a valid optional choice when maximizing measured throughput on this exact B200 setup is worth the memory. Production defaults remain unchanged.

All four numerical/order/LR gates passed. MBS8/chunk256 first-gradient relativeL2=0.0007558, cosine=0.999999714, final small-setNLL delta=+0.000544nats versus firstMBS4; repeatMBS4 delta=+0.000596. Effective globalbatch262144tokens and sequence4096 remain unchanged. Only16-update screening; no long-run quality or statistical-significance claim. No further GPUjobs launched.

[Summary](results/a2_mbs8_20261029/final/summary.json), [terminal verification](results/a2_mbs8_20261029/final/terminal-verification.json), [SHA receipts](results/a2_mbs8_20261029/final/SHA256SUMS). Raw local `temp/a2_mbs8_health01.log`, SHA256 `b7ff4e7fe570b5fac415c8c7b2be4613401f125601184b49ca338f47c7cb7376`; full extraction `temp/a2_mbs8_health01/`. Node root `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-mbs8-20261029-a01`.
