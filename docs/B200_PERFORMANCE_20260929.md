# B200 training performance investigation — 2026-09-29

Status: all 14 cases completed across the original queue and a corrected FA4
retry. All measurements, input hashes and automatic final burns are verified.

The measured gains come from disabling unnecessary decoder recomputation and
using implicit causal backbone attention. Keep Trainer/Accelerate, microbatch 16
and PyTorch SDPA. The combined tested path improves throughput by 1.52x for B,
1.40x for F and 1.36x for G. Microbatch 32 and FA4 add no compelling benefit
in these short tests. Production settings have not been changed.

## Scope and method

The user reported roughly one hour per 1,000 updates on H100/H200 and little
speed improvement from increasing microbatch 16 to 32. The historical script
at `9a4a08c:scripts/train_qwen3_0.6b_baseline.sh` used eight H200 GPUs,
microbatch 16, accumulation 4, sequence length 2,048, and did not request
activation checkpointing. Its referenced `smoke_train.py` is unavailable;
this investigation cannot establish a controlled B200/H200 hardware ratio.

The benchmark uses the current full Qwen3-0.6B model and real packed English
training data. Every variant uses eight GPUs and 1,048,576 input tokens per
update. Each starts fresh, retains the 28,600-update LR schedule and 1,430-update
warmup, and stops after 18 updates. Timings cover updates 6–18, taking the
slowest rank's interval for each update. Initialization, profiling at update 3,
final evaluation and checkpoint writes are outside the measurement window.
These are short throughput tests, not convergence experiments or full-job ETAs.

Each variant first passes a tiny-model CUDA BF16 objective/gradient comparison
against the production model. The branch output is nonzero during validation,
so branch-to-backbone gradients are exercised. Five local CPU tests also cover
native-forward equivalence, all measured arms' causal-mask equivalence,
checkpoint/chunk equivalence, rejection of padded/segmented fast-path inputs,
and the pinned FA4 tuple-return adapter API.
The final summary checks all eight ranks, step counts, accumulation counts,
and identical first-global-batch row hashes across variants. Microbatch 32
uses accumulation 2; global batch and data examples remain fixed.

`train.py`, `deep_kv/model.py`, and `deep_kv/training.py` remain unchanged.
`scripts/benchmark_training.py` installs experimental subclasses only inside
fresh benchmark worker processes. Scientific checkpoints are not resumed,
overwritten, or evaluated by this benchmark.

## Variants

- **Current A/B/F/G:** unchanged production computation.
- **Native A:** native Qwen forward, no decoder checkpointing, native causal
  mask handling, and full vocabulary loss. This is a combined control, not
  an isolation of any single change.
- **No-checkpoint B:** disable decoder activation checkpointing only.
- **Chunk512 B:** enlarge vocabulary-loss chunks from 128 to 512 only.
- **Causal B:** replace the backbone's explicit 4D causal mask with implicit
  causal attention for fully packed, unpadded, unsegmented contexts only.
- **Fast B/F/G:** combine no decoder checkpointing, chunk size 512 and implicit
  backbone causality. Auxiliary strict-past attention and F/G losses remain
  unchanged, including detached targets and FP32 functional-loss calculations.
- **Batch32 B:** original path, microbatch 32, accumulation 2.
- **perf-env-fast-B / FA4 B:** repeat fast B in the isolated environment, then
  change only its backbone attention to FlashAttention-4. This separates
  environment changes from the attention implementation change.

## Environment and execution

Production `train_env` was preserved. The controller created `perf_env` from
`envs/perf_env.txt`; installation commit `93a404d` completed at 12:05:23 UTC.
CPU-only preflight `634b56b` at 12:08:39 confirmed FA4 imports and versions:
PyTorch 2.14.0+cu130, Transformers 5.9.0, Accelerate 1.13.0, datasets 4.8.5,
FlashAttention-4 4.0.0b32, CUTLASS DSL 4.8.0, cuDNN 9.24.0.43, Triton 3.8.0.
Both environments share the same core framework versions.

The eight B200 GPUs have 1,000 W power limits and NV18 links between every
pair. Active benchmark snapshots showed SM clocks around 1,830–1,965 MHz;
they do not indicate a low configured power cap or PCIe-only communication.

Launch commit `9aa7132cb2db92e737af85642701182ec578cc0d`, job
`th2-78gg-performance-benchmark-20260929-a01`, started its queue at 12:13:59 UTC.
Root: `/mnt/local/_outputs/deep-llms_th2/performance-20260929-a01`.
The resource Accelerate config was copied to the actual HF cache, byte-checked,
and verified with `accelerate env` for both interpreters. The existing
supervisor reclaimed only freshly verified approved burn workers and recorded
all eight GPUs free before launching. It owns cleanup and automatic enhanced
burn restoration on success or failure.

## Results

All 13 SDPA/native cases completed and 138 artifacts were retrieved and
SHA256-verified. FA4 initially failed in its precheck because the adapter did
not unpack the pinned API's `(output, lse)` tuple. This was a benchmark adapter
bug, not a model or hardware failure. The fix and a tuple-API regression test
passed all five CPU tests; retry commit `a5ede38` ran only FA4 in fresh root
`performance-fa4-20260929-a02`. The original supervisor restored burns at
12:42:15 UTC, and the 12:45 UTC live audit verified all eight ranks with
cycles 230→240 and collective payload 255.38→266.48 GiB. Timings are
seconds per optimizer update; SD is across the 13 measured update intervals,
not an independent-run confidence interval. Memory is peak PyTorch allocated
memory per GPU during measured training, not total device occupancy.

| Case | Seconds/update | SD (seconds) | Peak allocated GiB/GPU |
|---|---:|---:|---:|
| current-A | 3.3018 | 0.1300 | 19.51 |
| native-A | 2.0829 | 0.0024 | 138.38 |
| current-B | 3.3662 | 0.1349 | 19.66 |
| no-checkpoint-B | 2.5963 | 0.0033 | 94.88 |
| chunk512-B | 3.3165 | 0.0987 | 30.10 |
| causal-B | 2.8380 | 0.1297 | 19.41 |
| fast-B | 2.2075 | 0.0025 | 98.32 |
| batch32-B | 3.2795 | 0.1988 | 27.48 |
| current-F | 4.0102 | 0.1217 | 19.72 |
| fast-F | 2.8547 | 0.0037 | 98.38 |
| current-G | 4.3710 | 0.1335 | 19.84 |
| fast-G | 3.2174 | 0.0044 | 98.45 |
| perf-env-fast-B | 2.2077 | 0.0027 | 98.32 |
| fa4-B (corrected retry) | 2.2281 | 0.0025 | 98.32 |

The batch-size result reproduces the user's observation: doubling microbatch
from 16 to 32 gives only about 2.6% less time, with substantial timing variation.
Disabling decoder checkpointing alone reduces B time by 22.9%; implicit causal
masking alone reduces it by 15.7%. Increasing loss chunks alone shows little
benefit. The combined tested path gives 1.52x B, 1.40x F and 1.36x G throughput.
B's optimized 2.21 seconds/update corresponds to about 37 minutes per 1,000
updates before evaluation/checkpoint overhead. The separate environment control
is effectively identical to train_env, so it is not responsible for the gain.
FA4 is about 0.9% slower than optimized SDPA in the matched environment; no
speedup was observed. The trace confirms actual CUTLASS SM100 FA4 forward and
backward kernels, with only the unchanged auxiliary branch still using cuDNN.
FA4's BF16 precheck loss difference was 0.0000272 and gradient relative L2 was
0.00737 (0.74%), within the predefined 0.01 / 0.03 thresholds. No tolerance was
relaxed. All 112 rank prechecks passed across the 14 successful probes.

## Final verification and artifacts

The corrected FA4 training and summary both exited zero; the queue completed
at 12:53:42 UTC. Automatic burn handoff passed at 12:54:53. Read-only audit
`96305a0` at 12:55:00–12:55:12 verified eight approved burn workers, all-rank
readiness, unchanged process identities, released guard, and progressing cycles
30→40 / collective payload 33.31→44.41 GiB. All GPUs showed 100% utilization
and 155,212 MiB occupancy. This verifies recovery after the successful retry as
well as the original queue's failure recovery.

Local artifacts:

- `artifacts/performance-20260929-a01/`: 138 verified source artifacts and a
  locally recomputed summary of the 13 completed original cases. The original
  queue's failed status is preserved; it is not rewritten as a successful queue.
- `artifacts/performance-fa4-20260929-a02/`: 20 verified artifacts including
  the successful FA4 queue completion manifest and live burn receipt.
- `artifacts/performance-20260929-a01/combined-summary.json`: all 14 results,
  with cross-run equality of the first 512-row global batch and data fingerprint.

Raw audit SHA256s:

- `temp/perf-failure-audit-20260929-a01.log`:
  `e20839ab944703191a60a8f827fbe8953282800e8e98fe351f393389be611fe6`
- `temp/perf-fa4-final-20260929-a02.log`:
  `7d4f388de1021f8cc47c6e3abc57530fbd6b2a17432b8be5f9fd6acb85ac5aac`

Every pulled artifact matched its source hash; required result/rank files also
matched runner artifact manifests. Timings were recomputed locally from all
eight rank records. Production source hashes remained unchanged. The only
benchmark source change between queues was unpacking FA4's output tuple.
`commands.sh` is returned to inactive `#0`; no training remains queued.

## What the profiler and source establish

The original path already executes cuDNN SM100 flash-attention forward and
backward kernels. The missing standalone `flash-attn` package was not evidence
that attention was unfused. Installing another package alone is not a fix.

The installed Transformers SDPA adapter enables native grouped-query attention
only when `attention_mask is None`. An explicit mask causes K/V head expansion
and sets `is_causal=False`, with causality still correctly encoded by the mask.
An implicit causal mask enables causal specialization and native GQA. This
explains a concrete source of avoidable work in the fully packed backbone path.
Auxiliary strict-past masks remain explicit and unchanged.

Rank 0's profiled B cuDNN forward/backward totals fell from approximately
311/247 ms to 33/103 ms with the combined fast path. Mask-only B measured
65/103 ms; disabling checkpointing alone measured 155/248 ms. These are
instrumented kernel totals, distinct from the unprofiled full-update timings.

The old H200 script did not request decoder checkpointing; the current custom
model recomputes every decoder layer. The isolated no-checkpoint test measures
the cost of that memory-saving choice. Its extra memory use fits on B200 at
microbatch 16; it should not be assumed to fit at microbatch 32.

Both native and custom controls report `DistributedDataParallel` as the wrapped
model. The existing HF loop already suppresses DDP gradient synchronization on
intermediate accumulation microbatches. Rank 0's first current-A profile spends
about 12 ms in NCCL all-reduce kernels versus seconds per update. This does not
measure every possible CPU/waiting overhead, but does not support rewriting
Trainer as the first optimization.

Profiler output includes CUDA annotation spans (for example
`DistributedDataParallel.forward`) alongside kernels. Do not sum all records,
double-count operator and kernel times, or call that annotation DDP overhead.

## Further work and limits

BF16-equivalent kernels need not produce bit-identical long-run weights. Use
the same chosen performance settings across scientific arms. The current
checkpointed production recipe remains unchanged pending a deliberate switch.

Many pointwise multiplication and cast kernels remain in the trace. Selective
compilation or fused normalization/activation/loss kernels are plausible next
candidates, not measured speedups here. A blanket native-Qwen patch is unsafe
for this wrapper: the consumer/target blocks and per-example losses have custom
paths. Any such follow-up must verify objective and gradients, including F/G
teacher-target precision, before timing real training.

Official references:

- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [FlashAttention-4 for Hopper/Blackwell](https://github.com/Dao-AILab/flash-attention#flashattention-4-cutedsl)
- [Accelerate gradient synchronization](https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization)
- [PyTorch normalization compilation](https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/)
- [Transformers fused training kernels](https://huggingface.co/docs/transformers/kernels)
