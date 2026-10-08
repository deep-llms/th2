# P6/P7 execution optimization — 8 October 2026

This changes execution of the existing arms, not their definitions. Targets,
gradient routing, losses, gates, parameter initialization, optimizer, data order,
document isolation and attention backends are unchanged. No new CLI options,
parameters or checkpoint fields are introduced.

## Changes

- `deep_kv/proxy.py`: calculate all target-normalization inverse scales with one
  batched median/floor/rsqrt operation per forward. Reuse the centered target
  already calculated for statistics, and count valid tokens once. Auxiliary-only
  forwards no longer calculate clipping counts that are discarded. Bootstrap
  mean/variance passes and inference skip the normalization work they do not use.
- `deep_kv/proxy_memory.py`: rotate each P7 proxy key once. Previously the Qwen
  query/key helper rotated the same key twice and discarded one result.
- P7 FA4: share doubled document boundaries across proxy layers and activation
  checkpoint replay. Interleaving, duplicated queries and the kernel are unchanged.

Caches belong only to the current forward. Statistics updates and checkpoint
reloads therefore cannot leave stale normalization scales or document layouts.
The shared normalization change also applies to earlier proxy arms.

## Verification

`tests/test_proxy_optimization.py` compares against the previous normalization
and key-RoPE expressions, including nontrivial means/variances and zero variance.
It covers the P6 family, all P7-simple variants and original P7, FP32/BF16,
SDPA and an independent CPU oracle for FA4, with checkpoint replay. Outputs
and every parameter gradient must match exactly (`rtol=atol=0`). Additional
checks exercise fresh scales after buffer updates and layout lifetime.

The three focused tests passed in 8.655 seconds. Logs and local profiling
artifacts are under `temp/proxy-optimization-20261008/` (Git-ignored).
The twelve shared P1/P3 head tests also passed in 11.481 seconds.
The broader 79-test regression passed in 661.568 seconds: anticipatory and
memory proxies, normalization, FA4 CPU oracle, Trainer/Accelerate integration,
two-rank CPU DDP, gradient accumulation, exact resume, checkpoint evaluation
and supervised fine-tuning. **94 tests passed across these three invocations.**
Commands (all with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`):

```bash
temp/p6-review-env/bin/python -m unittest tests.test_proxy_optimization
temp/p6-review-env/bin/python -m unittest tests.test_proxy_heads
temp/p6-review-env/bin/python -m unittest tests.test_anticipatory_proxy tests.test_proxy_memory tests.test_proxy_normalization tests.test_fa4_proxy tests.test_proxy_training tests.test_eval tests.test_finetune
```

Logs: `reference-tests.log`, `shared-heads.log`, `regression.log`. CPU profiling
prints an old-driver CUDA-device-query warning; no CUDA computation was used.

## Profiling scope

The local benchmark compares the old code imported before editing with the new
code, using seed 20261008 and all eight P6-iso/P7-simple parent/variant settings.
It uses 28 tiny Qwen blocks (hidden 128, intermediate 256, head dimension 16,
vocabulary 257), predictor width 32, batch 2, sequence 128 and documents of
32 tokens. SDPA, FP32 and CPU BF16; all activation checkpointing is disabled.
Each measurement includes LM and auxiliary forward, target statistics and
backward, but no optimizer step. Two warmups, seven timed iterations, then one
operator-profiled iteration. Saved outputs/gradients support a separate exact
before/after comparison at full depth.

All **16 full-depth comparisons matched every saved output and parameter
gradient exactly**, including target-statistics outputs. In twelve-location
P6-iso, full forward/backward `aten::rsqrt` count fell from 149 to 138,
subtractions from 36 to 24 and sums from 547 to 536. P7-simple additionally
reduced concatenations from 121 to 109 and multiplies from 1742 to 1707.
Normalization now invokes one batched median instead of twelve individual
medians (six for sparse); the profiler records two `aten::median` scopes
for that batched call, rather than twelve/six scalar calls.

Exploratory median forward/backward milliseconds (before → after):

| Arm | FP32 CPU | BF16 CPU |
|---|---:|---:|
| P6-iso | 595.8 → 565.9 | 1913.4 → 1795.7 |
| P6-iso-sparse | 582.2 → 540.4 | 1869.2 → 1752.2 |
| P6-iso-short | 602.1 → 558.9 | 1890.0 → 1788.4 |
| P6-iso-weighted | 607.3 → 568.7 | 1830.9 → 1794.3 |
| P6-iso-layernorm | 565.1 → 566.6 | 1783.8 → 1789.9 |
| P7-simple | 603.3 → 606.8 | 1833.4 → 1838.0 |
| P7-simple-sparse | 565.3 → 569.1 | 1775.6 → 1834.6 |
| P7-simple-short | 603.3 → 609.4 | 1853.0 → 1880.7 |

These runs **do not establish an overall speedup**, particularly for P7.
Timing differences range from 7.2% shorter to 3.3% longer and are confounded
by changing background load. The initial partial `before/` run was stopped
and excluded; only the fully seeded `before-seeded/` and `after-seeded/` runs
are compared in `comparison.json`. The benchmark and comparator scripts are
saved alongside these artifacts.

CPU timings are exploratory: concurrent local regression tests add noise, and
these small tensors do not predict B200 performance. Operator counts establish
removed work, not an end-to-end GPU speedup. No CUDA kernel, full-size B200 or
eight-GPU throughput test was run for this change. Existing B200 workloads and
environments were untouched; GPU validation is still required before deployment.
