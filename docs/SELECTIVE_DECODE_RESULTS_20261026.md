# Selective decode and retrieval results — 26 October 2026 SGT

**Decision:** local/global allocation preserves this simple retrieval task, but learned routing has not demonstrated a speed advantage over dense attention using the same kernel. Do not scale up router training based on these timings. The new kernel remains experimental because it failed the predeclared BF16 native-trajectory gate.

## What was tested

Frozen original 28-layer Qwen3-0.6B-Base and the exact previously fitted routers. No model/router training, package installation, driver change or old-output cleanup. Each layer keeps four of eight GQA groups global; other groups keep four sink tokens and the recent 128 tokens. All query heads participate. All native KV remains allocated.

- **Dense:** original native SDPA. **Dense kernel:** all groups global using the new split-K kernel.
- **Fixed groups:** calibration-selected global groups, native sliced SDPA. **Fixed kernel:** identical policy with selective kernel.
- **Joint reference:** learned joint router, native full/local branches combined. **Joint kernel:** identical router with selective KV reads.
- **Independent router:** matched classifier fitted with independent group-cost targets; included in retrieval.
- The inference-only kernel preserves native KV layout and reads only selected full/local histories. Prefill retains the reference path. No training backward, paged cache or KV-capacity saving is claimed.

Implementation: [kernel](../efficient_llm/selective_decode.py), [policy integration](../efficient_llm/context_attention.py), [screen](../efficient_llm/context_deploy.py), [precision controls](../efficient_llm/decode_numerics.py), [lookup construction](../efficient_llm/lookup_probe.py). [Predeclared plan and subsequent investigation](SELECTIVE_DECODE_PLAN_20261026.md).

## Whole-model cached decoding

Milliseconds per decode step, including QKV/MLP/logits/router and Python overhead. BF16, eager HF, StaticCache, synthetic teacher-forced tokens, no CUDA graphs or full compilation. Compilation/prefill excluded. Three repeats of 32 steps, rotating policy order; table is the mean of two GPUs’ median latencies. The kernel timings are **diagnostic only**, not an accepted drop-in backend.

| Batch | Context | Dense SDPA | Dense kernel | Fixed SDPA | Fixed kernel | Joint reference | Joint kernel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 20.84 | 20.44 | 25.68 | 20.23 | 28.28 | 23.36 |
| 8 | 2048 | 21.32 | 20.55 | 25.75 | 21.18 | 28.39 | 23.68 |
| 1 | 8192 | 21.87 | 21.07 | 27.02 | 20.99 | 30.46 | 24.23 |
| 8 | 8192 | 32.72 | 20.74 | 28.70 | 20.84 | 34.96 | 24.26 |

At batch 8/context 8k, joint selective execution is 1.349× faster than native SDPA, but **16.9% slower than dense using the same kernel**. Dense kernel itself is 1.577× faster than native SDPA. Fixed kernel is essentially tied with dense kernel. At the other three settings, joint selective execution is about 11–12% slower than native SDPA. These results do not establish a routing speedup; a stronger dense implementation removes the apparent advantage.

Prior natural-text quality remains [the context-budget result](CONTEXT_BUDGET_RESULTS_20261026.md): joint PPL is 1.17%/2.15% above dense at 2k/8k. This run does not re-estimate full-corpus PPL for the selective kernel.

## Retrieval: established SDPA policies only

Eight-choice single-token record lookup, chance 12.5%. Sixteen balanced cases in each cell, fixed seed 9042. A target is placed near the end or at 10%, 50%, or 90% of the prompt. All methods have the same records/queries. Records repeat across cells, so these are correlated sanity checks, not 128 independent standard-benchmark examples. The selective kernel is excluded.

| Context | Target position | Dense | Fixed groups | Joint router | Independent router | Local only |
|---:|---|---:|---:|---:|---:|---:|
| 2048 | near | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 2048 | 10% | 16/16 | 16/16 | 16/16 | 16/16 | 2/16 |
| 2048 | 50% | 16/16 | 16/16 | 16/16 | 16/16 | 2/16 |
| 2048 | 90% | 16/16 | 16/16 | 16/16 | 16/16 | 3/16 |
| 8192 | near | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 8192 | 10% | 16/16 | 16/16 | 16/16 | 16/16 | 2/16 |
| 8192 | 50% | 16/16 | 16/16 | 16/16 | 16/16 | 2/16 |
| 8192 | 90% | 16/16 | 16/16 | 16/16 | 16/16 | 2/16 |

Dense/fixed/joint/independent: **128/128** each. Local only: **45/128**, comprising 32/32 near and 13/96 distant. The positive controls pass. Half-global allocation preserves this task, while local-only largely fails at distance. The test is saturated among the four global-capable methods: it cannot establish a learned-router advantage or general downstream parity.

## Numerical checks and the failed gate

80 local tests across the maintained CPU suites, including Triton interpreter math, real tiny-model caching, HF two-rank router fits, route replay and job manifests. The review also fixed an existing cache-ownership bug: retain a unique object token instead of a recyclable numeric Python object ID. Native attention calculations remain unchanged.

Compiled B200 checks passed for FP32/BF16, lengths 1–8192, batches 1/2/8, mixed global/local groups, padded NaNs and inaccessible-history NaNs. FP32 attention max error was 8.35e-7; BF16 versus rounded FP32 reference max error was 0.0009766. Original-weight FP32 full-model comparisons (64 tokens) had max logit error 3.82e-5.

The initial queue nevertheless **failed** its BF16 cached native-comparison threshold: document 4 had +0.015731 nats mean NLL difference over 32 positions, above the preset absolute .01 limit. The queue stopped and restored burns; it did not run retrieval. The kernel and threshold were not changed to obtain a pass.

The follow-up measured all eight original documents (256 cached positions):

| Control | Mean NLL delta (right − left) | Changed next-token choices |
|---|---:|---:|
| Dense native → dense kernel | −0.000862 | 2/256 |
| Joint native → joint kernel | +0.000420 | 8/256 |
| Joint native → FP32 attention on BF16 QKV | −0.001874 | 6/256 |
| Joint native → kernel with replayed native routes | +0.001063 | 10/256 |

Native/kernel routing choices differ at 129/7168 layer-token sites (1.80%). Holding the same BF16-rounded weights but computing the model in FP32 restores agreement on all eight documents (maximum logit difference 2.87e-5). The FP32-attention control also deviates after casting each attention output back to BF16. Replaying routes reduces extreme logit deviations but does not remove all NLL differences.

Our inference is arithmetic/quantization sensitivity, amplified at some hard router decisions; no indexing/masking failure was found in the tested cases. PyTorch documents [backend-dependent SDPA numerical differences](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). This is not a proof of long-generation equivalence. The small aggregate loss delta does not erase individual failed checks.

The final profile intentionally records status `measured`, `diagnostic_only=true`, and `native_nll_gate_passed=false`. Its completed job means measurement completed; **it does not mean numerical acceptance passed**. Retrieval uses the existing SDPA implementation and therefore does not validate the kernel.

## Evidence and execution

- Initial gate/source: `9481bee`, run `6y3o-efficient-selective-decode-20261026-a01`; preserved passing compiled gate and failed profile execution.
- Numerical diagnosis/source: `25dc043`, run `6y3o-efficient-decode-numerics-20261026-a01`; eight-GPU precision/replay controls.
- Final measurement/source: `8adb821`, run `6y3o-efficient-selective-research-20261026-a01`; profile 81.5 seconds, retrieval 43.1 seconds; both exited zero at **23:52:48 SGT**.
- Each launch checked pinned inputs, copied/verified Accelerate and inspected its environment, stopped only verified owned burn workers, and verified all eight GPUs free. No environments/drivers changed.
- Burns restored and communicating at **23:53:59 SGT**, independently healthy **23:54:51 SGT**; eight ready ranks, collective probe 36 and advancing payload, guard released. Workers 602209–602216, launcher 602140 at observation. Reinspect before any future stop.
- Raw logs: `temp/deploy_health01.log`, `temp/numerics_health01.log`, `temp/research_health01.log`. Prior router weights remain in `temp/context_results_20261026`.
- Durable [initial gate](results/selective_decode_20261026/initial-gate/supervised/run/gate.json) and [numerical controls](results/selective_decode_20261026/numerics/supervised/run/numerics.json). Final [timings](results/selective_decode_20261026/research/supervised/run/profile.json), [all 640 scored lookups](results/selective_decode_20261026/research/supervised/run/retrieval.json), [verification](results/selective_decode_20261026/research/verification.json), and [hashes](results/selective_decode_20261026/research/artifact-hashes.json). All eight exported files verified against the node SHA256s. Raw archive and frozen lookup tensors: `temp/research_results_20261026`. Export commit `b48013a`; commands.sh restored to #0 after verification.

## Next decision

Keep the existing SDPA path as the default. Do not expand router training yet. The remaining efficiency question is whether selective context helps against a strong dense decoder in a genuinely attention-bound workload (longer context and/or less Python launch overhead). A small profile of that workload is more informative than another training sweep. Static half-global allocation is a necessary reference: it already solves every lookup case here. No novelty, broad quality parity or KV-capacity reduction is established by this screen.
