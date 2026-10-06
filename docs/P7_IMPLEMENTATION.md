# P7 implementation and validation

Implemented against `proxy_arm_P7_spec.md`, Revision 2. The operator explicitly
requested both dense SDPA and FA4; this extends the specification's SDPA-only
implementation wording without changing the attention function or objective.

## Entry points

Use the existing `train.py` arguments and recipe, changing only:

```text
--arm P7 --attention_backend sdpa
--arm P7 --attention_backend fa4
```

Also supported: `P7-kq`, `P7-ems`, and `P7-mlp`, with either backend.
The usual `python -m deep_kv make-jobs ... --arms A P7 --seeds 42` generates
sequential, eight-GPU experiments with the existing comparison stage. Specify
the desired arms explicitly. Select the backend in the recipe passed to this
command. Existing backend/configuration guards reject incompatible resumes.

The shared tokenizer, EOS packing, document boundaries/reset positions, data
cache, sampler, HF Trainer/Accelerate loop, scheduler, optimizer, stopping and
checkpoint handling are reused. No new preprocessing or training loop is added.
This implementation task did not launch or change a B200 workload.

## Model and attention

`deep_kv/proxy_memory.py` contains the causal four-tap estimator, independent
proxy projections, deterministic query sampling and relational KL. Integration
with placement, normalization, targets and losses is in `deep_kv/proxy.py`.

For the full model, blocks 2,4,...,24 add proxy entries to KV groups 6–7,
serving query heads 12–15. The other heads use native attention. Native queries,
keys and values retain their normal Qwen projections, normalization and RoPE.
Both entries for a token use that token's reset position.

- SDPA uses a shared boolean rectangular mask `[M | M]` and one softmax over
  `[native; proxy]` in the selected heads. The mask is reused by proxy layers.
- FA4 interleaves entries `[native_0, proxy_0, native_1, proxy_1, ...]`, duplicates
  each query, doubles each fragment's cumulative length and retains only odd
  query outputs. At position `2*t+1`, causal attention sees exactly both entries
  for each `j <= t`. Discarded even-query outputs have zero adjoints. Native
  and proxy entries still compete in one softmax. No dense attention mask or
  unequal-length causal alignment assumption is used in FA4 training.
- Proxy ablation removes the entire proxy-entry set by calling native attention.
  This is mathematically equivalent to masking the proxy half, and keeps the
  vanilla kernel shape for the baseline-equivalence check.

The predictor reads detached block inputs; its output is detached before the
proxy K/V projections. Consequently auxiliary backward has no edge into the
backbone/proxy projections, and LM backward has no edge into the predictor.
There is no alpha gate. New modules use a separate RNG scope, default seed 43;
the backbone initialization matches A exactly.

Default targets are detached FP32 four-block increments
`h[layer+3] - h[layer-1]`, standardized with existing per-channel running
statistics, two-pass initialization, clipping and frozen evaluation/replay.
Metadata uses `target_version=p7-r1`. P7-mlp instead uses the specified MLP sum.
P7-ems adds the existing document-reset EMS at 0.9 and 0.99. P7-kq duplicates
native values and does not instantiate the unused proxy value projection.

The conditional learnable logit-bias option is omitted. Gradient-bearing float
masks are not assumed supported by the selected fused backends. The training
wrapper supports full-sequence evaluation, not incremental generation with a
second KV cache; the specification's cache design is not implemented here.

## Loss scaling and reproducibility

The objective is `LM + lambda * (cosine + 0.5 * relational_KL)`, averaged over
proxy layers, with the specified 250-step auxiliary ramp to 0.1. Relational
logits use FP32 normalized predictions **with gradient**, detached targets,
temperature 0.1, and strictly earlier same-document candidates. Empty candidate
queries are excluded; zero-query sequences contribute differentiable zero.

LM targets, cosine tokens and sampled relational queries have **three separate
denominators**. Trainer sums each across the entire accumulation window and
all ranks. DDP's gradient average is compensated once; no extra accumulation
divisor is applied. Logs and evaluation use the same separate denominators.

Sampling uses a fresh CPU generator for each `(global_step, sequence_index)`.
The training index is `row_ordinal_within_rank_update * world_size + rank`;
row ordinals span all microbatches in the update. Sampling does not consume
model/data RNG, mutate checkpoint state or depend on layer/checkpoint replay.
Each sequence selects at most 256 unique eligible queries. Evaluation uses a
deterministic row ordinal and resets it for normal and ablated passes. This
rule is recorded in the saved experiment metadata. Identical recipe, world
size and data order reproduce the sampling after resume.

Every log interval includes per-layer cosine loss, relational loss, mean cosine
and existing normalization diagnostics. Evaluation adds proxy-masked LM loss
and per-layer proxy attention mass on the first at most eight held-out sequences
globally. Mass uses eager FP32 probabilities in 128-query chunks, without
changing normal training attention or normalization state.
P7 requests an additional evaluation at step 1,000 for the specified mid-run
check and records `p7_mean_cosine` and `p7_low_cosine_trigger` (threshold 0.3).
The trigger is a heuristic. Follow-up variants are available through the
existing queue interface; this flag does not silently start extra experiments.

## Cost and launch checks

The compute report distinguishes mathematical P7 attention from the FA4 adapter.
For production dimensions and one 2048-token document per sequence:

- P7 adds 12,596,736 parameters.
- Mathematical forward addition: 25,184,256 MAC/token, excluding training-only
  losses and normalization.
- The duplicated-query FA4 adapter adds 50,350,080 MAC/token under the same
  causal-work accounting. Discarding even outputs does not avoid their forward
  computation. These are operation counts, not measured wall-clock overhead.
- Relational target **and** prediction similarity products together add about
  6,291,456 forward MAC/token across the twelve layers, before backward/softmax.
  Counting just one similarity product underestimates this part of the spec.

`scripts/check_fa4_proxy.py worker --arm P7 ...` now includes relational loss in
the numerical comparison and supports all four names. Its smoke validator
expects 40 attention calls for enabled full-size P7, checks the new checkpoint
parameters/metrics, and retains the original optimizer/scheduler/eight-rank RNG
and backend checks. `scripts/profile_proxy_training.py` labels estimator,
proxy projections, memory attention and relational loss within real Trainer
updates. Its profile remains a disposable wrapper, not part of training.

Before production, run the full-model CUDA comparisons and eight-GPU Trainer
smokes/profile on the intended backend and recipe. Actual FA4 CUDA numerics,
cuDNN rectangular-mask dispatch, peak memory and throughput are **not yet
validated for P7**. No overhead or production-readiness claim follows from
the CPU checks.

## Local tests

Use the matching pinned environment, with GPUs hidden:

```bash
CUDA_VISIBLE_DEVICES='' /home/users/thien/miniconda3/envs/sampling_b200/bin/python -m unittest \
  tests.test_proxy_memory tests.test_proxy_heads tests.test_proxy_training \
  tests.test_anticipatory_proxy tests.test_fa4_baseline tests.test_fa4_proxy \
  tests.test_fa4_screen_validation -v
```

P7 acceptance covers placement/GQA, masked-baseline equivalence, unchanged
native heads, joint-softmax outputs and gradients, document/causal isolation,
target definitions/detachment, loss routing, query selection, checkpointing,
BF16 finiteness, sampling RNG isolation, accumulation, parameter counts and
attention mass. Both paths are exercised; **CPU FA4 tests substitute an
independent per-document SDPA oracle for the CUDA kernel**.

Real CPU Trainer tests cover save/resume for every arm/backend, two-rank Gloo
DDP for all four arms, and 300-token sequences exercising the sampled-query
branch. The latter compares resumed weights, optimizer moments and scheduler
exactly against uninterrupted training. Legacy proxy/backend tests check the
shared code paths. Three older queue assertions assumed the recipe still used
SDPA: reproducing them against unmodified HEAD confirmed they were stale.
They now select screen arms explicitly and expect the existing FA4 rejection;
production queue behavior was not changed to satisfy the tests.

Initial implementation validation: **63 unique tests passed**. The combined acceptance and
regression run passed 62 tests in 322.031 seconds; the additional step-1,000
evaluation test passed separately. The strengthened FA4 observer-count check
also passed. `git diff --check` passed. Logs for this session:
`/tmp/p7-final-tests.log`, `/tmp/p7-midrun-final.log`, and
`/tmp/p7-midrun-tests.log` (the last includes the observer test and an earlier
test-fixture error subsequently corrected). No CUDA training was run.

## Second review and performance work (2026-10-06)

Rechecked joint attention/GQA, target windows and detachment, loss denominators,
sampling, checkpoint replay, evaluation and resume. The SDPA and FA4 attention
functions are unchanged by this optimization pass. Changes are limited to:

- Batched relational similarity products, replacing two matrix multiplications
  per sequence with two batched operations per proxy layer. At microbatch16,
  that is32forward matmul calls replaced by2. Arithmetic and sampled queries
  are unchanged; this is not a16x throughput claim.
- Reuse of convolution boundary masks and relational query indices, masks and
  counts across all proxy layers and checkpoint replay. Autograd shares the
  cached relational mask rather than keeping separately created inverse masks.
- One transfer for the sampled query indices of a microbatch. Query sampling
  and its GPU-to-CPU synchronization are skipped when auxiliary loss is disabled,
  including target initialization and the attention-mass diagnostic.
- P7 normalization moments use the existing fused reduction helper: one
  collective instead of four per update, and one instead of three per bootstrap
  pass. No change to the mathematical statistics/update rule.

Also tightened an existing configuration check: a nonfinite auxiliary-loss
weight is now rejected during model construction, before training/output setup.

The batched loss pads uneven sampled-query counts. Dummy queries have one
harmless candidate to keep softmax finite, but their loss and gradient are
removed. Dedicated tests compare the original loop implementation with the
batched version, including zero-query rows, singleton candidate sets, sampling
above256queries, FP32/BF16 inputs, zero vectors, full-model gradients and both
attention paths. The original loop remains only as a test oracle.

FP32 batched and loop derivatives agree within the strict test tolerance.
An initial test applied FP32 tolerances to BF16 leaf gradients; a tiny FP32
reduction difference can round to adjacent BF16 values. The check now separately
validates the FP32 derivative on the quantized inputs and bounds the BF16 result
by the neighboring representable values (one ULP). No attention acceptance
tolerance was changed. Batching/reduction fusion preserves the objective but
does not promise bitwise identity to the old execution order. Exact resume is
tested between interrupted and uninterrupted runs of the optimized code.

These changes reduce repeated work and GPU launch/communication overhead.
CPU component timings are exploratory, not full training or B200 measurements.
The larger FA4 query-duplication cost remains; improving that path requires
separate forward/backward kernel validation. Full-model CUDA correctness,
memory and end-to-end throughput measurements remain required before launch.

Exploratory CPU relational-loss forward+backward timing (one thread, FP32,
two documents per row, fixed queries/masks, warmup followed by nine alternating
measurements; dimensions are batch/sequence/feature):

| Shape | Original loop median | Batched median |
|---|---:|---:|
| 16 × 512 × 128 | 169.08 ms | 144.34 ms |
| 4 × 2048 × 128 | 129.65 ms | 136.72 ms |

These small CPU probes show that batching is not a universal wall-clock win.
The intended GPU benefit is fewer launches and collectives; neither the table
nor operation counts establish B200 end-to-end speed. Source-session timing
log: `/tmp/p7-cpu-component-final.log`.

Second-review result: **all65tests passed in319.199seconds**, including the
new loop/batched comparisons, all-arm CPU Trainer runs, two-rank DDP, exact
300-token sampled-query resume and legacy proxy/backend regressions. The
nonfinite-weight construction guard was also verified in a targeted rerun.
`git diff --check` passed. Logs: `/tmp/p7-review-final.log` and
`/tmp/p7-config-guard.log`. FA4 CPU coverage still uses the reference kernel;
no B200 workload, environment or deployment was changed.
