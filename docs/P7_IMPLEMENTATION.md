# P7 implementation and validation

Forward-local normalization/layout reuse and proxy-key RoPE optimization are
documented in [the 8 October performance review](PROXY_OPTIMIZATION_20261008.md).

Implemented against `proxy_arm_P7_spec.md`, Revision 2. The operator explicitly
requested both dense SDPA and FA4; this extends the specification's SDPA-only
implementation wording without changing the attention function or objective.

## P7-simple variants based on P6-iso (8 October 2026)

These transfer the two new P6-iso ablations to P7-simple. They are separate
experiments, using the existing P7 consumer and training loop.

| Arm | Proxy blocks, 1-based | Detached MLP target | P6 counterpart |
|---|---|---|---|
| `P7-simple` | 2, 4, …, 24 | `m_l + m_(l+1) + m_(l+2) + m_(l+3)` | `P6-iso` |
| `P7-simple-sparse` | 2, 6, 10, 14, 18, 22 | Same four-block sum | `P6-iso-sparse` |
| `P7-simple-short` | 2, 4, …, 24 | `m_l + m_(l+1)` | `P6-iso-short` |

At each selected block, the bias-free 1024→256→1024 SiLU predictor receives
`stopgrad(RMSNorm(h))`. Its output is RMS-normalized, detached for LM use, and
projected into independent proxy K/V entries. The last four query heads
(12–15, KV groups 6–7) attend jointly to native and proxy entries with one
softmax. The other twelve query heads keep native attention. P7 does not add
P6's residual injection or channel gate: the predicted information is consumed
through attention. Causal document isolation and reset RoPE positions apply to
both entry types. Dense SDPA and FA4 use the existing tested implementations.

The target, predictor architecture, normalization, cosine auxiliary loss and
its ramp `0.1 * min(step / 250, 1)` match the respective P6-iso counterpart.
Auxiliary gradients reach only the predictor; LM gradients reach the backbone
and proxy projections. There is no convolution, EMS or relational KL. Loss is
averaged over tokens and active locations, so sparse placement preserves the
auxiliary coefficient. The short variant keeps twelve locations and does not
add late-block injections. Both variants' final target window ends at block 25.

Sparse initialization preserves all retained predictors **and memory projections**
from P7-simple under the same seeds; omitted heads do not remain in the model
or optimizer. Its extra parameter count is 6,292,224, versus 12,584,448 for
P7-simple/short. Sparse uses 34 attention calls per 28-block forward, versus 40
for P7-simple/short. Actual throughput requires measurement. Both new arms
require fresh training and cannot resume under a different arm or target.

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --arms P7-simple-sparse P7-simple-short --seeds 42 --stop-after 2500 \
  --output temp/p7-p6-variants-jobs.json
```

This generates the existing sequential eight-GPU queue only. Defaults resolve
from the arm; omit `proxy_lookahead`. Conflicting placement, target, isolation,
head allocation or gate overrides are rejected. Checkpoint evaluation and
optional supervised fine-tuning support both variants, with task gradients
re-enabled in fine-tuning as for P7-simple. Existing P7 and P6 definitions are
unchanged. No research results or full-size CUDA validation exist for these
new arms yet; they have not been added to the running B200 queue.

Local verification: all 41 regression tests passed in 353.541 seconds, including
full-depth target/initialization checks, gradient routing and accumulation,
two-rank CPU DDP, exact Trainer resume, report guards, evaluation and fine-tuning.
Three focused checks also passed (2.084 seconds). Logs:
temp/p7-p6-regression.log and temp/p7-p6-focused.log. Both attention interfaces
were tested on CPU; FA4 used an independent SDPA oracle, not its CUDA kernel.

## P7 P6-variant follow-up review

No production-code defect was found. Added two independent full-depth regression
checks using 28 blocks with tiny hidden dimensions. In FP32 and BF16, on dense
SDPA and the CPU FA4 oracle, short exactly matches the parent's initial LM
hidden states and gradients; sparse exactly matches P7-simple with only the
omitted memory consumers bypassed. The independent cosine-loss calculation uses
nontrivial per-layer means/variances and clipping, confirms averaging over tokens
and active locations, and confirms that only predictor parameters receive
auxiliary gradients. Three full-depth checks passed in 9.575 seconds; strict
checkpoint reload and supervised gradient-routing checks passed in 3.995 seconds.
Logs: temp/p7-p6-review-full-depth.log and temp/p7-p6-review-reload.log.
The three integration checks (exact Trainer resume/report/queue, accumulation,
and screen-result validation) also passed in temp/p7-p6-review-integration.log.
That invocation included a misspelled evaluation test class; the correctly named
reload test passed in the separate two-test run above. Eight intended checks
passed across these runs; the previous complete 41-test regression also passed.
Actual FA4 CUDA kernels/full-size B200 validation remain pending. Existing
training code and B200 workloads were not changed during this review.

## P7-simple — controlled memory comparison (6 October 2026)

The operator approved a simpler P7 following the design review. Use
`--arm P7-simple --attention_backend fa4` (or `sdpa`) with the existing recipe.
This is a new arm; existing P7/P7-kq/P7-ems/P7-mlp definitions stay unchanged.

**Head decision:** use the last **four query heads**, zero-based 12–15 in
Qwen3-0.6B, corresponding to KV groups 6–7. The other twelve query heads retain
native attention. This matches P4-iso-4h and the existing P7 head allocation,
keeps most attention native and avoids adding a head-count sweep before there
is evidence of benefit. Four is a controlled starting choice, not an empirically
established optimum. The arm requires two query heads per KV group and rejects
incompatible GQA layouts rather than silently changing the number of heads.

At blocks ℓ = 2, 4, …, 24:

- Source: the current block's input RMSNorm output `u_ℓ`.
- Predictor: `p_ℓ = W2 SiLU(W1 stopgrad(u_ℓ))`, bias-free 1024 → 256 → 1024.
  No convolution or EMS. Width remains configurable for tiny tests.
- Target: detached sum of the current and next three blocks' MLP outputs,
  `Σ(i=ℓ..ℓ+3) MLP_i`, standardized by the existing running per-channel mean
  and variance, then clipped exactly as in P4-iso; cosine loss normalizes vector
  directions. Target version
  is `p4p6-r1`; moments bootstrap/update/checkpoint exactly as before.
- Loss: cosine only, averaged over tokens and proxy layers. Total objective is
  `LM_loss + λ(step) cosine_loss`, with default `λ = 0.1 min(step/250, 1)`.
  There is no relational loss, sampled-query denominator or query sampling.
- Consumer: RMS-normalize `stopgrad(p_ℓ)` and project independent proxy K/V.
  Proxy K uses Qwen K normalization and the token's document-reset RoPE position.
  Four heads jointly attend to native and proxy entries through **one softmax**.
  Both kinds of entries obey causal, same-document visibility. There is no gate.
- Gradients: LM trains the backbone and proxy K/V projections; cosine trains
  only the predictor. Targets and normalization statistics are detached.

All predictor weights initialize identically to P4-iso/P4-iso-4h under the
same explicit module seed (default 43). Memory projections initialize only
after all predictors, so they do not perturb that RNG sequence. The head choice,
predictor, target and loss are matched to P4-iso-4h; changing consumption also
adds independent K/V projection parameters and removes P4's additive gate.
It is therefore an architecture comparison, not a parameter-matched control.
At production dimensions it adds 12,584,448 parameters across twelve blocks.

Implementation reuses `MemoryHead.entries`, `ProxyModel.memory_attention`,
FA4's interleaved varlen adapter, dense SDPA and the existing HF Trainer.
`SimpleMemoryHead` in `deep_kv/proxy_memory.py` supplies only the tokenwise MLP.
The model distinguishes memory attention from relational supervision so the
new arm uses ordinary LM-token/cosine-token accumulation denominators.
Logs retain per-layer cosine, auxiliary loss, attention mass and proxy-disabled
LM evaluation. Result metadata explicitly excludes relational loss, and the
report rejects an increment target or relational metadata for this arm.

The existing sequential queue accepts `--arms A P4-iso-4h P7-simple --seeds 42`
with `proxy_heads.b200.json`. It preserves microbatch 16, accumulation 4,
eight GPUs, 2,048-token packing with EOS/document isolation, the 28,600-step
schedule, 1,430 warmup steps and 2,500-step cutoff. Generating this queue does
not launch it. Existing result directories and checkpoints must not be
relabelled as the new arm.

Local validation covers exact P4 predictor initialization/output/gradients,
exact MLP-sum targets, no relational work, four-head joint softmax, untouched
native heads, gradient routes, causality and document isolation, checkpoint
recomputation, BF16 finiteness, accumulation, resume, reports and two-rank DDP.
CPU FA4 checks use an independent SDPA oracle in place of the CUDA kernel.
The earlier B200 results below cover the original four variants only:
**P7-simple subsequently passed its own CUDA/full-model smoke; see below.**
Local implementation validation covered 66 distinct tests. The careful review
adds two regression tests that compare against P4-iso-4h directly:

- With both consumers disabled, the same backbone produces matching target
  bootstrap moments, cosine loss, predictor gradients and running-statistic
  updates on both attention paths. This tests equality under a controlled
  forward pass; active P4 and P7 naturally produce different activations.
- All twelve predictors in a 28-block test model initialize identically under
  a second module seed, and both arms use the same default lambda schedule.
- Real tiny Trainer runs for A/P4-iso-4h/P7-simple share a data fingerprint and
  pass the comparison report on both backends. A changed auxiliary weight is
  rejected. Their queue preserves the requested arm order and eight GPUs/run.

The two new tests passed in 7.163 seconds, and the fresh 66-test regression
suite passed in a single invocation: all 68 tests covered by this review passed.
Logs are `/tmp/p7-simple-review-tests.log` and
`/tmp/p7-simple-review-control.log`. No training implementation issue has been
found in this review; only regression coverage and this report were extended.
No B200 process or execution command was changed.

## P7-simple B200 acceptance and production start

Launch `258bde6` under the authorized ten-arm remaining queue. The actual CUDA
SDPA/FA4 comparison passed (loss absolute difference 0.0002040863; whole-model
and proxy-only gradient relative L2 0.00592701 / 0.00513579; document-isolation
output and gradient differences both zero). The 25-step eight-GPU Trainer
smoke and checkpoint/backend validator passed, peak allocated 114.2105 GiB.
The exact matched-A recipe/data-order guard also passed. At 13:49:06 UTC,
monitor `e759c88` verified fresh P7-simple production training on all eight
GPUs with finite losses/gradients; it began at 13:47:01 UTC and will stop at
2,500 steps using the original full schedule. This establishes launch and
smoke correctness, not a long-run quality improvement.

Root: `/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01`.
See CURRENT_TASK.md for the ten-arm priority order and automatic burn handoff.
Startup evidence: `artifacts/proxy-remaining-launch-20261006/`.

## Entry points

Use the existing `train.py` arguments and recipe, changing only:

```text
--arm P7 --attention_backend sdpa
--arm P7 --attention_backend fa4
```

Also supported: `P7-kq`, `P7-ems`, `P7-mlp`, and `P7-simple`, with either backend.
The usual `python -m deep_kv make-jobs ... --arms A P7 --seeds 42` generates
sequential, eight-GPU experiments with the existing comparison stage. Specify
the desired arms explicitly. Select the backend in the recipe passed to this
command. Existing backend/configuration guards reject incompatible resumes.

The shared tokenizer, EOS packing, document boundaries/reset positions, data
cache, sampler, HF Trainer/Accelerate loop, scheduler, optimizer, stopping and
checkpoint handling are reused. No new preprocessing or training loop is added.
Initial implementation was local only. The operator subsequently authorized B200
acceptance tests; see the dated CUDA validation section below.

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

The initial CPU checks required full-model CUDA comparisons and eight-GPU
Trainer smokes on the intended recipe before production. Those subsequent
B200 results are recorded below; CPU checks alone did not establish actual
kernel behavior, memory use or throughput.

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
memory and end-to-end throughput measurements were required before launch
and are recorded in the subsequent B200 validation section.

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

## B200 validation — 6 October 2026 (passed)

Launch `2729792`, root `/mnt/local/_outputs/deep-llms_th2/p7-checks-20261006-a01`.
The existing `attention_bench` environment and driver were retained. Accelerate
was copied to its actual cache and verified as eight-GPU BF16. The supervisor
stopped only the inspected approved burn workers and verified all GPUs free.

Full Qwen3-0.6B CUDA checks passed separate LM, cosine and relational gradient
routing for P7/P7-kq/P7-ems/P7-mlp on SDPA and FA4. Auxiliary gradients reached
only the 36 estimator parameter tensors; LM gradients reached the backbone and
independent proxy projections. Running-statistics buffers stayed unchanged.

At production relational tensor shape `[16,2048,1024]`, the original loop and
batched loss matched (absolute loss difference zero); maximum FP32 derivative
difference was `3.41e-13`, relative L2 `1.26e-7`. The check also passed with
BF16-representable input values, mixed query counts, and a no-query row.
Alternating CUDA-event timing, excluding two warmups: median forward+backward
10.3406 ms (loop), 3.1768 ms (batched), approximately 3.25x component speedup.
This is **not** the end-to-end training speedup. Query plans were fixed for this
component timing; the full Trainer measurements include plan construction.

Cross-backend full-model numerical checks passed for all four variants:

| Arm | Absolute objective difference | Overall gradient relative L2 | Proxy gradient relative L2 |
|---|---:|---:|---:|
| P7 | 0.00002098 | 0.005925 | 0.005351 |
| P7-kq | 0.00006580 | 0.005497 | 0.010543 |
| P7-ems | 0.00003529 | 0.005971 | 0.005240 |
| P7-mlp | 0.00002193 | 0.005959 | 0.005407 |

No tolerance was relaxed. All variants had zero cross-document gradient and
output influence in the full-model FA4 perturbation test; parameters and
normalization buffers were unchanged by the numerical captures.

All ten 25-step eight-GPU Trainer runs and both backend validators passed.
They used the unchanged 28,600-step schedule, micro16/GAS4, sequence2048 and 32 eval
rows. They are acceptance/throughput tests, not evidence of model quality.

Small prior P4 result files are retained in
`artifacts/p7-review-20261006/p4-results/`: 42 files checked against producer
SHA256 hashes, archive `ff145becef7fd37c44896f22258646531df8bfe23d7a95c1d50c8f50e73f6285`.
Both P4 arms completed 2,500 steps with passing validators and automatic
communicating-burn handoff. Held-out LM: A 3.477173383, P4-iso-4h 3.479563634,
P4-4h 3.479081029. Neither P4 result improves on A in this single-seed screen.

FA4 steady-state measurements (median updates 10–25; update 6 is profiled and
excluded) at the production micro16/GAS4/global1,048,576-token batch:

| Arm | Seconds/update | Overhead versus FA4 A | Peak allocated GiB |
|---|---:|---:|---:|
| A | 2.0691 | — | 103.62 |
| P7 | 2.6094 | 26.11% | 116.27 |
| P7-kq | 2.6077 | 26.03% | 118.48 |
| P7-ems | 2.6336 | 27.28% | 118.72 |
| P7-mlp | 2.6338 | 27.30% | 116.27 |

Each P7 receipt recorded 40 FA4 calls and 40 Q/K/V gradient callbacks; no dense
mask or SDPA fallback. All final checkpoints contained optimizer/scheduler
state and eight rank-specific RNG files. Proxy attention-mass/eval diagnostics
were finite and checkpoint parameters passed validation. Profiled kernel
sums are diagnostic, not critical-path timings; the table uses unprofiled
wall-clock update measurements.

SDPA steady-state measurements under the same recipe:

| Arm | Seconds/update | Overhead versus SDPA A | Peak allocated GiB |
|---|---:|---:|---:|
| A | 2.4286 | — | 110.62 |
| P7 | 2.9736 | 22.44% | 125.89 |
| P7-kq | 2.9721 | 22.38% | 125.84 |
| P7-ems | 2.9978 | 23.44% | 126.07 |
| P7-mlp | 3.0005 | 23.55% | 125.89 |

For P7, FA4 reduces update time by about 12.25% versus SDPA (2.6094 versus
2.9736 seconds), with about 9.62 GiB less peak allocated memory. Both backends
passed the existing validators; no acceptance threshold was loosened. Actual
SDPA receipts identify cuDNN forward and backward on every call, including
rectangular P7 attention, with no math fallback or unattributed calls.
Final paired validation LM differences after 25 updates were at most 0.000043
across A and the four variants. These are only 25-step, 32-row smoke results;
they do not establish long-run convergence, quality, or a P7 improvement over A.

The full supervised queue exited zero and completed automatic burn restoration
at **10:17:19 UTC / 18:17:19 Singapore**. A fresh 10:18–10:19 UTC read-only
inspection verified exactly eight approved burn workers, GPU guard enabled,
and rank-zero collective cycles advancing from 130 to 150. No driver or
environment installation was performed. Original P4 results/checkpoints and
training data/cache were preserved.

Evidence: remote root above; local verified archive
`artifacts/p7-review-20261006/p7-results.tar.gz`, SHA256
`4e170e0184e61c68d3f4826b5ed95deecac60fe994e8e632ba58ca57f13a4955`,
161 small result/config/log/receipt files (no model weights). Producer final
log: `temp/p7-final-results.log`, SHA256
`b8ee127fa987aba70d1732060dc72aad17f61abd0ad0d57e0cb61136ea19f981`.

Local extraction verified the archive and all 161 manifest entries, all 23
queue stages, all ten final Trainer states, and recorded source hashes against
the current code. Runner completion is `status="ok"` in `complete.json` (the
filename denotes completion); the local verifier uses that existing schema.
`commands.sh` was returned to inactive `#0` after retrieval.
