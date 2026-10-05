# P1/P3 implementation and validation

Implements [revision 7 of the proxy-head specification](proxy_heads_P1_P3_spec_v3.md).
The user confirmed dense SDPA and document-local position IDs. This is a new
screening series; previous non-isolated A–G checkpoints are not matched controls.
No B200 experiment was launched during implementation.

## Model and shared training flow

`deep_kv/proxy.py` adds a model adapter to the existing Qwen backbone. P1 replaces
the last two KV groups in eligible even layers using a gated estimate of a future
MLP-output sum. P3 uses three document-reset exponential moving sums and full-width
deep-band increment targets. Queries and the K/V projection parameters remain native; the
model makes one SDPA call per layer. GQA ordering, QK normalization and RoPE follow
Qwen. Gates start at zero; estimator weights use normal initialization.

Every screening arm, including A/V1/V3, uses the same boolean causal/same-document
mask with `is_causal=False`. The existing collator resets positions from the same
document IDs used by P3. EOS remains at each real document boundary; literal EOS
inside source text does not create a false boundary. LM targets across boundaries
are excluded. Existing packing, caching and train/eval selection are reused.

The older DeepKV path uses an additive dense mask. Its earlier B200 dispatch and
throughput audit is useful background, **not validation of this new boolean-mask
proxy implementation**. The existing first-train/first-eval runtime audit remains
enabled and will record actual GPU dispatch on the next launch.

`deep_kv/proxy_training.py` supplies only the additional loss, normalization and logging
adapters. Optimizer, scheduler, dataloader, accumulation, distributed execution,
saving, resume and stopping remain HF Trainer/Accelerate through `train.py`.
No extra data-export or preprocessing command is required.

Targets use per-channel running standardization, relative variance floor 0.01,
clipping at ±10, epsilon 1e-6 and momentum 0.99. Cosine remains the default;
`proxy_loss_form=smooth_l1` is an opt-in ablation (beta=1), not part of the default
screen. No static channel mask or external calibration is used. The old mask
argument and disabled-centering option fail explicitly instead of silently
changing the new method.

P1 normalizes each detached FP32 MLP window sum. P3 subtracts the detached
FP32 input residual `h[layer-1]` from each deep-band state, normalizes each
increment with its own `(proxy_layer, deep_layer)` statistics, averages the
normalized increments, then applies the same document-reset EMS as the estimator.
The default P1/P3 buffers contain 12/58 mean-variance pairs respectively.

Initialization uses two no-grad forwards on exactly the first training
microbatch of each rank: globally reduced means, then globally reduced squared
deviations. It does not advance the loader or optimizer and skips the unneeded
vocabulary projection. Shifted sums, squares and counts accumulate once in the
original forward, outside checkpointed functions. After all accumulation
microbatches, all-reduce and update once per optimizer step. Variance is the EMA
of **per-step** variances, not pooled historical variance. Statistics remain
frozen during the step, backward recomputation and evaluation. Lambda-zero arms
initialize/update the same statistics but calculate no-grad diagnostic cosines
only at logging intervals. Mean, variance and initialization state are checkpoint
buffers. Alpha gates have zero weight decay; targets remain detached.

Each normalization buffer logs clipping fraction, floored-channel count, median
variance, normalized mean lag and step/running variance ratio. Lag ratios use FP64 and guard exactly zero variance with the smallest
normal FP32 value to keep dead-channel diagnostics finite. These metrics
describe the pre-update statistics used by the last optimizer step in the log
interval. Investigation thresholds in the spec are not automatic stopping rules.

P3 uses the specified two-level chunkwise scan: local 64×64 products, chunk-end
products and carry terms, in FP32. It builds coefficient/mask tensors once per
microbatch and reuses them across layers and targets. There are no token/chunk
Python scan loops, inverse decay powers, random target sketches or custom kernels.

## Arms and recipe

| CLI arm | Purpose |
|---|---|
| `A` with `proxy_screen=true` | Matched unwidened baseline |
| `V1`, `V3` | MLP widening by +72/+88 in all 28 blocks |
| `P1-lambda0`, `P3-lambda0` | Proxy architecture trained through LM only |
| `P1-block`, `P1-flow` | P1 auxiliary gradient-routing comparison |
| `P3-block` | Main P3 arm |
| `P3-flow` | Optional; excluded from the default screening queue |

`proxy_heads.b200.json` retains the full 28,600-update schedule, 1,430-update LR
warmup, cutoff 2,500, sequence 2,048, microbatch 16, accumulation 4 and eight GPUs:
1,048,576 input tokens/update. Auxiliary lambda uses the absolute completed-update
count: 0 for the first update, 0.1 at count 250, then constant. Optional decay is
off. The screen recipe explicitly disables decoder, LM and auxiliary activation
checkpointing, matching the completed isolated A baseline. The existing resume
checks still govern performance changes. Full-size proxy-arm capacity must be
smoke-tested before the screen; baseline capacity alone does not prove it fits.

Queue generation (creates a manifest only):

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --output temp/proxy-screen-jobs.json
python run_experiments.py --config temp/proxy-screen-jobs.json --list
```

Default seeds are 42, 1042, 2042, with matching model/data seeds and deterministic
estimator seed `seed + 1`. Output directories are `seed-N/ARM`. The queue has 24
training jobs, three within-seed reports and one aggregate report, run sequentially.
Each training job uses all eight GPUs. Use `--arms ... --seeds ...` to select a
subset. Historical `deep_kv.b200.json` queue defaults remain A/B/C/D.
Selecting proxy arms explicitly also selects the proxy baseline path for A;
mixing legacy B–G/bottleneck arms into that queue is rejected before any launch.

Data order is shared across arms for a given seed. Report generation checks
recipes, dataset fingerprints, token counts and stopping steps. The aggregate
reports paired differences and seed spread; it does not declare significance.
Evaluation includes the same model with proxy gates disabled and reports the LM
loss increase. Saved Trainer/W&B logs include per-layer losses/cosines, mean gate
magnitudes, lambda, LM loss, gradient norms and seconds/update.

For the full config, added parameters including gates are 6,303,744 for P1 and
7,099,392 for P3. V1/V3 add 6,193,152/7,569,408. Architecture MAC counts are computed
from the actual config; P3 includes chunk-end/carry work as well as local products.
Widening mismatch is below 0.2% of total forward MACs. Training-only target and
auxiliary-recomputation work is excluded from that match and must be measured.
Each result records actual parameter counts, the budget and runtime/memory data.
V3 construction uses the configured sequence length for its scan budget, matching
the reported budget; it does not assume 2,048 when a custom recipe changes length.

## Optional FA4-isolated arm A

`train.py --attention_backend fa4` selects a separate vanilla-A backend
experiment. It automatically uses the matching ProxyModel/ProxyTrainer baseline
path, but has no proxy heads or auxiliary loss. Only arm A is accepted. The
proxy screen and ordinary A default remain dense SDPA.

`baseline_a_fa4.b200.json` copies the current screen recipe, changing only the
attention selection and output directory. Sequence2048, micro16/accum4/eight
GPUs, all checkpointing off, full28600 schedule/warmup1430, cutoff2500 and the
same sampled English data are retained. Choose an explicit seed for a single run:

```bash
python -m deep_kv make-jobs --config baseline_a_fa4.b200.json \
  --seeds 42 --output temp/baseline-a-fa4-jobs.json
python run_experiments.py --config temp/baseline-a-fa4-jobs.json --list
```

This generates a manifest only. Execution requires the separately provisioned
`envs/attention_bench.txt` environment with `flash-attn-4==4.0.0b33`; do not
reinstall a live environment. Missing/other FA4 versions fail explicitly, without
a fallback. The existing eight-GPU launcher and supervisor workflow still apply.

The variant reuses packing, document IDs, EOS handling, reset RoPE positions,
next-token target eligibility, global loss normalization, optimizer/scheduler,
checkpointing and resume. Each contiguous document fragment becomes one varlen
sequence; every packed row starts a new fragment even when adjacent rows share a
document ID. One int32 cu_seqlens layout is built per microbatch and reused across
layers. Native GQA, QK normalization and RoPE precede the FA4 call. No dense mask
is allocated. The adapter follows the prior document-training benchmark's API.

`train_config.json` records `attention_backend=fa4`; default SDPA omits the new
field to preserve historical dense-run resume identity. Backend changes on resume
are rejected even with allow_performance_change_on_resume. Reports retain strict
backend matching; this is a separate backend experiment, not a replacement for
the dense A control. FA4 queues cannot reuse the completed dense baseline.

Results contain attention_runtime package/API metadata and fa4_receipts for the
first actual train/eval GPU microbatch on rank zero. Those receipts record calls,
query-gradient callbacks, dtype, fragment counts and maximum fragment length;
they are not CUDA profiler traces or numerical equivalence certificates.
SDPA receipts remain empty for FA4, rather than mislabeling a no-SDPA run.

Local tests use an independent per-fragment CPU attention stand-in to verify
outputs, all parameter gradients, exact isolation, no dense-mask allocation,
checkpoint recomputation, Trainer save/resume, queue scope and receipts.
Validation: 28 focused tests passed in 36.654 s; all 131 offline CPU tests
passed in 142.696 s (temp/fa4-baseline-full-20261005-a01.log). The queue manifest
passed the runner loader. A separate checkpoint-enabled receipt check verified
recomputation and query-gradient callbacks.

The production FA4 A path subsequently passed B200 numerical/smoke checks and
was launched separately; see CURRENT_TASK.md. No GPU job is launched by these tests.

## Reusing a completed baseline

Reports compare the shared scientific recipe strictly (including attention,
data fingerprints, schedule, execution settings and cutoff). Target metadata is
irrelevant for A/V, so a matching completed dense A remains usable. Proxy arms
must use r7 and matching normalization/loss settings. Within each family, target
quantity and index sets must also match, including lambda-zero controls. Checks
also apply across seeds. FA4 A remains a separate backend comparison.

The saved config records target_version=r7, raw target quantity, complete target
index sets, k, floor, clip, epsilon, momentum and loss form. Strict resume rejects
older definitions or changed hyperparameters, including with the performance-only
override. No target-migration override is provided. A/V retain their historical
saved config and state layout; their unused legacy all-ones mask buffer has no
role in r7 proxy target normalization.

To reuse an existing A without copying weights or altering its saved config:

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --reuse-baseline 42=/local/completed-baseline/seed-42/A \
  --output temp/proxy-screen-jobs.json
```

The supplied recipe is the ordinary screen recipe; no calibration is required.
The reused baseline path above is a placeholder.
`--reuse-baseline` accepts `SEED=/absolute/path/to/A` entries for selected seeds.
The queue validates the reused baseline's completion artifacts, seed and cutoff,
skips only that seed's A training, and includes it in both reports. With one
reused baseline the default screen has 23 new training jobs. New arms still
undergo full recipe/data matching against A when their results are compared.
References point to the original output directory; keep it available and unchanged.
Other seeds still train their own A. No baseline is reused implicitly.

Standalone reports accept `report --baseline-dir /path/to/A`; `report-seeds`
accepts the same `--reuse-baseline SEED=/path/to/A` mapping as queue generation.
The report records the external baseline path; source artifacts are read-only.

## Historical calibration utility

`scripts/calibrate_proxy_mask.py` is retained for inspecting earlier experiments.
Its output is not accepted by revision-7 training. Do not run it as a prerequisite
for the current screen.

## Revision-7 validation (2026-10-05)

All 136 offline regression tests passed in132.101s. A final focused six-test run
passed in4.841s, covering the additional BF16-statistics check, two-pass startup,
frozen accumulation buffers and zero-variance diagnostic guard. The full-depth
independent reference now checks P3 increments and all58 statistics pairs, not
the former normalized full-state targets. Logs:
`temp/proxy-r7-full-tests-20261005-a01.log`,
`temp/proxy-r7-final-t9-tests-a02.log`.

T9 FP32/FP64 maximum relative errors over500 updates: mean8.77e-7 and variance
1.89e-6. After1000 updates with4096 tokens each, a100000-token stationary held-out
batch had max absolute channel mean0.007692 and variance[0.987434,1.010180].
Clipping/variance-floor behavior, target detachment, frozen recomputation/eval,
BF16 compute with FP32 moments/loss, and exact saved-buffer/next-target replay pass.

Two-process CPU training covers A, P1-flow, P3-block and P3-lambda0. Resumed versus
uninterrupted weights and buffers match bitwise; mean/variance buffers also match
bitwise between ranks. Accumulated distributed loss gradients match a single
global batch within3.73e-9. Receipt:
`temp/proxy-r7-ddp-20261005-a01/verified.json`.
The distributed worker accepts any rank count>=2 and now uses FP32 for strict
bitwise resume checks. Existing older eight-rank BF16 evidence below belongs to
revision4. Full-size B200 capacity, dispatch and throughput for revision7 still
need a separate smoke test before screening. This implementation was tested only
on the dev machine and was not pushed to the execution remote.

### Follow-up review

All138 offline tests passed in144.624s (`temp/proxy-r7-review-full-a01.log`).
The additional test verifies unequal-size microbatch pooling and distinguishes
within-step variance from variation between historical step means. Eight CPU
ranks passed (`temp/proxy-r7-review-ddp8-a02/verified.json`): input order and
pre-interruption checkpoints match exactly, normalization buffers match bitwise
immediately after updates (before DDP can broadcast them) and after resume.
Resumed **parameters** differ by at most3.73e-9, including vanilla A; global-batch
versus distributed gradients differ by at most3.73e-9. The worker therefore keeps
buffers exact and separately bounds/reports parameter roundoff (rtol2e-6,
atol2e-8). The initial all-parameter bitwise assertion failed on A and was
corrected after measuring the difference. Two-rank bitwise behavior should not
be generalized to all distributed runs. Production training code was unchanged
by this review; full-size GPU validation is still pending.

## Earlier revision-4 validation (historical)

Local tests use tiny randomly initialized models and synthetic text in the pinned
`sampling_b200` environment. They do not estimate research gains.

Final full offline suite: **118 tests passed in 137.825 seconds**, including the
103 existing regression tests. Log: `temp/proxy-full-suite-final-a02.log`.
Python compilation and Git whitespace checks also passed.

- T1–T7: initialization equivalence, GQA replacement, exact document-2 activation/
  scan/target isolation and zero document-1 input gradients, chunkwise/dense/
  sequential scan agreement, gradient routing, target detachment and lambda-zero
  behavior. Tight checks use FP32 and fixed means; tolerances were not relaxed.
- T4 covers all specified boundaries at chunk size 64, with well-conditioned
  positive inputs for pointwise relative error and random signed backward probes.
- T8 architecture counts/widening and timing logging are tested locally. **Full
  B200 per-arm seconds/update, peak memory and dispatch remain pending.** The old
  2.425 s/update baseline must not be advertised as this implementation's speed.
- Actual `train.py` entry-point tests cover all eight arms, save/resume, centering,
  reliance evaluation and recorded diagnostics. Single-process resumed versus
  uninterrupted model/buffer states match bitwise.
- Eight-process CPU BF16 train/resume checks cover A, P1-flow, P3-block and
  P3-lambda0. Maximum resumed-state difference is 1.862645149230957e-9; centering
  buffers agree exactly across ranks. FP32 masked, accumulated distributed SGD
  versus one global batch differs by at most 3.725290298461914e-9 for both tested
  auxiliary routes. Receipt: `temp/proxy-ddp-a02/verified.json`.

During development, tests caught a malformed tiny config, diagnostics added too
late for saved HF logs, and the distributed test reporter reducing an empty A
buffer. A direct-loss logging test also needed Trainer's normal logging-state
initialization in its fixture. Each was fixed and the relevant tests rerun; no failed candidate was
used for a production run. The direct `hidden_states()` method also now uses the
proxy path, matching the training forward method.

Run the local checks with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/proxy_resume_worker.py \
  --output temp/proxy-resume-check
```

Use a fresh output for the distributed check. Before real screening, verify the
current B200 environment/data, copy and check Accelerate configuration, safely
reclaim authorized idle workers, and measure full-size capacity/throughput in
separate smoke outputs. Keep automatic burn recovery around any authorized GPU
queue. `commands.sh` is still inactive (`#0`). Decode-time cached EMS and the
optional offline estimability study are outside this screening implementation.

## Follow-up code review

A second review found and fixed two non-default configuration issues: custom
queues could select legacy A alongside proxy arms, and V3 construction used a
hard-coded length of 2,048 while its report used the configured length. The
standard 2,048-token eight-arm recipe is numerically unchanged. Aggregate reports
now also reject duplicate seeds so one run cannot be counted repeatedly.

An additional independent reference test covers all 28 layers at reduced width,
16 query/8 KV heads and query width twice hidden width, with nonzero gates and
centering means and excluded channels. It independently constructs P1 window
targets, P3 normalized deep-band targets, sequential EMS, auxiliary losses and
all parameter gradients for block/flow routing. All pass the unchanged FP32
tolerances. Local checks do not replace the pending B200 smoke test.

Review validation: **121 offline tests passed in 140.462 seconds**
(`temp/proxy-review-full-a01.log`). The repeated eight-process CPU BF16 resume
and global-gradient checks also passed (`temp/proxy-review-ddp-a01/verified.json`):
maximum resumed-state difference 1.862645149230957e-9 and global-gradient
difference 3.725290298461914e-9, with identical centering buffers across ranks.
Compilation and whitespace checks passed. No B200 workload was started.

## FA4 extension (2026-10-05)

Following the operator's acceptance of the Arm A FA4 evidence, the same
`attention_backend=fa4` interface now supports V1/V3 and all P1/P3 variants.
The native/replaced K/V projections, QK normalization, RoPE, target normalization,
EMS, losses and Trainer remain shared. FA4 replaces only the final attention
operation, using one document-fragment layout per microbatch. EOS ownership,
position resets, and next-document target masking remain unchanged.

The CLI default remains SDPA for compatibility. Select FA4 explicitly in a recipe
and select the proxy arms explicitly in `make-jobs`; the standalone FA4 baseline
recipe still defaults to A. A matched screen must use one backend across all arms.
Backend changes on resume and reuse of a dense baseline in an FA4 queue are refused.
FA4 stays pinned to 4.0.0b33 with no fallback. Training receipts now check Q, K and V
backward calls, including the replaced groups.

Local CPU tests use an independent per-document attention oracle, including
nonzero proxy gates, all routes and lambda-zero controls, auxiliary-only routing,
loss/parameter-gradient agreement, frozen normalization buffers, checkpoint
recomputation, isolation and exact native Trainer save/resume. These do not test
the CUDA kernel. `scripts/check_fa4_proxy.py` supplies separate full-model B200
numerical checks and artifact validation for three-step, eight-GPU Trainer smokes.
B200 execution and outcomes will be recorded after verification; no full screen
is launched by this validation task.
