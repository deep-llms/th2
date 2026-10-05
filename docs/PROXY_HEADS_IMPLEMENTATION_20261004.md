# P1/P3 implementation and validation

Implements [revision 4 of the proxy-head specification](proxy_heads_P1_P3_spec_v3.md).
The user confirmed dense SDPA and document-local position IDs. This is a new
screening series; previous non-isolated A–G checkpoints are not matched controls.
No B200 experiment was launched during implementation.

## Model and shared training flow

`deep_kv/proxy.py` adds a model adapter to the existing Qwen backbone. P1 replaces
the last two KV groups in eligible even layers using a gated estimate of a future
MLP-output sum. P3 uses three document-reset exponential moving sums and full-width
deep-band targets. Queries and the K/V projection parameters remain native; the
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

`deep_kv/proxy_training.py` supplies only the additional loss, centering and logging
adapters. Optimizer, scheduler, dataloader, accumulation, distributed execution,
saving, resume and stopping remain HF Trainer/Accelerate through `train.py`.
No extra data-export or preprocessing command is required.

Centering means bootstrap from each rank's first microbatch, then update once per
optimizer step after all-reducing detached sums/counts. Means stay fixed within
the step and during checkpoint recomputation. Lambda-zero controls also update
means, but compute diagnostic cosines under no-grad only at logging intervals.
All means, decay constants and channel masks are checkpoint buffers. Alpha gates
have zero weight decay. Block auxiliary gradients reach only estimator parameters;
flow auxiliary gradients also reach the backbone. Targets are detached.

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

Actual FA4 CUDA numerical behavior/throughput for this new production path still
requires a B200 smoke test. No GPU training is launched by implementation tests.

## Reusing a completed baseline

Reports ignore channel-mask settings only for the computation of vanilla A/V
controls. Both the mask path and full receipt must match between every P1/P3
arm (including lambda-zero controls), and across seeds. All other recipe fields,
including checkpointing flags, dataset fingerprints and stopping steps remain
strict. This does not relax checkpoint-resume validation.

To reuse an existing A without copying weights or altering its saved config:

```bash
python -m deep_kv make-jobs --config /local/calibrated-screen-recipe.json \
  --reuse-baseline 42=/local/completed-baseline/seed-42/A \
  --output temp/proxy-screen-jobs.json
```

The supplied recipe should be a copy of `proxy_heads.b200.json` with the shared
`proxy_channel_mask` path set after calibration. Paths above are placeholders.
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

## Optional channel calibration

Without a calibration file, the code explicitly logs an empty exclusion set.
The specification permits that fallback only when no checkpoint is available.
A trained isolated baseline now exists, so calibrate it before the proxy screen.
`scripts/calibrate_proxy_mask.py` can create one from a local trained original
arm-A checkpoint (either DeepKV or ProxyModel), using isolated text packing
and approximately 1M tokens. Loading is strict; proxy block outputs are observed
directly, since decoder forward hooks do not fire on the ProxyModel path.
Calibration requires every block to account for every processed token:

```bash
python -m scripts.calibrate_proxy_mask \
  --checkpoint /local/old-A/model.safetensors --config /local/model/config.json \
  --tokenizer /local/model --data-dir /local/english/train \
  --tokens 1048576 --output /local/proxy-channel-mask.json
```

Paths above are placeholders. Add `--device cuda --bf16` only for an approved GPU
calibration run. Pass the result as `proxy_channel_mask` in the shared recipe.
The file records per-block statistics and source hashes; the training recipe
records the mask SHA256. Changing it on resume is rejected. No calibration job
or checkpoint download was started here.

## Validation evidence and limits

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
