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
off. Checkpointing follows existing defaults; performance changes require the
existing resume checks. Real GPU capacity for this combination is still unmeasured.

Queue generation (creates a manifest only):

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --output temp/proxy-screen-jobs.json
python run_experiments.py --config temp/proxy-screen-jobs.json --list
```

Default seeds are 42, 43, 44, with matching model/data seeds and deterministic
estimator seed `seed + 1`. Output directories are `seed-N/ARM`. The queue has 24
training jobs, three within-seed reports and one aggregate report, run sequentially.
Each training job uses all eight GPUs. Use `--arms ... --seeds ...` to select a
subset. Historical `deep_kv.b200.json` queue defaults remain A/B/C/D.

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

## Optional channel calibration

Without a calibration file, the code explicitly logs an empty exclusion set.
`scripts/calibrate_proxy_mask.py` can create one from a local trained original
arm-A checkpoint, using the same isolated text packing and approximately 1M tokens:

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
