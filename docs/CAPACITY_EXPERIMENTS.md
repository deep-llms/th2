# Capacity-allocation experiments: implementation and use

Implements §12 of `Capacity_Allocation_Research_Project_v0.5_Final_Reviewed.md`.
This is a from-scratch **English pilot**, not pretrained GPT-2/Llama/Qwen tuning.
No research training has been launched by this implementation task.

## Implemented arms

| CLI arm | Body widths | Vocabulary interface | Unique parameters |
|---|---|---|---:|
| `B0` | 1024 × 6 | One genuinely tied 1024-wide table | 128,546,816 |
| `A128` | 1024 × 6 | Independent input 128 / output 896 | 129,595,392 |
| `A256` | 1024 × 6 | Independent input 256 / output 768 | 129,595,392 |
| `A512` | 1024 × 6 | Independent input 512 / output 512 | 129,595,392 |
| `C` | 256,256,512,512,1024,1024 | Independent 128 / 896 | 86,893,568 |
| `D` | 512,512,1024,1024,1280,1280 | Independent 128 / 896 | 126,285,056 |

Validate B0/A128 first; screen uniform arms before C/D. A128 is C/D's fixed-interface
reference. These are **not total-parameter-matched** models. Uniform arms also
support `--depth 12`; C/D reject that option until a deeper width budget is designed.
T2/T3, nonlinear input, N128, multilingual/Qwen and additional budget controls remain
deferred, as specified in the design. No automatic full-grid launch is configured.

`capacity_allocation/modeling.py` uses stock HF `LlamaForCausalLM` for B0.
Custom models reuse HF Llama layers, masks, RoPE, causal loss and dynamic KV cache.
C/D change both residual width and MHA head count with fixed head dimension 64.
T1 projections occur **after** complete residual blocks at changing-width boundaries.
Final RMSNorm precedes the output adapter. All matrices are bias-free.
No unused full-width vocabulary table, extra output table, or extra norm is retained.

Initialization policy is `fanin_scaled_head`: custom projections use normal std
`1/sqrt(fan_in)`, input lookup std 0.02, output table std
`0.02*sqrt(1024/output_rank)`. There is no forward-time logit multiplier.
Stock body initialization is retained. A shared seed does not imply identical
body tensors across architectures with different RNG consumption.

## Environment and correctness gate

Core template utilities still have no third-party dependencies. ML requirements
are separate in `requirements-capacity.txt`. The tested local environment uses
Python 3.11, torch 2.7.1+cu118, Transformers 5.9.0, Accelerate 1.13.0.
The tests so far use CPU; this is **not a verified B200 CUDA environment**.
Choose the torch CUDA wheel for the destination hardware/driver before installing
the remaining pins, then verify CUDA and run the GPU smoke gate. Do not silently
upgrade Transformers: the custom stage implementation depends on its layer API.

From the project root with the selected environment active:

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke_capacity --output temp/capacity_smoke.json
torchrun --standalone --nproc_per_node=2 -m scripts.smoke_capacity \
  --output temp/capacity_ddp_cpu.json
```

The smoke output path must be fresh. For an **authorized, free** GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scripts.smoke_capacity --device cuda --precision bf16 \
  --output temp/capacity_ddp_gpu.json
```

This never stops burns or other jobs. Follow `GPU_SAFETY.md` separately before
claiming GPUs. Tiny tests are correctness checks, not speed/quality measurements.
Full-size parameter counts are tested on the meta device without allocating weights.

## Prepare one frozen local English dataset

1. Obtain an authorized, pinned public CulturaX English release and a pinned
   `openai-community/gpt2` tokenizer through the run system's download mode.
   Training and preprocessing here make **no network requests**.
2. Create a source inventory JSON. Use real commit and SHA256 values, not these
   placeholders. Paths are relative to the inventory; absolute paths also work.
   Include a declared selection across English shards, not just a stream prefix.

```json
{
  "dataset": "uonlp/CulturaX",
  "language": "en",
  "revision": "REPLACE_WITH_40_CHARACTER_PUBLIC_RELEASE_COMMIT",
  "selection_rule": "REPLACE_WITH_RELEASE_WIDE_SHARD_SELECTION_RULE",
  "shards": [
    {"path": "en/en_part_00000.parquet", "sha256": "REPLACE_WITH_FILE_SHA256"}
  ]
}
```

3. Materialize the common packs into a **new** output directory:

```bash
python -m capacity_allocation.data \
  --source-manifest /path/to/source_inventory.json \
  --tokenizer-dir /path/to/gpt2_tokenizer \
  --tokenizer-revision REPLACE_WITH_TOKENIZER_COMMIT \
  --output /path/to/english_gpt2_packs \
  --seed 0 --sample-fraction 1 \
  --validation-fraction 0.004 --test-fraction 0.004
```

The CLI requires at least 5B packed training tokens and 20M each in validation/test
by default. Hash fractions are **document/cluster fractions**, not exact token
reservations: inspect measured counts and revise the preparation plan if needed.
The source inventory fixes order; hash sampling scans every listed shard rather
than stopping after a token prefix. Global exact duplicates are removed using an
on-disk SQLite index. Stable IDs preserve source/URL/timestamp/content hash and
shard/row locators. Splits use content hashes (or `--cluster-field FIELD` for
duplicate clusters), keeping exact duplicate content out of different splits.
Cluster IDs must be consistent for identical content.

Documents are tokenized without truncation or added vocabulary entries; append
one existing EOS, then pack continuously within each split. A final short tail
is retained in the binary file but unused; no tail is discarded per document or
tokenizer batch. Each full 2048-token pack has **2047 scored next-token targets**.
EOS targets are not masked. Attention can cross EOS within a pack.

Outputs: `train.bin`, `validation.bin`, `test.bin` (little-endian uint32),
`documents.sqlite`, `source_manifest.json`, and completion `manifest.json`.
The latter records file hashes, revisions, split rules and actual counts. Failed
preparation does not write a completion manifest and cannot overwrite its directory.
Preparation is currently single-process; no untested parallel sampler is implied.

**Before research training:** verify mapping to the public release and tokenizer
commit, inspect source/domain composition from the document manifest, and perform
the cross-split near-duplicate audit. The code records `near_duplicate_audit:
not_performed`; exact deduplication is not a substitute for that audit. No real
source inventory, tokenizer revision or sampled corpus has been selected here.

## Train a registered arm

`train_capacity.py` uses HF Trainer/Accelerate with one process per GPU. Supply
optimizer settings explicitly; the design has not selected their final values.
An example command structure, **after** those shell variables have been selected:

```bash
torchrun --standalone --nproc_per_node="$GPU_COUNT" train_capacity.py \
  --arm B0 --data "$PACKS" --output "$RUN_OUTPUT" \
  --train-tokens 1000000000 \
  --batch-per-device "$BATCH_PER_DEVICE" \
  --gradient-accumulation "$ACCUMULATION" \
  --learning-rate "$LR" --warmup-ratio "$WARMUP_RATIO" \
  --weight-decay "$WEIGHT_DECAY" --beta2 "$BETA2" \
  --precision bf16 --attention sdpa
```

Use `--arm A128`, etc., with separate fresh output directories. Keep
`world_size * batch_per_device * accumulation * 2048` identical within a
same-token comparison. All arms use AdamW, cosine decay, explicit warmup, no TF32,
no dropout, and `ddp_find_unused_parameters=False`; gradient flow is tested for
all parameters. Optional non-reentrant `--gradient-checkpointing` is available;
register its use and measure its speed/memory effect. Default beta1=0.9,
epsilon=1e-8 and clipping=1.0 are explicit implementation defaults, not tuned results.

The token horizon is rounded **up** to whole optimizer steps and the extra tokens
are recorded. Sufficient distinct packs must exist for this rounded horizon;
implicit repeated epochs are rejected. The trainer shuffles global pack indices
once with `--data-seed` and reuses that order independently of model RNG.
It never evaluates on the reserved test set during training.

The pinned Trainer normally counts unshifted labels when normalizing loss; our
override counts only `labels[:, 1:]`, preserving HF's accumulation/DDP reduction.
Validation gathers per-example NLL sums and target counts, not repeated batch
means. This lets Accelerate trim duplicated final-batch examples exactly. The
reported validation target count must match every full held-out pack once.

`--stop-at-step N` stops/saves without shortening the configured LR schedule.
For example, stopping at step N under a 5B-token horizon is not a completed
1B-token cosine run. `max_steps` is computed internally from the *full* token
horizon, never from the stop override. `result.json` distinguishes
`stopped_at_step` from `horizon_complete`. Periodic Trainer checkpoints contain
optimizer/scheduler/RNG state; `final/` is for model evaluation, not exact resume.
`--resume /same/run/checkpoint-N` resumes **interrupted** runs with unchanged
data/code/software/settings. Existing successful `result.json` is protected;
deliberate continuation requires a separately reviewed continuation workflow.

Artifacts also include `run_specification.json`, `parameters.json`,
`initial_scales.json`, `trainer_state.json`, periodic checkpoints and validation
metrics. Report processed tokens and scored targets separately. The recorded
train-call wall time includes periodic evaluation and checkpointing; it is **not**
an isolated kernel-throughput measurement. Peak memory is labeled rank zero.
HF's `total_flos` field is a library estimate, not measured FLOPs or a validated
variable-width compute accounting result; do not use it to claim equal compute.

To reload any checkpoint, import the registrations first:

```python
import capacity_allocation
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/path/to/run/final", local_files_only=True)
```

Dynamic cached decoding is covered by tests, including different KV head counts
at each stage. Tensor parallelism/static-cache/compiled deployment are not verified.

For a saved checkpoint, single-process held-out PPL can also be reproduced with:

```bash
python evaluate_capacity.py --checkpoint /path/to/run/final \
  --data /path/to/english_gpt2_packs --split validation \
  --device cuda --precision bf16 --batch-size 1 \
  --output /path/to/new_validation_result.json
```

Use the same precision across comparisons. `--split test` is an explicit final
confirmation action, not part of screening. The evaluator verifies pack hashes,
counts scored targets exactly and records checkpoint file hashes. For CPU
verification use `--device cpu --precision fp32`. Neither script downloads data.

The generic `run_experiments.py` can sequence explicit training argv lists and
verify each `result.json` contract; `commands.sh` remains inactive. No new remote
or automatic burn/handoff script is inferred from this implementation request.
