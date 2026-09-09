# Capacity-allocation diagnostics

Implements the measurements in
[the reviewed design](Diagnostics_TokenFrequency_EmbeddingSpectra_GradientStatistics_v0.2_Reviewed.md).
These supplement ordinary English PPL and downstream benchmarks; they are not
new training objectives. Nothing here starts, stops, or modifies a remote run.

## Workflow

1. Prepare one immutable diagnostic bundle from the same local training text,
   tokenizer and packing configuration as the experiment.
2. Compare matched saved checkpoints using that bundle: frequency-binned NLL,
   embedding spectra, and fixed-probe gradients.
3. Optionally include these workers in `eval.eval_parallel`. Fine-tuning remains
   a separate subsequent queue; diagnostics use the original checkpoints, not
   fine-tuned weights.

For the current screening, begin with all six `checkpoint-5000` checkpoints.
Earlier checkpoints can be added later in matched sets. Do not silently
substitute a different step, regenerate missing initialization weights, or
compare models on different held-out data.

## Freeze the inputs once

Use an installed compatible environment. Everything is local-only; on B200,
any missing downloads/installations must go through authorized `#d`/`#i`.
Example paths below describe the existing th2 layout, not a submitted job:

```bash
python -m eval.diagnostic_data \
  --train-dir /mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B/train \
  --eval-dir /mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B/eval \
  --tokenizer-name /mnt/local/_models/deep-llms_th2/Qwen3-0.6B \
  --languages en --vocab-size 151936 --block-size 2048 \
  --preprocessing-num-workers 160 --preprocessing-batch-size 1000 \
  --preprocessing-cache-dir /mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000 \
  --training-shuffle-seed 42 --probe-blocks 8 --seed 42 \
  --output-dir /mnt/local/_data/deep-llms_th2/swt/diagnostics/qwen_en_pool_v1
```

This is CPU preprocessing/counting, not a GPU training job. Reading the entire
packed training pool is a real I/O task; do it once, not once per checkpoint.
The shared cache can be reused, but this preparer should be scheduled separately
from active training to avoid CPU/I/O contention. It does not delete caches.

The fresh output contains:

- `frequencies.npz`: int64 input-token counts and saved token-to-bucket mapping.
- `eval/<language>/`: frozen packed held-out blocks, including labels.
- `manifest.json`: file SHA256/size, tokenizer content identity, HF raw/packed
  fingerprints, packing options, counts, seed and exact probe block IDs.

The large training corpus is not duplicated. Source training identity uses HF
dataset fingerprints; the bundle's files and tokenizer files additionally have
cryptographic hashes. Retain the original sampled-data manifest as well.
Production workers check the parent `train_config.json`, shuffled training
fingerprint and packing settings; English workers also check the held-out
fingerprint against training's validation data. A mismatch fails, not warns.
Frozen artifact hashes are verified on loading; never regenerate the manifest
merely to hide a mismatch.

### What the frequencies mean

Counts describe the **full selected packed training pool**, not the exact tokens
consumed by a particular stopped checkpoint. In the current experiment the
available pool is larger than the approximately 5.24B input tokens consumed by
5000 steps. Accordingly, `seen`/`unseen` mean seen/unseen in the frozen pool,
not proof that a row was/was not accessed before step5000. Reconstructing the
actual per-step shuffled/DDP exposure stream would be a separate diagnostic.

Counting follows training: concatenate within each map batch, discard short
tails, add no EOS or special tokens, and count complete input blocks (including
their first tokens). Loss scores only shifted targets. These counts are recorded
separately. For current full-batch, first-epoch checkpoints, consumed input and
scored-target budgets are derived from saved step × effective batch × block
length (or block length minus one), never inferred from folder names alone.

Seen non-special token types are sorted by `(count descending, ID ascending)`.
Cumulative boundaries are `ceil(N × [1%,10%,50%,90%,100%])`; small vocabularies
may have empty buckets. Unseen, EOS and padding have separate entries. If EOS
and padding share an ID, actual unmasked EOS belongs to EOS; masking, not that
ID alone, removes padding from loss. Overall NLL includes all valid targets,
including EOS; no unobserved bucket receives a fabricated zero NLL/PPL.

## Run one checkpoint

After the normal environment and free-GPU checks:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.diagnostics_checkpoint \
  --checkpoint /absolute/run/B0/checkpoint-5000 \
  --diagnostic-bundle /absolute/qwen_en_pool_v1 \
  --languages en --diagnostics frequency spectra gradients \
  --precision bf16 --gradient-precision fp32 --batch-size 1 \
  --output /absolute/fresh_results/B0_diagnostics.json
```

These are three different measurements:

- **Frequency loss:** full frozen held-out set, FP32 cross-entropy from BF16
  model calls, chunked along the sequence to bound temporary logits memory.
  Counts and sums are aggregated before computing NLL/PPL. Future multilingual
  runs preserve per-language results and a target-weighted aggregate, without
  packing across language boundaries.
- **Spectra:** raw input/output tables and effective body-facing products,
  centered over seen rows (primary) and all model-vocabulary rows (secondary).
  Float64 CPU covariance uses centered chunk merging, with four CPU threads per
  worker by default. Factorized effective covariance is transformed by its
  adapter exactly, avoiding materializing a full V × body-width matrix. C/D
  input and output body widths are taken from actual shapes, not assumed equal.
  `--spectrum-buckets` optionally adds every frequency subset. Output distinguishes
  covariance eigenvalues, `sigma/sqrt(N)` and literal singular values, plus
  energy fractions, effective/stable rank, numerical rank and k90/k95.
- **Probe gradients:** eight fixed held-out blocks per language by default,
  token-mean loss across all probe microbatches, no clipping or optimizer step,
  evaluation mode/dropout off. Default FP32 is deliberate for the tied-gradient
  conservation check; it is recorded separately from frequency-loss precision.
  BF16 probe mode is available with a looser, reported additivity tolerance.
  Tables, adapters, transitions, norms, attention q/o and FFN up/down projections
  are measured. Row statistics are dimension-normalized RMS with mean, median,
  p90, zero fraction and bucket population.

B0 additionally uses three diagnostic forwards at unchanged weights: total,
input-path only, output-path only. Their aggregate gradients must satisfy the
reported relative-L2 additivity tolerance. Reports include full/bucket cosines,
active input-row counts/fractions and active-row-only cosine. A zero-norm cosine
is null, not zero. Small probes can have low coverage; increase the probe set
by creating a new shared bundle before comparing models, never per candidate.

Weights/checkpoint files are not modified. Gradient buffers are cleared and
model mode restored on exit. These gradients are **not historical optimizer
training gradients**, and cannot establish actual AdamW update sizes.

## Alongside the existing eval queue

Add the following to the normal `python -m eval.eval_parallel ...` command:

```bash
--diagnostic-bundle /absolute/qwen_en_pool_v1 \
--diagnostics frequency spectra gradients
```

Without that flag the old two workers per checkpoint (PPL and benchmarks)
remain unchanged. With all diagnostics there are five workers per checkpoint:
**30 jobs for six checkpoints**, queued onto the selected free GPUs. Each
worker has its own `CUDA_VISIBLE_DEVICES` and fresh result file. Spectra use CPU
linear algebra even though their worker loads the checkpoint on its assigned
GPU; do not mistake this for a GPU utilization benchmark.

Diagnostic workers default to batch1, FP32 gradient probes and four CPU threads.
Use the standalone entry point for explicit alternative probe precision, batch
size or bucket spectra. All results bind to the same frozen manifest hash;
missing sections or a mismatched manifest prevent queue success. The queue
does not run fine-tuning or GPU burns itself. The outer authorized handoff must
wait for the **entire queue**, verify completion, then follow the usual GPU gates.

Compare frequency workers after completion:

```bash
python -m eval.compare_diagnostics \
  --reference /absolute/results/B0/diagnostic_frequency.json \
  --candidates /absolute/results/A128/diagnostic_frequency.json \
               /absolute/results/A256/diagnostic_frequency.json \
  --output /absolute/fresh_comparison.json
```

The comparison rejects different manifests, checkpoint budgets, scoring
precision, languages or target coverage. Delta NLL is candidate minus reference.
Confidence intervals and plots are not generated automatically. Add paired
uncertainty estimates before making a central claim from small tail differences.

## Future live-training instrumentation (not attached to current runs)

`eval.diagnostic_training.attach_training_diagnostics` is an opt-in observer:

```python
# After constructing a future CausalTrainer, before calling trainer.train():
from eval.diagnostic_training import attach_training_diagnostics
attach_training_diagnostics(
    trainer, "/absolute/fresh_training_statistics",
    every=100, update_every=1000, mapping=token_to_bucket,
)
```

It supports ordinary unsharded FP32/BF16 Trainer/DDP and requires positive
`max_grad_norm`. It rejects FP16 gradient scaling, FSDP/ZeRO and other unsupported
distributed modes. Observations occur after DDP reduction and before clipping;
HF5.9's `on_pre_optimizer_step` is too late for this. Optional actual updates
compare selected weights immediately before and after the real optimizer step,
including clipping, optimizer state and weight decay. Copying tables is costly;
keep update logging infrequent, or set `update_every=0` to disable it.

Only rank0 writes; no additional backward, collective, clipping or optimizer
step is performed. Resuming instrumentation requires a new output directory.
This helper is not imported/attached by the current `train.py`. Historical
pre-clip gradients and actual updates cannot be reconstructed from final weights.

## Tests and deferred B200 gate

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s tests -p test_diagnostics.py -v
```

Tests cover direct-SVD equivalence, exact adapter orientation for every arm,
counts/masks, PPL agreement, immutable inputs, FP32/BF16 probes, accumulation,
additivity, nonmutation and actual Trainer observer behavior. They use tiny CPU
models, not full-size B200 resource measurements. Run a destination smoke with
real frozen inputs on a free GPU before scheduling the full diagnostics.

The earlier **benchmark-harness precision bug remains a separate open gate**:
its nested autocast currently disables the requested BF16 model calls. The
diagnostics above call the model directly and do not use that harness wrapper.
`scripts/check_eval_precision.py` now preserves the actual-forward check:

```bash
# Later, after explicit authorization, compatible eval env and free-GPU checks:
CUDA_VISIBLE_DEVICES=0 python -m scripts.check_eval_precision \
  --device cuda --output /absolute/fresh_eval_precision_check.json
```

It performs tiny forward calls for all six arms, requires actual FP32 and BF16
projection outputs as requested, and exits nonzero on a mismatch. It does not
score benchmarks or download anything. **Expect BF16 failure until the harness
is explicitly configured with mixed_precision_dtype** (and FP32 softmax is
recommended). Do not launch full benchmark evaluation with the known mismatch,
or mistake the existing 72-test pass for proof of BF16 operation. The user deferred
this destination test; no B200 job has been submitted by this implementation.
