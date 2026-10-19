# Training and evaluation guide

Run commands from the project root in the selected ML environment. This guide
preserves the detailed training contracts; start with [README.md](../README.md)
for project setup and [JOBS.md](JOBS.md#training--resume--evaluation-example) for
the sequential train/resume/evaluate example.

- [Initialize and train](#initialize-and-train)
- [Full schedule and screening cutoff](#full-schedule-and-screening-cutoff)
- [Resume safeguards](#resume-safeguards)
- [Data, packing and document boundaries](#data-packing-and-document-boundaries)
- [Attention backends](#attention-backends)
- [Caches and data identity](#caches-and-data-identity)
- [Accelerate launcher](#accelerate-launcher)
- [Training tests](#training-tests)
- [Held-out validation loss and perplexity](#held-out-validation-loss-and-perplexity)
- [Extending a new project](#extending-a-new-project)

`train.py` uses native HF models and Trainer/Accelerate. `PackedTrainer` is a
small subclass that prepares document masks on the training device and checks
loss-count assumptions. The native causal-LM loss, optimizer, scheduler,
accumulation, checkpointing and saving remain HF behavior. Use `build_model()`
for future model changes; custom losses require separate normalization tests.
Default document isolation is validated for **full-attention Qwen3 with default
RoPE**. Other architectures, sliding-window layers and RoPE variants are rejected
until separately validated; some dynamic RoPE schemes depend on batch-wide length.

Choose a compatible environment containing PyTorch, Transformers, Datasets,
Accelerate, Tokenizers and Safetensors. Create a project-specific environment
specification as described in [commands.md](commands.md#1-install-environments);
this project now provides envs/train.txt and envs/eval.txt. See
[BDOT_IMPLEMENTATION.md](BDOT_IMPLEMENTATION.md) for the new model and runtime.
The runner installs these as conda environments `train` and `eval` under its
Python 3.11 prefix. Both specs include the tested CUDA 13 FA4 dependencies;
`eval` additionally includes the pinned lm-eval Hugging Face integration and
SentencePiece. Each file is self-contained for the controller installer.
`envs/train_fa2.txt` separately requests an experimental FA2 environment with
matching shared pins; its compiled wheel availability and B200 behavior are
unverified. See [FA2 environment notes](FA2_B200_ENVIRONMENT_20261019.md).
Add W&B or TensorBoard only if requested.
Tested locally on CPU with torch 2.14.0+cu130, transformers 5.9.0,
datasets 4.8.5 and accelerate 1.13.0; these are a test receipt, not a GPU lockfile.
Copying the template performs no environment installation or GPU validation.
Historical B200 checks are summarized in [VALIDATION.md](VALIDATION.md);
validate your own hardware and pinned environment before a new experiment.

All model/tokenizer/data paths must be local. Download through the approved
controller before running. Hub access is offline (including publication requests);
reporting defaults to none, with optional offline W&B or local TensorBoard.

## Initialize and train

Random initialization:

```bash
python train.py \
  --config_name /absolute/model/config.json \
  --tokenizer_name /absolute/tokenizer \
  --data_dir /absolute/sampled/train \
  --output_dir /absolute/outputs/new-baseline \
  --block_size 2048 --per_device_train_batch_size 2 --report_to none
```

For pretrained initialization, replace `--config_name` with
`--model_name_or_path /absolute/model`; the tokenizer defaults to that directory.
This starts a new optimizer. Ordinary HF resume is available through
`--resume_from_checkpoint /absolute/run/checkpoint-N`, or automatic last-checkpoint
selection in the output directory. Data/tokenizer/packing compatibility is checked against checkpoint
`data_recipe.json`. A separate `training_recipe.json` now guards the model,
training configuration and runtime. HF Trainer still restores the model,
optimizer, scheduler, RNG and consumed-data position; no restore logic is replaced.
Nonempty output directories without a checkpoint are refused.

## Full schedule and screening cutoff

For a full 30,000-step schedule that stops after 10,000 optimizer updates, add:

```bash
--max_steps 30000 --stop_after 10000 --warmup_steps 1500
```

Here the scheduler uses **30,000 steps**, with 1,500 warmup steps (5%).
Use one warmup option only. In the tested Transformers 5.9, a supplied
`warmup_ratio` overrides `warmup_steps`, even when the ratio is zero.
The example shell launcher supplies `--warmup_steps 500`; override it with
`--warmup_steps 1500` for this schedule. Recompute that integer when changing
full schedule length, not when changing the screening cutoff.
`stop_after` is an absolute global optimizer step, not microbatches or additional
steps since resume. A small HF callback stops after a complete update and forces
a full `checkpoint-10000` save, even with `save_strategy=no` or between normal
save intervals. Model, optimizer, scheduler and RNG states use native HF saving;
`save_only_model=True` and `ignore_data_skip=True` are rejected with this cutoff.

To continue, keep `max_steps=30000` and the same recipe, and set
`stop_after=20000` (or omit `stop_after` to finish the full schedule). Resume via
`--resume_from_checkpoint /absolute/run/checkpoint-10000`, or let the same output
directory select its latest checkpoint. Changing the explicit schedule length or
requesting a stopping step already reached by that checkpoint is rejected.
Without `stop_after`, ordinary HF max-step/epoch stopping remains in effect.
`train_results.json` reports both `global_step` and `schedule_steps`; the cutoff
is also saved in `train_config.json`. The normal `save_total_limit` option still
controls retention; leave it unset to retain all checkpoints.

## Resume safeguards

Every checkpoint and the run root contain `training_recipe.json`. It records the
constructed model configuration and tensor schema, training arguments, effective
per-process/global batch sizes, process count, precision/backend settings,
PyTorch/Transformers/Accelerate versions, CUDA/cuDNN versions and training-code
hashes. Data, tokenizer, packing and source identity remain in `data_recipe.json`.
Initial model/config paths alone are not used as model identity: Trainer restores
checkpoint weights, and the actual constructed architecture must match.

Resume compares all TrainingArguments except an explicit operational allowlist.
Changing optimizer/LR/warmup/schedule, model structure, seeds, batch/accumulation,
process count, precision, attention backend, checkpointing or data-loader settings
is rejected with the differing field names. This is deliberately conservative:
changing the microbatch while keeping the global batch equal is still rejected.
Changing `stop_after` to a later step, output location, logging, save frequency or
`save_total_limit` is allowed. Runtime/code changes require revalidation; there is
no silent override that labels a changed experiment as an equivalent resume.

The checkpoint must include nonempty optimizer and scheduler files and the RNG
file for **every rank** (plus the CUDA FP16 scaler when applicable). Missing state
fails before training instead of silently starting a fresh optimizer. A receipt
is written after normal checkpoint saving. These checks validate compatibility
and file presence, not every byte of checkpoint contents; HF handles loading and
reports corrupt state. Inputs/checkpoints must be complete and immutable.

Older checkpoints without this receipt are refused. Keep their original software
for an exact resume, or deliberately use `--model_name_or_path` with a **fresh
output directory** to initialize a new optimizer and new experiment. A final
model directory alone is not a resumable optimizer checkpoint.

This completeness guard covers ordinary Trainer and DDP. DeepSpeed/FSDP checkpoint
layouts and automatic batch-size adaptation require separate validation and are
rejected on guarded resume. CPU distributed training normalizes Accelerate's
`cpu:0` device to `cpu` for native optimizer loading; CUDA devices are unchanged.
Single- and two-process CPU tests compare resumed batches, LR, weights, optimizer,
scheduler and RNG with an uninterrupted reference. The separate GPU tests are described in [VALIDATION.md](VALIDATION.md);
these checks do not guarantee bitwise reproducibility across hardware or runtimes.

## Data, packing and document boundaries

`--data_dir` accepts a local Parquet file, a directory of sorted Parquet shards,
a single HF `save_to_disk` Dataset, or a tree of language/shard datasets. Select
a **single split** explicitly. DatasetDict roots and visible mixtures of named
train/validation/test splits are rejected. The loader uses only the `text` column;
null/non-string documents fail instead of being silently removed. File ordering
is lexical, followed by each source's stored row order. No network download or
manual conversion to `save_to_disk` is needed.

Training performs the usual two HF maps: tokenize documents, then concatenate
and split into fixed-length blocks. It appends **one document-end token after
every document**, including an empty document. Literal occurrences in the text
remain untouched. No automatic BOS, truncation or padding is applied. The final
incomplete block is dropped **per map batch within each worker shard**, as in
the original CLM pipeline; this is not one globally concatenated token stream.

`--document_end_token` defaults to the tokenizer's EOS. For a chat tokenizer,
that may be a chat-turn marker: choose the actual document marker explicitly.
The Qwen example uses `--document_end_token '<|endoftext|>'`; no Qwen token ID
is hardcoded. The marker must already encode as a single vocabulary token.
**Document isolation is now enabled by default**, using the baseline
boundary semantics with either dense SDPA (default) or optional FA4:

- Source row IDs travel alongside tokens through packing; literal EOS inside a
  document never creates a new segment. The appended EOS belongs to its document.
- Attention sees only the same document at the current or earlier token position.
- Positions reset to zero at each document fragment in a block. A long document
  continuing in a new block starts at zero there, without cross-block context.
- Labels at document starts and the first token of every block are ignored.
  Predicting each nonempty document's EOS remains part of the LM objective.
  Empty/single-token fragments have no within-document prediction target.

The independent settings are `--isolate_documents true` and
`--reset_document_positions true`. Isolation can be used without resets. For a
cross-document control, explicitly set **both** to false; resets without isolation
are rejected. These settings are saved in the data recipe and invalidate
incompatible caches/checkpoints.

`PackedCollator` keeps CPU batches linear in sequence length. With SDPA,
`PackedTrainer` builds the boolean `[batch, 1, length, length]` mask after device
transfer. With FA4, the collator prepares cumulative fragment lengths once on
CPU and transfers those small offsets; no dense attention mask is built. Model
weights remain a standard HF checkpoint. Packed evaluation must use the same collator and mask
preparation; loading weights alone does not invent document boundaries.

HF computes the denominator over non-ignored labels across all accumulation
microbatches and ranks (`average_tokens_across_devices=True`). Masking label zero
makes that count match the shifted LM targets. A microbatch/rank may contribute
zero targets; a whole global update with zero targets is rejected. Label
smoothing/custom loss functions require separate validation and are not enabled.

## Attention backends

Select `--attention_backend sdpa` (default) or `--attention_backend fa4`.
FA4 uses `packed_fa4.py`, a small HF `AttentionInterface` adapter around
`flash_attn.cute.interface.flash_attn_varlen_func`. Qwen's native projections,
QK normalization, RoPE, grouped KV heads and output projection remain unchanged.
Neither path uses a training KV cache.

FA4 requires a compatible CUDA environment, BF16/FP16 mixed precision, and the
pinned **flash-attn-4==4.0.0b33** API validated for this template (its CUDA 13
runtime used `flash-attn-4[cu13]==4.0.0b33` and `nvidia-cutlass-dsl==4.8.0`).
The B200 environment specs include these dependencies and `quack-kernels==0.6.5`.
Copying the source alone does not install them. Missing or wrong
versions fail explicitly; there is no silent SDPA fallback. This adapter requires
full-attention Qwen3, default RoPE, zero attention dropout, full unpadded blocks,
and document isolation. Position resets may be enabled or disabled independently
of the kernel. Accelerate `split_batches` and `dispatch_batches` must be false:
each rank collates its own batches and document offsets. Generation/KV-cache
support is outside this training adapter's scope.

Both backends consume identical packed sequences and can reuse the same HF
preprocessing caches. The selected backend and FA4 package/adapter identity are
recorded in the run/checkpoint recipe; changing backend on resume is rejected.
For direct packed model calls, select the backend with `build_model()`, then
use matching `PackedCollator` and `model_inputs()`; loading HF weights alone does
not register FA4 or reconstruct document boundaries. The packed evaluator below uses this same integration; other evaluation paths
need their own integration before claiming isolation.

CPU reference tests establish adapter semantics, not CUDA kernel accuracy or
speed. Before using this template's FA4 path for a real experiment, run the
included real-kernel smoke on an authorized free GPU in the configured environment:

```bash
python tests/ml/fa4_gpu_probe.py /absolute/new-fa4-receipt.json
```

It checks BF16 outputs/loss/parameter gradients against dense SDPA, document
leakage, and activation checkpointing on/off. Its explicit smoke tolerances are
2% logit and 5% global gradient relative L2 error; this is not long-run validation.
Historical B200 probes passed; see [VALIDATION.md](VALIDATION.md).
They do not replace validation after changing hardware, dependencies or the model.

## Caches and data identity

HF dataset caches remain shared across matching arms. Optional `--data_cache_dir`
selects the Parquet loader's HF cache; saved datasets use their normal local
HF map caches. `--overwrite_cache true` recomputes tokenization, packing and
shuffle without changing the recipe or deleting shared cache trees. No token
export step is required. `--map_batch_size` defaults to 1000 and worker count
defaults to one; **keep both fixed across compared runs**, because grouping and
remainder dropping can change with these settings. Different shuffle seeds reuse
tokenization/packing caches. `--data_seed` controls packed-data shuffle when
supplied; otherwise it follows `--seed`, and Trainer handles subsequent sampling.

A `data_recipe.json` is written at the run root before training and in each
checkpoint. It records the ordered sources, file sizes/mtimes, HF fingerprints,
tokenizer identity and asset hashes, EOS marker, worker/batch/block settings,
shuffle seed, code hash, library versions, and packed input/scored-token counts.
Source file stats/HF fingerprints are **not cryptographic corpus verification**;
use a trusted file manifest when that is needed. The selected inputs must already
be complete and immutable. This loader cannot infer hidden leakage or unfinished
unlabeled downloads from filenames.

Cache hits and recomputation produce the same sequence order with identical
inputs, tokenizer, versions, preprocessing settings and seed. Cached data from
old packing/attention policies are not reused. Resume requires a matching data receipt;
old checkpoints without one, relocated/stat-changed inputs and changed recipes
are rejected rather than silently assumed equivalent. Weight initialization into
a fresh run remains available with `--model_name_or_path`.

## Accelerate launcher

The example GPU launcher requires all paths explicitly:

```bash
bash scripts/train_qwen3_0.6b_baseline.sh \
  /absolute/accelerate.yaml /absolute/model/config.json /absolute/tokenizer \
  /absolute/sampled/train /absolute/outputs/new-baseline \
  --preprocessing_num_workers 8
```

Activate the chosen ML environment first, or set `PYTHON_BIN` to its interpreter.
The launcher uses BF16; the supplied Accelerate configuration selects GPU count.
Its example batch is 16 sequences/device, accumulation 4, sequence length 2048:
1,048,576 input tokens/update **only with eight ranks**. Other defaults retain
one epoch, LR 3e-4, cosine decay to 10%, and 500 warmup steps. Trailing Trainer
arguments override defaults. Set `--report_to wandb` with your own `WANDB_PROJECT`
for offline logging. The included single-process Accelerate example is not an
eight-GPU production configuration. GPU ownership checks remain the caller's job.

## Training tests

Optional real CPU tests (tiny local Qwen, no downloaded assets or GPU use):

```bash
python -m unittest discover -s tests/ml -v
```

They exercise training from scratch through Accelerate on CPU, pretrained
training with activation checkpointing, saved optimizer/RNG state, exact model
reload, CLI/JSON arguments, and shell launcher argument forwarding. Cutoff tests
compare uninterrupted and stopped/resumed training across warmup and an epoch
boundary: actual sampled batches, LR sequence, final weights, optimizer/scheduler
states and PyTorch RNG match exactly in the CPU test, with nonzero dropout. Data tests
cover Parquet/Arrow equivalence, EOS edge cases, sorted input order, split
rejection, real cache hits/rebuilds, multiprocessing packing, and resume guards.
Isolation tests compare packed vs independent document logits, losses and
parameter gradients; perturb earlier/future documents; cover literal EOS, empty
documents and chunk continuations; and repeat leakage checks with checkpointing.
FA4 adapter tests explicitly replace the CUDA kernel with a CPU per-document
reference and compare outputs, parameter/input gradients, token-weighted Trainer
updates, native checkpoint reloads and cache reuse; production has no CPU fallback.

An optional two-rank CPU probe checks the complete distributed Trainer update
with accumulation and unequal target counts against a full-batch SGD reference:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ml/ddp_isolation_probe.py \
  temp/new-isolation-ddp-probe
```

Use a fresh output path. It creates only synthetic CPU data and writes
`verification.json` after both ranks' update comparison. No GPUs are required.
Add `--fa4-reference` after the output path to test the FA4 adapter with the CPU
reference kernel across both ranks. This does not exercise actual FA4 CUDA code.
Add `--cutoff` to stop at step 1 of a 4-step schedule and verify a distributed
checkpoint including each rank's RNG state.
A separate two-rank CPU probe verifies stopped/resumed training against a full
run, including recipe validation and each rank's batch order and RNG state:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ml/ddp_resume_probe.py \
  temp/new-resume-ddp-probe
```

Use a fresh output path. Each rank writes `resume_rankN.json` only after exact
model/optimizer/scheduler/RNG and batch/LR comparisons pass.
The ordinary `tests/` suite remains standard-library-only.

## Held-out validation loss and perplexity

### During training

Add these options to the existing `train.py` command or pass them through the
baseline shell launcher:

```bash
--eval_data_dir /absolute/data/validation \
--eval_strategy steps --eval_steps 500 \
--per_device_eval_batch_size 2
```

HF Trainer evaluates after every 500 **optimizer updates**, after accumulation
and the optimizer update have completed, then continues training. Validation is
disabled by default. `--eval_strategy epoch` uses the native epoch schedule;
`--eval_on_start true` adds a starting evaluation. `--do_eval true` with
`--eval_strategy no` requests only a final evaluation. Periodic validation does
not force an extra evaluation at a cutoff that falls between eval intervals.

The held-out split uses the same tokenizer, EOS, packing, document isolation,
position resets and masked targets as training. Its packed shuffle has fixed seed
0. The default scores the entire split; `--eval_max_sequences N` selects a fixed
prefix for cheaper monitoring. Source overlap is rejected by resolved paths,
not content deduplication. Validation identity and selection are saved under
`validation` in each `training_recipe.json` and must match on resume.

The existing `PackedTrainer` keeps HF's scheduling, logging, evaluation callbacks
and native optimizer/checkpoint flow. It reuses the standalone scoring code for
token-weighted loss/PPL, with disjoint unpadded shards and final sum reductions.
It logs `eval_loss`, `eval_perplexity`, `eval_nll_sum`, `eval_num_tokens` and
`eval_num_sequences` in Trainer history/checkpoints and the configured logger.
It does not collect predictions or run downstream benchmarks.

Validation runs without gradients and restores module train/eval modes and
Python/NumPy/PyTorch CPU/CUDA RNG states, including when scoring raises. It uses
the training precision and a private evaluation dataloader RNG; it does not
change training batches, optimizer or schedule. No extra model replica is loaded.
Ordinary single-device and CPU/CUDA DDP are supported; sharded training,
`torch_compile`, full-eval weight casting, schedule-free optimizers, metric-driven
LR schedules and loading best weights are rejected for this monitoring path.
Custom callbacks that deliberately modify weights or control training are outside
this guarantee.

CPU tests compare validation on/off and cutoff/resume with dropout, accumulation
and activation checkpointing: batches, LR, weights, optimizer/scheduler and RNG
match exactly. The two-rank probe also covers uneven shards and an empty rank:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/ml/ddp_periodic_eval_probe.py \
  temp/new-periodic-eval-probe
```

Use a fresh output path. These checks do not establish GPU bitwise reproducibility
or actual FA4 CUDA kernel correctness. The existing strict code/runtime resume
guard still applies; this change does not silently migrate older checkpoints.

### Standalone checkpoint evaluation

Use `eval/packed_ppl.py` for checkpoints produced by this template. It requires
`data_recipe.json` and local tokenizer/model assets, and reads the checkpoint's
actual sequence length, EOS marker, map batch/worker settings, document isolation,
position resets and target masking. It reuses `prepare_training_data` and ordinary
HF caches; there is no token-export step or training update during scoring.

For a local CPU check:

```bash
python eval/packed_ppl.py \
  --checkpoint /absolute/run/checkpoint-10000 \
  --eval_data_dir /absolute/data/validation \
  --output_file /absolute/results/checkpoint-10000-ppl.json \
  --attention_backend sdpa --use_cpu true --batch_size 2
```

For an authorized GPU evaluation, use your configured Accelerate environment:

```bash
accelerate launch --config_file /absolute/accelerate.yaml eval/packed_ppl.py \
  --checkpoint /absolute/run/checkpoint-10000 \
  --eval_data_dir /absolute/data/validation \
  --output_file /absolute/results/checkpoint-10000-ppl.json \
  --precision bf16 --batch_size 16
```

These are examples, not GPU allocation or launch authorization. Pass the selected
precision explicitly (`fp32`, `bf16`, or `fp16`); it defaults to FP32. The backend
defaults to the checkpoint's recorded choice. Explicit `--attention_backend sdpa`
or `fa4` allows an identical-weights comparison, with both choices recorded in the
result. FA4 requires the same optional pinned kernel, CUDA and BF16/FP16 as training.
Historical FA4 CUDA probes passed on B200; ordinary CPU tests use a reference kernel.

- Select one held-out split explicitly, as Parquet or a saved Dataset/tree.
  Shared resolved source paths with the recorded training data are rejected.
  This detects path reuse, **not copied documents or corpus-level overlap**.
- EOS and document-start masking match training. The incomplete tail is dropped
  per map batch/worker, and chunk continuations reset positions as in training.
  This measures the packed training objective, not sliding-window document PPL.
- Evaluation uses a fixed shuffle seed of zero, independent of the training seed.
  By default every packed sequence is scored. Optional `--max_sequences N` selects
  a fixed prefix for a smoke check; the limit and actual counts appear in the JSON.
  Keep validation files, tokenizer and packing settings fixed across comparisons.
- Each process loads one model replica and scores distinct complete packed rows.
  No sequence is duplicated to pad distributed batches. Uneven shards and empty
  ranks are valid; only the final loss sum and token/sequence counts are reduced.
- Loss is total negative log likelihood divided by total eligible next-token
  count, and PPL is exp(loss). It never averages batch PPL or per-batch mean losses.
  CE is evaluated in FP32 chunks and summed in FP64; all-boundary batches contribute
  zero. Zero global target count, nonfinite loss and PPL overflow fail explicitly.
- Outputs include loss, PPL, token/sequence counts, checkpoint step when present,
  backend/precision/world size, evaluation recipe/fingerprint, evaluator/config
  hashes and weight-file metadata. File metadata is not a full weight checksum.
  A complete JSON is published without overwriting an existing result.

Queue packed evaluator commands with the generic `run_experiments.py`, passing each output
file under the run directory and requiring its JSON `status` to equal `ok`.
Periodic evaluation uses the same scoring implementation; downstream benchmarks
remain a separate task.

Tests compare packed scoring with independent documents, different batch sizes,
zero-target batches and the FA4 reference adapter. Real checkpoint CLI tests
compare one versus two CPU processes with uneven shards and with an empty rank;
the training smoke also evaluates an actual saved checkpoint on held-out Parquet.

## Extending a new project

Keep native HF training, optimizer, scheduling and checkpoint handling unless
your experiment requires changing them. Add the model through `build_model()`.
`PackedTrainer` deliberately rejects unvalidated custom losses: when introducing
one, implement and test its accumulation/DDP token normalization explicitly.
The default isolated-attention path supports full-attention Qwen3/default RoPE;
this is a reusable infrastructure template, not universal model support.

No corpus, language mixture, downstream benchmark suite or finetuner is selected.
Implement those from the new project's requirements. The old CulturaX-specific
sampler and incompatible legacy benchmark/PPL entrypoints are not shipped.
Training already accepts prepared local text and performs tokenization/packing
with HF caching; a separate token-export step is unnecessary.
