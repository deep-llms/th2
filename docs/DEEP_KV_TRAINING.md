# Four-arm anticipatory K/V training

Implementation of the arms in `anticipatory_deep_kv_four_arm_pilot_v2.md`, with
the user's 2026-09-27 budget override: approximately 30B English tokens and 1M
tokens per optimizer update, with optional matched iteration cutoffs. Entry point:
`python -m deep_kv`. Training now uses Hugging Face Trainer and Accelerate,
following `train.py` and `scripts/train_qwen3_0.6b_baseline.sh`. The old scripts
remain historical EmbHub examples; the active launcher is `scripts/train_deep_kv.sh`. No model weights are loaded: Qwen3 is initialized
from its local config with a fixed seed. Use Transformers **5.9.0**, as installed
in B200 `train_env` and local `sampling_b200`, with Accelerate **1.13.0**.

## Fixed experiment

| Arm | Auxiliary attention | Alignment |
|---|---|---|
| A | None | None |
| B | Strict-past, native block-5 query | None |
| C | Identical to B | Native block-5 K/V |
| D | Identical to B | Native block-21 K/V |

Targets are normalized native pre-RoPE keys and native values, computed once in
the same model forward. Targets alone are detached. Alignment averages direct
L1 over heads/features and valid tokens, with `(L_K + L_V) / 2`, coefficient 1.
Native attention and auxiliary attention use the same positions. Auxiliary
visibility intersects the backbone mask with `j < t`; empty-source outputs are
exactly zero. Its bias-free output projection starts at zero. The entire
backbone is trainable. All arms share identical initial backbone tensors;
B/C/D additionally share identical branch tensors. Activation checkpointing
recomputes ordinary blocks during backward without mutable capture hooks.

| Setting | Value |
|---|---|
| Backbone | Qwen3 0.6B geometry, 28 blocks, random initialization |
| Context | 2048 |
| Global input tokens/update | 1,048,576 (512 full contexts) |
| Full schedule updates/arm | 28,600 (includes packing margin below 30B) |
| Full schedule input tokens/arm | 29,989,273,600 |
| Selected run cutoff | 2,000 updates = 2,097,152,000 input tokens/arm |
| Execution | Eight GPUs per arm, sequential A → B → C → D |
| Microbatch/rank | 16; four accumulation passes (64 contexts/rank) |
| Master parameters / compute | float32 / bfloat16 autocast |
| Optimizer | Trainer fused AdamW on CUDA; betas (0.9, 0.95), epsilon 1e-8 |
| Peak LR / weight decay | 3e-4 / 0.1; Trainer parameter groups exclude biases/norms |
| Schedule | Baseline 500-step warmup; HF cosine_with_min_lr, minimum 10% of peak |
| Gradient clipping | Global norm 1.0 |
| Model / data seed | 42 / 42 (baseline seed) |
| Monitoring | Every 512 updates, fixed 128 evaluation contexts |
| Final evaluation | Fixed 4,882 contexts = 9,998,336 input tokens |
| Checkpoint | Every 250 updates and at a cutoff/final update; retain two |
| Logging / data-loader workers | Every 10 updates / 8 workers per rank |
| Experiment logging | Offline W&B on GPU; JSON history; CPU tests disable W&B |

Optimizer, batch size, scheduler, seed, logging and worker settings follow the
baseline launch script. The fixed evaluation and matched cutoff protocol remain
project-specific. Trainer uses its native scheduler step convention (the first
warmup update has LR zero). A cutoff shortens execution while retaining this full
schedule. Microbatch may be any divisor of 64; accumulation adapts to preserve
the global batch. The selected B200 microbatch is 16. Synthetic CPU
smoke runs use a separate, clearly labeled tiny recipe.

The original 28,610-update budget was slightly larger than the expected packed
English split. The 28,600-update schedule follows the margin recommended in
[the shortfall analysis](DEEP_KV_DATA_SHORTFALL_20260927.md). Its capacity estimate
is not a measured packed count; preparation still validates the exact budget.

## Data

`deep_kv.b200.json` selects the already sampled **English** training/evaluation
directories. No resampling of CulturaX or full-weight download is needed.
Preparation appends `<|endoftext|>` (151643), concatenates each 1,000-document
batch, splits into 2048-token contexts and drops that batch's remainder, matching
the current packing policy. Documents can attend across EOS boundaries within
a context; EOS does not imply segment isolation.

Preparation takes enough contexts from the ordered source prefix for the fixed
30B budget, then shuffles those contexts once with the locked data seed. It
writes uint32 token files and a fixed context permutation, about 120 GB total,
with checksums. Preparation covers the full budget even for an early-stop run,
so changing the cutoff preserves the exact data order and allows resume.
Every arm verifies and reuses these exact files. If packing leaves too few
contexts, preparation fails explicitly; it never repeats data to fill a budget.
Train and evaluation sources differ. Data reuse requires the same `updates`,
`tokens_per_update`, `context`, `eval_rows` and `data_seed`, plus matching source
paths, model-config checksum, packing policy and token/order checksums. Logging,
checkpoint/monitor intervals, workers, optimizer settings, model seed and
microbatch may change for a new run without preparing tokens again. The full
original preparation recipe stays in the manifest for provenance. Resume and
arm comparison still require identical full run settings. Changing only the
cutoff never requires preparation. The sampled text needs no resampling.
Trainer uses a sequential sampler over the existing fixed permutation, then
Accelerate shards batches across ranks; it does not apply a second shuffle.
These local token files are experiment inputs; the sampled Arrow data stays text.

## Plan and sequential launch

The following are launch instructions, not an automatic deployment. Existing
GPU workloads must be handled under the repository's GPU ownership rules first.
The queue only checks GPUs and refuses occupied devices; it never reclaims them.

```bash
python -m deep_kv plan --config deep_kv.b200.json --stop-after 2000
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 2000 \
  --output temp/deep-kv-jobs.json
python run_experiments.py --config temp/deep-kv-jobs.json --list
python run_experiments.py --config temp/deep-kv-jobs.json \
  --run-dir /mnt/local/_outputs/deep-llms_th2/deep-kv-hf-seed42-2k
```

Use the selected environment's Python for all commands. The generated queue
prepares data on CPU, runs each arm with eight-process `accelerate launch`, then produces
`comparison.json` on CPU. Each arm must exit successfully and publish the exact
2,000-update/2,097,152,000-token stopped marker before the queue advances.
Omitting `--stop-after` instead requires full 28,600-update/29,989,273,600-token
completion.
Rank zero writes offline W&B files to `<run-dir>/<arm>/wandb/` by default;
an explicit `WANDB_DIR` overrides that location. Each training call owns and
closes its W&B run, including on failure, so sequential calls in one Python
process use separate runs and correct directories. An existing active W&B run
must be finished before calling this trainer. Normal CPU tests disable W&B;
the dedicated logging regression enables the offline SDK on CPU.
The report verifies shared data/recipe/initialization and checkpoint identities.
It reports LM-loss differences B−A, C−B, D−B, D−C and D−A; negative favors the
first arm. Single-seed differences are exploratory, not statistical proof.

To prepare once and reuse in another explicitly planned queue:

```bash
python -m deep_kv prepare --config deep_kv.b200.json --output /PATH/prepared
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 2000 \
  --data-dir /PATH/prepared --output temp/deep-kv-reuse-jobs.json
```

## Checkpoints and stopping

Each arm writes `run.json`, `metrics.json`, and standard Trainer
`checkpoint-N/` directories: model safetensors, optimizer, scheduler, per-rank
RNG, training arguments and trainer state. The custom wrapper preserves both
names of tied embedding/output weights when saving, so standard HF restore
loads every tensor. The latest two certified checkpoints are retained. Allow roughly
8 GB per checkpoint, plus space for a third while a new save is being written.

After all ranks finish saving, `checkpoint-N/deep_kv.json` records the run
identity, step, component-loss history, and hashes of every required checkpoint
file. Resume selects the latest certified save and checks its identity/files;
unmarked partial saves are ignored. Native HF rotation is disabled: cleanup runs
only after certification and retains the two latest certified saves. This keeps
an interrupted replacement save from deleting the last recoverable checkpoint.
Partial directories do not count toward that limit and may require extra disk
space after an interruption. The report also verifies checkpoint hashes.
Old custom-loop `checkpoint.pt` files are not HF checkpoints and are rejected.
These checkpoints contain our custom model's full state; restoring it requires
the DeepKV wrapper, rather than treating it as an unmodified Qwen model.

The selected experiment uses a common **2,000-update cutoff**, consuming
2,097,152,000 tokens per arm (8,388,608,000 across four arms). Generate its queue
with `--stop-after 2000`; omitting that argument requests the full schedule:

```bash
python -m deep_kv plan --config deep_kv.b200.json --stop-after 2000
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 2000 \
  --data-dir /PATH/prepared --output temp/deep-kv-cutoff-jobs.json
python run_experiments.py --config temp/deep-kv-cutoff-jobs.json --run-dir /PATH/run
```

The queue passes the same cutoff to A/B/C/D and the comparison. Each arm retains
the 28,600-update LR schedule, evaluates all 4,882 fixed evaluation contexts,
saves a resumable checkpoint, and writes `stopped.json`. The cutoff queue checks
the exact iteration and token count before advancing. `comparison.json` records
`compared_update` and `training_complete: false`; its `status: complete` means
the comparison finished. A full-budget queue still requires `complete.json`.
The CLI default remains full training. A cutoff equal to 28,600 is full completion.

The trainer and report also accept `--stop-after N` individually. To continue
an arm to the full budget, omit the cutoff when resuming:

```bash
bash scripts/train_deep_kv.sh --config deep_kv.b200.json \
  --data-dir /PATH/prepared --output /PATH/run/D --arm D --resume
```

Resume rejects changed data, code, software, model configuration, precision or
world size. It continues at the next exact global context batch. The generic
queue itself uses fresh output directories; resume interrupted arms explicitly.
Resume removes the old stop marker after checkpoint validation. If interruption
occurs after saving the final checkpoint but before publishing results, resuming
restores `metrics.json` before publishing `complete.json` without another update.
There is no automatic training extension or GPU-burn management in this module.

## Loss and evaluation normalization

`DeepKVTrainer.compute_loss` returns a microbatch mean:
`LM_sum / LM_target_count + (K_sum + V_sum) / (2 * input_token_count)`.
The LM denominator excludes the first token of each causal context. The K/V
loss averages heads/features and all input positions; only native targets are
detached. Every training microbatch is full and equal-sized. The subclass sets
`model_accepts_loss_kwargs=False`: Trainer divides by four accumulation steps,
and DDP averages over eight ranks. There is no extra manual scaling by either
factor. Baseline/NoAlign simply have zero K/V terms.

Evaluation returns five small statistics per context, avoiding vocabulary-size
logit storage. Trainer gathers these and removes duplicate padding examples
from uneven distributed evaluation batches before computing token-weighted LM,
K, V and total losses. Model selection/report comparisons use **LM loss**;
HF's `eval_loss` is the full training objective for aligned arms.

## Local verification

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_deep_kv.py -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m deep_kv smoke --output temp/kv-smoke
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  -m deep_kv smoke --stop-after 2 --output temp/kv-smoke-ddp8
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  tests/deep_kv_resume_worker.py --data-dir temp/kv-smoke-ddp8/prepared \
  --output temp/kv-resume-check --bf16
```

CPU tests exercise the real Trainer loop, custom accumulation gradients,
mechanism acceptance, fixed data, checkpoint identity, exact single-process
resume with worker prefetch, interrupted-save recovery, and cutoff reporting.
The Gloo worker checks all four arms with eight CPU processes and four
accumulation steps, asserting the actual projection dtype during execution.
CPU tests use non-fused AdamW;
CUDA/NCCL, fused AdamW and full-context B200 throughput remain hardware checks.
The pinned libraries need a small CPU-only optimizer restore adjustment:
Accelerate's `cpu:0` device is normalized to `cpu` for `torch.load`; CUDA uses
Trainer's standard restore path.

Earlier full-geometry CPU tests established 28-layer model forward/backward
and zero-branch Base equivalence; they were not GPU capacity measurements.
The local English evaluation audit found 4,883 complete contexts, enough for
the fixed 4,882-context evaluation budget.
