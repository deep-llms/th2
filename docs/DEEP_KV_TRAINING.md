# Four-arm anticipatory K/V training

Implementation of the arms in `anticipatory_deep_kv_four_arm_pilot_v2.md`, with
the user's 2026-09-27 budget override: approximately 30B English tokens and 1M
tokens per optimizer update, with optional matched iteration cutoffs. Entry point:
`python -m deep_kv`. This is separate from the legacy EmbHub `train.py` and
pretrained PCC experiments. No model weights are loaded: Qwen3 is initialized
from its local config with a fixed seed. Use Transformers **5.9.0**, as installed
in B200 `train_env` and local `sampling_b200`.

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
| Updates/arm | 28,610 (floor of 30B / 1,048,576) |
| Input tokens/arm | 29,999,759,360 |
| Execution | Eight GPUs per arm, sequential A → B → C → D |
| Microbatch/rank | 1 by default; 64 accumulation passes (64 contexts/rank) |
| Master parameters / compute | float32 / bfloat16 autocast |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8 |
| Peak LR / weight decay | 3e-4 / 0.1; no decay on vectors/norms |
| Schedule | 1,431 warmup updates (5%, rounded up); cosine decay to 10% of peak |
| Gradient clipping | Global norm 1.0 |
| Model / data seed | 2901 / 20260922 |
| Monitoring | Every 512 updates, fixed 128 evaluation contexts |
| Final evaluation | Fixed 4,882 contexts = 9,998,336 input tokens |
| Checkpoint | Every 512 updates and at a graceful cutoff/final update |

The optimizer and evaluation values fill settings left unspecified by the
four-arm document. They are explicit implementation defaults, not previously
measured optimal values. A cutoff shortens execution while retaining this full
schedule. Microbatch may be any divisor of 64; accumulation adapts to preserve
the global batch. Larger microbatches require a GPU memory check. Synthetic CPU
smoke runs use a separate, clearly labeled tiny recipe.

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
Train and evaluation sources differ. Old 1B/32K prepared streams are rejected
because their recorded recipe differs; the sampled text needs no resampling.
These local token files are experiment inputs; the sampled Arrow data stays text.

## Plan and sequential launch

The following are launch instructions, not an automatic deployment. Existing
GPU workloads must be handled under the repository's GPU ownership rules first.
The queue only checks GPUs and refuses occupied devices; it never reclaims them.

```bash
python -m deep_kv plan --config deep_kv.b200.json
python -m deep_kv make-jobs --config deep_kv.b200.json --output temp/deep-kv-jobs.json
python run_experiments.py --config temp/deep-kv-jobs.json --list
python run_experiments.py --config temp/deep-kv-jobs.json \
  --run-dir /mnt/local/_outputs/deep-llms_th2/deep-kv-v2-seed2901
```

Use the selected environment's Python for all commands. The generated queue
prepares data on CPU, runs each arm with eight-rank `torchrun`, then produces
`comparison.json` on CPU. Each arm must exit successfully and publish the exact
28,610-update/29,999,759,360-token completion marker before the queue advances.
The report verifies shared data/recipe/initialization and checkpoint identities.
It reports LM-loss differences B−A, C−B, D−B, D−C and D−A; negative favors the
first arm. Single-seed differences are exploratory, not statistical proof.

To prepare once and reuse in another explicitly planned queue:

```bash
python -m deep_kv prepare --config deep_kv.b200.json --output /PATH/prepared
python -m deep_kv make-jobs --config deep_kv.b200.json \
  --data-dir /PATH/prepared --output temp/deep-kv-reuse-jobs.json
```

## Checkpoints and stopping

Each arm writes `run.json`, `checkpoint.pt`, `metrics.json`, and on full
completion `complete.json`. The checkpoint contains the full model, optimizer,
per-rank RNG and exact update cursor. The latest checkpoint is replaced atomically;
older periodic checkpoints are not accumulated. Budget roughly 7–8 GB per arm
for float32 model plus Adam moments, with additional temporary space during save.

For a deliberate common cutoff, generate the queue with `--stop-after N`.
For example, 1,000 optimizer updates consume 1,048,576,000 tokens per arm:

```bash
python -m deep_kv plan --config deep_kv.b200.json --stop-after 1000
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 1000 \
  --data-dir /PATH/prepared --output temp/deep-kv-cutoff-jobs.json
python run_experiments.py --config temp/deep-kv-cutoff-jobs.json --run-dir /PATH/run
```

The queue passes the same cutoff to A/B/C/D and the comparison. Each arm retains
the 28,610-update LR schedule, evaluates all 4,882 fixed evaluation contexts,
saves a resumable checkpoint, and writes `stopped.json`. The cutoff queue checks
the exact iteration and token count before advancing. `comparison.json` records
`compared_update` and `training_complete: false`; its `status: complete` means
the comparison finished. A full-budget queue still requires `complete.json`.
No cutoff is selected by default. A cutoff equal to 28,610 is full completion.

The trainer and report also accept `--stop-after N` individually. To continue
an arm to the full budget, omit the cutoff when resuming:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  --max-restarts=0 -m deep_kv train --config deep_kv.b200.json \
  --data-dir /PATH/prepared --output /PATH/run/D --arm D --resume
```

Resume rejects changed data, code, software, model configuration, precision or
world size. It continues at the next exact global context batch. The generic
queue itself uses fresh output directories; resume interrupted arms explicitly.
Resume removes the old stop marker after checkpoint validation. If interruption
occurs after saving the final checkpoint but before publishing results, resuming
restores `metrics.json` before publishing `complete.json` without another update.
There is no automatic training extension or GPU-burn management in this module.

## Local verification

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_deep_kv.py -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m deep_kv smoke --output temp/kv-smoke
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  -m deep_kv smoke --output temp/kv-smoke-ddp8
```

CPU tests cover the specification's six acceptance requirements, activation
checkpoint gradients, bfloat16 arithmetic, fixed packing, exact checkpoint resume,
and queue budgets. The eight-process Gloo smoke run checks distributed training
and uneven evaluation shards, including ranks with no evaluation rows.
GPU/NCCL capacity and throughput still require a short authorized B200 check.

The 2026-09-27 review additionally verified that B/C/D have identical LM outputs
and LM gradients for shared weights with a nonzero auxiliary output projection,
in both float32 and bfloat16. An eight-process CPU interruption/resume check
matched uninterrupted training within 7.5e-9 in every arm's parameters. Repeat
that check after creating a synthetic smoke's prepared data with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  tests/deep_kv_resume_worker.py --data-dir temp/kv-smoke/prepared \
  --output temp/kv-resume-check
```

The actual 28-layer, 600,244,352-parameter arm D passed a CPU forward/backward
check on eight tokens, including zero-output Base equivalence and finite
gradients. This verifies real model geometry, not 2048-token GPU memory capacity.
The locally reproduced English evaluation set yields 4,883 complete contexts
under the current EOS packing policy, enough for the fixed 4,882-context budget.
