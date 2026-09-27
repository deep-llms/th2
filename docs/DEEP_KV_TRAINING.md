# Four-arm anticipatory K/V training

Implementation of `anticipatory_deep_kv_four_arm_pilot_v2.md`. Entry point:
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
| Global input tokens/update | 32,768 (16 full contexts) |
| Updates/arm | 30,518 |
| Input tokens/arm | 1,000,013,824 |
| Execution | Eight GPUs per arm, sequential A → B → C → D |
| Microbatch/rank | 1 by default; two accumulation passes |
| Master parameters / compute | float32 / bfloat16 autocast |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8 |
| Peak LR / weight decay | 3e-4 / 0.1; no decay on vectors/norms |
| Schedule | 1,526 warmup updates; cosine decay to 10% of peak |
| Gradient clipping | Global norm 1.0 |
| Model / data seed | 2901 / 20260922 |
| Monitoring | Every 512 updates, fixed 128 evaluation contexts |
| Final evaluation | Fixed 4,882 contexts = 9,998,336 input tokens |
| Checkpoint | Every 512 updates and at a graceful cutoff/final update |

The optimizer and evaluation values fill settings left unspecified by the
four-arm document. They are explicit implementation defaults, not previously
measured optimal values. A configuration cannot silently shorten or extend the
pilot. Synthetic CPU smoke runs use a separate, clearly labeled tiny recipe.

## Data

`deep_kv.b200.json` selects the already sampled **English** training/evaluation
directories. No resampling of CulturaX or full-weight download is needed.
Preparation appends `<|endoftext|>` (151643), concatenates each 1,000-document
batch, splits into 2048-token contexts and drops that batch's remainder, matching
the current packing policy. Documents can attend across EOS boundaries within
a context; EOS does not imply segment isolation.

Preparation takes enough contexts from the ordered source prefix for the fixed
budget, then shuffles those contexts once with the locked data seed. It does
not tokenize/shuffle the entire 30B-token English pool. It writes uint32 token
files and a fixed context permutation, about 4 GB total, with checksums. Every
arm verifies and reuses these exact files. Train and evaluation sources differ.
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
30,518-update/1,000,013,824-token completion marker before the queue advances.
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

For a deliberate common cutoff, invoke all arms with the same `--stop-after N`.
This preserves the 30,518-update LR schedule and writes `stopped.json`, never
`complete.json`; the full-run queue therefore cannot mistake it for success.
For example, an individual arm may be resumed with:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  --max-restarts=0 -m deep_kv train --config deep_kv.b200.json \
  --data-dir /PATH/prepared --output /PATH/run/D --arm D --resume
```

Resume rejects changed data, code, software, model configuration, precision or
world size. It continues at the next exact global context batch. The generic
queue itself uses fresh output directories; resume interrupted arms explicitly.
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
