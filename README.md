# Anticipatory Deep-KV four-arm pilot

The active task is [the four-arm specification](docs/anticipatory_deep_kv_four_arm_pilot_v2.md).
Train Qwen3 from scratch and compare:

| Arm | Auxiliary attention | Alignment target |
|---|---|---|
| A | None | None |
| B | Strict-past attention at block 5 | None |
| C | Same as B | Native block-5 K/V |
| D | Same as B | Native block-21 K/V |

The current recipe is approximately 30B English tokens, 1,048,576 input tokens
per update, and an optional matched 2,500-update cutoff. Every arm uses eight
GPUs; arms run sequentially. The decisive comparison is D versus C, alongside
D versus B and A. Lower held-out LM loss is better.

## Code

- `train.py`: the baseline Hugging Face training flow, with the Deep-KV model
  and loss replacing the old project's model customization.
- `deep_kv/model.py`: the four arms and their causal masks.
- `deep_kv/packing.py`: unchanged EOS tokenization and batched CLM packing.
- `deep_kv/training.py`: custom loss, compact evaluation, component logging,
  fixed-step stopping, and the small adapters needed to save/restore the wrapper.
- `deep_kv/__main__.py` and `report.py`: queue generation and matched comparisons.
- `deep_kv.b200.json`: standard HF training arguments plus model/data/arm settings.
- `run_experiments.py`: existing sequential job runner.

Read [the training guide](docs/DEEP_KV_TRAINING.md) for commands, checkpoint
behavior, and validation. There is no separate data-preparation step before
training: the normal Hugging Face Dataset cache is shared across arms.

## Run the approved pilot

After verifying the machine, environment, local data, and GPU availability:

```bash
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 2500 \
  --output temp/deep-kv-jobs.json
python run_experiments.py --config temp/deep-kv-jobs.json --list
python run_experiments.py --config temp/deep-kv-jobs.json \
  --run-dir /mnt/local/_outputs/deep-llms_th2/deep-kv-2500
```

`commands.sh` is an operator-controlled deployment command. Runner pushes and GPU process management
follow [docs/commands.md](docs/commands.md), [docs/GIT_PUSH.md](docs/GIT_PUSH.md),
and [docs/GPU_SAFETY.md](docs/GPU_SAFETY.md).

## Local checks

Use the pinned `sampling_b200`/`train_env` environment (Transformers 5.9.0,
Accelerate 1.13.0). Tests are synthetic and CPU-only:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/deep_kv_resume_worker.py \
  --output temp/deep-kv-resume-check
```

## Earlier research

The old PCC diagnostics, privileged-teacher, distillation, and joint-training
implementations have been removed from the active tree. Their source is retained
in Git history (before this refactor, commit `63bcc61`). `docs/PCC_*` and older
research/result notes are historical records, not current launch instructions.
The inherited `eval/` scripts target ordinary Hugging Face model checkpoints;
they are not loaders for the custom Deep-KV wrapper. Use the final evaluation
performed by `train.py` for this pilot.

The independent corpus preparation project at `/disk/thuat/prepare_pretraining_data`
and its running sampler are separate from this training refactor.
