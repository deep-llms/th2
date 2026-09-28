# Deep-KV training through train.py

The mechanism is defined in [the four-arm specification](anticipatory_deep_kv_four_arm_pilot_v2.md).
The user's later budget overrides its original 1B/32K pilot: approximately
30B English tokens, 1,048,576 tokens per update, and a matched 2,500-step cutoff.

## Ordinary training follows the baseline

`train.py` is the single training entry point. It uses HfArgumentParser,
TrainingArguments, set_seed, HF Dataset.map/cache/shuffle, default_data_collator,
and Trainer.train/save_model/save_metrics/save_state. HF/Accelerate initializes
distributed training, places the model, shards batches, creates the optimizer
and scheduler, scales accumulated gradients, and manages checkpoints/resume.
There is no separate custom training loop or distributed launcher.

The old EmbHub customization is replaced with DeepKV. The necessary Trainer
subclass supplies the combined LM/KV loss and five per-example evaluation
statistics, avoiding full vocabulary-logit storage. It also handles tied tensor
storage when saving the nn.Module wrapper and a pinned-version CPU optimizer
restore issue; CUDA restore uses HF unchanged. A small callback logs loss
components and requests stop/save at the selected update. W&B stays offline,
with files under the arm output directory (or an explicit WANDB_DIR).

The old PCC pipelines, Recipe class, preparation commands, per-step history
ledger, checkpoint hashes/certification/rotation, and duplicate train CLI are
removed. Checkpoints now use ordinary HF handling and save_total_limit=2.

## Recipe

The executable source of ordinary hyperparameters is `deep_kv.b200.json`:

| Setting | Value |
|---|---|
| Model | Local Qwen3-0.6B-Base config, random initialization, 28 blocks |
| Arms | A Base; B extra attention; C shallow alignment; D deep alignment; E same as D with weight .3 |
| Consumer / deep target | Block 5 / block 21, 1-based |
| Context | 2048 |
| GPUs / microbatch / accumulation | 8 / 16 / 4 |
| Tokens per update | 1,048,576 |
| Full schedule | 28,600 updates = 29,989,273,600 input tokens |
| Selected cutoff | 2,500 updates = 2,621,440,000 input tokens per arm |
| Seed / data seed | 42 / 42 |
| Optimizer | Native HF fused AdamW on CUDA, betas .9/.95, weight decay .1 |
| LR | 3e-4, cosine_with_min_lr, minimum .1 of peak, warmup 1,430 (5% of full schedule) |
| BF16 / gradient clipping | Enabled / 1.0 |
| Logging / saving | Every 10 / 250 updates, and save at cutoff |
| Monitoring | 128 fixed contexts every 512 updates |
| Final evaluation | 4,882 fixed contexts, 9,998,336 input tokens |

Warmup uses 5% of the full 28,600-update schedule, as requested on 2026-09-28,
including when stopping at 2,500 updates. This supersedes the baseline's 500-step
warmup. Changing stop_after never changes max_steps or the LR curve. Startup validates
capacity for the full schedule even when stopping early. All arms receive the
same configuration except arm, output directory, and run name. B/C/D/E share
identical auxiliary initialization. Alignment uses detached native pre-RoPE
normalized K and native V from the same forward pass, direct L1. C/D use weight 1;
E uses weight 0.3 on `(L_K + L_V) / 2`, so its individual K/V coefficients are
0.15. E has exactly D's initialization, architecture, targets, stop-gradient,
optimizer, data order, and schedule. The coefficient is fixed by arm identity;
recorded `pilot.arm` also prevents resuming D as E. Training, evaluation and
objective logging use the same coefficient; logged K/V losses remain unweighted.

## Packing is unchanged

`deep_kv/packing.py` is a mechanical move of the previous `pcc/packing.py`.
It appends one explicit `<|endoftext|>` (151643) to each document, concatenates
within each HF map batch, splits into fixed contexts, and drops that batch's
remainder. It does not carry remainders between map batches. EOS is not a
segment attention barrier. Tokenization/grouping still use the two Dataset.map
calls inside main_process_first, followed by shuffle(seed=42) and the normal
Trainer sampler. There is no raw-text export or pretokenization job.

Training uses 160 preprocessing workers; evaluation retains one worker and its
existing fixed prefix. Keep source files, tokenizer, worker count, and software
identical across arms. A cache miss recomputes the same token sequences/order.
Moving the helper may cause an HF cache rebuild, without changing packing.
The sampled text and the separately running corpus preparation job are unchanged.

## Commands

Use Transformers 5.9.0 and Accelerate 1.13.0. Model/config/tokenizer/data paths
must already exist locally. The entry point disables online HF access and W&B
uploads. No explicit redundant upload flag is supplied to TrainingArguments;
the pinned HF default keeps uploads disabled.

For a single arm, use a standard HF JSON configuration:

```bash
bash scripts/train_deep_kv.sh deep_kv.b200.json
```

That config defaults to Arm A with a 2,500-update cutoff on the full schedule. For another arm or a cutoff,
copy the JSON and set arm, output_dir, and stop_after. Alternatively pass the
standard HF CLI arguments directly to train.py through the same shell launcher.

For the intended sequential 2,500-step experiment:

```bash
python -m deep_kv make-jobs --config deep_kv.b200.json --stop-after 2500 \
  --output temp/deep-kv-jobs.json
python run_experiments.py --config temp/deep-kv-jobs.json --list
python run_experiments.py --config temp/deep-kv-jobs.json --run-dir /PATH/fresh-run
```

The default queue remains A/B/C/D. Select the new E follow-up explicitly, using
the identical base recipe and a fresh output directory:

```bash
python -m deep_kv make-jobs --config deep_kv.b200.json --arms E \
  --output temp/deep-kv-E-jobs.json
python run_experiments.py --config temp/deep-kv-E-jobs.json --list
python run_experiments.py --config temp/deep-kv-E-jobs.json --run-dir /PATH/fresh-E-run
```

`--arms D E` generates a sequential D/E comparison; `--arms A B C D E` includes
all five. `python -m deep_kv report --run-dir /PATH/run --arms D E` compares
matched D/E results in that directory. An E-only queue validates E's result
and reports its LM loss without claiming a comparison to absent arms.

Both launch routes use `resources/accelerate_config.yaml`.

`train.py` also defaults `NCCL_NVLS_ENABLE=0` before distributed imports, so
the generated queue retains the original Qwen script's NCCL setting without
requiring a shell export. Direct launches may explicitly override that default.
W&B remains offline and uses the `deep2shallow` project by default.
Before a B200 launch, copy the config to the machine's actual HF Accelerate
default config and check `accelerate env`, as previously requested. These instructions do not themselves
launch anything or stop GPU processes.

By default the queue runs A/B/C/D then report, requiring successful exit and each arm's
result.json at the exact requested step before advancing. Report checks matched
configuration, dataset fingerprints, steps, final evaluation, and saved model
presence. It reports B−A, C−B, D−B, D−C and D−A in held-out LM loss. Negative
favors the first arm; one seed does not establish statistical significance.

### Functional-loss arms F/G

[The functional-loss specification](deep_route_kl_variant.md) adds two arms
with exactly D's forward, initialization and recipe:

- F: `LM + 0.3 * route_KL`.
- G: `LM + 0.3 * route_KL + 0.3 * message_SmoothL1` (no division by two).

Routing is `KL(deep || predicted)` at temperature 1, using the exact rotated
shallow query/predicted keys from the auxiliary forward and native rotated
block-21 keys, with the normal GQA repetition. Query and deep references are
detached only in the auxiliary loss. Empty strict-past rows are excluded before
softmax. KL averages valid queries and heads; message SmoothL1 (beta 1) also
averages head features, before the output projection. F has no auxiliary value
gradient. Neither F nor G computes raw K/V alignment.

FP32 score/loss computation uses 128-query chunks with activation recomputation
during training, preserving every allowed source and the exact objective while
bounding temporary score memory. This does add computation; B200 throughput and
capacity must be measured with the real configuration before a long launch.
The normal Trainer/Accelerate loop, packing, caches, schedule and evaluation
selection remain unchanged. Metrics are `loss_route` and `loss_msg`, with the
same weighted objective for training and evaluation. Compact per-example
statistics preserve correct denominators under distributed evaluation padding.

```bash
python -m deep_kv make-jobs --config deep_kv.b200.json --arms F G \
  --output temp/deep-kv-FG-jobs.json
python run_experiments.py --config temp/deep-kv-FG-jobs.json --list
```

This queues F, then G, then result validation/comparison; no previous arm is
implicitly retrained. Reporting matched results with `--arms A B D E F G`
includes the required F-B, G-F and G-B comparisons and prior controls. Compare
`eval_lm_loss`, not differently weighted auxiliary objectives.

## Resume and output

Each arm writes native checkpoint-N directories, a final model/tokenizer,
trainer_state.json, train_results.json, eval_results.json, train_config.json,
and result.json. The latter appears only after the requested update and final
evaluation finish. There is no custom checkpoint format or certificate.

Like the original train.py, rerunning the same output automatically resumes its
latest HF checkpoint, or accepts resume_from_checkpoint explicitly within the
same arm's output directory. Checkpoints from another output are rejected. Saved
configuration and data fingerprints must match; stop_after may change. Resuming
at an already reached cutoff restores/evaluates without taking another update.
A nonempty directory with no checkpoint is rejected. An interrupted/incomplete
native checkpoint can fail to load; select an earlier intact native checkpoint
explicitly. There is no automatic certified-checkpoint recovery layer.
Old Deep-KV/PCC checkpoint formats are not supported by this entry point.

The saved weights belong to the DeepKV wrapper; construct the same model and
load its state when evaluating outside train.py. They are not a bare Qwen
AutoModelForCausalLM checkpoint.

## Verification

The tests retain the specification's mechanism checks: base equivalence,
strict-past/empty-source masks, native target correctness, detached targets,
and live-branch gradient paths. Integration checks cover the real train.py entry
point, unchanged EOS packing/cache rebuild order, native accumulation scaling,
matched cutoff/reporting, and native checkpoint resume. The eight-process CPU
worker uses BF16, microbatch 16, accumulation 4, and uneven evaluation shards.
Local checks do not establish B200 CUDA/NCCL capacity or throughput.
